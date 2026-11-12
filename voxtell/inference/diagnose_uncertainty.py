#!/usr/bin/env python3
"""Frozen-model uncertainty diagnostics for DWI liver-cancer cases.

This module is deliberately evaluation-only: it creates no optimizer, sets every
VoxTell parameter to ``requires_grad=False``, and performs all prediction under
``torch.inference_mode``. Ground truth is used only after prediction to compute
offline Dice values; it is never passed to an uncertainty calculation.

The repository does not contain CUE-SFDA Stage 1. Consequently, an existing
per-case CUE CSV can be joined with ``--cue-csv`` without replacing CUE by an
unrelated proxy.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

# Keep the diagnostic offline, like the existing multiclass inference entrypoint.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch


PROMPTS = (
    "liver cancer",
    "liver tumor",
    "HCC",
    "hepatocellular carcinoma",
)
PROMPT_DICE_COLUMNS = (
    "dice_liver_cancer",
    "dice_liver_tumor",
    "dice_hcc",
    "dice_hepatocellular_carcinoma",
)
CSV_FIELDS = (
    "case_id",
    "baseline_dice",
    "cue_score",
    "cue_uncertainty",
    "image_pairwise_dice",
    "image_uncertainty",
    "image_variance",
    "prompt_pairwise_dice",
    "prompt_uncertainty",
    "prompt_variance",
    *PROMPT_DICE_COLUMNS,
)
UNCERTAINTY_COLUMNS = (
    "cue_uncertainty",
    "image_uncertainty",
    "prompt_uncertainty",
)


def case_id_from_path(path: str | Path) -> str:
    """Return a stable case ID while handling the compound NIfTI suffix."""
    name = Path(str(path)).name
    return name[:-7] if name.lower().endswith(".nii.gz") else Path(name).stem


def soft_dice(a: np.ndarray, b: np.ndarray, epsilon: float = 1e-8) -> float:
    """Soft Dice with an explicit empty/empty convention of one."""
    first = np.asarray(a, dtype=np.float64).reshape(-1)
    second = np.asarray(b, dtype=np.float64).reshape(-1)
    denominator = float(first.sum() + second.sum())
    if denominator <= epsilon:
        return 1.0
    return float(2.0 * np.dot(first, second) / denominator)


def mean_pairwise_soft_dice(probabilities: Sequence[np.ndarray]) -> float:
    if len(probabilities) < 2:
        raise ValueError("at least two predictions are required")
    values = [
        soft_dice(probabilities[i], probabilities[j])
        for i in range(len(probabilities))
        for j in range(i + 1, len(probabilities))
    ]
    return float(np.mean(values, dtype=np.float64))


def mean_voxel_variance(probabilities: Sequence[np.ndarray]) -> float:
    """Mean population variance across predictions, averaged over all crop voxels."""
    if len(probabilities) < 2:
        raise ValueError("at least two predictions are required")
    stacked = np.stack(probabilities).astype(np.float32, copy=False)
    return float(np.var(stacked, axis=0, ddof=0, dtype=np.float64).mean())


def binary_dice_from_crop(
    probability: np.ndarray,
    target_full: np.ndarray,
    crop_slices: tuple[slice, ...],
    threshold: float,
) -> float:
    """Dice against full GT without allocating a restored full probability map."""
    prediction = np.asarray(probability) > float(threshold)
    target_full = np.asarray(target_full, dtype=bool)
    target_crop = target_full[crop_slices]
    if prediction.shape != target_crop.shape:
        raise ValueError(
            f"prediction/GT crop shape mismatch: {prediction.shape} vs {target_crop.shape}"
        )
    intersection = int(np.count_nonzero(prediction & target_crop))
    denominator = int(np.count_nonzero(prediction)) + int(np.count_nonzero(target_full))
    return 1.0 if denominator == 0 else float(2.0 * intersection / denominator)


def bbox_to_slices(bbox: Sequence[Sequence[int]], shape: Sequence[int]) -> tuple[slice, ...]:
    if len(bbox) != 3 or len(shape) != 3:
        raise ValueError(f"expected 3-D bbox/shape, got {bbox}, {shape}")
    result = []
    for axis, ((start, stop), size) in enumerate(zip(bbox, shape)):
        start, stop, size = int(start), int(stop), int(size)
        if start < 0 or stop > size or stop <= start:
            raise ValueError(f"invalid bbox on axis {axis}: {(start, stop)} for size {size}")
        result.append(slice(start, stop))
    return tuple(result)


def _bias_field(shape: Sequence[int], rng: np.random.Generator, amplitude: float) -> np.ndarray:
    """Cheap smooth multiplicative field (constant/linear/quadratic, no deformation)."""
    field = np.zeros(tuple(int(v) for v in shape), dtype=np.float32)
    for axis, size in enumerate(shape):
        coordinate = np.linspace(-1.0, 1.0, int(size), dtype=np.float32)
        view_shape = [1, 1, 1]
        view_shape[axis] = int(size)
        coordinate = coordinate.reshape(view_shape)
        linear, quadratic = rng.normal(0.0, 0.35, size=2)
        field += linear * coordinate + quadratic * (coordinate * coordinate - 1.0 / 3.0)
    maximum = float(np.max(np.abs(field)))
    if maximum > 0:
        field *= float(amplitude) / maximum
    return 1.0 + field


def weak_augment(
    preprocessed: torch.Tensor,
    rng: np.random.Generator,
    noise_std_max: float = 0.03,
    bias_max: float = 0.05,
) -> torch.Tensor:
    """Apply only anatomy-preserving intensity perturbations to normalized data."""
    array = preprocessed.detach().cpu().numpy().astype(np.float32, copy=True)
    if array.ndim != 4 or array.shape[0] != 1:
        raise ValueError(f"expected single-channel (C,D,H,W), got {array.shape}")

    # Mild scale/offset/contrast and signed gamma preserve spatial correspondence.
    scale = rng.uniform(0.95, 1.05)
    offset = rng.uniform(-0.05, 0.05)
    contrast = rng.uniform(0.90, 1.10)
    gamma = rng.uniform(0.90, 1.10)
    spatial = array[0]
    mean = float(np.mean(spatial, dtype=np.float64))
    spatial = (spatial - mean) * contrast + mean
    spatial = np.sign(spatial) * np.power(np.abs(spatial), gamma)
    spatial = (spatial * scale + offset) * _bias_field(spatial.shape, rng, bias_max)
    noise_std = rng.uniform(0.0, noise_std_max)
    spatial += rng.normal(0.0, noise_std, size=spatial.shape).astype(np.float32)
    return torch.from_numpy(spatial[None].astype(np.float32, copy=False))


def _case_rng(seed: int, case_id: str) -> np.random.Generator:
    digest = hashlib.sha256(case_id.encode("utf-8")).digest()
    case_bits = int.from_bytes(digest[:8], "little")
    return np.random.default_rng(np.uint64(seed) ^ np.uint64(case_bits))


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks for ties, equivalent to scipy.stats.rankdata(method='average')."""
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0 + 1.0
        start = stop
    return ranks


def correlation(x: Iterable[float], y: Iterable[float], rank: bool = False) -> tuple[float, int]:
    first = np.asarray(list(x), dtype=np.float64)
    second = np.asarray(list(y), dtype=np.float64)
    valid = np.isfinite(first) & np.isfinite(second)
    first, second = first[valid], second[valid]
    if first.size < 2 or np.std(first) == 0 or np.std(second) == 0:
        return float("nan"), int(first.size)
    if rank:
        first, second = _rankdata(first), _rankdata(second)
    return float(np.corrcoef(first, second)[0, 1]), int(first.size)


def classification_metrics(
    uncertainty: Iterable[float], dice: Iterable[float], quality_threshold: float
) -> tuple[float, float, int]:
    """AUROC/AUPRC for high quality, using -uncertainty as the decision score."""
    uncertainty = np.asarray(list(uncertainty), dtype=np.float64)
    dice = np.asarray(list(dice), dtype=np.float64)
    valid = np.isfinite(uncertainty) & np.isfinite(dice)
    labels = dice[valid] > float(quality_threshold)
    if valid.sum() < 2 or np.unique(labels).size < 2:
        return float("nan"), float("nan"), int(valid.sum())
    from sklearn.metrics import average_precision_score, roc_auc_score

    quality_score = -uncertainty[valid]
    return (
        float(roc_auc_score(labels, quality_score)),
        float(average_precision_score(labels, quality_score)),
        int(valid.sum()),
    )


def normalize_cue_case_id(value: Any) -> str:
    return case_id_from_path(str(value).strip())


def load_cue_scores(
    path: Path | None, case_column: str | None, score_column: str | None
) -> tuple[dict[str, float], str | None, str | None]:
    if path is None:
        return {}, None, None
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if case_column is None:
            case_column = next((x for x in ("case_id", "filename", "case", "id") if x in fields), None)
        if score_column is None:
            score_column = next(
                (x for x in ("cue_uncertainty", "cue_score", "uncertainty", "quality_score") if x in fields),
                None,
            )
        if case_column not in fields or score_column not in fields:
            raise ValueError(
                f"CUE CSV columns not found; fields={fields}, case={case_column}, score={score_column}"
            )
        result: dict[str, float] = {}
        for row in reader:
            case_id = normalize_cue_case_id(row[case_column])
            if case_id in result:
                raise ValueError(f"duplicate CUE case ID: {case_id}")
            try:
                result[case_id] = float(row[score_column])
            except (TypeError, ValueError):
                result[case_id] = float("nan")
    return result, case_column, score_column


def _find_cases(
    images: Path, labels: Path, split_json: Path | None, split_key: str
) -> list[tuple[str, Path, Path]]:
    if split_json is not None:
        with split_json.open("r", encoding="utf-8") as handle:
            split = json.load(handle)
        if split_key not in split or not isinstance(split[split_key], list):
            raise ValueError(f"split JSON has no list key {split_key!r}")
        image_paths = [Path(item) for item in split[split_key]]
    else:
        image_paths = sorted(images.glob("*.nii.gz"))
    result = []
    for image_path in image_paths:
        if not image_path.is_absolute():
            image_path = images / image_path.name
        case_id = case_id_from_path(image_path)
        label_path = labels / image_path.name
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        if not label_path.is_file():
            raise FileNotFoundError(label_path)
        result.append((case_id, image_path, label_path))
    return result


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _predict_probabilities(
    predictor: Any, image: torch.Tensor, embeddings: torch.Tensor
) -> np.ndarray:
    """Reuse the existing Gaussian sliding-window inference and return crop probabilities."""
    with torch.inference_mode():
        logits = predictor.predict_sliding_window_return_logits(image, embeddings)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        probabilities = torch.sigmoid(logits.float()).cpu().numpy()
    return probabilities.astype(np.float32, copy=False)


def run_case(
    predictor: Any,
    reader: Any,
    prompt_embeddings: torch.Tensor,
    baseline_embedding: torch.Tensor,
    case: tuple[str, Path, Path],
    cue_scores: dict[str, float],
    cue_direction: str,
    image_views: int,
    seed: int,
    threshold: float,
) -> dict[str, Any]:
    case_id, image_path, label_path = case
    image, _ = reader.read_images([str(image_path)])
    preprocessed, bbox, original_shape = predictor.preprocess(image)
    crop_slices = bbox_to_slices(bbox, original_shape)

    # Original image + four prompts share a single forward pass. Index zero is
    # the baseline liver-cancer prediction and also the identity image view.
    prompt_probabilities = _predict_probabilities(predictor, preprocessed, prompt_embeddings)
    if prompt_probabilities.shape[0] != len(PROMPTS):
        raise RuntimeError(f"expected {len(PROMPTS)} prompt outputs, got {prompt_probabilities.shape}")

    image_probabilities = [prompt_probabilities[0]]
    rng = _case_rng(seed, case_id)
    for _ in range(1, image_views):
        augmented = weak_augment(preprocessed, rng)
        image_probability = _predict_probabilities(predictor, augmented, baseline_embedding)
        image_probabilities.append(image_probability[0])

    image_pairwise = mean_pairwise_soft_dice(image_probabilities)
    prompt_pairwise = mean_pairwise_soft_dice(list(prompt_probabilities))

    # Load GT only after every uncertainty value has been determined. This
    # makes the no-GT uncertainty boundary explicit rather than merely
    # relying on call-site discipline.
    label, _ = reader.read_images([str(label_path)])
    if image.shape[1:] != label.shape[1:]:
        raise ValueError(f"{case_id}: image/GT shape mismatch: {image.shape} vs {label.shape}")
    target = np.asarray(label[0]) > 0
    prompt_dices = [
        binary_dice_from_crop(probability, target, crop_slices, threshold)
        for probability in prompt_probabilities
    ]
    cue_score = cue_scores.get(case_id, float("nan"))
    cue_uncertainty = cue_score if cue_direction == "uncertainty" else -cue_score
    return {
        "case_id": case_id,
        "baseline_dice": prompt_dices[0],
        "cue_score": cue_score,
        "cue_uncertainty": cue_uncertainty,
        "image_pairwise_dice": image_pairwise,
        "image_uncertainty": 1.0 - image_pairwise,
        "image_variance": mean_voxel_variance(image_probabilities),
        "prompt_pairwise_dice": prompt_pairwise,
        "prompt_uncertainty": 1.0 - prompt_pairwise,
        "prompt_variance": mean_voxel_variance(list(prompt_probabilities)),
        **dict(zip(PROMPT_DICE_COLUMNS, prompt_dices)),
    }


def summarize(rows: Sequence[dict[str, Any]], quality_threshold: float) -> dict[str, Any]:
    dice = [float(row["baseline_dice"]) for row in rows]
    summary: dict[str, Any] = {}
    for column in UNCERTAINTY_COLUMNS:
        values = [float(row[column]) for row in rows]
        pearson, correlation_cases = correlation(values, dice)
        spearman, _ = correlation(values, dice, rank=True)
        auroc, auprc, classification_cases = classification_metrics(
            values, dice, quality_threshold
        )
        summary[column] = {
            "pearson_vs_baseline_dice": pearson,
            "spearman_vs_baseline_dice": spearman,
            "auroc_high_quality": auroc,
            "auprc_high_quality": auprc,
            "correlation_cases": correlation_cases,
            "classification_cases": classification_cases,
        }
    return summary


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def make_plots(rows: Sequence[dict[str, Any]], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dice = np.asarray([float(row["baseline_dice"]) for row in rows])
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    labels = ("CUE", "Image", "Prompt")
    for axis, column, label in zip(axes, UNCERTAINTY_COLUMNS, labels):
        values = np.asarray([float(row[column]) for row in rows])
        valid = np.isfinite(values) & np.isfinite(dice)
        axis.scatter(values[valid], dice[valid], alpha=0.75, s=24)
        axis.set_xlabel(f"{label} uncertainty")
        axis.set_ylabel("Baseline Dice")
        axis.set_title(f"{label} uncertainty vs Dice (n={int(valid.sum())})")
        axis.grid(alpha=0.25)
        if not valid.any():
            axis.text(0.5, 0.5, "unavailable", ha="center", va="center", transform=axis.transAxes)
    figure.savefig(output_dir / "uncertainty_vs_dice.png", dpi=180)
    plt.close(figure)

    image_uncertainty = np.asarray([float(row["image_uncertainty"]) for row in rows])
    prompt_uncertainty = np.asarray([float(row["prompt_uncertainty"]) for row in rows])
    valid = np.isfinite(image_uncertainty) & np.isfinite(prompt_uncertainty) & np.isfinite(dice)
    figure, axis = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)
    scatter = axis.scatter(
        image_uncertainty[valid], prompt_uncertainty[valid], c=dice[valid],
        cmap="viridis", vmin=0.0, vmax=1.0, alpha=0.8, s=30,
    )
    axis.set_xlabel("Image uncertainty")
    axis.set_ylabel("Prompt uncertainty")
    axis.set_title("Image vs prompt uncertainty")
    axis.grid(alpha=0.25)
    figure.colorbar(scatter, ax=axis, label="Baseline Dice")
    figure.savefig(output_dir / "image_vs_prompt_uncertainty.png", dpi=180)
    plt.close(figure)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Frozen VoxTell image/prompt uncertainty diagnostic for DWI liver cancer"
    )
    parser.add_argument("--images", required=True, help="DWI image directory")
    parser.add_argument("--labels", required=True, help="binary liver-cancer GT directory")
    parser.add_argument("--model", required=True, help="VoxTell model directory")
    parser.add_argument("--text-model", default="model/Qwen3-Embedding-4B")
    parser.add_argument(
        "--text-device", default="cpu",
        help="device used only while embedding the four fixed prompts (default: cpu)",
    )
    parser.add_argument(
        "--text-dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16",
    )
    parser.add_argument("--split-json", default=None)
    parser.add_argument("--split-key", default="test_cases")
    parser.add_argument("--cue-csv", default=None, help="existing CUE-SFDA Stage 1 per-case CSV")
    parser.add_argument("--cue-case-column", default=None)
    parser.add_argument("--cue-score-column", default=None)
    parser.add_argument(
        "--cue-direction", choices=("uncertainty", "quality"), default="uncertainty",
        help="meaning of the raw CUE score; quality is negated only for cue_uncertainty",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image-views", type=int, default=6,
                        help="total views including identity (default: 6)")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--quality-threshold", type=float, default=0.7)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", default="diagnostic/dwi_uncertainty")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.image_views < 2:
        raise ValueError("--image-views must be at least 2")
    if not 0.0 <= args.threshold <= 1.0 or not 0.0 <= args.quality_threshold <= 1.0:
        raise ValueError("thresholds must be in [0,1]")
    images, labels, model = Path(args.images), Path(args.labels), Path(args.model)
    text_model = Path(args.text_model)
    for path in (images, labels, model, text_model):
        if not path.exists():
            raise FileNotFoundError(path)
    split_json = Path(args.split_json) if args.split_json else None
    cue_csv = Path(args.cue_csv) if args.cue_csv else None
    for path in (split_json, cue_csv):
        if path is not None and not path.is_file():
            raise FileNotFoundError(path)

    cases = _find_cases(images, labels, split_json, args.split_key)
    if args.limit is not None:
        cases = cases[:args.limit]
    if not cases:
        raise RuntimeError("no image/label pairs found")
    cue_scores, cue_case_column, cue_score_column = load_cue_scores(
        cue_csv, args.cue_case_column, args.cue_score_column
    )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA unavailable for --device {args.device}")
    from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
    from voxtell.inference.predictor import VoxTellPredictor

    text_device = torch.device(args.text_device)
    if text_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA unavailable for --text-device {args.text_device}")
    text_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.text_dtype]
    predictor = VoxTellPredictor(
        model_dir=str(model), device=device, text_encoding_model=str(text_model),
        text_device=text_device, text_dtype=text_dtype,
    )
    predictor.network.eval()
    predictor.network.requires_grad_(False)
    predictor.text_backbone.eval()
    predictor.text_backbone.requires_grad_(False)
    if predictor.network.training or any(parameter.requires_grad for parameter in predictor.network.parameters()):
        raise RuntimeError("failed to freeze VoxTell network")

    # Embed once. The first slice is reused for every augmented image view.
    prompt_embeddings = predictor.embed_text_prompts(list(PROMPTS))
    baseline_embedding = prompt_embeddings[:, :1].contiguous()
    predictor.network = predictor.network.to(device).eval()
    reader = NibabelIOWithReorient()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "per_case_uncertainty.csv"
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases, 1):
        print(f"[{index}/{len(cases)}] {case[0]}", flush=True)
        row = run_case(
            predictor, reader, prompt_embeddings, baseline_embedding, case,
            cue_scores, args.cue_direction, args.image_views, args.seed, args.threshold,
        )
        rows.append(row)
        _write_csv(output_csv, rows)  # checkpoint long DWI runs after every case

    statistics = summarize(rows, args.quality_threshold)
    summary = _json_safe({
        "cases": len(rows),
        "prompts": list(PROMPTS),
        "protocol": {
            "model_updates": False,
            "network_frozen": True,
            "same_model_instance_for_all_cases_and_views": True,
            "image_views_total_including_identity": args.image_views,
            "image_augmentations": [
                "mild intensity scale/offset", "mild contrast", "signed gamma",
                "Gaussian noise", "smooth mild multiplicative bias field",
            ],
            "uncertainty_definition": "1 - mean pairwise soft Dice",
            "variance_definition": "mean voxel-wise population variance of probabilities",
            "variance_domain": "VoxTell nonzero crop",
            "ground_truth_use": "offline Dice only; never used by uncertainty",
            "high_quality_label": f"baseline_dice > {args.quality_threshold}",
            "classification_score": "negative uncertainty (higher means predicted quality)",
        },
        "cue": {
            "repository_stage1_implementation_found": False,
            "source_csv": str(cue_csv.resolve()) if cue_csv else None,
            "case_column": cue_case_column,
            "score_column": cue_score_column,
            "raw_score_direction": args.cue_direction,
            "matched_cases": sum(np.isfinite(float(row["cue_score"])) for row in rows),
            "note": "No proxy substituted when external CUE scores are absent.",
        },
        "statistics": statistics,
    })
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=False)
    make_plots(rows, output_dir)

    print("\nmetric                 Pearson   Spearman   AUROC     AUPRC     n", flush=True)
    for column in UNCERTAINTY_COLUMNS:
        item = statistics[column]
        print(
            f"{column:22s} {item['pearson_vs_baseline_dice']:9.4f} "
            f"{item['spearman_vs_baseline_dice']:10.4f} "
            f"{item['auroc_high_quality']:9.4f} {item['auprc_high_quality']:9.4f} "
            f"{item['correlation_cases']:5d}",
            flush=True,
        )
    print(f"\nSaved: {output_dir.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
