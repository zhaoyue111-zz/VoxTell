#!/usr/bin/env python3
"""Offline text-causal pseudo-label quality audit for the fixed ``liver`` prompt.

The experiment changes only the transformer decoder's memory-key visibility.  It
does not update parameters, alter images/text/encoder skips, or participate in
SFDA training.  D5 is the model output at list index 0 (highest resolution).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F


PROMPT = "liver"
CSV_FIELDS = [
    "case_id", "dice", "precision", "recall", "fp_ratio", "fn_ratio",
    "high_confidence_fp_ratio", "q_similarity_in", "q_similarity_out",
    "q_causal_score", "pred_softdice_in", "pred_softdice_out",
    "pred_causal_score", "entropy", "cac", "d25_consistency",
    "memory_shape_dhw", "memory_tokens", "attention_blocked_mass_in",
    "attention_blocked_mass_out", "attention_visible_mass_in",
    "attention_visible_mass_out", "attention_fallback_in",
    "attention_fallback_out",
]
CORRELATION_TARGETS = ("dice", "precision", "recall", "high_confidence_fp_ratio")
UNSUPERVISED_SCORES = (
    "q_similarity_in", "q_similarity_out", "q_causal_score",
    "pred_softdice_in", "pred_softdice_out", "pred_causal_score",
    "entropy", "cac", "d25_consistency",
)


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else float("nan")


def binary_case_metrics(probability: np.ndarray, target: np.ndarray,
                        threshold: float = 0.5,
                        high_confidence_threshold: float = 0.9) -> dict[str, float]:
    prediction = np.asarray(probability) >= float(threshold)
    target = np.asarray(target, dtype=bool)
    if prediction.shape != target.shape:
        raise ValueError(f"prediction/GT shape mismatch: {prediction.shape} vs {target.shape}")
    tp = int(np.count_nonzero(prediction & target))
    fp = int(np.count_nonzero(prediction & ~target))
    fn = int(np.count_nonzero(~prediction & target))
    high = np.asarray(probability) >= float(high_confidence_threshold)
    high_fp = int(np.count_nonzero(high & ~target))
    high_count = int(np.count_nonzero(high))
    return {
        "dice": safe_ratio(2 * tp, 2 * tp + fp + fn),
        "precision": safe_ratio(tp, tp + fp),
        "recall": safe_ratio(tp, tp + fn),
        "fp_ratio": safe_ratio(fp, tp + fp),
        "fn_ratio": safe_ratio(fn, tp + fn),
        "high_confidence_fp_ratio": safe_ratio(high_fp, high_count),
    }


def soft_dice(probability_a: torch.Tensor, probability_b: torch.Tensor,
              epsilon: float = 1e-8) -> torch.Tensor:
    """Differentiable Dice with a finite empty/empty convention."""
    a = probability_a.float().reshape(-1)
    b = probability_b.float().reshape(-1)
    denominator = a.sum() + b.sum()
    score = (2.0 * (a * b).sum()) / denominator.clamp_min(float(epsilon))
    return torch.where(denominator > float(epsilon), score, torch.ones_like(score))


def align_gt_nearest(gt: np.ndarray, target_shape: Sequence[int]) -> np.ndarray:
    gt = np.asarray(gt)
    shape = tuple(int(v) for v in target_shape)
    if gt.shape == shape:
        return gt.astype(bool, copy=False)
    tensor = torch.from_numpy(gt.astype(np.float32))[None, None]
    return F.interpolate(tensor, size=shape, mode="nearest")[0, 0].numpy().astype(bool)


def _safe_padding_mask(mask: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Return a bool key-padding mask and count deterministic all-blocked fallbacks."""
    mask = mask.to(dtype=torch.bool)
    if mask.ndim != 2:
        raise ValueError(f"expected (batch,tokens) mask, got {tuple(mask.shape)}")
    all_blocked = mask.all(dim=1)
    fallback_count = int(all_blocked.sum().item())
    if fallback_count:
        mask = mask.clone()
        mask[all_blocked, 0] = False
    return mask, fallback_count


def build_memory_masks(pseudo_mask: torch.Tensor,
                       memory_shape_dhw: Sequence[int]) -> dict[str, Any]:
    """Downsample D5 mask with nearest interpolation and flatten H,W,D order.

    ``VoxTellModel`` rearranges encoder memory from ``(B,C,D,H,W)`` to
    ``(H*W*D,B,C)``.  The permutation below must therefore precede flattening.
    """
    if pseudo_mask.ndim != 4:
        raise ValueError(f"expected pseudo mask (B,D,H,W), got {tuple(pseudo_mask.shape)}")
    dhw = tuple(int(v) for v in memory_shape_dhw)
    grid = F.interpolate(
        pseudo_mask.float().unsqueeze(1), size=dhw, mode="nearest"
    ).squeeze(1) >= 0.5
    inside_tokens = grid.permute(0, 2, 3, 1).reshape(grid.shape[0], -1)
    only_in, fallback_in = _safe_padding_mask(~inside_tokens)
    only_out, fallback_out = _safe_padding_mask(inside_tokens)
    return {
        "only_in": only_in,
        "only_out": only_out,
        "inside_tokens": inside_tokens,
        "fallback_in": fallback_in,
        "fallback_out": fallback_out,
        "memory_shape_dhw": dhw,
    }


def _safe_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    denominator = a.norm() * b.norm()
    if float(denominator) == 0.0:
        return 0.0
    value = torch.dot(a, b) / denominator
    return float(value) if torch.isfinite(value) else 0.0


def _attention_stats(attention: Any, blocked: torch.Tensor) -> tuple[float, float]:
    """Return mean attention mass on blocked and visible memory tokens."""
    if attention is None:
        return float("nan"), float("nan")
    weights = attention.detach().float()
    if weights.ndim == 4:  # tolerate (B, heads, target, memory)
        weights = weights.mean(dim=1)
    if weights.ndim != 3:
        return float("nan"), float("nan")
    blocked = blocked.to(device=weights.device)
    visible = ~blocked
    blocked_mass = weights.masked_select(blocked[:, None, :]).mean() if bool(blocked.any()) else torch.tensor(0.0, device=weights.device)
    visible_mass = weights.masked_select(visible[:, None, :]).mean() if bool(visible.any()) else torch.tensor(0.0, device=weights.device)
    return float(blocked_mass.cpu()), float(visible_mass.cpu())


def _as_d5_logits(predictions: Any) -> torch.Tensor:
    if not isinstance(predictions, (list, tuple)):
        raise ValueError("diagnostic inference requires all decoder outputs")
    if len(predictions) < 1:
        raise ValueError("model returned no decoder outputs")
    return predictions[0]


def _require_finite(name: str, tensor: torch.Tensor) -> None:
    if not bool(torch.isfinite(tensor).all()):
        raise RuntimeError(f"{name} contains NaN/Inf during causal audit")


def _resize_logits(logits: torch.Tensor, spatial_shape: Sequence[int]) -> torch.Tensor:
    shape = tuple(int(v) for v in spatial_shape)
    if tuple(logits.shape[-3:]) == shape:
        return logits
    return F.interpolate(logits, size=shape, mode="trilinear", align_corners=False)


def _spearman(values: Iterable[float], target: Iterable[float]) -> float:
    values = np.asarray(list(values), dtype=np.float64)
    target = np.asarray(list(target), dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(target)
    if int(valid.sum()) < 2:
        return float("nan")
    from scipy.stats import rankdata
    first, second = rankdata(values[valid]), rankdata(target[valid])
    if np.std(first) == 0 or np.std(second) == 0:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def _pairwise_mask_dice(masks: Sequence[np.ndarray]) -> float:
    values = []
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            a, b = masks[i], masks[j]
            values.append(safe_ratio(2 * np.count_nonzero(a & b), np.count_nonzero(a) + np.count_nonzero(b)))
    return float(np.nanmean(values)) if values and np.isfinite(values).any() else float("nan")


def _run_causal_case(predictor: Any, image: torch.Tensor, text_embedding: torch.Tensor,
                     target: np.ndarray, threshold: float,
                     high_confidence_threshold: float) -> dict[str, Any]:
    from acvl_utils.cropping_and_padding.padding import pad_nd_image
    from nnunetv2.inference.sliding_window_prediction import compute_gaussian

    device = predictor.device
    network = predictor.network.to(device).eval()
    padded, revert_padding = pad_nd_image(
        image, predictor.patch_size, "constant", {"value": 0}, True, None
    )
    slicers = predictor._internal_get_sliding_window_slicers(padded.shape[1:])
    gaussian = compute_gaussian(
        tuple(predictor.patch_size), sigma_scale=1.0 / 8,
        value_scaling_factor=10, device=torch.device("cpu")
    ).float()
    padded_shape = tuple(int(v) for v in padded.shape[1:])
    n_outputs = len(network.decoder.stages)
    normal_sum = [torch.zeros(padded_shape, dtype=torch.float32) for _ in range(n_outputs)]
    in_sum = torch.zeros(padded_shape, dtype=torch.float32)
    out_sum = torch.zeros(padded_shape, dtype=torch.float32)
    denominator = torch.zeros(padded_shape, dtype=torch.float32)
    q_in, q_out, d_in, d_out = [], [], [], []
    attention_values = {"in": [], "out": []}
    fallback = {"in": 0, "out": 0}
    memory_shape = None

    for patch_index, slicer in enumerate(slicers):
        tile = padded[slicer][None].to(device)
        amp = torch.autocast(device_type=device.type, enabled=device.type == "cuda")
        with torch.inference_mode(), amp:
            normal_outputs, normal_diag = network(
                tile, text_embedding, return_decoder_outputs=True,
                return_diagnostics=True,
            )
        d5_normal = _as_d5_logits(normal_outputs)
        _require_finite("normal D5 logits", d5_normal)
        d5_probability = torch.sigmoid(d5_normal[:, 0].float())
        memory_shape = tuple(int(v) for v in normal_diag["memory_shape"])
        masks = build_memory_masks(d5_probability >= float(threshold), memory_shape)
        with torch.inference_mode(), amp:
            in_outputs, in_diag = network(
                tile, text_embedding, return_decoder_outputs=True,
                memory_key_padding_mask=masks["only_in"], return_diagnostics=True,
            )
            out_outputs, out_diag = network(
                tile, text_embedding, return_decoder_outputs=True,
                memory_key_padding_mask=masks["only_out"], return_diagnostics=True,
            )
        _require_finite("normal q", normal_diag["mask_embedding"])
        _require_finite("only-in q", in_diag["mask_embedding"])
        _require_finite("only-out q", out_diag["mask_embedding"])
        d5_in = _as_d5_logits(in_outputs)
        d5_out = _as_d5_logits(out_outputs)
        _require_finite("only-in D5 logits", d5_in)
        _require_finite("only-out D5 logits", d5_out)
        probability_in = torch.sigmoid(d5_in[:, 0].float())
        probability_out = torch.sigmoid(d5_out[:, 0].float())
        q = normal_diag["mask_embedding"][0, 0]
        q_in.append(_safe_cosine(q, in_diag["mask_embedding"][0, 0]))
        q_out.append(_safe_cosine(q, out_diag["mask_embedding"][0, 0]))
        d_in.append(float(soft_dice(d5_probability, probability_in).cpu()))
        d_out.append(float(soft_dice(d5_probability, probability_out).cpu()))
        attention_values["in"].append(_attention_stats(
            in_diag["cross_attention"][-1], masks["only_in"]
        ))
        attention_values["out"].append(_attention_stats(
            out_diag["cross_attention"][-1], masks["only_out"]
        ))
        fallback["in"] += masks["fallback_in"]
        fallback["out"] += masks["fallback_out"]

        tile_shape = tuple(int(s.stop - s.start) for s in slicer[1:])
        weight = gaussian
        for output_index, output in enumerate(normal_outputs):
            output = _resize_logits(output.float(), tile_shape)[0, 0].cpu()
            normal_sum[output_index][slicer[1:]] += output * weight
        in_patch = _resize_logits(d5_in.float(), tile_shape)[0, 0].cpu()
        out_patch = _resize_logits(d5_out.float(), tile_shape)[0, 0].cpu()
        in_sum[slicer[1:]] += in_patch * weight
        out_sum[slicer[1:]] += out_patch * weight
        denominator[slicer[1:]] += weight

    crop = revert_padding[1:]
    denom = denominator[crop].clamp_min(torch.finfo(torch.float32).eps)
    normal_logits = [value[crop] / denom for value in normal_sum]
    in_logits, out_logits = in_sum[crop] / denom, out_sum[crop] / denom
    normal_probabilities = [torch.sigmoid(value) for value in normal_logits]
    probability = normal_probabilities[0].numpy()
    target = align_gt_nearest(target, probability.shape)
    metrics = binary_case_metrics(probability, target, threshold, high_confidence_threshold)
    d25_masks = [(p.numpy() >= float(threshold)) for p in normal_probabilities[:4]]
    probability_tensor = torch.from_numpy(probability).clamp(1e-6, 1 - 1e-6)
    entropy = -(probability_tensor * probability_tensor.log() +
                (1 - probability_tensor) * (1 - probability_tensor).log()).mean()
    attention = {
        side: {
            "blocked": float(np.nanmean([v[0] for v in values])) if values else float("nan"),
            "visible": float(np.nanmean([v[1] for v in values])) if values else float("nan"),
        }
        for side, values in attention_values.items()
    }
    row = {
        "dice": metrics["dice"], "precision": metrics["precision"],
        "recall": metrics["recall"], "fp_ratio": metrics["fp_ratio"],
        "fn_ratio": metrics["fn_ratio"],
        "high_confidence_fp_ratio": metrics["high_confidence_fp_ratio"],
        "q_similarity_in": float(np.nanmean(q_in)),
        "q_similarity_out": float(np.nanmean(q_out)),
        "q_causal_score": float(np.nanmean(q_in) - np.nanmean(q_out)),
        "pred_softdice_in": float(np.nanmean(d_in)),
        "pred_softdice_out": float(np.nanmean(d_out)),
        "pred_causal_score": float(np.nanmean(d_in) - np.nanmean(d_out)),
        "entropy": float(entropy),
        # CAC is not implemented in this VoxTell repository.  Do not replace it
        # with an unrelated confidence proxy; correlation is reported as NaN.
        "cac": float("nan"),
        "d25_consistency": _pairwise_mask_dice(d25_masks),
        "memory_shape_dhw": json.dumps(list(memory_shape)),
        "memory_tokens": int(np.prod(memory_shape)),
        "attention_blocked_mass_in": attention["in"]["blocked"],
        "attention_blocked_mass_out": attention["out"]["blocked"],
        "attention_visible_mass_in": attention["in"]["visible"],
        "attention_visible_mass_out": attention["out"]["visible"],
        "attention_fallback_in": fallback["in"],
        "attention_fallback_out": fallback["out"],
    }
    return row


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    correlations = {}
    for score in UNSUPERVISED_SCORES:
        correlations[score] = {
            target: {
                "spearman": _spearman(
                    [row[score] for row in rows], [row[target] for row in rows]
                ),
                "valid_cases": int(sum(
                    np.isfinite(row[score]) and np.isfinite(row[target]) for row in rows
                )),
            }
            for target in CORRELATION_TARGETS
        }
    return _json_safe({
        "prompt": PROMPT,
        "d5": {
            "name": "Decoder 5",
            "model_output_list_index": 0,
            "role": "final/highest-resolution output",
        },
        "memory": {
            "token_order": "H,W,D C-order after (B,C,D,H,W)->(B,H,W,D,C)",
            "mask_interpolation": "nearest",
            "key_padding_mask_true": "blocked",
        },
        "cac": {
            "available": False,
            "reason": "CAC implementation is not present in the VoxTell repository; no proxy substituted",
        },
        "correlations": correlations,
        "cases": len(rows),
    })


def _find_cases(images: Path, labels: Path) -> list[tuple[str, Path, Path]]:
    image_paths = sorted(images.glob("*.nii.gz")) if images.is_dir() else sorted(images.parent.glob(images.name))
    result = []
    for image in image_paths:
        label = labels / image.name
        if label.is_file():
            result.append((image.name.removesuffix(".nii.gz"), image, label))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline liver text-causal pseudo-label quality audit")
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--text-model", default="Qwen/Qwen3-Embedding-4B")
    parser.add_argument("--gt-label", type=int, default=5,
                        help="Liver label value; use 0 to treat all non-zero labels as liver")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--high-confidence-threshold", type=float, default=0.9)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-csv", default="output/text_causal_quality_per_case.csv")
    parser.add_argument("--summary-json", default="output/text_causal_quality_summary.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0 <= args.threshold <= 1 or not 0 <= args.high_confidence_threshold <= 1:
        raise ValueError("thresholds must be in [0,1]")
    if args.high_confidence_threshold < args.threshold:
        raise ValueError("high-confidence threshold must be >= threshold")
    from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
    from voxtell.inference.predictor_multiclass import VoxTellPredictor

    device = torch.device(f"cuda:{args.gpu}" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    cases = _find_cases(Path(args.images), Path(args.labels))
    if args.limit is not None:
        cases = cases[:args.limit]
    if not cases:
        raise RuntimeError("No image/label pairs found")
    predictor = VoxTellPredictor(str(args.model), device=device, text_encoding_model=args.text_model)
    reader = NibabelIOWithReorient()
    text_embedding = predictor.embed_text_prompts([PROMPT])
    rows = []
    for case_id, image_path, label_path in cases:
        image, _ = reader.read_images([str(image_path)])
        label, _ = reader.read_images([str(label_path)])
        image_tensor, bbox, _ = predictor.preprocess(image)
        labels = np.rint(label[0]).astype(np.int64)
        gt_full = labels != 0 if args.gt_label == 0 else labels == args.gt_label
        gt_crop = gt_full[tuple(slice(int(lo), int(hi)) for lo, hi in bbox)]
        row = _run_causal_case(
            predictor, image_tensor, text_embedding, gt_crop,
            args.threshold, args.high_confidence_threshold,
        )
        row["case_id"] = case_id
        rows.append(row)
        print(f"[{len(rows)}/{len(cases)}] {case_id} Dice={row['dice']:.4f} Qcausal={row['q_causal_score']:.4f}")
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    summary["threshold"] = args.threshold
    summary["high_confidence_threshold"] = args.high_confidence_threshold
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
