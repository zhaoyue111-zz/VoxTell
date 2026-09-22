#!/usr/bin/env python3
"""Analyse decoder probability distributions in GT-defined regions.

The predictor returns decoder outputs from highest to lowest spatial
resolution.  This script deliberately maps those outputs back to the model's
actual low-to-high decoder order: D1 is the earliest (usually 12^3) output and
D5 is the final (usually 192^3) output.

The D1-D4 maps are not raw low-resolution maps: each decoder head is
upsampled inside a patch and then fused over the full image by sliding-window
inference. D5 is the final full-volume sliding-window output. Only
per-region statistics are retained, and each decoder volume is released after
its row has been computed.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


DECODER_COUNT = 5
REGIONS = (
    "gt_foreground", "gt_background_valid", "gt_background_all",
    "d5_false_negative",
)
SUMMARY_REGIONS = ("gt_foreground", "gt_background_valid", "d5_false_negative")
STAT_FIELDS = ("voxel_count", "mean", "std", "min", "p05", "p25",
               "median", "p75", "p95", "max")
PER_CASE_FIELDS = (
    "case", "prompt", "decoder_stage", "is_final_output",
    "model_output_list_index", "internal_stage_index", "encoder_skip_index",
    "upsampling_order", "raw_shape", "aligned_shape", "interpolation",
    "probability_shape", "probability_map_semantics", "valid_mask_source",
    "gt_interpolation", "region", "threshold", *STAT_FIELDS,
)
SUMMARY_FIELDS = (
    "case", "case_count", "prompt", "decoder_stage", "is_final_output",
    "raw_shape", "aligned_shape", "probability_shape",
    "probability_map_semantics", "valid_mask_source", "interpolation",
    "gt_interpolation", "region", "threshold", *STAT_FIELDS,
)


def decoder_metadata(
    observed_output_shapes: Sequence[Sequence[int]] | None = None,
    aligned_shape: Sequence[int] | None = None,
    count: int = DECODER_COUNT,
) -> list[dict[str, Any]]:
    """Return D1..D5 metadata in actual low-to-high upsampling order.

    ``observed_output_shapes`` follows the predictor's returned list order
    (highest resolution first), just like ``return_all_layers=True``.
    """
    if count != DECODER_COUNT:
        raise ValueError(f"expected {DECODER_COUNT} decoder stages, got {count}")
    if observed_output_shapes is not None and len(observed_output_shapes) != count:
        raise ValueError("one observed shape is required for every decoder output")

    aligned = list(map(int, aligned_shape)) if aligned_shape is not None else None
    result = []
    for stage in range(1, count + 1):
        list_index = count - stage
        raw = (list(map(int, observed_output_shapes[list_index]))
               if observed_output_shapes is not None else None)
        result.append({
            "decoder_stage": f"D{stage}",
            "stage_number": stage,
            "model_output_list_index": list_index,
            "internal_stage_index": stage - 1,
            "encoder_skip_index": count - stage,
            "upsampling_order": stage,
            "is_final_output": stage == count,
            "raw_shape": raw,
            "aligned_shape": aligned,
        })
    return result


def _as_spatial_array(array: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(array, dtype=np.float32)
    if value.ndim != 3:
        raise ValueError(f"{name} must be a 3-D spatial array, got {value.shape}")
    return value


def align_probability_trilinear(probability: np.ndarray,
                                target_shape: Sequence[int]) -> np.ndarray:
    """Interpolate one probability map to ``target_shape`` in float32.

    The predictor already returns a full-volume map after patch upsampling and
    sliding-window fusion. Avoid a no-op interpolation when GT and prediction
    already have the same shape.
    """
    import torch
    import torch.nn.functional as F

    probability = _as_spatial_array(probability, "probability")
    target = tuple(int(v) for v in target_shape)
    if len(target) != 3 or any(v < 1 for v in target):
        raise ValueError(f"target_shape must contain three positive dimensions, got {target}")
    if probability.shape == target:
        return probability
    value = torch.from_numpy(probability)[None, None]
    aligned = F.interpolate(value, size=target, mode="trilinear", align_corners=False)
    return aligned[0, 0].numpy().astype(np.float32, copy=False)


def align_gt_nearest(gt: np.ndarray, target_shape: Sequence[int]) -> np.ndarray:
    """Interpolate the GT mask to ``target_shape`` with nearest-neighbour mode."""
    import torch
    import torch.nn.functional as F

    gt = _as_spatial_array(gt, "gt")
    target = tuple(int(v) for v in target_shape)
    if gt.shape == target:
        return gt.astype(bool, copy=False)
    value = torch.from_numpy(gt.astype(np.float32, copy=False))[None, None]
    aligned = F.interpolate(value, size=target, mode="nearest")
    return aligned[0, 0].numpy().astype(bool, copy=False)


def probability_stats(values: np.ndarray | Iterable[float]) -> dict[str, float | int]:
    """Calculate exact finite-value statistics without retaining other regions."""
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"voxel_count": 0, **{name: float("nan") for name in STAT_FIELDS[1:]}}
    quantiles = np.percentile(values, [5, 25, 50, 75, 95]).astype(np.float32)
    return {
        "voxel_count": int(values.size),
        "mean": float(np.mean(values, dtype=np.float32)),
        "std": float(np.std(values, dtype=np.float32)),
        "min": float(np.min(values)),
        "p05": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p95": float(quantiles[4]),
        "max": float(np.max(values)),
    }


def _json_shape(shape: Sequence[int] | None) -> str:
    return json.dumps(list(map(int, shape))) if shape is not None else ""


def _finite_mean(values: Iterable[Any]) -> float:
    values = np.asarray([v for v in values if v is not None], dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else float("nan")


def analyze_case(
    probabilities_in_returned_order: Sequence[np.ndarray],
    gt: np.ndarray,
    *,
    case: str = "",
    prompt: str = "liver",
    threshold: float = 0.5,
    metadata: Sequence[Mapping[str, Any]] | None = None,
    valid_inference_mask: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Return per-region rows without retaining aligned probability maps.

    The input probability sequence must use the predictor's returned order
    (high-to-low).  D1..D5 mapping is performed internally and is never based
    on the sequence's apparent stage name or on ``decoder_metrics.csv``.
    """
    if len(probabilities_in_returned_order) != DECODER_COUNT:
        raise ValueError(f"expected {DECODER_COUNT} decoder probabilities")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")

    raw = [_as_spatial_array(p, f"decoder output {i}")
           for i, p in enumerate(probabilities_in_returned_order)]
    gt_array = _as_spatial_array(gt, "gt")
    target_shape = gt_array.shape
    gt_aligned = align_gt_nearest(gt_array, target_shape)
    if valid_inference_mask is None:
        valid_mask_aligned = np.ones(target_shape, dtype=bool)
        valid_mask_source = "all_voxels_default"
    else:
        valid_mask = _as_spatial_array(valid_inference_mask, "valid_inference_mask")
        valid_mask_aligned = align_gt_nearest(valid_mask, target_shape)
        valid_mask_source = "predictor_crop_to_nonzero_bbox"

    md = list(metadata) if metadata is not None else decoder_metadata(
        [p.shape for p in raw], target_shape
    )
    if len(md) != DECODER_COUNT:
        raise ValueError(f"expected metadata for {DECODER_COUNT} decoder stages")

    # D5 alone defines the FN region.  Keep only this boolean mask, not D5's
    # full probability map, after the D5 map has been used for its own row.
    d5_info = md[DECODER_COUNT - 1]
    d5_raw = raw[int(d5_info["model_output_list_index"])]
    d5_aligned = align_probability_trilinear(d5_raw, target_shape)
    regions = {
        "gt_foreground": gt_aligned,
        "gt_background_valid": (~gt_aligned) & valid_mask_aligned,
        "gt_background_all": ~gt_aligned,
        "d5_false_negative": gt_aligned & (d5_aligned < np.float32(threshold)),
    }

    rows: list[dict[str, Any]] = []
    for info in md:
        stage = int(info["stage_number"])
        list_index = int(info["model_output_list_index"])
        probability = d5_aligned if stage == DECODER_COUNT else align_probability_trilinear(
            raw[list_index], target_shape
        )
        probability_interpolation = (
            "none_same_size" if raw[list_index].shape == target_shape else "trilinear"
        )
        probability_semantics = (
            "final_output_sliding_window_fused_full_volume"
            if stage == DECODER_COUNT
            else "patch_upsampled_then_sliding_window_fused_full_volume"
        )
        base = {
            "case": case,
            "prompt": prompt,
            "decoder_stage": info["decoder_stage"],
            "is_final_output": bool(info["is_final_output"]),
            "model_output_list_index": list_index,
            "internal_stage_index": info["internal_stage_index"],
            "encoder_skip_index": info["encoder_skip_index"],
            "upsampling_order": info["upsampling_order"],
            "raw_shape": _json_shape(info.get("raw_shape") or raw[list_index].shape),
            "aligned_shape": _json_shape(target_shape),
            "probability_shape": _json_shape(raw[list_index].shape),
            "probability_map_semantics": probability_semantics,
            "valid_mask_source": valid_mask_source,
            "interpolation": probability_interpolation,
            "gt_interpolation": "nearest",
            "threshold": float(threshold),
        }
        for region_name, region_mask in regions.items():
            row = dict(base)
            row["region"] = region_name
            row.update(probability_stats(probability[region_mask]))
            rows.append(row)
        if stage != DECODER_COUNT:
            del probability
    del d5_aligned
    return rows


def summarize_rows(
    rows: Sequence[Mapping[str, Any]],
    regions: Sequence[str] = SUMMARY_REGIONS,
) -> list[dict[str, Any]]:
    """Average case-level rows, so each case has equal weight.

    ``gt_background_all`` remains available in the per-case CSV but is not
    included by default because it contains restored zero-filled voxels
    outside the predictor's crop-to-nonzero inference region.
    """
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["region"] not in regions:
            continue
        groups[(str(row["decoder_stage"]), str(row["region"]))].append(row)

    summary: list[dict[str, Any]] = []
    for (stage, region), group in sorted(groups.items(), key=lambda item: (
            int(item[0][0][1:]), item[0][1])):
        first = group[0]
        result: dict[str, Any] = {
            "case": "__case_mean__",
            "case_count": len(group),
            "prompt": first.get("prompt", ""),
            "decoder_stage": stage,
            "is_final_output": first.get("is_final_output", ""),
            "raw_shape": (first.get("raw_shape", "")
                          if len({str(r.get("raw_shape", "")) for r in group}) == 1
                          else "mixed"),
            "aligned_shape": (first.get("aligned_shape", "")
                              if len({str(r.get("aligned_shape", "")) for r in group}) == 1
                              else "mixed"),
            "probability_shape": (first.get("probability_shape", "")
                                  if len({str(r.get("probability_shape", "")) for r in group}) == 1
                                  else "mixed"),
            "probability_map_semantics": first.get("probability_map_semantics", ""),
            "valid_mask_source": first.get("valid_mask_source", ""),
            "interpolation": first.get("interpolation", "trilinear"),
            "gt_interpolation": first.get("gt_interpolation", "nearest"),
            "region": region,
            "threshold": _finite_mean(r.get("threshold") for r in group),
        }
        for field in STAT_FIELDS:
            result[field] = _finite_mean(r.get(field) for r in group)
        summary.append(result)
    return summary


def _write_rows(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _iter_cases(images: str, labels: Path) -> list[tuple[Path, Path]]:
    image_path = Path(images)
    if image_path.is_dir():
        image_paths = sorted(image_path.glob("*.nii.gz"))
    else:
        image_paths = sorted(image_path.parent.glob(image_path.name))
    return [(path, labels / path.name) for path in image_paths
            if (labels / path.name).is_file()]


def select_label_value(label_map: np.ndarray, explicit_value: int | None,
                       case: str = "") -> int:
    """Resolve a binary GT label without silently choosing among classes."""
    values = sorted(int(value) for value in np.unique(label_map) if int(value) != 0)
    if explicit_value is not None:
        if explicit_value not in values:
            raise ValueError(
                f"{case}: requested --label-value {explicit_value}, "
                f"but non-zero GT labels are {values}"
            )
        return int(explicit_value)
    if len(values) == 1:
        return values[0]
    if len(values) > 1:
        raise ValueError(
            f"{case}: found multiple non-zero GT labels {values}; "
            "pass --label-value explicitly"
        )
    raise ValueError(
        f"{case}: found no non-zero GT label; pass --label-value explicitly "
        "if an all-zero label map is intentional"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, help="Image directory or *.nii.gz glob")
    parser.add_argument("--labels", required=True, help="GT label directory")
    parser.add_argument("--model", default="model")
    parser.add_argument("--text-model", default="/mnt/afs2/models/huggingface/hub/models--Qwen--Qwen3-Embedding-4B/snapshots/5cf2132abc99cad020ac570b19d031efec650f2b")
    parser.add_argument("--prompt", default="liver")
    parser.add_argument("--label-value", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Prediction and D5-FN threshold (default: 0.5)")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", default="decoder_probability_distribution")
    parser.add_argument(
        "--all-layers-on-device", action="store_true",
        help="Keep return_all_layers full-volume accumulation on the inference device "
             "for speed; default is safer CPU accumulation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    if args.device == "cuda":
        import torch
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    else:
        import torch
        device = torch.device("cpu")

    from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
    from voxtell.inference.predictor_multiclass import VoxTellPredictor

    cases = _iter_cases(args.images, Path(args.labels))
    if args.limit is not None:
        cases = cases[:args.limit]
    if not cases:
        raise RuntimeError("no image/label pairs found")

    reader = NibabelIOWithReorient()
    resolved_label_values: dict[str, int] = {}
    # Validate all label maps before loading the model. In particular, do not
    # silently choose the first class in a multi-label map.
    for image_path, label_path in cases:
        case = image_path.name.removesuffix(".nii.gz")
        label, _ = reader.read_images([str(label_path)])
        label_map = np.rint(label[0]).astype(np.int64)
        resolved_label_values[case] = select_label_value(
            label_map, args.label_value, case
        )
        del label_map, label

    predictor = VoxTellPredictor(
        str(args.model), device=device, text_encoding_model=args.text_model,
        return_all_layers_on_cpu=not args.all_layers_on_device,
    )
    per_case_rows: list[dict[str, Any]] = []
    for case_no, (image_path, label_path) in enumerate(cases, 1):
        case = image_path.name.removesuffix(".nii.gz")
        image, _ = reader.read_images([str(image_path)])
        label, _ = reader.read_images([str(label_path)])
        label_map = np.rint(label[0]).astype(np.int64)
        label_value = resolved_label_values[case]
        gt = label_map == label_value

        predictions = predictor.predict_single_image(
            image, [args.prompt], output_type="probabilities", return_all_layers=True
        )
        # Predictor outputs are high-to-low; the analyser performs the D1..D5
        # mapping and explicit interpolation independently of decoder_metrics.csv.
        raw_predictions = [np.asarray(prediction[0], dtype=np.float32)
                           for prediction in predictions]
        valid_inference_mask = predictor.get_last_valid_inference_mask()
        observed_shapes: list[Sequence[int] | None] = [None] * DECODER_COUNT
        # ``decoder_output_metadata`` is already in D1..D5 order and carries
        # the raw model output shapes captured before predictor-side alignment.
        # Keep that information while using the fresh GT shape as the analysis
        # target shape.
        for info in predictor.decoder_output_metadata:
            list_index = int(info["model_output_list_index"])
            observed_shapes[list_index] = info.get("observed_raw_patch_shape")
        metadata = decoder_metadata(
            [shape if shape is not None else raw_predictions[i].shape
             for i, shape in enumerate(observed_shapes)],
            gt.shape,
        )
        rows = analyze_case(raw_predictions, gt, case=case, prompt=args.prompt,
                            threshold=args.threshold, metadata=metadata,
                            valid_inference_mask=valid_inference_mask)
        per_case_rows.extend(rows)
        del raw_predictions, predictions, valid_inference_mask, image, label, gt
        print(f"[{case_no}/{len(cases)}] {case}: {len(rows)} distribution rows")

    _write_rows(output_dir / "decoder_probability_distribution_per_case.csv",
                PER_CASE_FIELDS, per_case_rows)
    _write_rows(output_dir / "decoder_probability_distribution_summary.csv",
                SUMMARY_FIELDS, summarize_rows(per_case_rows))
    print(f"Saved outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
