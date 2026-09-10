#!/usr/bin/env python3
"""Offline complementarity analysis for VoxTell's five image-decoder heads.

The model returns heads in highest-to-lowest resolution order.  This module
labels them by the decoder's actual low-to-high upsampling order, obtains full
volume probabilities through the existing sliding-window predictor, and never
changes training, TSE, model weights, or the default prediction API.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy import ndimage


CSV_FIELDS = [
    "record_type", "case", "name", "decoder_stage", "model_output_list_index",
    "internal_stage_index", "encoder_skip_index", "upsampling_order", "is_final_output",
    "head_type", "observed_raw_patch_shape",
    "aligned_patch_shape", "upsample_factor_to_final", "prompt", "threshold",
    "gt_voxels", "gt_volume_mm3", "pred_voxels", "pred_volume_mm3",
    "pred_gt_volume_ratio", "dice", "iou", "precision", "recall",
    "cc_count", "cc_volume_median_voxels", "cc_volume_median_mm3",
    "small_cc_count_fraction", "small_cc_voxel_fraction", "small_component_max_voxels", "lhs", "rhs",
    "pairwise_dice", "fp_overlap", "fp_overlap_dice", "fn_overlap",
    "fn_overlap_dice", "intersection_voxels", "union_voxels", "mean_probability",
    "d25_intersection_voxels", "d25_union_voxels", "majority_vote_voxels",
    "independent_tp_voxels", "independent_fp_voxels",
    "independent_fn_recovery", "r1_voxels", "r1_volume_mm3", "r1_tp_voxels", "r1_fp_voxels",
    "extra_precision", "fn_recovery", "merge_added_tp_voxels",
    "merge_added_fp_voxels", "merge_dice", "merge_recall", "merge_precision",
    "delta_dice", "delta_recall", "delta_precision",
]


def safe_div(num: float, den: float) -> float:
    """Return NaN for an empty denominator (never silently coerce it to 0/1)."""
    return float(num) / float(den) if den else float("nan")


def binary_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    pred_b, gt_b = np.asarray(pred, dtype=bool), np.asarray(gt, dtype=bool)
    if pred_b.shape != gt_b.shape:
        raise ValueError(f"shape mismatch: prediction {pred_b.shape}, GT {gt_b.shape}")
    inter = int(np.count_nonzero(pred_b & gt_b))
    pred_n, gt_n = int(pred_b.sum()), int(gt_b.sum())
    union = int(np.count_nonzero(pred_b | gt_b))
    return {
        "dice": safe_div(2 * inter, pred_n + gt_n),
        "iou": safe_div(inter, union),
        "precision": safe_div(inter, pred_n),
        "recall": safe_div(inter, gt_n),
        "intersection_voxels": inter,
        "union_voxels": union,
        "pred_voxels": pred_n,
        "gt_voxels": gt_n,
    }


def overlap_metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    a_b, b_b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    inter = int(np.count_nonzero(a_b & b_b))
    union = int(np.count_nonzero(a_b | b_b))
    return {
        "overlap": safe_div(inter, union),
        "overlap_dice": safe_div(2 * inter, int(a_b.sum()) + int(b_b.sum())),
        "intersection_voxels": inter,
        "union_voxels": union,
    }


def connected_component_metrics(mask: np.ndarray, voxel_volume_mm3: float = 1.0,
                                 small_component_max_voxels: int = 100) -> dict[str, float]:
    mask_b = np.asarray(mask, dtype=bool)
    structure = ndimage.generate_binary_structure(mask_b.ndim, mask_b.ndim)
    labels, count = ndimage.label(mask_b, structure=structure)
    sizes = np.bincount(labels.ravel())[1:].astype(np.int64)
    small = sizes <= int(small_component_max_voxels)
    return {
        "cc_count": int(count),
        "cc_volume_median_voxels": float(np.median(sizes)) if sizes.size else float("nan"),
        "cc_volume_median_mm3": safe_div(float(np.median(sizes)) * voxel_volume_mm3, 1)
        if sizes.size else float("nan"),
        "small_cc_count_fraction": safe_div(float(small.sum()), float(count)),
        "small_cc_voxel_fraction": safe_div(float(sizes[small].sum()), float(sizes.sum()))
        if sizes.size else float("nan"),
    }


def majority_vote(masks: Sequence[np.ndarray], votes: int = 3) -> np.ndarray:
    if not masks:
        raise ValueError("at least one mask is required")
    stack = np.stack([np.asarray(m, dtype=bool) for m in masks], axis=0)
    return stack.sum(axis=0) >= int(votes)


def align_gt_nearest(gt: np.ndarray, target_shape: Sequence[int]) -> np.ndarray:
    """Align GT explicitly with nearest interpolation when shapes differ."""
    gt = np.asarray(gt)
    target_shape = tuple(int(v) for v in target_shape)
    if gt.shape == target_shape:
        return gt.astype(bool)
    import torch
    import torch.nn.functional as F
    x = torch.from_numpy(gt.astype(np.float32))[None, None]
    return F.interpolate(x, size=target_shape, mode="nearest")[0, 0].numpy().astype(bool)


def _metric_row(name: str, stage: int, probs: np.ndarray, gt: np.ndarray,
                threshold: float, voxel_volume_mm3: float,
                metadata: Mapping[str, Any], small_component_max_voxels: int = 100) -> dict[str, Any]:
    mask = probs >= threshold
    result = binary_metrics(mask, gt)
    result.update(connected_component_metrics(mask, voxel_volume_mm3, small_component_max_voxels))
    result.update({
        "record_type": "layer", "name": name, "decoder_stage": stage,
        "model_output_list_index": metadata.get("model_output_list_index"),
        "internal_stage_index": metadata.get("internal_stage_index"),
        "head_type": metadata.get("head_type"),
        "pred_volume_mm3": float(result["pred_voxels"] * voxel_volume_mm3),
        "gt_volume_mm3": float(result["gt_voxels"] * voxel_volume_mm3),
        "pred_gt_volume_ratio": safe_div(result["pred_voxels"], result["gt_voxels"]),
        "threshold": threshold,
    })
    return result


def analyze_case(probabilities: Sequence[np.ndarray], gt: np.ndarray, threshold: float = 0.5,
                 voxel_volume_mm3: float = 1.0, metadata: Sequence[Mapping[str, Any]] | None = None,
                 prompt: str = "target", small_component_max_voxels: int = 100) -> dict[str, Any]:
    """Compute all case-level quantities; inputs are Decoder 1..5 probabilities."""
    if len(probabilities) != 5:
        raise ValueError(f"expected 5 decoder probabilities, got {len(probabilities)}")
    probs = [np.asarray(p, dtype=np.float32) for p in probabilities]
    final_shape = probs[0].shape
    if any(p.shape != final_shape for p in probs):
        raise ValueError("all probabilities must already be aligned to final volume space")
    gt_b = align_gt_nearest(gt, final_shape)
    md = list(metadata or ({"model_output_list_index": 4 - i, "internal_stage_index": i,
                            "head_type": "final_mask_einsum" if i == 4 else "intermediate_segmentation_conv"}
                           for i in range(5)))
    rows: list[dict[str, Any]] = []
    masks = [p >= threshold for p in probs]
    for i, (p, m) in enumerate(zip(probs, masks), start=1):
        row = _metric_row(f"Decoder {i}", i, p, gt_b, threshold, voxel_volume_mm3, md[i - 1], small_component_max_voxels)
        row["prompt"] = prompt
        # Keep the configured component threshold visible in machine-readable output.
        row["small_component_max_voxels"] = int(small_component_max_voxels)
        rows.append(row)

    # Decoder 5 is the actual final output (last upsampling stage, list index 0).
    final = masks[4]
    consensus = majority_vote(masks[1:5], votes=3)
    r1 = masks[0] & ~consensus
    final_fn = gt_b & ~final
    r1_tp = r1 & gt_b
    r1_fp = r1 & ~gt_b
    merged = final | r1
    final_m = binary_metrics(final, gt_b)
    merged_m = binary_metrics(merged, gt_b)
    rows.append({
        "record_type": "decoder1_extra", "name": "Decoder 1 extra R1", "decoder_stage": 1,
        "model_output_list_index": md[0].get("model_output_list_index"),
        "internal_stage_index": md[0].get("internal_stage_index"), "prompt": prompt,
        "threshold": threshold, "gt_voxels": int(gt_b.sum()), "r1_voxels": int(r1.sum()),
        "r1_volume_mm3": float(r1.sum() * voxel_volume_mm3),
        "r1_tp_voxels": int(r1_tp.sum()), "r1_fp_voxels": int(r1_fp.sum()),
        "extra_precision": safe_div(r1_tp.sum(), r1.sum()),
        "fn_recovery": safe_div((r1 & final_fn).sum(), final_fn.sum()),
        "merge_added_tp_voxels": int((merged & gt_b & ~final).sum()),
        "merge_added_fp_voxels": int((merged & ~gt_b & ~final).sum()),
        "merge_dice": merged_m["dice"], "merge_recall": merged_m["recall"],
        "merge_precision": merged_m["precision"],
        "delta_dice": merged_m["dice"] - final_m["dice"] if np.isfinite(merged_m["dice"]) and np.isfinite(final_m["dice"]) else float("nan"),
        "delta_recall": merged_m["recall"] - final_m["recall"] if np.isfinite(merged_m["recall"]) and np.isfinite(final_m["recall"]) else float("nan"),
        "delta_precision": merged_m["precision"] - final_m["precision"] if np.isfinite(merged_m["precision"]) and np.isfinite(final_m["precision"]) else float("nan"),
    })

    for i in range(1, 5):
        for j in range(i + 1, 5):
            fp_i, fp_j = masks[i] & ~gt_b, masks[j] & ~gt_b
            fn_i, fn_j = gt_b & ~masks[i], gt_b & ~masks[j]
            fp_o, fn_o = overlap_metrics(fp_i, fp_j), overlap_metrics(fn_i, fn_j)
            rows.append({
                "record_type": "pairwise", "lhs": f"Decoder {i + 1}", "rhs": f"Decoder {j + 1}",
                "threshold": threshold, "pairwise_dice": binary_metrics(masks[i], masks[j])["dice"],
                "fp_overlap": fp_o["overlap"], "fp_overlap_dice": fp_o["overlap_dice"],
                "fn_overlap": fn_o["overlap"], "fn_overlap_dice": fn_o["overlap_dice"],
                "intersection_voxels": fp_o["intersection_voxels"],
                "union_voxels": fp_o["union_voxels"],
            })

    for i in range(1, 5):
        independent_tp = masks[i] & gt_b & ~final
        independent_fp = masks[i] & ~gt_b & ~final
        rows.append({
            "record_type": "independent_vs_final", "name": f"Decoder {i + 1}",
            "decoder_stage": i + 1, "threshold": threshold,
            "independent_tp_voxels": int(independent_tp.sum()),
            "independent_fp_voxels": int(independent_fp.sum()),
            "independent_fn_recovery": safe_div(independent_tp.sum(), final_fn.sum()),
        })

    stack = np.stack(probs[1:5], axis=0)
    ensembles = {
        "d2_d5_intersection": np.all(np.stack(masks[1:5], axis=0), axis=0),
        "d2_d5_union": np.any(np.stack(masks[1:5], axis=0), axis=0),
        "d2_d5_majority_vote": consensus,
        "d2_d5_average_probability": stack.mean(axis=0) >= threshold,
    }
    d25_intersection_voxels = int(np.all(np.stack(masks[1:5], axis=0), axis=0).sum())
    d25_union_voxels = int(np.any(np.stack(masks[1:5], axis=0), axis=0).sum())
    for name, mask in ensembles.items():
        mm = binary_metrics(mask, gt_b)
        rows.append({
            "record_type": "ensemble", "name": name, "threshold": threshold,
            **mm, "mean_probability": float(stack.mean()),
            "d25_intersection_voxels": d25_intersection_voxels,
            "d25_union_voxels": d25_union_voxels,
            "majority_vote_voxels": int(consensus.sum()),
        })
    return {"rows": rows, "masks": masks, "gt": gt_b, "consensus": consensus, "r1": r1,
            "final": final, "merged": merged, "probabilities": probs}


def _finite_stats(values: Iterable[Any]) -> dict[str, Any]:
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    return {"mean": float(np.mean(vals)) if vals else float("nan"),
            "median": float(np.median(vals)) if vals else float("nan"),
            "std": float(np.std(vals)) if vals else float("nan"), "valid_cases": len(vals)}


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping): return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.ndarray): return _json_safe(value.tolist())
    return value


def summarize_rows(rows: Sequence[Mapping[str, Any]], total_cases: int,
                   metadata: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    layer_rows = [r for r in rows if r.get("record_type") == "layer"]
    extra_rows = [r for r in rows if r.get("record_type") == "decoder1_extra"]
    pair_rows = [r for r in rows if r.get("record_type") == "pairwise"]
    indep_rows = [r for r in rows if r.get("record_type") == "independent_vs_final"]
    ens_rows = [r for r in rows if r.get("record_type") == "ensemble"]
    metrics = ("dice", "iou", "precision", "recall", "pred_voxels", "pred_gt_volume_ratio",
               "cc_count", "cc_volume_median_voxels", "small_cc_count_fraction",
               "small_cc_voxel_fraction")
    layer_summary = {}
    for stage in range(1, 6):
        rr = [r for r in layer_rows if r.get("decoder_stage") == stage]
        layer_summary[f"decoder_{stage}"] = {
            metric: {**_finite_stats(r.get(metric) for r in rr), "total_cases": total_cases}
            for metric in metrics
        }
    extra_summary = {m: {**_finite_stats(r.get(m) for r in extra_rows), "total_cases": total_cases}
                     for m in ("extra_precision", "fn_recovery", "r1_voxels", "r1_tp_voxels",
                               "r1_volume_mm3", "r1_fp_voxels", "merge_added_tp_voxels", "merge_added_fp_voxels",
                               "merge_dice", "merge_recall", "merge_precision", "delta_dice",
                               "delta_recall", "delta_precision")}
    pair_summary = {}
    for lhs in range(2, 6):
        for rhs in range(lhs + 1, 6):
            rr = [r for r in pair_rows if r.get("lhs") == f"Decoder {lhs}" and r.get("rhs") == f"Decoder {rhs}"]
            pair_summary[f"decoder_{lhs}_vs_decoder_{rhs}"] = {
                m: {**_finite_stats(r.get(m) for r in rr), "total_cases": total_cases}
                for m in ("pairwise_dice", "fp_overlap", "fp_overlap_dice", "fn_overlap", "fn_overlap_dice")}
    independent_summary = {}
    for stage in range(2, 6):
        rr = [r for r in indep_rows if r.get("decoder_stage") == stage]
        independent_summary[f"decoder_{stage}"] = {
            m: {**_finite_stats(r.get(m) for r in rr), "total_cases": total_cases}
            for m in ("independent_tp_voxels", "independent_fp_voxels", "independent_fn_recovery")}
    ensemble_summary = {}
    for name in sorted({r.get("name") for r in ens_rows}):
        rr = [r for r in ens_rows if r.get("name") == name]
        ensemble_summary[name] = {m: {**_finite_stats(r.get(m) for r in rr), "total_cases": total_cases}
                                  for m in ("dice", "iou", "precision", "recall", "pred_voxels",
                                            "mean_probability", "d25_intersection_voxels",
                                            "d25_union_voxels", "majority_vote_voxels")}
    extra_p = extra_summary["extra_precision"]["mean"]
    fn_r = extra_summary["fn_recovery"]["mean"]
    positive = [r.get("delta_dice") for r in extra_rows if np.isfinite(r.get("delta_dice", np.nan))]
    positive_fraction = safe_div(sum(float(v) > 0 for v in positive), len(positive))
    fp_overlaps = [r.get("fp_overlap") for r in pair_rows]
    fn_overlaps = [r.get("fn_overlap") for r in pair_rows]
    d1_supported = bool(np.isfinite(extra_p) and np.isfinite(fn_r) and np.isfinite(positive_fraction)
                        and extra_p >= config.get("min_extra_precision", .5)
                        and fn_r >= config.get("min_fn_recovery", .05)
                        and positive_fraction >= config.get("min_positive_case_fraction", .75))
    overlap_mean = np.nanmean(fp_overlaps + fn_overlaps) if any(np.isfinite(x) for x in fp_overlaps + fn_overlaps) else np.nan
    complementary = bool(np.isfinite(overlap_mean) and overlap_mean <= config.get("max_error_overlap", .5))
    d25_good = all(
        np.isfinite(layer_summary[f"decoder_{stage}"]["dice"]["mean"])
        and layer_summary[f"decoder_{stage}"]["dice"]["mean"] >= .5
        for stage in range(2, 6)
    )
    if d25_good and np.isfinite(overlap_mean) and overlap_mean > config.get("max_error_overlap", .5):
        error_redundancy = "性能相近但错误冗余"
    elif complementary:
        error_redundancy = "错误重合较低，存在互补证据"
    else:
        error_redundancy = "证据不足，不能声称多层输出提供独立证据"
    conclusion = {
        "decoder1_extra_region_supported": d1_supported,
        "decoder2_to_5_truly_complementary": complementary,
        "positive_merge_dice_case_fraction": positive_fraction,
        "mean_pairwise_error_overlap": float(overlap_mean) if np.isfinite(overlap_mean) else float("nan"),
        "decoder2_to_5_error_redundancy_assessment": error_redundancy,
        "recommendation": "仅当两个问题都有证据支持时才使用多 decoder 证据；否则保留 Decoder 5 单层最终输出",
        "evidence_rule": "Decoder 1 需同时满足 ExtraPrecision/FNRecovery 阈值和稳定的 Dice 正变化；Decoder 2-5 需满足平均 FP/FN 重合低于阈值。",
    }
    return _json_safe({"schema_version": 1, "case_count": total_cases, "config": dict(config),
        "decoder_mapping": list(metadata), "layers": layer_summary, "decoder1_extra": extra_summary,
        "pairwise_decoder2_to_5": pair_summary, "independent_vs_final": independent_summary,
        "ensembles": ensemble_summary, "conclusion": conclusion})


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def _save_visual(path: Path, payload: Mapping[str, Any], title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    image = payload["image"]
    gt, d1, con, r1, final = (payload[k].astype(bool) for k in ("gt", "d1", "consensus", "r1", "final"))
    fp, fn = final & ~gt, gt & ~final
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    panels = [(image, "image", "gray"), (gt, "GT", "viridis"), (d1, "Decoder 1", "Blues"),
              (con, "D2-5 consensus", "Blues"), (r1, "R1 (D1\\consensus)", "magma"),
              (final, "final Decoder 5", "Blues"), (fp, "final FP", "Reds"), (fn, "final FN", "Purples")]
    for ax, (arr, name, cmap) in zip(axes.ravel(), panels):
        if name == "image":
            lo, hi = np.percentile(arr[np.isfinite(arr)], [1, 99]) if np.any(np.isfinite(arr)) else (0, 1)
            ax.imshow(arr, cmap=cmap, vmin=lo, vmax=hi)
        else: ax.imshow(arr, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(name); ax.axis("off")
    fig.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140); plt.close(fig)


def _visual_payload(image: np.ndarray, result: Mapping[str, Any]) -> dict[str, Any]:
    gt, masks, con, r1, final = result["gt"], result["masks"], result["consensus"], result["r1"], result["final"]
    score = gt | masks[0] | con | r1 | final
    # Display the most informative axial slice, retaining full 3-D analysis.
    axes = tuple(range(score.ndim - 1))
    slice_index = int(np.argmax(score.sum(axis=axes))) if score.ndim > 2 else 0
    slicer = (slice(None),) * (score.ndim - 1) + (slice_index,)
    img = np.asarray(image[0] if image.ndim == score.ndim + 1 else image)[slicer]
    return {"image": img, "gt": gt[slicer], "d1": masks[0][slicer], "consensus": con[slicer],
            "r1": r1[slicer], "final": final[slicer], "slice_index": slice_index}


def _smoke() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    shape = (6, 6, 4); gt = np.zeros(shape, bool); gt[1:4, 1:4, 1:3] = 1
    p = [np.zeros(shape, np.float32) for _ in range(5)]
    p[4][gt] = .9; p[0][0, 0, 0] = .9; p[0][3, 3, 2] = .9; p[1][gt] = .8; p[2][gt] = .8; p[3][gt] = .8
    result = analyze_case(p, gt, threshold=.5)
    assert result["rows"]
    return result["rows"], [{"name": f"Decoder {i}", "decoder_stage": i,
        "model_output_list_index": 5 - i, "internal_stage_index": i - 1,
        "head_type": "final_mask_einsum" if i == 5 else "intermediate_segmentation_conv"}
        for i in range(1, 6)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images", help="Image directory or *.nii.gz glob")
    p.add_argument("--labels", help="GT label directory")
    p.add_argument("--model", default="model")
    p.add_argument("--text-model", default="Qwen/Qwen3-Embedding-4B")
    p.add_argument("--prompt", default="liver")
    p.add_argument("--label-value", type=int, default=None)
    p.add_argument("--threshold", type=float, default=.5)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--output-dir", default="decoder_complementarity")
    p.add_argument("--small-component-max-voxels", type=int, default=100)
    p.add_argument("--save-probabilities", action="store_true")
    p.add_argument("--min-extra-precision", type=float, default=.5)
    p.add_argument("--min-fn-recovery", type=float, default=.05)
    p.add_argument("--min-positive-case-fraction", type=float, default=.75)
    p.add_argument("--max-error-overlap", type=float, default=.5)
    p.add_argument("--smoke-test", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args(); out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy(); config.pop("smoke_test", None)
    config.update({"decoder_output_interpolation": "trilinear",
                   "gt_interpolation": "nearest",
                   "decoder_order": "actual_low_to_high_upsampling_order"})
    log_path = out / "decoder_complementarity.log"
    log_path.write_text(f"Args: {vars(args)}\n", encoding="utf-8")
    if args.smoke_test:
        rows, metadata = _smoke(); _write_csv(out / "decoder_complementarity_per_case.csv", rows)
        (out / "decoder_complementarity_summary.json").write_text(json.dumps(
            summarize_rows(rows, 1, metadata, config), indent=2, allow_nan=False), encoding="utf-8")
        print(f"smoke test outputs: {out}"); return 0
    if not args.images or not args.labels:
        raise SystemExit("--images and --labels are required unless --smoke-test is used")
    os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
    from voxtell.inference.predictor_multiclass import VoxTellPredictor
    images_path, labels_dir = Path(args.images), Path(args.labels)
    image_paths = sorted(images_path.glob("*.nii.gz")) if images_path.is_dir() else sorted(images_path.parent.glob(images_path.name))
    cases = [(p, labels_dir / p.name) for p in image_paths if (labels_dir / p.name).is_file()]
    if args.limit is not None: cases = cases[:args.limit]
    if not cases: raise RuntimeError("no image/label pairs found")
    import torch
    device = torch.device(f"cuda:{args.gpu}" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    predictor = VoxTellPredictor(str(args.model), device=device, text_encoding_model=args.text_model)
    reader = NibabelIOWithReorient(); all_rows=[]; mapping=None; candidates={}
    print("Decoder mapping (actual low->high upsampling order; returned list is high->low):")
    with log_path.open("a", encoding="utf-8") as log:
        for info in predictor.decoder_output_metadata:
            line = json.dumps(info, sort_keys=True)
            print("  " + line); log.write(line + "\n")
    for case_no, (image_path, label_path) in enumerate(cases, 1):
        case = image_path.name.removesuffix(".nii.gz"); image, props = reader.read_images([str(image_path)]); label, _ = reader.read_images([str(label_path)])
        label_map = np.rint(label[0]).astype(np.int64); value = args.label_value if args.label_value is not None else next((int(v) for v in np.unique(label_map) if v), 1); gt = label_map == value
        predictions = predictor.predict_single_image(image, [args.prompt], output_type="probabilities", return_all_layers=True)
        # Returned list is highest->lowest. Metadata maps it to actual Decoder 1..5.
        mapping = predictor.decoder_output_metadata
        probs = [np.asarray(predictions[item["model_output_list_index"]][0], dtype=np.float32) for item in mapping]
        spacing = props.get("spacing", props.get("original_spacing", (1, 1, 1))); voxel_volume = float(np.prod(spacing))
        result = analyze_case(probs, gt, args.threshold, voxel_volume, mapping, args.prompt, args.small_component_max_voxels)
        for row in result["rows"]:
            if row.get("decoder_stage"):
                info = mapping[int(row["decoder_stage"]) - 1]
            else:
                info = {}
            row.update({"case": case, "encoder_skip_index": info.get("encoder_skip_index", ""),
                        "upsampling_order": info.get("upsampling_order", ""),
                        "is_final_output": info.get("is_final_output", ""),
                        "observed_raw_patch_shape": json.dumps(info.get("observed_raw_patch_shape", [])),
                        "aligned_patch_shape": json.dumps(info.get("aligned_patch_shape", [])),
                        "upsample_factor_to_final": json.dumps(info.get("upsample_factor_to_final", []))})
        all_rows.extend(result["rows"])
        if args.save_probabilities:
            case_out = out / "probabilities" / case; case_out.mkdir(parents=True, exist_ok=True)
            for i, prob in enumerate(probs, 1): np.save(case_out / f"decoder_{i}.npy", prob)
        pair = [r for r in result["rows"] if r.get("record_type") == "pairwise"]
        extra = next(r for r in result["rows"] if r.get("record_type") == "decoder1_extra")
        consistency = np.nanmean([r["pairwise_dice"] for r in pair]);
        scores = {"d1_effective_recovery": (extra.get("fn_recovery", np.nan) * extra.get("extra_precision", np.nan)), "d1_false_positive": (extra.get("merge_added_fp_voxels", 0) * (1 - (extra.get("extra_precision", 0) if np.isfinite(extra.get("extra_precision", np.nan)) else 0))), "d25_consistent": consistency, "d25_inconsistent": 1 - consistency}
        for category, score in scores.items():
            if np.isfinite(score) and (category not in candidates or score > candidates[category]["score"]): candidates[category] = {"score": float(score), "case": case, "payload": _visual_payload(image, result)}
        print(f"[{case_no}/{len(cases)}] {case}: D1 ExtraPrecision={extra['extra_precision']!r}, FNRecovery={extra['fn_recovery']!r}")
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"[{case_no}/{len(cases)}] {case}: {extra!r}\n")
    if mapping is None: raise RuntimeError("no predictions")
    _write_csv(out / "decoder_complementarity_per_case.csv", all_rows)
    summary = summarize_rows(all_rows, len(cases), mapping, config); summary["representative_cases"] = {k: {"case": v["case"], "score": v["score"]} for k,v in candidates.items()}
    (out / "decoder_complementarity_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    for category, candidate in candidates.items(): _save_visual(out / "visualizations" / f"{category}__{candidate['case']}.png", candidate["payload"], f"{category}: {candidate['case']}")
    print(f"Saved {out / 'decoder_complementarity_per_case.csv'} and {out / 'decoder_complementarity_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
