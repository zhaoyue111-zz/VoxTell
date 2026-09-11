#!/usr/bin/env python3
"""Offline text-causal pseudo-label quality audit for the fixed ``liver`` prompt.

The experiment changes only the transformer decoder's memory-key visibility.  It
does not update parameters, alter images/text/encoder skips, or participate in
SFDA training.  It first fuses normal D5 logits over all sliding windows,
then derives one global pseudo-label and crops it for the masked second pass.
D5 is the model output at list index 0 (highest resolution).
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
    "q_similarity_in_median", "q_similarity_out_median", "q_causal_score",
    "pred_softdice_in", "pred_softdice_out", "pred_causal_score", "entropy",
    "cac", "d25_consistency", "valid_patch_count", "total_patch_count",
    "causal_invalid_reason",
    "memory_shape_dhw", "memory_tokens", "attention_blocked_mass_in",
    "attention_blocked_mass_out", "attention_visible_mass_in",
    "attention_visible_mass_out", "attention_fallback_in",
    "attention_fallback_out", "attention_coverage_normal",
    "attention_coverage_normal_weighted", "attention_coverage_normal_median",
    "attention_coverage_valid_patches", "causal_support_voxels",
    "causal_support_ratio",
]
CORRELATION_TARGETS = ("dice", "precision", "recall", "high_confidence_fp_ratio")
UNSUPERVISED_SCORES = (
    "q_similarity_in", "q_similarity_out", "q_causal_score",
    "q_similarity_in_median", "q_similarity_out_median",
    "pred_softdice_in", "pred_softdice_out", "pred_causal_score",
    "entropy", "cac", "d25_consistency",
    "attention_coverage_normal", "attention_coverage_normal_weighted",
    "attention_coverage_normal_median",
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


def fused_prediction_softdices(normal_logits: torch.Tensor,
                               in_logits: torch.Tensor,
                               out_logits: torch.Tensor,
                               valid_patch_count: int,
                               valid_support: torch.Tensor | None = None) -> tuple[float, float]:
    """Compute causal prediction scores from final fused whole-volume logits."""
    if int(valid_patch_count) == 0:
        return float("nan"), float("nan")
    prob_normal = torch.sigmoid(normal_logits)
    prob_in = torch.sigmoid(in_logits)
    prob_out = torch.sigmoid(out_logits)
    if valid_support is not None:
        valid_support = valid_support.to(device=prob_normal.device, dtype=torch.bool)
        if not bool(valid_support.any()):
            return float("nan"), float("nan")
        prob_normal = prob_normal[valid_support]
        prob_in = prob_in[valid_support]
        prob_out = prob_out[valid_support]
    return (
        float(soft_dice(prob_normal, prob_in)),
        float(soft_dice(prob_normal, prob_out)),
    )


def align_gt_nearest(gt: np.ndarray, target_shape: Sequence[int]) -> np.ndarray:
    """Validate GT shape; axis/crop errors must not be hidden by resizing."""
    gt = np.asarray(gt)
    shape = tuple(int(v) for v in target_shape)
    if gt.shape == shape:
        return gt.astype(bool, copy=False)
    raise ValueError(
        "prediction/GT spatial shape mismatch; refusing implicit nearest resize: "
        f"prediction={shape}, GT={tuple(gt.shape)}"
    )


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


def crop_global_mask_to_patch(global_mask: torch.Tensor,
                              slicer: Sequence[slice]) -> torch.Tensor:
    """Crop a padded ``(B,D,H,W)`` global mask using a predictor slicer."""
    if global_mask.ndim != 4:
        raise ValueError(f"expected global mask (B,D,H,W), got {tuple(global_mask.shape)}")
    if len(slicer) != 4:
        raise ValueError(f"expected (channel,D,H,W) slicer, got {len(slicer)} entries")
    patch = global_mask[(slice(None), *tuple(slicer[1:]))]
    return patch


def _safe_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    denominator = a.norm() * b.norm()
    if float(denominator) == 0.0:
        return 0.0
    value = torch.dot(a, b) / denominator
    return float(value) if torch.isfinite(value) else 0.0


def _attention_stats(attention: Any, blocked: torch.Tensor) -> tuple[float, float]:
    """Return total attention mass on blocked and visible memory-token sets.

    Attention is summed over the memory-token dimension first, then averaged
    over batch/query/head dimensions. This makes the statistic independent of
    how many tokens belong to each set.
    """
    if attention is None:
        return float("nan"), float("nan")

    weights = attention.detach().float()
    if weights.ndim not in (3, 4):
        return float("nan"), float("nan")

    blocked = blocked.to(device=weights.device, dtype=torch.bool)
    if blocked.ndim != 2:
        return float("nan"), float("nan")

    if weights.shape[0] != blocked.shape[0] or weights.shape[-1] != blocked.shape[1]:
        return float("nan"), float("nan")

    visible = ~blocked

    if weights.ndim == 3:
        blocked_mask = blocked[:, None, :]
        visible_mask = visible[:, None, :]
    else:
        blocked_mask = blocked[:, None, None, :]
        visible_mask = visible[:, None, None, :]

    blocked_mass = (
        (weights * blocked_mask.to(weights.dtype)).sum(dim=-1).mean()
        if bool(blocked.any())
        else torch.zeros((), dtype=weights.dtype, device=weights.device)
    )
    visible_mass = (
        (weights * visible_mask.to(weights.dtype)).sum(dim=-1).mean()
        if bool(visible.any())
        else torch.zeros((), dtype=weights.dtype, device=weights.device)
    )

    blocked_value = (
        float(blocked_mass.cpu()) if torch.isfinite(blocked_mass) else float("nan")
    )
    visible_value = (
        float(visible_mass.cpu()) if torch.isfinite(visible_mass) else float("nan")
    )
    return blocked_value, visible_value

def attention_coverage(attention: Any, inside: torch.Tensor) -> float:
    """Fraction of normal attention mass landing inside the global pseudo-label."""
    if attention is None:
        return float("nan")
    weights = attention.detach().float()
    if weights.ndim == 4:
        weights = weights.mean(dim=1)
    if weights.ndim != 3:
        return float("nan")
    inside = inside.to(device=weights.device, dtype=weights.dtype)
    total = weights.sum()
    if float(total) <= 0 or not torch.isfinite(total):
        return float("nan")
    value = (weights * inside[:, None, :]).sum() / total
    return float(value.cpu()) if torch.isfinite(value) else float("nan")


def aggregate_attention_coverages(coverages: Sequence[float],
                                  weights: Sequence[float]) -> tuple[float, float, float]:
    """Aggregate normal-attention coverage over valid (non-fallback) patches."""
    values = np.asarray(coverages, dtype=np.float64)
    patch_weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(patch_weights) & (patch_weights > 0)
    if not bool(valid.any()):
        return float("nan"), float("nan"), float("nan")
    values, patch_weights = values[valid], patch_weights[valid]
    return (
        float(values.mean()),
        float(np.average(values, weights=patch_weights)),
        float(np.median(values)),
    )


def accumulate_causal_patch_logits(causal_normal_sum: torch.Tensor,
                                   in_sum: torch.Tensor,
                                   out_sum: torch.Tensor,
                                   causal_denominator: torch.Tensor,
                                   normal_patch: torch.Tensor,
                                   in_patch: torch.Tensor,
                                   out_patch: torch.Tensor,
                                   spatial_slicer: Sequence[slice],
                                   weight: torch.Tensor | float,
                                   valid_patch: bool) -> None:
    """Accumulate all three causal streams over exactly the same valid window."""
    if not valid_patch:
        return
    region = tuple(spatial_slicer)
    causal_normal_sum[region] += normal_patch * weight
    in_sum[region] += in_patch * weight
    out_sum[region] += out_patch * weight
    causal_denominator[region] += weight


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
    causal_normal_sum = torch.zeros(padded_shape, dtype=torch.float32)
    in_sum = torch.zeros(padded_shape, dtype=torch.float32)
    out_sum = torch.zeros(padded_shape, dtype=torch.float32)
    denominator = torch.zeros(padded_shape, dtype=torch.float32)
    causal_denominator = torch.zeros(padded_shape, dtype=torch.float32)
    normal_diagnostics = []
    attention_values = {"in": [], "out": []}
    attention_coverages = []
    attention_coverage_weights = []
    fallback = {"in": 0, "out": 0}
    valid_patch_count = 0
    q_in_values, q_out_values, q_weights = [], [], []

    # Round 1: fuse every normal D5 logit before constructing any mask.
    for slicer in slicers:
        tile = padded[slicer][None].to(device)
        amp = torch.autocast(device_type=device.type, enabled=device.type == "cuda")
        with torch.inference_mode(), amp:
            normal_outputs, normal_diag = network(
                tile, text_embedding, return_decoder_outputs=True,
                return_diagnostics=True,
            )
        d5_normal = _as_d5_logits(normal_outputs)
        _require_finite("normal D5 logits", d5_normal)
        memory_shape = tuple(int(v) for v in normal_diag["memory_shape"])
        tile_shape = tuple(int(s.stop - s.start) for s in slicer[1:])
        normal_d5_patch = _resize_logits(d5_normal.float(), tile_shape)[0, 0].cpu()
        normal_diagnostics.append({
            "slicer": slicer,
            "q": normal_diag["mask_embedding"].detach().cpu(),
            "cross_attention": normal_diag["cross_attention"][-1].detach().cpu(),
            "memory_shape": memory_shape,
            "normal_d5_patch": normal_d5_patch,
        })
        weight = gaussian
        for output_index, output in enumerate(normal_outputs):
            output = _resize_logits(output.float(), tile_shape)[0, 0].cpu()
            normal_sum[output_index][slicer[1:]] += output * weight
        denominator[slicer[1:]] += weight

    padded_denom = denominator.clamp_min(torch.finfo(torch.float32).eps)
    padded_normal_logits = [value / padded_denom for value in normal_sum]
    global_probability = torch.sigmoid(padded_normal_logits[0])
    global_mask = (global_probability >= float(threshold)).unsqueeze(0)

    # Round 2: crop the global D5 pseudo-label at each exact sliding-window
    # coordinate, then run only-in/only-out masked cross-attention.
    for diagnostic in normal_diagnostics:
        slicer = diagnostic["slicer"]
        tile = padded[slicer][None].to(device)
        amp = torch.autocast(device_type=device.type, enabled=device.type == "cuda")
        patch_mask = crop_global_mask_to_patch(global_mask, slicer)
        masks = build_memory_masks(patch_mask, diagnostic["memory_shape"])
        with torch.inference_mode(), amp:
            in_outputs, in_diag = network(
                tile, text_embedding, return_decoder_outputs=True,
                memory_key_padding_mask=masks["only_in"], return_diagnostics=True,
            )
            out_outputs, out_diag = network(
                tile, text_embedding, return_decoder_outputs=True,
                memory_key_padding_mask=masks["only_out"], return_diagnostics=True,
            )
        _require_finite("normal q", diagnostic["q"])
        _require_finite("only-in q", in_diag["mask_embedding"])
        _require_finite("only-out q", out_diag["mask_embedding"])
        d5_in = _as_d5_logits(in_outputs)
        d5_out = _as_d5_logits(out_outputs)
        _require_finite("only-in D5 logits", d5_in)
        _require_finite("only-out D5 logits", d5_out)
        q = diagnostic["q"][0, 0]
        fallback_patch = masks["fallback_in"] > 0 or masks["fallback_out"] > 0
        if not fallback_patch:
            valid_patch_count += 1
            token_weight = max(1, int(masks["inside_tokens"].sum().item()))
            q_in_current = in_diag["mask_embedding"][0, 0].detach().cpu()
            q_out_current = out_diag["mask_embedding"][0, 0].detach().cpu()
            q_in_values.append(_safe_cosine(q, q_in_current))
            q_out_values.append(_safe_cosine(q, q_out_current))
            q_weights.append(token_weight)
            attention_coverages.append(
                attention_coverage(diagnostic["cross_attention"], masks["inside_tokens"])
            )
            attention_coverage_weights.append(
                max(1, int(masks["inside_tokens"].sum().item()))
            )
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
        in_patch = _resize_logits(d5_in.float(), tile_shape)[0, 0].cpu()
        out_patch = _resize_logits(d5_out.float(), tile_shape)[0, 0].cpu()
        accumulate_causal_patch_logits(
            causal_normal_sum, in_sum, out_sum, causal_denominator,
            diagnostic["normal_d5_patch"], in_patch, out_patch,
            slicer[1:], weight, not fallback_patch,
        )

    crop = revert_padding[1:]
    normal_logits = [value[crop] for value in padded_normal_logits]
    causal_denom = causal_denominator[crop]
    valid_support = causal_denom > 0
    causal_normal_logits = torch.zeros_like(causal_denom)
    in_logits = torch.zeros_like(causal_denom)
    out_logits = torch.zeros_like(causal_denom)
    causal_normal_logits[valid_support] = (
        causal_normal_sum[crop][valid_support] / causal_denom[valid_support]
    )
    in_logits[valid_support] = in_sum[crop][valid_support] / causal_denom[valid_support]
    out_logits[valid_support] = out_sum[crop][valid_support] / causal_denom[valid_support]
    normal_probabilities = [torch.sigmoid(value) for value in normal_logits]
    probability = normal_probabilities[0].numpy()
    target = align_gt_nearest(target, probability.shape)
    metrics = binary_case_metrics(probability, target, threshold, high_confidence_threshold)

    causal_support_voxels = int(valid_support.sum().item())
    prediction_volume_voxels = int(valid_support.numel())
    causal_support_ratio = safe_ratio(
        causal_support_voxels,
        prediction_volume_voxels,
    )

    pred_softdice_in, pred_softdice_out = fused_prediction_softdices(
        causal_normal_logits, in_logits, out_logits, valid_patch_count, valid_support
    )
    if q_weights:
        q_similarity_in = float(np.average(q_in_values, weights=q_weights))
        q_similarity_out = float(np.average(q_out_values, weights=q_weights))
        q_similarity_in_median = float(np.median(q_in_values))
        q_similarity_out_median = float(np.median(q_out_values))
    else:
        q_similarity_in = q_similarity_out = float("nan")
        q_similarity_in_median = q_similarity_out_median = float("nan")
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
    coverage_mean, coverage_weighted, coverage_median = aggregate_attention_coverages(
        attention_coverages, attention_coverage_weights
    )
    invalid_reason = None if valid_patch_count else "no_valid_patch_after_empty_or_full_global_mask"
    row = {
        "dice": metrics["dice"], "precision": metrics["precision"],
        "recall": metrics["recall"], "fp_ratio": metrics["fp_ratio"],
        "fn_ratio": metrics["fn_ratio"],
        "high_confidence_fp_ratio": metrics["high_confidence_fp_ratio"],
        "q_similarity_in": q_similarity_in,
        "q_similarity_out": q_similarity_out,
        "q_similarity_in_median": q_similarity_in_median,
        "q_similarity_out_median": q_similarity_out_median,
        "q_causal_score": q_similarity_in - q_similarity_out if valid_patch_count else float("nan"),
        "pred_softdice_in": pred_softdice_in,
        "pred_softdice_out": pred_softdice_out,
        "pred_causal_score": pred_softdice_in - pred_softdice_out if valid_patch_count else float("nan"),
        "valid_patch_count": valid_patch_count,
        "total_patch_count": len(slicers),
        "causal_invalid_reason": invalid_reason,
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
        "attention_coverage_normal": coverage_mean,
        "attention_coverage_normal_weighted": coverage_weighted,
        "attention_coverage_normal_median": coverage_median,
        "attention_coverage_valid_patches": len(attention_coverages),
        "causal_support_voxels": causal_support_voxels,
        "causal_support_ratio": causal_support_ratio,
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
        "protocol": {
            "sliding_window_passes": 2,
            "global_mask": "sigmoid(Gaussian-fused normal D5 logits) >= threshold",
            "masked_pass_mask_source": "global mask cropped with the exact predictor slicer",
            "fallback_patches_excluded": True,
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
