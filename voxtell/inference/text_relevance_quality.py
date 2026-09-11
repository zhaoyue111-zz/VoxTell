#!/usr/bin/env python3
"""Offline VoxTell language-conditioned relevance audit.

This script keeps VoxTell inference unchanged. It does NOT mask foreground or
background tokens. Instead it measures how strongly the normal ``liver``
segmentation decision depends on transformer image-memory tokens.

Main ideas
----------
1. Run normal sliding-window inference and build ONE Gaussian-fused global D5
   pseudo-label.
2. Re-run the same normal model without any attention mask, with gradients
   enabled only for attribution.
3. Capture the projected transformer image memory with a forward hook.
4. For each valid patch, define the decision score as the mean D5 liver logit
   inside the cropped global pseudo-label.
5. Compute token relevance by |activation * gradient| summed over channels.
6. Compare relevance inside the pseudo-label with many equal-size random
   background token sets. This yields:
      - relevance_enrichment = inside_mean / matched_background_mean
      - lrs_z = (inside_mean - bg_mean) / bg_std
7. Also report raw normal-attention enrichment:
      attention_coverage / inside_token_ratio

Ground-truth labels are used ONLY for evaluation correlations.
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
    "case_id",
    "dice",
    "precision",
    "recall",
    "fp_ratio",
    "fn_ratio",
    "high_confidence_fp_ratio",
    "entropy",
    "d25_consistency",
    "global_mask_equals_metric_mask",
    "inside_voxel_ratio",
    "inside_token_ratio_mean",
    "inside_token_ratio_median",
    "attention_coverage_mean",
    "attention_coverage_median",
    "attention_enrichment_mean",
    "attention_enrichment_median",
    "relevance_inside_mean",
    "relevance_background_mean",
    "relevance_enrichment_mean",
    "relevance_enrichment_median",
    "lrs_z_mean",
    "lrs_z_median",
    "valid_relevance_patches",
    "total_patch_count",
]

CORRELATION_TARGETS = (
    "dice",
    "precision",
    "recall",
    "high_confidence_fp_ratio",
)

UNSUPERVISED_SCORES = (
    "entropy",
    "d25_consistency",
    "inside_token_ratio_mean",
    "inside_token_ratio_median",
    "attention_coverage_mean",
    "attention_coverage_median",
    "attention_enrichment_mean",
    "attention_enrichment_median",
    "relevance_inside_mean",
    "relevance_background_mean",
    "relevance_enrichment_mean",
    "relevance_enrichment_median",
    "lrs_z_mean",
    "lrs_z_median",
)


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else float("nan")


def binary_case_metrics(
    probability: np.ndarray,
    target: np.ndarray,
    threshold: float = 0.5,
    high_confidence_threshold: float = 0.9,
) -> dict[str, float]:
    prediction = np.asarray(probability) >= float(threshold)
    target = np.asarray(target, dtype=bool)
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction/GT shape mismatch: {prediction.shape} vs {target.shape}"
        )

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


def _resize_logits(
    logits: torch.Tensor, spatial_shape: Sequence[int]
) -> torch.Tensor:
    shape = tuple(int(v) for v in spatial_shape)
    if tuple(logits.shape[-3:]) == shape:
        return logits
    return F.interpolate(
        logits,
        size=shape,
        mode="trilinear",
        align_corners=False,
    )


def _as_d5_logits(predictions: Any) -> torch.Tensor:
    if not isinstance(predictions, (list, tuple)):
        raise ValueError("relevance audit requires all decoder outputs")
    if len(predictions) < 1:
        raise ValueError("model returned no decoder outputs")
    return predictions[0]


def _pairwise_mask_dice(masks: Sequence[np.ndarray]) -> float:
    values: list[float] = []
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            a = masks[i]
            b = masks[j]
            denominator = np.count_nonzero(a) + np.count_nonzero(b)
            values.append(
                safe_ratio(
                    2 * np.count_nonzero(a & b),
                    denominator,
                )
            )
    arr = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(arr)) if arr.size and np.isfinite(arr).any() else float("nan")


def _spearman(values: Iterable[float], target: Iterable[float]) -> float:
    values = np.asarray(list(values), dtype=np.float64)
    target = np.asarray(list(target), dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(target)
    if int(valid.sum()) < 2:
        return float("nan")

    from scipy.stats import rankdata

    first = rankdata(values[valid])
    second = rankdata(target[valid])
    if np.std(first) == 0 or np.std(second) == 0:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def _finite_mean(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def _finite_median(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else float("nan")


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def crop_global_mask_to_patch(
    global_mask: torch.Tensor,
    slicer: Sequence[slice],
) -> torch.Tensor:
    """Crop padded global mask with the exact predictor sliding-window slicer."""
    if global_mask.ndim != 4:
        raise ValueError(
            f"expected global mask (B,D,H,W), got {tuple(global_mask.shape)}"
        )
    if len(slicer) != 4:
        raise ValueError(
            f"expected (channel,D,H,W) slicer, got {len(slicer)} entries"
        )
    return global_mask[(slice(None), *tuple(slicer[1:]))]


def pseudo_mask_to_memory_grid(
    pseudo_mask: torch.Tensor,
    memory_shape_dhw: Sequence[int],
) -> torch.Tensor:
    """Map D5 pseudo-mask to projected-memory grid.

    Input pseudo_mask is (B,D,H,W).
    Returned grid is (B,Hm,Wm,Dm), exactly matching the spatial ordering of
    VoxTell's project_bottleneck_embed output before flattening.
    """
    if pseudo_mask.ndim != 4:
        raise ValueError(
            f"expected pseudo mask (B,D,H,W), got {tuple(pseudo_mask.shape)}"
        )
    dhw = tuple(int(v) for v in memory_shape_dhw)
    grid_dhw = F.interpolate(
        pseudo_mask.float().unsqueeze(1),
        size=dhw,
        mode="nearest",
    ).squeeze(1) >= 0.5
    return grid_dhw.permute(0, 2, 3, 1).contiguous()


def attention_coverage(
    attention: Any,
    inside_flat: torch.Tensor,
) -> float:
    """Fraction of normal cross-attention mass landing inside pseudo-label."""
    if attention is None:
        return float("nan")

    weights = attention.detach().float()

    # PyTorch MultiheadAttention commonly returns (B,Q,S).
    # Keep a tolerant path for (B,H,Q,S).
    if weights.ndim == 4:
        weights = weights.mean(dim=1)
    if weights.ndim != 3:
        return float("nan")

    inside_flat = inside_flat.to(
        device=weights.device,
        dtype=weights.dtype,
    )
    if tuple(inside_flat.shape) != (
        weights.shape[0],
        weights.shape[-1],
    ):
        return float("nan")

    total = weights.sum()
    if not torch.isfinite(total) or float(total) <= 0:
        return float("nan")

    inside_mass = (
        weights
        * inside_flat[:, None, :]
    ).sum()

    value = inside_mass / total
    return float(value.detach().cpu()) if torch.isfinite(value) else float("nan")


def matched_background_statistics(
    relevance_flat: torch.Tensor,
    inside_flat: torch.Tensor,
    num_samples: int,
    generator: torch.Generator,
    eps: float = 1e-8,
) -> dict[str, float] | None:
    """Compare pseudo-label relevance with equal-size background samples.

    The network is NOT perturbed. Sampling happens only on the already-computed
    relevance values.
    """
    relevance_flat = relevance_flat.detach().float().reshape(-1).cpu()
    inside_flat = inside_flat.detach().bool().reshape(-1).cpu()

    inside_values = relevance_flat[inside_flat]
    outside_values = relevance_flat[~inside_flat]

    n_inside = int(inside_values.numel())
    n_outside = int(outside_values.numel())

    if n_inside == 0 or n_outside < n_inside:
        return None

    inside_mean = float(inside_values.mean())

    bg_means: list[float] = []
    for _ in range(int(num_samples)):
        indices = torch.randperm(
            n_outside,
            generator=generator,
        )[:n_inside]
        bg_means.append(float(outside_values[indices].mean()))

    bg = np.asarray(bg_means, dtype=np.float64)
    bg_mean = float(bg.mean())
    bg_std = float(bg.std(ddof=1)) if bg.size > 1 else 0.0

    scale_eps = max(
        float(eps),
        1e-6 * max(abs(inside_mean), abs(bg_mean), 1e-12),
    )

    enrichment = inside_mean / (bg_mean + scale_eps)
    lrs_z = (inside_mean - bg_mean) / (bg_std + scale_eps)

    return {
        "inside_mean": inside_mean,
        "background_mean": bg_mean,
        "background_std": bg_std,
        "relevance_enrichment": float(enrichment),
        "lrs_z": float(lrs_z),
    }


def _run_case(
    predictor: Any,
    image: torch.Tensor,
    text_embedding: torch.Tensor,
    target: np.ndarray,
    threshold: float,
    high_confidence_threshold: float,
    background_samples: int,
    seed: int,
) -> dict[str, Any]:
    from acvl_utils.cropping_and_padding.padding import pad_nd_image
    from nnunetv2.inference.sliding_window_prediction import compute_gaussian

    device = predictor.device
    network = predictor.network.to(device).eval()

    padded, revert_padding = pad_nd_image(
        image,
        predictor.patch_size,
        "constant",
        {"value": 0},
        True,
        None,
    )
    slicers = predictor._internal_get_sliding_window_slicers(
        padded.shape[1:]
    )

    gaussian = compute_gaussian(
        tuple(predictor.patch_size),
        sigma_scale=1.0 / 8,
        value_scaling_factor=10,
        device=torch.device("cpu"),
    ).float()

    padded_shape = tuple(int(v) for v in padded.shape[1:])
    n_outputs = len(network.decoder.stages)

    normal_sum = [
        torch.zeros(padded_shape, dtype=torch.float32)
        for _ in range(n_outputs)
    ]
    denominator = torch.zeros(
        padded_shape,
        dtype=torch.float32,
    )

    # ------------------------------------------------------------------
    # Pass 1: unchanged normal VoxTell inference.
    # Build ONE Gaussian-fused global D5 pseudo-label.
    # ------------------------------------------------------------------
    for slicer in slicers:
        tile = padded[slicer][None].to(device)

        with torch.inference_mode():
            with torch.autocast(
                device_type=device.type,
                enabled=device.type == "cuda",
            ):
                outputs = network(
                    tile,
                    text_embedding,
                    return_decoder_outputs=True,
                )

        tile_shape = tuple(
            int(s.stop - s.start)
            for s in slicer[1:]
        )

        for output_index, output in enumerate(outputs):
            patch_logits = _resize_logits(
                output.float(),
                tile_shape,
            )[0, 0].cpu()

            normal_sum[output_index][slicer[1:]] += (
                patch_logits * gaussian
            )

        denominator[slicer[1:]] += gaussian

    denom = denominator.clamp_min(
        torch.finfo(torch.float32).eps
    )
    padded_normal_logits = [
        value / denom
        for value in normal_sum
    ]

    global_probability = torch.sigmoid(
        padded_normal_logits[0]
    )
    global_mask = (
        global_probability >= float(threshold)
    ).unsqueeze(0)

    # Remove image padding for all final case-level metrics.
    crop = revert_padding[1:]
    normal_logits = [
        value[crop]
        for value in padded_normal_logits
    ]
    normal_probabilities = [
        torch.sigmoid(value)
        for value in normal_logits
    ]

    probability = normal_probabilities[0].numpy()

    if tuple(target.shape) != tuple(probability.shape):
        raise ValueError(
            "prediction/GT shape mismatch: "
            f"{probability.shape} vs {target.shape}"
        )

    metrics = binary_case_metrics(
        probability,
        target,
        threshold,
        high_confidence_threshold,
    )

    metric_prediction = torch.from_numpy(
        probability >= float(threshold)
    )
    cropped_global_mask = global_mask[0][crop].cpu()

    # This must always be True. It proves the pseudo-mask used by relevance
    # comes from the exact same fused D5 prediction used for GT metrics.
    global_mask_equals_metric_mask = bool(
        torch.equal(
            metric_prediction,
            cropped_global_mask,
        )
    )
    if not global_mask_equals_metric_mask:
        raise RuntimeError(
            "global pseudo-mask differs from the binary mask used "
            "for full-volume GT metrics"
        )

    inside_voxel_ratio = float(
        cropped_global_mask.float().mean()
    )

    # Baselines retained from the earlier audit.
    d25_masks = [
        p.numpy() >= float(threshold)
        for p in normal_probabilities[:4]
    ]
    d25_consistency = _pairwise_mask_dice(
        d25_masks
    )

    probability_tensor = torch.from_numpy(
        probability
    ).clamp(1e-6, 1 - 1e-6)
    entropy = -(
        probability_tensor * probability_tensor.log()
        + (1 - probability_tensor)
        * (1 - probability_tensor).log()
    ).mean()

    # ------------------------------------------------------------------
    # Pass 2: unchanged normal inference, but gradients enabled.
    # We capture ONLY the projected transformer image-memory tensor.
    # No foreground/background token is masked.
    # ------------------------------------------------------------------
    captured: dict[str, torch.Tensor] = {}

    def capture_projected_memory(
        _module: torch.nn.Module,
        _inputs: tuple[Any, ...],
        output: torch.Tensor,
    ) -> None:
        captured["memory"] = output

    hook = network.project_bottleneck_embed.register_forward_hook(
        capture_projected_memory
    )

    # ``predictor.embed_text_prompts`` may return a PyTorch InferenceTensor
    # because text encoding is normally executed under inference mode.
    # Clone it here, outside inference_mode, to obtain a normal tensor for
    # the attribution pass. We do not need gradients w.r.t. the text itself.
    text_embedding_grad = text_embedding.detach().clone().to(device)
    text_embedding_grad.requires_grad_(False)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    inside_ratios: list[float] = []
    attention_coverages: list[float] = []
    attention_enrichments: list[float] = []

    relevance_inside: list[float] = []
    relevance_background: list[float] = []
    relevance_enrichments: list[float] = []
    lrs_values: list[float] = []

    valid_relevance_patches = 0

    try:
        for slicer in slicers:
            patch_mask = crop_global_mask_to_patch(
                global_mask,
                slicer,
            )

            # Patches with no pseudo-foreground cannot define a foreground
            # decision score, so they are intentionally skipped.
            if not bool(patch_mask.any()):
                continue

            tile = padded[slicer][None].to(device)
            network.zero_grad(set_to_none=True)
            captured.clear()

            # Full float32 is intentional here. Attribution is more stable and
            # easier to interpret than mixed-precision gradients.
            with torch.enable_grad():
                outputs, diagnostics = network(
                    tile,
                    text_embedding_grad,
                    return_decoder_outputs=True,
                    return_diagnostics=True,
                )

                if "memory" not in captured:
                    raise RuntimeError(
                        "project_bottleneck_embed hook did not capture memory"
                    )

                projected_memory = captured["memory"]
                # Shape: (B,Hm,Wm,Dm,C)
                if projected_memory.ndim != 5:
                    raise RuntimeError(
                        "unexpected projected-memory shape: "
                        f"{tuple(projected_memory.shape)}"
                    )

                d5 = _as_d5_logits(outputs)
                tile_shape = tuple(
                    int(s.stop - s.start)
                    for s in slicer[1:]
                )
                d5_patch = _resize_logits(
                    d5.float(),
                    tile_shape,
                )

                # The pseudo-mask is detached: it selects where the current
                # decision is evaluated, but is NOT part of the gradient path.
                decision_mask = (
                    patch_mask
                    .to(
                        device=d5_patch.device,
                        dtype=d5_patch.dtype,
                    )
                    .detach()
                )

                selected_logits = (
                    d5_patch[:, 0]
                    * decision_mask
                ).sum()
                selected_count = decision_mask.sum()

                if float(selected_count) <= 0:
                    continue

                score = selected_logits / selected_count

                gradient = torch.autograd.grad(
                    score,
                    projected_memory,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=False,
                )[0]

            # Language-path token relevance:
            # |activation * gradient| across embedding channels.
            relevance_grid = (
                projected_memory.detach().float()
                * gradient.detach().float()
            ).abs().sum(dim=-1)
            # Shape: (B,Hm,Wm,Dm)

            memory_shape_dhw = tuple(
                int(v)
                for v in diagnostics["memory_shape"]
            )

            inside_grid = pseudo_mask_to_memory_grid(
                patch_mask,
                memory_shape_dhw,
            ).cpu()
            inside_flat = inside_grid.reshape(
                inside_grid.shape[0],
                -1,
            )

            n_tokens = int(inside_flat.numel())
            n_inside = int(inside_flat.sum().item())
            if n_inside == 0 or n_inside >= n_tokens:
                continue

            inside_ratio = n_inside / n_tokens
            inside_ratios.append(
                float(inside_ratio)
            )

            # Normal raw cross-attention is an auxiliary metric only.
            # We use the last transformer layer here.
            cross_attention = diagnostics[
                "cross_attention"
            ]
            last_attention = (
                cross_attention[-1]
                if isinstance(
                    cross_attention,
                    (list, tuple),
                )
                else cross_attention
            )

            coverage = attention_coverage(
                last_attention,
                inside_flat,
            )
            attention_coverages.append(
                coverage
            )

            attention_enrichment = (
                coverage / inside_ratio
                if np.isfinite(coverage)
                and inside_ratio > 0
                else float("nan")
            )
            attention_enrichments.append(
                float(attention_enrichment)
            )

            stats = matched_background_statistics(
                relevance_grid.reshape(-1),
                inside_flat.reshape(-1),
                num_samples=background_samples,
                generator=generator,
            )
            if stats is None:
                continue

            valid_relevance_patches += 1
            relevance_inside.append(
                stats["inside_mean"]
            )
            relevance_background.append(
                stats["background_mean"]
            )
            relevance_enrichments.append(
                stats["relevance_enrichment"]
            )
            lrs_values.append(
                stats["lrs_z"]
            )

            # Release references to the autograd graph promptly.
            del outputs
            del diagnostics
            del projected_memory
            del gradient
            del relevance_grid

    finally:
        hook.remove()

    return {
        "dice": metrics["dice"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "fp_ratio": metrics["fp_ratio"],
        "fn_ratio": metrics["fn_ratio"],
        "high_confidence_fp_ratio": metrics[
            "high_confidence_fp_ratio"
        ],
        "entropy": float(entropy),
        "d25_consistency": d25_consistency,
        "global_mask_equals_metric_mask": (
            global_mask_equals_metric_mask
        ),
        "inside_voxel_ratio": inside_voxel_ratio,
        "inside_token_ratio_mean": _finite_mean(
            inside_ratios
        ),
        "inside_token_ratio_median": _finite_median(
            inside_ratios
        ),
        "attention_coverage_mean": _finite_mean(
            attention_coverages
        ),
        "attention_coverage_median": _finite_median(
            attention_coverages
        ),
        "attention_enrichment_mean": _finite_mean(
            attention_enrichments
        ),
        "attention_enrichment_median": _finite_median(
            attention_enrichments
        ),
        "relevance_inside_mean": _finite_mean(
            relevance_inside
        ),
        "relevance_background_mean": _finite_mean(
            relevance_background
        ),
        "relevance_enrichment_mean": _finite_mean(
            relevance_enrichments
        ),
        "relevance_enrichment_median": _finite_median(
            relevance_enrichments
        ),
        "lrs_z_mean": _finite_mean(
            lrs_values
        ),
        "lrs_z_median": _finite_median(
            lrs_values
        ),
        "valid_relevance_patches": (
            valid_relevance_patches
        ),
        "total_patch_count": len(slicers),
    }


def summarize(
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    correlations: dict[str, Any] = {}

    for score in UNSUPERVISED_SCORES:
        correlations[score] = {}
        for target in CORRELATION_TARGETS:
            score_values = [
                row[score]
                for row in rows
            ]
            target_values = [
                row[target]
                for row in rows
            ]
            valid_cases = int(
                sum(
                    np.isfinite(a)
                    and np.isfinite(b)
                    for a, b in zip(
                        score_values,
                        target_values,
                    )
                )
            )
            correlations[score][target] = {
                "spearman": _spearman(
                    score_values,
                    target_values,
                ),
                "valid_cases": valid_cases,
            }

    return _json_safe(
        {
            "prompt": PROMPT,
            "method": {
                "normal_inference_unchanged": True,
                "hard_in_out_masking": False,
                "relevance": (
                    "abs(projected_memory * "
                    "d(mean_D5_logit_inside_pseudo_mask)"
                    "/d(projected_memory))"
                ),
                "background_control": (
                    "equal-size random outside-token sets"
                ),
                "lrs_z": (
                    "(inside_mean - matched_bg_mean)"
                    " / matched_bg_std"
                ),
                "attention_enrichment": (
                    "normal_attention_coverage"
                    " / inside_token_ratio"
                ),
            },
            "correlations": correlations,
            "cases": len(rows),
        }
    )


def _find_cases(
    images: Path,
    labels: Path,
) -> list[tuple[str, Path, Path]]:
    image_paths = (
        sorted(images.glob("*.nii.gz"))
        if images.is_dir()
        else sorted(
            images.parent.glob(images.name)
        )
    )

    result: list[
        tuple[str, Path, Path]
    ] = []

    for image in image_paths:
        label = labels / image.name
        if label.is_file():
            result.append(
                (
                    image.name.removesuffix(
                        ".nii.gz"
                    ),
                    image,
                    label,
                )
            )

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline VoxTell language-conditioned "
            "pseudo-label relevance audit"
        )
    )

    parser.add_argument(
        "--images",
        required=True,
    )
    parser.add_argument(
        "--labels",
        required=True,
    )
    parser.add_argument(
        "--model",
        required=True,
    )
    parser.add_argument(
        "--text-model",
        default="Qwen/Qwen3-Embedding-4B",
    )
    parser.add_argument(
        "--gt-label",
        type=int,
        default=5,
        help=(
            "Liver label value; use 0 to treat "
            "all non-zero labels as liver"
        ),
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--high-confidence-threshold",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--background-samples",
        type=int,
        default=32,
        help=(
            "Number of equal-size random "
            "background samples per valid patch"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260911,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--output-csv",
        default=(
            "output/"
            "text_relevance_quality_per_case.csv"
        ),
    )
    parser.add_argument(
        "--summary-json",
        default=(
            "output/"
            "text_relevance_quality_summary.json"
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not 0 <= args.threshold <= 1:
        raise ValueError(
            "threshold must be in [0,1]"
        )
    if not 0 <= args.high_confidence_threshold <= 1:
        raise ValueError(
            "high-confidence threshold "
            "must be in [0,1]"
        )
    if (
        args.high_confidence_threshold
        < args.threshold
    ):
        raise ValueError(
            "high-confidence threshold "
            "must be >= threshold"
        )
    if args.background_samples < 2:
        raise ValueError(
            "background-samples must be >= 2"
        )

    from nnunetv2.imageio.nibabel_reader_writer import (
        NibabelIOWithReorient,
    )
    from voxtell.inference.predictor_multiclass import (
        VoxTellPredictor,
    )

    device = torch.device(
        f"cuda:{args.gpu}"
        if args.device == "cuda"
        and torch.cuda.is_available()
        else "cpu"
    )

    cases = _find_cases(
        Path(args.images),
        Path(args.labels),
    )
    if args.limit is not None:
        cases = cases[: args.limit]

    if not cases:
        raise RuntimeError(
            "No image/label pairs found"
        )

    predictor = VoxTellPredictor(
        str(args.model),
        device=device,
        text_encoding_model=args.text_model,
    )

    reader = NibabelIOWithReorient()

    # Fixed prompt by design.
    text_embedding = predictor.embed_text_prompts(
        [PROMPT]
    )

    rows: list[dict[str, Any]] = []

    for case_index, (
        case_id,
        image_path,
        label_path,
    ) in enumerate(cases):
        image, _ = reader.read_images(
            [str(image_path)]
        )
        label, _ = reader.read_images(
            [str(label_path)]
        )

        image_tensor, bbox, _ = predictor.preprocess(
            image
        )

        labels = np.rint(
            label[0]
        ).astype(np.int64)

        gt_full = (
            labels != 0
            if args.gt_label == 0
            else labels == args.gt_label
        )

        gt_crop = gt_full[
            tuple(
                slice(
                    int(lo),
                    int(hi),
                )
                for lo, hi in bbox
            )
        ]

        row = _run_case(
            predictor,
            image_tensor,
            text_embedding,
            gt_crop,
            args.threshold,
            args.high_confidence_threshold,
            args.background_samples,
            args.seed + case_index,
        )
        row["case_id"] = case_id
        rows.append(row)

        print(
            f"[{len(rows)}/{len(cases)}] "
            f"{case_id} "
            f"Dice={row['dice']:.4f} "
            f"LRS(z)={row['lrs_z_median']:.4f} "
            f"RelEnrich="
            f"{row['relevance_enrichment_median']:.4f} "
            f"AttnEnrich="
            f"{row['attention_enrichment_median']:.4f}"
        )

    output_csv = Path(
        args.output_csv
    )
    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=CSV_FIELDS,
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    summary["threshold"] = args.threshold
    summary[
        "high_confidence_threshold"
    ] = args.high_confidence_threshold
    summary[
        "background_samples"
    ] = args.background_samples
    summary["seed"] = args.seed

    summary_json = Path(
        args.summary_json
    )
    summary_json.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    summary_json.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
