"""Oracle-support localization diagnostic for VoxTell.

This module is deliberately independent of training. It reuses VoxTell's
preprocessing, patch size, sliding-window coordinates, and nnU-Net Gaussian
weighting while exposing the D5 visual tensor immediately before its final
text-conditioned mask head.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform
from scipy import ndimage
from sklearn.metrics import average_precision_score, roc_auc_score


SCORE_MODES = ("max", "top3_mean", "top5_mean")
BANK_NAMES = ("oracle", "prediction", "random")
ALPHAS = (0.25, 0.5, 1.0, 2.0)
SEED = 42
EPS = 1e-8


@dataclass(frozen=True)
class CaseFiles:
    case_id: str
    image: Path
    label: Path
    prediction: Path | None
    zero_shot_dice: float
    roi_bbox: tuple[tuple[int, int], tuple[int, int], tuple[int, int]] | None = None


def normalize_case_id(value: Any) -> str:
    text = str(value).strip().replace("\\", "/").rsplit("/", 1)[-1]
    lower = text.lower()
    for suffix in (".nii.gz", ".nii", ".npy", ".npz"):
        if lower.endswith(suffix):
            return text[: -len(suffix)]
    return text


def _json_preview(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_preview(v) for k, v in list(value.items())[:3]}
    if isinstance(value, list):
        return [_json_preview(v) for v in value[:2]]
    return value


def inspect_split_json(path: Path, available_ids: set[str]) -> list[str]:
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    print(f"[schema] split JSON type={type(payload).__name__}")
    if isinstance(payload, dict):
        print(f"[schema] split JSON top-level keys={list(payload.keys())}")
        train_keys = [
            key for key in payload
            if str(key).strip().lower() in {"train", "training"}
        ]
        if len(train_keys) != 1:
            raise ValueError(
                "split JSON must have exactly one explicit 'train' or 'training' "
                f"key; found {train_keys}. Refusing to guess."
            )
        entries = payload[train_keys[0]]
    elif isinstance(payload, list):
        print("[schema] split JSON is a bare list; treating the list as training cases")
        entries = payload
    else:
        raise ValueError("split JSON must be an object or list")
    print(f"[schema] split JSON sample={_json_preview(entries)}")
    if not isinstance(entries, list) or not entries:
        raise ValueError("training split must be a non-empty list")

    if all(isinstance(item, (str, int)) for item in entries):
        ids = [normalize_case_id(item) for item in entries]
    elif all(isinstance(item, dict) for item in entries):
        common = set(entries[0])
        for item in entries[1:]:
            common &= set(item)
        scored: list[tuple[int, str]] = []
        for key in sorted(common):
            values = [item[key] for item in entries]
            if all(isinstance(v, (str, int)) for v in values):
                score = sum(normalize_case_id(v) in available_ids for v in values)
                scored.append((score, str(key)))
        if not scored:
            raise ValueError("no scalar case-ID field exists in split entries")
        best = max(score for score, _ in scored)
        winners = [key for score, key in scored if score == best and score > 0]
        print(f"[schema] split ID-field overlap candidates={scored}")
        if len(winners) != 1:
            raise ValueError(
                f"split case-ID field is ambiguous ({winners}); refusing to guess"
            )
        selected = winners[0]
        print(f"[schema] resolved split case-ID field={selected!r}")
        ids = [normalize_case_id(item[selected]) for item in entries]
    else:
        raise ValueError("split training list mixes incompatible entry types")
    if len(ids) != len(set(ids)):
        duplicates = sorted({x for x in ids if ids.count(x) > 1})
        raise ValueError(f"duplicate training case IDs in split JSON: {duplicates}")
    return ids


def inspect_dice_csv(
    path: Path,
    train_ids: set[str],
    available_ids: set[str],
) -> dict[str, float]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    print(f"[schema] Dice CSV columns={fieldnames}")
    print(f"[schema] Dice CSV first rows={rows[:2]}")
    if not fieldnames or not rows:
        raise ValueError("Dice CSV is empty or has no header")

    universe = train_ids | available_ids
    overlap_scores: list[tuple[int, str]] = []
    for field in fieldnames:
        overlap = sum(normalize_case_id(row.get(field, "")) in universe for row in rows)
        overlap_scores.append((overlap, field))
    best = max(score for score, _ in overlap_scores)
    id_fields = [field for score, field in overlap_scores if score == best and score > 0]
    print(f"[schema] CSV case-ID overlap candidates={overlap_scores}")
    if len(id_fields) != 1:
        raise ValueError(f"CSV case-ID column is ambiguous ({id_fields}); refusing to guess")
    id_field = id_fields[0]

    dice_fields = [field for field in fieldnames if "dice" in field.strip().lower()]
    numeric_dice_fields = []
    for field in dice_fields:
        values = []
        for row in rows:
            case_id = normalize_case_id(row.get(id_field, ""))
            if case_id not in train_ids:
                continue
            try:
                values.append(float(row[field]))
            except (TypeError, ValueError):
                pass
        if values and all(np.isfinite(values)):
            numeric_dice_fields.append(field)
    if len(numeric_dice_fields) != 1:
        raise ValueError(
            "expected exactly one finite numeric Dice column for training cases; "
            f"found {numeric_dice_fields}. Refusing to guess."
        )
    dice_field = numeric_dice_fields[0]
    print(f"[schema] resolved CSV columns: case_id={id_field!r}, dice={dice_field!r}")

    result: dict[str, float] = {}
    for row in rows:
        case_id = normalize_case_id(row.get(id_field, ""))
        if case_id not in train_ids:
            continue
        if case_id in result:
            raise ValueError(f"duplicate Dice CSV row for case {case_id}")
        try:
            value = float(row[dice_field])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"non-numeric Dice for case {case_id}: {row[dice_field]!r}") from exc
        if not np.isfinite(value) or not (0.0 <= value <= 1.0):
            raise ValueError(f"Dice outside [0,1] for case {case_id}: {value}")
        result[case_id] = value
    missing = sorted(train_ids - set(result))
    if missing:
        raise FileNotFoundError(f"training cases missing from Dice CSV ({len(missing)}): {missing}")
    return result


def index_case_files(directory: Path, extensions: Sequence[str]) -> dict[str, Path]:
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    indexed: dict[str, Path] = {}
    collisions: dict[str, list[Path]] = {}
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        if not any(path.name.lower().endswith(ext) for ext in extensions):
            continue
        case_id = normalize_case_id(path.name)
        if case_id in indexed:
            collisions.setdefault(case_id, [indexed[case_id]]).append(path)
        else:
            indexed[case_id] = path
    if collisions:
        detail = {key: [str(p) for p in value] for key, value in collisions.items()}
        raise ValueError(f"ambiguous files: multiple paths map to one case ID: {detail}")
    return indexed


def audit_inputs(args: argparse.Namespace) -> tuple[list[CaseFiles], list[CaseFiles], dict[str, Any]]:
    image_index = index_case_files(Path(args.image_dir), (".nii.gz", ".nii"))
    label_index = index_case_files(Path(args.label_dir), (".nii.gz", ".nii"))
    prediction_index = (
        index_case_files(Path(args.prediction_dir), (".nii.gz", ".nii", ".npy", ".npz"))
        if args.prediction_dir else {}
    )
    available_ids = set(image_index) | set(label_index) | set(prediction_index)
    train_ids = inspect_split_json(Path(args.split_json), available_ids)
    dice = inspect_dice_csv(Path(args.dice_csv), set(train_ids), available_ids)

    supports = [case_id for case_id in train_ids if dice[case_id] > args.support_threshold]
    queries = [case_id for case_id in train_ids if dice[case_id] < args.query_threshold]
    print(
        f"[audit] train={len(train_ids)} support(Dice>{args.support_threshold})={len(supports)} "
        f"query(Dice<{args.query_threshold})={len(queries)}"
    )
    if not supports or not queries:
        raise ValueError("support or query set is empty at the requested thresholds")

    needed = set(supports) | set(queries)
    checked_train = set(train_ids)
    missing_images = sorted(checked_train - set(image_index))
    missing_labels = sorted(checked_train - set(label_index))
    missing_predictions = sorted(checked_train - set(prediction_index)) if args.prediction_dir else []
    print(f"[audit] missing images={missing_images}")
    print(f"[audit] missing labels={missing_labels}")
    print(f"[audit] missing predictions={missing_predictions}")
    if missing_images or missing_labels or missing_predictions:
        raise FileNotFoundError("required case files are missing; see audit above")

    roi_manifest_status = "not_found; supplied volumes treated as complete liver ROIs"
    roi_bboxes: dict[str, tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] = {}
    roi_manifest = Path(args.image_dir).parent / "roi_info.csv"
    if roi_manifest.is_file():
        with roi_manifest.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            roi_fields = list(reader.fieldnames or [])
            roi_rows = list(reader)
        print(f"[schema] liver ROI CSV columns={roi_fields}")
        required_roi_fields = {"filename", "size_d", "size_h", "size_w"}
        if not roi_rows or not required_roi_fields.issubset(roi_fields):
            raise ValueError(
                f"{roi_manifest}: expected explicit ROI fields {sorted(required_roi_fields)}"
            )
        roi_by_id: dict[str, tuple[int, int, int]] = {}
        for row in roi_rows:
            case_id = normalize_case_id(row["filename"])
            if case_id in roi_by_id:
                raise ValueError(f"duplicate ROI-manifest row for {case_id}")
            roi_by_id[case_id] = tuple(int(row[key]) for key in ("size_d", "size_h", "size_w"))
            coordinate_fields = ("d0", "d1", "h0", "h1", "w0", "w1")
            if all(key in row and row[key] != "" for key in coordinate_fields):
                roi_bboxes[case_id] = (
                    (int(row["d0"]), int(row["d1"])),
                    (int(row["h0"]), int(row["h1"])),
                    (int(row["w0"]), int(row["w1"])),
                )
        missing_roi_rows = sorted(checked_train - set(roi_by_id))
        if missing_roi_rows:
            raise FileNotFoundError(f"cases missing from liver ROI manifest: {missing_roi_rows}")
        for case_id in sorted(checked_train):
            header_image = nib.load(str(image_index[case_id]))
            reoriented = header_image.as_reoriented(io_orientation(header_image.affine))
            reader_shape = tuple(int(v) for v in reoriented.shape[::-1])
            if reader_shape != roi_by_id[case_id]:
                raise ValueError(
                    f"{case_id}: image shape {reader_shape} differs from "
                    f"roi_info.csv {roi_by_id[case_id]}"
                )
        roi_manifest_status = f"validated:{roi_manifest}"
        print(f"[audit] validated liver ROI manifest for all {len(checked_train)} training cases")
    else:
        print(
            "[audit] no sibling roi_info.csv; applying the documented contract that "
            "each supplied image is already one complete liver ROI"
        )

    def make(case_id: str) -> CaseFiles:
        return CaseFiles(
            case_id=case_id,
            image=image_index[case_id],
            label=label_index[case_id],
            prediction=prediction_index.get(case_id),
            zero_shot_dice=dice[case_id],
            roi_bbox=roi_bboxes.get(case_id),
        )

    support_cases = [make(case_id) for case_id in supports]
    query_cases = [make(case_id) for case_id in queries]
    metadata = {
        "train_cases": len(train_ids),
        "support_cases": len(support_cases),
        "query_cases": len(query_cases),
        "support_threshold": args.support_threshold,
        "query_threshold": args.query_threshold,
        "prediction_dir_supplied": bool(args.prediction_dir),
        "roi_manifest": roi_manifest_status,
        "liver_roi_contract": (
            "Each --image-dir volume is treated as one precomputed complete liver ROI; "
            "VoxTell crop_to_nonzero defines the valid inference subregion."
        ),
    }
    return support_cases, query_cases, metadata


def align_mask_nearest(mask: np.ndarray, target_shape: Sequence[int]) -> np.ndarray:
    source = np.asarray(mask, dtype=np.float32)
    if source.ndim != 3 or len(target_shape) != 3:
        raise ValueError("mask and target must be 3-D")
    tensor = torch.from_numpy(source)[None, None]
    aligned = F.interpolate(tensor, size=tuple(target_shape), mode="nearest")[0, 0]
    return aligned.numpy() > 0.5


def connected_lesions(mask: np.ndarray) -> tuple[np.ndarray, int]:
    return ndimage.label(np.asarray(mask, dtype=bool), structure=np.ones((3, 3, 3), dtype=bool))


def erode_lesions(mask: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]]]:
    labels, count = connected_lesions(mask)
    eroded = np.zeros_like(mask, dtype=bool)
    records = []
    structure = np.ones((3, 3, 3), dtype=bool)
    for lesion_id in range(1, count + 1):
        lesion = labels == lesion_id
        candidate = ndimage.binary_erosion(lesion, structure=structure, iterations=1)
        fallback = not bool(candidate.any())
        selected = lesion if fallback else candidate
        eroded |= selected
        records.append(
            {
                "lesion_id": lesion_id,
                "original_voxels": int(lesion.sum()),
                "prototype_voxels": int(selected.sum()),
                "erosion_fallback": fallback,
            }
        )
    return eroded, records


def normalized_prototype(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("prototype needs a non-empty [voxels, channels] array")
    values = values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), EPS)
    prototype = values.mean(axis=0)
    norm = float(np.linalg.norm(prototype))
    if not np.isfinite(norm) or norm <= EPS:
        raise ValueError("prototype mean has zero or non-finite norm")
    return (prototype / norm).astype(np.float32)


def exact_topk_mask(scores: np.ndarray, roi: np.ndarray, k: int) -> np.ndarray:
    roi_flat = np.flatnonzero(np.asarray(roi, dtype=bool).ravel())
    requested = int(k)
    k = min(max(requested, 0), int(roi_flat.size))
    result = np.zeros(np.asarray(scores).size, dtype=bool)
    if k:
        values = np.asarray(scores, dtype=np.float64).ravel()[roi_flat]
        if k == len(values):
            chosen = roi_flat
        else:
            threshold = np.partition(values, len(values) - k)[len(values) - k]
            higher_local = np.flatnonzero(values > threshold)
            tied_local = np.flatnonzero(values == threshold)
            remaining = k - len(higher_local)
            # Deterministic tie break: roi_flat and tied_local are ordered, so
            # take the smallest flat voxel indices without sorting all scores.
            selected_local = np.concatenate((higher_local, tied_local[:remaining]))
            chosen = roi_flat[selected_local]
        result[chosen] = True
    result = result.reshape(np.asarray(scores).shape)
    assert int(result.sum()) == k
    return result


def gaussian_fuse_patches(
    patches: Sequence[np.ndarray],
    slicers: Sequence[tuple[slice, ...]],
    gaussian: np.ndarray,
    output_shape: Sequence[int],
) -> np.ndarray:
    total = np.zeros(tuple(output_shape), dtype=np.float32)
    denominator = np.zeros(tuple(output_shape), dtype=np.float32)
    for patch, slicer in zip(patches, slicers):
        spatial = slicer[-3:]
        total[spatial] += np.asarray(patch, np.float32) * gaussian
        denominator[spatial] += gaussian
    if np.any(denominator <= 0):
        raise AssertionError("sliding-window Gaussian left uncovered voxels")
    return total / denominator


def binary_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    pred = np.asarray(prediction, dtype=bool)
    gt = np.asarray(target, dtype=bool)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    pred_n = int(pred.sum())
    gt_n = int(gt.sum())
    dice_den = 2 * tp + fp + fn
    return {
        "dice": float(2 * tp / dice_den) if dice_den else 1.0,
        "precision": float(tp / (tp + fp)) if (tp + fp) else (1.0 if gt_n == 0 else 0.0),
        "recall": float(tp / (tp + fn)) if (tp + fn) else 1.0,
        "pred_foreground_voxels": pred_n,
        "gt_foreground_voxels": gt_n,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float, str]:
    y = np.asarray(labels, dtype=bool).ravel()
    s = np.asarray(scores, dtype=np.float64).ravel()
    classes = np.unique(y)
    if classes.size == 0:
        return math.nan, math.nan, "empty_evaluation_region"
    if classes.size < 2:
        reason = "single_class_no_positive" if not bool(classes[0]) else "single_class_no_negative"
        return math.nan, math.nan, reason
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s)), ""


def localization_metrics(
    score: np.ndarray,
    target: np.ndarray,
    roi: np.ndarray,
    baseline_probability: np.ndarray,
) -> dict[str, Any]:
    gt = np.asarray(target, bool) & roi
    gt_n = int(gt.sum())
    if gt_n == 0:
        raise ValueError("query GT is empty; volume-based localization is undefined")
    auroc, auprc, auc_reason = safe_auc(gt[roi], score[roi])
    top1 = exact_topk_mask(score, roi, gt_n)
    result: dict[str, Any] = {
        "auroc": auroc,
        "auprc": auprc,
        "auc_nan_reason": auc_reason,
        "dice_at_oracle_volume": binary_metrics(top1, gt)["dice"],
    }
    for multiplier in (1, 2, 5):
        selected = exact_topk_mask(score, roi, multiplier * gt_n)
        result[f"recall_at_{multiplier}v"] = float(np.logical_and(selected, gt).sum() / gt_n)
    lesion_labels, lesion_count = connected_lesions(gt)
    hits = [bool(np.logical_and(top1, lesion_labels == lesion_id).any()) for lesion_id in range(1, lesion_count + 1)]
    result["lesion_count"] = lesion_count
    result["lesion_level_hit_rate"] = float(np.mean(hits)) if hits else math.nan
    precision_top = float(np.logical_and(top1, gt).sum() / max(int(top1.sum()), 1))
    prevalence = float(gt_n / max(int(roi.sum()), 1))
    result["top_score_foreground_enrichment"] = float(precision_top / prevalence) if prevalence else math.nan

    fn_roi = roi & (baseline_probability < 0.5)
    fn_gt = gt & fn_roi
    fn_n = int(fn_gt.sum())
    fn_auroc, fn_auprc, fn_reason = safe_auc(fn_gt[fn_roi], score[fn_roi])
    result.update(
        {
            "fn_region_voxels": int(fn_roi.sum()),
            "fn_positive_voxels": fn_n,
            "fn_region_auroc": fn_auroc,
            "fn_region_auprc": fn_auprc,
            "fn_auc_nan_reason": fn_reason,
        }
    )
    for multiplier in (1, 2, 5):
        if fn_n == 0:
            recall = math.nan
        else:
            selected = exact_topk_mask(score, fn_roi, multiplier * fn_n)
            recall = float(np.logical_and(selected, fn_gt).sum() / fn_n)
        result[f"fn_recall_at_{multiplier}v"] = recall
    return result


def score_modes(features: torch.Tensor, prototypes: torch.Tensor, chunk_size: int = 262_144) -> dict[str, torch.Tensor]:
    if prototypes.ndim != 2 or prototypes.shape[0] == 0:
        raise ValueError("prototype bank is empty")
    channels = features.shape[0]
    flat = features.permute(1, 2, 3, 0).reshape(-1, channels)
    outputs = {mode: [] for mode in SCORE_MODES}
    bank = F.normalize(prototypes.float(), dim=1)
    for start in range(0, flat.shape[0], chunk_size):
        values = F.normalize(flat[start : start + chunk_size].float(), dim=1)
        similarities = values @ bank.T
        outputs["max"].append(similarities.max(dim=1).values)
        for requested, mode in ((3, "top3_mean"), (5, "top5_mean")):
            k = min(requested, similarities.shape[1])
            outputs[mode].append(torch.topk(similarities, k=k, dim=1).values.mean(dim=1))
    spatial = tuple(int(v) for v in features.shape[1:])
    return {key: torch.cat(value).reshape(spatial) for key, value in outputs.items()}


def _bbox_slices(bbox: Sequence[Sequence[int]]) -> tuple[slice, ...]:
    return tuple(slice(int(lo), int(hi)) for lo, hi in bbox)


def _pad_spatial(array: np.ndarray, padded_shape: Sequence[int], revert: Sequence[slice]) -> np.ndarray:
    result = np.zeros(tuple(padded_shape), dtype=array.dtype)
    result[tuple(revert[1:])] = array
    return result


def _restore_crop(array: np.ndarray, bbox: Sequence[Sequence[int]], original_shape: Sequence[int]) -> np.ndarray:
    result = np.zeros(tuple(original_shape), dtype=array.dtype)
    result[_bbox_slices(bbox)] = array
    return result


def _load_nifti(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient

    data, properties = NibabelIOWithReorient().read_images([str(path)])
    return np.asarray(data[0]), properties


def load_case(case: CaseFiles) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    image, properties = _load_nifti(case.image)
    label, label_properties = _load_nifti(case.label)
    if image.shape != label.shape:
        raise ValueError(f"{case.case_id}: image/label shape mismatch {image.shape} vs {label.shape}")
    image_affine = properties["nibabel_stuff"]["original_affine"]
    label_affine = label_properties["nibabel_stuff"]["original_affine"]
    if not np.allclose(image_affine, label_affine, atol=1e-4):
        raise ValueError(f"{case.case_id}: image/label affine mismatch")
    if not np.all(np.isfinite(image)) or not np.all(np.isfinite(label)):
        raise ValueError(f"{case.case_id}: image or label contains non-finite values")
    rounded_label = np.rint(label)
    if not np.allclose(label, rounded_label, atol=1e-5):
        raise ValueError(f"{case.case_id}: tumor GT is not integer-valued")
    positive_values = np.unique(rounded_label[rounded_label > 0])
    if positive_values.size > 1:
        raise ValueError(
            f"{case.case_id}: tumor GT has ambiguous positive labels {positive_values.tolist()}; "
            "provide a binary tumor-GT directory"
        )
    return image.astype(np.float32, copy=False), rounded_label > 0, properties


def load_probability(
    path: Path,
    expected_shape: Sequence[int],
    roi_bbox: tuple[tuple[int, int], tuple[int, int], tuple[int, int]] | None = None,
) -> tuple[np.ndarray, str]:
    lower = path.name.lower()
    if lower.endswith((".nii", ".nii.gz")):
        array, _ = _load_nifti(path)
    elif lower.endswith(".npy"):
        array = np.load(path, allow_pickle=False)
    elif lower.endswith(".npz"):
        with np.load(path, allow_pickle=False) as archive:
            if len(archive.files) != 1:
                raise ValueError(f"{path}: NPZ has multiple arrays {archive.files}; refusing to guess")
            array = archive[archive.files[0]]
    else:
        raise ValueError(f"unsupported prediction file: {path}")
    array = np.squeeze(np.asarray(array, dtype=np.float32))
    if tuple(array.shape) != tuple(expected_shape) and roi_bbox is not None:
        slices = _bbox_slices(roi_bbox)
        if all(0 <= int(s.start) < int(s.stop) <= array.shape[i] for i, s in enumerate(slices)):
            cropped = array[slices]
            if tuple(cropped.shape) == tuple(expected_shape):
                print(
                    f"[geometry] {path.name}: cropped full-volume prediction {array.shape} "
                    f"to liver ROI {cropped.shape} using roi_info.csv"
                )
                array = cropped
    if tuple(array.shape) != tuple(expected_shape):
        raise ValueError(
            f"{path}: probability shape {array.shape} != liver-ROI image shape "
            f"{tuple(expected_shape)}; no unique valid roi_info.csv crop resolved it"
        )
    if not np.all(np.isfinite(array)) or float(array.min()) < 0 or float(array.max()) > 1:
        raise ValueError(f"{path}: expected finite probability/mask values in [0,1]")
    kind = "binary_mask" if np.all(np.isin(np.unique(array), (0.0, 1.0))) else "probability"
    return array, kind


def _compute_denominator(slicers: Sequence[tuple[slice, ...]], gaussian: torch.Tensor, shape: Sequence[int]) -> np.ndarray:
    denominator = np.zeros(tuple(shape), dtype=np.float32)
    weight = gaussian.cpu().numpy().astype(np.float32, copy=False)
    for slicer in slicers:
        denominator[slicer[1:]] += weight
    if not np.all(np.isfinite(denominator)) or np.any(denominator <= 0):
        raise AssertionError("invalid Gaussian denominator or uncovered sliding-window voxel")
    return denominator


def _network_patch(predictor: Any, patch: torch.Tensor, text_embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.inference_mode():
        autocast = torch.autocast(device_type=predictor.device.type, enabled=predictor.device.type == "cuda")
        with autocast:
            prediction, features = predictor.network(
                patch,
                text_embedding,
                return_features=True,
            )
    if isinstance(prediction, list):
        prediction = prediction[0]
    d5 = features["d5_pre_head"]
    if d5.ndim != 6 or d5.shape[1] != 1:
        raise AssertionError(f"unexpected D5 pre-head feature shape: {tuple(d5.shape)}")
    return prediction[0, 0].float(), d5[0, 0].float()


def extract_support_case(
    predictor: Any,
    text_embedding: torch.Tensor,
    case: CaseFiles,
) -> tuple[list[np.ndarray], list[np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    from acvl_utils.cropping_and_padding.padding import pad_nd_image
    from nnunetv2.inference.sliding_window_prediction import compute_gaussian

    image, gt, _ = load_case(case)
    preprocessed, bbox, original_shape = predictor.preprocess(image)
    crop = _bbox_slices(bbox)
    gt_crop = gt[crop]
    padded, revert = pad_nd_image(preprocessed, predictor.patch_size, "constant", {"value": 0}, True, None)
    padded_shape = tuple(int(v) for v in padded.shape[1:])
    slicers = predictor._internal_get_sliding_window_slicers(padded_shape)
    gaussian = compute_gaussian(tuple(predictor.patch_size), sigma_scale=1 / 8, value_scaling_factor=10, device=torch.device("cpu")).float()
    denominator = _compute_denominator(slicers, gaussian, padded_shape)

    if case.prediction is not None:
        probability, probability_kind = load_probability(case.prediction, original_shape, case.roi_bbox)
        pred_crop = probability[crop] >= 0.5
    else:
        probability = None
        probability_kind = "model_probability"
        pred_crop = None

    gt_aligned = align_mask_nearest(gt_crop, gt_crop.shape)
    _, gt_records = erode_lesions(gt_aligned)
    gt_labels, gt_count = connected_lesions(gt_aligned)
    gt_selected_labels = np.zeros_like(gt_labels)
    for record in gt_records:
        lesion_id = int(record["lesion_id"])
        lesion = gt_labels == lesion_id
        candidate = ndimage.binary_erosion(lesion, structure=np.ones((3, 3, 3), bool))
        gt_selected_labels[lesion if record["erosion_fallback"] else candidate] = lesion_id

    gt_padded = _pad_spatial(gt_selected_labels.astype(np.int32), padded_shape, revert)
    pred_labels_padded = None
    baseline_sum = np.zeros(padded_shape, dtype=np.float32)
    feature_shape = None
    channels = None
    feature_accumulators: dict[str, np.ndarray] = {}
    selected_indices: dict[str, np.ndarray] = {}

    # If external predictions are available, their connected components are known
    # before inference. Otherwise pass one first establishes the model mask.
    passes = 1 if pred_crop is not None else 2
    for pass_index in range(passes):
        if pass_index == 1:
            model_probability = 1 / (1 + np.exp(-np.clip(baseline_sum / denominator, -50, 50)))
            pred_crop = model_probability[tuple(revert[1:])] >= 0.5
        if pred_crop is not None and pred_labels_padded is None:
            pred_labels, _ = connected_lesions(align_mask_nearest(pred_crop, pred_crop.shape))
            pred_labels_padded = _pad_spatial(pred_labels.astype(np.int32), padded_shape, revert)

        collect_features = pass_index == passes - 1
        if collect_features and not selected_indices:
            if pred_labels_padded is None:
                raise AssertionError("prediction labels were not established before feature collection")
            masks = {"oracle": gt_padded > 0, "prediction": pred_labels_padded > 0}
            # Channel count is learned from the first real patch below.
            for name, mask in masks.items():
                selected_indices[name] = np.flatnonzero(mask.ravel())
        for slicer in slicers:
            patch = padded[slicer][None].to(predictor.device)
            logits, features = _network_patch(predictor, patch, text_embedding)
            raw_shape = tuple(int(v) for v in features.shape[1:])
            tile_shape = tuple(int(s.stop - s.start) for s in slicer[1:])
            feature_shape = raw_shape
            channels = int(features.shape[0])
            if raw_shape != tile_shape:
                raise AssertionError(
                    "D5 pre-head grid unexpectedly differs from its sliding-window grid: "
                    f"D5={raw_shape}, tile={tile_shape}. Refusing implicit feature resampling."
                )
            if tuple(logits.shape) != tile_shape:
                logits = F.interpolate(logits[None, None], size=tile_shape, mode="trilinear", align_corners=False)[0, 0]
            if pass_index == 0:
                baseline_sum[slicer[1:]] += (logits.cpu() * gaussian).numpy()
            if not collect_features:
                continue
            if not feature_accumulators:
                for name, indices in selected_indices.items():
                    feature_accumulators[name] = np.zeros((len(indices), channels), dtype=np.float32)
            feature_flat = features.permute(1, 2, 3, 0).reshape(-1, channels).cpu().numpy()
            weight_flat = gaussian.numpy().ravel()
            d_slice, h_slice, w_slice = slicer[1:]
            patch_shape = tile_shape
            for name, label_volume in (("oracle", gt_padded), ("prediction", pred_labels_padded)):
                local_labels = label_volume[slicer[1:]]
                local_flat = np.flatnonzero(local_labels.ravel() > 0)
                if not len(local_flat):
                    continue
                local_coords = np.stack(np.unravel_index(local_flat, patch_shape), axis=1)
                global_d = local_coords[:, 0] + int(d_slice.start)
                global_h = local_coords[:, 1] + int(h_slice.start)
                global_w = local_coords[:, 2] + int(w_slice.start)
                global_flat = np.ravel_multi_index((global_d, global_h, global_w), padded_shape)
                positions = np.searchsorted(selected_indices[name], global_flat)
                if not np.array_equal(selected_indices[name][positions], global_flat):
                    raise AssertionError("foreground compact-index mapping failed")
                feature_accumulators[name][positions] += feature_flat[local_flat] * weight_flat[local_flat, None]

    if feature_shape is None or channels is None:
        raise RuntimeError("no sliding-window patches were processed")
    oracle_prototypes: list[np.ndarray] = []
    prediction_prototypes: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for bank, label_volume, records in (
        ("oracle", gt_padded, gt_records),
        ("prediction", pred_labels_padded, None),
    ):
        indices = selected_indices[bank]
        if not len(indices):
            rows.append({"bank": bank, "case_id": case.case_id, "status": "empty_source_mask"})
            continue
        fused = feature_accumulators[bank] / denominator.ravel()[indices, None]
        compact_labels = label_volume.ravel()[indices]
        lesion_ids = sorted(int(v) for v in np.unique(compact_labels) if v > 0)
        for lesion_id in lesion_ids:
            lesion_features = fused[compact_labels == lesion_id]
            prototype = normalized_prototype(lesion_features)
            target_list = oracle_prototypes if bank == "oracle" else prediction_prototypes
            target_list.append(prototype)
            erosion_record = records[lesion_id - 1] if records is not None else {}
            rows.append(
                {
                    "bank": bank,
                    "prototype_index_in_case": len(target_list) - 1,
                    "case_id": case.case_id,
                    "lesion_id": lesion_id,
                    "zero_shot_dice": case.zero_shot_dice,
                    "source_voxels": int((compact_labels == lesion_id).sum()),
                    "original_gt_voxels": erosion_record.get("original_voxels", ""),
                    "erosion_fallback": erosion_record.get("erosion_fallback", ""),
                    "feature_channels": channels,
                    "raw_d5_patch_shape": "x".join(map(str, feature_shape)),
                    "aligned_patch_shape": "x".join(map(str, predictor.patch_size)),
                    "feature_to_roi_mapping": "D5 patch coordinates -> sliding-window slicer -> preprocessed ROI bbox -> original ROI",
                    "prototype_l2_norm": float(np.linalg.norm(prototype)),
                    "probability_source_type": probability_kind,
                    "status": "ok",
                }
            )
    return oracle_prototypes, prediction_prototypes, rows, {
        "feature_channels": channels,
        "raw_d5_patch_shape": feature_shape,
        "aligned_patch_shape": tuple(int(v) for v in predictor.patch_size),
        "gt_lesions": gt_count,
        "gt_erosion_fallbacks": sum(bool(r["erosion_fallback"]) for r in gt_records),
    }


def run_query_case(
    predictor: Any,
    text_embedding: torch.Tensor,
    case: CaseFiles,
    prototype_banks: dict[str, np.ndarray],
    output_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    from acvl_utils.cropping_and_padding.padding import pad_nd_image
    from nnunetv2.inference.sliding_window_prediction import compute_gaussian

    image, gt, properties = load_case(case)
    preprocessed, bbox, original_shape = predictor.preprocess(image)
    crop = _bbox_slices(bbox)
    padded, revert = pad_nd_image(preprocessed, predictor.patch_size, "constant", {"value": 0}, True, None)
    padded_shape = tuple(int(v) for v in padded.shape[1:])
    slicers = predictor._internal_get_sliding_window_slicers(padded_shape)
    gaussian = compute_gaussian(tuple(predictor.patch_size), sigma_scale=1 / 8, value_scaling_factor=10, device=torch.device("cpu")).float()
    gaussian_np = gaussian.numpy()
    denominator = _compute_denominator(slicers, gaussian, padded_shape)
    baseline_sum = np.zeros(padded_shape, np.float32)

    temp_dir = output_dir / ".score_buffers" / case.case_id
    temp_dir.mkdir(parents=True, exist_ok=True)
    buffers: dict[tuple[str, str], np.memmap] = {}
    for bank in BANK_NAMES:
        for mode in SCORE_MODES:
            buffer = np.memmap(temp_dir / f"{bank}_{mode}.dat", mode="w+", dtype=np.float32, shape=padded_shape)
            buffer[:] = 0
            buffers[(bank, mode)] = buffer
    bank_tensors = {
        name: torch.from_numpy(np.asarray(values, np.float32)).to(predictor.device)
        for name, values in prototype_banks.items()
    }
    feature_observation = None
    for slicer in slicers:
        patch = padded[slicer][None].to(predictor.device)
        logits, features = _network_patch(predictor, patch, text_embedding)
        raw_feature_shape = tuple(int(v) for v in features.shape[1:])
        tile_shape = tuple(int(s.stop - s.start) for s in slicer[1:])
        feature_observation = (int(features.shape[0]), raw_feature_shape, tile_shape)
        if raw_feature_shape != tile_shape:
            raise AssertionError(
                "D5 pre-head grid unexpectedly differs from its sliding-window grid: "
                f"D5={raw_feature_shape}, tile={tile_shape}. Refusing implicit feature resampling."
            )
        if tuple(logits.shape) != tile_shape:
            logits = F.interpolate(logits[None, None], size=tile_shape, mode="trilinear", align_corners=False)[0, 0]
        baseline_sum[slicer[1:]] += (logits.cpu() * gaussian).numpy()
        for bank, prototypes in bank_tensors.items():
            modes = score_modes(features, prototypes)
            for mode, values in modes.items():
                buffers[(bank, mode)][slicer[1:]] += values.cpu().numpy() * gaussian_np

    model_probability_crop = torch.sigmoid(torch.from_numpy(baseline_sum / denominator)).numpy()[tuple(revert[1:])]
    if case.prediction is not None:
        baseline_probability, probability_kind = load_probability(case.prediction, original_shape, case.roi_bbox)
    else:
        baseline_probability = _restore_crop(model_probability_crop, bbox, original_shape)
        probability_kind = "model_probability"
    roi = np.zeros(original_shape, dtype=bool)
    roi[crop] = True
    gt = gt & roi
    baseline_prediction = baseline_probability >= 0.5
    baseline = binary_metrics(baseline_prediction & roi, gt)
    baseline_empty = int(np.logical_and(baseline_prediction, roi).sum()) == 0

    localization_rows: list[dict[str, Any]] = []
    fusion_rows: list[dict[str, Any]] = []
    baseline_logit = np.log(np.clip(baseline_probability, 1e-6, 1 - 1e-6) / np.clip(1 - baseline_probability, 1e-6, 1))
    fused_primary = None
    oracle_similarity = None
    prediction_similarity = None

    def append_localization(bank: str, mode: str, score: np.ndarray) -> None:
        metrics = localization_metrics(score, gt, roi, baseline_probability)
        requested_k = {"native": 1, "max": 1, "top3_mean": 3, "top5_mean": 5}[mode]
        effective_k = 1 if bank == "baseline_probability" else min(requested_k, len(prototype_banks[bank]))
        localization_rows.append(
            {
                "case_id": case.case_id,
                "zero_shot_dice_csv": case.zero_shot_dice,
                "bank": bank,
                "score_mode": mode,
                "effective_prototype_top_k": effective_k,
                "baseline_dice": baseline["dice"],
                "baseline_precision": baseline["precision"],
                "baseline_recall": baseline["recall"],
                "baseline_pred_foreground_voxels": baseline["pred_foreground_voxels"],
                "baseline_empty_prediction": baseline_empty,
                "probability_source_type": probability_kind,
                **metrics,
            }
        )

    append_localization("baseline_probability", "native", baseline_probability)
    for bank in BANK_NAMES:
        for mode in SCORE_MODES:
            key = (bank, mode)
            buffer = buffers[key]
            fused_crop = (np.asarray(buffer) / denominator)[tuple(revert[1:])]
            proto_score = _restore_crop(fused_crop.astype(np.float32), bbox, original_shape)
            buffer.flush()
            append_localization(bank, mode, proto_score)
            if bank == "oracle" and mode == "max":
                oracle_similarity = proto_score.copy()
            if bank == "prediction" and mode == "max":
                prediction_similarity = proto_score.copy()
            roi_values = proto_score[roi]
            mean = float(roi_values.mean())
            std = float(roi_values.std())
            if std <= EPS or not np.isfinite(std):
                proto_score[:] = 0
                standardization_status = "zero_variance_score"
            else:
                proto_score[roi] = (roi_values - mean) / std
                proto_score[~roi] = 0
                standardization_status = "ok"
            for alpha in ALPHAS:
                fused = (baseline_logit + alpha * proto_score) >= 0.0
                metrics = binary_metrics(fused & roi, gt)
                fusion_rows.append(
                    {
                        "case_id": case.case_id,
                        "bank": bank,
                        "score_mode": mode,
                        "alpha": alpha,
                        "threshold": 0.5,
                        "dice": metrics["dice"],
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "delta_dice": metrics["dice"] - baseline["dice"],
                        "delta_precision": metrics["precision"] - baseline["precision"],
                        "delta_recall": metrics["recall"] - baseline["recall"],
                        "baseline_empty_prediction": baseline_empty,
                        "recovered_empty_case": bool(baseline_empty and metrics["tp"] > 0),
                        "score_mean_in_roi": mean,
                        "score_std_in_roi": std,
                        "standardization_status": standardization_status,
                    }
                )
                if bank == "oracle" and mode == "max" and alpha == 1.0:
                    fused_primary = fused & roi
            del proto_score, fused_crop
            buffer._mmap.close()
            del buffers[key]

    if fused_primary is None or oracle_similarity is None or prediction_similarity is None:
        raise AssertionError("required fixed fusion/similarity outputs were not generated")
    case_dir = output_dir / case.case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    save_float_nifti(baseline_probability, case_dir / "baseline_probability.nii.gz", properties)
    save_float_nifti(oracle_similarity, case_dir / "oracle_similarity.nii.gz", properties)
    save_float_nifti(prediction_similarity, case_dir / "prediction_similarity.nii.gz", properties)
    save_float_nifti(fused_primary.astype(np.uint8), case_dir / "fused_prediction.nii.gz", properties, integer=True)
    save_visualization(
        image, gt, baseline_probability, oracle_similarity, fused_primary,
        case_dir / "diagnostic.png", case.case_id,
    )
    shutil.rmtree(temp_dir)
    return localization_rows, fusion_rows, {
        "case_id": case.case_id,
        "baseline_empty": baseline_empty,
        "feature_observation": feature_observation,
    }


def save_float_nifti(array: np.ndarray, path: Path, properties: dict[str, Any], integer: bool = False) -> None:
    data = np.asarray(array).transpose((2, 1, 0))
    dtype = np.uint8 if integer else np.float32
    image = nib.Nifti1Image(data.astype(dtype), affine=properties["nibabel_stuff"]["reoriented_affine"])
    img_ornt = io_orientation(properties["nibabel_stuff"]["original_affine"])
    from_canonical = ornt_transform(axcodes2ornt("RAS"), img_ornt)
    restored = image.as_reoriented(from_canonical)
    if not np.allclose(properties["nibabel_stuff"]["original_affine"], restored.affine, atol=1e-4):
        raise ValueError(f"affine restoration failed for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(restored, str(path))


def save_visualization(
    image: np.ndarray,
    gt: np.ndarray,
    baseline_probability: np.ndarray,
    oracle_similarity: np.ndarray,
    fused: np.ndarray,
    path: Path,
    case_id: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per_slice = gt.sum(axis=(1, 2))
    z = int(np.argmax(per_slice)) if bool(gt.any()) else int(image.shape[0] // 2)
    fn_response = np.where(gt & (baseline_probability < 0.5), oracle_similarity, np.nan)
    panels = [
        (image[z], "HBP", "gray", None, None),
        (gt[z], "GT", "gray", 0, 1),
        (baseline_probability[z], "baseline probability", "magma", 0, 1),
        ((baseline_probability[z] >= 0.5), "baseline prediction", "gray", 0, 1),
        (oracle_similarity[z], "oracle similarity", "viridis", -1, 1),
        (fn_response[z], "prototype response in baseline FN", "viridis", -1, 1),
        (fused[z], "fixed fusion: oracle/max, alpha=1", "gray", 0, 1),
    ]
    figure, axes = plt.subplots(2, 4, figsize=(18, 9))
    for axis, panel in zip(axes.ravel(), panels):
        values, title, cmap, vmin, vmax = panel
        axis.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(title)
        axis.axis("off")
    axes.ravel()[-1].axis("off")
    figure.suptitle(f"{case_id}, axial D={z}")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _nanmean(rows: Sequence[dict[str, Any]], field: str) -> float:
    values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
    return float(np.nanmean(values)) if np.isfinite(values).any() else math.nan


def summarize(
    localization_rows: Sequence[dict[str, Any]],
    fusion_rows: Sequence[dict[str, Any]],
    audit: dict[str, Any],
    feature_metadata: dict[str, Any],
) -> dict[str, Any]:
    def local(bank: str, mode: str) -> list[dict[str, Any]]:
        return [r for r in localization_rows if r["bank"] == bank and r["score_mode"] == mode]

    baseline = local("baseline_probability", "native")
    oracle = local("oracle", "max")
    prediction = local("prediction", "max")
    random_rows = local("random", "max")
    metric_fields = (
        "auprc", "dice_at_oracle_volume", "recall_at_1v", "recall_at_2v", "recall_at_5v",
        "fn_region_auprc", "fn_recall_at_1v", "fn_recall_at_2v", "fn_recall_at_5v",
    )
    means = {
        name: {field: _nanmean(rows, field) for field in metric_fields}
        for name, rows in (("baseline_probability", baseline), ("oracle", oracle), ("prediction", prediction), ("random", random_rows))
    }
    primary_fusion = [
        row for row in fusion_rows
        if row["bank"] == "oracle" and row["score_mode"] == "max" and float(row["alpha"]) == 1.0
    ]
    alpha_counts = {}
    alpha_means = {}
    for alpha in ALPHAS:
        rows = [r for r in fusion_rows if r["bank"] == "oracle" and r["score_mode"] == "max" and float(r["alpha"]) == alpha]
        deltas = np.asarray([float(r["delta_dice"]) for r in rows])
        alpha_counts[str(alpha)] = {
            "improved": int((deltas > 1e-8).sum()),
            "declined": int((deltas < -1e-8).sum()),
            "unchanged": int((np.abs(deltas) <= 1e-8).sum()),
            "empty_predictions_recovered": int(sum(bool(r["recovered_empty_case"]) for r in rows)),
        }
        alpha_means[str(alpha)] = {
            "mean_delta_dice": _nanmean(rows, "delta_dice"),
            "mean_delta_precision": _nanmean(rows, "delta_precision"),
            "mean_delta_recall": _nanmean(rows, "delta_recall"),
        }
    oracle_better_baseline = (
        means["oracle"]["auprc"] > means["baseline_probability"]["auprc"]
        and means["oracle"]["dice_at_oracle_volume"] > means["baseline_probability"]["dice_at_oracle_volume"]
    )
    oracle_better_random = (
        means["oracle"]["auprc"] > means["random"]["auprc"]
        and means["oracle"]["dice_at_oracle_volume"] > means["random"]["dice_at_oracle_volume"]
    )
    fn_better_random = means["oracle"]["fn_region_auprc"] > means["random"]["fn_region_auprc"]
    precision_delta = _nanmean(primary_fusion, "delta_precision")
    recall_delta = _nanmean(primary_fusion, "delta_recall")
    probability_source_types = sorted({str(row["probability_source_type"]) for row in baseline})
    warnings = []
    if "binary_mask" in probability_source_types:
        warnings.append(
            "At least one --prediction-dir input is binary rather than probabilistic. "
            "Localization metrics remain valid, but baseline logits are saturated, so fixed-alpha fusion is conservative."
        )
    return {
        "experiment": "oracle_support_low_quality_localization",
        "seed": SEED,
        "audit": audit,
        "feature_metadata": feature_metadata,
        "metric_emphasis": ["AUPRC", "Dice@OracleVolume", "FN-region recovery"],
        "probability_source_types": probability_source_types,
        "warnings": warnings,
        "definitions": {
            "random_control": "independent fixed-seed channel permutation of each normalized oracle prototype",
            "lesion_hit": "a GT 26-connected component overlaps the score's top-1V mask",
            "top_score_foreground_enrichment": "precision in top-1V divided by GT prevalence in the valid ROI",
            "fn_volume": "number of GT-positive voxels where baseline probability < 0.5",
            "severe_precision_drop": "mean absolute Precision drop greater than 0.10 at fixed oracle/max alpha=1.0",
            "nan_auc": "AUROC/AUPRC are NaN when the evaluated region contains only one class",
        },
        "primary_max_score_means": means,
        "answers": {
            "oracle_better_than_baseline_probability": oracle_better_baseline,
            "oracle_better_than_random_prototype": oracle_better_random,
            "oracle_clearly_better_than_random_in_fn_region": fn_better_random,
            "fusion_alpha_1_case_changes": alpha_counts["1.0"],
            "empty_prediction_cases": int(sum(bool(r["baseline_empty_prediction"]) for r in baseline)),
            "empty_prediction_cases_recovered_alpha_1": alpha_counts["1.0"]["empty_predictions_recovered"],
            "mean_recall_delta_alpha_1": recall_delta,
            "mean_precision_delta_alpha_1": precision_delta,
            "recall_gain_with_severe_precision_drop": bool(recall_delta > 0 and precision_delta < -0.10),
            "gt_vs_prediction_prototype_gap": {
                field: means["oracle"][field] - means["prediction"][field]
                for field in metric_fields
            },
        },
        "oracle_max_fusion_counts_by_fixed_alpha": alpha_counts,
        "oracle_max_fusion_mean_deltas_by_fixed_alpha": alpha_means,
    }


def print_summary(summary: dict[str, Any]) -> None:
    answers = summary["answers"]
    print("\n=== Oracle support diagnostic summary ===")
    print(f"Oracle > baseline probability: {answers['oracle_better_than_baseline_probability']}")
    print(f"Oracle > random prototype: {answers['oracle_better_than_random_prototype']}")
    print(f"Oracle > random in baseline-FN region: {answers['oracle_clearly_better_than_random_in_fn_region']}")
    print("Primary max-score means (AUPRC / Dice@OracleVolume / FN-AUPRC):")
    for bank, metrics in summary["primary_max_score_means"].items():
        print(
            f"  {bank}: {metrics['auprc']:.6f} / "
            f"{metrics['dice_at_oracle_volume']:.6f} / {metrics['fn_region_auprc']:.6f}"
        )
    print(f"Fixed alpha=1 case changes: {answers['fusion_alpha_1_case_changes']}")
    print(
        "Empty predictions recovered: "
        f"{answers['empty_prediction_cases_recovered_alpha_1']}/{answers['empty_prediction_cases']}"
    )
    print(
        "Fixed alpha=1 mean delta: "
        f"Recall={answers['mean_recall_delta_alpha_1']:.6f}, "
        f"Precision={answers['mean_precision_delta_alpha_1']:.6f}; "
        f"severe precision drop={answers['recall_gain_with_severe_precision_drop']}"
    )
    print(f"GT - prediction-mask prototype gap: {answers['gt_vs_prediction_prototype_gap']}")
    for warning in summary.get("warnings", []):
        print(f"WARNING: {warning}")


def dry_run_checks() -> None:
    mask = np.zeros((4, 4, 4), bool)
    mask[1:3, 1:3, 1:3] = True
    assert align_mask_nearest(mask, (8, 8, 8)).sum() == 64
    prototype = normalized_prototype(np.asarray([[3, 0], [1, 0]], np.float32))
    assert np.allclose(np.linalg.norm(prototype), 1)
    scores = np.arange(8).reshape(2, 2, 2)
    assert exact_topk_mask(scores, np.ones_like(scores, bool), 3).sum() == 3
    fused = gaussian_fuse_patches(
        [np.ones((2, 2, 2)), np.full((2, 2, 2), 3)],
        [(slice(0, 2), slice(0, 2), slice(0, 2)), (slice(1, 3), slice(0, 2), slice(0, 2))],
        np.ones((2, 2, 2), np.float32),
        (3, 2, 2),
    )
    assert np.allclose(fused[1], 2)
    empty = binary_metrics(np.zeros((2, 2, 2), bool), np.zeros((2, 2, 2), bool))
    assert empty["dice"] == 1 and empty["pred_foreground_voxels"] == 0
    print("[dry-run] numerical invariants passed: alignment, Gaussian fusion, prototype norm, top-K, empty mask")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-json", required=True)
    parser.add_argument("--dice-csv", required=True)
    parser.add_argument("--image-dir", required=True, help="Directory of precomputed complete liver-ROI HBP NIfTI volumes")
    parser.add_argument("--label-dir", required=True)
    parser.add_argument("--prediction-dir", default=None)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--text-model", required=True)
    parser.add_argument("--prompt", default="HCC lesion")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--support-threshold", type=float, default=0.8)
    parser.add_argument("--query-threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true", help="Audit schemas/files and run numerical invariants without loading models")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    for path in (args.split_json, args.dice_csv, args.image_dir, args.label_dir, args.model_dir, args.text_model):
        if not Path(path).exists():
            raise FileNotFoundError(path)
    if args.prediction_dir and not Path(args.prediction_dir).exists():
        raise FileNotFoundError(args.prediction_dir)
    if not (0 <= args.query_threshold < args.support_threshold <= 1):
        raise ValueError("require 0 <= query-threshold < support-threshold <= 1")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    support_cases, query_cases, audit = audit_inputs(args)
    if args.dry_run:
        # Load one case from each group to validate geometry, not only filenames.
        for case in (support_cases[0], query_cases[0]):
            image, label, _ = load_case(case)
            if case.prediction is not None:
                load_probability(case.prediction, image.shape, case.roi_bbox)
            print(f"[dry-run] {case.case_id}: image/label geometry OK, shape={image.shape}, GT={int(label.sum())}")
        dry_run_checks()
        print("[dry-run] completed before model loading; no experiment outputs were produced")
        return 0

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested {device}, but CUDA is unavailable")
    from voxtell.inference.predictor import VoxTellPredictor

    predictor = VoxTellPredictor(model_dir=args.model_dir, device=device, text_encoding_model=args.text_model)
    predictor.network = predictor.network.to(device).eval()
    text_embedding = predictor.embed_text_prompts(args.prompt)
    support_rows: list[dict[str, Any]] = []
    oracle_prototypes: list[np.ndarray] = []
    prediction_prototypes: list[np.ndarray] = []
    feature_metadata: dict[str, Any] = {}
    support_feature_observations: list[dict[str, Any]] = []
    for index, case in enumerate(support_cases, 1):
        print(f"[support {index}/{len(support_cases)}] {case.case_id}")
        oracle, prediction, rows, metadata = extract_support_case(predictor, text_embedding, case)
        oracle_prototypes.extend(oracle)
        prediction_prototypes.extend(prediction)
        support_rows.extend(rows)
        if feature_metadata and metadata["feature_channels"] != feature_metadata["feature_channels"]:
            raise AssertionError("D5 feature channel count changed across cases")
        if feature_metadata and metadata["raw_d5_patch_shape"] != feature_metadata["raw_d5_patch_shape"]:
            raise AssertionError("D5 spatial shape changed across support cases")
        if not feature_metadata:
            feature_metadata = {
                "feature_channels": metadata["feature_channels"],
                "raw_d5_patch_shape": metadata["raw_d5_patch_shape"],
                "aligned_patch_shape": metadata["aligned_patch_shape"],
                "correspondence": (
                    "D5 pre-head grid equals the 192^3 sliding-window grid; GT is nearest-neighbor "
                    "aligned to that grid, then Gaussian-fused coordinates map through crop_to_nonzero "
                    "bbox into the supplied liver ROI."
                ),
            }
        support_feature_observations.append({"case_id": case.case_id, **metadata})
    if not oracle_prototypes:
        raise RuntimeError("oracle prototype bank is empty")
    if not prediction_prototypes:
        raise RuntimeError("prediction-mask prototype bank is empty")
    oracle_bank = np.stack(oracle_prototypes).astype(np.float32)
    prediction_bank = np.stack(prediction_prototypes).astype(np.float32)
    rng = np.random.default_rng(SEED)
    random_bank = np.stack([prototype[rng.permutation(prototype.size)] for prototype in oracle_bank]).astype(np.float32)
    random_bank /= np.maximum(np.linalg.norm(random_bank, axis=1, keepdims=True), EPS)
    assert np.allclose(np.linalg.norm(oracle_bank, axis=1), 1, atol=1e-5)
    assert np.allclose(np.linalg.norm(prediction_bank, axis=1), 1, atol=1e-5)
    counters = {"oracle": 0, "prediction": 0}
    for row in support_rows:
        bank = row["bank"]
        if row.get("status") == "ok":
            row["prototype_index"] = counters[bank]
            counters[bank] += 1
        else:
            row["prototype_index"] = ""
    feature_metadata["support_cases"] = support_feature_observations
    feature_metadata["oracle_prototype_count"] = len(oracle_bank)
    feature_metadata["prediction_prototype_count"] = len(prediction_bank)
    _write_csv(output_dir / "support_bank.csv", support_rows)
    np.savez_compressed(
        output_dir / "oracle_prototypes.npz",
        oracle_prototypes=oracle_bank,
        prediction_prototypes=prediction_bank,
        random_prototypes=random_bank,
        random_seed=np.asarray(SEED),
        metadata_json=np.asarray(json.dumps(feature_metadata)),
    )

    localization_rows: list[dict[str, Any]] = []
    fusion_rows: list[dict[str, Any]] = []
    observations = []
    banks = {"oracle": oracle_bank, "prediction": prediction_bank, "random": random_bank}
    for index, case in enumerate(query_cases, 1):
        print(f"[query {index}/{len(query_cases)}] {case.case_id}")
        local, fusion, observation = run_query_case(predictor, text_embedding, case, banks, output_dir)
        localization_rows.extend(local)
        fusion_rows.extend(fusion)
        observations.append(observation)
        _write_csv(output_dir / "per_case_localization.csv", localization_rows)
        _write_csv(output_dir / "per_case_fusion.csv", fusion_rows)
    feature_metadata["query_observations"] = observations
    summary = summarize(localization_rows, fusion_rows, audit, feature_metadata)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=True)
    print_summary(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
