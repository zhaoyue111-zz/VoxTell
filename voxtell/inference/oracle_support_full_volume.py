"""Full-volume Oracle Support diagnostic for VoxTell.

The experiment is analysis-only. It follows the official VoxTell path:
original RAS-reoriented volume -> crop_to_nonzero -> ZScore -> sliding windows
-> nnU-Net Gaussian fusion -> restoration into the original full volume.
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
from typing import Any, Iterator, Sequence

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform
from scipy import ndimage
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score


SCORE_MODES = ("max", "top3_mean", "top5_mean")
BANK_NAMES = ("oracle", "prediction")
ALPHAS = (0.25, 0.5, 1.0, 2.0)
RANDOM_SEEDS = tuple(range(42, 52))
BASELINE_DICE_TOLERANCE = 1e-4
EPS = 1e-8


@dataclass(frozen=True)
class CaseFiles:
    case_id: str
    image: Path
    label: Path
    prediction: Path | None
    zero_shot_dice: float


def normalize_case_id(value: Any) -> str:
    text = str(value).strip().replace("\\", "/").rsplit("/", 1)[-1]
    lower = text.lower()
    for suffix in (".nii.gz", ".nii", ".npy", ".npz"):
        if lower.endswith(suffix):
            return text[: -len(suffix)]
    return text


def _preview(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _preview(v) for k, v in list(value.items())[:3]}
    if isinstance(value, list):
        return [_preview(v) for v in value[:2]]
    return value


def inspect_split_json(path: Path, available_ids: set[str]) -> list[str]:
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    print(f"[schema] split JSON type={type(payload).__name__}")
    if isinstance(payload, dict):
        print(f"[schema] split JSON top-level keys={list(payload)}")
        keys = [
            k for k in payload
            if str(k).strip().lower() in {"train", "training", "train_cases"}
        ]
        if len(keys) != 1:
            raise ValueError(f"expected exactly one train/training/train_cases key, found {keys}")
        entries = payload[keys[0]]
    elif isinstance(payload, list):
        entries = payload
    else:
        raise ValueError("split JSON must be an object or list")
    print(f"[schema] split JSON sample={_preview(entries)}")
    if not isinstance(entries, list) or not entries:
        raise ValueError("training split must be a non-empty list")
    if all(isinstance(item, (str, int)) for item in entries):
        ids = [normalize_case_id(item) for item in entries]
    elif all(isinstance(item, dict) for item in entries):
        common = set(entries[0])
        for item in entries[1:]:
            common &= set(item)
        candidates = []
        for key in sorted(common):
            values = [item[key] for item in entries]
            if all(isinstance(v, (str, int)) for v in values):
                candidates.append((sum(normalize_case_id(v) in available_ids for v in values), key))
        best = max((score for score, _ in candidates), default=0)
        winners = [key for score, key in candidates if score == best and score > 0]
        print(f"[schema] split ID-field overlap candidates={candidates}")
        if len(winners) != 1:
            raise ValueError(f"ambiguous split case-ID field: {winners}")
        ids = [normalize_case_id(item[winners[0]]) for item in entries]
    else:
        raise ValueError("split training list mixes entry types")
    duplicates = sorted({case_id for case_id in ids if ids.count(case_id) > 1})
    if duplicates:
        raise ValueError(f"duplicate training IDs: {duplicates}")
    return ids


def inspect_dice_csv(path: Path, train_ids: set[str], available_ids: set[str], prompt: str) -> dict[str, float]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    print(f"[schema] Dice CSV columns={fields}")
    print(f"[schema] Dice CSV first rows={rows[:2]}")
    if not fields or not rows:
        raise ValueError("Dice CSV is empty")
    prompt_fields = [field for field in fields if field.strip().lower() == "prompt"]
    if len(prompt_fields) > 1:
        raise ValueError(f"ambiguous CSV prompt columns: {prompt_fields}")
    if prompt_fields:
        prompt_field = prompt_fields[0]
        available_prompts = sorted({str(row[prompt_field]).strip() for row in rows})
        print(f"[schema] Dice CSV prompts={available_prompts}")
        rows = [row for row in rows if str(row[prompt_field]).strip() == prompt]
        if not rows:
            raise ValueError(f"Dice CSV has no rows for prompt {prompt!r}; available={available_prompts}")
    universe = train_ids | available_ids
    overlaps = [(sum(normalize_case_id(row.get(field, "")) in universe for row in rows), field) for field in fields]
    best = max(score for score, _ in overlaps)
    id_fields = [field for score, field in overlaps if score == best and score > 0]
    if len(id_fields) != 1:
        raise ValueError(f"ambiguous CSV case-ID column: {id_fields}")
    id_field = id_fields[0]
    dice_fields = [field for field in fields if "dice" in field.lower()]
    valid_dice_fields = []
    for field in dice_fields:
        try:
            values = [float(row[field]) for row in rows if normalize_case_id(row[id_field]) in train_ids]
        except (TypeError, ValueError):
            continue
        if values and all(np.isfinite(values)):
            valid_dice_fields.append(field)
    if len(valid_dice_fields) != 1:
        raise ValueError(f"ambiguous finite Dice column: {valid_dice_fields}")
    dice_field = valid_dice_fields[0]
    print(f"[schema] resolved CSV case={id_field!r}, dice={dice_field!r}")
    result: dict[str, float] = {}
    for row in rows:
        case_id = normalize_case_id(row[id_field])
        if case_id not in train_ids:
            continue
        if case_id in result:
            raise ValueError(f"duplicate Dice row: {case_id}")
        value = float(row[dice_field])
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"invalid Dice for {case_id}: {value}")
        result[case_id] = value
    missing = sorted(train_ids - set(result))
    if missing:
        raise FileNotFoundError(f"training IDs missing from Dice CSV: {missing}")
    return result


def index_case_files(directory: Path, extensions: Sequence[str]) -> dict[str, Path]:
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    result: dict[str, Path] = {}
    collisions: dict[str, list[str]] = {}
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        if not any(path.name.lower().endswith(ext) for ext in extensions):
            continue
        case_id = normalize_case_id(path.name)
        if case_id in result:
            collisions.setdefault(case_id, [str(result[case_id])]).append(str(path))
        result[case_id] = path
    if collisions:
        raise ValueError(f"ambiguous files for case IDs: {collisions}")
    return result


def audit_inputs(args: argparse.Namespace) -> tuple[list[CaseFiles], list[CaseFiles], dict[str, Any]]:
    images = index_case_files(Path(args.image_dir), (".nii.gz", ".nii"))
    labels = index_case_files(Path(args.label_dir), (".nii.gz", ".nii"))
    predictions = index_case_files(Path(args.prediction_dir), (".nii.gz", ".nii", ".npy", ".npz")) if args.prediction_dir else {}
    train_ids = inspect_split_json(Path(args.split_json), set(images) | set(labels) | set(predictions))
    dice = inspect_dice_csv(
        Path(args.dice_csv), set(train_ids), set(images) | set(labels) | set(predictions), args.prompt
    )
    support_ids = [case_id for case_id in train_ids if dice[case_id] > args.support_threshold]
    query_ids = [case_id for case_id in train_ids if dice[case_id] < args.query_threshold]
    assert_support_query_disjoint(support_ids, query_ids)
    print(f"[audit] train={len(train_ids)} support={len(support_ids)} query={len(query_ids)}")
    if not support_ids or not query_ids:
        raise ValueError("support or query set is empty")
    checked = set(train_ids)
    missing_images = sorted(checked - set(images))
    missing_labels = sorted(checked - set(labels))
    missing_predictions = sorted(checked - set(predictions)) if args.prediction_dir else []
    print(f"[audit] missing images={missing_images}")
    print(f"[audit] missing labels={missing_labels}")
    print(f"[audit] missing predictions={missing_predictions}")
    if missing_images or missing_labels or missing_predictions:
        raise FileNotFoundError("missing training files; see audit")

    def make(case_id: str) -> CaseFiles:
        return CaseFiles(case_id, images[case_id], labels[case_id], predictions.get(case_id), dice[case_id])

    return [make(x) for x in support_ids], [make(x) for x in query_ids], {
        "train_cases": len(train_ids), "support_cases": len(support_ids), "query_cases": len(query_ids),
        "support_threshold": args.support_threshold, "query_threshold": args.query_threshold,
        "image_space": "original HBP full volume", "image_directory": str(Path(args.image_dir).resolve()),
        "prediction_directory": str(Path(args.prediction_dir).resolve()) if args.prediction_dir else None,
    }


def assert_support_query_disjoint(support_ids: Sequence[str], query_ids: Sequence[str]) -> None:
    overlap = sorted(set(support_ids) & set(query_ids))
    if overlap:
        raise AssertionError(f"support/query overlap: {overlap}")


def _bbox_slices(bbox: Sequence[Sequence[int]]) -> tuple[slice, ...]:
    return tuple(slice(int(lo), int(hi)) for lo, hi in bbox)


def restore_crop_to_full(array: np.ndarray, bbox: Sequence[Sequence[int]], original_shape: Sequence[int]) -> np.ndarray:
    result = np.zeros(tuple(int(v) for v in original_shape), dtype=np.asarray(array).dtype)
    result[_bbox_slices(bbox)] = array
    return result


def valid_region_from_bbox(original_shape: Sequence[int], bbox: Sequence[Sequence[int]]) -> np.ndarray:
    result = np.zeros(tuple(int(v) for v in original_shape), dtype=bool)
    result[_bbox_slices(bbox)] = True
    return result


def align_mask_nearest(mask: np.ndarray, target_shape: Sequence[int]) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(mask, np.float32))[None, None]
    return F.interpolate(tensor, size=tuple(target_shape), mode="nearest")[0, 0].numpy() > 0.5


def connected_lesions(mask: np.ndarray) -> tuple[np.ndarray, int]:
    return ndimage.label(np.asarray(mask, bool), structure=np.ones((3, 3, 3), bool))


def eroded_component_labels(mask: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]]]:
    labels, count = connected_lesions(mask)
    output = np.zeros_like(labels, dtype=np.int32)
    records = []
    structure = np.ones((3, 3, 3), bool)
    for lesion_id in range(1, count + 1):
        lesion = labels == lesion_id
        candidate = ndimage.binary_erosion(lesion, structure=structure, iterations=1)
        fallback = not bool(candidate.any())
        selected = lesion if fallback else candidate
        output[selected] = lesion_id
        records.append({"lesion_id": lesion_id, "original_voxels": int(lesion.sum()), "prototype_voxels": int(selected.sum()), "erosion_fallback": fallback})
    return output, records


def normalized_prototype(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, np.float32)
    if values.ndim != 2 or not len(values):
        raise ValueError("prototype features must be non-empty [N,C]")
    values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), EPS)
    result = values.mean(axis=0)
    result /= max(float(np.linalg.norm(result)), EPS)
    return result.astype(np.float32)


def gaussian_fuse_features_before_similarity(
    feature_patches: Sequence[np.ndarray],
    slicers: Sequence[tuple[slice, slice, slice]],
    gaussian: np.ndarray,
    output_shape: Sequence[int],
    prototype: np.ndarray,
) -> np.ndarray:
    """Small reference implementation used to assert the required ordering."""
    channels = int(feature_patches[0].shape[0])
    feature_sum = np.zeros((channels, *output_shape), np.float32)
    denominator = np.zeros(tuple(output_shape), np.float32)
    for patch, slicer in zip(feature_patches, slicers):
        feature_sum[(slice(None), *slicer)] += np.asarray(patch, np.float32) * gaussian[None]
        denominator[slicer] += gaussian
    if np.any(denominator <= 0):
        raise AssertionError("uncovered reference-fusion voxel")
    fused = feature_sum / denominator[None]
    fused /= np.maximum(np.linalg.norm(fused, axis=0, keepdims=True), EPS)
    proto = np.asarray(prototype, np.float32)
    proto /= max(float(np.linalg.norm(proto)), EPS)
    return np.einsum("cdhw,c->dhw", fused, proto)


def binary_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    pred, gt = np.asarray(prediction, bool), np.asarray(target, bool)
    tp = int((pred & gt).sum()); fp = int((pred & ~gt).sum()); fn = int((~pred & gt).sum())
    den = 2 * tp + fp + fn
    return {
        "dice": float(2 * tp / den) if den else 1.0,
        "precision": float(tp / (tp + fp)) if tp + fp else (1.0 if not gt.any() else 0.0),
        "recall": float(tp / (tp + fn)) if tp + fn else 1.0,
        "pred_foreground_voxels": int(pred.sum()), "gt_foreground_voxels": int(gt.sum()),
        "tp": tp, "fp": fp, "fn": fn,
    }


def verify_baseline_dice(case_id: str, csv_dice: float, recomputed_dice: float, tolerance: float = BASELINE_DICE_TOLERANCE) -> float:
    gap = abs(float(csv_dice) - float(recomputed_dice))
    if gap > tolerance:
        raise RuntimeError(
            f"{case_id}: baseline Dice mismatch: CSV={csv_dice:.10f}, recomputed={recomputed_dice:.10f}, "
            f"gap={gap:.10f} > {tolerance}; check prompt, preprocessing, prediction source, and orientation"
        )
    return gap


def _load_nifti(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
    data, properties = NibabelIOWithReorient().read_images([str(path)])
    return np.asarray(data[0]), properties


def load_case(case: CaseFiles) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    image, properties = _load_nifti(case.image)
    label, label_properties = _load_nifti(case.label)
    if image.shape != label.shape:
        raise ValueError(f"{case.case_id}: image/label shape mismatch")
    if not np.allclose(properties["nibabel_stuff"]["original_affine"], label_properties["nibabel_stuff"]["original_affine"], atol=1e-4):
        raise ValueError(f"{case.case_id}: image/label affine mismatch")
    rounded = np.rint(label)
    if not np.all(np.isfinite(image)) or not np.allclose(label, rounded, atol=1e-5):
        raise ValueError(f"{case.case_id}: non-finite image or non-integer GT")
    positive = np.unique(rounded[rounded > 0])
    if positive.size > 1:
        raise ValueError(f"{case.case_id}: ambiguous tumor labels {positive.tolist()}")
    return image.astype(np.float32, copy=False), rounded > 0, properties


def load_probability(
    path: Path,
    expected_shape: Sequence[int],
    expected_affine: np.ndarray | None = None,
) -> tuple[np.ndarray, str]:
    lower = path.name.lower()
    if lower.endswith((".nii", ".nii.gz")):
        array, properties = _load_nifti(path)
        if expected_affine is not None and not np.allclose(
            properties["nibabel_stuff"]["original_affine"], expected_affine, atol=1e-4
        ):
            raise ValueError(f"{path}: prediction affine does not match the original image")
    elif lower.endswith(".npy"):
        array = np.load(path, allow_pickle=False)
    elif lower.endswith(".npz"):
        with np.load(path, allow_pickle=False) as archive:
            if len(archive.files) != 1:
                raise ValueError(f"{path}: ambiguous NPZ arrays {archive.files}")
            array = archive[archive.files[0]]
    else:
        raise ValueError(path)
    array = np.squeeze(np.asarray(array, np.float32))
    if tuple(array.shape) != tuple(expected_shape):
        raise ValueError(f"{path}: prediction {array.shape} != full volume {tuple(expected_shape)}")
    if not np.all(np.isfinite(array)) or array.min() < 0 or array.max() > 1:
        raise ValueError(f"{path}: prediction must be finite in [0,1]")
    kind = "binary_mask" if np.all(np.isin(np.unique(array), (0, 1))) else "probability"
    return array, kind


def _network_patch(predictor: Any, patch: torch.Tensor, text_embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.inference_mode(), torch.autocast(device_type=predictor.device.type, enabled=predictor.device.type == "cuda"):
        prediction, returned = predictor.network(patch, text_embedding, return_features=True)
    if isinstance(prediction, list):
        prediction = prediction[0]
    feature = returned["d5_pre_head"]
    if feature.ndim != 6 or feature.shape[1] != 1:
        raise AssertionError(f"unexpected D5 feature {tuple(feature.shape)}")
    return prediction[0, 0].float(), feature[0, 0].float()


def _close_memmap(array: np.memmap | None) -> None:
    mapping = getattr(array, "_mmap", None) if array is not None else None
    if mapping is not None and not mapping.closed:
        array.flush(); array._mmap.close()


@dataclass
class FusedFeatureCase:
    case: CaseFiles
    image: np.ndarray
    gt: np.ndarray
    properties: dict[str, Any]
    bbox: tuple[tuple[int, int], ...]
    original_shape: tuple[int, ...]
    cropped_shape: tuple[int, ...]
    padded_shape: tuple[int, ...]
    revert_padding: tuple[slice, ...]
    feature_sum: np.memmap
    denominator: np.memmap
    baseline_probability: np.ndarray
    baseline_metrics: dict[str, Any]
    baseline_audit: dict[str, Any]
    feature_channels: int
    raw_patch_shape: tuple[int, ...]
    temp_dir: Path
    metadata: dict[str, Any]

    @property
    def padded_valid_slices(self) -> tuple[slice, ...]:
        return tuple(self.revert_padding[1:])

    def release_fusion_buffers(self) -> None:
        for array in (self.feature_sum, self.denominator):
            path = Path(array.filename)
            _close_memmap(array)
            if path.exists(): path.unlink()

    def cleanup(self) -> None:
        _close_memmap(self.feature_sum); _close_memmap(self.denominator)
        if self.temp_dir.exists(): shutil.rmtree(self.temp_dir)


def run_full_volume_feature_inference(predictor: Any, text_embedding: torch.Tensor, case: CaseFiles, output_dir: Path, prompt: str) -> FusedFeatureCase:
    from acvl_utils.cropping_and_padding.padding import pad_nd_image
    from nnunetv2.inference.sliding_window_prediction import compute_gaussian

    image, gt, properties = load_case(case)
    preprocessed, bbox, original_shape = predictor.preprocess(image)
    padded, revert = pad_nd_image(preprocessed, predictor.patch_size, "constant", {"value": 0}, True, None)
    padded_shape = tuple(int(v) for v in padded.shape[1:])
    slicers = predictor._internal_get_sliding_window_slicers(padded_shape)
    gaussian = compute_gaussian(tuple(predictor.patch_size), sigma_scale=1 / 8, value_scaling_factor=10, device=torch.device("cpu")).float().numpy()
    temp_dir = output_dir / ".oracle_support_work" / case.case_id
    if temp_dir.exists(): shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)
    denominator = np.memmap(temp_dir / "denominator.dat", "w+", np.float32, shape=padded_shape); denominator[:] = 0
    logits_sum = np.memmap(temp_dir / "logits.dat", "w+", np.float32, shape=padded_shape); logits_sum[:] = 0
    feature_sum = None; channels = 0; raw_patch_shape = None
    for slicer in slicers:
        logits, feature = _network_patch(predictor, padded[slicer][None].to(predictor.device), text_embedding)
        tile_shape = tuple(int(s.stop - s.start) for s in slicer[1:])
        observed = tuple(int(v) for v in feature.shape[1:])
        if observed != tile_shape or tuple(logits.shape) != tile_shape:
            raise AssertionError(f"D5/logit grid mismatch: feature={observed}, logits={tuple(logits.shape)}, tile={tile_shape}")
        if feature_sum is None:
            channels = int(feature.shape[0]); raw_patch_shape = observed
            feature_sum = np.memmap(temp_dir / "feature_sum.dat", "w+", np.float32, shape=(channels, *padded_shape)); feature_sum[:] = 0
        if int(feature.shape[0]) != channels or observed != raw_patch_shape:
            raise AssertionError("D5 shape changed across windows")
        feature_sum[(slice(None), *slicer[1:])] += feature.cpu().numpy() * gaussian[None]
        logits_sum[slicer[1:]] += logits.cpu().numpy() * gaussian
        denominator[slicer[1:]] += gaussian
    if feature_sum is None or np.any(denominator <= 0):
        raise AssertionError("empty inference or uncovered Gaussian voxel")
    cropped_probability = 1 / (1 + np.exp(-np.clip((logits_sum / denominator)[tuple(revert[1:])], -50, 50)))
    baseline_probability = restore_crop_to_full(cropped_probability.astype(np.float32), bbox, original_shape)
    baseline_metrics = binary_metrics(baseline_probability >= 0.5, gt)
    gap = verify_baseline_dice(case.case_id, case.zero_shot_dice, baseline_metrics["dice"])
    external_kind = "not_supplied"; external_gap = math.nan
    if case.prediction:
        external_probability, external_kind = load_probability(
            case.prediction, original_shape, properties["nibabel_stuff"]["original_affine"]
        )
        external_gap = verify_baseline_dice(case.case_id + ":external", case.zero_shot_dice, binary_metrics(external_probability >= 0.5, gt)["dice"])
    logits_path = Path(logits_sum.filename); _close_memmap(logits_sum); logits_path.unlink()
    bbox_tuple = tuple((int(a), int(b)) for a, b in bbox)
    cropped_shape = tuple(int(v) for v in preprocessed.shape[1:])
    audit = {"case_id": case.case_id, "baseline_dice_csv": float(case.zero_shot_dice), "baseline_dice_recomputed": float(baseline_metrics["dice"]), "baseline_dice_gap": float(gap)}
    metadata = {
        **audit, "prompt": prompt, "image_path": str(case.image), "prediction_source": "recomputed_voxtell_probability",
        "external_prediction_path": str(case.prediction) if case.prediction else None, "external_prediction_type": external_kind,
        "external_prediction_dice_gap": external_gap, "original_shape": list(map(int, original_shape)),
        "crop_to_nonzero_bbox": [list(x) for x in bbox_tuple], "cropped_shape": list(cropped_shape), "padded_shape": list(padded_shape),
        "patch_size": list(map(int, predictor.patch_size)), "tile_step_size": float(predictor.tile_step_size),
        "gaussian_sigma_scale": 0.125, "gaussian_value_scaling_factor": 10, "sliding_window_count": len(slicers),
        "feature_channels": channels, "raw_d5_patch_shape": list(raw_patch_shape),
    }
    return FusedFeatureCase(case, image, gt, properties, bbox_tuple, tuple(original_shape), cropped_shape, padded_shape, tuple(revert), feature_sum, denominator, baseline_probability, baseline_metrics, audit, channels, raw_patch_shape, temp_dir, metadata)


def iter_normalized_fused_feature_blocks(case: FusedFeatureCase, block_depth: int = 1) -> Iterator[tuple[int, int, np.ndarray]]:
    pd, ph, pw = case.padded_valid_slices
    for d0 in range(0, case.cropped_shape[0], block_depth):
        d1 = min(d0 + block_depth, case.cropped_shape[0])
        spatial = (slice(int(pd.start) + d0, int(pd.start) + d1), ph, pw)
        features = np.asarray(case.feature_sum[(slice(None), *spatial)], np.float32) / np.asarray(case.denominator[spatial], np.float32)[None]
        features /= np.maximum(np.linalg.norm(features, axis=0, keepdims=True), EPS)
        yield d0, d1, features


def prototypes_from_component_labels(case: FusedFeatureCase, labels: np.ndarray) -> list[np.ndarray]:
    count = int(labels.max())
    if count == 0: return []
    sums = np.zeros((count + 1, case.feature_channels), np.float64); counts = np.zeros(count + 1, np.int64)
    for d0, d1, features in iter_normalized_fused_feature_blocks(case):
        block_labels = labels[d0:d1].ravel(); selected = block_labels > 0
        if not selected.any(): continue
        vectors = features.reshape(case.feature_channels, -1).T[selected]
        ids = block_labels[selected]; np.add.at(sums, ids, vectors); counts += np.bincount(ids, minlength=count + 1)
    result = []
    for lesion_id in range(1, count + 1):
        if counts[lesion_id] == 0: raise AssertionError("lesion lost during feature alignment")
        vector = sums[lesion_id] / counts[lesion_id]; vector /= max(float(np.linalg.norm(vector)), EPS)
        result.append(vector.astype(np.float32))
    return result


def extract_support_case(case: FusedFeatureCase) -> tuple[list[np.ndarray], list[np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    crop = _bbox_slices(case.bbox)
    gt_crop = align_mask_nearest(case.gt[crop], case.cropped_shape)
    oracle_labels, erosion_records = eroded_component_labels(gt_crop)
    prediction_labels, prediction_count = connected_lesions(case.baseline_probability[crop] >= 0.5)
    banks = {"oracle": prototypes_from_component_labels(case, oracle_labels), "prediction": prototypes_from_component_labels(case, prediction_labels)}
    rows = []
    for bank, labels, records in (("oracle", oracle_labels, erosion_records), ("prediction", prediction_labels, None)):
        if not banks[bank]: rows.append({"bank": bank, "case_id": case.case.case_id, "status": "empty_source_mask"}); continue
        for index, prototype in enumerate(banks[bank], 1):
            record = records[index - 1] if records else {}
            rows.append({
                "bank": bank, "case_id": case.case.case_id, "lesion_id": index, "zero_shot_dice": case.case.zero_shot_dice,
                "source_voxels": int((labels == index).sum()), "original_gt_voxels": record.get("original_voxels", ""),
                "erosion_fallback": record.get("erosion_fallback", ""), "feature_channels": case.feature_channels,
                "raw_d5_patch_shape": "x".join(map(str, case.raw_patch_shape)), "fused_feature_shape": "x".join(map(str, case.cropped_shape)),
                "feature_mapping": "Gaussian-fused crop_to_nonzero grid restored by full-volume bbox", "prototype_l2_norm": float(np.linalg.norm(prototype)),
                "prediction_source": "recomputed_voxtell_probability", "status": "ok",
            })
    return banks["oracle"], banks["prediction"], rows, {**case.metadata, "gt_lesions": len(erosion_records), "gt_erosion_fallbacks": sum(bool(x["erosion_fallback"]) for x in erosion_records), "prediction_lesions": prediction_count}


def materialize_normalized_features(case: FusedFeatureCase) -> np.memmap:
    normalized = np.memmap(case.temp_dir / "normalized_features.dat", "w+", np.float16, shape=(case.feature_channels, *case.cropped_shape))
    for d0, d1, features in iter_normalized_fused_feature_blocks(case): normalized[:, d0:d1] = features.astype(np.float16)
    normalized.flush(); case.release_fusion_buffers()
    return normalized


def shuffled_prototype_bank(prototypes: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed)); bank = np.stack([row[rng.permutation(row.size)] for row in np.asarray(prototypes, np.float32)])
    bank /= np.maximum(np.linalg.norm(bank, axis=1, keepdims=True), EPS)
    return bank.astype(np.float32)


def _score_feature_matrix(features: np.ndarray, prototypes: np.ndarray, device: torch.device, chunk_size: int = 262_144) -> dict[str, np.ndarray]:
    flat = np.asarray(features, np.float32).reshape(features.shape[0], -1).T
    bank = torch.from_numpy(np.asarray(prototypes, np.float32)).to(device); outputs = {mode: [] for mode in SCORE_MODES}
    for start in range(0, len(flat), chunk_size):
        values = torch.from_numpy(flat[start:start + chunk_size]).to(device); similarities = values @ bank.T
        outputs["max"].append(similarities.max(1).values.cpu().numpy())
        for requested, mode in ((3, "top3_mean"), (5, "top5_mean")):
            k = min(requested, similarities.shape[1]); outputs[mode].append(torch.topk(similarities, k, dim=1).values.mean(1).cpu().numpy())
    return {mode: np.concatenate(parts) for mode, parts in outputs.items()}


def compute_similarity_after_feature_fusion(normalized: np.memmap, prototypes: np.ndarray, device: torch.device, directory: Path, prefix: str) -> dict[str, np.memmap]:
    if len(prototypes) == 0: raise ValueError(f"empty prototype bank: {prefix}")
    shape = tuple(normalized.shape[1:]); maps = {mode: np.memmap(directory / f"{prefix}_{mode}.dat", "w+", np.float32, shape=shape) for mode in SCORE_MODES}
    for d0 in range(shape[0]):
        block = np.asarray(normalized[:, d0:d0 + 1], np.float32)
        values = _score_feature_matrix(block, prototypes, device)
        for mode in SCORE_MODES: maps[mode][d0] = values[mode].reshape(shape[1:])
    for array in maps.values(): array.flush()
    return maps


def exact_topk_mask(scores: np.ndarray, valid_region: np.ndarray, k: int) -> np.ndarray:
    indices = np.flatnonzero(np.asarray(valid_region, bool).ravel()); k = min(max(int(k), 0), len(indices)); result = np.zeros(np.asarray(scores).size, bool)
    if k:
        values = np.asarray(scores, np.float64).ravel()[indices]
        if k == len(values): chosen = indices
        else:
            threshold = np.partition(values, len(values) - k)[len(values) - k]
            higher = np.flatnonzero(values > threshold); tied = np.flatnonzero(values == threshold)
            chosen = indices[np.concatenate((higher, tied[:k - len(higher)]))]
        result[chosen] = True
    return result.reshape(np.asarray(scores).shape)


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float, str]:
    y, s = np.asarray(labels, bool).ravel(), np.asarray(scores, np.float64).ravel(); classes = np.unique(y)
    if classes.size == 0: return math.nan, math.nan, "empty_region"
    if classes.size < 2: return math.nan, math.nan, "single_class_no_positive" if not classes[0] else "single_class_no_negative"
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s)), ""


def lesion_hits(prediction: np.ndarray, gt: np.ndarray) -> tuple[int, int, float]:
    labels, count = connected_lesions(gt); hits = sum(bool((prediction & (labels == lesion_id)).any()) for lesion_id in range(1, count + 1))
    return hits, count, float(hits / count) if count else math.nan


def localization_metrics(score: np.ndarray, target: np.ndarray, valid_region: np.ndarray, baseline_probability: np.ndarray) -> dict[str, Any]:
    gt = np.asarray(target, bool) & valid_region; volume = int(gt.sum())
    if volume == 0: raise ValueError("query GT is empty")
    auroc, auprc, reason = safe_auc(gt[valid_region], score[valid_region]); top1 = exact_topk_mask(score, valid_region, volume)
    hits, lesion_count, hit_rate = lesion_hits(top1, gt)
    precision_top = float((top1 & gt).sum() / max(int(top1.sum()), 1)); prevalence = volume / int(valid_region.sum())
    result = {"auroc": auroc, "auprc": auprc, "auc_nan_reason": reason, "dice_at_oracle_volume": binary_metrics(top1, gt)["dice"], "lesion_hits": hits, "lesion_count": lesion_count, "lesion_level_hit_rate": hit_rate, "top_score_foreground_enrichment": precision_top / prevalence}
    for multiple in (1, 2, 5): result[f"recall_at_{multiple}v"] = float((exact_topk_mask(score, valid_region, multiple * volume) & gt).sum() / volume)
    fn_region = valid_region & (baseline_probability < 0.5); fn_gt = gt & fn_region; fn_volume = int(fn_gt.sum())
    fn_auroc, fn_auprc, fn_reason = safe_auc(fn_gt[fn_region], score[fn_region]); result.update({"fn_region_auroc": fn_auroc, "fn_region_auprc": fn_auprc, "fn_auc_nan_reason": fn_reason, "fn_positive_voxels": fn_volume})
    for multiple in (1, 2, 5): result[f"fn_recall_at_{multiple}v"] = math.nan if fn_volume == 0 else float((exact_topk_mask(score, fn_region, multiple * fn_volume) & fn_gt).sum() / fn_volume)
    return result


def standardize_in_valid_region(score: np.ndarray, valid_region: np.ndarray) -> tuple[np.ndarray, float, float, str]:
    values = np.asarray(score, np.float32)[valid_region]; mean, std = float(values.mean()), float(values.std()); output = np.zeros_like(score, np.float32)
    if not np.isfinite(std) or std <= EPS: return output, mean, std, "zero_variance_score"
    output[valid_region] = (values - mean) / std; return output, mean, std, "ok"


def fuse_probability(baseline_probability: np.ndarray, standardized_score: np.ndarray, alpha: float) -> np.ndarray:
    probability = np.clip(np.asarray(baseline_probability, np.float32), 1e-6, 1 - 1e-6)
    baseline_logit = np.log(probability / (1 - probability))
    return baseline_logit + float(alpha) * np.asarray(standardized_score, np.float32) >= 0


def effective_empty_recovery(baseline_empty: bool, metrics: dict[str, Any], lesion_hit_count: int) -> bool:
    return bool(baseline_empty and metrics["dice"] >= 0.05 and metrics["recall"] >= 0.10 and lesion_hit_count >= 1)


def _localization_row(case: FusedFeatureCase, bank: str, mode: str, score: np.ndarray, random_seed: int | None = None) -> dict[str, Any]:
    crop = _bbox_slices(case.bbox); valid = np.ones(case.cropped_shape, bool); gt_crop = case.gt[crop]; baseline_crop = case.baseline_probability[crop]
    return {"case_id": case.case.case_id, "bank": bank, "score_mode": mode, "random_seed": random_seed if random_seed is not None else "", **case.baseline_audit, "baseline_precision": case.baseline_metrics["precision"], "baseline_recall": case.baseline_metrics["recall"], "baseline_pred_foreground_voxels": case.baseline_metrics["pred_foreground_voxels"], "baseline_empty_prediction": case.baseline_metrics["pred_foreground_voxels"] == 0, **localization_metrics(score, gt_crop, valid, baseline_crop)}


def run_query_case(case: FusedFeatureCase, prototype_banks: dict[str, np.ndarray], random_banks: dict[int, np.ndarray], output_dir: Path, device: torch.device) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    normalized = materialize_normalized_features(case); crop = _bbox_slices(case.bbox); valid = np.ones(case.cropped_shape, bool)
    localization_rows = [_localization_row(case, "baseline_probability", "native", case.baseline_probability[crop])]
    maps_by_bank = {bank: compute_similarity_after_feature_fusion(normalized, prototypes, device, case.temp_dir, bank) for bank, prototypes in prototype_banks.items()}
    for bank, maps in maps_by_bank.items():
        for mode, score in maps.items(): localization_rows.append(_localization_row(case, bank, mode, score))
    random_case_metrics = []
    for seed, bank in random_banks.items():
        maps = compute_similarity_after_feature_fusion(normalized, bank, device, case.temp_dir, f"random_{seed}")
        for mode, score in maps.items():
            row = _localization_row(case, "random", mode, score, seed); localization_rows.append(row)
            if mode == "max": random_case_metrics.append(row)
        for score in maps.values(): path = Path(score.filename); _close_memmap(score); path.unlink()
    fusion_rows = []; fused_primary = None
    baseline_empty = case.baseline_metrics["pred_foreground_voxels"] == 0
    for bank, maps in maps_by_bank.items():
        for mode, score in maps.items():
            standardized, mean, std, status = standardize_in_valid_region(score, valid)
            for alpha in ALPHAS:
                fused_crop = fuse_probability(case.baseline_probability[crop], standardized, alpha)
                fused_full = restore_crop_to_full(fused_crop, case.bbox, case.original_shape)
                metrics = binary_metrics(fused_full, case.gt); hits, lesion_count, hit_rate = lesion_hits(fused_full, case.gt)
                fusion_rows.append({"case_id": case.case.case_id, "bank": bank, "score_mode": mode, "alpha": alpha, "threshold": 0.5, **metrics, "lesion_hits": hits, "lesion_count": lesion_count, "lesion_level_hit_rate": hit_rate, "delta_dice": metrics["dice"] - case.baseline_metrics["dice"], "delta_precision": metrics["precision"] - case.baseline_metrics["precision"], "delta_recall": metrics["recall"] - case.baseline_metrics["recall"], "baseline_empty_prediction": baseline_empty, "effective_empty_recovery": effective_empty_recovery(baseline_empty, metrics, hits), "score_mean_in_valid_region": mean, "score_std_in_valid_region": std, "standardization_status": status})
                if bank == "oracle" and mode == "max" and alpha == 1.0: fused_primary = fused_full
    oracle_similarity = restore_crop_to_full(np.asarray(maps_by_bank["oracle"]["max"]), case.bbox, case.original_shape)
    prediction_similarity = restore_crop_to_full(np.asarray(maps_by_bank["prediction"]["max"]), case.bbox, case.original_shape)
    if fused_primary is None: raise AssertionError("fixed primary fusion missing")
    case_dir = output_dir / case.case.case_id; case_dir.mkdir(parents=True, exist_ok=True)
    save_nifti_like(case.baseline_probability, case_dir / "baseline_probability.nii.gz", case.properties)
    save_nifti_like(oracle_similarity, case_dir / "oracle_similarity.nii.gz", case.properties)
    save_nifti_like(prediction_similarity, case_dir / "prediction_similarity.nii.gz", case.properties)
    save_nifti_like(fused_primary.astype(np.uint8), case_dir / "fused_prediction.nii.gz", case.properties, integer=True)
    save_visualization(case.image, case.gt, case.baseline_probability, oracle_similarity, fused_primary, case_dir / "diagnostic.png", case.case.case_id)
    for maps in maps_by_bank.values():
        for score in maps.values(): _close_memmap(score)
    _close_memmap(normalized); case.cleanup()
    return localization_rows, fusion_rows, {**case.metadata, "random_seed_max_metrics": random_case_metrics}


def save_nifti_like(array: np.ndarray, path: Path, properties: dict[str, Any], integer: bool = False) -> None:
    data = np.asarray(array).transpose((2, 1, 0)).astype(np.uint8 if integer else np.float32)
    image = nib.Nifti1Image(data, affine=properties["nibabel_stuff"]["reoriented_affine"])
    transform = ornt_transform(axcodes2ornt("RAS"), io_orientation(properties["nibabel_stuff"]["original_affine"]))
    restored = image.as_reoriented(transform)
    if not np.allclose(restored.affine, properties["nibabel_stuff"]["original_affine"], atol=1e-4):
        raise ValueError(f"full-volume NIfTI geometry restoration failed: {path}")
    path.parent.mkdir(parents=True, exist_ok=True); nib.save(restored, str(path))
    roundtrip, roundtrip_properties = _load_nifti(path)
    if tuple(roundtrip.shape) != tuple(np.asarray(array).shape) or not np.allclose(
        roundtrip_properties["nibabel_stuff"]["original_affine"],
        properties["nibabel_stuff"]["original_affine"], atol=1e-4,
    ):
        raise ValueError(f"saved NIfTI does not match original full-volume shape/affine: {path}")


def save_visualization(image: np.ndarray, gt: np.ndarray, baseline: np.ndarray, oracle: np.ndarray, fused: np.ndarray, path: Path, case_id: str) -> None:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    z = int(np.argmax(gt.sum(axis=(1, 2)))) if gt.any() else image.shape[0] // 2
    fn_response = np.where(gt & (baseline < 0.5), oracle, np.nan)
    panels = [(image[z], "HBP", "gray", None, None), (gt[z], "GT", "gray", 0, 1), (baseline[z], "baseline probability", "magma", 0, 1), (baseline[z] >= .5, "baseline prediction", "gray", 0, 1), (oracle[z], "oracle similarity", "viridis", -1, 1), (fn_response[z], "response in baseline FN", "viridis", -1, 1), (fused[z], "oracle/max alpha=1", "gray", 0, 1)]
    figure, axes = plt.subplots(2, 4, figsize=(18, 9))
    for axis, (values, title, cmap, vmin, vmax) in zip(axes.ravel(), panels): axis.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax); axis.set_title(title); axis.axis("off")
    axes.ravel()[-1].axis("off"); figure.suptitle(f"{case_id}, axial D={z}"); figure.tight_layout(); figure.savefig(path, dpi=140); plt.close(figure)


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def paired_comparison(first: Sequence[float], second: Sequence[float], seed: int = 2026, samples: int = 5000) -> dict[str, Any]:
    a, b = np.asarray(first, np.float64), np.asarray(second, np.float64); keep = np.isfinite(a) & np.isfinite(b); differences = a[keep] - b[keep]
    if not len(differences): return {"n": 0, "mean_difference": math.nan, "median_difference": math.nan, "bootstrap_95_ci": [math.nan, math.nan], "wilcoxon_p": math.nan, "clearly_better": False}
    rng = np.random.default_rng(seed); boot = differences[rng.integers(0, len(differences), size=(samples, len(differences)))].mean(axis=1); ci = np.quantile(boot, [0.025, 0.975])
    try: p = float(wilcoxon(differences).pvalue) if np.any(differences != 0) else 1.0
    except ValueError: p = math.nan
    return {"n": int(len(differences)), "case_differences": differences.tolist(), "mean_difference": float(differences.mean()), "median_difference": float(np.median(differences)), "bootstrap_95_ci": [float(ci[0]), float(ci[1])], "wilcoxon_p": p, "clearly_better": bool(ci[0] > 0)}


def random_seed_summary(rows: Sequence[dict[str, Any]], metric: str) -> dict[str, Any]:
    """Summarize one case-averaged value per random seed."""
    by_seed: dict[int, list[float]] = {}
    for row in rows:
        seed = int(row["random_seed"])
        value = float(row[metric])
        if np.isfinite(value):
            by_seed.setdefault(seed, []).append(value)
    seed_means = {str(seed): float(np.mean(values)) for seed, values in sorted(by_seed.items()) if values}
    values = np.asarray(list(seed_means.values()), np.float64)
    if not len(values):
        return {
            "n_seeds": 0, "seed_case_means": {}, "mean": math.nan,
            "std": math.nan, "interval_95": [math.nan, math.nan],
        }
    return {
        "n_seeds": int(len(values)),
        "seed_case_means": seed_means,
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "interval_95": [float(x) for x in np.quantile(values, [0.025, 0.975])],
    }


def summarize(localization: Sequence[dict[str, Any]], fusion: Sequence[dict[str, Any]], audit: dict[str, Any], case_metadata: Sequence[dict[str, Any]]) -> dict[str, Any]:
    def select(bank: str, mode: str, seed: Any = None):
        rows = [r for r in localization if r["bank"] == bank and r["score_mode"] == mode]
        return rows if seed is None else [r for r in rows if r["random_seed"] == seed]
    baseline, oracle, prediction, random_rows = select("baseline_probability", "native"), select("oracle", "max"), select("prediction", "max"), select("random", "max")
    case_order = [r["case_id"] for r in oracle]
    def values(rows, metric): return [float(next(r for r in rows if r["case_id"] == case)[metric]) for case in case_order]
    def finite_mean(items: Sequence[float]) -> float:
        finite = np.asarray(items, np.float64)
        finite = finite[np.isfinite(finite)]
        return float(finite.mean()) if len(finite) else math.nan

    random_by_case = {
        case: [float(r["auprc"]) for r in random_rows if r["case_id"] == case]
        for case in case_order
    }
    comparisons = {
        "oracle_vs_baseline_auprc": paired_comparison(values(oracle, "auprc"), values(baseline, "auprc")),
        "oracle_vs_baseline_dice_at_oracle_volume": paired_comparison(values(oracle, "dice_at_oracle_volume"), values(baseline, "dice_at_oracle_volume")),
        "oracle_vs_random_auprc": paired_comparison(
            values(oracle, "auprc"), [finite_mean(random_by_case[c]) for c in case_order]
        ),
        "oracle_vs_random_fn_auprc": paired_comparison(
            values(oracle, "fn_region_auprc"),
            [finite_mean([float(r["fn_region_auprc"]) for r in random_rows if r["case_id"] == c]) for c in case_order],
        ),
        "oracle_vs_prediction_auprc": paired_comparison(values(oracle, "auprc"), values(prediction, "auprc")),
        "oracle_vs_prediction_dice_at_oracle_volume": paired_comparison(values(oracle, "dice_at_oracle_volume"), values(prediction, "dice_at_oracle_volume")),
        "oracle_vs_prediction_fn_auprc": paired_comparison(values(oracle, "fn_region_auprc"), values(prediction, "fn_region_auprc")),
    }
    fusion_counts = {}
    for alpha in ALPHAS:
        rows = [r for r in fusion if r["bank"] == "oracle" and r["score_mode"] == "max" and float(r["alpha"]) == alpha]; delta = np.asarray([float(r["delta_dice"]) for r in rows])
        recovered = [r for r in rows if r["effective_empty_recovery"]]
        recall_delta = np.asarray([float(r["delta_recall"]) for r in rows], np.float64)
        precision_delta = np.asarray([float(r["delta_precision"]) for r in rows], np.float64)
        severe_precision_drops = precision_delta < -0.10
        fusion_counts[str(alpha)] = {
            "improved_gt_0_02": int((delta > .02).sum()),
            "declined_lt_minus_0_02": int((delta < -.02).sum()),
            "within_plus_minus_0_02": int((np.abs(delta) <= .02).sum()),
            "mean_delta_dice": finite_mean(delta),
            "mean_delta_recall": finite_mean(recall_delta),
            "mean_delta_precision": finite_mean(precision_delta),
            "severe_precision_drop_cases": int(severe_precision_drops.sum()),
            "severe_precision_drop_fraction": float(severe_precision_drops.mean()) if len(rows) else math.nan,
            "recall_gain_with_severe_precision_loss": bool(
                len(rows) and finite_mean(recall_delta) > 0 and severe_precision_drops.mean() >= 0.25
            ),
            "effective_empty_recoveries": len(recovered),
            "empty_baselines": int(sum(bool(r["baseline_empty_prediction"]) for r in rows)),
            "effective_recovery_cases": [
                {"case_id": r["case_id"], "dice": r["dice"], "recall": r["recall"],
                 "precision": r["precision"], "tp": r["tp"], "fp": r["fp"],
                 "lesion_hits": r["lesion_hits"]}
                for r in recovered
            ],
        }
    return {
        "experiment": "oracle_support_full_volume", "audit": audit, "random_seeds": list(RANDOM_SEEDS), "case_inference_metadata": list(case_metadata),
        "definitions": {"valid_region": "crop_to_nonzero bbox in the original full volume", "clearly_better": "lower bound of paired bootstrap 95% CI is > 0", "effective_empty_recovery": "baseline empty and fused Dice>=0.05, Recall>=0.10, and >=1 GT lesion hit", "fusion_categories": "delta Dice >0.02, <-0.02, otherwise within +/-0.02", "severe_precision_drop": "delta Precision < -0.10; tradeoff flag requires this in >=25% of query cases while mean Recall increases"},
        "paired_comparisons": comparisons, "random_control": {metric: random_seed_summary(random_rows, metric) for metric in ("auprc", "dice_at_oracle_volume", "fn_region_auprc", "fn_recall_at_1v")},
        "fixed_alpha_fusion_counts": fusion_counts,
        "answers": {
            "oracle_clearly_better_than_baseline": bool(
                comparisons["oracle_vs_baseline_auprc"]["clearly_better"]
                and comparisons["oracle_vs_baseline_dice_at_oracle_volume"]["clearly_better"]
            ),
            "oracle_clearly_better_than_random": comparisons["oracle_vs_random_auprc"]["clearly_better"],
            "oracle_clearly_better_in_fn_region": comparisons["oracle_vs_random_fn_auprc"]["clearly_better"],
            "fixed_alpha_improved_declined_unchanged": {
                alpha: {
                    "improved": item["improved_gt_0_02"],
                    "declined": item["declined_lt_minus_0_02"],
                    "unchanged": item["within_plus_minus_0_02"],
                }
                for alpha, item in fusion_counts.items()
            },
            "empty_prediction_effective_recovery": {
                alpha: {
                    "recovered": item["effective_empty_recoveries"],
                    "total_empty": item["empty_baselines"],
                    "cases_with_fp": item["effective_recovery_cases"],
                }
                for alpha, item in fusion_counts.items()
            },
            "recall_precision_tradeoff": {
                alpha: {
                    "mean_delta_recall": item["mean_delta_recall"],
                    "mean_delta_precision": item["mean_delta_precision"],
                    "severe_precision_drop_cases": item["severe_precision_drop_cases"],
                    "recall_gain_with_severe_precision_loss": item["recall_gain_with_severe_precision_loss"],
                }
                for alpha, item in fusion_counts.items()
            },
            "gt_vs_prediction_prototype": {
                "auprc": comparisons["oracle_vs_prediction_auprc"],
                "dice_at_oracle_volume": comparisons["oracle_vs_prediction_dice_at_oracle_volume"],
                "fn_region_auprc": comparisons["oracle_vs_prediction_fn_auprc"],
            },
        },
    }


def dry_run_checks() -> None:
    crop = np.arange(24).reshape(2, 3, 4); restored = restore_crop_to_full(crop, ((1, 3), (2, 5), (1, 5)), (4, 7, 6)); assert np.array_equal(restored[1:3, 2:5, 1:5], crop)
    verify_baseline_dice("dry", .5, .50001); assert_support_query_disjoint(["a"], ["b"])
    metrics = {"dice": .01, "recall": .01}; assert not effective_empty_recovery(True, metrics, 1)
    print("[dry-run] full-volume restoration, baseline guard, split disjointness, and recovery guard passed")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-json", required=True); parser.add_argument("--dice-csv", required=True)
    parser.add_argument("--image-dir", required=True, help="Original HBP full-volume NIfTI directory"); parser.add_argument("--label-dir", required=True)
    parser.add_argument("--prediction-dir", default=None); parser.add_argument("--model-dir", required=True); parser.add_argument("--text-model", required=True)
    parser.add_argument("--prompt", default="HCC lesion"); parser.add_argument("--output-dir", required=True)
    parser.add_argument("--support-threshold", type=float, default=.8); parser.add_argument("--query-threshold", type=float, default=.5); parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true"); return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    for value in (args.split_json, args.dice_csv, args.image_dir, args.label_dir, args.model_dir, args.text_model):
        if not Path(value).exists(): raise FileNotFoundError(value)
    if args.prediction_dir and not Path(args.prediction_dir).exists(): raise FileNotFoundError(args.prediction_dir)
    if not 0 <= args.query_threshold < args.support_threshold <= 1: raise ValueError("invalid thresholds")
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    supports, queries, audit = audit_inputs(args)
    if args.dry_run:
        for case in (supports[0], queries[0]):
            image, label, properties = load_case(case)
            if case.prediction:
                load_probability(
                    case.prediction, image.shape, properties["nibabel_stuff"]["original_affine"]
                )
            print(f"[dry-run] {case.case_id}: full-volume shape={image.shape}, GT={int(label.sum())}")
        dry_run_checks(); print("[dry-run] completed without model loading"); return 0
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    from voxtell.inference.predictor import VoxTellPredictor
    predictor = VoxTellPredictor(args.model_dir, device=device, text_encoding_model=args.text_model); predictor.network = predictor.network.to(device).eval(); text = predictor.embed_text_prompts(args.prompt)
    support_rows = []; oracle = []; prediction = []; metadata = []; baseline_audits = []
    for index, case in enumerate(supports, 1):
        print(f"[support {index}/{len(supports)}] {case.case_id}"); fused = run_full_volume_feature_inference(predictor, text, case, output_dir, args.prompt)
        o, p, rows, info = extract_support_case(fused); oracle.extend(o); prediction.extend(p); support_rows.extend(rows); metadata.append(info); baseline_audits.append(fused.baseline_audit); fused.cleanup()
    if not oracle or not prediction: raise RuntimeError("empty oracle or prediction prototype bank")
    oracle_bank, prediction_bank = np.stack(oracle).astype(np.float32), np.stack(prediction).astype(np.float32)
    random_banks = {seed: shuffled_prototype_bank(oracle_bank, seed) for seed in RANDOM_SEEDS}
    counters = {"oracle": 0, "prediction": 0}
    for row in support_rows:
        if row["status"] == "ok": row["prototype_index"] = counters[row["bank"]]; counters[row["bank"]] += 1
    _write_csv(output_dir / "support_bank.csv", support_rows)
    np.savez_compressed(output_dir / "oracle_prototypes.npz", oracle_prototypes=oracle_bank, prediction_prototypes=prediction_bank, random_prototypes=np.stack([random_banks[s] for s in RANDOM_SEEDS]), random_seeds=np.asarray(RANDOM_SEEDS))
    localization = []; fusion_rows = []
    for index, case in enumerate(queries, 1):
        print(f"[query {index}/{len(queries)}] {case.case_id}"); fused = run_full_volume_feature_inference(predictor, text, case, output_dir, args.prompt); baseline_audits.append(fused.baseline_audit)
        rows, frows, info = run_query_case(fused, {"oracle": oracle_bank, "prediction": prediction_bank}, random_banks, output_dir, device); localization.extend(rows); fusion_rows.extend(frows); metadata.append(info)
        _write_csv(output_dir / "per_case_localization.csv", localization); _write_csv(output_dir / "per_case_fusion.csv", fusion_rows)
    audit.update({"prompt": args.prompt, "model_dir": str(Path(args.model_dir).resolve()), "text_model": str(Path(args.text_model).resolve()), "prediction_source": "recomputed_voxtell_probability", "baseline_dice_tolerance": BASELINE_DICE_TOLERANCE, "baseline_case_audits": baseline_audits})
    summary = summarize(localization, fusion_rows, audit, metadata)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle: json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=True)
    print(json.dumps(summary["answers"], indent=2, ensure_ascii=False, allow_nan=True)); return 0


if __name__ == "__main__": sys.exit(main())
