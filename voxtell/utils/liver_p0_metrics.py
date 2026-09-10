from __future__ import annotations

import csv
from pathlib import Path

import nibabel as nib
import numpy as np


def _binary_dice_iou(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> tuple[float, float]:
    pred = pred.astype(bool, copy=False)
    gt = gt.astype(bool, copy=False)

    intersection = np.logical_and(pred, gt).sum(dtype=np.int64)
    pred_sum = pred.sum(dtype=np.int64)
    gt_sum = gt.sum(dtype=np.int64)
    union = np.logical_or(pred, gt).sum(dtype=np.int64)

    if gt_sum == 0:
        empty_score = 1.0 if pred_sum == 0 else 0.0
        return empty_score, empty_score

    dice = (2.0 * intersection + eps) / (pred_sum + gt_sum + eps)
    iou = (intersection + eps) / (union + eps)
    return float(dice), float(iou)


def compute_p0_liver_metrics(
    pred_dir: str | Path = "/data/zy/VoxTell_from_disk/out_multi/P0",
    gt_dir: str | Path = "/data/zy/CT_MRI_DATA_3D/labels/P0",
    liver_label: int = 5,
    output_csv: str | Path | None = None,
) -> list[dict[str, float | str | int]]:
    """Compute per-case liver Dice and IoU for P0 multiclass NIfTI masks.

    `pred_dir` is treated as the prediction directory and `gt_dir` as the
    ground-truth directory. Both masks are converted to binary liver masks by
    `mask == liver_label`.
    """

    pred_dir = Path(pred_dir)
    gt_dir = Path(gt_dir)
    rows: list[dict[str, float | str | int]] = []

    for pred_path in sorted(pred_dir.glob("*.nii.gz")):
        gt_path = gt_dir / pred_path.name
        if not gt_path.exists():
            rows.append(
                {
                    "case": pred_path.name,
                    "dice": np.nan,
                    "iou": np.nan,
                    "pred_liver_voxels": -1,
                    "gt_liver_voxels": -1,
                    "status": "missing_gt",
                }
            )
            continue

        pred_arr = np.asanyarray(nib.load(str(pred_path)).dataobj)
        gt_arr = np.asanyarray(nib.load(str(gt_path)).dataobj)
        if pred_arr.shape != gt_arr.shape:
            raise ValueError(f"Shape mismatch for {pred_path.name}: pred={pred_arr.shape}, gt={gt_arr.shape}")

        pred_liver = pred_arr == liver_label
        gt_liver = gt_arr == liver_label
        dice, iou = _binary_dice_iou(pred_liver, gt_liver)
        rows.append(
            {
                "case": pred_path.name,
                "dice": dice,
                "iou": iou,
                "pred_liver_voxels": int(pred_liver.sum(dtype=np.int64)),
                "gt_liver_voxels": int(gt_liver.sum(dtype=np.int64)),
                "status": "ok",
            }
        )

    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["case", "dice", "iou", "pred_liver_voxels", "gt_liver_voxels", "status"],
            )
            writer.writeheader()
            writer.writerows(rows)

    ok_rows = [row for row in rows if row["status"] == "ok"]
    if ok_rows:
        mean_dice = float(np.mean([float(row["dice"]) for row in ok_rows]))
        mean_iou = float(np.mean([float(row["iou"]) for row in ok_rows]))
        print(f"P0 liver mean Dice={mean_dice:.6f}, mean IoU={mean_iou:.6f}, n={len(ok_rows)}")

    return rows


if __name__ == "__main__":
    compute_p0_liver_metrics(output_csv="/data/zy/VoxTell_from_disk/out_multi/P0_liver_metrics.csv")
