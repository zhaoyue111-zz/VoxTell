import numpy as np
import torch
from requests.utils import dict_from_cookiejar
from medpy.metric.binary import hd95, asd

def _confusion_from_binary(pred01: np.ndarray, gt01: np.ndarray):
    """
    pred01, gt01: ndarray, values in {0,1}
    """
    pred = (pred01 > 0).astype(np.bool_)
    gt = (gt01 > 0).astype(np.bool_)

    tp = np.logical_and(pred, gt).sum(dtype=np.int64)
    fp = np.logical_and(pred, np.logical_not(gt)).sum(dtype=np.int64)
    fn = np.logical_and(np.logical_not(pred), gt).sum(dtype=np.int64)
    tn = np.logical_and(np.logical_not(pred), np.logical_not(gt)).sum(dtype=np.int64)
    return tp, fp, fn, tn

# def dice_iou(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7):
#     """
#     pred, gt: ndarray shape (P, X, Y, Z), values {0,1}
#     """
#     if pred.shape != gt.shape:
#         raise ValueError(f"Shape mismatch: pred={pred.shape}, gt={gt.shape}. Expected same shape (P,X,Y,Z).")
#     if pred.ndim != 4:
#         raise ValueError(f"Expected pred/gt ndim=4 (P,X,Y,Z), got pred.ndim={pred.ndim}")
#
#     pred=pred[0]
#     gt=gt[0]
#
#     tp, fp, fn, tn = _confusion_from_binary(pred, gt)
#
#     dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
#     iou = (tp + eps) / (tp + fp + fn + eps)
#
#     return float(dice), float(iou)


def dice_iou(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7):
    """
    Args:
        pred: ndarray of shape (P, Z, X, Y) with values {0, 1}.
        gt: ndarray of shape (P, Z, X, Y) with values {0, 1}.
        eps: Small epsilon value to avoid division by zero.

    Returns:
        Tuple of (dice_scores, iou_scores), each a list[float] of length P.
    """
    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: pred={pred.shape}, gt={gt.shape}.")
    if pred.ndim != 4:
        raise ValueError(f"Expected 4D array (P,Z,X,Y), got {pred.ndim}")
    if pred.shape[1] == 0:
        raise ValueError(f"Expected non-empty Z dimension, got shape {pred.shape}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pred_t = torch.from_numpy(pred).to(device).float()
    target_t = torch.from_numpy(gt).to(device).float()

    pred_f = pred_t.flatten(1)
    tgt_f = target_t.flatten(1)

    inter = (pred_f * tgt_f).sum(1)  # [P]
    pred_sum = pred_f.sum(1)  # [P]
    tgt_sum = tgt_f.sum(1)  # [P]
    union_iou = pred_sum + tgt_sum - inter  # [P]
    union_dice = pred_sum + tgt_sum  # [P]

    iou_vec = (inter + eps) / (union_iou + eps)
    dice_vec = (2 * inter + eps) / (union_dice + eps)

    gt_empty = (tgt_sum == 0)
    pred_empty = (pred_sum == 0)

    iou_vec = torch.where(gt_empty, pred_empty.float(), iou_vec)
    dice_vec = torch.where(gt_empty, pred_empty.float(), dice_vec)

    return dice_vec.detach().cpu().tolist(), iou_vec.detach().cpu().tolist()

def compute_metrics(pred, gt):
    """
    Args:
        pred: numpy array of shape (P, Z, X, Y) with values {0, 1}.
        gt: numpy array of shape (P, Z, X, Y) with values {0, 1}.

    Returns:
        Tuple of (dice_scores, iou_scores), each a list[float] of length P.
    """
    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: pred={pred.shape}, gt={gt.shape}.")
    if pred.ndim != 4:
        raise ValueError(f"Expected 4D array (P,Z,X,Y), got {pred.ndim}")

    num_classes = pred.shape[0]
    dice_scores = np.zeros(num_classes, dtype=np.float64)
    iou_scores = np.zeros(num_classes, dtype=np.float64)

    for class_idx in range(num_classes):
        pred_3d = pred[class_idx]  # [Z, X, Y]
        gt_3d = gt[class_idx]  # [Z, X, Y]

        mdice = 0.0
        miou = 0.0
        Z = pred_3d.shape[0]
        for z in range(Z):
            pred_slice = pred_3d[z, :, :]  # [X,Y]
            gt_slice = gt_3d[z, :, :]  # [X,Y]

            pred_bool = pred_slice.astype(bool)
            gt_bool = gt_slice.astype(bool)

            intersection = np.logical_and(pred_bool, gt_bool).sum()
            union = np.logical_or(pred_bool, gt_bool).sum()
            iou = intersection / union if union > 0 else 1.0

            denom = pred_bool.sum() + gt_bool.sum()
            dice = (2 * intersection) / denom if denom > 0 else 1.0

            mdice += dice
            miou += iou

        dice_scores[class_idx] = mdice / Z
        iou_scores[class_idx] = miou / Z

    return dice_scores.tolist(), iou_scores.tolist()

def compute_boundary_metrics(pred, gt, spacing):
    pred = pred[0].astype(bool)
    gt = gt[0].astype(bool)

    # case过滤
    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0, 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return np.nan, np.nan

    try:
        hd = hd95(pred, gt, voxelspacing=spacing)
        asd_val = asd(pred, gt, voxelspacing=spacing)
    except:
        hd, asd_val = np.nan, np.nan

    return hd, asd_val
