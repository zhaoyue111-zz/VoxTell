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


def compute_metrics(pred, gt, eps=1e-7):
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

    P = pred.shape[0]
    dice = np.zeros(P, dtype=np.float64)
    iou = np.zeros(P, dtype=np.float64)

    for c in range(P):
        p = pred[c].astype(bool)
        g = gt[c].astype(bool)

        inter = np.logical_and(p, g).sum()
        union = np.logical_or(p, g).sum()
        ps = p.sum()
        gs = g.sum()

        dice[c] = (2 * inter + eps) / (ps + gs + eps)
        iou[c] = (inter + eps) / (union + eps)

    return dice.tolist(), iou.tolist()
