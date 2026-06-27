"""
metrics.py — segmentation evaluation metrics.

Class map (used throughout the regolith pipeline)
--------------------------------------------------
  { regolith: 0, rock: 1, sky: 2 }

The primary metric is rock-IoU (class 1) — rocks are the hazard class.
Mean-IoU is reported for completeness, but model selection (best checkpoint)
uses val rock-IoU.

All functions operate on integer NumPy arrays and have no dependency on PyTorch,
so they can run in any environment that has NumPy installed.
"""

from __future__ import annotations

import warnings

import numpy as np


def class_iou(pred: np.ndarray, label: np.ndarray, cls: int) -> float:
    """Compute Intersection-over-Union for a single class.

    Parameters
    ----------
    pred : np.ndarray, shape (H, W) or (N, H, W) or flat
        Predicted class indices (int).
    label : np.ndarray, shape matching pred
        Ground-truth class indices (int).
    cls : int
        Class index to evaluate. For rock-IoU (the hazard class), use cls=1.

    Returns
    -------
    float
        IoU for the specified class (0.0–1.0).
        Returns ``float('nan')`` if the class is absent from both pred and label
        (union == 0), so it can be excluded from means rather than counted as
        perfect or zero.
    """
    pred = np.asarray(pred)
    label = np.asarray(label)

    pred_mask = pred == cls
    label_mask = label == cls

    intersection = int(np.logical_and(pred_mask, label_mask).sum())
    union = int(np.logical_or(pred_mask, label_mask).sum())

    if union == 0:
        return float("nan")
    return intersection / union


def mean_iou(pred: np.ndarray, label: np.ndarray, num_classes: int = 3) -> float:
    """Compute mean Intersection-over-Union across all classes.

    Parameters
    ----------
    pred : np.ndarray, shape (H, W) or (N, H, W)
        Predicted class indices (int).
    label : np.ndarray, shape matching pred
        Ground-truth class indices (int).
    num_classes : int
        Number of classes. Default: 3 (regolith=0, rock=1, sky=2).

    Returns
    -------
    float
        Mean IoU across all present classes (0.0–1.0).
        Classes absent from both pred and label (IoU == NaN) are excluded
        from the mean. Returns ``float('nan')`` if all classes are absent.
    """
    ious = np.array(
        [class_iou(pred, label, cls) for cls in range(num_classes)],
        dtype=np.float64,
    )
    # np.nanmean returns nan if all values are nan — exactly what we want.
    # Suppress the RuntimeWarning emitted when the slice is all-NaN.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(np.nanmean(ious))


def confusion_counts(
    pred: np.ndarray, label: np.ndarray, cls: int
) -> tuple[int, int, int]:
    """Return (tp, fp, fn) counts for a single class.

    Parameters
    ----------
    pred : np.ndarray
        Predicted class indices (int).
    label : np.ndarray
        Ground-truth class indices (int).
    cls : int
        Class index to evaluate.

    Returns
    -------
    tuple[int, int, int]
        ``(tp, fp, fn)`` where:
        - tp = pixels predicted as *cls* that are truly *cls*
        - fp = pixels predicted as *cls* that are NOT *cls*
        - fn = pixels that are truly *cls* but NOT predicted as *cls*
    """
    pred = np.asarray(pred)
    label = np.asarray(label)

    pred_mask = pred == cls
    label_mask = label == cls

    tp = int(np.logical_and(pred_mask, label_mask).sum())
    fp = int(np.logical_and(pred_mask, ~label_mask).sum())
    fn = int(np.logical_and(~pred_mask, label_mask).sum())
    return tp, fp, fn


def precision_recall(
    pred: np.ndarray, label: np.ndarray, cls: int
) -> tuple[float, float]:
    """Compute precision and recall for a single class.

    Parameters
    ----------
    pred : np.ndarray
        Predicted class indices (int).
    label : np.ndarray
        Ground-truth class indices (int).
    cls : int
        Class index to evaluate. For the rock hazard class use cls=1.

    Returns
    -------
    tuple[float, float]
        ``(precision, recall)``.
        - precision = tp / (tp + fp); ``float('nan')`` when tp + fp == 0
          (no positive predictions).
        - recall    = tp / (tp + fn); ``float('nan')`` when tp + fn == 0
          (class absent from ground truth).
    """
    tp, fp, fn = confusion_counts(pred, label, cls)

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else float("nan")
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
    return precision, recall
