"""
metrics.py — segmentation evaluation metrics.

Class map (used throughout the regolith pipeline)
--------------------------------------------------
  { regolith: 0, rock: 1, sky: 2 }

The primary metric is rock-IoU (class 1) — rocks are the hazard class.
Mean-IoU is reported for completeness, but model selection (best checkpoint)
uses val rock-IoU.
"""

from __future__ import annotations


def mean_iou(pred, label, num_classes: int = 3) -> float:
    """Compute mean Intersection-over-Union across all classes.

    Parameters
    ----------
    pred : array-like, shape (H, W) or (N, H, W)
        Predicted class indices (int).
    label : array-like, shape (H, W) or (N, H, W)
        Ground-truth class indices (int). Same shape as pred.
    num_classes : int
        Number of classes. Default: 3 (regolith=0, rock=1, sky=2).

    Returns
    -------
    float
        Mean IoU across all classes (0.0–1.0).
        Classes absent from both pred and label are excluded from the mean.
    """
    # TODO (Task 3): implement
    raise NotImplementedError("mean_iou: implement in Task 3 (training)")


def class_iou(pred, label, cls: int) -> float:
    """Compute Intersection-over-Union for a single class.

    Parameters
    ----------
    pred : array-like, shape (H, W) or (N, H, W)
        Predicted class indices (int).
    label : array-like, shape (H, W) or (N, H, W)
        Ground-truth class indices (int). Same shape as pred.
    cls : int
        Class index to evaluate. For rock-IoU (the hazard class), use cls=1.

    Returns
    -------
    float
        IoU for the specified class (0.0–1.0).
        Returns 0.0 if the class is absent from both pred and label.
    """
    # TODO (Task 3): implement
    raise NotImplementedError("class_iou: implement in Task 3 (training)")
