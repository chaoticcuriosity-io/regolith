"""
test_metrics.py — pytest unit tests for training/metrics.py.

All expected values are computed by hand; arithmetic is shown inline.
Class map: { regolith: 0, rock: 1, sky: 2 }
"""

import math
import numpy as np
import pytest

from training.metrics import class_iou, confusion_counts, mean_iou, precision_recall


# ---------------------------------------------------------------------------
# Fixtures / shared arrays
# ---------------------------------------------------------------------------

# 4×4 mixed array — chosen so every class appears in both pred and label
# and produces non-trivial IoU values that are easy to verify.
#
#   label:                 pred:
#   0 0 1 1               0 0 0 1
#   0 1 1 2               0 1 1 2
#   2 2 1 0               2 2 1 0
#   2 2 0 0               2 2 0 1   ← last pixel differs
#
#   Class 0 (regolith):
#     label pixels: (0,0)(0,1)(1,0)(2,3)(3,2)(3,3) → 6 pixels
#     pred  pixels: (0,0)(0,1)(0,2)(1,0)(2,3)(3,2) → 6 pixels
#     intersection (both==0): (0,0)(0,1)(1,0)(2,3)(3,2) → 5
#     union = 6+6-5 = 7   → IoU = 5/7
#
#   Class 1 (rock):
#     label pixels: (0,2)(0,3)(1,1)(1,2)(2,2) → 5 pixels
#     pred  pixels: (0,3)(1,1)(1,2)(2,2)(3,3) → 5 pixels
#     intersection (both==1): (0,3)(1,1)(1,2)(2,2) → 4
#     union = 5+5-4 = 6   → IoU = 4/6 = 2/3
#
#   Class 2 (sky):
#     label pixels: (1,3)(2,0)(2,1)(3,0)(3,1) → 5 pixels
#     pred  pixels: (1,3)(2,0)(2,1)(3,0)(3,1) → 5 pixels
#     intersection (both==2): all 5 → 5
#     union = 5+5-5 = 5   → IoU = 1.0
#
#   mean_iou = (5/7 + 2/3 + 1.0) / 3

LABEL_4X4 = np.array(
    [[0, 0, 1, 1],
     [0, 1, 1, 2],
     [2, 2, 1, 0],
     [2, 2, 0, 0]],
    dtype=np.int64,
)

PRED_4X4 = np.array(
    [[0, 0, 0, 1],
     [0, 1, 1, 2],
     [2, 2, 1, 0],
     [2, 2, 0, 1]],
    dtype=np.int64,
)


# ---------------------------------------------------------------------------
# class_iou
# ---------------------------------------------------------------------------

class TestClassIoU:
    def test_regolith_iou(self):
        # 5/7 ≈ 0.7142857…
        result = class_iou(PRED_4X4, LABEL_4X4, cls=0)
        assert result == pytest.approx(5 / 7, rel=1e-6)

    def test_rock_iou(self):
        # 4/6 = 2/3 ≈ 0.6666…
        result = class_iou(PRED_4X4, LABEL_4X4, cls=1)
        assert result == pytest.approx(2 / 3, rel=1e-6)

    def test_sky_iou(self):
        # Perfect overlap → 1.0
        result = class_iou(PRED_4X4, LABEL_4X4, cls=2)
        assert result == pytest.approx(1.0, rel=1e-6)

    def test_class_absent_from_both_returns_nan(self):
        # Tiny 2×2 with only classes 0 and 1; asking for class 2 → NaN
        pred = np.array([[0, 1], [0, 0]], dtype=np.int64)
        label = np.array([[0, 0], [1, 0]], dtype=np.int64)
        result = class_iou(pred, label, cls=2)
        assert np.isnan(result)

    def test_perfect_single_class(self):
        arr = np.array([[1, 1], [1, 1]], dtype=np.int64)
        assert class_iou(arr, arr, cls=1) == pytest.approx(1.0)

    def test_zero_overlap(self):
        # pred has all 0, label has all 1 → IoU for class 1 = 0/1 = 0.0
        pred = np.zeros((3, 3), dtype=np.int64)
        label = np.ones((3, 3), dtype=np.int64)
        assert class_iou(pred, label, cls=1) == pytest.approx(0.0)

    def test_flat_1d_array(self):
        # Function should also work on flat arrays
        pred = np.array([0, 1, 1, 2])
        label = np.array([0, 1, 0, 2])
        # cls=1: intersection=(idx1), union=(idx1,idx2) → 1/2
        assert class_iou(pred, label, cls=1) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# mean_iou
# ---------------------------------------------------------------------------

class TestMeanIoU:
    def test_mean_iou_4x4(self):
        expected = (5 / 7 + 2 / 3 + 1.0) / 3
        result = mean_iou(PRED_4X4, LABEL_4X4, num_classes=3)
        assert result == pytest.approx(expected, rel=1e-6)

    def test_perfect_prediction(self):
        arr = np.array([[0, 1, 2], [2, 0, 1]], dtype=np.int64)
        assert mean_iou(arr, arr, num_classes=3) == pytest.approx(1.0)

    def test_absent_class_excluded_from_mean(self):
        # Only classes 0 and 1 appear — class 2 absent from both.
        # mean_iou with num_classes=3 should average only the two present IoUs.
        pred = np.array([[0, 1], [1, 0]], dtype=np.int64)
        label = np.array([[0, 1], [1, 0]], dtype=np.int64)
        # Both IoUs = 1.0; class 2 → NaN (excluded) → mean = 1.0
        result = mean_iou(pred, label, num_classes=3)
        assert result == pytest.approx(1.0)

    def test_all_classes_nan_returns_nan(self):
        # Pathological: empty arrays — all classes have union 0
        pred = np.zeros((0,), dtype=np.int64)
        label = np.zeros((0,), dtype=np.int64)
        result = mean_iou(pred, label, num_classes=3)
        assert np.isnan(result)


# ---------------------------------------------------------------------------
# confusion_counts
# ---------------------------------------------------------------------------

class TestConfusionCounts:
    def test_rock_counts_4x4(self):
        # Class 1 (rock):
        #   label==1: (0,2)(0,3)(1,1)(1,2)(2,2) → 5 pixels
        #   pred==1 : (0,3)(1,1)(1,2)(2,2)(3,3) → 5 pixels
        #   TP (pred==1 AND label==1): (0,3)(1,1)(1,2)(2,2) → 4
        #   FP (pred==1 AND label!=1): (3,3) → 1
        #   FN (pred!=1 AND label==1): (0,2) → 1
        tp, fp, fn = confusion_counts(PRED_4X4, LABEL_4X4, cls=1)
        assert tp == 4
        assert fp == 1
        assert fn == 1

    def test_perfect_prediction_counts(self):
        arr = np.ones((4, 4), dtype=np.int64)
        tp, fp, fn = confusion_counts(arr, arr, cls=1)
        assert tp == 16
        assert fp == 0
        assert fn == 0

    def test_no_overlap_counts(self):
        pred = np.zeros((3, 3), dtype=np.int64)
        label = np.ones((3, 3), dtype=np.int64)
        tp, fp, fn = confusion_counts(pred, label, cls=1)
        assert tp == 0
        assert fp == 0
        assert fn == 9


# ---------------------------------------------------------------------------
# precision_recall
# ---------------------------------------------------------------------------

class TestPrecisionRecall:
    def test_rock_precision_recall_4x4(self):
        # From confusion_counts above: tp=4, fp=1, fn=1
        # precision = tp / (tp + fp) = 4/5 = 0.8
        # recall    = tp / (tp + fn) = 4/5 = 0.8
        p, r = precision_recall(PRED_4X4, LABEL_4X4, cls=1)
        assert p == pytest.approx(0.8, rel=1e-6)
        assert r == pytest.approx(0.8, rel=1e-6)

    def test_perfect_precision_recall(self):
        arr = np.array([[1, 1], [0, 2]], dtype=np.int64)
        p, r = precision_recall(arr, arr, cls=1)
        assert p == pytest.approx(1.0)
        assert r == pytest.approx(1.0)

    def test_zero_denominator_precision_is_nan(self):
        # pred has no class-1 pixels, label has no class-1 pixels → tp=fp=fn=0
        pred = np.zeros((3, 3), dtype=np.int64)
        label = np.zeros((3, 3), dtype=np.int64)
        p, r = precision_recall(pred, label, cls=1)
        assert np.isnan(p)
        assert np.isnan(r)

    def test_all_false_negatives(self):
        # pred all 0, label all 1 → tp=0, fp=0, fn=9
        # precision: 0/(0+0) → nan; recall: 0/(0+9) = 0.0
        pred = np.zeros((3, 3), dtype=np.int64)
        label = np.ones((3, 3), dtype=np.int64)
        p, r = precision_recall(pred, label, cls=1)
        assert np.isnan(p)
        assert r == pytest.approx(0.0)
