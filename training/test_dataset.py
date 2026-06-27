"""
test_dataset.py — pytest unit tests for training/dataset.py

Creates a tiny synthetic split (3 fake rgb PNGs + 3 mask PNGs with class ids
{0, 1, 2, 255}) in a temp directory and verifies the dataset contract:

  - Correct pair count.
  - Image shape [3, 512, 512], float32, ImageNet-normalized (values outside [0,1]).
  - Mask shape [512, 512], int64 (LongTensor), values ⊆ {0, 1, 2, 255}.
  - class_pixel_frequencies returns expected per-class counts (255 excluded).

Dependencies: torch, PIL (Pillow), numpy — no GPU, no real dataset, no
HuggingFace Transformers needed.

Run from repo root:
    python -m pytest training/test_dataset.py -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from training.dataset import SyntheticLunarDataset, class_pixel_frequencies


# ---------------------------------------------------------------------------
# Fixture: tiny synthetic split (module-scoped so it is created once per run)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_split(tmp_path_factory) -> Path:
    """Create a minimal 3-image split with deterministic class layout.

    Mask layout (per image, 512×512):
      rows   0–127  → regolith (0):  128 rows × 512 cols = 65 536 px
      rows 128–255  → rock     (1):  128 rows × 512 cols = 65 536 px
      rows 256–383  → sky      (2):  128 rows × 512 cols = 65 536 px
      rows 384–511  → ignore (255):  128 rows × 512 cols = 65 536 px

    Per class across 3 images: 3 × 65 536 = 196 608 px each.
    """
    root     = tmp_path_factory.mktemp("tiny_split")
    rgb_dir  = root / "rgb"
    mask_dir = root / "mask"
    rgb_dir.mkdir()
    mask_dir.mkdir()

    rng = np.random.default_rng(42)

    for i in range(3):
        # Random uint8 RGB so the image spans the full [0, 255] range
        rgb_arr = rng.integers(0, 256, (512, 512, 3), dtype=np.uint8)
        Image.fromarray(rgb_arr, mode="RGB").save(rgb_dir / f"rgb_{i:05d}.png")

        # Structured mask with all four label values present
        mask_arr = np.zeros((512, 512), dtype=np.uint8)
        mask_arr[128:256, :] = 1    # rock
        mask_arr[256:384, :] = 2    # sky
        mask_arr[384:,   :] = 255   # ignore / background
        Image.fromarray(mask_arr, mode="L").save(mask_dir / f"mask_{i:05d}.png")

    return root


# ---------------------------------------------------------------------------
# SyntheticLunarDataset tests
# ---------------------------------------------------------------------------

class TestSyntheticLunarDataset:

    def test_pair_count(self, tiny_split: Path) -> None:
        """Dataset length must equal the number of rgb/mask file pairs."""
        ds = SyntheticLunarDataset(tiny_split)
        assert len(ds) == 3

    def test_image_shape(self, tiny_split: Path) -> None:
        """Image must be [3, 512, 512]."""
        ds = SyntheticLunarDataset(tiny_split)
        img, _ = ds[0]
        assert img.shape == torch.Size([3, 512, 512])

    def test_image_dtype_float(self, tiny_split: Path) -> None:
        """Image must be float32."""
        ds = SyntheticLunarDataset(tiny_split)
        img, _ = ds[0]
        assert img.dtype == torch.float32

    def test_image_normalized_range(self, tiny_split: Path) -> None:
        """After ImageNet normalization, values span roughly (-3, +3).

        For random uint8 images the minimum will be negative (from subtracting
        the channel mean) and the maximum will exceed 1.0 — both confirm that
        normalization has been applied and the tensor is NOT a raw [0,1] image.
        """
        ds = SyntheticLunarDataset(tiny_split)
        img, _ = ds[0]
        min_val = img.min().item()
        max_val = img.max().item()
        # ImageNet-normalized values: min ≈ (0 - 0.485)/0.229 = -2.12
        assert min_val < 0.0,  f"Expected negative values after normalization; got min={min_val}"
        # ImageNet-normalized values: max ≈ (1 - 0.406)/0.225 = 2.64
        assert max_val > 1.0,  f"Expected values > 1 after normalization; got max={max_val}"
        # Sanity: not extreme garbage
        assert min_val > -10.0
        assert max_val <  10.0

    def test_mask_shape(self, tiny_split: Path) -> None:
        """Mask must be [512, 512]."""
        ds = SyntheticLunarDataset(tiny_split)
        _, mask = ds[0]
        assert mask.shape == torch.Size([512, 512])

    def test_mask_dtype_long(self, tiny_split: Path) -> None:
        """Mask must be int64 (LongTensor) for CrossEntropyLoss compatibility."""
        ds = SyntheticLunarDataset(tiny_split)
        _, mask = ds[0]
        assert mask.dtype == torch.int64

    def test_mask_values_subset(self, tiny_split: Path) -> None:
        """All mask pixel values must be in {0, 1, 2, 255} (no other labels)."""
        ds = SyntheticLunarDataset(tiny_split)
        valid = {0, 1, 2, 255}
        for i in range(len(ds)):
            _, mask = ds[i]
            unique = set(mask.unique().tolist())
            assert unique.issubset(valid), (
                f"Sample {i}: unexpected mask values {unique - valid}"
            )

    def test_ignore_255_preserved(self, tiny_split: Path) -> None:
        """The ignore label (255) must survive the mask pipeline unchanged."""
        ds = SyntheticLunarDataset(tiny_split)
        _, mask = ds[0]
        assert (mask == 255).any(), "Expected some ignore (255) pixels in mask"

    def test_augment_flag_preserves_shapes(self, tiny_split: Path) -> None:
        """augment=True must not alter tensor shapes (hflip is shape-preserving)."""
        ds = SyntheticLunarDataset(tiny_split, augment=True)
        # Force the flip branch by setting a deterministic seed
        torch.manual_seed(0)
        for i in range(len(ds)):
            img, mask = ds[i]
            assert img.shape  == torch.Size([3, 512, 512])
            assert mask.shape == torch.Size([512, 512])

    def test_all_items_accessible(self, tiny_split: Path) -> None:
        """All items must be retrievable without error."""
        ds = SyntheticLunarDataset(tiny_split)
        for i in range(len(ds)):
            img, mask = ds[i]
            assert img is not None
            assert mask is not None


# ---------------------------------------------------------------------------
# class_pixel_frequencies tests
# ---------------------------------------------------------------------------

class TestClassPixelFrequencies:

    def test_returns_ndarray_shape3(self, tiny_split: Path) -> None:
        freqs = class_pixel_frequencies(tiny_split)
        assert isinstance(freqs, np.ndarray)
        assert freqs.shape == (3,)

    def test_all_classes_present(self, tiny_split: Path) -> None:
        """All three classes should appear (each has 128 rows × 512 cols × 3 images)."""
        freqs = class_pixel_frequencies(tiny_split)
        assert all(freqs > 0), f"Expected all classes present; got {freqs}"

    def test_ignore_255_excluded(self, tiny_split: Path) -> None:
        """Sum of class counts < total pixels because 255-pixels are excluded."""
        freqs    = class_pixel_frequencies(tiny_split)
        total_px = 3 * 512 * 512   # 3 images, each 512×512
        assert freqs.sum() < total_px, (
            f"Expected sum < {total_px} (some pixels are ignore-255); "
            f"got sum={freqs.sum()}"
        )

    def test_expected_counts(self, tiny_split: Path) -> None:
        """
        Per image:
          rows 0-127   → regolith (0): 128 * 512 = 65 536 px
          rows 128-255 → rock     (1): 128 * 512 = 65 536 px
          rows 256-383 → sky      (2): 128 * 512 = 65 536 px
          rows 384-511 → ignore (255): excluded
        3 images → 3 × 65 536 = 196 608 px per class.
        """
        freqs    = class_pixel_frequencies(tiny_split)
        expected = 3 * 128 * 512
        np.testing.assert_array_equal(
            freqs, [expected, expected, expected],
            err_msg=f"Unexpected class frequencies: {freqs}",
        )

    def test_dtype_int64(self, tiny_split: Path) -> None:
        freqs = class_pixel_frequencies(tiny_split)
        assert freqs.dtype == np.int64
