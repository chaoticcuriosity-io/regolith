"""
dataset.py — PyTorch Dataset for the synthetic lunar segmentation dataset.

Expected split layout (produced by the dataset generator):
  <split_dir>/
    rgb/
      rgb_XXXXX.png        512×512 RGB (DLSS-upscaled)
    mask/
      mask_XXXXX.png       512×512 single-channel uint8
                           class ids: 0=regolith, 1=rock, 2=sky
                           255=ignore/background (never equals a class id)
    manifest.json
    frames.jsonl

Class map: { regolith: 0, rock: 1, sky: 2 }
rock (class 1) is the hazard class of interest.
255 is the ignore label — passed through unchanged; CrossEntropyLoss ignores it
via ignore_index=255.

Pairs are matched by sorted filename index (rgb_XXXXX ↔ mask_XXXXX).
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch import FloatTensor, LongTensor
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


# ImageNet normalization constants (same as SegFormer / most HuggingFace models)
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


class SyntheticLunarDataset(Dataset):
    """PyTorch Dataset yielding (image, mask) pairs from a synthetic lunar split.

    Parameters
    ----------
    split_dir : str | Path
        Root directory of the split (contains rgb/ and mask/ subdirectories).
    augment : bool
        If True, apply random horizontal flip during __getitem__.
        Kept minimal intentionally — the synthetic-to-real domain gap is the
        main challenge; heavy augmentation risks distorting the rock geometry.

    Yields (per __getitem__)
    -----------------------
    image : FloatTensor[3, 512, 512]
        RGB image normalized with ImageNet mean/std.
    mask : LongTensor[512, 512]
        Per-pixel class index: 0=regolith, 1=rock, 2=sky, 255=ignore.
    """

    def __init__(
        self,
        split_dir: str | Path,
        augment: bool = False,
    ) -> None:
        self.split_dir = Path(split_dir)
        self.augment   = augment

        rgb_dir  = self.split_dir / "rgb"
        mask_dir = self.split_dir / "mask"

        if not rgb_dir.is_dir():
            raise FileNotFoundError(f"rgb/ subdirectory not found in {split_dir}")
        if not mask_dir.is_dir():
            raise FileNotFoundError(f"mask/ subdirectory not found in {split_dir}")

        rgb_files  = sorted(rgb_dir.glob("rgb_*.png"))
        mask_files = sorted(mask_dir.glob("mask_*.png"))

        if len(rgb_files) == 0:
            raise ValueError(f"No rgb_*.png files found in {rgb_dir}")
        if len(rgb_files) != len(mask_files):
            raise ValueError(
                f"rgb/mask count mismatch in {split_dir}: "
                f"{len(rgb_files)} rgb vs {len(mask_files)} mask files"
            )

        self._rgb_files  = rgb_files
        self._mask_files = mask_files

        # Always-applied image transforms
        self._to_tensor = transforms.ToTensor()  # PIL → [C,H,W] float32 in [0,1]
        self._normalize = transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)

    def __len__(self) -> int:
        return len(self._rgb_files)

    def __getitem__(self, idx: int) -> Tuple[FloatTensor, LongTensor]:
        img  = Image.open(self._rgb_files[idx]).convert("RGB")
        mask = Image.open(self._mask_files[idx])  # single-channel uint8

        # Light augmentation: horizontal flip only (preserves rock/sky geometry)
        if self.augment and torch.rand(1).item() > 0.5:
            img  = transforms.functional.hflip(img)
            mask = transforms.functional.hflip(mask)

        # Image: PIL RGB → [3, H, W] float32 in [0,1] → ImageNet normalized
        image_tensor: FloatTensor = self._normalize(self._to_tensor(img))

        # Mask: PIL L → numpy int64 → LongTensor (255 preserved for ignore_index)
        mask_np = np.array(mask, dtype=np.int64)
        mask_tensor: LongTensor = torch.from_numpy(mask_np).long()

        return image_tensor, mask_tensor


# ---------------------------------------------------------------------------
# Class-weight helper
# ---------------------------------------------------------------------------

def class_pixel_frequencies(split_dir: str | Path) -> np.ndarray:
    """Count pixels per semantic class across an entire split, ignoring 255.

    Used to compute inverse-frequency class weights for CrossEntropyLoss so
    the rare rock class (class 1) is not overwhelmed by the dominant regolith.

    Parameters
    ----------
    split_dir : str | Path
        Root directory of the split (contains mask/).

    Returns
    -------
    np.ndarray, shape (3,), dtype int64
        counts[c] = total pixel count for class c across all mask files.
        Order: [regolith, rock, sky] = [0, 1, 2].
        Pixels with value 255 (ignore/background) are excluded.
    """
    split_dir = Path(split_dir)
    mask_dir  = split_dir / "mask"

    counts = np.zeros(3, dtype=np.int64)
    for mask_path in sorted(mask_dir.glob("mask_*.png")):
        mask = np.array(Image.open(mask_path), dtype=np.int64)
        for cls in range(3):
            counts[cls] += int((mask == cls).sum())

    return counts
