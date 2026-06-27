"""
dataset.py — PyTorch Dataset for the synthetic lunar segmentation dataset.

The dataset directory is produced by replicator/generate_dataset.py.
Expected layout:
  <root>/
    rgb/                    NNNNNN.png
    semantic_segmentation/  NNNNNN.png  (single-channel, class index per pixel)
    manifest.json           class map + metadata

Class map
---------
  { regolith: 0, rock: 1, sky: 2 }
  rock (class 1) is the hazard class of interest.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

# TODO (Task 3): add actual imports
# import torch
# from torch import FloatTensor, LongTensor
# from torch.utils.data import Dataset
# from torchvision import transforms
# from PIL import Image


# from torch.utils.data import Dataset
# class SyntheticLunarDataset(Dataset):  # TODO(Task 3): uncomment inheritance; fill __len__/__getitem__
class SyntheticLunarDataset:
    """PyTorch Dataset yielding (image, mask) pairs from a synthetic lunar dataset.

    Parameters
    ----------
    root : str | Path
        Root directory of the dataset (contains rgb/, semantic_segmentation/,
        manifest.json).
    split : str
        "train" or "val" — filters frames according to the split index in
        manifest.json (or a companion split_manifest.json).
    image_size : tuple[int, int]
        (H, W) to resize images and masks to. Default: (720, 1280).
    augment : bool
        Whether to apply random horizontal flip + color jitter during training.

    Yields
    ------
    image : FloatTensor[3, H, W]
        Normalized RGB image (ImageNet mean/std).
    mask : LongTensor[H, W]
        Per-pixel class index: 0=regolith, 1=rock, 2=sky.
    """

    # TODO (Task 3): implement __init__, __len__, __getitem__
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        image_size: Tuple[int, int] = (720, 1280),
        augment: bool = False,
    ) -> None:
        raise NotImplementedError("SyntheticLunarDataset: implement in Task 3 (training)")

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int):  # -> Tuple[FloatTensor, LongTensor]
        raise NotImplementedError
