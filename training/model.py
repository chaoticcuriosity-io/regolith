"""
model.py — segmentation model factory.

Wraps HuggingFace Transformers SegFormer. The default (segformer_b0) is
the smallest SegFormer variant — fast to fine-tune, reasonable on the
lunar domain. Swap to segformer_b2 / b5 if rock-IoU plateaus.

Class map: { regolith: 0, rock: 1, sky: 2 }
"""

from __future__ import annotations

# TODO (Task 3): add actual imports
# import torch.nn as nn
# from transformers import SegformerForSemanticSegmentation


# HuggingFace model IDs for each supported name
_MODEL_IDS = {
    "segformer_b0": "nvidia/segformer-b0-finetuned-ade-512-512",
    "segformer_b2": "nvidia/segformer-b2-finetuned-ade-512-512",
    "segformer_b5": "nvidia/segformer-b5-finetuned-ade-512-512",
}


def build_model(name: str = "segformer_b0", num_classes: int = 3) -> "nn.Module":
    """Build and return a SegFormer segmentation model.

    Loads the pretrained ImageNet / ADE20k backbone and replaces the
    segmentation head with a new head for `num_classes` output channels.
    The head is randomly initialized; the backbone is pretrained.

    Parameters
    ----------
    name : str
        One of "segformer_b0", "segformer_b2", "segformer_b5".
        Default: "segformer_b0" (fastest, sufficient for 3-class lunar).
    num_classes : int
        Number of output classes. Default: 3 (regolith, rock, sky).

    Returns
    -------
    nn.Module
        SegFormerForSemanticSegmentation with the new head attached.
        Move to device after calling: model.to(device).

    Notes
    -----
    See docs/reports/03-training.md for fine-tuning details and val rock-IoU results.
    """
    # TODO (Task 3): implement
    raise NotImplementedError("build_model: implement in Task 3 (training)")
