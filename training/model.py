"""
model.py — segmentation model factory for the regolith pipeline.

Default: SegFormer-B0 (nvidia/mit-b0 backbone, HuggingFace Transformers)
  - Smallest SegFormer variant; fast to fine-tune on 3-class lunar data.
  - Choice rationale: SegFormer-B0 has strong sim-to-real transfer from its
    ImageNet encoder; its hierarchical Mix Transformer captures multi-scale
    rock/regolith texture better than a ResNet backbone at this parameter
    budget. Low inference latency is also a plus for eventual onboard deployment.
  - CRITICAL: outputs logits at H/4 × W/4 resolution. Callers MUST upsample:
      F.interpolate(logits, size=(512,512), mode='bilinear', align_corners=False)
    before computing loss or argmax. (train.py handles this automatically.)
  - Swap to segformer_b2/b5 if val rock-IoU plateaus after 50 epochs.

Fallback: DeepLabV3-ResNet50 (torchvision, name="deeplabv3")
  - Use when HuggingFace is unavailable or the network is down on the Spark.
  - Returns full-resolution logits (output["out"]); no interpolation needed.
  - ImageNet-pretrained backbone; fresh 3-class ASPP head.

Class map: { regolith: 0, rock: 1, sky: 2 }
"""

from __future__ import annotations

import torch.nn as nn


# HuggingFace pretrained backbone IDs — ImageNet pretrained encoders only
# (not ADE20K-finetuned, so the decode head starts fresh for our 3 classes)
_HF_BACKBONE_IDS: dict[str, str] = {
    "segformer_b0": "nvidia/mit-b0",
    "segformer_b2": "nvidia/mit-b2",
    "segformer_b5": "nvidia/mit-b5",
}


def build_model(name: str = "segformer_b0", num_classes: int = 3) -> nn.Module:
    """Build and return a segmentation model with a fresh head for `num_classes`.

    Parameters
    ----------
    name : str
        "segformer_b0"  — HuggingFace SegFormer-B0, ImageNet backbone (default)
        "segformer_b2"  — HuggingFace SegFormer-B2 (larger; better if B0 plateaus)
        "segformer_b5"  — HuggingFace SegFormer-B5 (largest)
        "deeplabv3"     — torchvision DeepLabV3-ResNet50 (offline fallback)
    num_classes : int
        Number of output classes. Default: 3 (regolith, rock, sky).

    Returns
    -------
    nn.Module
        Model with ImageNet-pretrained backbone and a fresh `num_classes` head.
        Call ``model.to(device)`` after this function.

    Notes
    -----
    SegFormer variants output logits at H/4 × W/4 — interpolate in train.py
    before loss + argmax. DeepLab returns full-resolution output["out"].
    """
    if name in _HF_BACKBONE_IDS:
        return _build_segformer(name, num_classes)
    if name == "deeplabv3":
        return _build_deeplabv3(num_classes)
    raise ValueError(
        f"Unknown model name '{name}'. "
        f"Valid choices: {sorted(_HF_BACKBONE_IDS.keys()) + ['deeplabv3']}"
    )


# ---------------------------------------------------------------------------
# Internal builders — imports deferred so the module loads without these deps
# ---------------------------------------------------------------------------

def _build_segformer(name: str, num_classes: int) -> nn.Module:
    """Load MIT backbone from HuggingFace; attach a fresh decode head."""
    from transformers import SegformerForSemanticSegmentation, SegformerConfig

    backbone_id = _HF_BACKBONE_IDS[name]

    # Load backbone config and override the number of output labels
    config = SegformerConfig.from_pretrained(backbone_id)
    config.num_labels = num_classes
    config.id2label   = {0: "regolith", 1: "rock", 2: "sky"}
    config.label2id   = {"regolith": 0, "rock": 1, "sky": 2}

    # from_pretrained loads the MIT encoder weights; ignore_mismatched_sizes=True
    # lets the decode head be randomly initialized (its size changed with num_labels)
    model = SegformerForSemanticSegmentation.from_pretrained(
        backbone_id,
        config=config,
        ignore_mismatched_sizes=True,
    )
    return model


def _build_deeplabv3(num_classes: int) -> nn.Module:
    """torchvision DeepLabV3-ResNet50: ImageNet backbone, fresh 3-class head."""
    import torchvision
    from torchvision.models import ResNet50_Weights
    from torchvision.models.segmentation import deeplabv3_resnet50

    model = deeplabv3_resnet50(
        weights=None,                        # no pretrained segmentation head
        weights_backbone=ResNet50_Weights.DEFAULT,  # ImageNet backbone
        num_classes=num_classes,
    )
    return model
