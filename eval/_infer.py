"""
_infer.py — shared inference helpers for the regolith evaluation scripts.

Both eval_synth.py (synthetic hold-out) and eval_real.py (real lunar photos)
load a trained checkpoint, run a forward pass, and colorize the prediction with
the project's canonical palette. That logic lives here so the two scripts agree
exactly — same normalization, same upsample, same colors.

Canonical class map (matches training/dataset.py and training/metrics.py):
  { regolith: 0, rock: 1, sky: 2 }, 255 = ignore

CRITICAL inference details (must match training/train.py):
  * Images are ImageNet-normalized — the same mean/std SegFormer was trained on.
  * SegFormer logits come out at H/4 x W/4; we bilinearly upsample to the input
    size BEFORE argmax (identical to train.py's upsample()).
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Repo root on sys.path so `from training.model import build_model` works when
# this module is imported from eval/ (the scripts add the repo root themselves).
import sys
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training.model import build_model  # noqa: E402

# ImageNet normalization — identical to training/dataset.py.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Canonical palette — used by both eval_synth.py and eval_real.py so real-image
# overlays match the synthetic ablation figures already in the report.
#   regolith = tan, rock = RED (the hazard class), sky = blue, ignore = black
PALETTE: dict[int, tuple[int, int, int]] = {
    0: (170, 140, 110),
    1: (220, 45, 40),
    2: (70, 120, 205),
    255: (0, 0, 0),
}
CLASS_NAMES = ["regolith", "rock", "sky"]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint: str | Path, device: torch.device) -> Tuple[torch.nn.Module, dict]:
    """Load a best.pt checkpoint and return (eval-mode model, checkpoint dict).

    The checkpoint stores ``model_name`` (e.g. "segformer_b0") and ``state_dict``
    (written by training/train.py). We rebuild the architecture and load weights.
    """
    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    model_name = ckpt.get("model_name", "segformer_b0")
    model = build_model(model_name, num_classes=3).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def is_segformer(model: torch.nn.Module) -> bool:
    """True for HuggingFace SegFormer models (logits at H/4, need upsampling)."""
    return hasattr(model, "config") and "Segformer" in type(model).__name__


# ---------------------------------------------------------------------------
# Preprocess / predict
# ---------------------------------------------------------------------------

def preprocess(rgb: np.ndarray) -> torch.Tensor:
    """RGB uint8 HxWx3 -> normalized float tensor [1, 3, H, W] (ImageNet stats)."""
    x = rgb.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = np.transpose(x, (2, 0, 1))
    return torch.from_numpy(x).unsqueeze(0)


@torch.no_grad()
def predict(model: torch.nn.Module, rgb: np.ndarray, device: torch.device) -> np.ndarray:
    """Run the model on an HxWx3 uint8 RGB image; return an HxW int64 class map.

    Mirrors train.py exactly: SegFormer logits (H/4) are bilinearly upsampled to
    the input resolution before argmax; DeepLab is already full-resolution.
    """
    h, w = rgb.shape[:2]
    x = preprocess(rgb).to(device)
    if is_segformer(model):
        logits = model(pixel_values=x).logits
        logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
    else:
        logits = model(x)["out"]
    return logits.argmax(1)[0].cpu().numpy().astype(np.int64)


# ---------------------------------------------------------------------------
# Image fitting + colorizing
# ---------------------------------------------------------------------------

def fit_square(rgb: np.ndarray, size: int = 512, mode: str = "letterbox") -> np.ndarray:
    """Resize an arbitrary RGB image to size x size.

    mode="letterbox": preserve aspect ratio, pad the short side with black.
        (For a square input this is a plain resize — no bars. Apollo Hasselblad
        surface frames are natively square, so most inputs hit this no-bar path.)
    mode="cover": preserve aspect ratio, center-crop the long side, then resize
        (no black bars, but crops content).
    """
    img = Image.fromarray(rgb)
    w, h = img.size

    if mode == "cover":
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        img = img.crop((left, top, left + side, top + side))
        img = img.resize((size, size), Image.BILINEAR)
        return np.array(img)

    # letterbox
    scale = size / max(w, h)
    new_w, new_h = round(w * scale), round(h * scale)
    img = img.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(img, ((size - new_w) // 2, (size - new_h) // 2))
    return np.array(canvas)


def colorize(mask: np.ndarray) -> np.ndarray:
    """Class-index map HxW -> RGB uint8 HxWx3 using the canonical palette."""
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for k, c in PALETTE.items():
        out[mask == k] = c
    return out


def overlay(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Alpha-blend the colorized prediction over the RGB image.

    A blend (not a solid fill) is deliberate for real photos: you can see the
    actual rock/regolith texture *under* the colored mask and judge for yourself
    whether the red lands on a real rock.
    """
    color = colorize(mask).astype(np.float32)
    base = rgb.astype(np.float32)
    blend = (1.0 - alpha) * base + alpha * color
    return np.clip(blend, 0, 255).astype(np.uint8)


def class_fractions(mask: np.ndarray) -> dict[str, float]:
    """Fraction of pixels assigned to each class (over the full image)."""
    total = mask.size
    return {name: float((mask == i).sum()) / total for i, name in enumerate(CLASS_NAMES)}
