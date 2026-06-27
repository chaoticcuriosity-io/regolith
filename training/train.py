"""
train.py — fine-tune SegFormer-B0 (or DeepLab) on the synthetic lunar dataset.

Usage
-----
  python training/train.py \\
    --train-split /workspace/datasets/train_dr \\
    --val-split   /workspace/datasets/test_photoreal \\
    --epochs 50   --out /workspace/checkpoints/segformer_b0_dr

  # With explicit options:
  python training/train.py \\
    --train-split /workspace/datasets/train_dr \\
    --val-split   /workspace/datasets/test_photoreal \\
    --epochs 50 --model segformer_b0 --lr 6e-5 --batch 8 --seed 0 \\
    --out /workspace/checkpoints/segformer_b0_dr

Key design decisions
--------------------
* SegFormer logits are at H/4×W/4 — interpolated to 512×512 BEFORE loss + argmax.
* Loss: class-weighted CrossEntropy with ignore_index=255 (ignore label passes
  through the dataset unchanged; the loss simply skips those pixels).
* Class weights = inverse pixel frequency (normalized to sum to num_classes).
  Rock (class 1) is typically rare; up-weighting prevents it being ignored.
* Model selection: highest val rock-IoU saves best.pt (not mIoU). Rock is the
  hazard class — that is what matters for the downstream safety system.
* Cosine LR decay (T_max = epochs, eta_min = 1e-7) + AdamW.
* Per-epoch confusion counts are accumulated incrementally (not stored as full
  prediction arrays) to keep memory usage constant regardless of dataset size.

Hardware note (DGX Spark)
--------------------------
  ssh spark "docker exec <container> bash -lc \\
    'cd /workspace/regolith && python training/train.py \\
      --train-split /workspace/datasets/train_dr \\
      --val-split   /workspace/datasets/test_photoreal \\
      --epochs 50   --out /workspace/checkpoints/segformer_b0_dr'"
  See docs/reports/03-training.md § Reproduce.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

# Ensure repo root is on sys.path when this script is run directly
# (python training/train.py adds training/ to sys.path, not the repo root)
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training.dataset import SyntheticLunarDataset, class_pixel_frequencies
from training.model import build_model
from training import metrics as M


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune SegFormer (or DeepLab) on the synthetic lunar dataset."
    )
    p.add_argument("--train-split", required=True, metavar="DIR",
                   help="Training split directory (rgb/ + mask/ subdirs)")
    p.add_argument("--val-split",   required=True, metavar="DIR",
                   help="Validation split directory (rgb/ + mask/ subdirs)")
    p.add_argument("--epochs",      type=int,   default=50,
                   help="Number of training epochs (default: 50)")
    p.add_argument("--out",         required=True, metavar="DIR",
                   help="Output directory for checkpoints + metrics files")
    p.add_argument("--model",       default="segformer_b0",
                   choices=["segformer_b0", "segformer_b2", "segformer_b5", "deeplabv3"],
                   help="Model variant (default: segformer_b0)")
    p.add_argument("--lr",          type=float, default=6e-5,
                   help="Peak learning rate, AdamW (default: 6e-5, SegFormer paper)")
    p.add_argument("--batch",       type=int,   default=8,
                   help="Batch size per forward/backward step (default: 8)")
    p.add_argument("--seed",        type=int,   default=0,
                   help="Global random seed for reproducibility (default: 0)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic conv (may be slower; acceptable for research training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def compute_class_weights(train_split: str | Path, device: torch.device) -> torch.Tensor:
    """Inverse-frequency class weights, normalized so they sum to num_classes=3.

    Steps:
      1. Count pixels per class via class_pixel_frequencies (ignores 255).
      2. Replace any zero count with 1 to avoid division by zero.
      3. weight[c] = (1 / count[c]) / sum(1/count) * 3
         → rare classes get higher weight; total weight budget is preserved.

    Returns a float32 tensor on `device` ready for nn.CrossEntropyLoss.
    """
    freqs = class_pixel_frequencies(train_split).astype(np.float64)
    freqs = np.where(freqs == 0, 1.0, freqs)   # guard against absent classes
    inv   = 1.0 / freqs
    weights = inv / inv.sum() * len(freqs)      # normalize; sum = num_classes
    return torch.tensor(weights, dtype=torch.float32, device=device)


def get_logits(model: nn.Module, images: torch.Tensor, is_segformer: bool) -> torch.Tensor:
    """Forward pass; return logits tensor [B, C, H, W] ready for loss."""
    if is_segformer:
        return model(pixel_values=images).logits   # [B, C, H/4, W/4]
    return model(images)["out"]                    # [B, C, H, W] full-res


def upsample(logits: torch.Tensor, is_segformer: bool) -> torch.Tensor:
    """Upsample SegFormer's H/4×W/4 logits to 512×512; DeepLab is already full-res."""
    if is_segformer:
        return F.interpolate(
            logits, size=(512, 512), mode="bilinear", align_corners=False
        )
    return logits


# ---------------------------------------------------------------------------
# Incremental confusion counter (avoids storing full-epoch pred arrays)
# ---------------------------------------------------------------------------

class _ConfusionAccum:
    """Accumulates (tp, fp, fn) per class and total loss across batches."""

    def __init__(self, num_classes: int = 3) -> None:
        self.num_classes = num_classes
        self.reset()

    def reset(self) -> None:
        self.tp   = np.zeros(self.num_classes, dtype=np.int64)
        self.fp   = np.zeros(self.num_classes, dtype=np.int64)
        self.fn   = np.zeros(self.num_classes, dtype=np.int64)
        self._loss_sum = 0.0
        self._n        = 0

    def update(self, pred: np.ndarray, label: np.ndarray, loss_val: float, n: int) -> None:
        for cls in range(self.num_classes):
            tp, fp, fn = M.confusion_counts(pred, label, cls)
            self.tp[cls] += tp
            self.fp[cls] += fp
            self.fn[cls] += fn
        self._loss_sum += loss_val * n
        self._n        += n

    def compute(self) -> dict:
        ious: list[float] = []
        for cls in range(self.num_classes):
            union = int(self.tp[cls] + self.fp[cls] + self.fn[cls])
            ious.append(float(self.tp[cls] / union) if union > 0 else float("nan"))
        miou     = float(np.nanmean(ious))
        rock_iou = ious[1]
        avg_loss = self._loss_sum / max(1, self._n)
        return {"loss": avg_loss, "miou": miou, "rock_iou": rock_iou}


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    is_segformer: bool,
    train: bool = True,
) -> dict:
    """Run one training or evaluation epoch.

    Returns dict with keys: loss, miou, rock_iou.
    """
    model.train(train)
    accum = _ConfusionAccum(num_classes=3)
    ctx   = torch.enable_grad() if train else torch.no_grad()

    with ctx:
        for images, masks in loader:
            images = images.to(device)           # [B, 3, 512, 512]
            masks  = masks.to(device)            # [B, 512, 512] LongTensor

            # Forward
            logits = get_logits(model, images, is_segformer)
            # SegFormer: logits are [B, C, 128, 128] → upsample to [B, C, 512, 512]
            logits = upsample(logits, is_segformer)

            # Loss (ignore_index=255 skips background pixels)
            loss = criterion(logits, masks)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Metrics — argmax over class dim; flatten to 1D for confusion_counts
            pred_np  = logits.detach().argmax(dim=1).cpu().numpy().reshape(-1).astype(np.int64)
            label_np = masks.cpu().numpy().reshape(-1).astype(np.int64)
            accum.update(pred_np, label_np, loss.item(), images.size(0))

    return accum.compute()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[regolith/train] device={device}  model={args.model}  "
          f"epochs={args.epochs}  lr={args.lr}  batch={args.batch}  seed={args.seed}")

    # ---- Datasets & loaders -----------------------------------------------
    train_ds = SyntheticLunarDataset(args.train_split, augment=True)
    val_ds   = SyntheticLunarDataset(args.val_split,   augment=False)
    print(f"[regolith/train] train={len(train_ds)} samples  val={len(val_ds)} samples")

    num_workers = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=num_workers, pin_memory=device.type == "cuda",
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=num_workers, pin_memory=device.type == "cuda",
    )

    # ---- Model -------------------------------------------------------------
    is_segformer = args.model != "deeplabv3"
    model = build_model(name=args.model, num_classes=3).to(device)

    # ---- Class weights from training split ---------------------------------
    class_weights = compute_class_weights(args.train_split, device)
    print(f"[regolith/train] class weights (regolith, rock, sky): "
          f"{class_weights.cpu().tolist()}")

    # ---- Loss / optimizer / scheduler -------------------------------------
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)

    # ---- Training loop ----------------------------------------------------
    best_rock_iou  = -1.0
    metrics_path   = out_dir / "metrics.jsonl"
    best_ckpt_path = out_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        # Train
        train_stats = run_epoch(
            model, train_loader, criterion, optimizer,
            device, is_segformer, train=True,
        )
        scheduler.step()

        # Validate
        val_stats = run_epoch(
            model, val_loader, criterion, None,
            device, is_segformer, train=False,
        )

        lr_now = scheduler.get_last_lr()[0]

        # Append to metrics.jsonl (one JSON object per line)
        row: dict = {
            "epoch":          epoch,
            "lr":             lr_now,
            "train_loss":     train_stats["loss"],
            "train_miou":     train_stats["miou"],
            "train_rock_iou": train_stats["rock_iou"],
            "val_loss":       val_stats["loss"],
            "val_miou":       val_stats["miou"],
            "val_rock_iou":   val_stats["rock_iou"],
        }
        with open(metrics_path, "a") as f:
            f.write(json.dumps(row) + "\n")

        # Checkpoint if this is the best val rock-IoU so far
        improved = (
            not np.isnan(val_stats["rock_iou"])
            and val_stats["rock_iou"] > best_rock_iou
        )
        if improved:
            best_rock_iou = val_stats["rock_iou"]
            torch.save(
                {
                    "epoch":        epoch,
                    "model_name":   args.model,
                    "state_dict":   model.state_dict(),
                    "val_rock_iou": best_rock_iou,
                    "val_miou":     val_stats["miou"],
                    "args":         vars(args),
                },
                best_ckpt_path,
            )
            marker = "★ best"
        else:
            marker = ""

        print(
            f"  epoch {epoch:3d}/{args.epochs}"
            f"  train_loss={train_stats['loss']:.4f}"
            f"  val_rock_iou={val_stats['rock_iou']:.4f}"
            f"  val_miou={val_stats['miou']:.4f}"
            f"  {marker}"
        )

    # ---- Final summary -----------------------------------------------------
    summary = {
        "model":          args.model,
        "epochs_trained": args.epochs,
        "best_val_rock_iou": best_rock_iou,
        "best_ckpt":      str(best_ckpt_path),
        "train_split":    str(args.train_split),
        "val_split":      str(args.val_split),
        "seed":           args.seed,
        "lr":             args.lr,
        "batch":          args.batch,
        "class_weights":  class_weights.cpu().tolist(),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[regolith/train] Done.  Best val rock-IoU: {best_rock_iou:.4f}")
    print(f"[regolith/train] Outputs: {out_dir}")


if __name__ == "__main__":
    main()
