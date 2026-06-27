"""
train.py — fine-tune SegFormer on the synthetic lunar dataset.

Usage
-----
  python training/train.py \\
    --train-split /workspace/datasets/train_dr \\
    --val-split   /workspace/datasets/val_dr \\
    --epochs 50 \\
    --out /workspace/checkpoints/segformer_b0_dr

What it does
------------
1. Loads train + val SyntheticLunarDataset.
2. Builds a SegFormer-B0 (build_model).
3. Trains with class-weighted cross-entropy
   (rock class up-weighted due to pixel imbalance).
4. Validates after each epoch; saves the best checkpoint on val rock-IoU.
5. Writes a metrics.json to <out>/ with per-epoch val mIoU and rock-IoU.

Hardware note
-------------
  Run inside the regolith Docker container on the DGX Spark:
    ssh spark "docker exec <container> bash -lc \\
      'cd /workspace/regolith && python training/train.py \\
        --train-split /workspace/datasets/train_dr \\
        --val-split /workspace/datasets/val_dr \\
        --epochs 50 --out /workspace/checkpoints/segformer_b0_dr'"
  See docs/reports/03-training.md ## Reproduce.
"""

from __future__ import annotations

import argparse
import sys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune SegFormer on the synthetic lunar segmentation dataset."
    )
    p.add_argument(
        "--train-split",
        required=True,
        metavar="DIR",
        help="Path to the training split directory (output of generate_dataset.py)",
    )
    p.add_argument(
        "--val-split",
        required=True,
        metavar="DIR",
        help="Path to the validation split directory",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=50,
        metavar="INT",
        help="Number of training epochs (default: 50)",
    )
    p.add_argument(
        "--out",
        required=True,
        metavar="DIR",
        help="Output directory for checkpoints and metrics.json",
    )
    p.add_argument(
        "--model",
        default="segformer_b0",
        choices=["segformer_b0", "segformer_b2", "segformer_b5"],
        help="SegFormer variant (default: segformer_b0)",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=6e-5,
        help="Learning rate (default: 6e-5, SegFormer paper recommendation)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size per GPU step (default: 8)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # TODO (Task 3): implement
    #   1. Build datasets (dataset.SyntheticLunarDataset)
    #   2. Build model (model.build_model)
    #   3. Compute class weights from train split class frequencies
    #   4. Training loop: class-weighted CE, AdamW, cosine LR schedule
    #   5. Val loop: compute mIoU and rock-IoU (metrics.mean_iou, metrics.class_iou)
    #   6. Save best checkpoint (highest val rock-IoU)
    #   7. Write metrics.json
    raise NotImplementedError("train: implement in Task 3 (training)")


if __name__ == "__main__":
    main()
