"""
eval_synth.py — evaluate the trained model on the synthetic hold-out test split.

Produces:
  - metrics.json   : mIoU, per-class IoU (regolith/rock/sky), frame count
  - overlays/      : qualitative PNG overlays (predicted mask on RGB image)
                     sampled every N frames

Usage
-----
  python eval/eval_synth.py \\
    --checkpoint /workspace/checkpoints/segformer_b0_dr/best.pt \\
    --test-split /workspace/datasets/test_photoreal \\
    --out /workspace/eval_results/synth

The rock-IoU reported here is the "ceiling" — synthetic-to-synthetic transfer.
Compare it to eval_real.py rock-IoU to measure the sim-to-real gap.

See docs/reports/04-sim-to-real.md for the full analysis.
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate SegFormer on the synthetic lunar test split."
    )
    p.add_argument("--checkpoint", required=True, metavar="PT", help="Path to best.pt")
    p.add_argument("--test-split", required=True, metavar="DIR", help="Test dataset directory")
    p.add_argument("--out", required=True, metavar="DIR", help="Output directory")
    p.add_argument("--overlay-every", type=int, default=100, help="Save overlay every N frames")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # TODO (Task 4): implement
    raise NotImplementedError("eval_synth: implement in Task 4 (evaluation)")


if __name__ == "__main__":
    main()
