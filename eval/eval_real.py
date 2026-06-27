"""
eval_real.py — evaluate the trained model on real lunar surface imagery.

Produces:
  - metrics.json   : mIoU, per-class IoU on real images
  - overlays/      : qualitative PNG overlays on real images

Usage
-----
  python eval/eval_real.py \\
    --checkpoint /workspace/checkpoints/segformer_b0_dr/best.pt \\
    --real-data /workspace/datasets/real_lunar \\
    --out /workspace/eval_results/real

The sim-to-real gap = synth rock-IoU (eval_synth.py) minus real rock-IoU (this script).

Data note
---------
  Real images are downloaded/prepared by eval/data/download_real.py.
  The real_lunar/ directory lives on the DGX Spark (not in git).

See docs/reports/04-sim-to-real.md for the full gap analysis.
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate SegFormer on real lunar imagery (sim-to-real gap measurement)."
    )
    p.add_argument("--checkpoint", required=True, metavar="PT", help="Path to best.pt")
    p.add_argument("--real-data", required=True, metavar="DIR", help="Real lunar images directory")
    p.add_argument("--out", required=True, metavar="DIR", help="Output directory")
    p.add_argument("--overlay-every", type=int, default=5, help="Save overlay every N images")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # TODO (Task 4): implement
    raise NotImplementedError("eval_real: implement in Task 4 (evaluation)")


if __name__ == "__main__":
    main()
