"""
render_predictions.py — cinematic RTX render with segmentation predictions overlaid.

Uses the NVIDIA Omniverse RTX renderer to produce a high-quality cinematic
sequence over the lunar stage (built by scene/build_lunar_stage.py).
Loads the trained model, runs inference per frame, and composites the
per-pixel segmentation mask onto the RTX render.

Output
------
  <out>/
    frames/       NNNNNN.png   — individual frames (RGB + overlay)
    render.mp4                 — assembled cinematic video

Usage
-----
  python render/render_predictions.py \\
    --checkpoint /workspace/checkpoints/segformer_b0_dr/best.pt \\
    --out /workspace/renders/v1 \\
    --fps 30 \\
    --duration-s 30

Video note
----------
  render.mp4 lives on the DGX Spark (not in git — too large).
  A compressed preview is published to GitHub Pages in docs/reports/assets/.
  See docs/reports/05-the-render.md for the full render walkthrough.

Hardware note
-------------
  NVIDIA DGX Spark (GB10 Grace Blackwell, aarch64, CUDA 13, sm_121,
  128 GB unified memory — nvidia-smi reports VRAM as N/A, which is expected)
  Run inside the regolith Docker container:
    ssh spark "docker exec <container> bash -lc 'python render/render_predictions.py ...'"
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cinematic RTX render with SegFormer segmentation overlay."
    )
    p.add_argument("--checkpoint", required=True, metavar="PT", help="Path to best.pt")
    p.add_argument("--out", required=True, metavar="DIR", help="Output directory")
    p.add_argument("--fps", type=int, default=30, help="Output video frame rate (default: 30)")
    p.add_argument("--duration-s", type=float, default=30.0, help="Render duration in seconds")
    p.add_argument("--seed", type=int, default=0, help="Scene seed for build_lunar_stage")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # TODO (Task 5): implement
    #   1. build_lunar_stage(seed=args.seed)
    #   2. Load model checkpoint
    #   3. Configure Omniverse RTX renderer (high SPP, path tracing)
    #   4. Camera fly-through path
    #   5. Per-frame: render → inference → composite overlay → save PNG
    #   6. Assemble frames to render.mp4 (ffmpeg)
    raise NotImplementedError("render_predictions: implement in Task 5 (render)")


if __name__ == "__main__":
    main()
