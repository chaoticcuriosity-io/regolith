"""
generate_dataset.py — CLI entry point for synthetic dataset generation.

Builds the lunar stage, registers Replicator randomizations, and runs
the Replicator orchestrator to emit N labeled frames.

Usage
-----
  python generate_dataset.py --config configs/train_dr.yaml --n 10000 --out /workspace/datasets/train_dr

Output layout (per <out>/)
--------------------------
  rgb/              NNNNNN.png       — RGB renders
  semantic_segmentation/  NNNNNN.png  — per-pixel class labels (class map below)
  instance/         NNNNNN.png       — per-instance labels
  distance_to_camera/ NNNNNN.npy     — depth maps
  manifest.json     — class map, frame count, seeds, config path, generation timestamp

Class map
---------
  { "regolith": 0, "rock": 1, "sky": 2 }

Hardware note
-------------
  Run inside the regolith Docker container on the DGX Spark:
    ssh spark "docker exec <container> bash -lc \\
      'cd /workspace/regolith && python replicator/generate_dataset.py --config configs/train_dr.yaml --n 10000 --out /workspace/datasets/train_dr'"
  See docs/reports/02-domain-randomization.md ## Reproduce.
"""

from __future__ import annotations

import argparse
import sys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a labeled synthetic lunar dataset with NVIDIA Omniverse Replicator."
    )
    p.add_argument(
        "--config",
        required=True,
        metavar="YAML",
        help="Path to a YAML config in replicator/configs/ (e.g. configs/train_dr.yaml)",
    )
    p.add_argument(
        "--n",
        type=int,
        required=True,
        metavar="INT",
        help="Number of frames to generate",
    )
    p.add_argument(
        "--out",
        required=True,
        metavar="DIR",
        help="Output directory for the dataset",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # TODO (Task 2): implement
    #   1. Load cfg from args.config (yaml.safe_load)
    #   2. Call scene.build_lunar_stage.build_lunar_stage(seed=cfg.seed)
    #   3. Call replicator.randomizers.register(cfg)
    #   4. Configure Replicator writers (rgb, semantic_segmentation, instance, distance_to_camera)
    #   5. rep.orchestrator.run(args.n)
    #   6. Write manifest.json (class map, counts, seeds, config path, timestamp)
    raise NotImplementedError("generate_dataset: implement in Task 2 (Replicator SDG)")


if __name__ == "__main__":
    main()
