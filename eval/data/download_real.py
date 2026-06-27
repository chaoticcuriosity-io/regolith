"""
download_real.py — download and prepare real lunar surface imagery for evaluation.

Sources (to be confirmed in Task 4)
-------------------------------------
  - Apollo surface photography (NASA archive, public domain)
  - Lunar Reconnaissance Orbiter LROC NAC surface images (NASA PDS)
  - Possibly: Chang'e mission imagery (where available with permissive licensing)

Output
------
  <out>/
    images/   *.jpg or *.png   — real lunar surface images
    masks/    *.png            — manually annotated segmentation masks
                                 (regolith=0, rock=1, sky=2)
                                 NOTE: mask annotation is the bottleneck;
                                 start with a small set (~200 images).
    metadata.json             — source URL, license, annotation author

Usage
-----
  python eval/data/download_real.py --out /workspace/datasets/real_lunar

Data note
---------
  Real images + masks live on the DGX Spark (not in git).
  See docs/reports/04-sim-to-real.md for dataset details and annotation notes.
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download and prepare real lunar imagery for sim-to-real evaluation."
    )
    p.add_argument("--out", required=True, metavar="DIR", help="Output directory")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # TODO (Task 4): implement download + basic preprocessing
    raise NotImplementedError("download_real: implement in Task 4 (evaluation)")


if __name__ == "__main__":
    main()
