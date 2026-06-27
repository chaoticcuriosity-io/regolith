#!/usr/bin/env python3
"""
download_real.py — fetch the Artificial Lunar Rocky Landscape Dataset for sim-to-real evaluation.

Dataset
-------
  Kaggle: romainpessia/artificial-lunar-rocky-landscape-dataset
  Authors: Romain Pessia & Genya Ishigami, Space Robotics Group, Keio University

  Contains 9 766 photorealistic synthetic renders + paired RGB segmentation masks,
  plus ~36 real Chang'e 3 rover photographs — the genuine sim-to-real test set.

Usage
-----
  python eval/data/download_real.py --out /workspace/datasets/lunar_landscape

Credential / fallback priority
-------------------------------
  1. Kaggle API  (~/.kaggle/kaggle.json  OR  KAGGLE_USERNAME + KAGGLE_KEY env vars)
  2. HuggingFace datasets library (no known mirror as of 2026-06-27 — skipped with notice)
  3. Manual download instructions printed to stdout; script exits non-zero.

Idempotent: re-running when the dataset is already present is a no-op unless --force.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

KAGGLE_SLUG = "romainpessia/artificial-lunar-rocky-landscape-dataset"
KAGGLE_URL = f"https://www.kaggle.com/datasets/{KAGGLE_SLUG}"
ZIP_NAME = "artificial-lunar-rocky-landscape-dataset.zip"

# If this path exists inside --out the dataset is already extracted.
SENTINEL_SUBPATH = "images/render"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Download the Artificial Lunar Rocky Landscape Dataset (Kaggle) "
            "for sim-to-real evaluation."
        )
    )
    p.add_argument("--out", required=True, metavar="DIR",
                   help="Destination directory; dataset is extracted here.")
    p.add_argument("--force", action="store_true",
                   help="Re-download even if the dataset appears to be present.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Credential detection
# ---------------------------------------------------------------------------

def _kaggle_creds_available() -> bool:
    """Return True if Kaggle credentials can be located."""
    if (Path.home() / ".kaggle" / "kaggle.json").exists():
        return True
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return False


# ---------------------------------------------------------------------------
# Source 1: Kaggle API
# ---------------------------------------------------------------------------

def _try_kaggle(out_dir: Path) -> bool:
    """Attempt download via the kaggle CLI. Returns True on success."""
    if not _kaggle_creds_available():
        print("[kaggle] No credentials found (~/.kaggle/kaggle.json or "
              "KAGGLE_USERNAME/KAGGLE_KEY) — skipping.")
        return False

    if not shutil.which("kaggle"):
        print("[kaggle] 'kaggle' CLI not found. Install with: pip install kaggle")
        print("[kaggle] Skipping Kaggle source.")
        return False

    print(f"[kaggle] Credentials found. Downloading {KAGGLE_SLUG} ...")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Try with --unzip first (kaggle CLI >= 1.5)
    cmd = ["kaggle", "datasets", "download", "-d", KAGGLE_SLUG,
           "-p", str(out_dir), "--unzip"]
    result = subprocess.run(cmd, check=False)

    if result.returncode == 0:
        print(f"[kaggle] Dataset extracted to {out_dir}")
        return True

    # --unzip flag may not be available — fall back to manual unzip
    print("[kaggle] --unzip flag failed; retrying without it ...")
    cmd_plain = ["kaggle", "datasets", "download", "-d", KAGGLE_SLUG,
                 "-p", str(out_dir)]
    result2 = subprocess.run(cmd_plain, check=False)
    if result2.returncode != 0:
        print(f"[kaggle] Download failed (exit {result2.returncode})")
        return False

    zip_path = out_dir / ZIP_NAME
    if not zip_path.exists():
        # kaggle sometimes names the zip differently — find any zip
        zips = list(out_dir.glob("*.zip"))
        if zips:
            zip_path = zips[0]
        else:
            print("[kaggle] Download succeeded but no ZIP found — unexpected.")
            return False

    print(f"[kaggle] Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    zip_path.unlink(missing_ok=True)

    print(f"[kaggle] Dataset extracted to {out_dir}")
    return True


# ---------------------------------------------------------------------------
# Source 2: HuggingFace
# ---------------------------------------------------------------------------

def _try_huggingface(out_dir: Path) -> bool:
    """Attempt download via HuggingFace datasets. Returns True on success.

    NOTE: As of 2026-06-27 no HuggingFace mirror of this dataset is known.
    This is a documented fallback hook — update the HF_REPO constant if a
    mirror appears (e.g., search https://huggingface.co/datasets?search=lunar+landscape).
    """
    HF_REPO: str | None = None  # e.g. "username/artificial-lunar-landscape"

    if HF_REPO is None:
        print("[huggingface] No known HuggingFace mirror for this dataset — skipping.")
        return False

    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        print("[huggingface] 'datasets' library not installed. "
              "Install with: pip install datasets")
        return False

    print(f"[huggingface] Downloading from {HF_REPO} ...")
    ds = load_dataset(HF_REPO)
    # Save locally — adjust for actual dataset structure when a mirror exists
    ds.save_to_disk(str(out_dir))
    print(f"[huggingface] Dataset saved to {out_dir}")
    return True


# ---------------------------------------------------------------------------
# Fallback: manual instructions
# ---------------------------------------------------------------------------

def _print_manual_instructions(out_dir: Path) -> None:
    print()
    print("=" * 72)
    print("MANUAL DOWNLOAD REQUIRED")
    print("=" * 72)
    print()
    print("No automated download source is available (no Kaggle credentials,")
    print("no HuggingFace mirror). Follow these steps:")
    print()
    print("OPTION A — Kaggle API (recommended for DGX Spark):")
    print("  1. Create a free Kaggle account: https://www.kaggle.com/account/login")
    print("  2. Accept dataset terms:  " + KAGGLE_URL)
    print("  3. Download your API token from https://www.kaggle.com/settings")
    print("     (Account → API → Create New Token → saves ~/.kaggle/kaggle.json)")
    print("  4. On the Spark:")
    print("     pip install kaggle")
    print(f"     kaggle datasets download -d {KAGGLE_SLUG} \\")
    print(f"         -p {out_dir} --unzip")
    print()
    print("OPTION B — browser download, then scp to Spark:")
    print(f"  1. Download the ZIP from: {KAGGLE_URL}")
    print(f"  2. scp {ZIP_NAME} spark:/workspace/datasets/")
    print(f"  3. ssh spark 'unzip /workspace/datasets/{ZIP_NAME} -d {out_dir}'")
    print()
    print("Expected structure after extraction:")
    print(f"  {out_dir}/")
    print(f"  ├── images/")
    print(f"  │   ├── render/          ← 9 766 synthetic renders (720×480 PNG)")
    print(f"  │   ├── ground/          ← RGB segmentation masks (preferred)")
    print(f"  │   └── clean/           ← cleaned masks (fewer small-pixel artifacts)")
    print(f"  └── real_moon_images/    ← ~36 Chang'e 3 rover photos (true sim-to-real set)")
    print()
    print("Re-run this script after placing the dataset at the above path.")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    sentinel = out_dir / SENTINEL_SUBPATH

    # Idempotency check
    if not args.force and sentinel.exists():
        print(f"[download_real] Dataset already present (sentinel: {sentinel})")
        print("  Pass --force to re-download.")
        sys.exit(0)

    # Try sources in priority order
    for attempt in (_try_kaggle, _try_huggingface):
        if attempt(out_dir):
            print("[download_real] Done.")
            sys.exit(0)

    # Nothing worked — print manual instructions and exit non-zero
    _print_manual_instructions(out_dir)
    sys.exit(1)


if __name__ == "__main__":
    main()
