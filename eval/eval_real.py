"""
eval_real.py — run trained checkpoints on REAL lunar surface photographs.

This is the genuine sim-to-real test. The model was trained only on synthetic
renders (chapters 02-03); here we point it at real Apollo / Surveyor surface
photos (NASA, public domain — see eval/real_images/sources.md) and look at what
it predicts. There is NO ground truth for these images, so the output is
QUALITATIVE: colored overlays, not an IoU. (Fabricating an IoU on unlabeled real
images would be dishonest — don't.)

Two-model comparison
--------------------
Pass --compare-checkpoint to render the money panel for each photo:

    [ real RGB | no-DR prediction overlay | DR-1500 prediction overlay ]

with the canonical palette (regolith=tan, rock=RED, sky=blue). This shows on
real pixels whether domain randomization transfers the way it did on synthetic.

Produces (in --out):
  - real-<id>.png        one comparison panel per input photo
  - contactsheet.png     all panels stacked into one sheet
  - predictions.json     per-image predicted class fractions for each model

Usage
-----
  python eval/eval_real.py \\
    --checkpoint         /workspace/regolith/outputs/runs/dr_1500/best.pt \\
    --compare-checkpoint /workspace/regolith/outputs/runs/nodr_750/best.pt \\
    --image-dir /workspace/regolith/eval/real_images \\
    --out /workspace/regolith/outputs/eval_real

Single-model mode (no --compare-checkpoint) renders [ RGB | prediction ] panels.
See docs/reports/04-sim-to-real.md for the gap assessment.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval._infer import (  # noqa: E402
    class_fractions, fit_square, load_model, overlay, predict,
)

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run trained checkpoint(s) on real lunar photos (qualitative)."
    )
    p.add_argument("--checkpoint", required=True, metavar="PT",
                   help="Primary checkpoint (rightmost overlay). Use the DR model.")
    p.add_argument("--image-dir", required=True, metavar="DIR",
                   help="Directory of real lunar photos (png/jpg/...).")
    p.add_argument("--out", required=True, metavar="DIR", help="Output directory")
    p.add_argument("--compare-checkpoint", metavar="PT", default=None,
                   help="Optional second checkpoint (middle overlay). Use no-DR.")
    p.add_argument("--label", default="DR-1500",
                   help="Label for the primary checkpoint (default: DR-1500)")
    p.add_argument("--compare-label", default="no-DR",
                   help="Label for the compare checkpoint (default: no-DR)")
    p.add_argument("--size", type=int, default=512, help="Square model input size")
    p.add_argument("--fit", choices=["letterbox", "cover"], default="letterbox",
                   help="How to make non-square photos square (default: letterbox)")
    p.add_argument("--alpha", type=float, default=0.45, help="Overlay opacity")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_primary, ck_p = load_model(args.checkpoint, device)
    print(f"[eval_real] device={device}  primary={ck_p.get('model_name')} "
          f"({args.label})  val_rock_iou(stored)={ck_p.get('val_rock_iou')}")

    model_compare = None
    if args.compare_checkpoint:
        model_compare, ck_c = load_model(args.compare_checkpoint, device)
        print(f"[eval_real] compare={ck_c.get('model_name')} ({args.compare_label}) "
              f"val_rock_iou(stored)={ck_c.get('val_rock_iou')}")

    images = sorted(p for p in Path(args.image_dir).iterdir()
                    if p.suffix.lower() in _IMAGE_EXTS)
    if not images:
        raise SystemExit(f"No images found in {args.image_dir}")
    print(f"[eval_real] {len(images)} real photo(s)")

    panel_paths: list[Path] = []
    predictions: dict[str, dict] = {}

    for img_path in images:
        stem = img_path.stem
        rgb_full = np.array(Image.open(img_path).convert("RGB"))
        rgb = fit_square(rgb_full, size=args.size, mode=args.fit)

        pred_p = predict(model_primary, rgb, device)
        ov_p = overlay(rgb, pred_p, alpha=args.alpha)

        cols = [("real photo", rgb)]
        rec: dict = {"primary_frac": class_fractions(pred_p)}
        if model_compare is not None:
            pred_c = predict(model_compare, rgb, device)
            ov_c = overlay(rgb, pred_c, alpha=args.alpha)
            cols.append((f"{args.compare_label} prediction", ov_c))
            rec["compare_frac"] = class_fractions(pred_c)
        cols.append((f"{args.label} prediction", ov_p))
        predictions[stem] = rec

        n = len(cols)
        fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.9))
        for ax, (title, im) in zip(axes, cols):
            ax.imshow(im)
            ax.set_title(title, fontsize=11)
            ax.axis("off")
        fig.suptitle(
            f"{stem}    palette: regolith=tan  rock=red  sky=blue   "
            f"(real NASA photo — no ground truth, qualitative)",
            fontsize=10.5, y=1.02,
        )
        fig.tight_layout()
        panel_path = out_dir / f"real-{stem}.png"
        fig.savefig(panel_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        panel_paths.append(panel_path)
        rock = rec["primary_frac"]["rock"] * 100
        print(f"[eval_real] {stem:28s} {args.label} rock={rock:4.1f}%  -> {panel_path.name}")

    # Contact sheet: stack all panels vertically.
    if panel_paths:
        sheets = [Image.open(p).convert("RGB") for p in panel_paths]
        width = max(s.width for s in sheets)
        scaled = []
        for s in sheets:
            if s.width != width:
                s = s.resize((width, round(s.height * width / s.width)), Image.BILINEAR)
            scaled.append(s)
        total_h = sum(s.height for s in scaled)
        sheet = Image.new("RGB", (width, total_h), (255, 255, 255))
        y = 0
        for s in scaled:
            sheet.paste(s, (0, y))
            y += s.height
        sheet.save(out_dir / "contactsheet.png")
        print(f"[eval_real] wrote contactsheet.png ({width}x{total_h})")

    (out_dir / "predictions.json").write_text(json.dumps(predictions, indent=2))
    print(f"[eval_real] wrote {out_dir/'predictions.json'}")


if __name__ == "__main__":
    main()
