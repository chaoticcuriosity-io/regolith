"""
eval_synth.py — evaluate a trained checkpoint on a synthetic hold-out split.

This formalizes the chapter-03 ablation eval: load a checkpoint, run it over a
labeled synthetic split (rgb/ + mask/), and report rock-IoU / mIoU computed the
same way training did — globally accumulated confusion counts (tp/fp/fn summed
over every pixel of every frame), not a mean of per-frame IoUs. Run against
``test_photoreal`` with the dr_1500 checkpoint to reproduce val rock-IoU ~0.815.

Produces:
  - metrics.json   : global mIoU + per-class IoU (regolith/rock/sky), per-frame
                     mean rock-IoU, and frame count
  - overlays/      : qualitative PNG triptychs (RGB | ground truth | prediction)
                     saved every --overlay-every frames

Usage
-----
  python eval/eval_synth.py \\
    --checkpoint /workspace/regolith/outputs/runs/dr_1500/best.pt \\
    --test-split /workspace/datasets/test_photoreal \\
    --out /workspace/regolith/outputs/eval_synth

The rock-IoU reported here is the synthetic "ceiling" — synthetic-to-synthetic
transfer. eval_real.py is the genuine sim-to-real test on real lunar photos.
See docs/reports/04-sim-to-real.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval._infer import (  # noqa: E402
    CLASS_NAMES, colorize, load_model, predict,
)
from training import metrics as M  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a SegFormer checkpoint on a synthetic lunar split."
    )
    p.add_argument("--checkpoint", required=True, metavar="PT", help="Path to best.pt")
    p.add_argument("--test-split", required=True, metavar="DIR",
                   help="Split directory (rgb/ + mask/ subdirs)")
    p.add_argument("--out", required=True, metavar="DIR", help="Output directory")
    p.add_argument("--overlay-every", type=int, default=50,
                   help="Save a qualitative overlay every N frames (default: 50)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    (out_dir / "overlays").mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_model(args.checkpoint, device)
    print(f"[eval_synth] device={device}  model={ckpt.get('model_name')}  "
          f"checkpoint val_rock_iou(stored)={ckpt.get('val_rock_iou')}")

    split = Path(args.test_split)
    rgb_files = sorted((split / "rgb").glob("rgb_*.png"))
    mask_files = sorted((split / "mask").glob("mask_*.png"))
    if len(rgb_files) == 0:
        raise SystemExit(f"No rgb_*.png in {split/'rgb'}")
    if len(rgb_files) != len(mask_files):
        raise SystemExit(f"rgb/mask count mismatch: {len(rgb_files)} vs {len(mask_files)}")
    print(f"[eval_synth] {len(rgb_files)} frames in {split.name}")

    # Global confusion accumulators (tp/fp/fn per class) — matches train.py.
    tp = np.zeros(3, dtype=np.int64)
    fp = np.zeros(3, dtype=np.int64)
    fn = np.zeros(3, dtype=np.int64)
    per_frame_rock_iou: list[float] = []

    for i, (rgb_path, mask_path) in enumerate(zip(rgb_files, mask_files)):
        rgb = np.array(Image.open(rgb_path).convert("RGB"))
        gt = np.array(Image.open(mask_path), dtype=np.int64)
        pred = predict(model, rgb, device)

        for cls in range(3):
            c_tp, c_fp, c_fn = M.confusion_counts(pred, gt, cls)
            tp[cls] += c_tp
            fp[cls] += c_fp
            fn[cls] += c_fn

        per_frame_rock_iou.append(M.class_iou(pred, gt, cls=1))

        if i % args.overlay_every == 0:
            panel = np.concatenate([rgb, colorize(gt), colorize(pred)], axis=1)
            Image.fromarray(panel).save(out_dir / "overlays" / f"{rgb_path.stem}.png")

    # Global IoU = tp / (tp + fp + fn) per class (NaN if class wholly absent).
    class_iou = []
    for cls in range(3):
        union = int(tp[cls] + fp[cls] + fn[cls])
        class_iou.append(float(tp[cls] / union) if union > 0 else float("nan"))
    miou = float(np.nanmean(class_iou))
    rock_iou = class_iou[1]
    mean_frame_rock_iou = float(np.nanmean(per_frame_rock_iou))

    result = {
        "checkpoint": str(args.checkpoint),
        "model_name": ckpt.get("model_name"),
        "split": split.name,
        "n_frames": len(rgb_files),
        "rock_iou_global": rock_iou,
        "miou_global": miou,
        "class_iou_global": {CLASS_NAMES[c]: class_iou[c] for c in range(3)},
        "rock_iou_mean_per_frame": mean_frame_rock_iou,
        "stored_val_rock_iou": ckpt.get("val_rock_iou"),
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2))

    print(f"[eval_synth] rock-IoU (global)   = {rock_iou:.4f}")
    print(f"[eval_synth] mIoU     (global)   = {miou:.4f}")
    print(f"[eval_synth] class-IoU            = "
          f"regolith {class_iou[0]:.4f}  rock {class_iou[1]:.4f}  sky {class_iou[2]:.4f}")
    print(f"[eval_synth] rock-IoU (per-frame) = {mean_frame_rock_iou:.4f}")
    print(f"[eval_synth] wrote {out_dir/'metrics.json'}")


if __name__ == "__main__":
    main()
