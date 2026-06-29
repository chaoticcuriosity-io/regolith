#!/usr/bin/env python3
"""
make_results_v3.py — aggregate the v3 ablation into RESULTS.md + results.json.

Reads (all under /workspace/regolith):
  outputs/runs_v3/<run>/summary.json            (best_epoch, val rock-IoU, wall)
  outputs/eval_v3/eval_synth/<run>/metrics.json (synth mIoU + per-class IoU)
  outputs/eval_v3/eval_real/<run>/predictions.json (per-image rock fraction)

Writes:
  outputs/runs_v3/results.json
  outputs/runs_v3/RESULTS.md

"flood" = fraction of real-image pixels predicted class 1 (rock). Real lunar
surface is mostly safe regolith, so LOWER flood = better sim-to-real transfer.
The rock IoU/flood numbers are reported plainly; no spin.
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path("/workspace/regolith")
RUNS = ROOT / "outputs/runs_v3"
ESYN = ROOT / "outputs/eval_v3/eval_synth"
EREAL = ROOT / "outputs/eval_v3/eval_real"
ORDER = ["nodr_750", "dr_750", "dr_1500"]

# Baselines from prior generations (given in the task brief / runs_v2 summaries).
BASE = {
    "v1": {"dr_1500": {"synth_rock_iou": 0.815, "real_flood_pct": 52.0}},
    "v2": {
        "nodr_750": {"synth_rock_iou": 0.8025},
        "dr_750":   {"synth_rock_iou": 0.8486},
        "dr_1500":  {"synth_rock_iou": 0.8521, "real_flood_pct": 83.0},
    },
}


def load_json(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception as e:
        print(f"[make_results] WARN could not read {p}: {e}")
        return None


def fmt(x, nd=4):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "n/a"


def pct(x, nd=1):
    return f"{x*100:.{nd}f}%" if isinstance(x, (int, float)) else "n/a"


results = {"runs": {}, "baselines": BASE}
for run in ORDER:
    rec: dict = {}
    s = load_json(RUNS / run / "summary.json")
    if s:
        rec["best_epoch"] = s.get("best_epoch")
        rec["epochs_run"] = s.get("epochs_run")
        rec["best_val_rock_iou"] = s.get("best_val_rock_iou")
        rec["best_val_miou"] = s.get("best_val_miou")
        rec["train_wall_s"] = s.get("wall_time_s")
    m = load_json(ESYN / run / "metrics.json")
    if m:
        rec["synth_rock_iou"] = m.get("rock_iou_global")
        rec["synth_miou"] = m.get("miou_global")
        rec["synth_class_iou"] = m.get("class_iou_global")
        rec["synth_rock_iou_per_frame"] = m.get("rock_iou_mean_per_frame")
    pr = load_json(EREAL / run / "predictions.json")
    if pr:
        floods = []
        per_img = {}
        for stem, d in pr.items():
            f = d.get("primary_frac", {}).get("rock")
            if f is not None:
                floods.append(f)
                per_img[stem] = f
        if floods:
            rec["real_flood_mean"] = sum(floods) / len(floods)
            rec["real_flood_min"] = min(floods)
            rec["real_flood_max"] = max(floods)
            rec["real_n_images"] = len(floods)
            rec["real_flood_per_image"] = per_img
    results["runs"][run] = rec

(RUNS / "results.json").write_text(json.dumps(results, indent=2))

# ----------------------------- RESULTS.md ---------------------------------
L = []
L.append("# Regolith v3 — Retrain + Eval Results\n")
L.append("SegFormer-b0 (`nvidia/mit-b0`), 3-run domain-randomization ablation on the "
         "v3 dataset (realistic cratered/displaced dark-regolith ground). "
         "Hyperparameters identical to v1/v2: `--epochs 40 --patience 8 --lr 6e-5 "
         "--batch 8 --seed 0`, best checkpoint selected on validation rock-IoU "
         "(val split = `test_photoreal`, 300 frames).\n")

v3 = results["runs"]
dr1500 = v3.get("dr_1500", {})
flood = dr1500.get("real_flood_mean")
headline = "n/a"
if isinstance(flood, (int, float)):
    fp = flood * 100
    if fp < 83.0:
        headline = (f"**HEADLINE: YES — the real flood DROPPED to {fp:.1f}% for v3 dr_1500, "
                    f"below the v2 baseline of ~83%** (v1 was ~52%).")
    else:
        headline = (f"**HEADLINE: NO — the real flood is {fp:.1f}% for v3 dr_1500, "
                    f"NOT below the v2 ~83% baseline** (v1 was ~52%). Reported plainly.")
L.append(headline + "\n")

# Per-run v3 table
L.append("## v3 per-run results\n")
L.append("| run | train data | best epoch | synth mIoU | synth rock-IoU | synth class-IoU (reg/rock/sky) | real mean flood | train wall |")
L.append("|---|---|---|---|---|---|---|---|")
data_desc = {"nodr_750": "train_nodr (750, no DR)",
             "dr_750": "first 750 of train_dr (DR)",
             "dr_1500": "full train_dr (1500, DR)"}
for run in ORDER:
    r = v3.get(run, {})
    ci = r.get("synth_class_iou") or {}
    ci_s = (f"{fmt(ci.get('regolith'),3)}/{fmt(ci.get('rock'),3)}/{fmt(ci.get('sky'),3)}"
            if ci else "n/a")
    wall = r.get("train_wall_s")
    wall_s = f"{wall/60:.1f} min" if isinstance(wall, (int, float)) else "n/a"
    L.append(f"| {run} | {data_desc[run]} | {r.get('best_epoch','n/a')} | "
             f"{fmt(r.get('synth_miou'),4)} | {fmt(r.get('synth_rock_iou'),4)} | {ci_s} | "
             f"{pct(r.get('real_flood_mean'))} | {wall_s} |")
L.append("")

# Cross-version comparison (dr_1500 lineage)
L.append("## Cross-version comparison (dr_1500)\n")
L.append("| version | synth rock-IoU (dr_1500) | real flood (dr_1500) | note |")
L.append("|---|---|---|---|")
L.append(f"| v1 | {fmt(BASE['v1']['dr_1500']['synth_rock_iou'],3)} | "
         f"~{BASE['v1']['dr_1500']['real_flood_pct']:.0f}% | first synthetic stage |")
L.append(f"| v2 | {fmt(BASE['v2']['dr_1500']['synth_rock_iou'],3)} | "
         f"~{BASE['v2']['dr_1500']['real_flood_pct']:.0f}% | photoreal rocks collapsed rock/regolith boundary |")
L.append(f"| **v3** | **{fmt(dr1500.get('synth_rock_iou'),3)}** | "
         f"**{pct(dr1500.get('real_flood_mean'))}** | realistic cratered dark-regolith ground |")
L.append("")

# Synthetic rock-IoU across runs vs v2
L.append("## Synthetic rock-IoU: v2 vs v3 (all runs)\n")
L.append("| run | v2 synth rock-IoU | v3 synth rock-IoU |")
L.append("|---|---|---|")
for run in ORDER:
    v2v = BASE["v2"].get(run, {}).get("synth_rock_iou")
    L.append(f"| {run} | {fmt(v2v,4)} | {fmt(v3.get(run,{}).get('synth_rock_iou'),4)} |")
L.append("")

# Real flood per image (dr_1500)
if dr1500.get("real_flood_per_image"):
    L.append("## v3 dr_1500 real-image flood (per image)\n")
    L.append("| image | rock flood |")
    L.append("|---|---|")
    for stem, f in sorted(dr1500["real_flood_per_image"].items(),
                          key=lambda kv: kv[1], reverse=True):
        L.append(f"| {stem} | {f*100:.1f}% |")
    L.append("")

L.append("## Method notes\n")
L.append("- **Synthetic eval**: `eval_synth.py` on `test_photoreal` (300 in-distribution "
         "photoreal frames); IoU is globally accumulated tp/fp/fn (matches training).")
L.append("- **Real eval**: `eval_real.py` on 21 NASA public-domain Apollo/Surveyor surface "
         "photos (`eval/real_images_v3`, no ground truth). Flood = mean fraction of pixels "
         "predicted `rock`. No Chang'e/CNSA imagery included.")
L.append("- Overlays (RGB + red rock mask) per run in `outputs/eval_v3/eval_real/<run>/` "
         "and the dr_1500-vs-nodr_750 money panels in "
         "`outputs/eval_v3/eval_real/compare_dr1500_vs_nodr750/`.")
L.append("")

(RUNS / "RESULTS.md").write_text("\n".join(L))
print("[make_results] wrote", RUNS / "RESULTS.md", "and", RUNS / "results.json")
print(headline)
for run in ORDER:
    r = v3.get(run, {})
    print(f"  {run:9s} synth_rock_iou={fmt(r.get('synth_rock_iou'))} "
          f"synth_miou={fmt(r.get('synth_miou'))} "
          f"real_flood={pct(r.get('real_flood_mean'))} best_epoch={r.get('best_epoch')}")
