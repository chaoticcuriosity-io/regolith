#!/usr/bin/env python3
"""
generate_dataset.py — domain-randomized synthetic dataset generator for the lunar stage.

For each frame i: draw a domain-randomized ``params`` dict (replicator.randomizers.
sample_params), author a fresh lunar stage (scene.build_lunar_stage), render one RGB +
raw-label-id frame with Omniverse Replicator's BasicWriter using the validated headless
poll-drain pattern, remap the raw mask to the canonical class map via
``canonical_mask_from_json``, and save ``rgb/rgb_XXXXX.png`` + ``mask/mask_XXXXX.png``.

ONE SimulationApp + render pipeline is reused across all frames (per-frame ``new_stage()``
swaps the scene); a persistent chmod-777 shader cache amortizes the ~150 s cold RTX
compile so warm frames are fast. See setup-notes.md ## Session 2-4 for the gotchas.

Usage (inside the Isaac container on the DGX Spark)
--------------------------------------------------
  /isaac-sim/python.sh replicator/generate_dataset.py \
      --config replicator/configs/train_dr.yaml \
      --n 1500 --out /home/.../regolith_data/train_dr --seed 42 --res 512

Output layout (per <out>/)
--------------------------
  rgb/    rgb_XXXXX.png    — RGB render (uint8)
  mask/   mask_XXXXX.png   — canonical class-id mask (uint8 {0=regolith,1=rock,2=sky,255=ignore})
  preview/ preview_XXXXX.png — fixed-colormap mask preview (every --preview-stride frames; gallery)
  frames.jsonl            — one JSON record per frame (crash-safe append: seed, params, stats)
  manifest.json           — class map, config, per-frame seeds+params, aggregate class stats
  progress.txt            — single-line live progress (frames done / total, s/frame, ETA)

Class map: { regolith: 0, rock: 1, sky: 2 }  (rock = hazard); ignore index = 255.
"""

from __future__ import annotations

import os
import sys
import glob
import json
import time
import shutil
import argparse
import datetime

# Make the repo importable (scene/, replicator/) regardless of CWD.
_THIS = os.path.abspath(__file__)
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Fixed legend shared with build_lunar_stage's verify path (gallery consistency).
_COLORMAP = {0: (120, 110, 96), 1: (224, 70, 38), 2: (38, 56, 110), 255: (0, 0, 0)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a domain-randomized synthetic lunar segmentation dataset."
    )
    p.add_argument("--config", required=True, metavar="YAML",
                   help="Path to a YAML config in replicator/configs/.")
    p.add_argument("--n", type=int, required=True, metavar="INT",
                   help="Number of frames to generate.")
    p.add_argument("--out", required=True, metavar="DIR",
                   help="Output directory for the dataset.")
    p.add_argument("--seed", type=int, default=None,
                   help="Base seed (overrides config 'seed'; default: config or 42).")
    p.add_argument("--res", type=int, default=None,
                   help="Square render resolution (overrides config 'res'; default 512).")
    p.add_argument("--subframes", type=int, default=None,
                   help="RTX accumulation subframes/frame (overrides config 'rt_subframes').")
    p.add_argument("--mode", choices=["dr", "nodr"], default=None,
                   help="Override config 'mode' (dr | nodr).")
    p.add_argument("--start-index", type=int, default=0,
                   help="First frame index (for resume); files are numbered from here.")
    p.add_argument("--preview-stride", type=int, default=25,
                   help="Write a colormap preview every Nth frame (0 = never).")
    p.add_argument("--cold-budget", type=float, default=260.0,
                   help="Drain budget (s) for the first frame (cold RTX compile).")
    p.add_argument("--warm-budget", type=float, default=120.0,
                   help="Drain budget (s) for subsequent (warm) frames.")
    return p.parse_args()


def _frame_seed(base_seed: int, i: int) -> int:
    """Well-spread, reproducible per-frame seed from a base seed + frame index."""
    return int((int(base_seed) ^ ((i + 1) * 2654435761)) & 0x7FFFFFFF)


def main() -> None:
    args = parse_args()

    # Load config before touching Kit.
    import yaml
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh) or {}

    if args.mode is not None:
        cfg["mode"] = args.mode
    mode = str(cfg.get("mode", "dr")).strip().lower()
    base_seed = args.seed if args.seed is not None else int(cfg.get("seed", 42))
    res = args.res if args.res is not None else int(cfg.get("res", 512))
    subframes = args.subframes if args.subframes is not None else int(cfg.get("rt_subframes", 3))
    n = int(args.n)
    start = int(args.start_index)
    out = os.path.abspath(args.out)

    rgb_dir = os.path.join(out, "rgb")
    mask_dir = os.path.join(out, "mask")
    prev_dir = os.path.join(out, "preview")
    scratch = os.path.join(out, ".scratch")
    for d in (rgb_dir, mask_dir, prev_dir, scratch):
        os.makedirs(d, exist_ok=True)

    frames_jsonl = os.path.join(out, "frames.jsonl")
    manifest_path = os.path.join(out, "manifest.json")
    progress_path = os.path.join(out, "progress.txt")

    t_start = time.time()

    def log(msg: str) -> None:
        print(">>> [gen] %s" % msg, flush=True)
        sys.stderr.flush()

    log("config=%s mode=%s base_seed=%d res=%d subframes=%d n=%d start=%d out=%s"
        % (args.config, mode, base_seed, res, subframes, n, start, out))

    # --- Boot Isaac (shield Kit from our CLI flags) ------------------------- #
    argv0 = sys.argv[0] if sys.argv else "generate_dataset.py"
    sys.argv = [argv0]

    import numpy as np
    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": True, "renderer": "RayTracedLighting"})
    log("SimulationApp ready (+%.1fs)" % (time.time() - t_start))

    import omni.usd
    import omni.replicator.core as rep
    for _ in range(10):
        simulation_app.update()
    rep.orchestrator.set_capture_on_play(False)
    log("replicator ready (+%.1fs)" % (time.time() - t_start))

    # Project imports (lazy pxr/numpy inside; safe after SimulationApp is up).
    from scene.build_lunar_stage import (
        build_lunar_stage, canonical_mask_from_json, CLASS_MAP, IGNORE_INDEX,
    )
    from replicator.randomizers import sample_params

    inv_class = {v: k for k, v in CLASS_MAP.items()}

    def _drain(check, budget):
        deadline = time.time() + budget
        while time.time() < deadline:
            simulation_app.update()
            if check():
                return True
        return False

    def _find(d, pred):
        return [q for q in glob.glob(os.path.join(d, "**", "*"), recursive=True)
                if os.path.isfile(q) and pred(os.path.basename(q))]

    from PIL import Image

    records = []
    # Running per-class pixel-fraction accumulators for the aggregate summary.
    frac_sum = {0: 0.0, 1: 0.0, 2: 0.0, 255: 0.0}
    n_ok = 0
    per_frame_secs = []

    def write_progress(done, total, last_sec, status="running"):
        elapsed = time.time() - t_start
        rate = (sum(per_frame_secs) / len(per_frame_secs)) if per_frame_secs else 0.0
        remaining = max(total - done, 0)
        eta_s = remaining * rate
        with open(progress_path, "w") as fh:
            fh.write(
                "status=%s split=%s done=%d/%d last_frame=%.1fs mean=%.1fs "
                "elapsed=%.0fmin eta=%.0fmin (%s)\n"
                % (status, os.path.basename(out), done, total, last_sec, rate,
                   elapsed / 60.0, eta_s / 60.0,
                   datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            )

    def write_manifest(status):
        agg = {}
        if n_ok:
            for cid in (0, 1, 2, 255):
                name = inv_class.get(cid, "ignore")
                agg[name] = frac_sum[cid] / n_ok
        manifest = {
            "class_map": CLASS_MAP,
            "ignore_index": IGNORE_INDEX,
            "config_path": os.path.abspath(args.config),
            "mode": mode,
            "base_seed": base_seed,
            "res": res,
            "rt_subframes": subframes,
            "n_requested": n,
            "start_index": start,
            "n_completed": n_ok,
            "status": status,
            "created_utc": datetime.datetime.utcfromtimestamp(t_start).isoformat() + "Z",
            "updated_utc": datetime.datetime.utcnow().isoformat() + "Z",
            "mean_seconds_per_frame": (sum(per_frame_secs) / len(per_frame_secs))
            if per_frame_secs else None,
            "aggregate_class_pixel_fraction": agg,
            "frames": records,
        }
        tmp = manifest_path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(manifest, fh, indent=2)
        os.replace(tmp, manifest_path)

    # Truncate the JSONL only on a fresh (start==0) run.
    if start == 0 and os.path.exists(frames_jsonl):
        os.remove(frames_jsonl)

    status = "running"
    try:
        for k in range(n):
            i = start + k
            seed_i = _frame_seed(base_seed, i)
            rng_i = np.random.RandomState(seed_i)
            params_i = sample_params(rng_i, cfg)

            t_f = time.time()

            # Clean scratch, author a fresh stage, render one frame.
            shutil.rmtree(scratch, ignore_errors=True)
            os.makedirs(scratch, exist_ok=True)

            omni.usd.get_context().new_stage()
            build_lunar_stage(seed_i, params_i)
            for _ in range(10):
                simulation_app.update()

            rp = rep.create.render_product("/World/RoverCam", (res, res))
            writer = rep.WriterRegistry.get("BasicWriter")
            writer.initialize(
                output_dir=scratch, rgb=True, semantic_segmentation=True,
                colorize_semantic_segmentation=False,
            )
            writer.attach([rp])
            for _ in range(15):
                simulation_app.update()

            budget = args.cold_budget if k == 0 else args.warm_budget
            rep.orchestrator.step(delta_time=0.0, rt_subframes=subframes)

            def _has_rgb():
                return bool(_find(scratch, lambda b: b.startswith("rgb") and b.endswith(".png")))

            def _has_seg():
                return bool(_find(scratch, lambda b: "semantic_segmentation" in b
                                  and b.endswith(".png")))

            ok = _drain(lambda: _has_rgb() and _has_seg(), budget=budget)
            try:
                rep.orchestrator.wait_until_complete()
            except Exception as e:  # noqa: BLE001
                log("wait_until_complete frame %d raised: %r" % (i, e))
            writer.detach()
            try:
                rp.destroy()
            except Exception:  # noqa: BLE001
                pass

            if not ok:
                log("FRAME %05d TIMEOUT after %.0fs — no files; skipping" % (i, budget))
                write_progress(k + 1, n, time.time() - t_f, status="running")
                continue

            # Locate the raw outputs in scratch.
            rgb_src = sorted(_find(scratch, lambda b: b.startswith("rgb") and b.endswith(".png")))[0]
            seg_src = sorted(_find(
                scratch,
                lambda b: "semantic_segmentation" in b and b.endswith(".png")
                and "labels" not in b,
            ))[0]
            json_src = sorted(_find(
                scratch, lambda b: b.endswith(".json") and "semantic_segmentation" in b
            ))[0]

            raw = np.array(Image.open(seg_src))
            canon = canonical_mask_from_json(raw, json_src)

            # Persist canonical mask + RGB under stable names.
            rgb_dst = os.path.join(rgb_dir, "rgb_%05d.png" % i)
            mask_dst = os.path.join(mask_dir, "mask_%05d.png" % i)
            shutil.copyfile(rgb_src, rgb_dst)
            Image.fromarray(canon).save(mask_dst)

            # Per-class pixel fractions.
            total = canon.size
            fr = {cid: float((canon == cid).sum()) / total for cid in (0, 1, 2, 255)}
            for cid in fr:
                frac_sum[cid] += fr[cid]

            # Occasional colormap preview (no extra render — pure numpy).
            if args.preview_stride and (k % args.preview_stride == 0):
                prev = np.zeros((canon.shape[0], canon.shape[1], 3), dtype=np.uint8)
                for cid, col in _COLORMAP.items():
                    prev[canon == cid] = col
                Image.fromarray(prev).save(os.path.join(prev_dir, "preview_%05d.png" % i))

            dt = time.time() - t_f
            per_frame_secs.append(dt)
            n_ok += 1

            rec = {
                "index": i,
                "seed": seed_i,
                "params": params_i,
                "class_pixel_fraction": {inv_class.get(c, "ignore"): fr[c]
                                         for c in (0, 1, 2, 255)},
                "seconds": round(dt, 2),
            }
            records.append(rec)
            with open(frames_jsonl, "a") as fh:
                fh.write(json.dumps(rec) + "\n")

            log("FRAME %05d ok  %.1fs  regolith=%.3f rock=%.3f sky=%.3f ignore=%.4f"
                % (i, dt, fr[0], fr[1], fr[2], fr[255]))
            write_progress(k + 1, n, dt, status="running")
            if (k + 1) % 25 == 0 or (k + 1) == n:
                write_manifest("running")

        status = "done"
    except Exception:
        import traceback
        log("FATAL:\n" + traceback.format_exc())
        status = "error"
    finally:
        write_manifest(status)
        write_progress(n_ok, n, per_frame_secs[-1] if per_frame_secs else 0.0, status=status)
        shutil.rmtree(scratch, ignore_errors=True)
        log("closing SimulationApp (status=%s, completed=%d/%d, +%.0fmin)"
            % (status, n_ok, n, (time.time() - t_start) / 60.0))
        simulation_app.close()

    print("GENERATE_DATASET_DONE status=%s out=%s completed=%d" % (status, out, n_ok),
          flush=True)


if __name__ == "__main__":
    main()
