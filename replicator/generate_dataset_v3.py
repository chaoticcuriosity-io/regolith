#!/usr/bin/env python3
"""generate_dataset_v3.py — domain-randomized SDG generator for the v3 PREMIUM stage.

v3 of replicator/generate_dataset.py. Same proven render loop (one SimulationApp reused
across frames; per-frame new_stage() swaps the scene; poll-for-files drain; canonical
mask remap; crash-safe manifest/progress), but:

  * authors scene/build_lunar_stage_v3.build_lunar_stage_v3 (cratered regolith ground +
    tiling normal map, power-law PER-MESH rocks each tagged "rock" -> correct semantics,
    pure black starfield) instead of the v2 stage;
  * renders the SURFACE-ONLY hazard cam (rover DISABLED — the rover is cinematic-only);
  * DISABLES RTX auto-exposure / eye-adaptation (the v3 look is calibrated for fixed
    exposure: auto-exposure clips every sunlit surface to white, killing the texture);
  * draws per-frame DR via replicator.randomizers_v3.sample_params_v3 (config-driven).

Canonical masks {regolith:0, rock:1, sky:2} (ignore=255) are produced by the SAME
label-name-based scene.build_lunar_stage.canonical_mask_from_json used by v2 — the
per-mesh rocks (incl. pebbles) land in class 1.

Usage (inside the Isaac container on the DGX Spark)
--------------------------------------------------
  /isaac-sim/python.sh replicator/generate_dataset_v3.py \
      --config replicator/configs_v3/train_dr.yaml \
      --n 1200 --out /workspace/data/train_dr --seed 42 --res 512

Output layout matches v2 (rgb/, mask/, preview/, frames.jsonl, manifest.json,
progress.txt). Class map: { regolith:0, rock:1, sky:2 }; ignore index = 255.
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

# Fixed legend shared with the v2 verify/gallery path (gallery consistency).
_COLORMAP = {0: (120, 110, 96), 1: (224, 70, 38), 2: (38, 56, 110), 255: (0, 0, 0)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a domain-randomized v3 lunar segmentation dataset."
    )
    p.add_argument("--config", required=True, metavar="YAML",
                   help="Path to a YAML config in replicator/configs_v3/.")
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
    p.add_argument("--cold-budget", type=float, default=300.0,
                   help="Drain budget (s) for the first frame (cold RTX compile).")
    p.add_argument("--warm-budget", type=float, default=180.0,
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
    subframes = args.subframes if args.subframes is not None else int(cfg.get("rt_subframes", 4))
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
        print(">>> [gen-v3] %s" % msg, flush=True)
        sys.stderr.flush()

    log("config=%s mode=%s base_seed=%d res=%d subframes=%d n=%d start=%d out=%s"
        % (args.config, mode, base_seed, res, subframes, n, start, out))

    # --- Boot Isaac (shield Kit from our CLI flags) ------------------------- #
    argv0 = sys.argv[0] if sys.argv else "generate_dataset_v3.py"
    sys.argv = [argv0]

    import numpy as np
    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": True, "renderer": "RayTracedLighting"})
    log("SimulationApp ready (+%.1fs)" % (time.time() - t_start))

    import carb
    import omni.usd
    import omni.replicator.core as rep

    # CRITICAL for v3: disable auto-exposure / eye-adaptation so brightness is driven
    # purely by sun intensity + albedo. With these ON, every sunlit surface clips to
    # white and the realistic regolith texture is lost (the exact bug v3 fixed).
    settings = carb.settings.get_settings()
    for key in ("/rtx/post/histogram/enabled",
                "/rtx/post/tonemap/enableAutoExposure",
                "/rtx/post/eyeAdaptation/enabled"):
        try:
            settings.set(key, False)
        except Exception:  # noqa: BLE001
            pass

    for _ in range(10):
        simulation_app.update()
    rep.orchestrator.set_capture_on_play(False)
    log("replicator ready, auto-exposure OFF (+%.1fs)" % (time.time() - t_start))

    # Project imports (lazy pxr/numpy inside; safe after SimulationApp is up).
    from scene.build_lunar_stage_v3 import build_lunar_stage_v3, CLASS_MAP, IGNORE_INDEX
    from scene.build_lunar_stage import canonical_mask_from_json
    from replicator.randomizers_v3 import sample_params_v3

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
    frac_sum = {0: 0.0, 1: 0.0, 2: 0.0, 255: 0.0}
    n_ok = 0
    per_frame_secs = []

    def write_progress(done, total, last_sec, status="running"):
        elapsed = time.time() - t_start
        # warm rate excludes the cold first frame for an honest ETA.
        warm = per_frame_secs[1:] if len(per_frame_secs) > 1 else per_frame_secs
        rate = (sum(warm) / len(warm)) if warm else 0.0
        remaining = max(total - done, 0)
        eta_s = remaining * rate
        with open(progress_path, "w") as fh:
            fh.write(
                "status=%s split=%s done=%d/%d last_frame=%.1fs warm_mean=%.1fs "
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
        warm = per_frame_secs[1:] if len(per_frame_secs) > 1 else per_frame_secs
        manifest = {
            "stage": "v3",
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
            "warm_mean_seconds_per_frame": (sum(warm) / len(warm)) if warm else None,
            "aggregate_class_pixel_fraction": agg,
            "frames": records,
        }
        tmp = manifest_path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(manifest, fh, indent=2)
        os.replace(tmp, manifest_path)

    if start == 0 and os.path.exists(frames_jsonl):
        os.remove(frames_jsonl)

    status = "running"
    try:
        for k in range(n):
            i = start + k
            seed_i = _frame_seed(base_seed, i)
            rng_i = np.random.RandomState(seed_i)
            params_i = sample_params_v3(rng_i, cfg)
            # Hard guarantee: the dataset is surface-only.
            params_i["rover_enabled"] = False

            t_f = time.time()

            shutil.rmtree(scratch, ignore_errors=True)
            os.makedirs(scratch, exist_ok=True)

            omni.usd.get_context().new_stage()
            cams = build_lunar_stage_v3(seed_i, params_i)
            cam_path = dict(cams).get("hazard", "/World/HazardCam")
            for _ in range(10):
                simulation_app.update()

            rp = rep.create.render_product(cam_path, (res, res))
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

            rgb_dst = os.path.join(rgb_dir, "rgb_%05d.png" % i)
            mask_dst = os.path.join(mask_dir, "mask_%05d.png" % i)
            shutil.copyfile(rgb_src, rgb_dst)
            Image.fromarray(canon).save(mask_dst)

            total = canon.size
            fr = {cid: float((canon == cid).sum()) / total for cid in (0, 1, 2, 255)}
            for cid in fr:
                frac_sum[cid] += fr[cid]

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

    print("GENERATE_DATASET_V3_DONE status=%s out=%s completed=%d" % (status, out, n_ok),
          flush=True)


if __name__ == "__main__":
    main()
