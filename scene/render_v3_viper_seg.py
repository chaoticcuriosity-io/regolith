"""
render_v3_viper_seg.py — v3 VIPER beauty stills + a VIVID ground-truth
rock-segmentation overlay on the forward hazard-cam view.

Companion render harness to build_lunar_stage_v3.py (scene authoring lives there;
this only renders + composites — LOOK gate, no dataset/training changes).

Two modes (env V3_MODE):

  diag   Fast verification (RayTracedLighting). Writes the hazard cam's
         GROUND-TRUTH semantic_segmentation AND instance_id_segmentation, then
         CROSS-TABULATES: for every pixel whose instance prim is under the
         PointInstancer (/World/Rocks/Scatter), what semantic class does it fall
         in?  Answers honestly whether the scattered pebbles emit into the "rock"
         class (critical for the upcoming dataset). Prints SCATTER_IN_ROCK=yes/no.

  final  Path-traced 1920x1080 deliverables:
           <out>/beauty/   VIPER third-person hero
           <out>/hazard/   forward hazard-cam clean beauty plate (+ the seg masks)
           <out>/overlay/  hazard plate + VIVID overlay (rock=opaque RED,
                          regolith=subtle green tint, sky untouched) built from the
                          ground-truth semantic mask, plus a flat mask key. Re-runs
                          the instancer cross-tab so the overlay is self-verifying.

All RGB + masks use the proven BasicWriter + poll-for-files drain (RTX
descriptor-leak rule). Segmentation is colorized; we read the color PNG + the
labels JSON (color -> class / prim-path) and build masks by exact color match —
robust, no id-packing assumptions.
"""
from __future__ import annotations
import os
import re
import sys
import glob
import json
import time


def _phase(t0, msg):
    print(">>> PHASE %s  (+%.1fs)" % (msg, time.time() - t0), flush=True)
    sys.stderr.flush()


def _latest(subdir, prefix):
    """Newest <prefix>_NNNN.png (data) and its sidecar json. Semantic seg writes
    <prefix>_labels_NNNN.json; instance_id seg writes <prefix>_mapping_NNNN.json."""
    pngs = [p for p in glob.glob(os.path.join(subdir, "**", prefix + "_*.png"), recursive=True)
            if "labels" not in os.path.basename(p)]
    jsons = (glob.glob(os.path.join(subdir, "**", prefix + "_labels_*.json"), recursive=True)
             + glob.glob(os.path.join(subdir, "**", prefix + "_mapping_*.json"), recursive=True))
    pngs.sort(); jsons.sort()
    return (pngs[-1] if pngs else None), (jsons[-1] if jsons else None)


def _load_colorized(subdir, prefix):
    """Return (rgba HxWx4 uint8, {name -> [ (r,g,b,a), ... ]}) for a colorized AOV."""
    import numpy as np
    from PIL import Image
    png, js = _latest(subdir, prefix)
    if not png or not js:
        return None, {}
    img = np.array(Image.open(png).convert("RGBA"))
    labels = json.load(open(js))
    name_to_colors = {}
    for col, val in labels.items():
        nums = tuple(int(x) for x in re.findall(r"\d+", col))
        if len(nums) < 3:
            continue
        if len(nums) == 3:
            nums = nums + (255,)
        if isinstance(val, dict):
            name = val.get("class", val.get("semantic", val.get("instance", str(val))))
        else:
            name = str(val)
        name_to_colors.setdefault(str(name), []).append(nums[:4])
    return img, name_to_colors


def _mask_for(img, colors):
    """Boolean HxW mask = union of pixels exactly equal to any (r,g,b,a) in colors."""
    import numpy as np
    m = np.zeros(img.shape[:2], dtype=bool)
    for c in colors:
        c = np.array(c[:4], dtype=img.dtype)
        m |= np.all(img == c[None, None, :], axis=-1)
    return m


def _drain(app, check, budget):
    dl = time.time() + budget
    while time.time() < dl:
        app.update()
        if check():
            return True
    return False


def main():
    import numpy as np
    from PIL import Image

    mode = os.environ.get("V3_MODE", "final")
    seed = int(os.environ.get("V3_SEED", "7") or "7")
    out_dir = os.environ.get("SMOKE_OUT", "/workspace/out")
    pathtrace = (mode == "final")
    if pathtrace:
        width = int(os.environ.get("V3_W", "1920") or "1920")
        height = int(os.environ.get("V3_H", "1080") or "1080")
        subframes = int(os.environ.get("V3_SUBFRAMES", "48") or "48")
    else:
        width = int(os.environ.get("V3_W", "1280") or "1280")
        height = int(os.environ.get("V3_H", "720") or "720")
        subframes = int(os.environ.get("V3_SUBFRAMES", "16") or "16")

    renderer = "PathTracing" if pathtrace else "RayTracedLighting"
    sys.argv = [sys.argv[0] if sys.argv else "render_v3_viper_seg.py"]
    t0 = time.time()
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)

    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": True, "renderer": renderer})
    _phase(t0, "simapp_ready mode=%s %dx%d" % (mode, width, height))

    import carb
    import omni.replicator.core as rep
    settings = carb.settings.get_settings()
    for key in ("/rtx/post/histogram/enabled",
                "/rtx/post/tonemap/enableAutoExposure",
                "/rtx/post/eyeAdaptation/enabled"):
        try:
            settings.set(key, False)
        except Exception:
            pass
    if pathtrace:
        pt_spp = int(os.environ.get("V3_PT_TOTALSPP", "384") or "384")
        pt_bounces = int(os.environ.get("V3_PT_BOUNCES", "5") or "5")
        settings.set("/rtx/rendermode", "PathTracing")
        settings.set("/rtx/pathtracing/spp", 1)
        settings.set("/rtx/pathtracing/totalSpp", pt_spp)
        settings.set("/rtx/pathtracing/maxBounces", pt_bounces)
        settings.set("/rtx/pathtracing/clampSpp", 0)
        try:
            settings.set("/rtx/pathtracing/optixDenoiser/enabled", False)
        except Exception:
            pass
        print(">>> pathtracing: totalSpp=%d maxBounces=%d" % (pt_spp, pt_bounces), flush=True)

    for _ in range(10):
        simulation_app.update()
    rep.orchestrator.set_capture_on_play(False)

    from build_lunar_stage_v3 import build_lunar_stage_v3
    try:
        cams = build_lunar_stage_v3(seed, {})
        for _ in range(12):
            simulation_app.update()
        _phase(t0, "scene_built")
        cam_map = dict(cams)
        haz_path = cam_map.get("hazard", "/World/HazardCam")

        # ---- hazard cam: rgb (final) + ground-truth semantic & instance masks --- #
        haz_dir = os.path.join(out_dir, "hazard")
        rp_haz = rep.create.render_product(haz_path, (width, height))
        w = rep.WriterRegistry.get("BasicWriter")
        w.initialize(
            output_dir=haz_dir,
            rgb=bool(pathtrace),
            semantic_segmentation=True,
            instance_id_segmentation=True,
            colorize_semantic_segmentation=True,
            colorize_instance_id_segmentation=True,
        )
        w.attach([rp_haz])
        for _ in range(20):
            simulation_app.update()
        _phase(t0, "warm_hazard")
        rep.orchestrator.step(delta_time=0.0, rt_subframes=subframes)
        _drain(simulation_app,
               lambda: _latest(haz_dir, "semantic_segmentation")[0] is not None,
               budget=480.0)
        try:
            rep.orchestrator.wait_until_complete()
        except Exception as e:
            print(">>> wait raised: %r" % e, flush=True)
        w.detach()
        _phase(t0, "hazard_masks")

        sem_img, sem_names = _load_colorized(haz_dir, "semantic_segmentation")
        inst_img, inst_names = _load_colorized(haz_dir, "instance_id_segmentation")
        if sem_img is None:
            print(">>> ERROR: no semantic mask written", flush=True)
            raise RuntimeError("no semantic mask")

        H, W = sem_img.shape[:2]
        rock_colors = sem_names.get("rock", [])
        reg_colors = sem_names.get("regolith", [])
        rock_mask = _mask_for(sem_img, rock_colors)
        reg_mask = _mask_for(sem_img, reg_colors)

        print(">>> [hazard] semantic classes present:", flush=True)
        for nm, cols in sorted(sem_names.items()):
            px = int(_mask_for(sem_img, cols).sum())
            print("      %-12s %8d px (%.1f%%)  colors=%d"
                  % (nm, px, 100.0 * px / (H * W), len(cols)), flush=True)

        # cross-tab: PointInstancer pixels -> what semantic class?
        scatter_names = [n for n in inst_names if "Rocks/Scatter" in n] if inst_img is not None else []
        scatter_px = scatter_in_rock = 0
        if inst_img is not None and scatter_names:
            scols = []
            for n in scatter_names:
                scols += inst_names[n]
            scatter_mask = _mask_for(inst_img, scols)
            scatter_px = int(scatter_mask.sum())
            if scatter_px:
                scatter_in_rock = int((scatter_mask & rock_mask).sum())
        frac = (scatter_in_rock / scatter_px) if scatter_px else 0.0
        verdict = "yes" if (scatter_px > 0 and frac >= 0.90) else (
                  "partial" if scatter_in_rock > 0 else "no")
        print(">>> INSTANCER CHECK: PointInstancer(/World/Rocks/Scatter) "
              "matched_instance_prims=%d scatter_px=%d in_rock=%d frac=%.3f -> SCATTER_IN_ROCK=%s"
              % (len(scatter_names), scatter_px, scatter_in_rock, frac, verdict), flush=True)

        if mode == "diag":
            _phase(t0, "closing")
            simulation_app.close()
            print("V3_DIAG_DONE", flush=True)
            return

        # ---- composite the VIVID overlay -------------------------------------- #
        rgbs = sorted([p for p in glob.glob(os.path.join(haz_dir, "**", "rgb*.png"), recursive=True)])
        ov_dir = os.path.join(out_dir, "overlay")
        os.makedirs(ov_dir, exist_ok=True)
        if not rgbs:
            print(">>> ERROR: no hazard rgb to composite", flush=True)
        else:
            beauty = np.array(Image.open(rgbs[0]).convert("RGB")).astype(np.float32)
            bh, bw = beauty.shape[:2]
            rm, gm = rock_mask, reg_mask
            if (bh, bw) != (H, W):
                rm = np.array(Image.fromarray(rock_mask).resize((bw, bh), Image.NEAREST))
                gm = np.array(Image.fromarray(reg_mask).resize((bw, bh), Image.NEAREST))
            overlay = beauty.copy()
            RED = np.array([242.0, 28.0, 28.0]); a_rock = 0.86
            overlay[rm] = a_rock * RED + (1.0 - a_rock) * beauty[rm]
            GREEN = np.array([46.0, 130.0, 52.0]); a_reg = 0.16
            overlay[gm] = (1.0 - a_reg) * beauty[gm] + a_reg * GREEN
            Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(
                os.path.join(ov_dir, "hazard_seg_overlay.png"))
            key = np.zeros((beauty.shape[0], beauty.shape[1], 3), np.uint8)
            key[gm] = (40, 110, 45); key[rm] = (240, 30, 30)
            Image.fromarray(key).save(os.path.join(ov_dir, "hazard_seg_key.png"))
            print(">>> overlay: rock_px=%d (%.2f%%) regolith_px=%d (%.2f%%) -> %s"
                  % (int(rm.sum()), 100.0 * rm.sum() / rm.size,
                     int(gm.sum()), 100.0 * gm.sum() / gm.size, ov_dir), flush=True)

        # ---- beauty hero cams (path-traced rgb) ------------------------------- #
        for name in ("beauty", "beauty2"):
            cpath = cam_map.get(name)
            if not cpath:
                continue
            sub_out = os.path.join(out_dir, name)
            rp = rep.create.render_product(cpath, (width, height))
            wb = rep.WriterRegistry.get("BasicWriter")
            wb.initialize(output_dir=sub_out, rgb=True)
            wb.attach([rp])
            for _ in range(8):
                simulation_app.update()
            _phase(t0, "warm_%s" % name)
            rep.orchestrator.step(delta_time=0.0, rt_subframes=subframes)
            ok = _drain(simulation_app,
                        lambda: bool(glob.glob(os.path.join(sub_out, "**", "rgb*.png"), recursive=True)),
                        budget=320.0)
            _phase(t0, ("got_%s" if ok else "TIMEOUT_%s") % name)
            try:
                rep.orchestrator.wait_until_complete()
            except Exception as e:
                print(">>> wait raised: %r" % e, flush=True)
            wb.detach()
            files = sorted(glob.glob(os.path.join(sub_out, "**", "rgb*.png"), recursive=True))
            if files:
                a = np.array(Image.open(files[0]).convert("RGB"))
                print(">>> %-8s mean=%.1f p1=%.0f p50=%.0f p99=%.0f -> %s"
                      % (name, float(a.mean()), float(np.percentile(a, 1)),
                         float(np.percentile(a, 50)), float(np.percentile(a, 99)),
                         files[0]), flush=True)
    except Exception:
        import traceback
        print(">>> BUILD/RENDER ERROR:\n" + traceback.format_exc(), flush=True)
    finally:
        _phase(t0, "closing")
        simulation_app.close()
    print("V3_STILLS_DONE output_dir=" + out_dir, flush=True)


if __name__ == "__main__":
    main()
