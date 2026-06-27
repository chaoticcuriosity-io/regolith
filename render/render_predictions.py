#!/usr/bin/env python3
"""
render_predictions.py — the cinematic centerpiece: an RTX lunar flythrough with
the synthetic-trained hazard model's predictions overlaid live (a rover's-eye
"hazard HUD").

The pipeline is THREE stages, deliberately split so Isaac Sim and PyTorch never
share a process (each is a heavyweight, mutually-incompatible runtime):

  Stage A — `render`   (Isaac Sim container, nvcr.io/nvidia/isaac-sim:6.0.0)
      Author the lunar stage ONCE with a hero seed + dramatic low-sun params
      (scene/build_lunar_stage.py), then dolly/orbit a single RTX camera across
      it, capturing one RGB frame per camera pose from the LdrColor annotator
      (read AFTER convergence updates so auto-exposure/DLSS have settled — a
      BasicWriter captures pre-convergence and persists a black buffer on this
      box). The camera path is saved to camera_path.json so it is reproducible.

  Stage B — `overlay`  (PyTorch container, nvcr.io/nvidia/pytorch:26.03-py3)
      Load the deployed dr_1500 checkpoint and, per RGB frame, run the EXACT
      eval/_infer.py inference path (ImageNet norm + SegFormer H/4 -> input
      upsample before argmax). Composite an honest hazard overlay:
        rock (hazard) = semi-transparent red + bright detection outline
        regolith/safe-ground = subtle green "safe-traverse" tint
        sky = untouched
      then brand it (Chaotic Curiosity wordmark + a small caption/legend).

  Stage C — `assemble` (anywhere with ffmpeg; run on the Spark host)
      ffmpeg the branded frames into a full-res faststart MP4, pull a few hero
      stills, and emit a web-optimized preview (<= ~8 MB) for embedding.

RTX descriptor-leak rule (setup-notes.md Session 5): the whole flythrough is
kept to <= 300 frames in ONE fresh Isaac container (well under the ~1500 clean-
pool budget), uses the poll-for-files drain loop (NOT bare wait_until_complete),
mounts a chmod-777 persistent shader cache, and the container is removed when
done. A single render product + single annotator are created ONCE and reused
across every camera pose (the per-frame render-product/writer churn that caused
the Session-5 leak is therefore avoided entirely).

Hardware: NVIDIA DGX Spark (GB10 Grace-Blackwell, aarch64, CUDA 13, sm_121,
128 GB unified memory — nvidia-smi reports VRAM as N/A, which is expected).

The full-resolution MP4 stays on the Spark (regolith_render/, git-excluded as a
large binary); only hero stills + a small preview are committed under
docs/reports/assets/. See setup-notes.md ## Session 8 for the full walkthrough.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import subprocess
import sys
import time


# --------------------------------------------------------------------------- #
# Hero scene parameters — a dramatic, IN-DISTRIBUTION lunar stage.
#
# Every value sits inside the train_dr.yaml domain ranges (sun el [5,40] /
# az [100,260] / I [9000,20000]; regolith albedo [0.10,0.28]; terrain amp
# [2.0,4.8]; camera height [1.5,2.7] / pitch [-5,1.5] / fov [52,72]) so the
# model sees an appearance it was trained on and the overlay reads accurately.
# Low sun (el 11 deg) + side-back azimuth (122 deg, behind the +Y-looking
# camera) rakes long hard shadows across the terrain; the dome sun-hole stays
# ~120 deg off the optical axis, far outside the frustum (setup-notes Gotcha 11).
# --------------------------------------------------------------------------- #
HERO_PARAMS = {
    "sun_elevation_deg": 11.0,
    "sun_azimuth_deg": 122.0,
    "sun_intensity": 17000.0,
    "regolith_albedo": 0.19,
    "regolith_roughness": 0.95,
    "terrain_amplitude": 3.7,
    "crater_count_range": (8, 13),
    "rock_count_range": (95, 125),
    "near_rock_count": 14,
    "rock_scale_range": (0.30, 1.30),
    "near_rock_scale_range": (1.4, 3.1),
    "near_rock_y_range": (-78.0, -44.0),
    "embedding_depth_frac": 0.30,
    "camera_height_m": 2.0,
    "camera_pitch_deg": -1.5,   # base; the path overrides orientation per frame
    "camera_fov_deg": 62.0,
    "dome_radius": 600.0,
    "star_count": 260,
}

# Cinematic camera move (Z-up, meters; the rover looks toward +Y / the horizon).
# A slow dolly FORWARD into the boulder field, with a gentle rightward arc, a
# slow pan, an easing downward tilt toward the approaching hazards, and a subtle
# rover bob. Cosine easing on the dolly/tilt so the move starts and stops gently.
CAM_PATH = {
    "dolly_y": (-95.0, -63.0),     # forward 32 m (stays behind most near rocks)
    "arc_x_amp": 7.0,              # lateral arc amplitude (0 -> +amp -> 0)
    "eye_z_base": 2.0,             # rover eye height
    "eye_z_bob": 0.10,             # subtle vertical bob amplitude
    "eye_z_bob_cycles": 1.5,
    "yaw_deg": (6.0, -6.0),        # slow pan (look slightly right -> slightly left)
    "pitch_deg": (-1.5, -3.0),     # ease the tilt down toward the hazards
}


def _ease(t: float) -> float:
    """Cosine ease-in-out on t in [0,1] -> [0,1] (gentle start/stop)."""
    return 0.5 - 0.5 * math.cos(math.pi * t)


def _pose_at(t: float, idx: int | None = None) -> dict:
    """Camera pose at parameter ``t`` (valid for any t, including t<0 for the
    continuous lead-in). ``fwd`` matches build_lunar_stage's convention
    (az=0 -> +Y, pitch<0 -> look down) so it feeds _basis_matrix() directly."""
    y0, y1 = CAM_PATH["dolly_y"]
    yaw0, yaw1 = CAM_PATH["yaw_deg"]
    p0, p1 = CAM_PATH["pitch_deg"]
    xamp = CAM_PATH["arc_x_amp"]
    zb = CAM_PATH["eye_z_base"]
    zbob = CAM_PATH["eye_z_bob"]
    zcyc = CAM_PATH["eye_z_bob_cycles"]

    te = _ease(t)
    ex = xamp * math.sin(math.pi * t)                 # gentle right arc, back to 0
    ey = y0 + (y1 - y0) * te                           # eased dolly forward
    ez = zb + zbob * math.sin(2.0 * math.pi * zcyc * t)  # subtle bob
    yaw = yaw0 + (yaw1 - yaw0) * t                     # linear pan
    pitch = p0 + (p1 - p0) * te                        # eased tilt-down
    az = math.radians(yaw)
    pr = math.radians(pitch)
    fwd = [math.cos(pr) * math.sin(az), math.cos(pr) * math.cos(az), math.sin(pr)]
    return {
        "idx": idx, "t": round(t, 6),
        "eye": [round(ex, 5), round(ey, 5), round(ez, 5)],
        "fwd": [round(c, 6) for c in fwd],
        "yaw_deg": round(yaw, 4), "pitch_deg": round(pitch, 4),
    }


def compute_camera_path(n_frames: int) -> list[dict]:
    """The hero shot's per-frame camera poses (t in [0,1]). Pure python."""
    if n_frames <= 1:
        return [_pose_at(0.0, 0)]
    return [_pose_at(k / (n_frames - 1), k) for k in range(n_frames)]


def compute_lead_in(n_frames: int, n_lead: int, dist_m: float = 20.0) -> list[dict]:
    """A dedicated, CONTINUOUS fast lead-in that warms the RTX temporal pipeline
    (DLSS/TAA history) before the hero shot. Warming needs a large cumulative
    camera TRANSLATION (~15-20 m) — which the hero path covers only slowly — so
    the lead-in is a fast ~dist_m dolly that ends EXACTLY at pose 0 (no seam
    discontinuity; a backward jump would reset DLSS history and black the first
    frames). A small yaw sweep adds the pose variation that helps it warm.
    Rendered frames are discarded; by pose 0 the readback is lit."""
    if n_frames <= 1 or n_lead <= 0:
        return []
    p0 = _pose_at(0.0)
    ex0, ey0, ez0 = p0["eye"]
    yaw0 = CAM_PATH["yaw_deg"][0]
    pitch0 = float(p0["pitch_deg"])
    out = []
    for i in range(n_lead):
        f = i / n_lead                       # 0 .. just-below-1 (ends at pose 0)
        decay = 1.0 - f                       # oscillation amplitude decays to 0 at pose 0
        # Lateral S-sweep + yaw sweep (big optical flow -> fills DLSS history),
        # plus a forward dolly. All terms vanish at f->1 so it ENDS exactly at
        # pose 0 (continuous, no seam). Forward-only motion produced too little
        # screen flow to warm; lateral/rotational motion does.
        ex = ex0 + 11.0 * math.sin(2.0 * math.pi * f) * decay
        ey = (ey0 - dist_m) + dist_m * f
        yaw = yaw0 + 13.0 * math.sin(2.0 * math.pi * f) * decay
        az = math.radians(yaw)
        pr = math.radians(pitch0)
        fwd = [math.cos(pr) * math.sin(az), math.cos(pr) * math.cos(az), math.sin(pr)]
        out.append({"idx": None, "t": None,
                    "eye": [round(ex, 5), round(ey, 5), ez0],
                    "fwd": [round(c, 6) for c in fwd]})
    return out


# =========================================================================== #
# Stage A — render the RTX flythrough RGB frames (runs inside Isaac Sim).
# =========================================================================== #
def cmd_render(args: argparse.Namespace) -> None:
    # Repo root importable (scene/, etc.) regardless of CWD.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    out = os.path.abspath(args.out)
    rgb_dir = os.path.join(out, "rgb")
    os.makedirs(rgb_dir, exist_ok=True)

    n = int(args.frames)
    W, H = int(args.width), int(args.height)
    path = compute_camera_path(n)

    # Persist the move + settings BEFORE booting Kit (reproducibility, and so a
    # crash still leaves the camera path on disk).
    meta = {
        "stage": "A/render",
        "seed": args.seed,
        "frames": n, "fps": args.fps, "width": W, "height": H,
        "renderer": args.renderer, "rt_subframes": args.subframes,
        "hero_params": {k: (list(v) if isinstance(v, tuple) else v)
                        for k, v in HERO_PARAMS.items()},
        "cam_path_spec": CAM_PATH,
        "camera_path": path,
    }
    with open(os.path.join(out, "camera_path.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    t0 = time.time()

    def log(msg: str) -> None:
        print(">>> [render] %s  (+%.1fs)" % (msg, time.time() - t0), flush=True)
        sys.stderr.flush()

    log("config: seed=%d frames=%d %dx%d renderer=%s subframes=%d out=%s"
        % (args.seed, n, W, H, args.renderer, args.subframes, out))

    # Shield Kit/SimulationApp from our CLI flags.
    argv0 = sys.argv[0] if sys.argv else "render_predictions.py"
    sys.argv = [argv0]

    from isaacsim import SimulationApp
    sim = SimulationApp({"headless": True, "renderer": args.renderer})
    log("SimulationApp ready")

    import carb
    import omni.usd
    import omni.replicator.core as rep
    for _ in range(10):
        sim.update()
    rep.orchestrator.set_capture_on_play(False)

    # RTX configuration. We keep this DELIBERATELY MINIMAL: the validated
    # Session 3/4 path used the bare SimulationApp renderer with NO carb-settings
    # overrides and produced well-lit frames (mean ~104). Overriding tonemap /
    # exposure / rendermode here once produced a constant-black buffer, so we
    # touch nothing for RayTracedLighting. PathTracing only needs the rendermode
    # switch; rt_subframes (passed to orchestrator.step) drives accumulation/spp.
    settings = carb.settings.get_settings()
    if args.renderer.lower() == "pathtracing":
        try:
            settings.set("/rtx/rendermode", "PathTracing")
            settings.set("/rtx/pathtracing/totalSpp", int(args.subframes))
        except Exception as e:  # noqa: BLE001 — best-effort
            log("pathtracing-settings note: %r" % e)

    # CRITICAL — disable auto-exposure (eye adaptation) and set a FIXED film
    # exposure. Auto-exposure ramps up from black over several frames (so the
    # first frames render dark/black) AND drifts during the camera move, which
    # would make the video flicker. A fixed exposure makes every frame
    # consistently lit from the very first capture, and lets us tune the hero
    # brightness to sit in the model's training distribution (mean ~65-95).
    try:
        settings.set("/rtx/post/histogram/enabled", False)
        settings.set("/rtx/post/tonemap/filmIso", float(args.film_iso))
        settings.set("/rtx/post/tonemap/cameraShutter", float(args.camera_shutter))
        settings.set("/rtx/post/tonemap/fNumber", float(args.f_number))
    except Exception as e:  # noqa: BLE001
        log("exposure-settings note: %r" % e)
    log("replicator ready (renderer=%s, fixed exposure iso=%.0f)"
        % (args.renderer, float(args.film_iso)))

    import numpy as np
    from PIL import Image
    from scene.build_lunar_stage import build_lunar_stage, _basis_matrix, _set_transform

    # Build the hero stage ONCE; the camera moves over a static scene.
    omni.usd.get_context().new_stage()
    build_lunar_stage(args.seed, HERO_PARAMS)
    for _ in range(25):
        sim.update()
    log("hero stage authored")

    stage = omni.usd.get_context().get_stage()
    cam_prim = stage.GetPrimAtPath("/World/RoverCam")

    # ONE render product + ONE LdrColor ANNOTATOR, reused across every pose.
    #
    # We capture via the annotator, NOT BasicWriter: on this box BasicWriter
    # writes the frame at step-time, BEFORE the RTX color pipeline (auto-exposure
    # + DLSS temporal accumulation) has converged -> it persists an all-black /
    # under-exposed buffer. The LdrColor annotator, read AFTER pumping a batch of
    # convergence updates, returns the fully-lit frame (verified: a BasicWriter
    # frame read mean 0, the same annotator read mean ~64 / fully lit). Reusing a
    # single render product across all poses also avoids the Session-5 per-frame
    # render-product churn that leaks RTX descriptors.
    rp = rep.create.render_product("/World/RoverCam", (W, H))
    ldr = rep.AnnotatorRegistry.get_annotator("LdrColor")
    ldr.attach(rp)
    log("render product + LdrColor annotator attached")

    n_warm = int(args.warmup_updates)
    n_conv = int(args.converge_updates)

    def _capture_rgb():
        """Read the current LdrColor frame as HxWx3 uint8."""
        d = np.asarray(ldr.get_data())
        return d[..., :3].astype(np.uint8)

    # ONE continuous capture pass over [lead-in poses] + [hero poses]. The lead-in
    # is a fast lateral/yaw S-sweep that warms the RTX temporal pipeline (DLSS/TAA
    # history) — forward-only motion produces too little screen flow to warm it,
    # and a separate warmup loop does not carry into the capture (the readback is
    # latency-bound). Running it as ONE pass and SKIP-saving the lead frames lets
    # the warm state flow into the hero frames. We additionally GATE hero saving
    # on the readback being lit (mean > threshold) and number saved frames
    # contiguously, so no black startup frame is ever written.
    lead = compute_lead_in(n, max(12, int(args.prime_steps)))
    all_poses = [(False, q) for q in lead] + [(True, q) for q in path]
    for _ in range(20):
        sim.update()

    per_frame = []
    means = []
    save_i = 0
    saving = False
    status = "done"
    try:
        for is_hero, pose in all_poses:
            tf = time.time()
            _set_transform(cam_prim, _basis_matrix(pose["fwd"], translate=tuple(pose["eye"])))
            for _ in range(n_warm):
                sim.update()
            rep.orchestrator.step(delta_time=0.0, rt_subframes=int(args.subframes))
            # Convergence updates: DLSS accumulation + exposure settle here.
            for _ in range(n_conv):
                sim.update()
            rgb = _capture_rgb()
            m = float(rgb.mean())

            if not is_hero:
                continue  # lead-in: warm only, never saved
            if not saving:
                if m <= float(args.lit_threshold):
                    continue  # still-dark hero frame at startup -> skip
                saving = True
            Image.fromarray(rgb).save(os.path.join(rgb_dir, "rgb_%05d.png" % save_i))
            dt = time.time() - tf
            per_frame.append(dt)
            means.append(round(m, 2))
            if save_i == 0 or (save_i + 1) % 10 == 0:
                warm = per_frame[1:] or [dt]
                log("FRAME %05d ok %.1fs (warm mean %.1fs) rgb_mean=%.1f"
                    % (save_i, dt, sum(warm) / len(warm), m))
            save_i += 1
    except Exception:
        import traceback
        log("FATAL:\n" + traceback.format_exc())
        status = "error"
    finally:
        try:
            ldr.detach()
            rp.destroy()
        except Exception:  # noqa: BLE001
            pass
        have = len(glob.glob(os.path.join(rgb_dir, "rgb_*.png")))
        warm = per_frame[1:] if len(per_frame) > 1 else per_frame
        meta["result"] = {
            "status": status, "frames_written": have,
            "cold_frame_s": round(per_frame[0], 1) if per_frame else None,
            "warm_mean_s": round(sum(warm) / len(warm), 2) if warm else None,
            "total_render_min": round((time.time() - t0) / 60.0, 1),
            "rgb_mean_min": min(means) if means else None,
            "rgb_mean_max": max(means) if means else None,
        }
        with open(os.path.join(out, "camera_path.json"), "w") as fh:
            json.dump(meta, fh, indent=2)
        log("closing SimulationApp (status=%s, %d/%d frames, %.1f min, rgb_mean %s..%s)"
            % (status, have, n, (time.time() - t0) / 60.0,
               meta["result"]["rgb_mean_min"], meta["result"]["rgb_mean_max"]))
        sim.close()
    print("RENDER_DONE status=%s out=%s frames=%d" % (status, out, have), flush=True)


# =========================================================================== #
# Stage B — overlay live hazard predictions + branding (runs inside PyTorch).
# =========================================================================== #
# Hazard-HUD palette (distinct from the report's flat analysis palette: this is
# a cinematic read, tuned for legibility over the lit RTX frame).
_ROCK_RGB = (228, 56, 46)      # hazard red
_ROCK_EDGE_RGB = (255, 132, 96)  # bright detection outline
_SAFE_RGB = (54, 200, 104)     # safe-traverse green

# Brand + caption strings.
_BRAND = "CHAOTIC  CURIOSITY"
_TAGLINE = "regolith · lunar hazard segmentation"
_CAPTION_1 = "Synthetic-trained hazard segmentation · live overlay"
_CAPTION_2 = "trained + rendered on one DGX Spark"
_MODEL_TAG = "SegFormer-B0 · domain-randomized · rock-IoU 0.815"


def _find_fonts():
    """Locate DejaVu Sans (regular + bold). matplotlib ships them; fall back to
    common system paths, then to PIL's bitmap default."""
    cands_reg, cands_bold = [], []
    try:
        import matplotlib.font_manager as fm
        cands_reg.append(fm.findfont("DejaVu Sans"))
        cands_bold.append(fm.findfont("DejaVu Sans:bold"))
    except Exception:  # noqa: BLE001
        pass
    cands_reg += ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    cands_bold += ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]
    reg = next((p for p in cands_reg if p and os.path.isfile(p)), None)
    bold = next((p for p in cands_bold if p and os.path.isfile(p)), None)
    return reg, bold


def _font(path, size):
    from PIL import ImageFont
    if path:
        try:
            return ImageFont.truetype(path, size)
        except Exception:  # noqa: BLE001
            pass
    return ImageFont.load_default()


def hazard_overlay(rgb, mask, rock_alpha: float, reg_alpha: float):
    """Composite the honest hazard read onto the RTX RGB frame.

    rock (1)     -> semi-transparent red fill + a bright eroded-edge outline
    regolith (0) -> subtle green safe-traverse tint
    sky (2)      -> untouched
    """
    import numpy as np
    from scipy import ndimage

    out = rgb.astype(np.float32)
    rock_m = mask == 1
    reg_m = mask == 0

    if reg_m.any():
        out[reg_m] = (1.0 - reg_alpha) * out[reg_m] + reg_alpha * np.array(_SAFE_RGB, np.float32)
    if rock_m.any():
        out[rock_m] = (1.0 - rock_alpha) * out[rock_m] + rock_alpha * np.array(_ROCK_RGB, np.float32)
        # Bright detection outline (HUD pop) — the rim of each rock region.
        eroded = ndimage.binary_erosion(rock_m, iterations=2)
        edge = ndimage.binary_dilation(rock_m & ~eroded, iterations=1)
        out[edge] = np.array(_ROCK_EDGE_RGB, np.float32)

    return np.clip(out, 0, 255).astype(np.uint8)


def add_branding(img, fonts, frame_idx: int, n_frames: int, rock_frac: float):
    """Draw the Chaotic Curiosity mark + caption/legend HUD onto a PIL RGB image.

    Tasteful, not cluttered: a soft lower-third gradient for legibility, a top-
    left wordmark, a two-line caption bottom-left, and a small legend + model
    stat bottom-right. Thin accent rule + corner ticks for a HUD feel.
    """
    from PIL import Image, ImageDraw

    reg_font, bold_font = fonts
    base = img.convert("RGBA")
    W, Hh = base.size
    s = Hh / 1080.0  # scale every dimension off a 1080p reference

    def px(v):
        return max(1, int(round(v * s)))

    # --- soft lower-third gradient (legibility) --------------------------- #
    import numpy as np
    grad_h = int(0.26 * Hh)
    g = np.zeros((grad_h, W, 4), dtype=np.uint8)
    ramp = np.linspace(0.0, 1.0, grad_h) ** 1.6
    g[..., 3] = (ramp[:, None] * 165).astype(np.uint8)  # black, rising alpha
    base.alpha_composite(Image.fromarray(g, "RGBA"), (0, Hh - grad_h))

    # thin top gradient too (frames the wordmark)
    tg = np.zeros((int(0.12 * Hh), W, 4), dtype=np.uint8)
    tramp = np.linspace(1.0, 0.0, tg.shape[0]) ** 1.6
    tg[..., 3] = (tramp[:, None] * 120).astype(np.uint8)
    base.alpha_composite(Image.fromarray(tg, "RGBA"), (0, 0))

    draw = ImageDraw.Draw(base)
    accent = (235, 96, 70, 255)   # warm hazard-orange accent
    white = (240, 240, 242, 255)
    dim = (196, 198, 205, 255)

    mx = px(46)   # left/right margin
    # --- top-left wordmark ------------------------------------------------- #
    wf = _font(bold_font, px(34))
    tf = _font(reg_font, px(19))
    draw.text((mx, px(40)), _BRAND, font=wf, fill=white)
    draw.text((mx, px(40) + px(40)), _TAGLINE, font=tf, fill=dim)
    # accent tick under the wordmark
    draw.rectangle([mx, px(40) + px(70), mx + px(58), px(40) + px(70) + px(4)], fill=accent)

    # --- top-right timecode / progress ------------------------------------ #
    tcf = _font(reg_font, px(18))
    tc = "FRAME %03d / %03d" % (frame_idx + 1, n_frames)
    tcw = draw.textlength(tc, font=tcf)
    draw.text((W - mx - tcw, px(44)), tc, font=tcf, fill=dim)

    # --- bottom-left caption ---------------------------------------------- #
    cf = _font(bold_font, px(24))
    cf2 = _font(reg_font, px(20))
    by = Hh - px(108)
    draw.text((mx, by), _CAPTION_1, font=cf, fill=white)
    draw.text((mx, by + px(34)), _CAPTION_2, font=cf2, fill=dim)

    # --- bottom-right legend + model stat --------------------------------- #
    lf = _font(reg_font, px(20))
    sw = px(20)        # swatch size
    gap = px(10)
    items = [("ROCK / HAZARD", _ROCK_RGB), ("SAFE REGOLITH", _SAFE_RGB)]
    # measure widths to right-align the legend column
    widths = [draw.textlength(t, font=lf) for t, _ in items]
    col_w = sw + gap + int(max(widths))
    lx = W - mx - col_w
    ly = Hh - px(104)
    for (label, color), tw in zip(items, widths):
        draw.rounded_rectangle([lx, ly, lx + sw, ly + sw], radius=px(4),
                               fill=color + (235,), outline=(0, 0, 0, 160), width=px(1))
        draw.text((lx + sw + gap, ly + px(1)), label, font=lf, fill=white)
        ly += sw + px(12)
    sf = _font(reg_font, px(17))
    stw = draw.textlength(_MODEL_TAG, font=sf)
    draw.text((W - mx - stw, ly + px(2)), _MODEL_TAG, font=sf, fill=dim)

    # --- thin HUD corner ticks (subtle) ----------------------------------- #
    tick = px(26)
    tl = px(2)
    for (cx, cy, dx, dy) in [
        (mx - px(14), px(34), 1, 1), (W - mx + px(14), px(34), -1, 1),
        (mx - px(14), Hh - px(34), 1, -1), (W - mx + px(14), Hh - px(34), -1, -1),
    ]:
        draw.line([(cx, cy), (cx + dx * tick, cy)], fill=accent, width=tl)
        draw.line([(cx, cy), (cx, cy + dy * tick)], fill=accent, width=tl)

    return base.convert("RGB")


def cmd_overlay(args: argparse.Namespace) -> None:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    import numpy as np
    import torch
    from PIL import Image

    # Reuse the EXACT eval inference path (ImageNet norm + H/4->input upsample).
    from eval._infer import load_model, predict

    rgb_dir = os.path.abspath(args.rgb_dir)
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    rgb_paths = sorted(glob.glob(os.path.join(rgb_dir, "**", "rgb*.png"), recursive=True))
    if not rgb_paths:
        raise SystemExit("no rgb*.png frames found under %s" % rgb_dir)
    if int(args.skip_head) > 0:
        # Drop the first few frames (the brief exposure fade-in at capture start),
        # so the branded sequence begins fully exposed.
        rgb_paths = rgb_paths[int(args.skip_head):]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_model(args.checkpoint, device)
    fonts = _find_fonts()
    print(">>> [overlay] model=%s device=%s frames=%d infer=%dx%d"
          % (ckpt.get("model_name", "?"), device, len(rgb_paths),
             args.infer_width, args.infer_height), flush=True)

    iw, ih = int(args.infer_width), int(args.infer_height)
    n = len(rgb_paths)
    frame_meta = []
    t0 = time.time()
    for k, p in enumerate(rgb_paths):
        rgb = np.array(Image.open(p).convert("RGB"))
        Hh, Ww = rgb.shape[:2]

        # Downscale to the inference resolution (keeps 16:9, ~training scale),
        # predict, then NEAREST-upsample the class map back to full render res.
        small = np.array(Image.fromarray(rgb).resize((iw, ih), Image.BILINEAR))
        mask_small = predict(model, small, device).astype(np.uint8)
        mask = np.array(Image.fromarray(mask_small).resize((Ww, Hh), Image.NEAREST))

        # Inference runs on the ORIGINAL RGB (above); the composite uses a
        # display-graded copy. The RTX renderer auto-exposes the lunar surface
        # brighter than the training distribution; a mild gain (<1) darkens it
        # toward a dramatic, in-distribution lunar look WITHOUT changing the
        # predictions (which are computed from the original pixels).
        disp = np.clip(rgb.astype(np.float32) * float(args.display_gain), 0, 255).astype(np.uint8)
        comp = hazard_overlay(disp, mask, args.rock_alpha, args.reg_alpha)
        branded = add_branding(Image.fromarray(comp), fonts, k, n,
                               float((mask == 1).mean()))
        branded.save(os.path.join(out_dir, "overlay_%05d.png" % k))

        fr = {
            "idx": k, "src": os.path.basename(p),
            "rock_frac": round(float((mask == 1).mean()), 5),
            "regolith_frac": round(float((mask == 0).mean()), 5),
            "sky_frac": round(float((mask == 2).mean()), 5),
        }
        frame_meta.append(fr)
        if k == 0 or (k + 1) % 20 == 0 or k == n - 1:
            print(">>> [overlay] %05d/%d rock=%.3f (%.2f s/frame)"
                  % (k + 1, n, fr["rock_frac"], (time.time() - t0) / (k + 1)), flush=True)

    with open(os.path.join(out_dir, "overlay_frames.json"), "w") as fh:
        json.dump({
            "checkpoint": os.path.abspath(args.checkpoint),
            "model_name": ckpt.get("model_name"),
            "infer_res": [iw, ih],
            "rock_alpha": args.rock_alpha, "reg_alpha": args.reg_alpha,
            "n_frames": n,
            "mean_rock_frac": round(float(np.mean([f["rock_frac"] for f in frame_meta])), 5),
            "frames": frame_meta,
        }, fh, indent=2)
    print("OVERLAY_DONE out=%s frames=%d (%.1f min)"
          % (out_dir, n, (time.time() - t0) / 60.0), flush=True)


# =========================================================================== #
# Stage C — assemble MP4 + stills + web preview (runs anywhere with ffmpeg).
# =========================================================================== #
def _run(cmd: list[str]) -> None:
    print(">>> [assemble] $ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _select_stills(overlay_dir: str, n_stills: int) -> list[int]:
    """Pick hero still indices: the single hazard-heaviest frame plus evenly
    spaced frames, biased toward decent rock coverage."""
    meta_path = os.path.join(overlay_dir, "overlay_frames.json")
    overlays = sorted(glob.glob(os.path.join(overlay_dir, "overlay_*.png")))
    n = len(overlays)
    if n == 0:
        return []
    picks = []
    if os.path.isfile(meta_path):
        with open(meta_path) as fh:
            frames = json.load(fh).get("frames", [])
        if frames:
            # Hazard-heaviest frame in the back half (we've approached the field).
            back = [f for f in frames if f["idx"] >= n // 2] or frames
            picks.append(max(back, key=lambda f: f["rock_frac"])["idx"])
    # evenly spaced anchors
    for frac in (0.20, 0.52, 0.84):
        picks.append(int(round(frac * (n - 1))))
    # dedupe, keep order, trim
    seen, ordered = set(), []
    for i in sorted(picks):
        if i not in seen:
            seen.add(i)
            ordered.append(i)
    return ordered[:n_stills]


def cmd_assemble(args: argparse.Namespace) -> None:
    overlay_dir = os.path.abspath(args.overlay_dir)
    out_root = os.path.abspath(args.out)
    stills_dir = os.path.join(out_root, "stills")
    os.makedirs(stills_dir, exist_ok=True)

    overlays = sorted(glob.glob(os.path.join(overlay_dir, "overlay_*.png")))
    if not overlays:
        raise SystemExit("no overlay_*.png frames under %s" % overlay_dir)
    n = len(overlays)
    pattern = os.path.join(overlay_dir, "overlay_%05d.png")
    fps = int(args.fps)

    # --- 1. full-res faststart MP4 (stays on the Spark) ------------------- #
    full_mp4 = os.path.join(out_root, "regolith_flythrough.mp4")
    _run(["ffmpeg", "-y", "-framerate", str(fps), "-i", pattern,
          "-c:v", "libx264", "-preset", "slow", "-crf", "17",
          "-pix_fmt", "yuv420p", "-movflags", "+faststart", full_mp4])

    # --- 2. hero stills (downsized for the repo) -------------------------- #
    still_idx = _select_stills(overlay_dir, int(args.n_stills))
    still_paths = []
    for n_out, i in enumerate(still_idx, start=1):
        src = pattern % i
        dst = os.path.join(stills_dir, "render-hero-%d.png" % n_out)
        _run(["ffmpeg", "-y", "-i", src, "-vf", "scale=%d:-1" % int(args.still_width), dst])
        still_paths.append(dst)

    # --- 3. web-optimized preview MP4 (<= ~8 MB, for embedding) ----------- #
    preview = os.path.join(out_root, "render-preview.mp4")
    _run(["ffmpeg", "-y", "-framerate", str(fps), "-i", pattern,
          "-vf", "scale=%d:-2" % int(args.preview_width),
          "-c:v", "libx264", "-preset", "slow", "-crf", str(int(args.preview_crf)),
          "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", preview])

    # --- 4. small looping GIF (most portable embed) ----------------------- #
    gif = os.path.join(out_root, "render-preview.gif")
    gfps = max(8, fps // 3)
    palette = os.path.join(out_root, ".palette.png")
    vf = "fps=%d,scale=%d:-1:flags=lanczos" % (gfps, int(args.gif_width))
    try:
        _run(["ffmpeg", "-y", "-framerate", str(fps), "-i", pattern,
              "-vf", vf + ",palettegen=stats_mode=diff", palette])
        _run(["ffmpeg", "-y", "-framerate", str(fps), "-i", pattern, "-i", palette,
              "-lavfi", vf + " [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=3", gif])
    except Exception as e:  # noqa: BLE001 — gif is a nice-to-have
        print(">>> [assemble] gif skipped: %r" % e, flush=True)
        gif = None

    def _mb(path):
        return round(os.path.getsize(path) / 1e6, 2) if path and os.path.isfile(path) else None

    summary = {
        "n_frames": n, "fps": fps,
        "full_mp4": full_mp4, "full_mp4_MB": _mb(full_mp4),
        "preview_mp4": preview, "preview_MB": _mb(preview),
        "gif": gif, "gif_MB": _mb(gif),
        "stills": still_paths, "still_indices": still_idx,
    }
    with open(os.path.join(out_root, "assemble_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print("ASSEMBLE_DONE " + json.dumps(summary), flush=True)


# =========================================================================== #
# CLI
# =========================================================================== #
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cinematic RTX lunar flythrough with live hazard-segmentation overlay.")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("render", help="Stage A: render RTX flythrough RGB (Isaac Sim).")
    a.add_argument("--out", required=True, help="Output root (writes rgb/ + camera_path.json).")
    a.add_argument("--seed", type=int, default=7)
    a.add_argument("--frames", type=int, default=240, help="<= 300 (RTX leak budget).")
    a.add_argument("--fps", type=int, default=30)
    a.add_argument("--width", type=int, default=1920)
    a.add_argument("--height", type=int, default=1080)
    a.add_argument("--renderer", default="RayTracedLighting",
                   choices=["RayTracedLighting", "PathTracing"])
    a.add_argument("--subframes", type=int, default=48, help="RTX subframes/spp per frame.")
    a.add_argument("--prime-steps", type=int, default=18,
                   help="Max real-motion pre-roll steps to warm the pipeline before frame 0.")
    a.add_argument("--lead-in-m", type=float, default=6.0,
                   help="Metres behind pose 0 the pre-roll starts (real forward motion).")
    a.add_argument("--lit-threshold", type=float, default=8.0,
                   help="rgb mean above which the warmed pipeline is considered lit.")
    a.add_argument("--warmup-updates", type=int, default=8,
                   help="Per-frame updates after the camera move, before the capture step.")
    a.add_argument("--converge-updates", type=int, default=40,
                   help="Per-frame updates after the step so LdrColor converges (lit).")
    a.add_argument("--film-iso", type=float, default=1000.0,
                   help="Fixed-exposure film ISO (higher = brighter).")
    a.add_argument("--camera-shutter", type=float, default=30.0,
                   help="Fixed-exposure shutter (1/s; higher = darker).")
    a.add_argument("--f-number", type=float, default=2.0,
                   help="Fixed-exposure aperture f-number (higher = darker).")
    a.set_defaults(func=cmd_render)

    b = sub.add_parser("overlay", help="Stage B: hazard overlay + branding (PyTorch).")
    b.add_argument("--checkpoint", required=True, help="Path to dr_1500 best.pt.")
    b.add_argument("--rgb-dir", required=True, help="Stage-A rgb/ directory.")
    b.add_argument("--out", required=True, help="Output dir for overlay_*.png.")
    b.add_argument("--infer-width", type=int, default=1024)
    b.add_argument("--infer-height", type=int, default=576)
    b.add_argument("--rock-alpha", type=float, default=0.42)
    b.add_argument("--reg-alpha", type=float, default=0.16)
    b.add_argument("--display-gain", type=float, default=0.78,
                   help="Brightness gain for the composite only (inference uses original).")
    b.add_argument("--skip-head", type=int, default=0,
                   help="Drop the first N rgb frames (the brief exposure fade-in).")
    b.set_defaults(func=cmd_overlay)

    c = sub.add_parser("assemble", help="Stage C: MP4 + stills + preview (ffmpeg).")
    c.add_argument("--overlay-dir", required=True, help="Stage-B overlay_*.png directory.")
    c.add_argument("--out", required=True, help="Output root for video/stills.")
    c.add_argument("--fps", type=int, default=30)
    c.add_argument("--n-stills", type=int, default=4)
    c.add_argument("--still-width", type=int, default=1536)
    c.add_argument("--preview-width", type=int, default=1280)
    c.add_argument("--preview-crf", type=int, default=30)
    c.add_argument("--gif-width", type=int, default=720)
    c.set_defaults(func=cmd_assemble)

    return p.parse_args(argv)


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
