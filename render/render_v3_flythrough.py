#!/usr/bin/env python3
"""
render_v3_flythrough.py — the v3 cinematic centerpiece: an RTX lunar flythrough
over the realistic v3 cratered surface featuring the VIPER rover, with the
synthetic-trained dr_1500 hazard model's predictions overlaid live (a rover's-eye
"hazard HUD") plus a third-person picture-in-picture establishing inset.

Adapts render/render_predictions.py (the v2 harness) to the v3 scene
(scene/build_lunar_stage_v3.py — realistic dark cratered regolith + power-law
rocks + detailed NASA VIPER rover + harsh low-sun lighting). Same proven,
leak-safe capture machinery: ONE render product + ONE LdrColor annotator created
ONCE and reused across every camera pose, read AFTER convergence updates (a
BasicWriter persists a pre-convergence black buffer on this box), with a
continuous lead-in that warms the RTX temporal pipeline (DLSS/TAA history).

THREE stages (Isaac Sim and PyTorch never share a process):

  Stage A — `render`  (Isaac Sim, nvcr.io/nvidia/isaac-sim:6.0.0)
      Build the v3 stage ONCE (deterministic seed), then dolly a single RTX
      camera. Two camera modes (--cam):
        forward : rover-mast forward hazard POV, a slow dolly into the boulder
                  field (the main view).
        pip     : a gentle third-person orbit of the VIPER on the surface (the
                  establishing inset).
      Each mode runs in its OWN fresh container (leak-safe + failure-isolated):
      if the PiP pass flakes, the forward pass has already shipped.

  Stage B — `overlay` (PyTorch, regolith-train-v3 / pytorch:26.03-py3)
      Load dr_1500 best.pt; per forward RGB frame run the EXACT eval/_infer.py
      path; composite a VIVID hazard read (rock = opaque red + bright outline,
      regolith = subtle green, sky untouched), then PiP-inset the matching
      third-person frame (corner, bordered, labelled) and brand it.

  Stage C — `assemble` (host ffmpeg)
      Full-res faststart MP4 + hero stills + a web preview + a looping GIF.

RTX descriptor-leak rule (setup-notes Session 5/11): each pass is one fresh
container, <= ~250 frames, single reused render product, poll/converge drain,
chmod-777 persistent shader cache mounted, container removed cleanly (never
force-killed mid-compile — that strands the cache lock and hangs the next boot).

Hardware: NVIDIA DGX Spark (GB10, aarch64, CUDA 13, sm_121, 128 GB unified).
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
# Forward hazard-cam path (world, Z-up; rover at origin facing +Y, sun az 135).
# A slow dolly FORWARD down the +Y traverse, closing on the hero boulder field
# (boulders live at y ~ 13..60), with a gentle lateral arc, a slow pan, an easing
# downward tilt toward the approaching hazards, and a subtle rover bob.
# Heights are RELATIVE to the scene ground_z (read from the authored HazardCam at
# runtime) so the camera tracks the surface instead of burying / floating.
# --------------------------------------------------------------------------- #
FWD_PATH = {
    "dolly_y": (2.2, 18.5),        # forward ~16 m toward the boulders
    "arc_x_amp": 2.4,              # lateral arc amplitude (0 -> +amp -> 0)
    "eye_h": (2.25, 1.95),         # eye height above ground_z (eases lower = drama)
    "eye_h_bob": 0.09,             # subtle vertical bob amplitude
    "eye_h_bob_cycles": 1.5,
    "yaw_deg": (5.0, -4.0),        # slow pan right -> left
    "pitch_deg": (-9.5, -6.5),     # ease the tilt up toward the boulders
}

# Third-person PiP orbit (world). A gentle reveal of the VIPER, kept in the
# front-right sunlit quadrant (sun comes from +X/-Y). Camera rides a circle of
# radius R around the rover, looking at the rover mid-body.
PIP_ORBIT = {
    "radius": 11.5,
    "theta_deg": (-62.0, -22.0),   # azimuth sweep (from +X axis); -Y/+X = sunlit
    "eye_h": (2.7, 2.35),          # height above ground_z
    "look_h": 1.15,                # look at rover mid-body height above ground_z
}


def _ease(t: float) -> float:
    """Cosine ease-in-out on t in [0,1] -> [0,1]."""
    return 0.5 - 0.5 * math.cos(math.pi * t)


def _fwd_pose_at(t: float, ground_z: float, idx=None) -> dict:
    y0, y1 = FWD_PATH["dolly_y"]
    yaw0, yaw1 = FWD_PATH["yaw_deg"]
    p0, p1 = FWD_PATH["pitch_deg"]
    h0, h1 = FWD_PATH["eye_h"]
    xamp = FWD_PATH["arc_x_amp"]
    bob = FWD_PATH["eye_h_bob"]
    cyc = FWD_PATH["eye_h_bob_cycles"]

    te = _ease(t)
    ex = xamp * math.sin(math.pi * t)
    ey = y0 + (y1 - y0) * te
    eh = h0 + (h1 - h0) * te + bob * math.sin(2.0 * math.pi * cyc * t)
    ez = ground_z + eh
    yaw = yaw0 + (yaw1 - yaw0) * t
    pitch = p0 + (p1 - p0) * te
    az = math.radians(yaw)
    pr = math.radians(pitch)
    fwd = [math.cos(pr) * math.sin(az), math.cos(pr) * math.cos(az), math.sin(pr)]
    return {"idx": idx, "t": round(t, 6),
            "eye": [round(ex, 5), round(ey, 5), round(ez, 5)],
            "fwd": [round(c, 6) for c in fwd],
            "yaw_deg": round(yaw, 4), "pitch_deg": round(pitch, 4)}


def _pip_pose_at(t: float, ground_z: float, idx=None) -> dict:
    th0, th1 = PIP_ORBIT["theta_deg"]
    h0, h1 = PIP_ORBIT["eye_h"]
    R = PIP_ORBIT["radius"]
    te = _ease(t)
    th = math.radians(th0 + (th1 - th0) * te)
    ex = R * math.cos(th)
    ey = R * math.sin(th)
    ez = ground_z + (h0 + (h1 - h0) * te)
    look = [0.0, 0.0, ground_z + PIP_ORBIT["look_h"]]
    fwd = [look[0] - ex, look[1] - ey, look[2] - ez]
    n = math.sqrt(sum(c * c for c in fwd)) or 1.0
    fwd = [c / n for c in fwd]
    return {"idx": idx, "t": round(t, 6),
            "eye": [round(ex, 5), round(ey, 5), round(ez, 5)],
            "fwd": [round(c, 6) for c in fwd]}


def compute_path(cam: str, n_frames: int, ground_z: float) -> list[dict]:
    fn = _fwd_pose_at if cam == "forward" else _pip_pose_at
    if n_frames <= 1:
        return [fn(0.0, ground_z, 0)]
    return [fn(k / (n_frames - 1), ground_z, k) for k in range(n_frames)]


def compute_lead_in(cam: str, n_lead: int, ground_z: float) -> list[dict]:
    """A continuous lead-in that warms the RTX temporal pipeline before pose 0.
    Ends EXACTLY at pose 0 (no seam). Big lateral/yaw flow fills DLSS history;
    forward-only motion warms too little. Frames are discarded."""
    if n_lead <= 0:
        return []
    p0 = compute_path(cam, 2, ground_z)[0]
    ex0, ey0, ez0 = p0["eye"]
    f0 = p0["fwd"]
    out = []
    for i in range(n_lead):
        f = i / n_lead
        decay = 1.0 - f
        ex = ex0 + 9.0 * math.sin(2.0 * math.pi * f) * decay
        ey = ey0 - 11.0 * decay + 11.0 * decay * f
        ez = ez0 + 0.6 * math.sin(2.0 * math.pi * f) * decay
        out.append({"idx": None, "t": None,
                    "eye": [round(ex, 5), round(ey, 5), round(ez, 5)],
                    "fwd": [round(c, 6) for c in f0]})
    return out


# =========================================================================== #
# Stage A — render RTX flythrough RGB frames (Isaac Sim).
# =========================================================================== #
def _world_translate(prim):
    from pxr import UsdGeom, Usd, Gf
    m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    t = m.ExtractTranslation()
    return float(t[0]), float(t[1]), float(t[2])


def cmd_render(args: argparse.Namespace) -> None:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    cam = args.cam
    out = os.path.abspath(args.out)
    rgb_dir = os.path.join(out, "rgb_%s" % cam)
    os.makedirs(rgb_dir, exist_ok=True)

    n = int(args.frames)
    W, H = int(args.width), int(args.height)

    t0 = time.time()

    def log(msg: str) -> None:
        print(">>> [render:%s] %s  (+%.1fs)" % (cam, msg, time.time() - t0), flush=True)
        sys.stderr.flush()

    log("config: seed=%d cam=%s frames=%d %dx%d renderer=%s subframes=%d out=%s"
        % (args.seed, cam, n, W, H, args.renderer, args.subframes, out))

    argv0 = sys.argv[0] if sys.argv else "render_v3_flythrough.py"
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

    # v3 LOOK: auto-exposure OFF (the approved v3 stage is lit purely by sun
    # intensity + albedo; auto-exposure clips the sunlit surface to white and
    # drifts/flickers during the move). NO film-iso override — match the approved
    # build_lunar_stage_v3 main() / render_v3_viper_seg settings exactly.
    settings = carb.settings.get_settings()
    for key in ("/rtx/post/histogram/enabled",
                "/rtx/post/tonemap/enableAutoExposure",
                "/rtx/post/eyeAdaptation/enabled"):
        try:
            settings.set(key, False)
        except Exception:
            pass
    if args.renderer.lower() == "pathtracing":
        try:
            settings.set("/rtx/rendermode", "PathTracing")
            settings.set("/rtx/pathtracing/spp", 1)
            settings.set("/rtx/pathtracing/totalSpp", int(args.subframes))
            settings.set("/rtx/pathtracing/maxBounces", 5)
        except Exception as e:  # noqa: BLE001
            log("pathtracing-settings note: %r" % e)
    log("replicator ready (renderer=%s, auto-exposure OFF)" % args.renderer)

    import numpy as np
    from PIL import Image
    from scene.build_lunar_stage_v3 import build_lunar_stage_v3, _basis_matrix, _set_transform

    omni.usd.get_context().new_stage()
    cams = build_lunar_stage_v3(args.seed, {})
    cam_map = dict(cams)
    for _ in range(25):
        sim.update()
    log("v3 stage authored: cams=%s" % cam_map)

    stage = omni.usd.get_context().get_stage()
    # Anchor heights from the authored cameras (HazardCam mast ~ ground_z + 2.2).
    haz_prim = stage.GetPrimAtPath(cam_map.get("hazard", "/World/HazardCam"))
    hx, hy, hz = _world_translate(haz_prim)
    ground_z = hz - 2.2
    log("anchor: hazardcam=(%.2f,%.2f,%.2f) -> ground_z=%.3f" % (hx, hy, hz, ground_z))

    # Persist the move BEFORE the heavy capture (reproducibility + crash safety).
    path = compute_path(cam, n, ground_z)
    meta = {
        "stage": "A/render", "cam": cam, "seed": args.seed,
        "frames": n, "fps": args.fps, "width": W, "height": H,
        "renderer": args.renderer, "rt_subframes": args.subframes,
        "ground_z": ground_z,
        "fwd_path_spec": FWD_PATH, "pip_orbit_spec": PIP_ORBIT,
        "camera_path": path,
    }
    with open(os.path.join(out, "camera_path_%s.json" % cam), "w") as fh:
        json.dump(meta, fh, indent=2)

    # We MOVE a dedicated free camera (not the authored one) so the authored
    # camera prims stay put (the PiP needs the rover/HazardCam untouched).
    from pxr import UsdGeom, Gf
    free_path = "/World/FlyCam"
    aspect = float(W) / float(H)
    haperture = 36.0
    hfov = 60.0 if cam == "forward" else 46.0
    focal = haperture / (2.0 * math.tan(math.radians(hfov) / 2.0))
    flycam = UsdGeom.Camera.Define(stage, free_path)
    flycam.CreateFocalLengthAttr(float(focal))
    flycam.CreateHorizontalApertureAttr(float(haperture))
    flycam.CreateVerticalApertureAttr(float(haperture / aspect))
    flycam.CreateClippingRangeAttr(Gf.Vec2f(0.02, 6000.0))
    cam_prim = flycam.GetPrim()

    rp = rep.create.render_product(free_path, (W, H))
    ldr = rep.AnnotatorRegistry.get_annotator("LdrColor")
    ldr.attach(rp)
    log("render product + LdrColor annotator attached (FlyCam, hfov=%.0f)" % hfov)

    n_warm = int(args.warmup_updates)
    n_conv = int(args.converge_updates)

    def _capture_rgb():
        d = np.asarray(ldr.get_data())
        return d[..., :3].astype(np.uint8)

    lead = compute_lead_in(cam, max(12, int(args.prime_steps)), ground_z)
    all_poses = [(False, q) for q in lead] + [(True, q) for q in path]
    for _ in range(20):
        sim.update()

    per_frame, means = [], []
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
            for _ in range(n_conv):
                sim.update()
            rgb = _capture_rgb()
            m = float(rgb.mean())
            if not is_hero:
                continue
            if not saving:
                if m <= float(args.lit_threshold):
                    continue
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
        with open(os.path.join(out, "camera_path_%s.json" % cam), "w") as fh:
            json.dump(meta, fh, indent=2)
        log("closing (status=%s, %d/%d frames, %.1f min, rgb_mean %s..%s)"
            % (status, have, n, (time.time() - t0) / 60.0,
               meta["result"]["rgb_mean_min"], meta["result"]["rgb_mean_max"]))
        sim.close()
    print("RENDER_DONE cam=%s status=%s out=%s frames=%d" % (cam, status, rgb_dir, have), flush=True)


# =========================================================================== #
# Stage B — overlay live hazard predictions + PiP inset + branding (PyTorch).
# =========================================================================== #
_ROCK_RGB = (236, 40, 36)        # VIVID hazard red
_ROCK_EDGE_RGB = (255, 150, 110)  # bright detection outline
_SAFE_RGB = (54, 200, 104)       # safe-traverse green

_BRAND = "CHAOTIC  CURIOSITY"
_TAGLINE = "regolith · lunar hazard segmentation"
_CAPTION_1 = "Synthetic-trained hazard segmentation · live overlay"
_CAPTION_2 = "VIPER on v3 cratered regolith · trained + rendered on one DGX Spark"
_MODEL_TAG = "SegFormer-B0 · domain-randomized · rock-IoU 0.887"
_PIP_LABEL = "THIRD-PERSON · VIPER"


def _find_fonts():
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
    """rock(1) -> VIVID red + bright outline; regolith(0) -> subtle green; sky(2) untouched."""
    import numpy as np
    from scipy import ndimage
    out = rgb.astype(np.float32)
    rock_m = mask == 1
    reg_m = mask == 0
    if reg_m.any():
        out[reg_m] = (1.0 - reg_alpha) * out[reg_m] + reg_alpha * np.array(_SAFE_RGB, np.float32)
    if rock_m.any():
        out[rock_m] = (1.0 - rock_alpha) * out[rock_m] + rock_alpha * np.array(_ROCK_RGB, np.float32)
        eroded = ndimage.binary_erosion(rock_m, iterations=2)
        edge = ndimage.binary_dilation(rock_m & ~eroded, iterations=1)
        out[edge] = np.array(_ROCK_EDGE_RGB, np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def composite_pip(base_img, pip_img, scale=0.30, margin_frac=0.022):
    """Paste the third-person PiP into the TOP-RIGHT corner with a thin border."""
    from PIL import Image, ImageDraw, ImageOps
    W, Hh = base_img.size
    iw = int(W * scale)
    ih = int(iw * pip_img.size[1] / pip_img.size[0])
    inset = pip_img.resize((iw, ih), Image.LANCZOS)
    bw = max(2, int(W * 0.0016))
    inset = ImageOps.expand(inset, border=bw, fill=(232, 232, 236))
    inset = ImageOps.expand(inset, border=bw, fill=(0, 0, 0))
    mx = int(W * margin_frac)
    my = int(W * margin_frac)
    x = W - inset.size[0] - mx
    y = my + int(Hh * 0.085)   # below the top wordmark band
    # soft drop shadow
    shadow = Image.new("RGBA", base_img.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    sd.rectangle([x + bw, y + bw, x + inset.size[0] + bw, y + inset.size[1] + bw],
                 fill=(0, 0, 0, 120))
    base_img = Image.alpha_composite(base_img.convert("RGBA"), shadow)
    base_img.paste(inset, (x, y))
    d = ImageDraw.Draw(base_img)
    reg, bold = _find_fonts()
    s = Hh / 1080.0
    lf = _font(bold, max(11, int(17 * s)))
    tw = d.textlength(_PIP_LABEL, font=lf)
    lx = x + (inset.size[0] - tw) / 2
    ly = y + inset.size[1] + int(4 * s)
    d.text((lx, ly), _PIP_LABEL, font=lf, fill=(228, 228, 232, 255))
    return base_img.convert("RGB")


def add_branding(img, fonts, frame_idx: int, n_frames: int):
    from PIL import Image, ImageDraw
    import numpy as np
    reg_font, bold_font = fonts
    base = img.convert("RGBA")
    W, Hh = base.size
    s = Hh / 1080.0

    def px(v):
        return max(1, int(round(v * s)))

    grad_h = int(0.26 * Hh)
    g = np.zeros((grad_h, W, 4), dtype=np.uint8)
    ramp = np.linspace(0.0, 1.0, grad_h) ** 1.6
    g[..., 3] = (ramp[:, None] * 165).astype(np.uint8)
    base.alpha_composite(Image.fromarray(g, "RGBA"), (0, Hh - grad_h))
    tg = np.zeros((int(0.12 * Hh), W, 4), dtype=np.uint8)
    tramp = np.linspace(1.0, 0.0, tg.shape[0]) ** 1.6
    tg[..., 3] = (tramp[:, None] * 120).astype(np.uint8)
    base.alpha_composite(Image.fromarray(tg, "RGBA"), (0, 0))

    draw = ImageDraw.Draw(base)
    accent = (235, 96, 70, 255)
    white = (240, 240, 242, 255)
    dim = (196, 198, 205, 255)
    mx = px(46)
    wf = _font(bold_font, px(34))
    tf = _font(reg_font, px(19))
    draw.text((mx, px(40)), _BRAND, font=wf, fill=white)
    draw.text((mx, px(40) + px(40)), _TAGLINE, font=tf, fill=dim)
    draw.rectangle([mx, px(40) + px(70), mx + px(58), px(40) + px(70) + px(4)], fill=accent)

    cf = _font(bold_font, px(24))
    cf2 = _font(reg_font, px(20))
    by = Hh - px(108)
    draw.text((mx, by), _CAPTION_1, font=cf, fill=white)
    draw.text((mx, by + px(34)), _CAPTION_2, font=cf2, fill=dim)

    lf = _font(reg_font, px(20))
    sw = px(20)
    gap = px(10)
    items = [("ROCK / HAZARD", _ROCK_RGB), ("SAFE REGOLITH", _SAFE_RGB)]
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
    return base.convert("RGB")


def cmd_overlay(args: argparse.Namespace) -> None:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    import numpy as np
    import torch
    from PIL import Image
    from eval._infer import load_model, predict

    rgb_dir = os.path.abspath(args.rgb_dir)
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    rgb_paths = sorted(glob.glob(os.path.join(rgb_dir, "**", "rgb*.png"), recursive=True))
    if not rgb_paths:
        raise SystemExit("no rgb*.png frames under %s" % rgb_dir)
    if int(args.skip_head) > 0:
        rgb_paths = rgb_paths[int(args.skip_head):]

    pip_paths = []
    if args.pip_dir and os.path.isdir(args.pip_dir):
        pip_paths = sorted(glob.glob(os.path.join(os.path.abspath(args.pip_dir), "**", "rgb*.png"),
                                     recursive=True))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_model(args.checkpoint, device)
    fonts = _find_fonts()
    print(">>> [overlay] model=%s device=%s frames=%d pip=%d infer=%dx%d"
          % (ckpt.get("model_name", "?"), device, len(rgb_paths), len(pip_paths),
             args.infer_width, args.infer_height), flush=True)

    iw, ih = int(args.infer_width), int(args.infer_height)
    n = len(rgb_paths)
    frame_meta = []
    t0 = time.time()
    for k, p in enumerate(rgb_paths):
        rgb = np.array(Image.open(p).convert("RGB"))
        Hh, Ww = rgb.shape[:2]
        small = np.array(Image.fromarray(rgb).resize((iw, ih), Image.BILINEAR))
        mask_small = predict(model, small, device).astype(np.uint8)
        mask = np.array(Image.fromarray(mask_small).resize((Ww, Hh), Image.NEAREST))
        disp = np.clip(rgb.astype(np.float32) * float(args.display_gain), 0, 255).astype(np.uint8)
        comp = hazard_overlay(disp, mask, args.rock_alpha, args.reg_alpha)
        comp_img = Image.fromarray(comp)
        if pip_paths:
            j = min(k, len(pip_paths) - 1)
            try:
                pip_img = Image.open(pip_paths[j]).convert("RGB")
                pip_disp = np.clip(np.array(pip_img).astype(np.float32) * float(args.display_gain),
                                   0, 255).astype(np.uint8)
                comp_img = composite_pip(comp_img, Image.fromarray(pip_disp), scale=float(args.pip_scale))
            except Exception as e:  # noqa: BLE001
                if k == 0:
                    print(">>> [overlay] PiP composite failed: %r (continuing without)" % e, flush=True)
                pip_paths = []  # fall back cleanly for the rest
        branded = add_branding(comp_img, fonts, k, n)
        branded.save(os.path.join(out_dir, "overlay_%05d.png" % k))
        fr = {"idx": k, "src": os.path.basename(p),
              "rock_frac": round(float((mask == 1).mean()), 5),
              "regolith_frac": round(float((mask == 0).mean()), 5),
              "sky_frac": round(float((mask == 2).mean()), 5)}
        frame_meta.append(fr)
        if k == 0 or (k + 1) % 20 == 0 or k == n - 1:
            print(">>> [overlay] %05d/%d rock=%.3f (%.2f s/frame)"
                  % (k + 1, n, fr["rock_frac"], (time.time() - t0) / (k + 1)), flush=True)

    with open(os.path.join(out_dir, "overlay_frames.json"), "w") as fh:
        json.dump({"checkpoint": os.path.abspath(args.checkpoint),
                   "model_name": ckpt.get("model_name"), "infer_res": [iw, ih],
                   "rock_alpha": args.rock_alpha, "reg_alpha": args.reg_alpha,
                   "pip": bool(pip_paths), "n_frames": n,
                   "mean_rock_frac": round(float(np.mean([f["rock_frac"] for f in frame_meta])), 5),
                   "frames": frame_meta}, fh, indent=2)
    print("OVERLAY_DONE out=%s frames=%d (%.1f min)"
          % (out_dir, n, (time.time() - t0) / 60.0), flush=True)


# =========================================================================== #
# Stage C — assemble MP4 + stills + web preview + GIF (ffmpeg).
# =========================================================================== #
def _run(cmd: list[str]) -> None:
    print(">>> [assemble] $ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _select_stills(overlay_dir: str, n_stills: int) -> list[int]:
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
            back = [f for f in frames if f["idx"] >= n // 2] or frames
            picks.append(max(back, key=lambda f: f["rock_frac"])["idx"])
    for frac in (0.18, 0.5, 0.82):
        picks.append(int(round(frac * (n - 1))))
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
        raise SystemExit("no overlay_*.png under %s" % overlay_dir)
    n = len(overlays)
    pattern = os.path.join(overlay_dir, "overlay_%05d.png")
    fps = int(args.fps)

    full_mp4 = os.path.join(out_root, "regolith_v3_flythrough.mp4")
    _run(["ffmpeg", "-y", "-framerate", str(fps), "-i", pattern,
          "-c:v", "libx264", "-preset", "slow", "-crf", "17",
          "-pix_fmt", "yuv420p", "-movflags", "+faststart", full_mp4])

    still_idx = _select_stills(overlay_dir, int(args.n_stills))
    still_paths = []
    for n_out, i in enumerate(still_idx, start=1):
        src = pattern % i
        dst = os.path.join(stills_dir, "render-v3-hero-%d.png" % n_out)
        _run(["ffmpeg", "-y", "-i", src, "-vf", "scale=%d:-1" % int(args.still_width), dst])
        still_paths.append(dst)

    preview = os.path.join(out_root, "render-v3-preview.mp4")
    _run(["ffmpeg", "-y", "-framerate", str(fps), "-i", pattern,
          "-vf", "scale=%d:-2" % int(args.preview_width),
          "-c:v", "libx264", "-preset", "slow", "-crf", str(int(args.preview_crf)),
          "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", preview])

    gif = os.path.join(out_root, "render-v3-preview.gif")
    gfps = max(8, fps // 2)
    palette = os.path.join(out_root, ".palette.png")
    vf = "fps=%d,scale=%d:-1:flags=lanczos" % (gfps, int(args.gif_width))
    try:
        _run(["ffmpeg", "-y", "-framerate", str(fps), "-i", pattern,
              "-vf", vf + ",palettegen=stats_mode=diff", palette])
        _run(["ffmpeg", "-y", "-framerate", str(fps), "-i", pattern, "-i", palette,
              "-lavfi", vf + " [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=3", gif])
    except Exception as e:  # noqa: BLE001
        print(">>> [assemble] gif skipped: %r" % e, flush=True)
        gif = None

    def _mb(path):
        return round(os.path.getsize(path) / 1e6, 2) if path and os.path.isfile(path) else None

    summary = {"n_frames": n, "fps": fps,
               "full_mp4": full_mp4, "full_mp4_MB": _mb(full_mp4),
               "preview_mp4": preview, "preview_MB": _mb(preview),
               "gif": gif, "gif_MB": _mb(gif),
               "stills": still_paths, "still_indices": still_idx}
    with open(os.path.join(out_root, "assemble_v3_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print("ASSEMBLE_DONE " + json.dumps(summary), flush=True)


# =========================================================================== #
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="v3 cinematic RTX lunar flythrough (VIPER + PiP + live hazard overlay).")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("render", help="Stage A: render RTX flythrough RGB (Isaac Sim).")
    a.add_argument("--out", required=True)
    a.add_argument("--cam", choices=["forward", "pip"], default="forward")
    a.add_argument("--seed", type=int, default=7)
    a.add_argument("--frames", type=int, default=192)
    a.add_argument("--fps", type=int, default=24)
    a.add_argument("--width", type=int, default=1280)
    a.add_argument("--height", type=int, default=720)
    a.add_argument("--renderer", default="RayTracedLighting", choices=["RayTracedLighting", "PathTracing"])
    a.add_argument("--subframes", type=int, default=48)
    a.add_argument("--prime-steps", type=int, default=16)
    a.add_argument("--lit-threshold", type=float, default=6.0)
    a.add_argument("--warmup-updates", type=int, default=8)
    a.add_argument("--converge-updates", type=int, default=40)
    a.set_defaults(func=cmd_render)

    b = sub.add_parser("overlay", help="Stage B: hazard overlay + PiP + branding (PyTorch).")
    b.add_argument("--checkpoint", required=True)
    b.add_argument("--rgb-dir", required=True)
    b.add_argument("--pip-dir", default="")
    b.add_argument("--out", required=True)
    b.add_argument("--infer-width", type=int, default=1024)
    b.add_argument("--infer-height", type=int, default=576)
    b.add_argument("--rock-alpha", type=float, default=0.60)
    b.add_argument("--reg-alpha", type=float, default=0.14)
    b.add_argument("--pip-scale", type=float, default=0.30)
    b.add_argument("--display-gain", type=float, default=1.0)
    b.add_argument("--skip-head", type=int, default=0)
    b.set_defaults(func=cmd_overlay)

    c = sub.add_parser("assemble", help="Stage C: MP4 + stills + preview + GIF (ffmpeg).")
    c.add_argument("--overlay-dir", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--fps", type=int, default=24)
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
