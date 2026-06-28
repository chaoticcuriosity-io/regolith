"""Generate the seamless tiling lunar-regolith tangent-space NORMAL map used by
build_lunar_stage_v3.py (the regolith material's UsdUVTexture -> normal input).

Content: fine granular bump field (periodic fBm) + scattered micro-pits (small
craters: bowl + raised rim) + tiny pebble bumps. Periodic, so it tiles seamlessly;
the scene tiles it every V3_REGOLITH_TILE metres across the terrain.

Usage:
    python scene/make_regolith_normal.py [out.png]
Default output: <repo>/assets/regolith_normal.png
"""
import os
import sys
import numpy as np
from PIL import Image

N = 1024          # texture resolution
TILE_M = 5.0      # physical size the tile represents (metres) — match V3_REGOLITH_TILE
rng = np.random.RandomState(20260628)


def periodic_fbm(n, octaves, base_period, gain=0.5, seed=0):
    """Band-limited periodic noise via FFT of white noise (guarantees seamless wrap)."""
    out = np.zeros((n, n), np.float64)
    amp = 1.0
    fy = np.fft.fftfreq(n) * n
    fx = np.fft.fftfreq(n) * n
    FX, FY = np.meshgrid(fx, fy)
    R = np.sqrt(FX ** 2 + FY ** 2)
    norm = 0.0
    for o in range(octaves):
        kc = max(1.0, n / (base_period * (2.0 ** o)))
        r = np.random.RandomState(seed + o * 17)
        W = np.fft.fft2(r.randn(n, n))
        band = np.exp(-((R - kc) ** 2) / (2.0 * (0.5 * kc) ** 2 + 1e-6))
        f = np.real(np.fft.ifft2(W * band))
        f /= (f.std() + 1e-9)
        out += amp * f
        norm += amp
        amp *= gain
    return out / norm


def build():
    h = 0.55 * periodic_fbm(N, 4, base_period=24.0, seed=1)     # fine grain
    h += 0.9 * periodic_fbm(N, 3, base_period=128.0, seed=50)   # medium clumps
    yy, xx = np.mgrid[0:N, 0:N].astype(np.float64)
    # micro-pits (bowl + raised rim), periodic
    for _ in range(140):
        cx = rng.uniform(0, N); cy = rng.uniform(0, N)
        R = rng.uniform(6, 34); depth = R * rng.uniform(0.10, 0.22)
        dx = np.abs(xx - cx); dx = np.minimum(dx, N - dx)
        dy = np.abs(yy - cy); dy = np.minimum(dy, N - dy)
        d = np.sqrt(dx ** 2 + dy ** 2); dn = d / R
        h += np.where(dn < 1.0, -depth * (1.0 - dn ** 2), 0.0)
        h += (depth * 0.45) * np.exp(-(((d - R) / (0.35 * R)) ** 2))
    # tiny pebble bumps
    for _ in range(400):
        cx = rng.uniform(0, N); cy = rng.uniform(0, N); R = rng.uniform(1.5, 5.0)
        dx = np.abs(xx - cx); dx = np.minimum(dx, N - dx)
        dy = np.abs(yy - cy); dy = np.minimum(dy, N - dy)
        d = np.sqrt(dx ** 2 + dy ** 2)
        h += (R * 0.25) * np.exp(-(d ** 2) / (2 * (0.6 * R) ** 2))
    # height -> tangent-space normal (mostly-up, gentle tactile perturbation)
    h = (h - h.mean()) / (h.std() + 1e-9)
    strength = 0.9
    gy, gx = np.gradient(h)
    nx = -gx * strength; ny = -gy * strength; nz = np.ones_like(h)
    ln = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
    nx /= ln; ny /= ln; nz /= ln
    rgb = np.stack([
        ((nx * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8),
        ((ny * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8),
        ((nz * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8),
    ], axis=-1)
    return rgb


def main():
    if len(sys.argv) > 1:
        out = sys.argv[1]
    else:
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out = os.path.join(repo, "assets", "regolith_normal.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    Image.fromarray(build(), "RGB").save(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
