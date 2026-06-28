"""
build_lunar_stage_v3.py — v3 "make it real" lunar scene + procedural rover.

Branched from scene/build_lunar_stage.py (v2). The v2 scene had good realistic
ROCK geometry but a FLAT, SMOOTH, untextured ground, only small/similar rocks, and
a flat-ish look. v3 is a fidelity jump aimed at a beauty-plate look test (NO dataset
/ training yet — this file authors a scene and renders TEST STILLS only):

  1. REGOLITH GROUND (biggest change): high-resolution heightfield carrying a
     MULTI-SCALE CRATER field (a few big + many medium + many small, each a bowl
     interior + raised rim) plus rolling hills, fBm surface roughness, and fine
     grain — all carried in GEOMETRY. Dark lunar regolith PBR (albedo ~0.10–0.12,
     high roughness) plus a tiling tangent-space REGOLITH NORMAL MAP (UsdUVTexture
     -> normal, world-XY st) that adds tactile sub-mesh-res relief the heightfield
     can't carry (pits/grain that catch the raking sun). A rough/cratered ground is
     both prettier AND the sim-to-real fix (v2's smooth ground let the model learn
     "rough=rock").
  2. ROCKS — POWER-LAW size distribution: thousands of instances spanning >2 orders
     of magnitude — MANY tiny pebbles, fewer small/mid rocks (a USD PointInstancer
     over a pool of noise-displaced basalt prototypes), and a FEW large hand-placed
     in-frame BOULDERS (individual high-subdiv displaced meshes).
  3. LIGHTING: harsh LOW sun (~12° elevation), long hard-edged shadows, pure black
     sky with crisp stars, high contrast — real airless-body lighting. A small,
     tasteful Earth low on the horizon (optional).
  4. ROVER — PREMIUM real-model path: references an external rover USD (NASA VIPER
     — Volatiles Investigating Polar Exploration Rover, the lunar rover — full PBR:
     gold MLI body, 4 legged wheels, the signature VERTICAL side solar panels, a
     tall front NavCam mast + headlights, and the TRIDENT drill), stood on the
     surface and faced down-traverse. (No CC0/public-domain VIPER mesh exists for
     download, so the asset is authored from primitives in Blender to VIPER's
     documented configuration — see scene/viper_build.py; same glb/Blender->USD
     route.) Falls back to a procedural rover if the asset is absent, so the script
     runs anywhere. The biggest in-frame BOULDERS are fractured/angular dark basalt
     (rock material hue is locked grey — no pink tint).

Cameras authored:
  /World/BeautyCam   — third-person hero: rover on the cratered surface among boulders.
  /World/BeautyCam2  — alternate low/opposite-side hero angle.
  /World/HazardCam   — rover-perspective forward view (future main hazard-cam plate).

Run (on the DGX Spark, inside nvcr.io/nvidia/isaac-sim:6.0.0 via run_isaac.sh-style
mount) — renders one 1920x1080 RGB beauty plate per camera into <out>/<camname>/:

  /isaac-sim/python.sh build_lunar_stage_v3.py

Env knobs: V3_SEED, V3_W, V3_H, V3_SUBFRAMES, V3_RENDERER (RayTracedLighting|PathTracing),
V3_PATHTRACE (0/1), V3_PT_TOTALSPP, V3_PT_BOUNCES, V3_ROVER_USD, V3_ROVER_SCALE,
V3_REGOLITH_NORMAL, V3_REGOLITH_TILE, SMOKE_OUT (output dir). The showpiece stills
use path tracing (soft contact shadows + regolith GI bounce); the dataset will render
the same geometry/materials under faster RayTracedLighting.

RTX descriptor-leak rule (see scripts/run_isaac.sh + dgx-spark manual): fresh
container, poll-for-files drain (NOT bare wait_until_complete), ≤300 frames/session.
This renders 3 stills — far under budget.
"""

from __future__ import annotations

import math

CLASS_MAP = {"regolith": 0, "rock": 1, "sky": 2}
IGNORE_INDEX = 255


def _resolve_asset(env_var, default_path, rel_path):
    """Resolve an external asset path. Order: $env_var -> default_path (the canonical
    Spark location) -> <script_dir>/../rel_path (repo-relative, for checkouts off the
    Spark). Returns the first that exists, else default_path so the caller's isfile()
    check drives the graceful fallback (procedural rover / no normal map)."""
    import os
    p = os.environ.get(env_var, "")
    if p:
        return p
    if os.path.isfile(default_path):
        return default_path
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        cand = os.path.normpath(os.path.join(here, "..", rel_path))
        if os.path.isfile(cand):
            return cand
    except Exception:
        pass
    return default_path


# =========================================================================== #
# Procedural noise (numpy). Lazy imports so importing this module outside the
# Isaac container does not require numpy/pxr.
# =========================================================================== #
def _hash2(ix, iy, seed):
    import numpy as np
    ix = ix.astype(np.int64)
    iy = iy.astype(np.int64)
    n = (ix * np.int64(73856093)) ^ (iy * np.int64(19349663)) ^ np.int64(int(seed) * 83492791)
    n = (n ^ (n >> np.int64(13))) * np.int64(1274126177)
    n = n & np.int64(0x7FFFFFFF)
    return n.astype(np.float64) / float(0x7FFFFFFF)


def _value_noise_2d(X, Y, seed, freq):
    import numpy as np
    xf = X * freq
    yf = Y * freq
    x0 = np.floor(xf).astype(np.int64)
    y0 = np.floor(yf).astype(np.int64)
    tx = xf - x0
    ty = yf - y0
    sx = tx * tx * (3.0 - 2.0 * tx)
    sy = ty * ty * (3.0 - 2.0 * ty)
    v00 = _hash2(x0, y0, seed)
    v10 = _hash2(x0 + 1, y0, seed)
    v01 = _hash2(x0, y0 + 1, seed)
    v11 = _hash2(x0 + 1, y0 + 1, seed)
    a = v00 * (1.0 - sx) + v10 * sx
    b = v01 * (1.0 - sx) + v11 * sx
    return a * (1.0 - sy) + b * sy


def _fractal_noise_2d(X, Y, seed, octaves=5, base_freq=0.012, lacunarity=2.0, gain=0.5):
    import numpy as np
    total = np.zeros_like(X, dtype=np.float64)
    amp = 1.0
    freq = base_freq
    norm = 0.0
    for o in range(octaves):
        total += amp * _value_noise_2d(X, Y, seed + o * 101, freq)
        norm += amp
        amp *= gain
        freq *= lacunarity
    return (total / norm) * 2.0 - 1.0


def _icosphere(subdiv):
    import numpy as np
    t = (1.0 + 5.0 ** 0.5) / 2.0
    verts = [
        (-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
        (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
        (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1),
    ]
    verts = [tuple((np.array(v) / np.linalg.norm(v)).tolist()) for v in verts]
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ]
    midcache = {}

    def midpoint(a, b):
        key = (a, b) if a < b else (b, a)
        if key in midcache:
            return midcache[key]
        va = np.array(verts[a]); vb = np.array(verts[b])
        vm = (va + vb) / 2.0
        vm = vm / np.linalg.norm(vm)
        verts.append(tuple(vm.tolist()))
        idx = len(verts) - 1
        midcache[key] = idx
        return idx

    for _ in range(int(subdiv)):
        newf = []
        for a, b, c in faces:
            ab = midpoint(a, b); bc = midpoint(b, c); ca = midpoint(c, a)
            newf += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        faces = newf
    return np.array(verts, dtype=np.float64), np.array(faces, dtype=np.int32)


def _hash3(ix, iy, iz, seed):
    import numpy as np
    ix = ix.astype(np.int64); iy = iy.astype(np.int64); iz = iz.astype(np.int64)
    n = (
        (ix * np.int64(73856093)) ^ (iy * np.int64(19349663)) ^ (iz * np.int64(83492791))
        ^ np.int64(int(seed) * 2654435761 & 0x7FFFFFFFFFFFFFFF)
    )
    n = (n ^ (n >> np.int64(13))) * np.int64(1274126177)
    n = n & np.int64(0x7FFFFFFF)
    return n.astype(np.float64) / float(0x7FFFFFFF)


def _value_noise_3d(P, seed, freq):
    import numpy as np
    pf = np.asarray(P, dtype=np.float64) * freq
    p0 = np.floor(pf).astype(np.int64)
    t = pf - p0
    s = t * t * (3.0 - 2.0 * t)
    x0, y0, z0 = p0[:, 0], p0[:, 1], p0[:, 2]
    sx, sy, sz = s[:, 0], s[:, 1], s[:, 2]

    def corner(dx, dy, dz):
        return _hash3(x0 + dx, y0 + dy, z0 + dz, seed)

    c000 = corner(0, 0, 0); c100 = corner(1, 0, 0)
    c010 = corner(0, 1, 0); c110 = corner(1, 1, 0)
    c001 = corner(0, 0, 1); c101 = corner(1, 0, 1)
    c011 = corner(0, 1, 1); c111 = corner(1, 1, 1)
    x00 = c000 * (1.0 - sx) + c100 * sx
    x10 = c010 * (1.0 - sx) + c110 * sx
    x01 = c001 * (1.0 - sx) + c101 * sx
    x11 = c011 * (1.0 - sx) + c111 * sx
    y0i = x00 * (1.0 - sy) + x10 * sy
    y1i = x01 * (1.0 - sy) + x11 * sy
    return y0i * (1.0 - sz) + y1i * sz


def _fbm_3d(P, seed, octaves=5, base_freq=1.0, lacunarity=2.0, gain=0.5):
    import numpy as np
    total = np.zeros(len(P), dtype=np.float64)
    amp = 1.0
    freq = base_freq
    norm = 0.0
    for o in range(int(octaves)):
        total += amp * _value_noise_3d(P, seed + o * 131, freq)
        norm += amp
        amp *= gain
        freq *= lacunarity
    return (total / norm) * 2.0 - 1.0


def _vertex_normals(points, faces):
    import numpy as np
    pts = np.asarray(points, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    n = np.zeros_like(pts)
    v0, v1, v2 = pts[f[:, 0]], pts[f[:, 1]], pts[f[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    for k in range(3):
        np.add.at(n, f[:, k], fn)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    ln[ln == 0.0] = 1.0
    return n / ln


def _displace_rock(base_dirs, rng, amp=0.34):
    """Deform a unit icosphere into an irregular, eroded lunar rock (v2 deformer)."""
    import numpy as np
    dirs = np.asarray(base_dirs, dtype=np.float64)
    off = rng.uniform(-41.0, 41.0, size=3)
    aniso = rng.uniform(0.70, 1.55, size=3)
    P = dirs * aniso[None, :] + off[None, :]
    seed_a = int(rng.randint(1, 1_000_000))
    seed_b = int(rng.randint(1, 1_000_000))
    bf = float(rng.uniform(1.05, 1.75))
    lump = _fbm_3d(P, seed_a, octaves=4, base_freq=bf * 0.85, lacunarity=2.05, gain=0.5)
    rid = _fbm_3d(P * 1.7, seed_b, octaves=4, base_freq=bf, lacunarity=2.2, gain=0.5)
    rid = 1.0 - np.abs(rid)
    rid = rid * rid
    med = _fbm_3d(P * 3.0, seed_a + 401, octaves=3, base_freq=bf * 1.7, lacunarity=2.3, gain=0.5)
    fine = _fbm_3d(P * 7.0, seed_b + 977, octaves=2, base_freq=bf * 2.7, lacunarity=2.5, gain=0.5)
    a_lump = float(amp) * rng.uniform(0.90, 1.15)
    a_rid = float(amp) * rng.uniform(0.26, 0.42)
    a_med = float(amp) * rng.uniform(0.12, 0.19)
    a_fine = float(amp) * rng.uniform(0.05, 0.09)
    radius = 1.0 + a_lump * lump + a_rid * (rid - 0.5) * 2.0 + a_med * med + a_fine * fine
    radius = np.clip(radius, 0.50, 1.85)
    return dirs * radius[:, None]


def _displace_rock_angular(base_dirs, rng, amp=0.40, n_facets=8):
    """Like _displace_rock but FRACTURED/ANGULAR — for the big hero boulders.

    Stronger, sharper ridges, then a handful of random cutting planes shear the
    outer shell flat, carving genuine angular facets (fractured basalt) instead of
    a rounded lump. Pair with flat (faceted) shading at author time."""
    import numpy as np
    dirs = np.asarray(base_dirs, dtype=np.float64)
    off = rng.uniform(-41.0, 41.0, size=3)
    aniso = rng.uniform(0.80, 1.45, size=3)
    P = dirs * aniso[None, :] + off[None, :]
    seed_a = int(rng.randint(1, 1_000_000))
    seed_b = int(rng.randint(1, 1_000_000))
    bf = float(rng.uniform(1.05, 1.6))
    lump = _fbm_3d(P, seed_a, octaves=4, base_freq=bf * 0.85, lacunarity=2.05, gain=0.5)
    rid = _fbm_3d(P * 1.7, seed_b, octaves=4, base_freq=bf, lacunarity=2.25, gain=0.5)
    rid = 1.0 - np.abs(rid)
    rid = rid ** 3                                  # sharper ridge crests
    med = _fbm_3d(P * 3.0, seed_a + 401, octaves=3, base_freq=bf * 1.7, lacunarity=2.3, gain=0.5)
    a_lump = float(amp) * rng.uniform(0.80, 1.00)
    a_rid = float(amp) * rng.uniform(0.45, 0.68)    # stronger ridges than rounded rock
    a_med = float(amp) * rng.uniform(0.10, 0.16)
    radius = 1.0 + a_lump * lump + a_rid * (rid - 0.5) * 2.0 + a_med * med
    radius = np.clip(radius, 0.55, 1.95)
    pts = dirs * radius[:, None]
    # Fracture: clip points beyond a few random planes back ONTO the plane -> facets.
    for _ in range(int(n_facets)):
        nrm = rng.normal(size=3)
        nrm = nrm / (np.linalg.norm(nrm) + 1e-9)
        proj = pts @ nrm
        d = float(rng.uniform(0.60, 0.90)) * float(proj.max())
        over = proj > d
        if np.any(over):
            pts[over] -= np.outer(proj[over] - d, nrm)
    return pts


def _sample_height(Zgrid, xs, ys, px, py):
    import numpy as np
    nx = len(xs); ny = len(ys)
    fx = (px - xs[0]) / (xs[-1] - xs[0]) * (nx - 1)
    fy = (py - ys[0]) / (ys[-1] - ys[0]) * (ny - 1)
    fx = min(max(fx, 0.0), nx - 1.000001)
    fy = min(max(fy, 0.0), ny - 1.000001)
    ix = int(np.floor(fx)); iy = int(np.floor(fy))
    tx = fx - ix; ty = fy - iy
    z00 = Zgrid[iy, ix]; z10 = Zgrid[iy, ix + 1]
    z01 = Zgrid[iy + 1, ix]; z11 = Zgrid[iy + 1, ix + 1]
    a = z00 * (1 - tx) + z10 * tx
    b = z01 * (1 - tx) + z11 * tx
    return float(a * (1 - ty) + b * ty)


# =========================================================================== #
# USD helpers (lazy pxr / omni imports).
# =========================================================================== #
def _add_semantics(prim, label):
    try:
        from isaacsim.core.utils.semantics import add_update_semantics
        add_update_semantics(prim, semantic_label=label, type_label="class")
        return
    except Exception:
        pass
    try:
        from omni.isaac.core.utils.semantics import add_update_semantics
        add_update_semantics(prim, semantic_label=label, type_label="class")
        return
    except Exception:
        pass
    from pxr import Semantics
    sem = Semantics.SemanticsAPI.Apply(prim, "Semantics")
    sem.CreateSemanticTypeAttr("class")
    sem.CreateSemanticDataAttr(label)


def _vec3f_array(np_pts):
    from pxr import Vt
    import numpy as np
    arr = np.asarray(np_pts, dtype=np.float32)
    try:
        return Vt.Vec3fArray.FromNumpy(arr)
    except Exception:
        try:
            return Vt.Vec3fArray(arr)
        except Exception:
            return Vt.Vec3fArray([tuple(float(c) for c in p) for p in arr.tolist()])


def _int_array(np_idx):
    from pxr import Vt
    import numpy as np
    arr = np.asarray(np_idx, dtype=np.int32)
    try:
        return Vt.IntArray.FromNumpy(arr)
    except Exception:
        try:
            return Vt.IntArray(arr)
        except Exception:
            return Vt.IntArray([int(i) for i in arr.tolist()])


def _author_mesh(stage, path, points, face_counts, face_indices, normals=None):
    from pxr import UsdGeom, Gf
    import numpy as np
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(_vec3f_array(points))
    mesh.CreateFaceVertexCountsAttr(_int_array(face_counts))
    mesh.CreateFaceVertexIndicesAttr(_int_array(face_indices))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    pmin = np.asarray(points, dtype=np.float64).min(axis=0)
    pmax = np.asarray(points, dtype=np.float64).max(axis=0)
    mesh.CreateExtentAttr([
        Gf.Vec3f(float(pmin[0]), float(pmin[1]), float(pmin[2])),
        Gf.Vec3f(float(pmax[0]), float(pmax[1]), float(pmax[2])),
    ])
    if normals is not None:
        mesh.CreateNormalsAttr(_vec3f_array(normals))
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    return mesh.GetPrim()


def _author_sky_dome(stage, path, radius, sun_dir, hole_deg, n_long=72, n_lat=36):
    import numpy as np
    lons = np.linspace(0.0, 2.0 * np.pi, n_long + 1)
    lats = np.linspace(-np.pi / 2.0, np.pi / 2.0, n_lat + 1)
    LO, LA = np.meshgrid(lons, lats)
    ux = np.cos(LA) * np.cos(LO)
    uy = np.cos(LA) * np.sin(LO)
    uz = np.sin(LA)
    unit = np.stack([ux.ravel(), uy.ravel(), uz.ravel()], axis=1)
    points = unit * radius
    s = np.asarray(sun_dir, dtype=np.float64)
    s = s / np.linalg.norm(s)
    cos_hole = np.cos(np.radians(hole_deg))
    stride = n_long + 1
    faces = []
    for i in range(n_lat):
        for j in range(n_long):
            a = i * stride + j
            b = a + 1
            c = a + stride + 1
            d = a + stride
            cdir = unit[a] + unit[b] + unit[c] + unit[d]
            cdir = cdir / np.linalg.norm(cdir)
            if float(np.dot(cdir, s)) > cos_hole:
                continue
            faces.append((a, b, c, d))
    face_indices = np.array(faces, dtype=np.int32).ravel()
    face_counts = np.full(len(faces), 4, dtype=np.int32)
    prim = _author_mesh(stage, path, points, face_counts, face_indices)
    from pxr import UsdGeom
    UsdGeom.Mesh(prim).CreateDoubleSidedAttr(True)
    return prim


def _make_preview_material(stage, path, diffuse, roughness, metallic=0.0, emissive=None, unlit=False):
    from pxr import UsdShade, Sdf, Gf
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*diffuse))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
    if unlit:
        shader.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(1)
        shader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0, 0, 0))
    if emissive is not None:
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emissive))
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def _make_regolith_material_normalmapped(stage, path, diffuse, roughness, emissive,
                                         normal_path, st_name="st"):
    """UsdPreviewSurface regolith material with a tiling tangent-space NORMAL map
    (UsdUVTexture -> normal). Adds tactile sub-mesh-res relief the heightfield can't
    carry. st is supplied by the bound mesh's primvars:st (world-XY / tile)."""
    from pxr import UsdShade, Sdf, Gf
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*diffuse))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    if emissive is not None:
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emissive))
    streader = UsdShade.Shader.Define(stage, path + "/stReader")
    streader.CreateIdAttr("UsdPrimvarReader_float2")
    streader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set(st_name)
    st_out = streader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    tex = UsdShade.Shader.Define(stage, path + "/NormalTex")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(normal_path)
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_out)
    tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
    tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
    tex.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2.0, 2.0, 2.0, 1.0))
    tex.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1.0, -1.0, -1.0, 0.0))
    tex_rgb = tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    shader.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(tex_rgb)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def _make_rock_material_pool(stage, base_path, rng, n, albedo_range, rough_range, emissive=0.0):
    import numpy as np
    mats = []
    lo, hi = float(albedo_range[0]), float(albedo_range[1])
    rlo, rhi = float(rough_range[0]), float(rough_range[1])
    # Neutral grey basalt — NO warm/pink hue. The earlier warm per-channel tint
    # ([1.035,1.0,0.955]) plus per-channel jitter pushed the lightest boulders
    # PINK. Lock hue to grey; jitter BRIGHTNESS only (same delta on all channels).
    tint = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    em = float(emissive)
    for k in range(int(n)):
        b = lo + (hi - lo) * float(rng.uniform(0.0, 1.0)) ** 1.6
        j = float(rng.uniform(-0.010, 0.010))
        diffuse = np.clip(b * tint + j, 0.02, 0.55)
        rough = float(rng.uniform(rlo, rhi))
        mat = _make_preview_material(
            stage, "%s/RockMat_%02d" % (base_path, k),
            diffuse=tuple(float(c) for c in diffuse), roughness=rough, metallic=0.0,
            emissive=((em, em, em) if em > 0 else None),
        )
        mats.append(mat)
    return mats


def _bind_material(prim, material):
    from pxr import UsdShade
    UsdShade.MaterialBindingAPI(prim).Bind(material)


def _basis_matrix(forward, translate, up=(0.0, 0.0, 1.0)):
    """Gf.Matrix4d for a frame whose local -Z points along ``forward``."""
    import numpy as np
    from pxr import Gf
    f = np.asarray(forward, dtype=np.float64)
    f = f / np.linalg.norm(f)
    u = np.asarray(up, dtype=np.float64)
    zc = -f
    if abs(float(np.dot(zc, u))) > 0.999:
        u = np.array([0.0, 1.0, 0.0])
    right = np.cross(u, zc)
    right /= np.linalg.norm(right)
    upc = np.cross(zc, right)
    t = np.asarray(translate, dtype=np.float64)
    return Gf.Matrix4d(
        float(right[0]), float(right[1]), float(right[2]), 0.0,
        float(upc[0]), float(upc[1]), float(upc[2]), 0.0,
        float(zc[0]), float(zc[1]), float(zc[2]), 0.0,
        float(t[0]), float(t[1]), float(t[2]), 1.0,
    )


def _set_transform(prim, matrix):
    from pxr import UsdGeom
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTransformOp().Set(matrix)


def _author_box(stage, path, dims, matrix, material=None):
    """Unit cube scaled to dims=(dx,dy,dz), then placed by ``matrix`` (applied after scale)."""
    from pxr import UsdGeom, Gf
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.CreateExtentAttr([Gf.Vec3f(-0.5, -0.5, -0.5), Gf.Vec3f(0.5, 0.5, 0.5)])
    dx, dy, dz = (float(d) for d in dims)
    m_scale = Gf.Matrix4d().SetScale(Gf.Vec3d(dx, dy, dz))
    _set_transform(cube.GetPrim(), m_scale * matrix)
    if material is not None:
        _bind_material(cube.GetPrim(), material)
    return cube.GetPrim()


def _author_cyl(stage, path, radius, height, axis, matrix, material=None):
    from pxr import UsdGeom, Gf
    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateRadiusAttr(float(radius))
    cyl.CreateHeightAttr(float(height))
    cyl.CreateAxisAttr(axis)
    r = float(radius); h = float(height) / 2.0
    if axis == "X":
        ext = [Gf.Vec3f(-h, -r, -r), Gf.Vec3f(h, r, r)]
    elif axis == "Y":
        ext = [Gf.Vec3f(-r, -h, -r), Gf.Vec3f(r, h, r)]
    else:
        ext = [Gf.Vec3f(-r, -r, -h), Gf.Vec3f(r, r, h)]
    cyl.CreateExtentAttr(ext)
    _set_transform(cyl.GetPrim(), matrix)
    if material is not None:
        _bind_material(cyl.GetPrim(), material)
    return cyl.GetPrim()


def _trans(x, y, z):
    from pxr import Gf
    return Gf.Matrix4d().SetTranslate(Gf.Vec3d(float(x), float(y), float(z)))


def _make_camera(stage, path, pos, look_at, hfov_deg, aspect=16.0 / 9.0):
    import numpy as np
    from pxr import UsdGeom, Gf
    haperture = 36.0
    vaperture = haperture / float(aspect)
    focal = haperture / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))
    cam = UsdGeom.Camera.Define(stage, path)
    cam.CreateFocalLengthAttr(float(focal))
    cam.CreateHorizontalApertureAttr(float(haperture))
    cam.CreateVerticalApertureAttr(float(vaperture))
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.02, 6000.0))
    fwd = np.asarray(look_at, dtype=np.float64) - np.asarray(pos, dtype=np.float64)
    _set_transform(cam.GetPrim(), _basis_matrix(fwd, translate=pos))
    return cam.GetPrim()


# =========================================================================== #
# v3 structural / look constants.
# =========================================================================== #
TERRAIN_SIZE_M = 760.0
TERRAIN_RES = 820            # 821^2 ~ 674k verts; cell ~0.93 m (fine close detail)
SUN_COLOR = (1.0, 0.965, 0.92)
SUN_ANGLE_DEG = 0.53

REGOLITH_TINT = (1.03, 1.00, 0.95)   # faint warm-gray; scaled by the gray albedo

DEFAULT_PARAMS = {
    # Sun — harsh, low, airless-body
    "sun_elevation_deg": 13.0,
    "sun_azimuth_deg": 135.0,        # light from +X/-Y -> long shadows toward -X/+Y
    "sun_intensity": 3600.0,         # auto-exposure OFF: lit regolith ~mid-grey (NOT
                                     # white) so dark-basalt albedo + surface texture read.
    # Regolith — DARK lunar grey + faint emissive floor (shadow fill; the enclosing
    # star dome blocks any dome/distant fill light, so a tiny constant emissive lifts
    # shadows off pure-black void to dark grey — reads as regolith, not a hole).
    "regolith_albedo": 0.12,
    "regolith_roughness": 0.97,
    "regolith_emissive": 0.013,
    "rock_emissive": 0.008,
    # Terrain layers (meters)
    "hill_amp": 9.0,                 # rolling hills (ragged horizon)
    "undulation_amp": 2.4,
    "roughness_amp": 0.72,
    "grain_amp": 0.15,
    # Crater field (multi-scale)
    "n_craters_big": 7,
    "n_craters_med": 42,
    "n_craters_small": 240,
    # Rock scatter (power-law via PointInstancer) + hero boulders
    "n_pebbles": 1300,
    "n_small": 420,
    "n_mid": 95,
    "embedding_depth_frac": 0.34,
    "rock_albedo_range": (0.045, 0.100),   # dark basalt, a touch below regolith for separation
    "rock_roughness_range": (0.85, 0.97),
    "rock_displacement_amp": 0.34,
    # Sky dome / stars / earth
    "dome_radius": 600.0,
    "star_count": 150,
    "earth_enabled": True,
    # Rover
    "rover_enabled": True,
    "rover_xy": (0.0, 0.0),
    "rover_heading_deg": 0.0,        # 0 = faces +Y
    # Surface-only dataset hazard cam (used ONLY when rover_enabled is False — the SDG
    # path). Eye height is GROUND-RELATIVE (the terrain has ~9 m hills, so an absolute-Z
    # camera can bury underground -> black frame). Looks down the +Y traverse, where the
    # power-law scatter + hero boulders live; yaw/pitch/fov are domain-randomizable per
    # frame. Small yaw + a rear/side sun azimuth keep the sky-dome's sun-hole out of frame.
    "cam_xy": (0.0, 0.0),
    "cam_height_m": 2.1,
    "cam_pitch_deg": -7.0,           # negative = look down
    "cam_fov_deg": 60.0,
    "cam_yaw_deg": 0.0,              # 0 = look +Y
    "cam_look_dist": 30.0,
}

_COUNT_RNG_XOR = 0xA11CE
DOME_HOLE_DEG = 34.0
STAR_INSET_M = 18.0


# =========================================================================== #
# Crater stamping (windowed — only touches grid cells near each crater).
# =========================================================================== #
def _stamp_crater(Z, xs, ys, cx, cy, R, depth, rim):
    """Add a bowl interior + raised Gaussian rim into heightfield Z, in place.

    Only the local window around the crater is touched (cheap for many craters)."""
    import numpy as np
    nx = len(xs); ny = len(ys)
    dx = xs[1] - xs[0]
    infl = R * 1.9
    ix0 = max(0, int(math.floor((cx - infl - xs[0]) / dx)))
    ix1 = min(nx, int(math.ceil((cx + infl - xs[0]) / dx)) + 1)
    iy0 = max(0, int(math.floor((cy - infl - ys[0]) / dx)))
    iy1 = min(ny, int(math.ceil((cy + infl - ys[0]) / dx)) + 1)
    if ix1 <= ix0 or iy1 <= iy0:
        return
    sx = xs[ix0:ix1]
    sy = ys[iy0:iy1]
    GX, GY = np.meshgrid(sx, sy)
    d = np.sqrt((GX - cx) ** 2 + (GY - cy) ** 2)
    dn = d / R
    bowl = np.where(dn < 1.0, -depth * (1.0 - dn ** 2), 0.0)
    rim_bump = rim * np.exp(-(((d - R) / (0.26 * R)) ** 2))
    Z[iy0:iy1, ix0:ix1] += bowl + rim_bump


# =========================================================================== #
# Real-model rover (referenced external USD) — PREMIUM path.
# =========================================================================== #
def _reference_rover_usd(stage, root_path, usd_path, base_xy, ground_z, heading_deg,
                         scale=1.0, wheel_local_z=0.0, sink=0.09):
    """Reference an external rover USD (NASA VIPER lunar rover, Blender->USD) and
    stand it on the surface.

    The asset's FRONT is -Y in its own frame (VIPER's NavCam mast + TRIDENT drill at
    -Y), so we add a 180 deg yaw to face scene-forward (+Y) before the scene heading.
    Wheels rest at local z=wheel_local_z (the asset's wheel-contact floor: 0.0 for
    the VIPER build; the legacy Perseverance asset used -1.0); we lift them onto
    ``ground_z`` and sink a few cm for contact. Returns the hazard-cam mount."""
    from pxr import UsdGeom, Gf
    xform = UsdGeom.Xform.Define(stage, root_path)
    xform.GetPrim().GetReferences().AddReference(usd_path)
    yaw_total = float(heading_deg) + 180.0
    m_scale = Gf.Matrix4d().SetScale(Gf.Vec3d(scale, scale, scale))
    m_yaw = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), yaw_total))
    lift = -float(wheel_local_z) * float(scale)
    m_tr = _trans(base_xy[0], base_xy[1], ground_z + lift - float(sink))
    _set_transform(xform.GetPrim(), m_scale * m_yaw * m_tr)

    # Hazard / nav-cam POV: position the camera AHEAD of and ABOVE the rover's
    # front edge (the 3.1 m rover spans ~1.5 m forward of centre + a ~2.8 m mast),
    # then look forward & slightly down. Placing it clear of the body is essential —
    # mounting near the rover centre lands the camera INSIDE the mesh -> black frame.
    yaw_scene = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), float(heading_deg)))
    down_deg = -13.0
    fwd_local = Gf.Vec3d(0.0, math.cos(math.radians(down_deg)), math.sin(math.radians(down_deg)))
    fwd_world = yaw_scene.TransformDir(fwd_local)
    fwd_xy = yaw_scene.TransformDir(Gf.Vec3d(0.0, 1.0, 0.0))
    front_off = 2.0 * float(scale)            # clear of the front mast/deck
    hx = base_xy[0] + float(fwd_xy[0]) * front_off
    hy = base_xy[1] + float(fwd_xy[1]) * front_off
    haz_h = ground_z + 2.2 * float(scale)     # ~mast-cam height
    return {
        "hazard_pos": (hx, hy, haz_h),
        "hazard_fwd": (float(fwd_world[0]), float(fwd_world[1]), float(fwd_world[2])),
    }


# =========================================================================== #
# Procedural rover (fallback when no rover USD is available).
# =========================================================================== #
def _author_rover(stage, root_path, base_xy, ground_z, heading_deg, mats):
    """Author a recognizable 6-wheel rover (Curiosity/Perseverance-style silhouette).

    Built from Cube/Cylinder primitives under ``root_path`` (a child Xform placed at
    (base_xy, ground_z) and yawed by ``heading_deg``; 0 deg faces +Y = "forward").
    Returns dict with the hazard-cam mount transform (world) so the caller can place
    a camera on the rover's mast.  All local coords are in the rover frame:
    +Y forward, +X right, +Z up; wheels rest at local z=0 (== ground)."""
    from pxr import UsdGeom, Gf

    UsdGeom.Xform.Define(stage, root_path)
    yaw = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), float(heading_deg)))
    place = yaw * _trans(base_xy[0], base_xy[1], ground_z)
    rover = UsdGeom.Xform.Define(stage, root_path)
    _set_transform(rover.GetPrim(), place)

    body_mat = mats["body"]
    deck_mat = mats["deck"]
    wheel_mat = mats["wheel"]
    metal_mat = mats["metal"]
    mast_mat = mats["mast"]
    white_mat = mats["white"]

    p = root_path
    # --- Chassis: warm gold thermal-blanket body, belly ~0.55 m off ground ----
    belly = 0.62
    body_h = 0.46
    body_cz = belly + body_h / 2.0
    _author_box(stage, p + "/Chassis", (1.45, 2.45, body_h), _trans(0, 0, body_cz), body_mat)
    # Sloped front bevel + rear equipment box for silhouette interest
    _author_box(stage, p + "/ChassisTop", (1.30, 2.10, 0.16),
                _trans(0, 0, body_cz + body_h / 2.0 + 0.06), deck_mat)
    # --- Wheels: 6, dark, cleated; rocker-bogie hint via strut boxes ----------
    wheel_r = 0.34
    wheel_w = 0.26
    wx = 0.92
    for side, sgn in (("L", -1.0), ("R", +1.0)):
        for k, wy in enumerate((-0.92, 0.0, 0.92)):
            _author_cyl(stage, "%s/Wheel_%s%d" % (p, side, k), wheel_r, wheel_w, "X",
                        _trans(sgn * wx, wy, wheel_r), wheel_mat)
            # strut from wheel hub up to the chassis belly
            _author_box(stage, "%s/Strut_%s%d" % (p, side, k), (0.06, 0.10, belly - wheel_r),
                        _trans(sgn * (wx - 0.12), wy, wheel_r + (belly - wheel_r) / 2.0), metal_mat)
        # rocker rail along each side
        _author_box(stage, "%s/Rocker_%s" % (p, side), (0.07, 2.10, 0.08),
                    _trans(sgn * (wx - 0.16), 0.0, belly - 0.02), metal_mat)
    # --- Camera mast (front-mounted), with a stereo "head" ---------------------
    mast_base_z = body_cz + body_h / 2.0
    mast_h = 1.15
    mast_top = mast_base_z + mast_h
    _author_cyl(stage, p + "/Mast", 0.05, mast_h, "Z",
                _trans(0.0, 0.95, mast_base_z + mast_h / 2.0), mast_mat)
    _author_box(stage, p + "/MastHead", (0.55, 0.16, 0.18),
                _trans(0.0, 0.95, mast_top + 0.02), deck_mat)
    # two camera "eyes" on the head, looking forward (+Y)
    for ex in (-0.18, 0.18):
        _author_cyl(stage, p + "/Eye_%s" % ("L" if ex < 0 else "R"), 0.045, 0.06, "Y",
                    _trans(ex, 0.95 + 0.10, mast_top + 0.02), metal_mat)
    # --- RTG (rear) + high-gain antenna dish -----------------------------------
    _author_cyl(stage, p + "/RTG", 0.20, 0.62, "Y",
                _trans(0.0, -1.45, body_cz + 0.18), metal_mat)
    # antenna dish: thin cylinder tilted up
    from pxr import Gf as _Gf
    dish_tilt = _Gf.Matrix4d().SetRotate(_Gf.Rotation(_Gf.Vec3d(1, 0, 0), -40.0)) * \
        _trans(0.45, -0.55, mast_base_z + 0.35)
    _author_cyl(stage, p + "/Antenna", 0.22, 0.05, "Z", dish_tilt, white_mat)
    _author_cyl(stage, p + "/AntMast", 0.03, 0.45, "Z",
                _trans(0.45, -0.55, mast_base_z + 0.18), metal_mat)

    # Hazard cam mount: just above the mast head, looking forward + slightly down.
    haz_local_pos = Gf.Vec3d(0.0, 1.02, mast_top + 0.05)
    world_pos = place.Transform(haz_local_pos)
    # forward+down in world (rotate the local forward by the yaw)
    fwd_local = Gf.Vec3d(0.0, math.cos(math.radians(-9.0)), math.sin(math.radians(-9.0)))
    fwd_world = yaw.TransformDir(fwd_local)
    return {
        "hazard_pos": (float(world_pos[0]), float(world_pos[1]), float(world_pos[2])),
        "hazard_fwd": (float(fwd_world[0]), float(fwd_world[1]), float(fwd_world[2])),
    }


# =========================================================================== #
# Scene authoring.
# =========================================================================== #
def build_lunar_stage_v3(seed: int, params: dict | None = None):
    """Author the v3 lunar scene + rover + cameras into the current USD context.

    Returns a list of (camera_name, camera_prim_path) for the render harness."""
    import os
    import numpy as np
    from pxr import UsdGeom, UsdLux, Gf

    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)

    rng = np.random.RandomState(int(seed) & 0x7FFFFFFF)
    count_rng = np.random.RandomState((int(seed) ^ _COUNT_RNG_XOR) & 0x7FFFFFFF)

    import omni.usd
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.Scope.Define(stage, "/World/Looks")

    # --- Materials --------------------------------------------------------- #
    tint = np.asarray(REGOLITH_TINT, dtype=np.float64)
    reg_a = float(p["regolith_albedo"])
    # Dark grey regolith: gray albedo carrying a faint warm tint + faint emissive floor.
    regolith_diffuse = tuple(float(c) for c in (reg_a * tint))
    reg_em = float(p["regolith_emissive"])
    reg_emissive = (reg_em * 1.03, reg_em, reg_em * 0.95)
    # Tiling regolith NORMAL map (tactile sub-mesh-res relief). Used when present;
    # otherwise fall back to the plain (smooth-between-pebbles) regolith material.
    normal_path = _resolve_asset(
        "V3_REGOLITH_NORMAL",
        "/home/chaotic-curiosity/regolith/assets/regolith_normal.png",
        "assets/regolith_normal.png")
    regolith_tile = float(os.environ.get("V3_REGOLITH_TILE", "5.0"))
    use_normal = bool(normal_path) and os.path.isfile(normal_path)
    if use_normal:
        regolith_mat = _make_regolith_material_normalmapped(
            stage, "/World/Looks/RegolithMat", diffuse=regolith_diffuse,
            roughness=float(p["regolith_roughness"]), emissive=reg_emissive,
            normal_path=normal_path)
    else:
        regolith_mat = _make_preview_material(
            stage, "/World/Looks/RegolithMat",
            diffuse=regolith_diffuse, roughness=float(p["regolith_roughness"]), metallic=0.0,
            emissive=reg_emissive,
        )
    rock_mat_rng = np.random.RandomState((int(seed) * 2246822519 + 0x20C) & 0x7FFFFFFF)
    rock_mats = _make_rock_material_pool(
        stage, "/World/Looks", rock_mat_rng, 12,
        p["rock_albedo_range"], p["rock_roughness_range"], emissive=float(p["rock_emissive"]),
    )
    # Dark neutral basalt for the biggest fractured hero boulders (darker than the
    # scatter pool so they read as heavy, in-frame basalt rather than light lumps).
    boulder_dark_mat = _make_preview_material(
        stage, "/World/Looks/BoulderDarkMat",
        diffuse=(0.038, 0.038, 0.040), roughness=0.93, metallic=0.0,
        emissive=(float(p["rock_emissive"]),) * 3,
    )
    sky_mat = _make_preview_material(
        stage, "/World/Looks/SkyMat", diffuse=(0, 0, 0), roughness=1.0, metallic=0.0,
        emissive=(0.004, 0.004, 0.007), unlit=True,
    )
    star_mat = _make_preview_material(
        stage, "/World/Looks/StarMat", diffuse=(0, 0, 0), roughness=1.0, metallic=0.0,
        emissive=(2.6, 2.6, 2.8), unlit=True,
    )
    earth_mat = _make_preview_material(
        stage, "/World/Looks/EarthMat", diffuse=(0.05, 0.08, 0.16), roughness=1.0,
        metallic=0.0, emissive=(0.18, 0.34, 0.62), unlit=True,
    )
    rover_mats = {
        "body": _make_preview_material(stage, "/World/Looks/RoverBody",
                                       diffuse=(0.62, 0.50, 0.20), roughness=0.45, metallic=0.5),
        "deck": _make_preview_material(stage, "/World/Looks/RoverDeck",
                                       diffuse=(0.30, 0.30, 0.33), roughness=0.55, metallic=0.4),
        "wheel": _make_preview_material(stage, "/World/Looks/RoverWheel",
                                        diffuse=(0.045, 0.045, 0.05), roughness=0.85, metallic=0.2),
        "metal": _make_preview_material(stage, "/World/Looks/RoverMetal",
                                        diffuse=(0.42, 0.43, 0.46), roughness=0.35, metallic=0.85),
        "mast": _make_preview_material(stage, "/World/Looks/RoverMast",
                                       diffuse=(0.50, 0.51, 0.54), roughness=0.30, metallic=0.9),
        "white": _make_preview_material(stage, "/World/Looks/RoverWhite",
                                        diffuse=(0.78, 0.78, 0.80), roughness=0.40, metallic=0.1),
    }

    # --- Regolith ground (multi-scale displaced heightfield) --------------- #
    size = TERRAIN_SIZE_M
    res = TERRAIN_RES
    xs = np.linspace(-size / 2.0, size / 2.0, res + 1)
    ys = np.linspace(-size / 2.0, size / 2.0, res + 1)
    X, Y = np.meshgrid(xs, ys, indexing="xy")

    Z = _fractal_noise_2d(X, Y, seed, octaves=5, base_freq=0.004) * float(p["hill_amp"])
    Z += _fractal_noise_2d(X, Y, seed + 7, octaves=5, base_freq=0.02) * float(p["undulation_amp"])
    Z += _fractal_noise_2d(X, Y, seed + 19, octaves=4, base_freq=0.08) * float(p["roughness_amp"])
    Z += _fractal_noise_2d(X, Y, seed + 31, octaves=3, base_freq=0.24) * float(p["grain_amp"])

    # Multi-scale crater field. Counts from the dedicated count RNG; per-crater
    # params (placement, depth, rim) from the main rng.
    def _craters(n, r_lo, r_hi, depth_k, rim_k):
        for _ in range(int(n)):
            cx = rng.uniform(-size / 2.3, size / 2.3)
            cy = rng.uniform(-size / 2.3, size / 2.3)
            R = rng.uniform(r_lo, r_hi)
            depth = R * depth_k * rng.uniform(0.75, 1.25)
            rim = depth * rim_k * rng.uniform(0.7, 1.3)
            _stamp_crater(Z, xs, ys, cx, cy, R, depth, rim)

    nb = int(count_rng.randint(max(1, p["n_craters_big"] - 2), p["n_craters_big"] + 3))
    nm = int(count_rng.randint(p["n_craters_med"] - 8, p["n_craters_med"] + 9))
    nsm = int(count_rng.randint(p["n_craters_small"] - 30, p["n_craters_small"] + 31))
    _craters(nb, 14.0, 40.0, 0.20, 0.22)
    _craters(nm, 3.0, 12.0, 0.18, 0.20)
    _craters(nsm, 0.6, 3.0, 0.15, 0.16)

    # Per-vertex normals from height gradient.
    dy = ys[1] - ys[0]
    dx = xs[1] - xs[0]
    dZdy, dZdx = np.gradient(Z, dy, dx)
    nrm = np.stack([-dZdx, -dZdy, np.ones_like(Z)], axis=-1)
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)

    points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    normals = nrm.reshape(-1, 3)
    nx = res + 1
    ixg, iyg = np.meshgrid(np.arange(res), np.arange(res))
    i00 = (iyg * nx + ixg).ravel()
    i10 = i00 + 1
    i01 = i00 + nx
    i11 = i00 + nx + 1
    face_indices = np.stack([i00, i10, i11, i01], axis=1).ravel()
    face_counts = np.full(res * res, 4, dtype=np.int32)
    regolith_prim = _author_mesh(stage, "/World/Regolith", points, face_counts, face_indices, normals=normals)
    # World-XY st (tiled every ``regolith_tile`` m) so the normal map repeats across
    # the terrain at a fixed physical scale.
    if use_normal:
        from pxr import Sdf, Vt
        st = np.stack([X.ravel() / regolith_tile, Y.ravel() / regolith_tile], axis=1).astype(np.float32)
        stpv = UsdGeom.PrimvarsAPI(regolith_prim).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        try:
            stpv.Set(Vt.Vec2fArray.FromNumpy(st))
        except Exception:
            stpv.Set(Vt.Vec2fArray([(float(a), float(b)) for a, b in st]))
    _bind_material(regolith_prim, regolith_mat)
    _add_semantics(regolith_prim, "regolith")

    # --- Rock prototype pool + PER-MESH power-law scatter ------------------ #
    # The scatter is authored as INDIVIDUAL Mesh prims (each tagged "rock"), NOT a
    # USD PointInstancer. A PointInstancer renders identically but its semantic
    # label — whether placed on the instancer prim OR on its child prototype meshes
    # — does NOT propagate to the point-instanced pixels in Isaac Sim 6.0 semantic
    # segmentation (verified both ways: every scattered pebble came back UNLABELLED,
    # a real hole for the upcoming dataset). Per-mesh scatter uses the SAME proven
    # path as the hero boulders (individually-tagged meshes -> reliably class
    # "rock"). RNG draw order is preserved exactly, so positions/scales/orientations
    # are bit-identical to the instancer version — the look is unchanged.
    UsdGeom.Xform.Define(stage, "/World/Rocks")
    UsdGeom.Scope.Define(stage, "/World/Rocks/Scatter")
    disp_amp = float(p["rock_displacement_amp"])
    n_low = 10   # subdiv-2 protos (pebbles/small)
    n_mid = 6    # subdiv-3 protos (mid rocks)
    proto_rng = np.random.RandomState((int(seed) * 7919 + 17) & 0x7FFFFFFF)
    protos = []   # reusable geometry pool: (points, face_counts, face_indices, normals, material)
    for k in range(n_low + n_mid):
        subdiv = 2 if k < n_low else 3
        bp, bf = _icosphere(subdiv)
        dp = _displace_rock(bp, proto_rng, amp=disp_amp)
        nn = _vertex_normals(dp, bf)
        protos.append((dp, np.full(len(bf), 3, dtype=np.int32), bf.ravel(), nn,
                       rock_mats[k % len(rock_mats)]))

    # --- Power-law scatter (pebbles -> small -> mid), each an own tagged Mesh -- #
    embed = float(p["embedding_depth_frac"])
    scatter_n = [0]

    def _scatter(n, s_lo, s_hi, xr, yr, proto_lo, proto_hi):
        for _ in range(int(n)):
            px = rng.uniform(-xr, xr)
            py = rng.uniform(yr[0], yr[1])
            base = math.exp(rng.uniform(math.log(s_lo), math.log(s_hi)))  # log-uniform
            sx = base * rng.uniform(0.8, 1.2)
            sy = base * rng.uniform(0.8, 1.2)
            sz = base * rng.uniform(0.55, 0.9)
            g = _sample_height(Z, xs, ys, px, py)
            cz = g + sz * (1.0 - embed)
            rot = (Gf.Rotation(Gf.Vec3d(1, 0, 0), rng.uniform(0, 360))
                   * Gf.Rotation(Gf.Vec3d(0, 1, 0), rng.uniform(0, 360))
                   * Gf.Rotation(Gf.Vec3d(0, 0, 1), rng.uniform(0, 360)))
            pk = int(rng.randint(proto_lo, proto_hi))
            pts, fc, fi, nn, mat = protos[pk]
            idx = scatter_n[0]; scatter_n[0] += 1
            prim = _author_mesh(stage, "/World/Rocks/Scatter/rock_%05d" % idx,
                                pts, fc, fi, normals=nn)
            m_scale = Gf.Matrix4d().SetScale(Gf.Vec3d(float(sx), float(sy), float(sz)))
            m_rot = Gf.Matrix4d().SetRotate(rot)
            _set_transform(prim, m_scale * m_rot * _trans(float(px), float(py), float(cz)))
            _bind_material(prim, mat)
            _add_semantics(prim, "rock")

    _scatter(p["n_pebbles"], 0.04, 0.18, 36.0, (-12.0, 82.0), 0, n_low)
    _scatter(p["n_small"], 0.18, 0.5, 55.0, (-30.0, 135.0), 0, n_low)
    _scatter(p["n_mid"], 0.5, 1.4, 72.0, (-40.0, 175.0), n_low, n_low + n_mid)
    print(">>> scatter: authored %d per-mesh rocks (each semantic=rock)" % scatter_n[0], flush=True)

    # --- Hand-placed hero boulders (the FEW large in-frame rocks) ---------- #
    UsdGeom.Xform.Define(stage, "/World/HeroRocks")
    # Layout: a few boulders FLANK the rover (frame it in the beauty shots) but stay
    # out of the rover's forward +Y corridor; the rest recede at varying distance/size
    # down the traverse (hazard-cam depth) while leaving a clear horizon gap.
    hero_specs = [
        (5.6, 0.5, 2.2), (-5.2, -1.2, 1.9), (6.8, -3.2, 1.6),     # flank the rover
        (-6.5, 13.0, 1.7), (7.5, 22.0, 2.3), (-9.5, 34.0, 2.9),    # forward, offset
        (11.0, 45.0, 2.6), (-4.5, 60.0, 3.4), (13.5, 30.0, 2.0),
        (-13.0, 7.0, 2.1), (14.0, 9.0, 2.4),                        # wide (beauty2)
    ]
    for hi, (hx, hy, hs) in enumerate(hero_specs):
        h_rng = np.random.RandomState((int(seed) * 524287 + hi * 1301 + 3) & 0x7FFFFFFF)
        # The prominent in-frame boulders (3 flanking the rover) + the genuinely
        # big ones -> angular fractured + dark basalt. Mid/far stay rounded/varied.
        big = (hi < 3) or (hs >= 2.3)
        subdiv = 5 if (big or hs >= 3.0) else 4   # finer facets on the angular ones
        bp, bf = _icosphere(subdiv)
        if big:
            dp = _displace_rock_angular(bp, h_rng, amp=disp_amp * 1.15)
            nn = None                                  # flat/faceted -> angular read
            hero_mat = boulder_dark_mat                # darker basalt
        else:
            dp = _displace_rock(bp, h_rng, amp=disp_amp)
            nn = _vertex_normals(dp, bf)
            hero_mat = rock_mats[h_rng.randint(len(rock_mats))]
        prim = _author_mesh(stage, "/World/HeroRocks/Hero_%02d" % hi, dp,
                            np.full(len(bf), 3, dtype=np.int32), bf.ravel(), normals=nn)
        sx = hs * h_rng.uniform(0.85, 1.25)
        sy = hs * h_rng.uniform(0.85, 1.25)
        sz = hs * h_rng.uniform(0.6, 0.92)
        g = _sample_height(Z, xs, ys, hx, hy)
        cz = g + sz * (1.0 - embed)
        m_scale = Gf.Matrix4d().SetScale(Gf.Vec3d(sx, sy, sz))
        rot = (Gf.Rotation(Gf.Vec3d(1, 0, 0), h_rng.uniform(0, 360))
               * Gf.Rotation(Gf.Vec3d(0, 1, 0), h_rng.uniform(0, 360))
               * Gf.Rotation(Gf.Vec3d(0, 0, 1), h_rng.uniform(0, 360)))
        m_rot = Gf.Matrix4d().SetRotate(rot)
        _set_transform(prim, m_scale * m_rot * _trans(hx, hy, cz))
        _bind_material(prim, hero_mat)
        _add_semantics(prim, "rock")

    # --- Sun (harsh distant light, low elevation) -------------------------- #
    el = math.radians(float(p["sun_elevation_deg"]))
    az = math.radians(float(p["sun_azimuth_deg"]))
    s = np.array([math.cos(el) * math.sin(az), math.cos(el) * math.cos(az), math.sin(el)])
    sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
    sun.CreateIntensityAttr(float(p["sun_intensity"]))
    sun.CreateColorAttr(Gf.Vec3f(*SUN_COLOR))
    sun.CreateAngleAttr(SUN_ANGLE_DEG)
    _set_transform(sun.GetPrim(), _basis_matrix(-s, translate=(0.0, 0.0, 200.0)))

    # --- Star dome + stars ------------------------------------------------- #
    dome_radius = float(p["dome_radius"])
    dome_prim = _author_sky_dome(stage, "/World/StarDome", dome_radius, sun_dir=s, hole_deg=DOME_HOLE_DEG)
    _bind_material(dome_prim, sky_mat)
    _add_semantics(dome_prim, "sky")
    try:
        UsdGeom.Scope.Define(stage, "/World/Stars")
        star_r = dome_radius - STAR_INSET_M
        for i in range(int(p["star_count"])):
            d = rng.normal(size=3)
            d = d / np.linalg.norm(d)
            d[2] = abs(d[2]) * 0.85 + 0.12
            d = d / np.linalg.norm(d)
            pos = d * star_r
            star = UsdGeom.Sphere.Define(stage, "/World/Stars/Star_%03d" % i)
            srad = float(rng.uniform(0.5, 1.2))
            star.CreateRadiusAttr(srad)
            star.CreateExtentAttr([Gf.Vec3f(-srad, -srad, -srad), Gf.Vec3f(srad, srad, srad)])
            _set_transform(star.GetPrim(), _trans(pos[0], pos[1], pos[2]))
            _bind_material(star.GetPrim(), star_mat)
            _add_semantics(star.GetPrim(), "sky")
    except Exception as exc:
        print(">>> star scatter skipped: %r" % exc, flush=True)

    # --- Earth (small, tasteful, low on horizon) --------------------------- #
    if bool(p["earth_enabled"]):
        e_az = math.radians(104.0)
        e_el = math.radians(17.0)
        ed = np.array([math.cos(e_el) * math.sin(e_az), math.cos(e_el) * math.cos(e_az), math.sin(e_el)])
        e_dist = dome_radius * 0.82
        e_pos = ed * e_dist
        e_rad = e_dist * math.tan(math.radians(1.7))   # ~3.4 deg angular size
        earth = UsdGeom.Sphere.Define(stage, "/World/Earth")
        earth.CreateRadiusAttr(float(e_rad))
        earth.CreateExtentAttr([Gf.Vec3f(-e_rad, -e_rad, -e_rad), Gf.Vec3f(e_rad, e_rad, e_rad)])
        _set_transform(earth.GetPrim(), _trans(e_pos[0], e_pos[1], e_pos[2]))
        _bind_material(earth.GetPrim(), earth_mat)
        _add_semantics(earth.GetPrim(), "sky")

    # --- Rover ------------------------------------------------------------- #
    # PREMIUM path: reference the NASA VIPER lunar-rover USD (Blender->USD, full PBR:
    # gold MLI body, vertical side solar panels, NavCam mast, TRIDENT drill). Falls
    # back to the procedural rover if the asset is absent (runnable anywhere).
    haz = None
    if bool(p["rover_enabled"]):
        rxy = p["rover_xy"]
        ground_z = _sample_height(Z, xs, ys, rxy[0], rxy[1])
        rover_usd = _resolve_asset(
            "V3_ROVER_USD",
            "/home/chaotic-curiosity/regolith/assets/rover/viper.usdc",
            "assets/rover/viper.usdc")
        rover_scale = float(os.environ.get("V3_ROVER_SCALE", "1.0"))
        # Wheel-contact floor of the referenced asset in its own frame (VIPER build
        # = 0.0; legacy Perseverance asset = -1.0). Override via V3_ROVER_FLOOR_Z.
        rover_floor_z = float(os.environ.get("V3_ROVER_FLOOR_Z", "0.0"))
        if rover_usd and os.path.isfile(rover_usd):
            haz = _reference_rover_usd(stage, "/World/Rover", rover_usd, rxy, ground_z,
                                       float(p["rover_heading_deg"]), scale=rover_scale,
                                       wheel_local_z=rover_floor_z)
            print(">>> rover: referenced real-model USD %s (scale=%.2f)"
                  % (rover_usd, rover_scale), flush=True)
        else:
            haz = _author_rover(stage, "/World/Rover", rxy, ground_z,
                                float(p["rover_heading_deg"]), rover_mats)
            print(">>> rover: USD '%s' not found -> procedural fallback" % rover_usd, flush=True)

    # --- Cameras ----------------------------------------------------------- #
    # Beauty cams must sit a fixed EYE HEIGHT above the LOCAL ground (the terrain has
    # ~9 m rolling hills; an absolute-Z camera ends up buried underground -> black frame).
    rxy = p["rover_xy"]
    rover_gz = _sample_height(Z, xs, ys, rxy[0], rxy[1])

    def _cam_on_ground(name, cxy, eye_h, look_xy, look_h, hfov):
        g = _sample_height(Z, xs, ys, cxy[0], cxy[1])
        _make_camera(stage, name, pos=(cxy[0], cxy[1], g + eye_h),
                     look_at=(look_xy[0], look_xy[1], rover_gz + look_h), hfov_deg=hfov)

    # Beauty hero: sunlit front-right 3/4 (sun is from +X/-Y), rover the clear
    # subject, boulders framing the sides without occluding, textured ground sweep.
    _cam_on_ground("/World/BeautyCam", (9.0, -8.5), 2.7, (-0.6, 2.4), 1.15, 41.0)
    # Closer dramatic 3/4: low + tight, rover large in frame with boulders flanking
    # (the premium "closer rover shot").
    _cam_on_ground("/World/BeautyCam2", (5.4, -5.4), 1.25, (-0.4, 1.5), 1.25, 49.0)
    # Hazard cam: on the rover mast, forward POV.
    if haz is not None:
        hp = haz["hazard_pos"]
        hf = haz["hazard_fwd"]
        _make_camera(stage, "/World/HazardCam",
                     pos=hp, look_at=(hp[0] + hf[0] * 30.0, hp[1] + hf[1] * 30.0, hp[2] + hf[2] * 30.0),
                     hfov_deg=58.0)
    else:
        # Surface-only dataset cam (no rover): ground-relative eye height, looking
        # down the +Y traverse with DR-able yaw / pitch / fov (the SDG hazard plate).
        cxy = p["cam_xy"]
        gcam = _sample_height(Z, xs, ys, cxy[0], cxy[1])
        eye = gcam + float(p["cam_height_m"])
        yaw = math.radians(float(p["cam_yaw_deg"]))
        pit = math.radians(float(p["cam_pitch_deg"]))
        cphi = math.cos(pit)
        # forward in XY = +Y rotated about Z by yaw; pitch tilts it down (-Z).
        fwd = (cphi * math.sin(yaw), cphi * math.cos(yaw), math.sin(pit))
        dist = float(p["cam_look_dist"])
        _make_camera(stage, "/World/HazardCam",
                     pos=(cxy[0], cxy[1], eye),
                     look_at=(cxy[0] + fwd[0] * dist,
                              cxy[1] + fwd[1] * dist,
                              eye + fwd[2] * dist),
                     hfov_deg=float(p["cam_fov_deg"]))

    print(">>> build_lunar_stage_v3: seed=%d  ground=%dx%d verts  craters(b/m/s)=%d/%d/%d"
          "  scatter(peb/sm/mid)=%d/%d/%d  hero=%d  sun(el=%.1f,az=%.1f,I=%.0f)  reg_albedo=%.3f"
          % (seed, nx, nx, nb, nm, nsm, p["n_pebbles"], p["n_small"], p["n_mid"], len(hero_specs),
             float(p["sun_elevation_deg"]), float(p["sun_azimuth_deg"]), float(p["sun_intensity"]),
             reg_a), flush=True)

    cams = [("beauty", "/World/BeautyCam"),
            ("hazard", "/World/HazardCam"),
            ("beauty2", "/World/BeautyCam2")]
    return cams


# =========================================================================== #
# Render harness (__main__): one 1920x1080 RGB beauty plate per camera.
# =========================================================================== #
def main():
    import os, sys, glob, time
    import numpy as np

    seed = int(os.environ.get("V3_SEED", "7"))
    out_dir = os.environ.get("SMOKE_OUT", "/workspace/out")
    width = int(os.environ.get("V3_W", "1920"))
    height = int(os.environ.get("V3_H", "1080"))
    subframes = int(os.environ.get("V3_SUBFRAMES", "48"))
    renderer = os.environ.get("V3_RENDERER", "RayTracedLighting")
    pathtrace = os.environ.get("V3_PATHTRACE", "0") == "1"
    sun_env = os.environ.get("V3_SUN", "")          # override sun intensity
    sweep_env = os.environ.get("V3_SUN_SWEEP", "")   # comma list -> exposure sweep (hazard cam)

    build_params = {}
    if sun_env:
        build_params["sun_intensity"] = float(sun_env)

    argv0 = sys.argv[0] if sys.argv else "build_lunar_stage_v3.py"
    sys.argv = [argv0]
    t0 = time.time()

    def phase(msg):
        print(">>> PHASE %s  (+%.1fs)" % (msg, time.time() - t0), flush=True)
        sys.stderr.flush()

    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": True, "renderer": renderer})
    phase("simapp_ready")

    import carb
    import omni.replicator.core as rep
    settings = carb.settings.get_settings()
    # Deterministic exposure: disable auto-exposure / eye-adaptation so brightness is
    # driven purely by sun intensity + material albedo (v2-style auto-exposure was
    # clipping every sunlit surface to white, killing all texture in the lit areas).
    for key in ("/rtx/post/histogram/enabled",
                "/rtx/post/tonemap/enableAutoExposure",
                "/rtx/post/eyeAdaptation/enabled"):
        try:
            settings.set(key, False)
        except Exception:
            pass
    if pathtrace:
        pt_spp = int(os.environ.get("V3_PT_TOTALSPP", "320"))
        pt_bounces = int(os.environ.get("V3_PT_BOUNCES", "5"))
        settings.set("/rtx/rendermode", "PathTracing")
        settings.set("/rtx/pathtracing/spp", 1)
        settings.set("/rtx/pathtracing/totalSpp", pt_spp)
        settings.set("/rtx/pathtracing/maxBounces", pt_bounces)
        settings.set("/rtx/pathtracing/clampSpp", 0)
        # Keep the OptiX denoiser OFF so it doesn't smear the fine regolith
        # normal-map relief; 320 spp is clean enough for these stills.
        try:
            settings.set("/rtx/pathtracing/optixDenoiser/enabled", False)
        except Exception:
            pass
        print(">>> pathtracing: totalSpp=%d maxBounces=%d" % (pt_spp, pt_bounces), flush=True)

    for _ in range(10):
        simulation_app.update()
    phase("ext_started")

    rep.orchestrator.set_capture_on_play(False)

    try:
        cams = build_lunar_stage_v3(seed, build_params)
        for _ in range(12):
            simulation_app.update()
        phase("scene_built")

        def drain(check, budget):
            dl = time.time() + budget
            while time.time() < dl:
                simulation_app.update()
                if check():
                    return True
            return False

        def _report(name, sub_out):
            rgbs = sorted(glob.glob(os.path.join(sub_out, "**", "rgb*.png"), recursive=True))
            if rgbs:
                try:
                    from PIL import Image
                    a = np.array(Image.open(rgbs[0]).convert("RGB"))
                    print(">>> %-10s RGB %s mean=%.1f p1=%.0f p50=%.0f p99=%.0f max=%d  -> %s"
                          % (name, a.shape, float(a.mean()), float(np.percentile(a, 1)),
                             float(np.percentile(a, 50)), float(np.percentile(a, 99)),
                             int(a.max()), rgbs[0]), flush=True)
                except Exception as e:
                    print(">>> brightness read failed: %r" % e, flush=True)

        # --- Exposure sweep mode: render hazard cam at several sun intensities --- #
        if sweep_env:
            import omni.usd
            from pxr import UsdLux
            stg = omni.usd.get_context().get_stage()
            sun = UsdLux.DistantLight(stg.GetPrimAtPath("/World/Sun"))
            haz_path = dict(cams).get("hazard", "/World/HazardCam")
            rp = rep.create.render_product(haz_path, (width, height))
            first_s = True
            for sval in [float(x) for x in sweep_env.split(",") if x.strip()]:
                sun.GetIntensityAttr().Set(float(sval))
                sub_out = os.path.join(out_dir, "sweep_%d" % int(sval))
                w = rep.WriterRegistry.get("BasicWriter")
                w.initialize(output_dir=sub_out, rgb=True)
                w.attach([rp])
                for _ in range(20 if first_s else 8):
                    simulation_app.update()
                phase("warm_sweep_%d" % int(sval))
                rep.orchestrator.step(delta_time=0.0, rt_subframes=min(subframes, 24))
                drain(lambda: bool(glob.glob(os.path.join(sub_out, "**", "rgb*.png"), recursive=True)),
                      budget=(430.0 if first_s else 120.0))
                try:
                    rep.orchestrator.wait_until_complete()
                except Exception as e:
                    print(">>> wait raised: %r" % e, flush=True)
                w.detach()
                _report("sweep_%d" % int(sval), sub_out)
                first_s = False
            phase("closing")
            simulation_app.close()
            print("V3_STILLS_DONE output_dir=" + out_dir, flush=True)
            return

        first = True
        for name, cam_path in cams:
            sub_out = os.path.join(out_dir, name)
            rp = rep.create.render_product(cam_path, (width, height))
            w = rep.WriterRegistry.get("BasicWriter")
            w.initialize(output_dir=sub_out, rgb=True)
            w.attach([rp])
            for _ in range(20 if first else 6):
                simulation_app.update()
            phase("warm_%s" % name)
            rep.orchestrator.step(delta_time=0.0, rt_subframes=subframes)
            ok = drain(lambda: bool(glob.glob(os.path.join(sub_out, "**", "rgb*.png"), recursive=True)),
                       budget=(430.0 if first else 220.0))
            phase(("got_%s" if ok else "TIMEOUT_%s") % name)
            try:
                rep.orchestrator.wait_until_complete()
            except Exception as e:
                print(">>> wait_until_complete raised: %r" % e, flush=True)
            w.detach()
            _report(name, sub_out)
            first = False
    except Exception:
        import traceback
        print(">>> BUILD/RENDER ERROR:\n" + traceback.format_exc(), flush=True)
        sys.stderr.flush()
    finally:
        phase("closing")
        simulation_app.close()

    print("V3_STILLS_DONE output_dir=" + out_dir, flush=True)


if __name__ == "__main__":
    main()
