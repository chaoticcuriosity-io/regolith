"""
build_lunar_stage.py — author a USD stage representing a lunar landing/traverse scene.

Called by the Replicator pipeline (generate_dataset.py) and directly (``__main__``)
for scene inspection / a single validation render.

Stage layout (Z-up, meters)
----------------------------
/World/
  Regolith       — displaced ground mesh (procedural noise + craters), class "regolith"
  Rocks/Rock_*   — scatter of noise-deformed icosphere meshes, class "rock"
  Sun            — UsdLux.DistantLight, harsh, low elevation (long shadows, no atmosphere)
  StarDome       — large emissive near-black sphere enclosing the scene, class "sky"
  StarDome/Star_* — tiny emissive star spheres on the dome, class "sky"
  RoverCam       — UsdGeom.Camera at rover/lander eye height, looking across the terrain

Class map (matches training/metrics.py, training/dataset.py, generate_dataset.py)
---------------------------------------------------------------------------------
  regolith: 0
  rock:     1
  sky:      2

Validation render (``python build_lunar_stage.py --seed 42 --out <dir>``) uses the
headless Replicator pattern validated on the DGX Spark (see replicator/_smoke_render.py
and setup-notes.md ## Session 2). It emits, for one 1024x1024 frame:
  - <out>/labelid/rgb_*.png                          — RGB render
  - <out>/labelid/semantic_segmentation_*.png        — RAW integer label-id mask (training)
  - <out>/labelid/semantic_segmentation_classid.png  — RAW mask remapped to the canonical
                                                        class map {regolith:0, rock:1, sky:2}
  - <out>/preview/semantic_segmentation_*.png        — colorized preview (figures)

Heavy artifacts (USD assets, datasets) live on the DGX Spark — not in git.
See docs/reports/01-the-lunar-stage.md for the full walkthrough.
"""

from __future__ import annotations

import math

# Canonical training class map (matches training/metrics.py, training/dataset.py,
# replicator/generate_dataset.py). rock (1) is the hazard class of interest.
CLASS_MAP = {"regolith": 0, "rock": 1, "sky": 2}
IGNORE_INDEX = 255  # unlabeled / background pixels in canonical masks

# Dedicated XOR for the scene-count RNG (n_rocks / crater count) so sweeping one
# domain-randomization knob doesn't shift the others' draw order (see build_lunar_stage).
_COUNT_RNG_XOR = 0xA11CE

# --------------------------------------------------------------------------- #
# Structural constants (the scene's fixed skeleton — NOT domain-randomized).
# --------------------------------------------------------------------------- #
TERRAIN_SIZE_M = 920.0        # meters across (centered on origin); edge past the horizon
TERRAIN_RES = 300             # cells per side -> (res+1)^2 vertices (~3 m cells)
TERRAIN_BASE_FREQ_LOW = 0.010   # primary fBm undulation frequency
TERRAIN_BASE_FREQ_HIGH = 0.045  # finer ripple frequency
TERRAIN_RIPPLE_AMP = 0.7        # amplitude of the finer ripple layer (relative to base)

SUN_COLOR = (1.0, 0.97, 0.92)   # warm white; no atmosphere to tint it
SUN_ANGLE_DEG = 0.53            # sun's angular size -> crisp, hard shadows
SUN_HEIGHT_M = 200.0            # light prim Z (cosmetic; DistantLight is at infinity)

REGOLITH_TINT = (0.22, 0.205, 0.19)   # warm-gray diffuse at the nominal albedo
ROCK_DIFFUSE = (0.155, 0.145, 0.135)  # darker than regolith
ROCK_ROUGHNESS = 0.90
ICOSPHERE_SUBDIV = 2          # rock base mesh: 162 verts / 320 faces

DOME_HOLE_DEG = 34.0          # spherical-cap cut around the sun (lets the sun in)
STAR_INSET_M = 18.0           # stars sit this far inside the dome
CAMERA_EYE_Y = -90.0          # camera Y position (looks across +Y toward the horizon)
CAMERA_APERTURE = 24.0        # horizontal == vertical aperture (square sensor)
CAMERA_VIEW_DIST = 160.0      # nominal forward throw used to derive the default pitch

# Default camera FOV (deg) equivalent to the validated focal=20 / aperture=24 lens,
# and default pitch reproducing the validated eye=(0,-90,2.0) -> target=(0,70,-0.5) look.
_DEFAULT_CAMERA_FOV_DEG = math.degrees(2.0 * math.atan(CAMERA_APERTURE / (2.0 * 20.0)))
_DEFAULT_CAMERA_PITCH_DEG = math.degrees(math.atan2(-2.5, CAMERA_VIEW_DIST))


# --------------------------------------------------------------------------- #
# Domain-randomizable parameters. build_lunar_stage(seed, params) reads every
# value below from `params`, falling back to these defaults (which reproduce the
# validated Session-3 scene). replicator/randomizers.py samples a dict with these
# keys per frame; the structural skeleton above stays fixed.
# --------------------------------------------------------------------------- #
DEFAULT_PARAMS = {
    # Sun
    "sun_elevation_deg": 17.0,
    "sun_azimuth_deg": 120.0,
    "sun_intensity": 14000.0,
    # Regolith material
    "regolith_albedo": 0.205,          # scalar gray; mean of REGOLITH_TINT -> exact default
    "regolith_roughness": 0.96,
    # Terrain
    "terrain_amplitude": 3.2,          # primary fBm displacement amplitude (meters)
    "crater_count_range": (6, 11),     # [lo, hi); actual count drawn from the count RNG
    # Rocks
    "rock_count_range": (75, 115),     # [lo, hi); actual count drawn from the count RNG
    "near_rock_count": 12,             # guaranteed near-field boulders (sane coverage)
    "rock_scale_range": (0.3, 1.3),    # far-field base scale (meters)
    "near_rock_scale_range": (1.2, 3.0),  # near-field base scale (meters)
    "near_rock_y_range": (-78.0, -48.0),  # near-rock placement band ahead of the camera
    "embedding_depth_frac": 0.30,      # fraction of a rock's Z half-extent sunk into ground
    # Camera
    "camera_height_m": 2.0,            # rover/lander eye height
    "camera_pitch_deg": _DEFAULT_CAMERA_PITCH_DEG,  # downward tilt (negative = look down)
    "camera_fov_deg": _DEFAULT_CAMERA_FOV_DEG,      # horizontal field of view
    # Sky dome / stars
    "dome_radius": 600.0,
    "star_count": 220,
}


def canonical_mask_from_json(raw_id_array, labels_json_path):
    """Map a raw Replicator semantic-id mask -> the canonical class map
    {regolith:0, rock:1, sky:2}, with unlabeled/background -> 255 (ignore index).

    The raw-id -> canonical-id mapping is derived from the labels JSON class
    NAMES (normalized ``.strip().lower()``) — never from hardcoded raw integers.
    Replicator assigns the raw ids dynamically (it reserves 0=BACKGROUND,
    1=UNLABELLED and numbers the scene's classes in registration order), so the
    only stable key is the class name string.

    Parameters
    ----------
    raw_id_array : np.ndarray
        Single-channel (or RGB, first channel taken) raw label-id image as written
        by ``BasicWriter(..., colorize_semantic_segmentation=False)``.
    labels_json_path : str
        Path to the ``semantic_segmentation_labels_*.json`` Replicator emitted
        alongside the mask (id -> {"class": name}).

    Returns
    -------
    np.ndarray
        ``uint8`` array, same H×W as the input, valued in {0, 1, 2, 255}.

    Raises
    ------
    RuntimeError
        If the labels JSON is missing or empty (without it the raw ids are
        unmappable, so a silent all-255 mask would be a data-corruption trap).
    """
    import os
    import json
    import numpy as np

    if not labels_json_path or not os.path.isfile(labels_json_path):
        raise RuntimeError("labels JSON missing: %r" % (labels_json_path,))
    with open(labels_json_path) as fh:
        raw = json.load(fh)
    if not raw:
        raise RuntimeError("labels JSON empty: %r" % (labels_json_path,))

    arr = np.asarray(raw_id_array)
    if arr.ndim == 3:
        arr = arr[..., 0]
    arr = arr.astype(np.int64)

    canon = np.full(arr.shape, IGNORE_INDEX, dtype=np.uint8)
    for k, v in raw.items():
        try:
            idv = int(k)
        except (ValueError, TypeError):
            continue
        name = v.get("class") if isinstance(v, dict) else v
        key = str(name).strip().lower()
        if key in CLASS_MAP:
            canon[arr == idv] = CLASS_MAP[key]
    return canon


# --------------------------------------------------------------------------- #
# Procedural geometry helpers (numpy). Imports are lazy so that importing this
# module outside the Isaac container (no numpy/pxr required) does not fail.
# --------------------------------------------------------------------------- #
def _hash2(ix, iy, seed):
    """Deterministic per-lattice-point hash -> float in [0, 1)."""
    import numpy as np

    ix = ix.astype(np.int64)
    iy = iy.astype(np.int64)
    n = (ix * np.int64(73856093)) ^ (iy * np.int64(19349663)) ^ np.int64(int(seed) * 83492791)
    n = (n ^ (n >> np.int64(13))) * np.int64(1274126177)
    n = n & np.int64(0x7FFFFFFF)
    return n.astype(np.float64) / float(0x7FFFFFFF)


def _value_noise_2d(X, Y, seed, freq):
    """Smooth value noise on a lattice -> array in [0, 1], same shape as X."""
    import numpy as np

    xf = X * freq
    yf = Y * freq
    x0 = np.floor(xf).astype(np.int64)
    y0 = np.floor(yf).astype(np.int64)
    tx = xf - x0
    ty = yf - y0
    sx = tx * tx * (3.0 - 2.0 * tx)  # smoothstep
    sy = ty * ty * (3.0 - 2.0 * ty)
    v00 = _hash2(x0, y0, seed)
    v10 = _hash2(x0 + 1, y0, seed)
    v01 = _hash2(x0, y0 + 1, seed)
    v11 = _hash2(x0 + 1, y0 + 1, seed)
    a = v00 * (1.0 - sx) + v10 * sx
    b = v01 * (1.0 - sx) + v11 * sx
    return a * (1.0 - sy) + b * sy


def _fractal_noise_2d(X, Y, seed, octaves=5, base_freq=0.012, lacunarity=2.0, gain=0.5):
    """Fractal (fBm) value noise -> array roughly in [-1, 1]."""
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
    """Unit-radius geodesic icosphere -> (points (N,3) float64, faces (M,3) int32)."""
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
        va = np.array(verts[a])
        vb = np.array(verts[b])
        vm = (va + vb) / 2.0
        vm = vm / np.linalg.norm(vm)
        verts.append(tuple(vm.tolist()))
        idx = len(verts) - 1
        midcache[key] = idx
        return idx

    for _ in range(int(subdiv)):
        newf = []
        for a, b, c in faces:
            ab = midpoint(a, b)
            bc = midpoint(b, c)
            ca = midpoint(c, a)
            newf += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        faces = newf
    return np.array(verts, dtype=np.float64), np.array(faces, dtype=np.int32)


def _rock_radius_multiplier(dirs, rng, amp=0.55, nlobes=6):
    """Per-vertex radial deformation for a unit sphere -> radius multipliers (N,)."""
    import numpy as np

    disp = np.zeros(len(dirs), dtype=np.float64)
    for _ in range(nlobes):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        freq = rng.uniform(1.5, 4.0)
        phase = rng.uniform(0.0, 6.2831853)
        a = rng.uniform(0.3, 1.0)
        disp += a * np.sin(freq * (dirs @ axis) + phase)
    disp /= nlobes
    return 1.0 + amp * disp


def _sample_height(Zgrid, xs, ys, px, py):
    """Bilinear sample of a height grid Zgrid[iy, ix] at world (px, py)."""
    import numpy as np

    nx = len(xs)
    ny = len(ys)
    fx = (px - xs[0]) / (xs[-1] - xs[0]) * (nx - 1)
    fy = (py - ys[0]) / (ys[-1] - ys[0]) * (ny - 1)
    fx = min(max(fx, 0.0), nx - 1.000001)
    fy = min(max(fy, 0.0), ny - 1.000001)
    ix = int(np.floor(fx))
    iy = int(np.floor(fy))
    tx = fx - ix
    ty = fy - iy
    z00 = Zgrid[iy, ix]
    z10 = Zgrid[iy, ix + 1]
    z01 = Zgrid[iy + 1, ix]
    z11 = Zgrid[iy + 1, ix + 1]
    a = z00 * (1 - tx) + z10 * tx
    b = z01 * (1 - tx) + z11 * tx
    return float(a * (1 - ty) + b * ty)


# --------------------------------------------------------------------------- #
# USD helpers (lazy pxr / omni imports).
# --------------------------------------------------------------------------- #
def _add_semantics(prim, label):
    """Tag a prim with a USD Semantics 'class' label that Replicator reads.

    Tries the Isaac core utility first (the recommended path on Isaac Sim 6.0),
    then falls back to the legacy ``pxr.Semantics.SemanticsAPI`` directly.
    """
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
    """Create a UsdGeom.Mesh from numpy arrays. Returns the Usd.Prim."""
    from pxr import UsdGeom, Gf
    import numpy as np

    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(_vec3f_array(points))
    mesh.CreateFaceVertexCountsAttr(_int_array(face_counts))
    mesh.CreateFaceVertexIndicesAttr(_int_array(face_indices))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    pmin = np.asarray(points, dtype=np.float64).min(axis=0)
    pmax = np.asarray(points, dtype=np.float64).max(axis=0)
    mesh.CreateExtentAttr(
        [
            Gf.Vec3f(float(pmin[0]), float(pmin[1]), float(pmin[2])),
            Gf.Vec3f(float(pmax[0]), float(pmax[1]), float(pmax[2])),
        ]
    )
    if normals is not None:
        mesh.CreateNormalsAttr(_vec3f_array(normals))
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    return mesh.GetPrim()


def _author_sky_dome(stage, path, radius, sun_dir, hole_deg, n_long=72, n_lat=36):
    """Author an inward-facing sky-dome sphere mesh with a spherical-cap HOLE cut
    around ``sun_dir``. The hole lets a DistantLight (at infinity) reach the scene
    — a fully closed dome would occlude the sun and leave the terrain in shadow —
    while the camera (which faces away from the sun) still sees only dome = sky.
    Returns the Usd.Prim.
    """
    import numpy as np

    lons = np.linspace(0.0, 2.0 * np.pi, n_long + 1)
    lats = np.linspace(-np.pi / 2.0, np.pi / 2.0, n_lat + 1)
    LO, LA = np.meshgrid(lons, lats)  # (n_lat+1, n_long+1)
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
                continue  # face lies in the sun-facing cap -> leave it open
            faces.append((a, b, c, d))

    face_indices = np.array(faces, dtype=np.int32).ravel()
    face_counts = np.full(len(faces), 4, dtype=np.int32)
    prim = _author_mesh(stage, path, points, face_counts, face_indices)
    from pxr import UsdGeom

    UsdGeom.Mesh(prim).CreateDoubleSidedAttr(True)
    return prim


def _make_preview_material(
    stage, path, diffuse, roughness, metallic=0.0, emissive=None, unlit=False
):
    """Author a UsdPreviewSurface material. Returns UsdShade.Material.

    ``unlit=True`` uses the specular workflow with specularColor=0 so the surface
    has NO diffuse and NO specular response — it renders as its emissiveColor
    regardless of scene lighting. Used for the sky dome so stray sun rays through
    the dome's open cap can't flare a grazing-angle Fresnel highlight onto it.
    """
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


def _bind_material(prim, material):
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI(prim).Bind(material)


def _basis_matrix(forward, translate, up=(0.0, 0.0, 1.0)):
    """Gf.Matrix4d for a frame whose local -Z points along ``forward`` (camera/light)."""
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


# --------------------------------------------------------------------------- #
# Scene authoring.
# --------------------------------------------------------------------------- #
def build_lunar_stage(seed: int, params: dict | None = None) -> None:
    """Author the lunar scene into the current Replicator / USD context.

    Parameters
    ----------
    seed : int
        Random seed for deterministic terrain displacement + rock scatter.
    params : dict | None
        Domain-randomization overrides. Any key from ``DEFAULT_PARAMS`` (sun
        elevation/azimuth/intensity, regolith albedo/roughness, terrain amplitude,
        rock/crater counts + scales, near-rock band, embedding depth, camera
        height/pitch/FOV, dome radius, star count) replaces its default. When
        ``params is None`` the scene reproduces the validated Session-3 nominal.

    Side effects
    ------------
    Authors /World/{Regolith, Rocks, Sun, StarDome, RoverCam} into the stage
    returned by ``omni.usd.get_context().get_stage()``. All visible geometry is
    tagged with USD Semantics classes (regolith / rock / sky) so Replicator can
    emit pixel-perfect masks. Returns None.
    """
    import numpy as np
    from pxr import Usd, UsdGeom, UsdLux, Gf, Sdf
    import omni.usd

    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)

    rng = np.random.RandomState(int(seed) & 0x7FFFFFFF)
    # Dedicated stream for scene-element COUNTS only, so changing any other DR knob
    # (which consumes a variable number of `rng` draws) doesn't shift n_rocks/craters.
    count_rng = np.random.RandomState((int(seed) ^ _COUNT_RNG_XOR) & 0x7FFFFFFF)

    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    # Side-effecting prim definition: authoring /World/Looks as a Scope so the
    # material prims below have a parent. Binding is the side effect; the handle
    # itself is intentionally unused.
    looks = UsdGeom.Scope.Define(stage, "/World/Looks")  # noqa: F841  (prim side-effect)

    # --- Materials ---------------------------------------------------------- #
    # Regolith albedo is a single gray scalar; scale the warm tint so the default
    # albedo (mean of REGOLITH_TINT) reproduces the validated diffuse exactly.
    tint = np.asarray(REGOLITH_TINT, dtype=np.float64)
    regolith_diffuse = tuple((float(p["regolith_albedo"]) / float(tint.mean())) * tint)
    regolith_mat = _make_preview_material(
        stage, "/World/Looks/RegolithMat",
        diffuse=regolith_diffuse, roughness=float(p["regolith_roughness"]), metallic=0.0,
    )
    rock_mat = _make_preview_material(
        stage, "/World/Looks/RockMat",
        diffuse=ROCK_DIFFUSE, roughness=ROCK_ROUGHNESS, metallic=0.0,
    )
    sky_mat = _make_preview_material(
        stage, "/World/Looks/SkyMat",
        diffuse=(0.0, 0.0, 0.0), roughness=1.0, metallic=0.0,
        emissive=(0.010, 0.010, 0.016), unlit=True,  # near-black space, ignores lighting
    )
    star_mat = _make_preview_material(
        stage, "/World/Looks/StarMat",
        diffuse=(0.0, 0.0, 0.0), roughness=1.0, metallic=0.0,
        emissive=(4.0, 4.0, 4.2), unlit=True,
    )

    # --- Regolith ground (displaced heightfield) ---------------------------- #
    size = TERRAIN_SIZE_M
    res = TERRAIN_RES
    xs = np.linspace(-size / 2.0, size / 2.0, res + 1)
    ys = np.linspace(-size / 2.0, size / 2.0, res + 1)
    X, Y = np.meshgrid(xs, ys, indexing="xy")  # shape (ny, nx) = (Y, X)

    # Layered undulation + finer ripples (primary amplitude is domain-randomized).
    terr_amp = float(p["terrain_amplitude"])
    Z = _fractal_noise_2d(X, Y, seed, octaves=5, base_freq=TERRAIN_BASE_FREQ_LOW) * terr_amp
    Z += (
        _fractal_noise_2d(X, Y, seed + 7, octaves=4, base_freq=TERRAIN_BASE_FREQ_HIGH)
        * (terr_amp / 3.2 * TERRAIN_RIPPLE_AMP)
    )

    # A handful of craters (parabolic bowl + raised rim). Count from the dedicated
    # count RNG (range is the DR knob); per-crater params stay on the main rng.
    cmin, cmax = p["crater_count_range"]
    n_craters = int(count_rng.randint(int(cmin), int(cmax)))
    for _ in range(n_craters):
        cx = rng.uniform(-size / 2.4, size / 2.4)
        cy = rng.uniform(-size / 2.4, size / 2.4)
        R = rng.uniform(10.0, 38.0)
        depth = rng.uniform(1.5, 5.5)
        rim = depth * rng.uniform(0.18, 0.35)
        d = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        dn = d / R
        bowl = np.where(dn < 1.0, -depth * (1.0 - dn ** 2), 0.0)
        rim_bump = rim * np.exp(-(((d - R) / (0.30 * R)) ** 2))
        Z += bowl + rim_bump

    # Per-vertex normals from the height gradient (Z-up heightfield).
    dy = ys[1] - ys[0]
    dx = xs[1] - xs[0]
    dZdy, dZdx = np.gradient(Z, dy, dx)
    nrm = np.stack([-dZdx, -dZdy, np.ones_like(Z)], axis=-1)
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)

    points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    normals = nrm.reshape(-1, 3)

    nx = res + 1
    ny = res + 1
    ixg, iyg = np.meshgrid(np.arange(res), np.arange(res))
    i00 = (iyg * nx + ixg).ravel()
    i10 = i00 + 1
    i01 = i00 + nx
    i11 = i00 + nx + 1
    face_indices = np.stack([i00, i10, i11, i01], axis=1).ravel()
    face_counts = np.full(res * res, 4, dtype=np.int32)

    regolith_prim = _author_mesh(
        stage, "/World/Regolith", points, face_counts, face_indices, normals=normals
    )
    _bind_material(regolith_prim, regolith_mat)
    _add_semantics(regolith_prim, "regolith")

    # --- Rocks (noise-deformed icospheres) ---------------------------------- #
    UsdGeom.Xform.Define(stage, "/World/Rocks")
    base_pts, base_faces = _icosphere(2)  # 162 verts / 320 faces
    face_counts_rock = np.full(len(base_faces), 3, dtype=np.int32)
    face_indices_rock = base_faces.ravel()

    # Camera sits near (0, -90); rocks are scattered ahead (toward +Y) with a
    # few deliberately large near-field boulders to guarantee sane coverage.
    rmin, rmax = p["rock_count_range"]
    n_rocks = int(count_rng.randint(int(rmin), int(rmax)))
    n_near = int(p["near_rock_count"])
    near_y_lo, near_y_hi = p["near_rock_y_range"]
    near_s_lo, near_s_hi = p["near_rock_scale_range"]
    far_s_lo, far_s_hi = p["rock_scale_range"]
    embed_frac = float(p["embedding_depth_frac"])
    for i in range(n_rocks):
        r_rng = np.random.RandomState((int(seed) * 100003 + i * 9176 + 1) & 0x7FFFFFFF)

        if i < n_near:
            px = r_rng.uniform(-55.0, 55.0)
            py = r_rng.uniform(float(near_y_lo), float(near_y_hi))  # ahead of the camera
            base_scale = r_rng.uniform(float(near_s_lo), float(near_s_hi))
        else:
            px = r_rng.uniform(-95.0, 95.0)
            py = r_rng.uniform(-78.0, 230.0)
            base_scale = r_rng.uniform(float(far_s_lo), float(far_s_hi))
            if r_rng.rand() < 0.15:
                base_scale *= r_rng.uniform(2.0, 3.3)  # occasional boulder

        mult = _rock_radius_multiplier(base_pts, r_rng)
        rock_pts = base_pts * mult[:, None]

        prim = _author_mesh(
            stage, "/World/Rocks/Rock_%03d" % i,
            rock_pts, face_counts_rock, face_indices_rock,
        )

        sx = base_scale * r_rng.uniform(0.8, 1.3)
        sy = base_scale * r_rng.uniform(0.8, 1.3)
        sz = base_scale * r_rng.uniform(0.5, 0.9)  # flattened boulders
        rx = r_rng.uniform(0, 360)
        ry = r_rng.uniform(0, 360)
        rz = r_rng.uniform(0, 360)
        ground = _sample_height(Z, xs, ys, px, py)
        cz = ground + sz * embed_frac  # partially embedded in the regolith

        m_scale = Gf.Matrix4d().SetScale(Gf.Vec3d(sx, sy, sz))
        rot = (
            Gf.Rotation(Gf.Vec3d(1, 0, 0), rx)
            * Gf.Rotation(Gf.Vec3d(0, 1, 0), ry)
            * Gf.Rotation(Gf.Vec3d(0, 0, 1), rz)
        )
        m_rot = Gf.Matrix4d().SetRotate(rot)
        m_trans = Gf.Matrix4d().SetTranslate(Gf.Vec3d(px, py, cz))
        _set_transform(prim, m_scale * m_rot * m_trans)
        _bind_material(prim, rock_mat)
        _add_semantics(prim, "rock")

    # --- Sun (harsh distant light at low elevation) ------------------------- #
    elevation_deg = float(p["sun_elevation_deg"])  # low sun -> long shadows; no atmosphere
    azimuth_deg = float(p["sun_azimuth_deg"])
    el = math.radians(elevation_deg)
    az = math.radians(azimuth_deg)
    # Direction from scene toward the sun, then the emission (travel) direction.
    s = np.array([math.cos(el) * math.sin(az), math.cos(el) * math.cos(az), math.sin(el)])
    emit_dir = -s

    sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
    sun.CreateIntensityAttr(float(p["sun_intensity"]))
    sun.CreateColorAttr(Gf.Vec3f(*SUN_COLOR))
    sun.CreateAngleAttr(SUN_ANGLE_DEG)  # sun's angular size -> crisp, hard shadows
    _set_transform(sun.GetPrim(), _basis_matrix(emit_dir, translate=(0.0, 0.0, SUN_HEIGHT_M)))

    # --- Star dome (near-enclosing emissive sphere, open at the sun) -------- #
    # A fully closed dome would occlude the distant sun and leave the whole scene
    # in shadow. We cut a cap around the sun direction (which is behind the camera,
    # so never in frame) so sun shadow rays escape while the camera still sees sky.
    dome_radius = float(p["dome_radius"])
    dome_prim = _author_sky_dome(
        stage, "/World/StarDome", dome_radius, sun_dir=s, hole_deg=DOME_HOLE_DEG
    )
    _bind_material(dome_prim, sky_mat)
    _add_semantics(dome_prim, "sky")

    # Stars: tiny emissive spheres just inside the dome, biased to the +Y/up sky
    # the camera actually sees. Tagged "sky" so they fold into the sky class.
    try:
        UsdGeom.Scope.Define(stage, "/World/Stars")
        n_stars = int(p["star_count"])
        star_r = dome_radius - STAR_INSET_M
        for i in range(n_stars):
            d = rng.normal(size=3)
            d = d / np.linalg.norm(d)
            d[1] = abs(d[1]) * 0.85 + 0.15      # bias toward +Y (camera look dir)
            d[2] = abs(d[2]) * 0.9 + 0.1        # bias upward
            d = d / np.linalg.norm(d)
            pos = d * star_r
            star = UsdGeom.Sphere.Define(stage, "/World/Stars/Star_%03d" % i)
            srad = float(rng.uniform(0.5, 1.1))
            star.CreateRadiusAttr(srad)
            star.CreateExtentAttr(
                [Gf.Vec3f(-srad, -srad, -srad), Gf.Vec3f(srad, srad, srad)]
            )
            _set_transform(
                star.GetPrim(),
                Gf.Matrix4d().SetTranslate(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2]))),
            )
            _bind_material(star.GetPrim(), star_mat)
            _add_semantics(star.GetPrim(), "sky")
    except Exception as exc:  # stars are cosmetic — never let them break the scene
        print(">>> star scatter skipped: %r" % exc, flush=True)

    # --- Camera (rover/lander eye height, looking across the terrain) ------- #
    # FOV -> focal length for a fixed square aperture; pitch -> look direction.
    fov_deg = float(p["camera_fov_deg"])
    focal = CAMERA_APERTURE / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    cam = UsdGeom.Camera.Define(stage, "/World/RoverCam")
    cam.CreateFocalLengthAttr(float(focal))
    cam.CreateHorizontalApertureAttr(CAMERA_APERTURE)
    cam.CreateVerticalApertureAttr(CAMERA_APERTURE)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 3000.0))
    pitch = math.radians(float(p["camera_pitch_deg"]))
    eye = (0.0, CAMERA_EYE_Y, float(p["camera_height_m"]))
    # Look toward +Y, tilted by pitch (negative = down). _basis_matrix normalizes.
    fwd = (0.0, math.cos(pitch), math.sin(pitch))
    _set_transform(cam.GetPrim(), _basis_matrix(fwd, translate=eye))

    print(
        ">>> build_lunar_stage: seed=%d  rocks=%d  craters=%d  sun(el=%.1f,az=%.1f,I=%.0f)"
        "  albedo=%.3f  terr_amp=%.2f  cam(h=%.2f,fov=%.1f)"
        % (
            seed, n_rocks, n_craters,
            elevation_deg, azimuth_deg, float(p["sun_intensity"]),
            float(p["regolith_albedo"]), terr_amp,
            float(p["camera_height_m"]), fov_deg,
        ),
        flush=True,
    )


# --------------------------------------------------------------------------- #
# Validation render (__main__).
# --------------------------------------------------------------------------- #
def _verify_and_report(raw_dir):
    """Read the raw label-id mask + labels JSON, remap to the canonical class
    map, and print unique ids + per-class pixel counts. Returns rock coverage %."""
    import os
    import glob
    import numpy as np

    segs = sorted(
        q for q in glob.glob(os.path.join(raw_dir, "**", "*.png"), recursive=True)
        if "semantic_segmentation" in os.path.basename(q)
        and "classid" not in os.path.basename(q)
    )
    jsons = sorted(glob.glob(os.path.join(raw_dir, "**", "*.json"), recursive=True))
    if not segs:
        print(">>> VERIFY: no semantic_segmentation PNG found in %s" % raw_dir, flush=True)
        return None
    # A single render product writes exactly one seg mask + one labels JSON per
    # pass; more than one means a stale prior run leaked in (ambiguous mapping).
    assert len(segs) == 1, "expected exactly 1 seg PNG in %s, found %d: %s" % (
        raw_dir, len(segs), segs,
    )
    assert len(jsons) == 1, "expected exactly 1 labels JSON in %s, found %d: %s" % (
        raw_dir, len(jsons), jsons,
    )
    seg_path = segs[0]
    json_path = jsons[0]

    from PIL import Image

    arr = np.array(Image.open(seg_path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    arr = arr.astype(np.int64)

    raw_unique, raw_counts = np.unique(arr, return_counts=True)
    print(">>> VERIFY semantic mask: %s  shape=%s" % (seg_path, arr.shape), flush=True)
    print(
        ">>> RAW mask unique ids: %s"
        % dict(zip(raw_unique.tolist(), raw_counts.tolist())),
        flush=True,
    )

    # Canonical remap via the shared module helper (raw ids -> {regolith:0,
    # rock:1, sky:2}, background/unlabeled -> 255), derived from the labels JSON.
    canon = canonical_mask_from_json(arr, json_path)

    total = canon.size
    canon_unique = np.unique(canon)
    print(">>> CANONICAL class-id mask (regolith=0, rock=1, sky=2):", flush=True)
    print("       unique values: %s" % canon_unique.tolist(), flush=True)
    for name, cid in CLASS_MAP.items():
        c = int((canon == cid).sum())
        print("       %-9s id=%d : %9d px (%6.2f%%)" % (name, cid, c, 100.0 * c / total), flush=True)
    unl = int((canon == 255).sum())
    if unl:
        print("       UNLABELED      : %9d px (%6.2f%%)" % (unl, 100.0 * unl / total), flush=True)

    rock_pct = 100.0 * float((canon == 1).sum()) / total
    print(">>> ROCK COVERAGE: %.2f%%  (sane target 3-50%%)" % rock_pct, flush=True)

    out_path = os.path.join(os.path.dirname(seg_path), "semantic_segmentation_classid.png")
    Image.fromarray(canon).save(out_path)
    print(">>> wrote canonical class-id mask: %s" % out_path, flush=True)

    # Deterministic fixed-colormap preview (consistent legend across the project).
    colormap = {0: (120, 110, 96), 1: (224, 70, 38), 2: (38, 56, 110), 255: (0, 0, 0)}
    rgb_prev = np.zeros((canon.shape[0], canon.shape[1], 3), dtype=np.uint8)
    for cid, col in colormap.items():
        rgb_prev[canon == cid] = col
    cmap_path = os.path.join(os.path.dirname(seg_path), "semantic_preview_colormap.png")
    Image.fromarray(rgb_prev).save(cmap_path)
    print(">>> wrote fixed-colormap preview: %s" % cmap_path, flush=True)

    # RGB brightness sanity (confirm the frame is actually lit, not all-black).
    rgbs = sorted(glob.glob(os.path.join(raw_dir, "**", "rgb*.png"), recursive=True))
    if rgbs:
        rgb = np.array(Image.open(rgbs[0]).convert("RGB"))
        print(
            ">>> RGB %s  mean=%.1f  p99=%.0f  max=%d"
            % (rgb.shape, float(rgb.mean()), float(np.percentile(rgb, 99)), int(rgb.max())),
            flush=True,
        )
    return rock_pct


def main():
    import os
    import sys
    import glob
    import time
    import argparse

    parser = argparse.ArgumentParser(description="Render one validation frame of the lunar stage.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=os.environ.get("SMOKE_OUT", "/workspace/out"))
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--subframes", type=int, default=16)
    args, _unknown = parser.parse_known_args()

    # Shield Kit/SimulationApp from our CLI flags.
    argv0 = sys.argv[0] if sys.argv else "build_lunar_stage.py"
    sys.argv = [argv0]

    out_dir = args.out
    raw_dir = os.path.join(out_dir, "labelid")
    prev_dir = os.path.join(out_dir, "preview")

    t0 = time.time()

    def phase(msg):
        print(">>> PHASE %s  (+%.1fs)" % (msg, time.time() - t0), flush=True)
        sys.stderr.flush()

    # 1. SimulationApp MUST be created before importing omni.replicator.core.
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True, "renderer": "RayTracedLighting"})
    phase("simapp_ready")

    import omni.replicator.core as rep

    for _ in range(10):
        simulation_app.update()
    phase("ext_started")

    rep.orchestrator.set_capture_on_play(False)

    try:
        # 2. Author the scene into the current USD context.
        build_lunar_stage(args.seed)
        for _ in range(10):
            simulation_app.update()
        phase("scene_built")

        # 3. Render in TWO sequential passes with one render product.
        #    Replicator's semantic_segmentation annotator is a SINGLETON in the
        #    SDG pipeline: its `colorize` flag is global, so two BasicWriters
        #    attached at once (one False, one True) clobber each other. Running
        #    the passes sequentially (attach -> step -> detach) avoids the clash:
        #      Pass A: colorize=False -> raw integer label-id mask (+ RGB), training
        #      Pass B: colorize=True  -> colorized preview, figures
        render_product = rep.create.render_product("/World/RoverCam", (args.width, args.height))

        def _drain(check, budget=300.0):
            deadline = time.time() + budget
            while time.time() < deadline:
                simulation_app.update()
                if check():
                    return True
            return False

        def _segs(d):
            return [
                p for p in glob.glob(os.path.join(d, "**", "*.png"), recursive=True)
                if "semantic_segmentation" in os.path.basename(p)
            ]

        # --- Pass A: raw label-ids + RGB ---
        writer_raw = rep.WriterRegistry.get("BasicWriter")
        writer_raw.initialize(
            output_dir=raw_dir, rgb=True, semantic_segmentation=True,
            colorize_semantic_segmentation=False,
        )
        writer_raw.attach([render_product])
        for _ in range(20):
            simulation_app.update()
        phase("passA_warmup")
        rep.orchestrator.step(delta_time=0.0, rt_subframes=args.subframes)
        ok_a = _drain(
            lambda: bool(glob.glob(os.path.join(raw_dir, "**", "rgb*.png"), recursive=True))
            and bool(_segs(raw_dir))
        )
        phase("passA_files" if ok_a else "passA_timeout")
        try:
            rep.orchestrator.wait_until_complete()
        except Exception as e:
            print(">>> wait_until_complete (A) raised: %r" % e, flush=True)
        writer_raw.detach()

        # --- Pass B: colorized preview ---
        writer_prev = rep.WriterRegistry.get("BasicWriter")
        writer_prev.initialize(
            output_dir=prev_dir, rgb=False, semantic_segmentation=True,
            colorize_semantic_segmentation=True,
        )
        writer_prev.attach([render_product])
        for _ in range(5):
            simulation_app.update()
        rep.orchestrator.step(delta_time=0.0, rt_subframes=args.subframes)
        # Pass B reuses Pass A's now-warm shader cache, so the frame lands fast —
        # a ~90 s budget is ample (vs the ~150 s cold-start budget for Pass A).
        ok_b = _drain(lambda: bool(_segs(prev_dir)), budget=90.0)
        phase("passB_files" if ok_b else "passB_timeout")
        try:
            rep.orchestrator.wait_until_complete()
        except Exception as e:
            print(">>> wait_until_complete (B) raised: %r" % e, flush=True)
        writer_prev.detach()
        phase("drained")

        # 4. Verify the 3-class mask before shutdown.
        _verify_and_report(raw_dir)
    except Exception:
        import traceback

        print(">>> BUILD/RENDER ERROR:\n" + traceback.format_exc(), flush=True)
        sys.stderr.flush()
    finally:
        phase("closing")
        simulation_app.close()

    print("LUNAR_STAGE_DONE output_dir=" + out_dir, flush=True)


if __name__ == "__main__":
    main()
