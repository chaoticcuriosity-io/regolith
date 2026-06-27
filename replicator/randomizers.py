"""
randomizers.py — sample domain-randomized parameters for the lunar stage.

The lunar scene is authored in pure Python (``scene/build_lunar_stage.py``), not as
a Replicator randomizer graph, so domain randomization happens here: ``sample_params``
draws a fresh ``params`` dict per frame (governed by ranges from a YAML config) that
``build_lunar_stage(seed, params)`` consumes. This keeps the randomization policy in
one inspectable place and lets the same scene builder serve every split.

Two knob families
-----------------
* **Scalar knobs** — one fresh value per frame, drawn uniformly from ``cfg["ranges"]``:
  sun elevation / azimuth / intensity, regolith albedo / roughness, terrain amplitude,
  embedding depth, camera height / pitch / FOV, star count.
* **Range pass-through knobs** — handed to ``build_lunar_stage`` *as a range*; the
  per-element variation (which rock is how big, how many rocks/craters this seed gets)
  happens inside the builder via its dedicated RNGs. These are: crater count, rock
  count, rock scale, near-rock scale, near-rock placement band. Holding the range fixed
  across frames (while the seed varies) means rock *layout* still differs frame-to-frame
  even in the no-DR control — only the *domain* (lighting / albedo / camera) is frozen.

Config schema (see replicator/configs/*.yaml)
--------------------------------------------
  mode: dr | nodr          # dr -> sample from `ranges`; nodr -> fixed `nominal`
  seed: <int>              # base seed (per-frame seed = base ^ frame-mix in generate_dataset)
  res: <int>               # render resolution (square)
  rt_subframes: <int>      # RTX accumulation subframes per frame
  ranges:                  # used when mode == dr
    sun_elevation_deg: [lo, hi]
    ...
  nominal:                 # used when mode == nodr (any omitted key -> build_lunar_stage default)
    sun_elevation_deg: <scalar>
    ...

Notes
-----
The build-time defaults (``scene.build_lunar_stage.DEFAULT_PARAMS``) reproduce the
validated Session-3 nominal scene; any key a config omits simply falls through to them.
"""

from __future__ import annotations

# Scalar DR knobs: one uniform draw per frame (mode == dr).
SCALAR_KEYS = (
    "sun_elevation_deg",
    "sun_azimuth_deg",
    "sun_intensity",
    "regolith_albedo",
    "regolith_roughness",
    "terrain_amplitude",
    "embedding_depth_frac",
    "camera_height_m",
    "camera_pitch_deg",
    "camera_fov_deg",
    "star_count",
)

# Scalar knobs that must be emitted as ints.
INT_SCALAR_KEYS = frozenset({"star_count"})

# Range pass-through knobs: config key -> build_lunar_stage params key.
RANGE_KEYS = {
    "crater_count": "crater_count_range",
    "rock_count": "rock_count_range",
    "rock_scale": "rock_scale_range",
    "near_rock_scale": "near_rock_scale_range",
    "near_rock_y": "near_rock_y_range",
}

# Range knobs whose bounds are integer counts (drawn with randint -> need ints).
INT_RANGE_KEYS = frozenset({"crater_count_range", "rock_count_range"})


def _coerce_scalar(key, value):
    return int(round(float(value))) if key in INT_SCALAR_KEYS else float(value)


def _coerce_range(param_key, lo, hi):
    if param_key in INT_RANGE_KEYS:
        return (int(lo), int(hi))
    return (float(lo), float(hi))


def _nominal_params(cfg) -> dict:
    """Build the fixed ``params`` dict for the no-DR ablation control.

    Reads ``cfg["nominal"]`` (scalars + ranges). Any knob the config omits is left
    out of the returned dict so ``build_lunar_stage`` falls back to its default —
    keeping the control honest (it differs from the validated nominal only where the
    config explicitly says so).
    """
    nom = (cfg.get("nominal") or {})
    p: dict = {}
    for k in SCALAR_KEYS:
        if k in nom:
            p[k] = _coerce_scalar(k, nom[k])
    for ck, pk in RANGE_KEYS.items():
        if ck in nom:
            lo, hi = nom[ck]
            p[pk] = _coerce_range(pk, lo, hi)
    return p


def sample_params(rng, cfg) -> dict:
    """Draw a domain-randomized ``params`` dict for ``build_lunar_stage``.

    Parameters
    ----------
    rng : numpy.random.RandomState
        Per-frame RNG (the caller seeds one stream per frame so frames are
        independent and reproducible).
    cfg : dict
        Parsed YAML config (see module docstring). ``cfg["mode"]`` selects the
        policy: ``"dr"`` samples scalars from ``cfg["ranges"]``; ``"nodr"`` returns
        the fixed ``cfg["nominal"]`` control (no draws from ``rng``).

    Returns
    -------
    dict
        JSON-serializable ``params`` (Python ``float``/``int`` scalars; ranges as
        2-tuples) ready to pass straight to ``build_lunar_stage(seed, params)`` and
        to record verbatim in the dataset manifest.
    """
    mode = str(cfg.get("mode", "dr")).strip().lower()
    if mode == "nodr":
        return _nominal_params(cfg)
    if mode != "dr":
        raise ValueError("config 'mode' must be 'dr' or 'nodr', got %r" % (mode,))

    ranges = cfg.get("ranges")
    if not ranges:
        raise ValueError("config mode 'dr' requires a non-empty 'ranges' block")

    p: dict = {}
    for k in SCALAR_KEYS:
        if k in ranges:
            lo, hi = ranges[k]
            p[k] = _coerce_scalar(k, rng.uniform(float(lo), float(hi)))
    for ck, pk in RANGE_KEYS.items():
        if ck in ranges:
            lo, hi = ranges[ck]
            p[pk] = _coerce_range(pk, lo, hi)
    return p
