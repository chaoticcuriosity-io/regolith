"""randomizers_v3.py — config-driven domain-randomization sampler for the v3 stage.

v3's scene builder (``scene/build_lunar_stage_v3.build_lunar_stage_v3``) consumes a
flat ``params`` dict that overrides ``DEFAULT_PARAMS``. Unlike v2 (whose
``replicator/randomizers.py`` hard-codes the knob names), this sampler is fully
DATA-DRIVEN: the YAML config names exactly which build params to randomize per frame,
which are integer counts, which are passed through verbatim as ``(lo, hi)`` tuples,
and which are held constant. Adding/removing a DR knob is a config edit, not a code
edit — and the keys are the literal ``build_lunar_stage_v3`` param names.

Config schema (replicator/configs_v3/*.yaml)
--------------------------------------------
  mode: dr | nodr
  seed, res, rt_subframes: <int>
  int_keys: [scalar keys to coerce to int — counts]
  const:                 # applied EVERY frame, BOTH modes (YAML lists -> tuples)
    rover_enabled: false
    cam_xy: [0.0, 0.0]
    ...
  passthrough_ranges:    # applied EVERY frame, BOTH modes: handed to the builder
    rock_albedo_range: [0.045, 0.10]   # verbatim as a (lo, hi) tuple (content range,
    rock_roughness_range: [0.85, 0.97] # not a per-frame domain knob)
  ranges:                # mode == dr: one fresh uniform draw per frame -> build param
    sun_elevation_deg: [9, 24]
    ...
  nominal:               # mode == nodr: fixed scalar/tuple per key (frozen domain)
    sun_elevation_deg: 14
    ...

The three splits keep the v2 meaning: ``train_dr`` (full DR), ``train_nodr`` (frozen
domain = ablation control; scene *content* still varies via the per-frame seed),
``test_photoreal`` (unseen ranges = domain-gap test).
"""

from __future__ import annotations


def _as_value(v):
    """YAML list -> tuple (build params like cam_xy / *_range expect tuples)."""
    if isinstance(v, (list, tuple)):
        return tuple(v)
    return v


def sample_params_v3(rng, cfg) -> dict:
    """Draw a JSON-serializable ``params`` dict for ``build_lunar_stage_v3``.

    Parameters
    ----------
    rng : numpy.random.RandomState
        Per-frame RNG (caller seeds one stream per frame -> reproducible draws).
    cfg : dict
        Parsed YAML config (see module docstring). ``cfg['mode']`` selects policy.
    """
    mode = str(cfg.get("mode", "dr")).strip().lower()
    int_keys = set(cfg.get("int_keys") or [])

    out: dict = {}
    # Constants (applied every frame, both modes).
    for k, v in (cfg.get("const") or {}).items():
        out[k] = _as_value(v)
    # Content-distribution ranges handed to the builder verbatim (both modes).
    for k, lohi in (cfg.get("passthrough_ranges") or {}).items():
        out[k] = (float(lohi[0]), float(lohi[1]))

    if mode == "nodr":
        for k, v in (cfg.get("nominal") or {}).items():
            if isinstance(v, (list, tuple)):
                out[k] = tuple(float(x) for x in v)
            elif k in int_keys:
                out[k] = int(round(float(v)))
            else:
                out[k] = float(v)
        return out

    if mode != "dr":
        raise ValueError("config 'mode' must be 'dr' or 'nodr', got %r" % (mode,))

    ranges = cfg.get("ranges")
    if not ranges:
        raise ValueError("config mode 'dr' requires a non-empty 'ranges' block")
    for k, lohi in ranges.items():
        lo, hi = float(lohi[0]), float(lohi[1])
        val = rng.uniform(lo, hi)
        out[k] = int(round(val)) if k in int_keys else float(val)
    return out
