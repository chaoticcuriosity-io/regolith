# setup-notes.md — regolith build log

This file is the running record of what we actually did — gotchas, deviations, commands that worked, commands that didn't. It absorbs all aarch64/memory gotchas so later sessions don't repeat the same investigations.

---

## Session 1 — 2026-06-26 — scaffold

### Goal
Task 0: scaffold the repo (house-style learning series + pipeline code skeleton), push as private GitHub repo. Task 1 (Isaac Lab setup smoke test) is the next session.

### Completed
- Repo created at `chaoticcuriosity-io/regolith` (private)
- Full directory structure + all stubs authored
- `docs/reports/00-primer.md` fully written
- `scripts/free_memory.sh` written and syntax-checked (`bash -n`)
- Initial commit pushed; SHA recorded in task0 report

### aarch64 / DGX Spark notes (accumulate here)

**Hardware:** NVIDIA DGX Spark (GB10 Grace Blackwell, aarch64, CUDA 13, sm_121, 128 GB unified memory — nvidia-smi reports VRAM as N/A, which is expected)

**Memory discipline:**
- `nvidia-smi` reports VRAM = N/A — this is expected on unified memory. Always use `free -h` instead.
- Keep ~110 GiB free before any Omniverse/training run.
- OOM on unified memory = swap death-spiral → hard reboot, not a clean CUDA OOM. No exceptions.
- Free memory helper: `bash scripts/free_memory.sh` (stops open-webui, ollama-compose, compose-arangodb-1).

**CUDA extension builds:**
- Set `TORCH_CUDA_ARCH_LIST="12.1"` (sm_120/121 binary-compatible).
- Use `pip install --no-build-isolation` (build needs the container's torch).
- On CUDA 13, add `#include <cstdint>` to any source using `uint32_t`/`uintptr_t`.

**Containers:**
- Long-lived: mjlab-dev (humanoid RL), gsplat-dev (3DGS), unsloth-dev — leave running unless near OOM.
- Stop before heavy runs: open-webui, ollama-compose, compose-arangodb-1.
- ComfyUI runs as systemd service (~170 MiB) — fine to leave.
- Base new GPU work on NGC `pytorch:26.03-py3` (PyTorch 2.11 / CUDA 13.2).

**Port forwards:** open-webui owns host :8080. Use `ssh -L <port>:localhost:<port> spark` for Viser / splat viewers.

---

## Session 2 — 2026-06-26 — Isaac Sim 6.0 + Replicator SDG smoke on the Spark

### Goal

Task 1: validate a headless Omniverse Replicator SDG run on the DGX Spark (aarch64, GB10).
No Isaac Lab, no from-source build — use NVIDIA's prebuilt `nvcr.io/nvidia/isaac-sim:6.0.0`
image directly and confirm it produces labeled synthetic images.

### Result

**SUCCESS.** Headless Replicator SDG ran on the Spark (GB10/sm_121) and produced valid
512×512 RGB + 512×512 semantic-segmentation PNGs of a cube on a plane.
Logged in `isaac_smoke.log` on the Spark; artifacts in `/home/chaotic-curiosity/regolith_smoke/`.

### Pinned image

```
nvcr.io/nvidia/isaac-sim:6.0.0
```

- Multi-arch manifest: pulls `linux/arm64` automatically on the Spark, `linux/amd64` on x86.
- Public — no NGC login required.
- 17.6 GB on disk; first pull ≈ 7 min.
- Version string: `6.0.0-rc.59+release.41464`.
- **Do NOT change the tag without re-running the smoke test.**

### Exact run command (generalised as `scripts/run_isaac.sh`)

```bash
docker run --name isaac-sim-smoke --entrypoint bash --gpus all \
  -e "ACCEPT_EULA=Y" -e "PRIVACY_CONSENT=Y" -e "SMOKE_OUT=/workspace/out" \
  --rm --network=host \
  -v /home/chaotic-curiosity/regolith_smoke.py:/workspace/regolith_smoke.py:ro \
  -v /home/chaotic-curiosity/regolith_smoke:/workspace/out:rw \
  nvcr.io/nvidia/isaac-sim:6.0.0 \
  -lc "/isaac-sim/python.sh /workspace/regolith_smoke.py"
```

Inside the container, always invoke scripts via `/isaac-sim/python.sh <script.py>` — this
wrapper sets PYTHONPATH, LD_LIBRARY_PATH, and all Omniverse/CarB env vars before launching
the embedded Python interpreter.

### Memory / co-tenant handling

- Stopped before launch: `docker stop open-webui ollama-compose compose-arangodb-1`.
- Peak Isaac usage: ~8.3 GiB. Available stayed ≥101 GiB throughout.
- Restarted after: `docker start open-webui ollama-compose compose-arangodb-1`.
  Final available: ~110 GiB.
- Use `bash scripts/free_memory.sh` to automate the stop-and-check step.

### Gotchas (aarch64 / GB10 — root cause + fix)

**Gotcha 1 — CRITICAL — do NOT bind-mount `/isaac-sim/.cache`:**
Container runs as uid 1234 (`isaac-sim`). A host-owned bind-mount at `/isaac-sim/.cache`
triggers `PermissionError [Errno 13]` inside `wp.init()` (NVIDIA Warp). This aborts
extension startup and cascades into a chain of misleading errors: "No writer 'BasicWriter'",
`OgnRefTimeGate TypeError`, `rep.orchestrator` becomes `NoneType`. Use the container-internal
cache, or a separate host dir that is `chmod 777`.

**Gotcha 2 — poll-for-files drain, not bare `wait_until_complete()`:**
First RTX frame on the Spark (GB10/sm_121) takes ~150 s. `rep.orchestrator.wait_until_complete()`
fires its internal drain timeout before the frame lands → output dir remains empty. Fix: after
`rep.orchestrator.step()`, pump `simulation_app.update()` in a loop and poll the output dir for
PNGs (180 s budget). Only call `wait_until_complete()` after ≥2 PNG files are confirmed.
A persistent (chmod 777) shader-cache mount eliminates this penalty on reruns.

**Gotcha 3 — don't call `set_extension_enabled_immediate()` synchronously:**
Calling it before the orchestrator is ready causes a crash during `replicator_yaml` startup.
A few `simulation_app.update()` ticks after `import omni.replicator.core` is all that's needed.

**Gotcha 4 — OmniHub/Nucleus "exited with exit status: 101" warnings are benign:**
For primitive-only scenes (no `omniverse://` assets) Nucleus is not required. These log lines
are harmless but can false-trigger naive log watchers looking for "exited with".

**Gotcha 5 — PhysX-GPU + RTX on GB10 (sm_121) work out of the box:**
No source build, no `TORCH_CUDA_ARCH_LIST`, no X11 headers required for the container path.
(The from-source playbook needs GCC-11 + X11 + `LD_PRELOAD=libgomp.so.1` — avoid it.)

**Gotcha 6 — shutdown abort is harmless:**
`simulation_app.close()` ends with the known Isaac shutdown segfault (`Aborted (core dumped)`)
AFTER output files are written. Ignore it; `SMOKE_TEST_DONE` in stdout may not print depending
on buffer state.

### Downstream format decision

The smoke used `colorize_semantic_segmentation=True` → RGBA preview PNG +
`semantic_segmentation_labels_*.json` (color→class mapping).

**For the actual SDG / training pipeline:**
- `colorize_semantic_segmentation=False` → single-channel label-id masks; `dataset.py` reads
  clean class indices directly (no color-to-id conversion step).
- Generate **both**: `colorize=False` label-ids for training, `colorize=True` (or a fixed
  colormap) for gallery/report figures.
- Class map: `{regolith: 0, rock: 1, sky: 2}` (rock = hazard).

### Artifacts left on the Spark

- `/home/chaotic-curiosity/regolith_smoke.py` — original smoke script (superseded by `replicator/_smoke_render.py`)
- `/home/chaotic-curiosity/run_isaac_smoke.sh` — original one-shot launcher (superseded by `scripts/run_isaac.sh`)
- `/home/chaotic-curiosity/regolith_smoke/` — validated output (rgb + semantic PNGs)
- `/home/chaotic-curiosity/isaac_smoke.log` — full stdout/stderr from the validated run

## Reproduce

Run the smoke test from the Spark (or via ssh):

```bash
# 1. Free memory
ssh spark "bash /home/chaotic-curiosity/regolith/scripts/free_memory.sh"

# 2. Run Isaac Sim smoke test
ssh spark "bash /home/chaotic-curiosity/regolith/scripts/run_isaac.sh \
  /home/chaotic-curiosity/regolith/replicator/_smoke_render.py \
  /home/chaotic-curiosity/regolith_smoke_out"

# 3. Verify outputs (expect >= 2 PNGs: rgb + semantic_segmentation)
ssh spark "find /home/chaotic-curiosity/regolith_smoke_out -name '*.png' | sort"

# 4. Restart co-tenants
ssh spark "docker start open-webui ollama-compose compose-arangodb-1"
```

---

## Session 3 — 2026-06-27 — the lunar stage (`scene/build_lunar_stage.py`)

### Goal

Task 1 (USD scene): implement `build_lunar_stage(seed)` — a procedural, asset-free
lunar scene (displaced regolith heightfield + noise-deformed rock scatter + harsh
low-sun + near-black star dome + rover camera) tagged with USD Semantics, plus a
`__main__` that renders one validated 3-class frame at 1024×1024. Live on the Spark.

### Result

**SUCCESS.** One 1024×1024 RGB + raw integer label-id mask + colorized preview, all
three classes present, no unlabeled pixels. Reference frames committed to
`docs/reports/assets/scene-reference-{rgb,seg}.png` (the first real lunar frames).

Per-class pixel counts (seed 42, canonical map `{regolith:0, rock:1, sky:2}`):

```
regolith  id=0 :  476877 px (45.48%)
rock      id=1 :   62360 px ( 5.95%)   <- hazard coverage, in the 3–50% sane band
sky       id=2 :  509339 px (48.57%)
UNLABELED      :       0 px ( 0.00%)
unique values  : [0, 1, 2]
RGB mean=104.5  p99=240  max=252  (lit, not black)
```

Render time: **~21 s warm** (hot shader cache), **~160 s cold** (first SimulationApp
init = RTX shader compile). Persistent dev container + chmod-777 cache mount made the
edit→render loop ~21 s per iteration.

### Scene design (all procedural / numpy — no external assets)

- **Regolith:** 920 m × 920 m grid (300×300 cells ≈ 3 m), fBm value-noise displacement
  + a few parabolic-bowl-with-rim craters; per-vertex normals from the height gradient;
  low-albedo gray `UsdPreviewSurface` (diffuse 0.22, roughness 0.96). Class `regolith`.
- **Rocks:** ~76 noise-deformed icospheres (subdiv-2, radial-sinusoid lumps), random
  non-uniform scale / rotation / placement, partially embedded; ~12 deliberate near-field
  boulders guarantee sane coverage. Class `rock`.
- **Sun:** one `UsdLux.DistantLight`, elevation 17°, azimuth 120°, intensity 14000,
  angle 0.53° (crisp shadows). Behind the camera → long shadows reach toward the lens.
- **Sky:** emissive near-black dome **mesh** (radius 600) with a 34° cap cut out around
  the sun; stars = tiny emissive spheres biased to the camera's view. Class `sky`.
- **Camera:** `/World/RoverCam` at (0, −90, 2.0) m eye height looking +Y across the
  terrain toward the horizon; ~62° FOV (focal 20 / aperture 24).

### Run pattern (one render product, TWO sequential passes — see Gotcha 7)

```
SMOKE_OUT=/workspace/out /isaac-sim/python.sh \
  /workspace/regolith/scene/build_lunar_stage.py --seed 42 --out /workspace/out
```
Pass A: `BasicWriter(rgb=True, semantic_segmentation=True, colorize=False)` → `out/labelid/`
(RGB + raw label-id PNG + int→class JSON). Pass B: `BasicWriter(semantic_segmentation=True,
colorize=True)` → `out/preview/` (colorized seg). The poll-for-files drain from Session 2
is reused for each pass.

### New gotchas (Task 1 scene authoring)

**Gotcha 7 — two BasicWriters share ONE semantic_segmentation annotator.**
The annotator lives once in the SDG pipeline (`/Render/PostProcess/SDGPipeline/
Replicator_semantic_segmentation`); its `colorize` flag is **global**. Attaching a
`colorize=False` writer and a `colorize=True` writer *simultaneously* (even on separate
render products) makes the second silently flip the first ("Annotator … already attached.
Modifying `colorize` from `False` to `True`") → the raw mask is corrupted/blank. Fix:
render **two sequential passes** — attach → step → drain → detach, then repeat.

**Gotcha 8 — a fully-enclosing sky dome occludes the DistantLight → all-black scene.**
A `DistantLight` is at infinity; every sun shadow ray from the terrain hits the
enclosing dome first → 100 % shadow → only emissive stars render. Fix: cut a spherical
cap (≈34°) out of the dome around the **sun** direction so shadow rays escape. Works
here because the sun is behind the camera (azimuth 120°, camera looks +Y) — the hole is
118° off the optical axis and never enters the ~44° half-diagonal frustum, so the camera
still sees only dome = sky (verified: 0 unlabeled pixels).

**Gotcha 9 — sun through the hole flares a Fresnel highlight on the far dome interior.**
Rays passing through the cap graze the camera-facing dome interior; even with
`diffuseColor=0`, UsdPreviewSurface's metallic-workflow dielectric specular (F0≈0.04)
blows up at grazing angles → a bright gray "sky" patch at the horizon. Fix: author the
sky material **unlit** — `useSpecularWorkflow=1` + `specularColor=(0,0,0)` +
`diffuseColor=0` + emissive — so it renders its emissive color regardless of lighting.

**Gotcha 10 — `Gf.Vec3f(numpy.float32, …)` raises `Boost.Python.ArgumentError`.**
Boost wants native Python floats. Cast numpy scalars: `Gf.Vec3f(float(a), float(b), float(c))`.
(Conversely, `Vt.Vec3fArray(np_float32_Nx3)` / `Vt.IntArray(np_int32)` *do* accept numpy
arrays — used for the 90k-vertex mesh points/indices.)

**Note — Replicator label-ids ≠ canonical class ids.** BasicWriter auto-assigns
`0=BACKGROUND, 1=UNLABELLED, 2=sky, 3=rock, 4=regolith` (reserves 0/1). The labels JSON
(`semantic_segmentation_labels_*.json`, keyed by id when `colorize=False`) must be parsed
to remap to the project's `{regolith:0, rock:1, sky:2}`. `build_lunar_stage.py` does this
remap in `_verify_and_report()` and writes `semantic_segmentation_classid.png`.

**Semantics that Replicator reads:** `isaacsim.core.utils.semantics.add_update_semantics(
prim, semantic_label=..., type_label="class")` (legacy `pxr.Semantics.SemanticsAPI` is the
fallback). Verified to populate the BasicWriter masks.

### Artifacts left on the Spark

- `/home/chaotic-curiosity/regolith/scene/build_lunar_stage.py` — synced scene script
- `/home/chaotic-curiosity/regolith_scene_out/labelid/` — rgb + raw label-id PNG + JSON +
  `semantic_segmentation_classid.png` (canonical) + `semantic_preview_colormap.png`
- `/home/chaotic-curiosity/regolith_scene_out/preview/` — colorized seg
- `/home/chaotic-curiosity/regolith_cache/` — persistent chmod-777 shader/Warp cache
- `/home/chaotic-curiosity/isaac_scene.log` — full stdout/stderr

### Reproduce

```bash
# 1. Free memory
ssh spark "bash /home/chaotic-curiosity/regolith/scripts/free_memory.sh"

# 2. Persistent dev container with a warm shader cache (chmod-777 host dir at /isaac-sim/.cache)
ssh spark "mkdir -p /home/chaotic-curiosity/regolith_cache /home/chaotic-curiosity/regolith_scene_out \
  && chmod 777 /home/chaotic-curiosity/regolith_cache /home/chaotic-curiosity/regolith_scene_out \
  && docker run -d --name isaac-dev --entrypoint bash --gpus all --network=host \
       -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
       -v /home/chaotic-curiosity/regolith:/workspace/regolith:rw \
       -v /home/chaotic-curiosity/regolith_scene_out:/workspace/out:rw \
       -v /home/chaotic-curiosity/regolith_cache:/isaac-sim/.cache:rw \
       nvcr.io/nvidia/isaac-sim:6.0.0 -lc 'sleep infinity'"

# 3. Render one validation frame (container uid 1234 owns its outputs — clean them inside)
ssh spark "docker exec isaac-dev bash -lc 'rm -rf /workspace/out/labelid /workspace/out/preview; \
  SMOKE_OUT=/workspace/out /isaac-sim/python.sh \
  /workspace/regolith/scene/build_lunar_stage.py --seed 42 --out /workspace/out'"

# 4. Tear down + restart co-tenants
ssh spark "docker rm -f isaac-dev; docker start open-webui ollama-compose compose-arangodb-1"
```

---

## Session 4 — 2026-06-27 — domain randomization + SDG pipeline (Task 2)

### Goal

Task 2: build the domain-randomized synthetic-data-generation pipeline — refactor the
scene builder to be params-driven, add a randomizer + dataset generator + per-split
configs, validate a small batch on the Spark, and launch the full dataset generation
detached. Datasets live on the Spark (`/home/chaotic-curiosity/regolith_data/`), not git.

### Result

**SUCCESS.** SDG pipeline validated end-to-end at 512² (20 `train_dr` + 10
`test_photoreal` frames, **all zero unlabeled** after the azimuth fix below), then the
full generation (1500 / 750 / 300 = **2550 frames**) launched detached. Warm throughput
**~1.5 s/frame** (subframes=3) / **~1.7 s/frame** (subframes=12); cold first frame ~157 s,
amortized by the persistent chmod-777 shader cache.

### Prerequisite refactor — `scene/build_lunar_stage.py`

- **`canonical_mask_from_json(raw_id_array, labels_json_path)`** — module-level helper
  mapping Replicator **raw** semantic ids → canonical `{regolith:0, rock:1, sky:2}` (bg/
  unlabeled → **255**). Mapping derived from the labels-JSON class **names**
  (`.strip().lower()`), never hardcoded raw ints (Replicator numbers classes dynamically:
  0=BACKGROUND, 1=UNLABELLED, then 2=sky, 3=rock, 4=regolith for this scene). Raises
  `RuntimeError` if the JSON is missing/empty. `__main__`'s verify path now calls it.
- **`build_lunar_stage(seed, params=None)`** — every DR-able literal (sun el/az/intensity,
  regolith albedo/roughness, terrain amplitude, rock/crater count ranges, rock + near-rock
  scale, near-rock band, embedding depth, camera height/pitch/FOV, dome radius, star count)
  reads from `params` with the validated literals as defaults (`DEFAULT_PARAMS`). Structural
  constants promoted to module level (`TERRAIN_SIZE_M`, `SUN_COLOR`, `ROCK_DIFFUSE`, …).
  `params is None` reproduces the Session-3 nominal (camera lens reconstructed from FOV+pitch
  defaults that match the old focal=20/target look to ~1e-5).
- **Decoupled scene-count RNG** — `n_rocks`/crater counts drawn from a dedicated
  `RandomState(seed ^ 0xA11CE)` so sweeping one DR knob (which consumes a variable number of
  `rng` draws) doesn't shift the counts. (Side effect: the per-seed count for a given seed
  differs from Session 3 — expected; re-validated.)
- Tidies: verify path now asserts **exactly one** seg PNG + one labels JSON; Pass B drain
  budget tightened to 90 s (warm); `looks` Scope annotated as a side-effecting prim def.

### SDG code

- **`replicator/randomizers.py` → `sample_params(rng, cfg)`** — draws a `params` dict.
  *Scalar knobs* (sun, albedo, roughness, terrain amp, embedding, camera, star count) get
  one fresh uniform draw/frame from `cfg["ranges"]`. *Range pass-through knobs* (crater/rock
  count, rock + near-rock scale, near-rock band) are handed through as ranges — per-element
  variation happens inside the builder. `mode: nodr` returns the fixed `cfg["nominal"]`
  (no draws) — the ablation control. Output is JSON-clean (Python float/int).
- **`replicator/generate_dataset.py`** — `--config --n --out --seed [--res 512] [--subframes]
  [--mode] [--start-index] [--preview-stride]`. Reuses **one** SimulationApp; per frame:
  `new_stage()` → `build_lunar_stage(seed_i, params_i)` → one render product on
  `/World/RoverCam` → `BasicWriter(colorize=False)` → validated poll-drain → remap via
  `canonical_mask_from_json` → save `rgb/rgb_XXXXX.png` + `mask/mask_XXXXX.png` (uint8
  {0,1,2,255}). Writes `frames.jsonl` (crash-safe), `manifest.json` (class map, per-frame
  seed+params, aggregate class fractions), `progress.txt` (live ETA). Colormap previews
  every `--preview-stride` frames (pure numpy on the canonical mask — no second render pass).
- **`replicator/configs/`** — `train_dr.yaml` (wide DR), `train_nodr.yaml` (frozen domain,
  same content distribution = ablation control), `test_photoreal.yaml` (subframes=12 +
  UNSEEN ranges: higher/brighter sun, brighter+smoother regolith, rougher terrain, bigger
  rocks, higher+wider+down-pitched camera = the domain-gap eval split).

### Small-batch validation (proof)

- **train_dr** (20 frames, subframes=3, 512²): mask ids exactly `{0,1,2}`, **0 unlabeled**;
  per-class fractions regolith ~0.49 / **rock mean ≈ 0.038 (range 0.010–0.079)** / sky ~0.46.
- **test_photoreal** (10 frames, subframes=12): ids `{0,1,2}`, **0 unlabeled**; rock spans
  0.014–0.534 (large near-field boulders fill some frames — intended for the harder split).
- **rock = id 1 confirmed** — visually (preview red blob aligns pixel-perfect with the RGB
  boulder) and statistically (rock is the sparse few-% class; regolith/sky dominate — not
  scrambled). RGB sane: mean ~82, p99 215, lit not black.

### New gotchas (Task 2)

**Gotcha 11 — full sun-azimuth DR puts the sky-dome's sun-hole IN FRAME → unlabeled pixels.**
The dome cuts a 34° cap around the sun so the DistantLight escapes (Session-3 Gotcha 8). With
the Session-3 fixed az=120 the hole was safely off the optical axis, but randomizing azimuth
over `[0,360]` swings it into the forward frustum at low sun → the camera sees through to the
void → 8–36 % BACKGROUND (→255) pixels. **Fix:** constrain `sun_azimuth_deg` to **`[100,260]`**
(camera looks +Y / az≈0; this keeps the hole ≥ ~57° off-axis even at the worst corner with a
512² camera offset −90 m in Y). Trade-off: sun stays in the rear/side hemisphere (back/side
lighting only) — a documented limitation of the procedural-dome approach, not a bug. After the
fix: 30/30 small-batch frames had **0 unlabeled**.

**Gotcha 12 — RayTracedLighting renders internally at ~256² then DLSS-upscales to 512².**
Log warns `DLSS … Render resolution of (256, 256) is below minimal input resolution of 300`.
Benign for us: the **semantic mask is exact at native 512²** (seg is not DLSS-upscaled); only
the RGB is mildly softened — acceptable for a training set, and it makes subframe count nearly
free (subframes 3 vs 12 ≈ 1.5 s vs 1.7 s/frame). If crisper RGB is ever needed, disable DLSS or
raise resolution.

**Note — no memory leak across frames.** Per-frame render-product `.destroy()` + writer
`.detach()` keep usage flat; the container sits at ~80 MiB idle between SimulationApp runs and
Isaac peaks ~7 GiB during generation. 103 GiB stayed free with all co-tenants up.

### Memory / co-tenant handling

Available memory was **ample (110 GiB at idle)**, Isaac peaks ~7 GiB, so per the run-pattern
guidance the co-tenants (`open-webui`, `ollama-compose`, `compose-arangodb-1`) were **LEFT
RUNNING** for the multi-hour job (103 GiB free during generation — comfortable margin). No
`free_memory.sh` needed this session.

### Full generation — chosen counts + launch

- **Counts (final): 1500 `train_dr` + 750 `train_nodr` + 300 `test_photoreal` = 2550 frames.**
  At ~1.5–1.7 s/frame this is **~75 min wall-clock** — far inside the 6–8 h budget (the
  measured throughput let us hit the full target rather than scaling down). Sized as a demo to
  show the DR-vs-no-DR lesson, not production scale.
- **Launched detached** inside the persistent `isaac-dev` container via `docker exec -d`,
  chaining the three splits sequentially, logging to `/home/chaotic-curiosity/regolith_data/gen.log`.
  Each split writes its own `progress.txt` + `manifest.json`. Datasets live under
  `/home/chaotic-curiosity/regolith_data/{train_dr,train_nodr,test_photoreal}/` — **not in git**.

### Reproduce

```bash
# 0. Persistent dev container (warm shader cache; data dir chmod 777; co-tenants left up)
ssh spark "mkdir -p /home/chaotic-curiosity/regolith_data && chmod 777 /home/chaotic-curiosity/regolith_data \
  && docker run -d --name isaac-dev --entrypoint bash --gpus all --network=host \
       -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
       -v /home/chaotic-curiosity/regolith:/workspace/regolith:rw \
       -v /home/chaotic-curiosity/regolith_data:/workspace/data:rw \
       -v /home/chaotic-curiosity/regolith_cache:/isaac-sim/.cache:rw \
       nvcr.io/nvidia/isaac-sim:6.0.0 -lc 'sleep infinity'"

# 1. Sync code (no local rsync on the Windows box -> tar over ssh)
#    tar --exclude=__pycache__ -cf - scene replicator | ssh spark "cd /home/chaotic-curiosity/regolith && tar -xf -"

# 2. Launch full generation (detached; survives the launching shell)
ssh spark "docker exec -d isaac-dev bash -lc '
  cd /workspace/regolith
  /isaac-sim/python.sh replicator/generate_dataset.py --config replicator/configs/train_dr.yaml      --n 1500 --out /workspace/data/train_dr       --seed 42   --res 512 >> /workspace/data/gen.log 2>&1
  /isaac-sim/python.sh replicator/generate_dataset.py --config replicator/configs/train_nodr.yaml    --n 750  --out /workspace/data/train_nodr     --seed 42   --res 512 >> /workspace/data/gen.log 2>&1
  /isaac-sim/python.sh replicator/generate_dataset.py --config replicator/configs/test_photoreal.yaml --n 300  --out /workspace/data/test_photoreal --seed 7777 --res 512 >> /workspace/data/gen.log 2>&1'"

# 3. Check progress (one-liner)
ssh spark "cat /home/chaotic-curiosity/regolith_data/{train_dr,train_nodr,test_photoreal}/progress.txt 2>/dev/null; tail -3 /home/chaotic-curiosity/regolith_data/gen.log"
```

---

## Session 5 — 2026-06-27 — RTX resource-descriptor leak: the detached full-gen stall + the fix

### What happened

The Session-4 detached full generation chained all three splits **sequentially inside ONE
long-lived `isaac-dev` container** (`docker exec -d`, three `generate_dataset.py` calls).
`train_dr` completed **1500/1500**. The next process, `train_nodr` (a fresh PID in the same
container), died at **frame 136** with:

```
[Warning] [carb] Plugin interface for a client: omni.hydratexture.plugin was already released.
[Warning] [omni.graph.core.plugin] …Replicator_semantic_segmentation… Illegal cycle connection
    from …Replicator_semantic_segmentation.outputs:exec to …WriterSyncGate.inputs:execIn ignored
[Fatal] [omni.rtx] Out of resource descriptors!
```

…then hung (PID spinning ~663 % CPU, GPU 94 %). `test_photoreal` never started. Memory was
never the problem (100+ GiB free throughout) — this is a **GPU resource-descriptor leak**, not RAM.

### Root cause (confirmed empirically, not just hypothesised)

1. **The SDG render loop leaks RTX resource descriptors per frame.** `generate_dataset.py`
   creates a fresh `render_product` + `BasicWriter` every frame (`rep.create.render_product(...)`
   → `writer.attach` → step → `writer.detach` → `rp.destroy()`). The hydra render target /
   SDG OmniGraph teardown is not clean (hence the `hydratexture … already released` +
   `Illegal cycle connection` flood). Each frame leaks ~one descriptor set. Frame time also
   creeps up as the leak grows (measured: 1.5 s → 4.1 s over 750 frames).
2. **`SimulationApp.close()` segfaults on shutdown without releasing GPU descriptors**
   (Session-2 Gotcha 6). So the leak is **not reclaimed on process exit**.
3. **The leaked pool is HOST/driver-level — it survives `docker rm -f` too.** Proven by the
   recovery sequence below: the budget fell monotonically across back-to-back runs
   **1500 → 136 → 18 → 0** (a genuinely-clean fresh container, launched immediately after
   killing the predecessor, died on **frame 0**).
4. **The pool reclaims only LAZILY**, after the leaking container is fully gone AND the GPU
   sits idle (~2–3 min at 0 % util). After ~5 min idle, a fresh `n=3` probe completed cleanly
   — the pool had recovered. A single fresh process from a **reclaimed** pool sustains
   **≥1500** frames (train_dr proved it).
5. **A GPU reset is impossible here.** `nvidia-smi --gpu-reset -i 0` →
   *"GPU … is the primary GPU"* (the single GB10 drives Xorg/the desktop). `sudo` is not
   passwordless, and a reboot would destroy the long-running co-tenants (gsplat/mjlab/unsloth).
   **Idle reclamation is the only practical recovery.**

**Why the original run failed:** all three splits shared **one** host descriptor pool with
**zero reclaim time** between them. `train_dr`'s 1500 frames drained the shared pool;
`train_nodr` started on the depleted remainder → exhausted it at frame 136. The Task-2 prompt's
"fresh container (clean GPU context) per split" framing is **necessary but not sufficient** —
a fresh *container* does NOT get a fresh *descriptor pool* (the pool is host-level). The missing
ingredient is an **idle-reclaim gap between splits**.

### The rule (durable fix)

- **One fresh container per split** (fresh process), AND
- **a GPU-idle reclaim gap between splits** (~2–3 min, GPU at 0 % util, no Isaac container up),
  so the driver reclaims the prior split's leaked descriptors before the next split starts.
- **Keep any single split well under the ~1500-frame clean-pool budget.** Chunk splits
  >~1000 frames into ≤~750-frame pieces, each in a fresh container + reclaim gap, continuing
  the frame indices and per-frame seed with `--start-index` (seeds are deterministic from
  `base_seed ^ ((i+1)*2654435761)`, so resuming at index N continues the exact sequence with
  no gaps/dupes). Here neither remaining split needed chunking (750 and 300 each < 1500).
- **NEVER run all splits back-to-back in one long-lived container.** Any future full-dataset
  regen must use `scripts/generate_all.sh` (one fresh container + reclaim gap per split).
- **Monitoring caveat:** when a watcher tails a log that the next run truncates with `>`, a
  `grep` for `Out of resource descriptors` can match the **previous** run's leftover line
  before truncation (a false positive that bit us twice). Guard it: confirm the new run's
  header line is present first.

### Recovery performed (this session)

1. `docker rm -f isaac-dev` — killed the hung PID 7080, freed the leaked GPU context. GPU → 0 %.
2. Deleted the partial `train_nodr` (136 frames) — files are owned by container uid 1234, so
   the host user can't `rm` them; deleted via a throwaway uid-1234 container
   (`docker run --rm --entrypoint bash -v …regolith_data:/workspace/data <image> -lc 'rm -rf …'`).
   **Left `train_dr` (1500) untouched.**
3. Diagnosed the host-level leak (budget 1500→136→18→0); established that ~5 min idle reclaims
   the pool (clean `n=3` probe).
4. **`train_nodr` (750, seed 42):** fresh container, launched after a confirmed idle gap.
   Passed frame 136 decisively and completed **750/750** (status=done, clean shutdown). This
   proved the fix.
5. Idle gap, then **`test_photoreal` (300, seed 7777, subframes=12):** fresh container,
   completed **300/300** (status=done). subframes=12 was not meaningfully slower than
   subframes=3 (DLSS upscales from ~256² regardless — Session-4 Gotcha 12).

### Final verification (no fabrication)

| split           | masks | rgb  | index range | contiguous | dupes | ignore | sample ids |
|-----------------|-------|------|-------------|------------|-------|--------|------------|
| train_dr        | 1500  | 1500 | 0..1499     | yes        | no    | 0.000  | {0,1,2}    |
| train_nodr      | 750   | 750  | 0..749      | yes        | no    | 0.000  | {0,1,2}    |
| test_photoreal  | 300   | 300  | 0..299      | yes        | no    | 0.000  | {0,1,2}    |

Mask spot-checks (first/middle/last + random): canonical ids exactly `{0,1,2}` (rock = 1),
**0.00000 ignore** in every sampled frame; rock is the sparse few-% class (train ~2–10 %,
test up to ~14 % with the larger boulders — as intended). Final host state: no Isaac
containers, GPU 0 % util (only ComfyUI 170 MiB + desktop), **110 GiB free**. Co-tenants were
left running throughout (never stopped → nothing to restore).

### Reproduce

```bash
# Durable full regen — one fresh container + reclaim gap per split:
ssh spark "bash /home/chaotic-curiosity/regolith/scripts/generate_all.sh"

# Per-split manual pattern (what this session did), e.g. train_nodr:
ssh spark "docker rm -f isaac-dev 2>/dev/null; \
  for i in 1 2 3 4 5 6 7; do nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader; sleep 20; done; \
  docker run -d --name isaac-dev --entrypoint bash --gpus all --network=host \
    -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
    -v /home/chaotic-curiosity/regolith:/workspace/regolith:rw \
    -v /home/chaotic-curiosity/regolith_data:/workspace/data:rw \
    -v /home/chaotic-curiosity/regolith_cache:/isaac-sim/.cache:rw \
    nvcr.io/nvidia/isaac-sim:6.0.0 -lc 'sleep infinity'; \
  docker exec -d isaac-dev bash -lc 'cd /workspace/regolith && \
    /isaac-sim/python.sh replicator/generate_dataset.py \
      --config replicator/configs/train_nodr.yaml --n 750 \
      --out /workspace/data/train_nodr --seed 42 --res 512 >> /workspace/data/gen_nodr.log 2>&1'"
# …then remove isaac-dev, idle-reclaim gap, repeat for test_photoreal (seed 7777). NEVER chain splits in one container.
```

---

## Session 6 — 2026-06-27 — training the model + the size-matched DR ablation (Task 3)

### Environment

GPU training ran in a dedicated container, **separate from the Isaac Sim image** (training needs
PyTorch + HuggingFace, not Omniverse):

```bash
# regolith-train: NGC PyTorch with transformers added
docker run -d --name regolith-train --entrypoint bash --gpus all --network=host \
  -v /home/chaotic-curiosity/regolith:/workspace/regolith:rw \
  -v /home/chaotic-curiosity/regolith_data:/workspace/datasets:rw \
  nvcr.io/nvidia/pytorch:26.03-py3 -lc 'pip install transformers && sleep infinity'
```

Versions inside the container: **torch 2.11.0a0 (nv26.03) · CUDA 13 · transformers 5.12.1 ·
matplotlib 3.10.8**, GPU reports as **NVIDIA GB10**. No descriptor-leak problem here — that was
an RTX/SDG render-loop issue (Session 5); plain PyTorch training is clean. Each epoch ran
~16–18 s (750 frames) to ~31 s (1500 frames); co-tenants left running throughout.

### Training-code fixes (reconciled into git this session)

The training pipeline (`training/`, committed in Task-3-prep) ran on the first GPU attempt with
no functional bugs — but it lacked the instrumentation a real ablation needs. The fixes
(all in `training/train.py`, brought back from the Spark; `dataset.py`/`metrics.py`/`model.py`
were unchanged, and `python -m pytest training/test_metrics.py training/test_dataset.py` still
passes **33/33** locally):

- **Early stopping** — new `--patience` flag (default 8) on val rock-IoU. Without it every run
  would burn all 40 epochs well past its peak; with it, no-DR stopped at epoch 17, DR-750 at 31,
  DR-1500 at 35.
- **Per-class IoU** recorded everywhere — added `class_iou: [regolith, rock, sky]` to the
  per-epoch `metrics.jsonl` row, to the saved checkpoint, and to `summary.json`. The headline
  table needs the per-class breakdown, not just rock-IoU + mIoU.
- **`best_epoch`, `epochs_run`, `epoch_sec`, `wall_time_s`** — timing + which epoch the saved
  checkpoint came from, for the results table and the overfitting analysis.

These are additive (no behavior change to the loss, optimizer, or metric definitions), so the
checkpoints produced by the Spark copy are bit-for-bit what the committed code would produce.

### The three runs (identical recipe; only the training data differs)

Common: SegFormer-B0 (`nvidia/mit-b0` ImageNet encoder + fresh 3-class decode head),
class-weighted CE (`ignore_index=255`, rock weight **~2.59** vs regolith/sky ~0.20),
AdamW lr **6e-5**, cosine decay, batch **8**, seed **0**, patience **8**, max 40 epochs,
**val split = `test_photoreal`** (unseen-domain generalization probe, not the training distribution).

| Run | `--train-split` | Frames | DR? |
|-----|-----------------|-------:|:---:|
| `nodr_750` | `train_nodr` | 750 | no |
| `dr_750` | `train_dr_750` | 750 | yes (size-matched) |
| `dr_1500` | `train_dr` | 1500 | yes (deployed) |

`train_dr_750` is the **first 750 frames of `train_dr`, symlinked** (`ln -s ../../train_dr/...`) —
same generation, same seed lineage, truncated to match the control's count. That makes
`dr_750` vs `nodr_750` an exact size-matched comparison: identical 750-frame budget, identical
content distribution, the *only* difference is whether appearance was domain-randomized.

### Results (best val checkpoint on `test_photoreal`; from each run's `summary.json`)

| Run | rock-IoU | mIoU | regolith / rock / sky IoU | best epoch | wall |
|-----|:--------:|:----:|:-------------------------:|:----------:|:----:|
| `nodr_750` | 0.689 | 0.863 | 0.928 / 0.689 / 0.973 | 9 | 278 s |
| `dr_750` | 0.788 | 0.903 | 0.947 / 0.788 / 0.974 | 23 | 507 s |
| `dr_1500` | 0.815 | 0.914 | 0.952 / 0.815 / 0.975 | 27 | 1024 s |

**Independent verification (no fabrication):** a fresh inference pass over all 300
`test_photoreal` frames, straight from `best.pt`, reproduced the global pixel-pooled numbers —
`nodr_750` → rock-IoU **0.6885** / mIoU **0.8629**, `dr_1500` → **0.8147** / **0.9139** — matching
the training-time `summary.json` to the third decimal. The committed figures/overlays come from
that same pass.

### Honest interpretation

- **Headline (size-matched DR):** rock-IoU **0.689 → 0.788 = +0.099 absolute (+14% relative)**,
  dataset size held constant at 750. This isolates domain randomization from data quantity.
- **Secondary (more data):** rock-IoU **0.788 → 0.815 = +0.027** going 750 → 1500 frames with DR
  held constant. Real but ~⅓ the DR effect. Comparing the deployed 1500 directly against the
  no-DR 750 (a +0.126 swing) would have **confounded** DR with data size — hence the symlinked
  size-matched split.
- The gains concentrate on **rock**. Regolith (0.928→0.952) and sky (0.973→0.975) were already
  near-ceiling and barely move; DR buys exactly the hazard-class capability it was meant to.
- **Mechanism = overfitting.** The no-DR model peaked at epoch 9 with train rock-IoU **0.830** but
  val **0.689** (0.141 gap) and *rising* val loss — it memorized its one frozen appearance. The
  DR model's **epoch-1** val rock-IoU (**0.712**) already beat no-DR's best-ever (0.689), because
  randomized appearance leaves nothing to memorize. DR trains longer before plateauing (1024 s vs
  278 s) — cheap for the capability.
- **Caveat that governs the claim:** `test_photoreal` is **synthetic → synthetic** unseen-domain
  transfer (held-out sim parameter ranges), **not** real lunar imagery. The +0.099 is a real
  generalization result but not yet the sim-to-real number. That is Task 4 / chapter 04.

### Post-processing (this session, Task 3 deliverables)

- `docs/reports/assets/ablation-rockiou.png` — rock-IoU bar chart (matplotlib, local), annotated
  with the +0.099 size-matched gain and the +0.027 more-data gain.
- `docs/reports/assets/ablation-results.json` — full results table + verification numbers.
- `docs/reports/assets/ablation-{00029,00118,00071,00154}.png` — 4-panel qualitative overlays
  (RGB | GT | no-DR pred | DR-1500 pred), generated on the Spark from `best.pt`, fixed project
  palette (rock = red). Frames span 2.7 %–30.4 % rock coverage; each shows DR recovering rocks
  the no-DR model drops (e.g. 00118: no-DR labels a bright boulder as *sky*; 00154: no-DR hollows
  a boulder's interior to regolith). DR beats no-DR on **267/300** frames; mean per-frame gain
  **+0.077** rock-IoU (the four chosen overlays, +0.10–0.14, are above-average **legible** cases,
  not the +0.3–0.5 outliers — stated plainly in the chapter).
- `docs/reports/03-training.md` — chapter 03, house voice, ~2.8k words.

Checkpoints (`best.pt`, ~15 MB each) **stay on the Spark** at
`/home/chaotic-curiosity/regolith/outputs/runs/{nodr_750,dr_750,dr_1500}/` — heavy binaries,
git-excluded. The committed artifacts are the figures, the results JSON, and the chapter.

### Reproduce

See `docs/reports/03-training.md` § Reproduce for the three `python training/train.py` commands
(the size-matched `train_dr_750` symlink setup is the first block there). Figure regeneration is
pure post-processing from the saved checkpoints — no retraining required.

## Session 7 — 2026-06-27 — sim-to-real evaluation on real lunar photos (Task 4)

### Goal

Spend the credibility: take the deployed `dr_1500` checkpoint (synthetic val rock-IoU 0.815) and
the `nodr_750` control, freeze them, and run them on **real public-domain lunar photographs** to
see — honestly — what survives contact with real pixels. A visible domain gap is a legitimate
result, not something to hide.

### Eval code (new, committed)

- `eval/_infer.py` — shared inference core: `load_model` (rebuilds `build_model(model_name)` from
  the checkpoint's `model_name`, `load_state_dict`), ImageNet-normalized `preprocess`, `predict`
  (SegFormer logits upsampled H/4 → input size **before** argmax, matching `train.py`), `colorize`
  + alpha `overlay` with the canonical palette `{regolith=(170,140,110), rock=(220,45,40),
  sky=(70,120,205)}`, and `fit_square` (letterbox/cover to 512²).
- `eval/eval_synth.py` — formalizes the held-out synthetic eval: globally-pooled tp/fp/fn → per-class
  IoU (same accumulation as training). **Sanity check passed:** `dr_1500` on `test_photoreal` →
  rock-IoU **0.8147** / mIoU 0.9139 (regolith 0.9521 / rock 0.8147 / sky 0.9750), reproducing the
  training-time 0.815 to the third decimal ⇒ the inference path is faithful.
- `eval/eval_real.py` — CLI (checkpoint, image-dir, out) + optional `--compare-checkpoint` for the
  money panel `[ real RGB | no-DR overlay | DR-1500 overlay ]`. No ground truth on real images ⇒
  **qualitative only, no fabricated IoU**.

Note: the HF `SegformerForSemanticSegmentation LOAD REPORT` (UNEXPECTED `classifier.*`, MISSING
`decode_head.*`) printed at load is **benign** — it's `build_model`'s pretrained-encoder init
*before* `load_state_dict` applies the trained weights; the exact 0.8147 match proves the trained
weights loaded.

### Real images (curation)

7 frames, NASA public domain, fetched over plain HTTPS from the NASA Image & Video Library
(`images-assets.nasa.gov`), spanning **Apollo 11/14/15/16/17** — 1 color + 6 B&W, with-sky and
without-sky compositions. Chosen terrain-dominant; astronaut/lander/rover/flag/Earth frames
rejected as unfair to a 3-class `{regolith,rock,sky}` model. All ~square (aspect 0.995–1.012) so
letterboxing is negligible. Capped to ≤1024 px, committed under `eval/real_images/` with
`sources.md` (NASA IDs + URLs + public-domain statement).

**Kaggle quantitative bonus: SKIPPED — no creds.** No `~/.kaggle/kaggle.json`, no
`KAGGLE_USERNAME`/`KAGGLE_KEY`, on host or in `regolith-train`. Per the plan, did **not** attempt
interactive auth ⇒ no quantitative cross-domain IoU this session. `download_real.py` /
`class_mapping.py` remain ready for a credentialed run.

### Honest findings (qualitative)

- **Transfers well:** the horizon / black-sky boundary (clean on every with-sky frame), and
  detection of **large real rocks** (boulders, the Apollo 16 block, the Apollo 15 cobble field all
  fire red).
- **The gap, visible:** (1) **over-segmentation of rock on fine regolith** — DR labels **~52%** of
  pixels rock on average across the 7 frames (no-DR ~61%), vs 1–8% in training; clearly false on
  smooth-soil frames (Apollo 11 = 42% rock). (2) **Dark shadows → sky** — deep real shadows get
  painted blue (vivid on the Apollo 16 block's cast shadow). (3) **Thin / lens-flare-hazed sky
  missed** (Apollo 15 Hadley).
- **DR vs no-DR on real:** no ground truth ⇒ no numeric claim, but DR consistently predicts **less
  false rock** (6/7 frames) and **largely removes the shadow-as-sky hallucination** no-DR commits
  (clear on Apollo 17 / Apollo 14) — the *same* improvement DR bought on synthetic, now visible on
  real pixels. **DR is better, not fixed**; both still show the core gap.
- Bonus: 6/7 frames are **B&W** (an unseen color regime) yet still produce coherent structure ⇒ the
  transferred cues are shape/shadow, not palette.

### Deliverables

- `docs/reports/04-sim-to-real.md` — chapter 04, house voice, ~2.3k words.
- `docs/reports/assets/real-{01..07}-*.png` — per-photo 3-panel overlays (downsized for repo
  weight); `real-contactsheet.png` overview; `real-predictions.json` (per-class fractions);
  `eval-synth-dr1500.json` (synthetic sanity numbers).

Checkpoints stay on the Spark (git-excluded). Eval ran in `regolith-train`; outputs at
`/workspace/regolith/outputs/eval_{synth,real}/`, pulled back via `scp`.

### Reproduce

See `docs/reports/04-sim-to-real.md` § Reproduce for the two `python eval/eval_{synth,real}.py`
commands (run inside `regolith-train`).

---

## Session 8 — 2026-06-27 — cinematic RTX flythrough + live hazard overlay (Task 5)

### Goal

The portfolio money-shot: a high-quality RTX lunar flythrough with the deployed
`dr_1500` hazard model's predictions overlaid live — a rover's-eye "hazard HUD."
Two-stage pipeline (Isaac renders RGB; PyTorch overlays predictions), assembled to
MP4 + stills + a web preview. Scripted in `render/render_predictions.py`
(`render` / `overlay` / `assemble` subcommands).

### Result

**SUCCESS.** A 142-frame branded flythrough — a slow dolly forward into a boulder
field with a gentle arc/pan/tilt — every boulder segmented as **rock (red + bright
detection outline)**, traversable ground as **safe regolith (green tint)**, sky
untouched. Predictions are the real `dr_1500` model's (mean rock fraction ~0.069,
climbing 5→9% as the camera approaches). Assembled at **24 fps ⇒ 5.9 s**.

- **Full-res MP4 (1920×1080, h264, faststart):**
  `/home/chaotic-curiosity/regolith_render/regolith_flythrough.mp4` — **1.23 MB**
  (stays on the Spark; git-excluded; a GitHub Release asset at publish).
- Committed to `docs/reports/assets/`: `render-hero-{1..4}.png` (1536-wide stills),
  `render-preview.mp4` (1280-wide, 0.18 MB), `render-preview.gif` (720-wide, 2.64 MB).

### Scene + camera (kept IN-DISTRIBUTION so the overlay reads accurately)

`build_lunar_stage(seed=7, HERO_PARAMS)` — a dramatic but in-distribution hero scene
(every value inside the `train_dr.yaml` ranges): low sun **el 11°**, side-back **az
122°** (long raking shadows; the dome sun-hole stays ~120° off-axis, far outside the
frustum), intensity 17000, regolith albedo 0.19, terrain amp 3.7, ~95–125 rocks + 14
near boulders, camera ~2.0 m / fov 62°. A single static stage; the camera MOVES over
it (a 252-pose path: dolly **y −95→−63 m**, lateral arc ±7 m, slow yaw pan +6→−6°,
eased tilt −1.5→−3°, subtle bob). Path saved to `regolith_render/camera_path.json`.

### Render settings + timing

1920×1080, **RayTracedLighting**, `rt_subframes=48`. Warm **~2.4 s/frame**; total
**10.4 min** for the run. rgb_mean stabilises ~108 (the first ~2 captures fade in
69→98→108; dropped via `--skip-head 2`).

### The hard part — RTX color capture (root-caused empirically)

**`BasicWriter` writes BLACK frames on this box.** It captures at step-time, BEFORE
the RTX color pipeline (DLSS temporal accumulation + auto-exposure) converges, and
persists an all-zero `LdrColor`. The *semantic* AOV was perfect throughout, which
masked it as a "scene is fine, only RGB is black" puzzle. Confirmed with an annotator
probe: a `BasicWriter` frame read **mean 0**, the **`LdrColor` annotator read mean ~64
/ fully lit**. **Fix: capture via the `LdrColor` annotator, not BasicWriter, and save
PNGs ourselves.**

**The color pipeline needs a real-motion warmup before it goes lit.** After attach,
the first ~5–100+ readbacks are black; only sustained camera motion (translation +
the path's yaw/arc; static or tiny-jitter does NOT warm it — verified) drives DLSS/
exposure to lit, after which it stays lit and stable. A separate warmup loop did not
carry into the capture (latency-bound). **Robust fix: one continuous capture pass;
SKIP-save until the readback mean > threshold, numbering saved frames contiguously**
— so no black frame is ever written. The warmup length varies (it ate ~108 of the
252 hero poses this run, leaving 144 lit frames = the dramatic *approach* segment).

**Exposure is auto-exposed to ~mean 108–128 (brighter than training ~82) and the
fixed-exposure settings tried (`/rtx/post/histogram/enabled`, `/rtx/post/tonemap/
filmIso|cameraShutter|fNumber`) are IGNORED on this build** (iso 650 vs 1000 gave the
same mean). Handled in Stage B with a `--display-gain 0.78` applied to the COMPOSITE
ONLY — inference runs on the original pixels, so predictions are unchanged; the video
just gets a more dramatic, in-distribution lunar grade.

### Pipeline

- **Stage A (`render`, Isaac):** build hero stage once → single render product + single
  `LdrColor` annotator (reused across all poses) → per-pose: move camera, step, pump
  convergence updates, read+save. Annotator+single-product reuse means NO per-frame
  render-product/writer churn → the Session-5 descriptor leak is avoided.
- **Stage B (`overlay`, PyTorch):** reuse `eval/_infer.py` (ImageNet norm + SegFormer
  H/4→input upsample before argmax) on each frame downscaled to 1024×576, NEAREST-
  upsample the mask to full res, composite (rock = semi-transp red + eroded-edge
  outline; regolith = green tint; sky untouched), then brand (Chaotic Curiosity
  wordmark + caption + legend + corner ticks). `dr_1500/best.pt`.
- **Stage C (`assemble`, host ffmpeg):** full-res faststart MP4 + 4 hero stills +
  a web preview MP4 + a GIF. Run on the **host** (`regolith-train`/overlay containers
  have no ffmpeg; the host does).

### Leak-safe approach used (RTX descriptor-leak rule)

- ONE fresh Isaac container; ONE render product + ONE annotator reused across all poses
  (no per-frame churn); ~260 `orchestrator.step()` calls total — well under the
  ~300-frame-in-one-container / ~1500 clean-pool budgets. Memory ample; co-tenants
  could stay up (I stopped `open-webui`/`ollama-compose` only while diagnosing the
  black-frame issue, then restored them).
- **New gotcha 13 — do NOT `docker rm -f` an Isaac container mid shader-compile.** It
  leaves a stale `regolith_cache/ov/_cache.lock`; the NEXT container then HANGS at boot
  (SimulationApp never starts, GPU 0%, only the config line logged). Fix: remove the
  lock (`rm` it, or via a throwaway uid-1234 container since it's container-owned) and
  relaunch. Let renders finish cleanly instead of force-killing.

### Where things live

- Full MP4: `/home/chaotic-curiosity/regolith_render/regolith_flythrough.mp4` (1.23 MB).
- RGB frames: `regolith_render/rgb/`; overlays: `regolith_render/overlay/`; stills:
  `regolith_render/stills/`; camera path + render meta: `regolith_render/camera_path.json`.
- Committed (git): `render/render_predictions.py`, `docs/reports/assets/render-hero-*.png`,
  `render-preview.mp4`, `render-preview.gif`. `.gitignore` excludes `*.mp4` /
  `regolith_render/` except the negated `docs/reports/assets/render-preview.mp4`.

### Reproduce

```bash
# Stage A — render RGB (fresh Isaac container; warm shader cache; let it FINISH cleanly)
ssh spark "docker run -d --name isaac-render --entrypoint bash --gpus all --network=host \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  -v /home/chaotic-curiosity/regolith:/workspace/regolith:rw \
  -v /home/chaotic-curiosity/regolith_render:/workspace/render_out:rw \
  -v /home/chaotic-curiosity/regolith_cache:/isaac-sim/.cache:rw \
  nvcr.io/nvidia/isaac-sim:6.0.0 -lc 'sleep infinity'"
ssh spark "docker exec isaac-render bash -lc 'cd /workspace/regolith && /isaac-sim/python.sh \
  render/render_predictions.py render --out /workspace/render_out --seed 7 --frames 252 \
  --width 1920 --height 1080 --renderer RayTracedLighting --subframes 48 \
  --prime-steps 8 --lit-threshold 50'"
ssh spark "docker rm -f isaac-render"   # clean shutdown, no force-kill mid-compile

# Stage B — overlay predictions + branding (PyTorch container with transformers)
ssh spark "docker exec regolith-overlay bash -lc 'cd /workspace/regolith && python \
  render/render_predictions.py overlay --checkpoint outputs/runs/dr_1500/best.pt \
  --rgb-dir /workspace/render_out/rgb --out /workspace/render_out/overlay \
  --display-gain 0.78 --skip-head 2'"

# Stage C — assemble MP4 + stills + preview (host ffmpeg)
ssh spark "python3 /home/chaotic-curiosity/regolith/render/render_predictions.py assemble \
  --overlay-dir /home/chaotic-curiosity/regolith_render/overlay \
  --out /home/chaotic-curiosity/regolith_render --fps 24"
```

---

## Session 9 — 2026-06-27 — public-release polish (Task 6)

### Goal

Final content polish before flipping the repo public. No new experiments; no new renders. Write the missing chapter, update all stale content, fix all placeholder links. Commit and push.

### Completed

**Chapter 05 (`docs/reports/05-the-render.md`) — written.** ~1,900 words in house voice covering:
- In-distribution framing: why the scene was chosen inside `train_dr.yaml` ranges, and why that makes the overlay fair vs. chapter 04's harder real-image test
- `--display-gain 0.78` on composite only — model sees original pixels; predictions are real
- RTX pipeline: `RayTracedLighting`, 48 subframes, 1920×1080, ~2.4 s/frame warm, ~10.4 min total
- The `BasicWriter` black-frame root cause and fix (`LdrColor` annotator + skip-save until mean > threshold + sustained-motion warmup) — documented as a teaching moment
- Three-stage pipeline: Stage A (Isaac render → raw PNGs), Stage B (PyTorch overlay + branding), Stage C (host ffmpeg → MP4 + stills + GIF)
- Reproduce commands (copy-pasteable `ssh spark "docker exec …"` blocks for all three stages)
- "## What you now understand" summary
- "## What you've built" series wrap-up table (00–05) with honest bottom line

**README.md — updated:**
- New `## Results` section with the real ablation numbers (no-DR 0.689, DR 750 0.788 +0.099, DR-1500 0.815), `ablation-rockiou.png`, one real-photo overlay (`real-03-apollo14-large-boulder.png`), `render-preview.gif`, and honest sim-to-real summary
- `## Status` section rewritten: all six chapters marked done with one-line descriptions; "not started" entries removed

**`docs/reports/README.md` — updated:** Status column removed; chapter descriptions rewritten to match what was actually built; 2,550-frame total corrected (was "10 k").

**`docs/index.md` — updated:** Results summary + GIF added above the series table; series table descriptions updated; sister-repo links present.

**Placeholder links fixed (all three):**
- `02-domain-randomization.md`: "coming once the model trains" → live link
- `03-training.md`: "coming next" → live link
- `04-sim-to-real.md`: "coming next" → live link

### Links verified

All internal chapter links (`00-primer.md` → `01-the-lunar-stage.md` → … → `05-the-render.md`) exist and resolve as files on disk. Sister-repo GitHub URLs match the established `chaoticcuriosity-io/` org pattern. GitHub Pages URL `https://chaoticcuriosity-io.github.io/regolith/` cannot be verified locally (Pages not yet active), but `docs/index.md` is correctly placed and formatted for Jekyll/GitHub Pages.

### Commit

`docs/` (all chapters + assets) + `README.md` + `setup-notes.md`
Message: `docs(task6): chapter 05 (the render) + README results section + finalize TOC/links for publish`

---

## Session 10 — 2026-06-27 — v2 realistic rocks: the full redo + the honest fidelity-vs-transfer finding

### Goal

Re-run the entire pipeline on **realistic rocks** (v2), preserve the v1 (low-poly) build as a documented "evolution," and reach the same publishable end-state as v1. The hypothesis going in was the obvious one: photoreal rocks → better model. Branch `v2-realistic` (off `rock-realism` @ `33a21fe`). Retraining approved.

### What changed in the scene (`scene/build_lunar_stage.py`, +286/-21)

Rocks only — the regolith ground was left as the plain displaced sheet (rocks were the ask). The old smooth low-poly faceted icospheres became **realistic noise-displaced basalt boulders**:

- **Subdivision scaled to on-screen size** (`_subdiv_for_scale`): far specks subdiv-2 (162 v), mid subdiv-3, near subdiv-4, hero boulders subdiv-5 (10,242 v / 20,480 f). ~0.7 M tris/scene.
- **Multi-octave 3-D fBm displacement** (`_displace_rock`, `_fbm_3d`, `_value_noise_3d`): coarse fBm lumps + a **ridged facet term** (`1 − |fBm|` → sharp creases/fractures) + medium bumps + fine grain, sampled anisotropically per rock (elongation). **Smooth per-vertex normals** (`_vertex_normals`) kill the faceting.
- **12-material dark-basalt PBR pool** (`_make_rock_material_pool`): per-rock albedo 0.058–0.130 (darker than regolith), warm-gray tint + jitter, roughness 0.85–0.97. All per-rock appearance knobs are domain-randomizable.

### Pipeline re-run (same infra as v1; RTX descriptor-leak rule from Session 5 honored)

- **Dataset** regenerated 1500/750/300 on the realistic rocks (`regolith_data_v2/`; v1 data kept). **~9.4 s/frame** (4–5× v1 — per-frame high-poly geometry rebuild is the cost). Masks `{0,1,2}`, sane fractions, 0 unlabeled, leak-safe.
- **Retrain** (`outputs/runs_v2/`), identical recipe to v1. Best epochs 18/19/13.

### Results — synthetic UP, real DOWN (the whole point)

**Synthetic (`test_photoreal`, val rock-IoU):**

| Run | v1 (blocky) | v2 (realistic) |
|-----|:-----------:|:--------------:|
| nodr_750 | 0.689 | **0.8025** |
| dr_750 (size-matched) | 0.788 | **0.8486** |
| dr_1500 (deployed) | 0.815 | **0.8521** |
| size-matched DR gain | +0.099 | **+0.046** |

dr_1500 per-class [regolith 0.9556, rock 0.8521, sky 0.9730], mIoU 0.9269. `eval_synth` reproduced rock-IoU **0.8520** from `best.pt` (faithful inference path). More data saturates: dr_1500 only +0.004 over dr_750. **Insight:** realistic geometry raised the no-DR floor a lot (0.689→0.8025) → DR has less headroom (gain shrinks +0.099→+0.046); fidelity and DR are partly redundant levers.

**Sim-to-real (7 real Apollo photos, no ground truth → qualitative + pixel-fraction counts):** the v2 model is **WORSE**. It **floods ~83% of real pixels as rock** (regolith ~4%), up from v1's ~52%. On real, `dr_1500` predicts **more** false rock than `nodr_750` on **all 7 frames** — the opposite of synthetic, where DR reduced over-segmentation. New **"rock-cloud"** artifact: rock hallucinated up into the black sky above the horizon (vivid on Apollo 14 cone-crater + large-boulder + Apollo 17). **Mechanism:** photoreal rocks (rough, gray, bumpy, matte) collapsed the rock-vs-regolith boundary toward "any rough gray texture = rock" — and real lunar regolith IS exactly that at photo resolution → it floods. v1's cruder rocks kept the classes visually separable, which accidentally protected transfer.

### Re-render

Reused the realistic beauty frames + **v2 `dr_1500` overlay**; clean on the in-distribution synthetic render (overlay caption shows rock-IoU 0.852). The point in the docs: the clean render is the *same* in-distribution flattery the synthetic benchmark gives — ch04 on real photos is the reality check.

### Framing decision (Don) — EMBRACE THE HONEST TRADEOFF

Headline is the **fidelity-vs-transfer lesson**, not "DR is magic": *higher fidelity + higher synthetic scores ≠ better real transfer; synthetic metrics can mislead; you must test on real.* Lead with it; don't bury it.

### Docs rewrite (this session)

- Rewrote `00`–`05` around the honest arc; **new chapter `06-rock-fidelity.md`** (the v1→v2 evolution, the technique, evolution-rocks + evolution-ablation figures, the full synthetic↑/real↓ story, the lesson) ends the series.
- Chapter `04` fully reoriented to the flood finding (~83% rock; DR worse on real; rock-cloud; mechanism). Chapter `03`: v2 ablation numbers + smaller DR gain + saturation; **removed the v1 per-frame "DR wins 267/300" claim and deleted the v1 `ablation-00XXX.png` / `ablation-perframe.json`** (4 overlays + 1 JSON).
- Swapped `eval-synth-dr1500.json` → v2 (0.852). Preserved v1 "before" figures as `*-v1.*` (ablation-rockiou-v1.png, dr-gallery-v1.png, real-apollo-v1.png, ablation-results-v1.json).
- README / `docs/reports/README.md` / `docs/index.md` reframed around the tradeoff; Status now 7 chapters (00–06).

### Artifacts

Checkpoints + datasets stay on the Spark (`outputs/runs_v2/`, `regolith_data_v2/`) — git-excluded. Committed: v2 figures, v2 JSON, chapters. Scratchpad working copies in `scratchpad/v2-review/`.
