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
