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
