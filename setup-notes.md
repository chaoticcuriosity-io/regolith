# setup-notes.md — regolith build log

This file is the running record of what we actually did — gotchas, deviations, commands that worked, commands that didn't. It absorbs all aarch64/memory gotchas so later sessions don't repeat the same investigations.

---

## Session 1 — 2026-06-26 — scaffold + Isaac smoke

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

## Session 2 — (planned) — Isaac Lab + Omniverse install
*Notes will go here.*
