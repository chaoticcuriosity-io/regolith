# DGX Spark ops manual — regolith

> **Sister document:** the authoritative machine-level manual lives at [`g1-humanoid-rl/docs/dgx-spark-manual.md`](https://github.com/chaoticcuriosity-io/g1-humanoid-rl/blob/main/docs/dgx-spark-manual.md). Read that first. This document is project-specific: it covers only what's different or additive for the `regolith` pipeline.

---

## 1. The machine

**NVIDIA DGX Spark (GB10 Grace Blackwell, aarch64, CUDA 13, sm_121, 128 GB unified memory — nvidia-smi reports VRAM as N/A, which is expected)**

Connect and verify:

```bash
ssh spark "uname -m; free -h; docker ps"
```

Expected output: `aarch64`, then a table with ~128 GB total memory, then the running container list.

---

## 2. Co-tenant containers

| Container | What it is | Policy |
|-----------|-----------|--------|
| `mjlab-dev` | G1 humanoid robot RL stack | Leave running |
| `gsplat-dev` | 3D Gaussian Splatting — graphdeco trainer + CUDA rasterizer | Leave running |
| `unsloth-dev` | LLM fine-tuning | Leave running |
| `open-webui` | Open WebUI (LLM chat frontend) | **Stop before heavy run** |
| `ollama-compose` | Ollama (local model server) | **Stop before heavy run** |
| `compose-arangodb-1` | ArangoDB | **Stop before heavy run** |

ComfyUI runs as a systemd service (~170 MiB) — fine to leave.

Free memory before a generation or training session:

```bash
bash scripts/free_memory.sh
```

This stops the three containers above, prints `free -h`, and exits non-zero if available memory is below 100 GiB. Restart them after your run:

```bash
ssh spark "docker start open-webui ollama-compose compose-arangodb-1"
```

---

## 3. Memory discipline

- Monitor memory with `free -h`, not `nvidia-smi` (VRAM is reported as N/A on unified memory — that's normal).
- Keep ~110 GiB free heading into any Omniverse or training run.
- A unified-memory OOM does not surface as a clean CUDA error — it becomes a swap death-spiral that may require a hard reboot. The 110 GiB margin is not optional.

---

## 4. Running regolith pipeline steps

All pipeline steps run inside a Docker container on the Spark. Base image: `nvcr.io/nvidia/pytorch:26.03-py3`. The training container is `regolith-train` (see `setup-notes.md` for full history).

General pattern:

```bash
ssh spark "docker exec <container> bash -lc 'cd /workspace/regolith && <command>'"
```

Step-by-step reproduce commands live in the `## Reproduce` section of each chapter report in `docs/reports/`.

---

## 5. Run Isaac Sim + Replicator

> **Sister document:** `setup-notes.md ## Session 2` has the full session log, all six
> gotchas with root causes, and the downstream segmentation-format decision.

### Pinned image

```
nvcr.io/nvidia/isaac-sim:6.0.0
```

Multi-arch manifest: pulls `linux/arm64` automatically on the Spark. Public — no NGC
login required. Do not change the tag without re-running the smoke test.

### Step-by-step

**Step 1 — Free memory:**

```bash
bash scripts/free_memory.sh
```

This stops open-webui, ollama-compose, and compose-arangodb-1 (see co-tenant table in
§ 2), prints `free -h`, and exits non-zero if available memory is below 100 GiB.
Isaac Sim peaked at ~8.3 GiB on the validated run; keep ≥110 GiB free as usual.

**Step 2 — Run Isaac Sim:**

```bash
bash scripts/run_isaac.sh <host-script-path> <host-output-dir>
```

Both paths are on the Spark host. The script mounts the Python script read-only and the
output dir read-write, then runs `/isaac-sim/python.sh <script>` inside the container.
The output dir is created and `chmod 777`'d automatically (required — see Warning 1 below).

Example using the smoke script:

```bash
bash scripts/run_isaac.sh \
  /home/chaotic-curiosity/regolith/replicator/_smoke_render.py \
  /home/chaotic-curiosity/regolith_out
```

**Step 3 — Restart co-tenants:**

```bash
docker start open-webui ollama-compose compose-arangodb-1
```

### Warnings

> **Warning — do NOT bind-mount `/isaac-sim/.cache`.**
> Container runs as uid 1234. A host-owned bind-mount at that path triggers
> `PermissionError` in `wp.init()` (NVIDIA Warp), aborting extension startup and
> cascading into misleading errors ("No writer 'BasicWriter'", orchestrator NoneType).
> `scripts/run_isaac.sh` deliberately omits this mount. Do not add it.

> **Warning — use poll-for-files drain, not bare `wait_until_complete()`.**
> First RTX frame on the Spark takes ~150 s. `wait_until_complete()`'s internal timeout
> fires before the frame lands → zero output files. The reference script
> `replicator/_smoke_render.py` implements the correct pattern: pump
> `simulation_app.update()` and poll the output dir for PNGs (180 s budget) before
> calling `wait_until_complete()`.

---

## 6. CUDA extension build notes

When building CUDA extensions from source inside the container:

```bash
TORCH_CUDA_ARCH_LIST="12.1" pip install --no-build-isolation <package>
```

On CUDA 13, any source that uses `uint32_t` or `uintptr_t` needs `#include <cstdint>` added at the top. See `setup-notes.md` for the specific packages this affected.

---

## 7. Port forwarding (web viewers)

`open-webui` occupies host port 8080. For Omniverse/Viser web viewers:

```bash
ssh -L <port>:localhost:<port> spark
```

Then open `http://localhost:<port>` in your local browser.
