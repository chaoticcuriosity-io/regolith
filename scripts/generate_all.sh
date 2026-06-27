#!/usr/bin/env bash
# generate_all.sh — full regolith dataset generation, the DURABLE way.
#
# Runs each split in its OWN fresh Isaac container with a GPU-idle reclaim gap
# between splits. This is the fix for the RTX resource-descriptor leak that
# stalled the Session-4 "all splits in one container" run.
#
# WHY (full root cause: setup-notes.md ## Session 5 — 2026-06-27):
#   The SDG render loop leaks RTX resource descriptors per frame, and
#   SimulationApp.close() segfaults without releasing them. The leaked pool is
#   HOST/driver-level — it survives both process exit AND `docker rm`. It is
#   reclaimed only LAZILY, after the leaking container is gone and the GPU sits
#   idle a few minutes. A single fresh process from a *reclaimed* pool sustains
#   >=1500 frames (train_dr proved it), but running splits back-to-back in one
#   container drains the shared pool (train_dr's 1500 -> train_nodr died at 136).
#   A GPU reset is impossible (the GB10 is the primary/display GPU), so the
#   reclaim gap below is the recovery mechanism — do NOT remove it.
#
# RULES baked in:
#   - one fresh container per split (no shared GPU descriptor pool);
#   - a RECLAIM_GAP of GPU-idle seconds between splits;
#   - keep any single split well under ~1500 frames; chunk bigger splits into
#     <=~750-frame pieces with --start-index (seeds continue deterministically).
#
# Usage (on the Spark, or via ssh):
#   ssh spark "bash /home/chaotic-curiosity/regolith/scripts/generate_all.sh"
#
# Env overrides:
#   DATA_DIR    (default /home/chaotic-curiosity/regolith_data)
#   REPO_DIR    (default /home/chaotic-curiosity/regolith)
#   CACHE_DIR   (default /home/chaotic-curiosity/regolith_cache, chmod 777)
#   RECLAIM_GAP (default 150 — seconds of GPU idle to wait between splits)
#
# Image: nvcr.io/nvidia/isaac-sim:6.0.0 (pinned; see CLAUDE.md / setup-notes.md).

set -euo pipefail

IMAGE="nvcr.io/nvidia/isaac-sim:6.0.0"
DATA_DIR="${DATA_DIR:-/home/chaotic-curiosity/regolith_data}"
REPO_DIR="${REPO_DIR:-/home/chaotic-curiosity/regolith}"
CACHE_DIR="${CACHE_DIR:-/home/chaotic-curiosity/regolith_cache}"
RECLAIM_GAP="${RECLAIM_GAP:-150}"
CONTAINER="isaac-dev"

mkdir -p "${DATA_DIR}" "${CACHE_DIR}"
chmod 777 "${DATA_DIR}" "${CACHE_DIR}" || true

# split | config | n | seed
SPLITS=(
  "train_dr|replicator/configs/train_dr.yaml|1500|42"
  "train_nodr|replicator/configs/train_nodr.yaml|750|42"
  "test_photoreal|replicator/configs/test_photoreal.yaml|300|7777"
)

gpu_util() { nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1; }

reclaim_gap() {
  # Wait for the GPU to go (and stay) idle so the driver reclaims leaked descriptors.
  echo ">>> reclaim gap: waiting ${RECLAIM_GAP}s of GPU idle before next split"
  local waited=0
  while [ "${waited}" -lt "${RECLAIM_GAP}" ]; do
    echo "    [$(date +%H:%M:%S)] gpu_util=$(gpu_util)% (waited ${waited}s)"
    sleep 20
    waited=$((waited + 20))
  done
}

run_split() {
  local name="$1" config="$2" n="$3" seed="$4"
  local out="${DATA_DIR}/${name}"
  local log="${DATA_DIR}/gen_${name}.log"

  echo ">>> [$(date -u)] split=${name} n=${n} seed=${seed}"
  docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true

  docker run -d --name "${CONTAINER}" --entrypoint bash --gpus all --network=host \
    -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
    -v "${REPO_DIR}:/workspace/regolith:rw" \
    -v "${DATA_DIR}:/workspace/data:rw" \
    -v "${CACHE_DIR}:/isaac-sim/.cache:rw" \
    "${IMAGE}" -lc 'sleep infinity' >/dev/null

  # Run in the foreground of the exec so this script blocks until the split finishes.
  docker exec "${CONTAINER}" bash -lc "
    cd /workspace/regolith
    echo '=== ${name} $(date -u) ===' > /workspace/data/gen_${name}.log
    /isaac-sim/python.sh replicator/generate_dataset.py \
      --config ${config} --n ${n} --out /workspace/data/${name} \
      --seed ${seed} --res 512 >> /workspace/data/gen_${name}.log 2>&1
    echo '=== DONE $(date -u) ===' >> /workspace/data/gen_${name}.log
  "

  local done
  done="$(tail -3 "${log}" 2>/dev/null | grep -oE 'completed=[0-9]+' | tail -1 || true)"
  echo ">>> split=${name} finished (${done:-completed=?}); $(grep -c 'Out of resource descriptors' "${log}" 2>/dev/null || echo 0) descriptor-fatal lines"

  # Tear down this split's container so its leaked descriptors can be reclaimed.
  docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true
}

i=0
for entry in "${SPLITS[@]}"; do
  IFS='|' read -r name config n seed <<<"${entry}"
  [ "${i}" -gt 0 ] && reclaim_gap
  run_split "${name}" "${config}" "${n}" "${seed}"
  i=$((i + 1))
done

echo ">>> ALL SPLITS DONE."
echo ">>> counts:"
for entry in "${SPLITS[@]}"; do
  IFS='|' read -r name _ _ _ <<<"${entry}"
  echo "    ${name}: rgb=$(ls "${DATA_DIR}/${name}/rgb" 2>/dev/null | wc -l) mask=$(ls "${DATA_DIR}/${name}/mask" 2>/dev/null | wc -l)"
done
