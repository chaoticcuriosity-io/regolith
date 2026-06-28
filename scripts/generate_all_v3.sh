#!/usr/bin/env bash
# generate_all_v3.sh — full v3 PREMIUM regolith dataset generation, the DURABLE way.
#
# Same leak-safe discipline as scripts/generate_all.sh (one FRESH Isaac container per
# split + a GPU-idle reclaim gap between splits, to recover from the per-frame RTX
# resource-descriptor leak — see setup-notes.md ## Session 5), but driving the v3 stage:
#   replicator/generate_dataset_v3.py + replicator/configs_v3/*  (cratered ground +
#   per-mesh rocks, SURFACE-ONLY / no rover) -> regolith_data_v3/.
#
# Usage (detached, on the Spark):
#   nohup bash /home/chaotic-curiosity/regolith/scripts/generate_all_v3.sh \
#       > /home/chaotic-curiosity/regolith_data_v3/master.log 2>&1 &
#
# Env overrides:
#   DATA_DIR    (default /home/chaotic-curiosity/regolith_data_v3)
#   REPO_DIR    (default /home/chaotic-curiosity/regolith)
#   CACHE_DIR   (default /home/chaotic-curiosity/regolith_cache, chmod 777)
#   RECLAIM_GAP (default 150 — seconds of GPU idle between splits)
#   N_TRAIN_DR / N_TRAIN_NODR / N_TEST   (per-split frame counts; defaults below)
#
# Image: nvcr.io/nvidia/isaac-sim:6.0.0 (pinned). rt_subframes come from each config.

set -euo pipefail

IMAGE="nvcr.io/nvidia/isaac-sim:6.0.0"
DATA_DIR="${DATA_DIR:-/home/chaotic-curiosity/regolith_data_v3}"
REPO_DIR="${REPO_DIR:-/home/chaotic-curiosity/regolith}"
CACHE_DIR="${CACHE_DIR:-/home/chaotic-curiosity/regolith_cache}"
RECLAIM_GAP="${RECLAIM_GAP:-150}"
CONTAINER="isaac-v3"

# Final per-split counts. Measured v3 warm s/frame is ~5s (train_dr, subframes=4) —
# the per-mesh-rock + cratered-ground scene is NOT slower than v2 in practice (USD
# authoring is cheap; the proto geometry pool is reused). So the full 1500/750/300
# target fits the overnight budget with wide margin (~4-5 h total incl. cold compiles
# + reclaim gaps). Override via env at launch.
N_TRAIN_DR="${N_TRAIN_DR:-1500}"
N_TRAIN_NODR="${N_TRAIN_NODR:-750}"
N_TEST="${N_TEST:-300}"

NORMAL_MAP="${REPO_DIR}/assets/regolith_normal.png"   # mounted at /workspace/regolith/...

mkdir -p "${DATA_DIR}" "${CACHE_DIR}"
chmod 777 "${DATA_DIR}" "${CACHE_DIR}" || true

# split | config | n | seed
SPLITS=(
  "train_dr|replicator/configs_v3/train_dr.yaml|${N_TRAIN_DR}|42"
  "train_nodr|replicator/configs_v3/train_nodr.yaml|${N_TRAIN_NODR}|42"
  "test_photoreal|replicator/configs_v3/test_photoreal.yaml|${N_TEST}|7777"
)

gpu_util() { nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1; }

reclaim_gap() {
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
  local log="${DATA_DIR}/gen_${name}.log"

  echo ">>> [$(date -u)] split=${name} n=${n} seed=${seed}"
  docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true

  # SURFACE-ONLY: V3_ROVER_USD intentionally UNSET (rover off). V3_REGOLITH_NORMAL
  # points at the in-container repo copy so the realistic regolith relief is present.
  docker run -d --name "${CONTAINER}" --entrypoint bash --gpus all --network=host \
    -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
    -e V3_REGOLITH_NORMAL=/workspace/regolith/assets/regolith_normal.png \
    -v "${REPO_DIR}:/workspace/regolith:rw" \
    -v "${DATA_DIR}:/workspace/data:rw" \
    -v "${CACHE_DIR}:/isaac-sim/.cache:rw" \
    "${IMAGE}" -lc 'sleep infinity' >/dev/null

  docker exec "${CONTAINER}" bash -lc "
    cd /workspace/regolith
    unset V3_ROVER_USD
    echo '=== ${name} $(date -u) ===' > /workspace/data/gen_${name}.log
    /isaac-sim/python.sh replicator/generate_dataset_v3.py \
      --config ${config} --n ${n} --out /workspace/data/${name} \
      --seed ${seed} --res 512 >> /workspace/data/gen_${name}.log 2>&1
    echo '=== DONE $(date -u) ===' >> /workspace/data/gen_${name}.log
  "

  local done
  done="$(tail -3 "${log}" 2>/dev/null | grep -oE 'completed=[0-9]+' | tail -1 || true)"
  echo ">>> split=${name} finished (${done:-completed=?}); $(grep -c 'Out of resource descriptors' "${log}" 2>/dev/null || echo 0) descriptor-fatal lines"

  docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true
}

i=0
for entry in "${SPLITS[@]}"; do
  IFS='|' read -r name config n seed <<<"${entry}"
  [ "${i}" -gt 0 ] && reclaim_gap
  run_split "${name}" "${config}" "${n}" "${seed}"
  i=$((i + 1))
done

echo ">>> ALL V3 SPLITS DONE."
echo ">>> counts:"
for entry in "${SPLITS[@]}"; do
  IFS='|' read -r name _ _ _ <<<"${entry}"
  echo "    ${name}: rgb=$(ls "${DATA_DIR}/${name}/rgb" 2>/dev/null | wc -l) mask=$(ls "${DATA_DIR}/${name}/mask" 2>/dev/null | wc -l)"
done
