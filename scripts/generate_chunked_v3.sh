#!/usr/bin/env bash
# generate_chunked_v3.sh — leak-RESILIENT chunked v3 dataset generation.
#
# Why this exists (see setup-notes.md "Session: v3 descriptor leak hit within-split"):
# the v3 PREMIUM scene authors ~1400-1700 INDIVIDUAL per-mesh `rock` prims per frame
# (the fix that makes pebbles segment into class 1). Those per-mesh prims churn RTX
# resource descriptors ~1.6x faster than v2, so the per-PROCESS descriptor budget is
# only ~954 frames — BELOW a single 1500-frame split. generate_all_v3.sh used one
# fresh container PER SPLIT; that still hit `[Fatal] [omni.rtx] Out of resource
# descriptors!` mid-train_dr (frame ~954) and HUNG (python spinning, never exits).
#
# Fix: CHUNK every split into pieces well under the ~954 budget (default 500 frames),
# each in a FRESH Isaac container (resets the descriptor pool), with a GPU-idle reclaim
# gap between chunks (the leak only reclaims after the container is gone AND the GPU
# goes idle — see setup-notes Session 5). A per-chunk WATCHDOG kills+retries on the
# descriptor fatal, a short process exit, or a stall (no new frame for >5 min) — so one
# hang can't freeze the whole run for hours.
#
# RESUME is index-exact and dup-free: generate_dataset_v3.py keys each frame's seed to
# its ABSOLUTE index (seed_i = _frame_seed(base_seed, start_index + k)) and numbers
# files rgb_%05d/mask_%05d by that index, and only wipes frames.jsonl when
# start_index==0. So a chunk is just `--start-index <#already-done> --n <chunk>`. This
# orchestrator RECOUNTS the mask/ dir before EVERY chunk, so retries resume exactly
# where the data ends — no gaps, no dups, even after a kill mid-chunk.
#
# Usage (detached, on the Spark):
#   nohup setsid bash /home/chaotic-curiosity/regolith/scripts/generate_chunked_v3.sh \
#       > /home/chaotic-curiosity/regolith_data_v3/master_chunked.log 2>&1 < /dev/null &
#
# Env overrides:
#   DATA_DIR REPO_DIR CACHE_DIR
#   CHUNK         (frames per chunk; default 500 — must stay < ~954 leak budget)
#   RECLAIM_GAP   (seconds of GPU-idle between chunks; default 240)
#   STALL_TIMEOUT (no-new-frame watchdog seconds; default 300)
#   POLL          (watchdog poll interval seconds; default 30)
#   MAX_STUCK     (consecutive no-progress attempts before aborting a split; default 5)
#   N_TRAIN_DR N_TRAIN_NODR N_TEST  (per-split targets)
#
# Image: nvcr.io/nvidia/isaac-sim:6.0.0 (pinned). rt_subframes come from each config.

set -uo pipefail   # NOT -e: chunk failures are handled (kill+retry), not fatal to the run.

IMAGE="nvcr.io/nvidia/isaac-sim:6.0.0"
DATA_DIR="${DATA_DIR:-/home/chaotic-curiosity/regolith_data_v3}"
REPO_DIR="${REPO_DIR:-/home/chaotic-curiosity/regolith}"
CACHE_DIR="${CACHE_DIR:-/home/chaotic-curiosity/regolith_cache}"
CHUNK="${CHUNK:-500}"
RECLAIM_GAP="${RECLAIM_GAP:-240}"
STALL_TIMEOUT="${STALL_TIMEOUT:-300}"
POLL="${POLL:-30}"
MAX_STUCK="${MAX_STUCK:-5}"
CONTAINER="isaac-v3-chunk"
CHUNK_DIR="${DATA_DIR}/chunks"

N_TRAIN_DR="${N_TRAIN_DR:-1500}"
N_TRAIN_NODR="${N_TRAIN_NODR:-750}"
N_TEST="${N_TEST:-300}"

mkdir -p "${DATA_DIR}" "${CACHE_DIR}" "${CHUNK_DIR}"
# CHUNK_DIR MUST be world-writable: the per-chunk log is created by the generator's
# stdout redirect INSIDE the Isaac container (UID 1234), but this dir is made here as
# the host user (UID 1001). Without 777, the redirect fails (Permission denied) and the
# generator never launches — the data split dirs are fine (the container creates them).
chmod 777 "${DATA_DIR}" "${CACHE_DIR}" "${CHUNK_DIR}" 2>/dev/null || true

trap 'echo ">>> [trap] removing container ${CONTAINER}"; docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true' EXIT

# split | config | total | seed
SPLITS=(
  "train_dr|replicator/configs_v3/train_dr.yaml|${N_TRAIN_DR}|42"
  "train_nodr|replicator/configs_v3/train_nodr.yaml|${N_TRAIN_NODR}|42"
  "test_photoreal|replicator/configs_v3/test_photoreal.yaml|${N_TEST}|7777"
)

ts() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }
gpu_util() { nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1; }

# Authoritative per-split progress = count of finished mask files (written last per frame).
mask_count() {
  local c
  c="$(ls "${DATA_DIR}/$1/mask"/mask_*.png 2>/dev/null | wc -l)"
  echo "${c//[^0-9]/}"
}

reclaim_gap() {
  echo ">>> [$(ts)] reclaim gap: ${RECLAIM_GAP}s GPU-idle before next chunk"
  local waited=0
  while [ "${waited}" -lt "${RECLAIM_GAP}" ]; do
    echo "    [$(date +%H:%M:%S)] gpu_util=$(gpu_util)% (waited ${waited}s)"
    sleep 20
    waited=$((waited + 20))
  done
}

# Run ONE chunk in a fresh container, watchdogged.
# Echoes nothing; returns 0 = target reached, 1 = needs-retry (fatal / short-exit / stall).
run_chunk() {
  local name="$1" config="$2" seed="$3" start="$4" this_n="$5"
  local target=$((start + this_n))
  local pad; pad="$(printf '%05d' "${start}")"
  local clog="${CHUNK_DIR}/gen_${name}_${pad}.log"
  local clog_in="/workspace/data/chunks/gen_${name}_${pad}.log"

  echo ">>> [$(ts)] CHUNK split=${name} start=${start} n=${this_n} -> target=${target} (seed=${seed}) log=${clog}"
  docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true
  docker run -d --name "${CONTAINER}" --entrypoint bash --gpus all --network=host \
    -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
    -e V3_REGOLITH_NORMAL=/workspace/regolith/assets/regolith_normal.png \
    -v "${REPO_DIR}:/workspace/regolith:rw" \
    -v "${DATA_DIR}:/workspace/data:rw" \
    -v "${CACHE_DIR}:/isaac-sim/.cache:rw" \
    "${IMAGE}" -lc 'sleep infinity' >/dev/null

  # Launch the generator DETACHED inside the container; sentinel "EXIT=<code>" on finish.
  docker exec -d "${CONTAINER}" bash -lc "
    cd /workspace/regolith
    unset V3_ROVER_USD
    echo '=== ${name} chunk start=${start} n=${this_n} '\"\$(date -u)\"' ===' > '${clog_in}'
    /isaac-sim/python.sh replicator/generate_dataset_v3.py \
      --config ${config} --n ${this_n} --out /workspace/data/${name} \
      --seed ${seed} --res 512 --start-index ${start} >> '${clog_in}' 2>&1
    echo \"=== EXIT=\$? \$(date -u) ===\" >> '${clog_in}'
  "

  local last_count cur now last_progress
  last_count="$(mask_count "${name}")"
  last_progress="$(date +%s)"

  while true; do
    sleep "${POLL}"
    cur="$(mask_count "${name}")"
    now="$(date +%s)"
    if [ "${cur}" -gt "${last_count}" ]; then
      echo "    [$(date +%H:%M:%S)] ${name} mask=${cur}/${target} (+$((cur - last_count)))"
      last_count="${cur}"
      last_progress="${now}"
    fi
    # success (checked first so an end-of-chunk fatal can't mask a completed chunk)
    if [ "${cur}" -ge "${target}" ]; then
      echo ">>> [$(ts)] CHUNK OK split=${name} reached ${cur}/${target}"
      docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true
      return 0
    fi
    # descriptor fatal -> kill + retry
    if grep -q 'Out of resource descriptors' "${clog}" 2>/dev/null; then
      echo ">>> [$(ts)] CHUNK FATAL (descriptor leak) split=${name} mask=${cur}/${target} -> kill+retry"
      docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true
      return 1
    fi
    # process exited without reaching target -> retry
    if grep -q 'EXIT=' "${clog}" 2>/dev/null; then
      echo ">>> [$(ts)] CHUNK ENDED-SHORT split=${name} mask=${cur}/${target} (process exited) -> retry"
      docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true
      return 1
    fi
    # stall watchdog
    if [ $((now - last_progress)) -ge "${STALL_TIMEOUT}" ]; then
      echo ">>> [$(ts)] CHUNK STALL split=${name} no new frame for $((now - last_progress))s at mask=${cur}/${target} -> kill+retry"
      docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true
      return 1
    fi
  done
}

echo "============================================================"
echo ">>> v3 CHUNKED generation start $(ts)"
echo ">>> chunk=${CHUNK} reclaim_gap=${RECLAIM_GAP}s stall_timeout=${STALL_TIMEOUT}s poll=${POLL}s"
echo ">>> targets: train_dr=${N_TRAIN_DR} train_nodr=${N_TRAIN_NODR} test_photoreal=${N_TEST}"
echo ">>> existing: train_dr=$(mask_count train_dr) train_nodr=$(mask_count train_nodr) test_photoreal=$(mask_count test_photoreal)"
echo "============================================================"

first_chunk=1
for entry in "${SPLITS[@]}"; do
  IFS='|' read -r name config total seed <<<"${entry}"
  echo ">>> ===== SPLIT ${name} target=${total} seed=${seed} ====="
  stuck=0
  while true; do
    done="$(mask_count "${name}")"
    if [ "${done}" -ge "${total}" ]; then
      echo ">>> SPLIT ${name} COMPLETE: ${done}/${total}"
      break
    fi
    remaining=$((total - done))
    this_n=$(( remaining < CHUNK ? remaining : CHUNK ))
    start="${done}"

    # GPU-idle reclaim before every chunk except the very first of the whole run.
    if [ "${first_chunk}" -eq 0 ]; then reclaim_gap; fi
    first_chunk=0

    if run_chunk "${name}" "${config}" "${seed}" "${start}" "${this_n}"; then
      stuck=0
    else
      new_done="$(mask_count "${name}")"
      if [ "${new_done}" -le "${done}" ]; then
        stuck=$((stuck + 1))
        echo ">>> SPLIT ${name} NO progress this attempt (still ${new_done}/${total}); stuck=${stuck}/${MAX_STUCK}"
        if [ "${stuck}" -ge "${MAX_STUCK}" ]; then
          echo ">>> [ABORT] SPLIT ${name} stuck at ${new_done}/${total} after ${MAX_STUCK} attempts."
          exit 2
        fi
      else
        stuck=0
        echo ">>> SPLIT ${name} partial progress ${done}->${new_done}/${total}; continuing from data end"
      fi
    fi
  done
done

echo "============================================================"
echo ">>> ALL V3 SPLITS DONE $(ts)"
for entry in "${SPLITS[@]}"; do
  IFS='|' read -r name _ total _ <<<"${entry}"
  echo "    ${name}: rgb=$(ls "${DATA_DIR}/${name}/rgb"/*.png 2>/dev/null | wc -l) mask=$(ls "${DATA_DIR}/${name}/mask"/*.png 2>/dev/null | wc -l) (target ${total})"
done
echo "============================================================"
