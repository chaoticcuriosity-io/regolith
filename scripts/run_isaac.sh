#!/usr/bin/env bash
# run_isaac.sh — reusable Isaac Sim + Replicator launcher for the DGX Spark.
#
# Usage (run on the Spark, or via ssh):
#   bash scripts/run_isaac.sh <host-script-path> <host-output-dir>
#
#   <host-script-path>   absolute path on the Spark host to the Python script to run;
#                        mounted read-only inside the container and invoked with
#                        /isaac-sim/python.sh.
#   <host-output-dir>    absolute path on the Spark host for Replicator output;
#                        created automatically, chmod 777'd (required — see gotchas),
#                        and mounted at /workspace/out inside the container.
#
# Example:
#   ssh spark "bash /home/chaotic-curiosity/regolith/scripts/run_isaac.sh \
#     /home/chaotic-curiosity/regolith/replicator/_smoke_render.py \
#     /home/chaotic-curiosity/regolith_out"
#
# GOTCHAS — read before modifying this script:
#   (full context: setup-notes.md ## Session 2 — 2026-06-26)
#
#   1. DO NOT add -v /isaac-sim/.cache:... — container runs as uid 1234 (user
#      "isaac-sim"); a host-owned bind-mount at that path triggers PermissionError
#      in wp.init() (NVIDIA Warp), aborting extension startup and cascading into
#      misleading errors ("No writer 'BasicWriter'", orchestrator NoneType).
#      For shader-cache persistence (kills the ~150 s first-frame penalty), mount a
#      SEPARATE host dir that is chmod 777 and set WARP_CACHE_PATH / HOME inside
#      the container — not /isaac-sim/.cache directly.
#
#   2. The Python script MUST use poll-for-files drain, not bare
#      wait_until_complete(). First RTX frame on the Spark is ~150 s;
#      wait_until_complete() times out before the frame lands → zero output files.
#      See replicator/_smoke_render.py for the reference implementation.
#
#   3. chmod 777 the host output dir (done below) — container uid 1234 must be able
#      to write there; a host-owned dir with default permissions causes silent failure.
#
# Image: nvcr.io/nvidia/isaac-sim:6.0.0 (pinned; multi-arch; pulls linux/arm64
#        on aarch64 automatically; public — no NGC login required; 17.6 GB on disk).
#        Do NOT change the pinned tag without re-running the smoke test.

set -euo pipefail

IMAGE="nvcr.io/nvidia/isaac-sim:6.0.0"

SCRIPT_HOST="${1:?Usage: bash scripts/run_isaac.sh <host-script-path> <host-output-dir>}"
OUT_HOST="${2:?Usage: bash scripts/run_isaac.sh <host-script-path> <host-output-dir>}"

SCRIPT_NAME="$(basename "${SCRIPT_HOST}")"
SCRIPT_CONTAINER="/workspace/${SCRIPT_NAME}"
OUT_CONTAINER="/workspace/out"

# Output dir must exist and be world-writable — container uid 1234 (isaac-sim) writes there.
mkdir -p "${OUT_HOST}"
chmod 777 "${OUT_HOST}"

docker run \
  --name "isaac-sim-run-$$" \
  --entrypoint bash \
  --gpus all \
  --network=host \
  --rm \
  -e "ACCEPT_EULA=Y" \
  -e "PRIVACY_CONSENT=Y" \
  -e "SMOKE_OUT=${OUT_CONTAINER}" \
  -v "${SCRIPT_HOST}:${SCRIPT_CONTAINER}:ro" \
  -v "${OUT_HOST}:${OUT_CONTAINER}:rw" \
  "${IMAGE}" \
  -lc "/isaac-sim/python.sh ${SCRIPT_CONTAINER}"
