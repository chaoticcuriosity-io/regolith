#!/usr/bin/env bash
# reproduce.sh — skeleton for the end-to-end pipeline reproduce sequence.
#
# Each step runs inside the regolith Docker container on the DGX Spark.
# Container name: set CONTAINER below once Task 1 establishes it.
# Fill in actual commands as each task is completed (see setup-notes.md).
#
# Usage: bash scripts/reproduce.sh
# (Or run steps individually — they're designed to be idempotent.)

set -euo pipefail

SPARK="spark"
CONTAINER="<fill-in-Task-1>"   # e.g. regolith-dev
WORKSPACE="/workspace/regolith"

run() {
    # Helper: run a command in the container on the Spark
    ssh "${SPARK}" "docker exec ${CONTAINER} bash -lc 'cd ${WORKSPACE} && $*'"
}

echo "=== regolith: end-to-end reproduce ==="
echo ""

# --- Step 0: Free memory on the Spark ---
echo "--- Step 0: Free memory ---"
ssh "${SPARK}" "bash ${WORKSPACE}/scripts/free_memory.sh"

# --- Step 1: Build the lunar USD stage ---
# TODO (Task 1): uncomment and fill in once Isaac + Omniverse are set up
# echo "--- Step 1: Build lunar stage ---"
# run "python scene/build_lunar_stage.py"

# --- Step 2: Generate training dataset (domain randomization) ---
# TODO (Task 2): uncomment once Replicator is installed
# echo "--- Step 2: Generate training dataset ---"
# run "python replicator/generate_dataset.py \
#       --config replicator/configs/train_dr.yaml \
#       --n 10000 \
#       --out /workspace/datasets/train_dr"
#
# run "python replicator/generate_dataset.py \
#       --config replicator/configs/test_photoreal.yaml \
#       --n 1000 \
#       --out /workspace/datasets/test_photoreal"

# --- Step 3: Train SegFormer ---
# TODO (Task 3): uncomment once dataset is ready
# echo "--- Step 3: Train ---"
# run "python training/train.py \
#       --train-split /workspace/datasets/train_dr \
#       --val-split   /workspace/datasets/val_dr \
#       --epochs 50 \
#       --out /workspace/checkpoints/segformer_b0_dr"

# --- Step 4a: Evaluate on synthetic hold-out ---
# TODO (Task 4): uncomment once model is trained
# echo "--- Step 4a: Eval (synthetic) ---"
# run "python eval/eval_synth.py \
#       --checkpoint /workspace/checkpoints/segformer_b0_dr/best.pt \
#       --test-split /workspace/datasets/test_photoreal \
#       --out /workspace/eval_results/synth"

# --- Step 4b: Evaluate on real imagery ---
# TODO (Task 4): uncomment once real data is downloaded
# echo "--- Step 4b: Eval (real) ---"
# run "python eval/data/download_real.py --out /workspace/datasets/real_lunar"
# run "python eval/eval_real.py \
#       --checkpoint /workspace/checkpoints/segformer_b0_dr/best.pt \
#       --real-data /workspace/datasets/real_lunar \
#       --out /workspace/eval_results/real"

# --- Step 5: Cinematic render ---
# TODO (Task 5): uncomment once eval is done
# echo "--- Step 5: Cinematic render ---"
# run "python render/render_predictions.py \
#       --checkpoint /workspace/checkpoints/segformer_b0_dr/best.pt \
#       --out /workspace/renders/v1 \
#       --fps 30 \
#       --duration-s 30"

echo ""
echo "=== Reproduce script: all steps either completed or skipped (see TODOs above) ==="
