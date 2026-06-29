#!/usr/bin/env bash
# run_pipeline_v3.sh — detached v3 retrain + eval pipeline (train -> eval_synth -> eval_real -> aggregate)
# Mirrors run_ablation_v2.sh hyperparameters exactly, pointed at datasets_v3 -> outputs/runs_v3 + outputs/eval_v3.
# Runs INSIDE the regolith-train-v3 container (paths are container paths).
set -u
cd /workspace/regolith
# mit-b0 backbone weights are pre-cached in this image (committed from the proven
# v2 container); force offline so from_pretrained never hangs on a network call.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p outputs/runs_v3 outputs/eval_v3/eval_synth outputs/eval_v3/eval_real
echo $$ > outputs/runs_v3/pipeline.pid
rm -f outputs/runs_v3/PIPELINE_DONE
echo "[pipeline_v3] START $(date -u)  pid=$$"

DATA=/workspace/datasets_v3
RUNS=outputs/runs_v3
ESYN=outputs/eval_v3/eval_synth
EREAL=outputs/eval_v3/eval_real
COMMON="--val-split $DATA/test_photoreal --epochs 40 --patience 8 --model segformer_b0 --lr 6e-5 --batch 8 --seed 0"

# ----------------------------- TRAIN (3 runs) -----------------------------
echo "[pipeline_v3] ===== TRAIN A: nodr_750 (train_nodr, 750, no DR) ====="
python training/train.py --train-split $DATA/train_nodr   $COMMON --out $RUNS/nodr_750 2>&1 | tee $RUNS/nodr_750.log
echo "[pipeline_v3] A done rc=${PIPESTATUS[0]} $(date -u)"

echo "[pipeline_v3] ===== TRAIN B: dr_750 (first 750 of train_dr, DR) ====="
python training/train.py --train-split $DATA/train_dr_750 $COMMON --out $RUNS/dr_750 2>&1 | tee $RUNS/dr_750.log
echo "[pipeline_v3] B done rc=${PIPESTATUS[0]} $(date -u)"

echo "[pipeline_v3] ===== TRAIN C: dr_1500 (full train_dr, 1500, DR) ====="
python training/train.py --train-split $DATA/train_dr     $COMMON --out $RUNS/dr_1500 2>&1 | tee $RUNS/dr_1500.log
echo "[pipeline_v3] C done rc=${PIPESTATUS[0]} $(date -u)"

# ------------------- EVAL SYNTH (test_photoreal, 300) ---------------------
for run in nodr_750 dr_750 dr_1500; do
  echo "[pipeline_v3] ===== EVAL_SYNTH: $run ====="
  python eval/eval_synth.py --checkpoint $RUNS/$run/best.pt \
      --test-split $DATA/test_photoreal \
      --out $ESYN/$run --overlay-every 50 2>&1 | tee $ESYN/$run.log
  echo "[pipeline_v3] eval_synth $run rc=${PIPESTATUS[0]} $(date -u)"
done

# ----------------------- EVAL REAL (21 real photos) -----------------------
for run in nodr_750 dr_750 dr_1500; do
  echo "[pipeline_v3] ===== EVAL_REAL: $run ====="
  python eval/eval_real.py --checkpoint $RUNS/$run/best.pt \
      --image-dir eval/real_images_v3 \
      --label $run \
      --out $EREAL/$run 2>&1 | tee $EREAL/$run.log
  echo "[pipeline_v3] eval_real $run rc=${PIPESTATUS[0]} $(date -u)"
done

# Money panel: dr_1500 (primary, right) vs nodr_750 (compare, middle)
echo "[pipeline_v3] ===== EVAL_REAL money panel: dr_1500 vs nodr_750 ====="
python eval/eval_real.py --checkpoint $RUNS/dr_1500/best.pt \
    --compare-checkpoint $RUNS/nodr_750/best.pt \
    --label dr_1500 --compare-label nodr_750 \
    --image-dir eval/real_images_v3 \
    --out $EREAL/compare_dr1500_vs_nodr750 2>&1 | tee $EREAL/compare.log
echo "[pipeline_v3] money panel rc=${PIPESTATUS[0]} $(date -u)"

# ----------------------------- AGGREGATE ----------------------------------
echo "[pipeline_v3] ===== AGGREGATE RESULTS ====="
python make_results_v3.py 2>&1 | tee $RUNS/results_agg.log

echo "[pipeline_v3] ALL DONE $(date -u)"
touch outputs/runs_v3/PIPELINE_DONE
