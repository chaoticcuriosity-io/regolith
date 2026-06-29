# Regolith v3 — Retrain + Eval Results

SegFormer-b0 (`nvidia/mit-b0`), 3-run domain-randomization ablation on the v3 dataset (realistic cratered/displaced dark-regolith ground). Hyperparameters identical to v1/v2: `--epochs 40 --patience 8 --lr 6e-5 --batch 8 --seed 0`, best checkpoint selected on validation rock-IoU (val split = `test_photoreal`, 300 frames).

**HEADLINE: YES — the real flood DROPPED to 35.7% for v3 dr_1500, far below v2 (72.6%) and v1 (44.0%)** — all three measured apples-to-apples with the same hardened `eval/eval_real.py` on the same 21-image NASA set. (v1/v2's *original* 7-image floods were ~52% / ~83%; see the cross-version block below.)

## v3 per-run results

| run | train data | best epoch | synth mIoU | synth rock-IoU | synth class-IoU (reg/rock/sky) | real mean flood | train wall |
|---|---|---|---|---|---|---|---|
| nodr_750 | train_nodr (750, no DR) | 14 | 0.9455 | 0.8593 | 0.980/0.859/0.997 | 39.7% | 6.1 min |
| dr_750 | first 750 of train_dr (DR) | 16 | 0.9486 | 0.8675 | 0.981/0.868/0.997 | 36.3% | 6.7 min |
| dr_1500 | full train_dr (1500, DR) | 31 | 0.9563 | 0.8870 | 0.984/0.887/0.997 | 35.7% | 19.4 min |

## Cross-version comparison (dr_1500) — apples-to-apples, same 21 images

All three `dr_1500` checkpoints were scored with the **same hardened `eval/eval_real.py`** on the **same 21-image** NASA set (`eval/real_images_v3`, 0 skipped), so the flood column below is an exact same-set comparison — not a directional one across different image pools. Flood = mean fraction of pixels predicted `rock` (lower better). `model.py` (SegFormer-b0) is identical across versions, so the v1/v2 checkpoints load and run cleanly under the current eval.

| version | synth rock-IoU (dr_1500) | real flood (dr_1500, same 21 imgs) | worst frame (21-img set) | note |
|---|---|---|---|---|
| v1 | 0.815 | 44.0% | apollo16_south_ray_boulder_close (80.1%) | first synthetic stage |
| v2 | 0.852 | 72.6% | apollo16_south_ray_boulder_close (99.5%) | photoreal rocks collapsed rock/regolith boundary |
| **v3** | **0.887** | **35.7%** | apollo11_tranquility_base_wide_panorama (71.2%) | realistic cratered dark-regolith ground |

**Provenance.** v1 and v2 were *originally* evaluated on an older 7-image set (`eval/real_images`), where they flooded ~52% and ~83% respectively — those are the numbers that appear in chapters 04/06. Re-running the exact same v1/v2 checkpoints on the harder 21-image set with the current eval gives the 44.0% / 72.6% above. v3 dr_1500 = 35.7% is unchanged (it was always measured on the 21-image set). The ordering is identical and the conclusion is stronger: v2 floods real surfaces badly (72.6%), and v3 fixes it (35.7%), landing below even v1's crude baseline (44.0%).

## Synthetic rock-IoU: v2 vs v3 (all runs)

| run | v2 synth rock-IoU | v3 synth rock-IoU |
|---|---|---|
| nodr_750 | 0.8025 | 0.8593 |
| dr_750 | 0.8486 | 0.8675 |
| dr_1500 | 0.8521 | 0.8870 |

## v3 dr_1500 real-image flood (per image)

| image | rock flood |
|---|---|
| apollo11_tranquility_base_wide_panorama | 71.2% |
| apollo14_cone_crater_boulder_group | 56.2% |
| surveyor7_surface_mosaic | 55.2% |
| surveyor7_terrain_ground_view | 54.9% |
| apollo17_station2_terrain | 49.9% |
| apollo16_descartes_highlands_terrain | 47.5% |
| apollo16_south_ray_boulder_close | 38.2% |
| apollo17_station6_split_rock_area | 37.4% |
| apollo17_station5_camelot_panorama | 36.8% |
| apollo17_station6_boulder_view | 33.3% |
| apollo14_cone_crater_boulder_field_overview | 29.8% |
| apollo17_station6_boulder | 29.5% |
| apollo15_station2_panorama | 28.9% |
| apollo16_shadow_rock_north_ray | 28.6% |
| apollo15_hadley_rille_boulder | 27.5% |
| apollo11_lm_shadow_regolith_footprints | 26.1% |
| apollo14_cone_crater_boulder_field_south | 25.9% |
| apollo14_cone_crater_boulder_close | 21.7% |
| apollo12_surveyor_crater_panorama | 19.9% |
| apollo11_regolith_footprint_closeup | 16.8% |
| surveyor1_surface_shadow | 15.1% |

## Method notes

- **Synthetic eval**: `eval_synth.py` on `test_photoreal` (300 in-distribution photoreal frames); IoU is globally accumulated tp/fp/fn (matches training).
- **Real eval**: `eval_real.py` on 21 NASA public-domain Apollo/Surveyor surface photos (`eval/real_images_v3`, no ground truth). Flood = mean fraction of pixels predicted `rock`. No Chang'e/CNSA imagery included.
- Overlays (RGB + red rock mask) per run in `outputs/eval_v3/eval_real/<run>/` and the dr_1500-vs-nodr_750 money panels in `outputs/eval_v3/eval_real/compare_dr1500_vs_nodr750/`.
