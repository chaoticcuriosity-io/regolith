# Reports — curriculum TOC

Read in order 00 → 06. Each chapter is self-contained but assumes the previous one. The series ends on a surprising sim-to-real result, not a clean win — read for the lesson.

| # | Chapter | What you build |
|---|---------|----------------|
| 00 | [The primer](00-primer.md) | Background: semantic segmentation, synthetic data, sim-to-real gap, domain randomization, and the DGX Spark |
| 01 | [The lunar stage](01-the-lunar-stage.md) | A procedural OpenUSD scene: regolith heightfield, realistic noise-displaced basalt rocks, sun, dome, rover camera; first pixel-accurate segmentation mask |
| 02 | [Domain randomization](02-domain-randomization.md) | Replicator pipeline; 2,550 labeled frames across three splits (DR, no-DR, unseen-domain test) with randomized lighting, materials, and geometry |
| 03 | [Training](03-training.md) | SegFormer-B0 fine-tuned on synthetic data; honest size-matched ablation — domain randomization adds **+0.046 rock-IoU**, DR-1500 ceiling **0.8521** |
| 04 | [Sim-to-real](04-sim-to-real.md) | Transfer evaluation on 7 real Apollo lunar photographs — the photoreal model **floods** ~83% of real regolith as rock |
| 05 | [The render](05-the-render.md) | Cinematic 1920×1080 RTX flythrough with live `dr_1500` hazard overlay — clean, because in-distribution |
| 06 | [Rock fidelity](06-rock-fidelity.md) | The v1→v2 evolution and the honest tradeoff: higher fidelity raised synthetic scores but worsened real transfer |

Heavy artifacts (datasets, checkpoints, full-res render) live on the DGX Spark, not in git. Each report's `## Reproduce` section has the copy-pasteable `ssh spark "docker exec …"` commands.
