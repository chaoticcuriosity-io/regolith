# Reports — curriculum TOC

Read in order 00 → 05. Each chapter is self-contained but assumes the previous one.

| # | Chapter | Status | What you build |
|---|---------|--------|----------------|
| 00 | [The primer](00-primer.md) | done | Background: semantic segmentation, synthetic data, sim-to-real gap, domain randomization |
| 01 | [The lunar stage](01-the-lunar-stage.md) | not started | USD scene: regolith heightfield, rock instancer, sun, dome, camera |
| 02 | [Domain randomization](02-domain-randomization.md) | not started | Replicator pipeline; 10 k labeled frames with randomized lighting + placement |
| 03 | [Training](03-training.md) | not started | SegFormer fine-tuned on synthetic data; best checkpoint on val rock-IoU |
| 04 | [Sim-to-real](04-sim-to-real.md) | not started | Transfer evaluation on real Apollo-era imagery; gap measured |
| 05 | [The render](05-the-render.md) | not started | Cinematic RTX `.mp4` with segmentation overlaid; published to GitHub Pages |

Heavy artifacts (datasets, checkpoints, renders) live on the DGX Spark, not in git. Each report's `## Reproduce` section has the copy-pasteable `ssh spark "docker exec …"` commands.
