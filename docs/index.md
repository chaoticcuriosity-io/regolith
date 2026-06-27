---
layout: default
title: Lunar hazard detection from synthetic data
---

# Lunar hazard detection from synthetic data

You want a rover to see rocks. There are almost no labeled real images of the lunar surface. So you render the data yourself — perfect labels, zero annotation cost — and train a model on it.

That's the whole idea. This series shows you the full loop: build a USD scene, run NVIDIA Omniverse Replicator to generate a dataset with domain randomization, fine-tune a SegFormer, measure the transfer gap to real imagery, and render a cinematic final output. Everything runs on a single NVIDIA DGX Spark.

---

## Results

Domain randomization adds **+0.099 rock-IoU** on an unseen synthetic domain (0.689 → 0.788, size-matched). The full DR-1500 model scores 0.815. On 7 real Apollo photographs the transfer is partial — large boulders and the horizon/sky boundary work; fine regolith is over-segmented; DR reduces false positives and largely removes the shadow-as-sky error that the no-DR model commits. DR is better, not fixed.

The series closes with a 1920×1080 cinematic RTX flythrough — a rover-eye dolly into a boulder field with the deployed model's predictions overlaid live on every frame.

![Preview animation — cinematic flythrough with live hazard overlay](reports/assets/render-preview.gif)

---

## Start here

- [00 — Primer: synthetic data, sim-to-real, and why the Moon is a hard problem](reports/00-primer.md)

## The series

| Chapter | Topic |
|---------|-------|
| [01 — The lunar stage](reports/01-the-lunar-stage.md) | USD scene: terrain, rocks, lighting |
| [02 — Domain randomization](reports/02-domain-randomization.md) | Replicator pipeline + 2,550-frame labeled dataset |
| [03 — Training](reports/03-training.md) | SegFormer fine-tuning + honest DR ablation |
| [04 — Sim-to-real](reports/04-sim-to-real.md) | Transfer gap, measured on real Apollo photos |
| [05 — The render](reports/05-the-render.md) | Cinematic 1920×1080 RTX flythrough with live hazard overlay |

---

*A [Chaotic Curiosity](https://chaoticcuriosity.io) project by Don Balanzat — sibling to [chaotic-fine-tuning](https://github.com/chaoticcuriosity-io/chaotic-fine-tuning) (LLM fine-tuning on the same Spark) and [g1-humanoid-rl](https://github.com/chaoticcuriosity-io/g1-humanoid-rl) (humanoid robot RL).*
