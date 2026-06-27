---
layout: default
title: Lunar hazard detection from synthetic data
---

# Lunar hazard detection from synthetic data

You want a rover to see rocks. There are almost no labeled real images of the lunar surface. So you render the data yourself — perfect labels, zero annotation cost — and train a model on it.

That's the whole idea. This series shows you the full loop: build a USD scene, run NVIDIA Omniverse Replicator to generate a dataset with domain randomization, fine-tune a SegFormer, measure the transfer gap to real imagery, and render a cinematic final output. Everything runs on a single NVIDIA DGX Spark.

---

## Start here

- [00 — Primer: synthetic data, sim-to-real, and why the Moon is a hard problem](reports/00-primer.md)

## The series

| Chapter | Topic |
|---------|-------|
| [01 — The lunar stage](reports/01-the-lunar-stage.md) | USD scene: terrain, rocks, lighting |
| [02 — Domain randomization](reports/02-domain-randomization.md) | Replicator pipeline + labeled dataset |
| [03 — Training](reports/03-training.md) | SegFormer fine-tuning |
| [04 — Sim-to-real](reports/04-sim-to-real.md) | Transfer gap, measured |
| [05 — The render](reports/05-the-render.md) | Cinematic RTX output |

---

*A [Chaotic Curiosity](https://chaoticcuriosity.io) project by Don Balanzat — sibling to [chaotic-fine-tuning](https://github.com/chaoticcuriosity-io/chaotic-fine-tuning) and [g1-humanoid-rl](https://github.com/chaoticcuriosity-io/g1-humanoid-rl).*
