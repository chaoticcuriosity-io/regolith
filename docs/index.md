---
layout: default
title: Lunar hazard detection from synthetic data
---

# Lunar hazard detection from synthetic data

You want a rover to see rocks. There are almost no labeled real images of the lunar surface. So you render the data yourself — perfect labels, zero annotation cost — and train a model on it.

That's the whole idea. This series shows you the full loop: build a USD scene, run NVIDIA Omniverse Replicator to generate a dataset with domain randomization, fine-tune a SegFormer, measure the transfer gap to real imagery, and render a cinematic final output. Everything runs on a single NVIDIA DGX Spark.

---

## Results — an honest surprise

We made the synthetic rocks photoreal. The synthetic benchmark went **up**: deployed rock-IoU 0.815 → **0.852** on an unseen synthetic domain, with domain randomization adding +0.046 size-matched. But real-world transfer got **worse**: on 7 real Apollo photographs the photoreal model **floods** ~83% of pixels with false rock (up from ~52%), because photoreal rocks (rough, gray, bumpy) collapsed the rock-vs-regolith boundary toward "any rough gray texture is rock" — and real lunar regolith is exactly that. **Higher fidelity + higher synthetic score ≠ better real transfer.** Synthetic metrics can mislead; you have to test on real data.

The series closes with a 1920×1080 cinematic RTX flythrough — a rover-eye dolly into a boulder field with the deployed model's predictions overlaid live. The overlay is clean, but only because the scene is in-distribution — the same flattery the synthetic score gives.

![Preview animation — cinematic flythrough with live hazard overlay](reports/assets/render-preview.gif)

---

## Start here

- [00 — Primer: synthetic data, sim-to-real, and why the Moon is a hard problem](reports/00-primer.md)

## The series

| Chapter | Topic |
|---------|-------|
| [01 — The lunar stage](reports/01-the-lunar-stage.md) | USD scene: terrain, realistic basalt rocks, lighting |
| [02 — Domain randomization](reports/02-domain-randomization.md) | Replicator pipeline + 2,550-frame labeled dataset |
| [03 — Training](reports/03-training.md) | SegFormer fine-tuning + honest DR ablation (+0.046; ceiling 0.852) |
| [04 — Sim-to-real](reports/04-sim-to-real.md) | The flood: ~83% of real regolith called rock |
| [05 — The render](reports/05-the-render.md) | Cinematic 1920×1080 RTX flythrough with live hazard overlay |
| [06 — Rock fidelity](reports/06-rock-fidelity.md) | The v1→v2 evolution and the honest fidelity-vs-transfer tradeoff |

---

*A [Chaotic Curiosity](https://chaoticcuriosity.io) project by Don Balanzat — sibling to [chaotic-fine-tuning](https://github.com/chaoticcuriosity-io/chaotic-fine-tuning) (LLM fine-tuning on the same Spark) and [g1-humanoid-rl](https://github.com/chaoticcuriosity-io/g1-humanoid-rl) (humanoid robot RL).*
