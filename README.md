# regolith

Train a lunar hazard segmentation model on 100% synthetic data — no real labeled images, no GPU cluster, no guesswork. Everything runs on a single NVIDIA DGX Spark, from scene authoring through model training to a cinematic render that shows the model's predictions overlaid on a procedurally generated moonscape.

*A [Chaotic Curiosity](https://chaoticcuriosity.io) project by Don Balanzat — sibling to [chaotic-fine-tuning](https://github.com/chaoticcuriosity-io/chaotic-fine-tuning) (LLM fine-tuning on the same Spark) and [g1-humanoid-rl](https://github.com/chaoticcuriosity-io/g1-humanoid-rl) (humanoid robot RL).*

---

## The throughline

You want a rover or lander to *see* rocks and hazards on the Moon. The catch: real labeled images of the lunar surface are almost nonexistent, and none exist for the specific terrain or lighting conditions you care about.

The answer is to manufacture the data yourself. Render a photorealistic lunar scene in a simulator — you set every rock, control every shadow, and get pixel-perfect labels for free. Train a segmentation model on those synthetic images. Then measure how well it transfers to reality.

That loop — generate synthetic data, train, measure the transfer gap, close it — is the arc of this series, honestly told. Failures stay documented as lessons. Every "it works" comes with a number or an artifact.

---

## Start here

If you've never touched synthetic data pipelines or Omniverse, start at the primer:
**[`docs/reports/00-primer.md`](docs/reports/00-primer.md)** — what the problem is, what the tools are, and what you'll build.

Then follow the series in order (01 → 05) in [`docs/reports/`](docs/reports/).

To run anything yourself, see the ops manual: [`docs/dgx-spark-regolith-manual.md`](docs/dgx-spark-regolith-manual.md).

---

## What's here

| Path | What it does |
|------|-------------|
| `docs/reports/` | The learning series — six chapters, 00 through 05 |
| `scene/` | USD scene builder — regolith terrain, rocks, lighting, camera |
| `replicator/` | NVIDIA Omniverse Replicator pipeline — randomizers + dataset generator |
| `training/` | SegFormer fine-tuning — dataset loader, model, metrics, train loop |
| `eval/` | Evaluation on synthetic hold-out and real lunar imagery |
| `render/` | Cinematic RTX render with predicted segmentation overlaid |
| `scripts/` | Ops helpers — memory management, end-to-end reproduce |
| `docs/` | GitHub Pages source + DGX Spark ops manual |

---

## The stack

| Layer | Tool |
|-------|------|
| Scene authoring | [OpenUSD](https://openusd.org) |
| Synthetic data generation | [NVIDIA Omniverse Replicator](https://developer.nvidia.com/omniverse/replicator) |
| Segmentation model | [SegFormer](https://huggingface.co/docs/transformers/model_doc/segformer) (HuggingFace Transformers) |
| Training framework | PyTorch |
| Rendering | NVIDIA Omniverse RTX renderer |
| Hardware | NVIDIA DGX Spark (GB10 Grace Blackwell, aarch64, CUDA 13, sm_121, 128 GB unified memory — nvidia-smi reports VRAM as N/A, which is expected) |

---

## Status

| Task | Status | Report |
|------|--------|--------|
| 00 — Scaffold | ✓ done | — |
| 01 — Lunar stage (USD scene) | not started | [01-the-lunar-stage.md](docs/reports/01-the-lunar-stage.md) |
| 02 — Domain randomization + dataset | not started | [02-domain-randomization.md](docs/reports/02-domain-randomization.md) |
| 03 — Training | not started | [03-training.md](docs/reports/03-training.md) |
| 04 — Sim-to-real evaluation | not started | [04-sim-to-real.md](docs/reports/04-sim-to-real.md) |
| 05 — Cinematic render | not started | [05-the-render.md](docs/reports/05-the-render.md) |

Heavy artifacts (datasets, checkpoints, `.pt`, `.mp4`) live on the DGX Spark, not in git. Reproduce commands are in each report's `## Reproduce` block.
