# regolith

Train a lunar hazard segmentation model on 100% synthetic data — then learn, the hard way, *where* realism actually buys you real-world transfer. We made the training **rocks** photoreal (v2): the synthetic benchmark rose (rock-IoU 0.815 → 0.852) but real transfer got **worse** — the model flooded ~83% of real Apollo regolith with false rock, up from ~52%, because "rough gray = rock" is a shortcut that breaks on real soil. That failure handed us a hypothesis: the fidelity was on the wrong surface. So we moved it to the **ground** (v3) — a realistic cratered, dark, displaced regolith floor — and the shortcut disappeared. **Synthetic rock-IoU climbed to 0.887 and the real-photo flood dropped to 35.7%, below both prior builds.** This repo is the honest, reproducible story of that arc: the obvious upgrade that backfired, and the diagnosis that turned it into a fix. Everything runs on a single NVIDIA DGX Spark, from scene authoring through model training to a cinematic render.

*A [Chaotic Curiosity](https://chaoticcuriosity.io) project by Don Balanzat — sibling to [chaotic-fine-tuning](https://github.com/chaoticcuriosity-io/chaotic-fine-tuning) (LLM fine-tuning on the same Spark) and [g1-humanoid-rl](https://github.com/chaoticcuriosity-io/g1-humanoid-rl) (humanoid robot RL).*

---

## The throughline

You want a rover or lander to *see* rocks and hazards on the Moon. The catch: real labeled images of the lunar surface are almost nonexistent, and none exist for the specific terrain or lighting conditions you care about.

The answer is to manufacture the data yourself. Render a photorealistic lunar scene in a simulator — you set every rock, control every shadow, and get pixel-perfect labels for free. Train a segmentation model on those synthetic images. Then measure how well it transfers to reality.

That loop — generate synthetic data, train, measure the transfer gap — is the arc of this series, honestly told. And the honest result has two turns: the obvious upgrade (make the rocks photoreal) raised every synthetic score while making real-world behavior worse — and then the *diagnosis* of that failure pointed to the real lever (make the **ground** realistic), which fixed it. Failures stay documented as lessons; here the failure is what told us where to look. Every "it works" comes with a number or an artifact — and so does every "it doesn't."

---

## Start here

If you've never touched synthetic data pipelines or Omniverse, start at the primer:
**[`docs/reports/00-primer.md`](docs/reports/00-primer.md)** — what the problem is, what the tools are, and what you'll build.

Then follow the series in order (01 → 07) in [`docs/reports/`](docs/reports/).

To run anything yourself, see the ops manual: [`docs/dgx-spark-regolith-manual.md`](docs/dgx-spark-regolith-manual.md).

---

## What's here

| Path | What it does |
|------|-------------|
| `docs/reports/` | The learning series — eight chapters, 00 through 07 |
| `scene/` | USD scene builder — regolith terrain, rocks, lighting, camera |
| `replicator/` | NVIDIA Omniverse Replicator pipeline — randomizers + dataset generator |
| `training/` | SegFormer fine-tuning — dataset loader, model, metrics, train loop |
| `eval/` | Evaluation on synthetic hold-out and real lunar imagery |
| `render/` | Cinematic RTX render with predicted segmentation overlaid |
| `scripts/` | Ops helpers — memory management (`free_memory.sh`), dataset assembly (`generate_all.sh`); per-step reproduce commands live in each chapter's `## Reproduce` block |
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

## Results

**v3 is the headline: we moved the fidelity from the rocks to the ground, and the real-world flood collapsed.** The cross-version arc, deployed `dr_1500` checkpoint each time, synthetic rock-IoU on the held-out `test_photoreal` split and false-rock flood on real NASA public-domain Apollo + Surveyor photographs:

| Version | rocks | ground | synth rock-IoU | real flood (same 21 images) |
|---------|-------|--------|:--------------:|:----------:|
| v1 | low-poly blobs | smooth heightfield | 0.815 | 44.0% |
| v2 | photoreal basalt | smooth heightfield | 0.852 | 72.6% |
| **v3** | power-law basalt | **cratered dark displaced** | **0.887** | **35.7%** |

The flood column is an **exact, apples-to-apples comparison**: all three `dr_1500` checkpoints are scored with the same hardened eval on the same 21 NASA photographs (`eval/real_images_v3`), not on different image pools. (For provenance, v1 and v2 were *originally* measured on an older 7-image set at ~52% / ~83% — the numbers chapters 04/06 report; the same checkpoints re-run on the harder 21-image set give 44.0% / 72.6%.)

v1→v2 raised the synthetic score and made real transfer *worse* — the failure. v2→v3 raised the synthetic score **and** dropped the real flood below even v1, by changing one design decision: the realism moved off the rocks and onto the **ground**.

![Preview animation of the v3 cinematic flythrough — VIPER's forward hazard-cam crossing a cratered dark-regolith boulder field with the live dr_1500 hazard overlay (boulders red and outlined, the rough regolith floor correctly left green and un-flooded)](docs/reports/assets/render-v3-preview.gif)

**Why the flood dropped.** v2's model leaned on a texture shortcut — "rough gray bumpy = rock" — which works in a simulator where only the rocks are rough, and breaks on real film where regolith is rough too. v3 makes the **ground** realistic (albedo ~0.08–0.12, cratered, displaced, normal-mapped) plus power-law rock sizes and harsh lunar lighting. Now both rock and ground are rough inside the simulator, so the texture shortcut no longer separates the classes — the model is forced to learn **shape, cast shadow, and scale**, cues that are real and transfer. The full v3 story is in [chapter 07](docs/reports/07-realistic-ground.md); the v2 failure that set it up is [chapter 06](docs/reports/06-rock-fidelity.md).

**The v3 size-matched ablation** (`test_photoreal`, 300 unseen-domain frames):

| Run | Frames | DR? | synth rock-IoU | real flood |
|-----|-------:|:---:|:--------------:|:----------:|
| `nodr_750` | 750 | no | 0.8593 | 39.7% |
| `dr_750` | 750 | **yes** | 0.8675 | 36.3% |
| `dr_1500` | 1,500 | yes | **0.8870** | **35.7%** |

Domain randomization now helps on **both** axes — the size-matched DR gain is +0.008 synth rock-IoU and DR *reduces* the real flood (39.7% → 35.7%), the opposite of v2, where DR worsened it. The DR gain is small because the realistic ground raised the no-DR floor (0.8593) above v2's *best deployed* model (0.8521); scale adds the +0.019 rest. Honest caveat: the Apollo 11 Tranquility wide panorama stays a stubborn outlier (~71% flood) — a low-contrast, distant-horizon frame with no strong shadows or discrete objects for the honest cue to grab. The numbers tie to [`outputs/runs_v3/RESULTS.md`](outputs/runs_v3/RESULTS.md).

**The render** is a 1920 × 1080 cinematic flythrough of NASA's **VIPER** rover crossing the v3 boulder field — a rover's-eye hazard HUD: the forward hazard-cam with the live `dr_1500` overlay. The overlay is clean *and*, unlike v2, the model behind it now also holds up far better on real photographs (35.7% flood). The render scene is still in-distribution, so it remains the flattering view — but it is finally backed by a real-photo number that moved the right way. The full 1080p MP4 ([`render-v3-preview.mp4`](docs/reports/assets/render-v3-preview.mp4)), hero stills, and beauty plates are in [`docs/reports/assets/`](docs/reports/assets/).

---

## Status

Series complete through v3. All eight chapters published; v3 (realistic-ground build) is the current headline.

| Chapter | Report |
|---------|--------|
| [00 — Primer](docs/reports/00-primer.md) | Synthetic data, sim-to-real gap, domain randomization, the DGX Spark |
| [01 — The lunar stage](docs/reports/01-the-lunar-stage.md) | USD scene: regolith heightfield, realistic noise-displaced basalt rocks, sun, dome, camera |
| [02 — Domain randomization](docs/reports/02-domain-randomization.md) | Replicator pipeline; 2,550 labeled frames across three splits |
| [03 — Training](docs/reports/03-training.md) | SegFormer fine-tuned on synthetic data; honest size-matched DR ablation |
| [04 — Sim-to-real](docs/reports/04-sim-to-real.md) | Transfer on real Apollo photographs — the v2 model floods ~83% of real regolith as rock |
| [05 — The render](docs/reports/05-the-render.md) | Cinematic 1920×1080 RTX flythrough with live hazard overlay (clean, in-distribution) |
| [06 — Rock fidelity](docs/reports/06-rock-fidelity.md) | The v1→v2 evolution and the failure: photoreal rocks raised synthetic scores but worsened real transfer |
| [07 — Realistic ground](docs/reports/07-realistic-ground.md) | **v3:** fidelity moved to the ground — synth rock-IoU 0.887, real flood down to 35.7%; VIPER render |

Heavy artifacts (datasets, checkpoints, `.pt`, full-res `.mp4`) live on the DGX Spark, not in git. Reproduce commands are in each report's `## Reproduce` block.
