# 00 — The primer: synthetic data, sim-to-real, and why the Moon is a hard problem

*Chaotic Curiosity | regolith series*

---

Before we write a line of code, let's get the map right. This chapter explains the problem, the approach, and the tools — in plain language. No background required.

---

## The problem

You want a rover or lander to *see* the terrain in front of it. Not just capture images — actually understand what's in them. Which pixels are solid ground? Which are rocks that could shred a wheel? Which is sky?

The technical name for this is **semantic segmentation** — the task of assigning a label (in our case: `regolith`, `rock`, or `sky`) to every single pixel in an image. It's a well-understood computer vision problem, with strong models available off the shelf. The challenge isn't the model. It's the data.

To train a segmentation model, you need thousands of labeled images — images where someone (or something) has already drawn the boundaries and said "this blob of pixels is a rock." On Earth, that data exists for roads, pedestrians, medical scans. On the Moon, it barely exists at all.

NASA and other agencies have returned a handful of surface photos with any useful labels. None cover the specific terrain you care about. None cover it under the specific lighting conditions that matter for your mission. And even if they did, the annotation process — a human drawing precise masks over tens of thousands of lunar images — would be slow, expensive, and inconsistent.

So you're stuck: the problem is real, the model architecture exists, but the training data doesn't.

---

## The idea: render it yourself

Here's the insight that unlocks everything: if you can *simulate* a scene precisely enough, the simulator already knows what every pixel is. You authored the scene — you placed every rock, set every light, defined every material. The labels aren't annotations; they're ground truth you generated for free as a byproduct of rendering.

This is **synthetic data** — training data that comes from a simulator or renderer rather than the real world.

To build the synthetic lunar world, we use two pieces of NVIDIA infrastructure:

**OpenUSD** (Universal Scene Description) is a file format and API, originally developed by Pixar, for describing 3D scenes. Think of it as the "source of truth" for geometry, materials, lights, and cameras. Everything in our scene — the rock-strewn regolith, the sun, the rover camera — lives in a USD file.

**NVIDIA Omniverse Replicator** is a Python framework built into NVIDIA Omniverse (it ships as part of Isaac Sim) that automates the process of generating datasets. You describe what you want varied (lighting angle, rock placement, surface texture), and Replicator renders thousands of frames — each with paired RGB images *and* perfect semantic segmentation masks.

We render once, label nothing, and get a training set.

---

## The catch: the sim-to-real gap

Here's the problem with synthetic data, told honestly: a model trained only on clean, perfect renders often fails on real images. The simulator can't perfectly replicate every artifact of real optics, sensor noise, dust scatter, or the subtle textures of actual regolith. The model learns to recognize features of the renderer, not features of the real world.

This is called the **sim-to-real gap** — the performance drop you observe when you move from simulated test conditions to real-world conditions. It's a known, documented problem, not a theory. You can measure it.

The standard fix is **domain randomization**: deliberately vary the simulation parameters wildly during training — sun elevation from horizon to zenith, rock albedo across a wide spectrum, camera field of view, regolith texture, rock count and scale — so that no single renderer artifact dominates. The model can't overfit to one lighting scheme or one texture style because it never sees the same one twice. What it *does* see, consistently, is the shapes and depth relationships between rock, ground, and sky — and those are real.

Domain randomization doesn't close the gap completely. Chapter 04 of this series characterizes the remaining gap on real Apollo imagery — the honest version of "it works."

---

## Why a DGX Spark

The full loop — build the USD scene, run Replicator to generate a dataset, fine-tune a segmentation model, evaluate it, render a cinematic output — requires both a capable GPU and enough CPU memory to hold the Omniverse simulation while training runs.

We run everything on a single **NVIDIA DGX Spark (GB10 Grace Blackwell, aarch64, CUDA 13, sm_121, 128 GB unified memory — nvidia-smi reports VRAM as N/A, which is expected)**. The unified memory architecture means there's no separate VRAM ceiling — the GPU and CPU share the full 128 GB pool. That's what lets us keep a full Omniverse environment resident alongside a training run without swapping.

There's one important gotcha: unified memory means an out-of-memory event doesn't surface as a clean CUDA OOM error — it becomes a swap death-spiral that can require a hard reboot. The `scripts/free_memory.sh` helper stops the non-essential co-tenant containers (`open-webui`, `ollama-compose`, `compose-arangodb-1`) before a big run. Keep at least 110 GiB free heading into any dataset generation or training session.

---

## What you'll build in this series

| Chapter | What you build | What you measure |
|---------|---------------|------------------|
| [01 — The lunar stage](01-the-lunar-stage.md) | A USD scene: displaced heightfield regolith, instanced rocks, a sun light, a star dome, a rover camera | Visual inspection of scene prims + render preview |
| [02 — Domain randomization](02-domain-randomization.md) | A Replicator pipeline that generates a labeled dataset with randomized lighting, textures, and rock placement | Frame count, class distribution, dataset size |
| [03 — Training](03-training.md) | A SegFormer fine-tuned on the synthetic dataset | val rock-IoU (the hazard class that matters most) |
| [04 — Sim-to-real evaluation](04-sim-to-real.md) | A transfer evaluation on real Apollo-era surface imagery | Qualitative transfer assessment — no fabricated real IoU |
| [05 — The render](05-the-render.md) | A cinematic RTX render with per-pixel segmentation overlaid | Artifact: `.mp4` render, published as a GitHub Release |

---

No pretense: the gap in chapter 04 will exist and may be significant. Domain randomization helps; it doesn't perform miracles. The point of this series is to build the full loop — generate, train, measure, render — and document what actually happens at each step.

Continue to [01 — The lunar stage](01-the-lunar-stage.md).
