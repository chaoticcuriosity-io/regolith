# eval/data

Real lunar imagery dataset support for sim-to-real evaluation. Fetches the **Artificial Lunar Rocky Landscape Dataset** (Kaggle), maps its mask palette to regolith canonical class IDs, and documents the sim-to-real evaluation split.

*Part of the [regolith](../..) demo — lunar hazard segmentation from 100% synthetic data. Task 4: sim-to-real evaluation.*

---

## The dataset

**Artificial Lunar Rocky Landscape Dataset**
Romain Pessia & Genya Ishigami, Space Robotics Group, Keio University, Japan.
Kaggle: [romainpessia/artificial-lunar-rocky-landscape-dataset](https://www.kaggle.com/datasets/romainpessia/artificial-lunar-rocky-landscape-dataset)

> If you use this dataset, please cite:
> Romain Pessia and Genya Ishigami. "Artificial Lunar Rocky Landscape Dataset." Kaggle, 2019.
> https://www.kaggle.com/datasets/romainpessia/artificial-lunar-rocky-landscape-dataset

**License:** check the Kaggle dataset page — the license was not publicly accessible without login as of 2026-06-27. Likely CC or "Other (specified in description)." Confirm before redistribution.

**Why this dataset?** It gives us two independent cross-checks:

1. *Domain-gap cross-check* — `images/render/` are photorealistic synthetic renders from Terragen (a different renderer/provenance than our Isaac Sim / Replicator renders). If the model generalizes across synthetic renderers, it is likely learning geometry rather than renderer-specific artifacts.
2. *True sim-to-real test* — `real_moon_images/` contains ~36 actual photographs from the Chang'e 3 lunar rover (PCAM + TCAM cameras). This is the genuine sim-to-real evaluation set — real dirt, real rocks, real cameras.

---

## Dataset structure

After running `download_real.py` the directory looks like this:

```
<out>/
├── images/
│   ├── render/           ← 9 766 synthetic renders (720×480 PNG)
│   │     render0001.png … render9766.png
│   ├── ground/           ← RGB segmentation masks (PREFERRED for eval)
│   │     ground0001.png … ground9766.png
│   └── clean/            ← cleaned masks (fewer small-pixel artifacts)
│         clean0001.png  … clean9766.png
└── real_moon_images/     ← ~36 Chang'e 3 rover photos (true sim-to-real set)
```

`render0042.png` pairs with `ground0042.png` and `clean0042.png` — same numeric index.

Prefer **ground/** masks over clean/ for evaluation — clean/ aggressively removes small labeled pixels, losing detail. A bounding-box CSV for large rocks is also included in the dataset root (useful for detection tasks; ignored here).

---

## Class mapping

The dataset masks are **RGB images** (not grayscale label maps). The palette uses four pure-primary colors:

| Dataset class   | Mask color | RGB value     | Canonical ID | Regolith class |
|-----------------|------------|---------------|:------------:|----------------|
| Ground/regolith | Black      | `(  0,  0,  0)` | **0**      | regolith       |
| Small rocks     | Green      | `(  0,255,  0)` | **1**      | rock (hazard)  |
| Large rocks     | Blue       | `(  0,  0,255)` | **1**      | rock (hazard)  |
| Sky             | Red        | `(255,  0,  0)` | **2**      | sky            |
| Other           | —          | —             | **255**      | ignore         |

Key design decisions:
- The dataset's "ground" (black background) is our **regolith(0)** — the two terms are semantically identical for a lunar surface scene.
- Small and large rocks are **collapsed into a single rock(1)** class — our model does not distinguish them, and both are hazards.
- `real_moon_images/` is the **genuine sim-to-real subset** — these are actual photographs, not renders. Some may have hand-drawn masks (verify structure when downloading; encoding may differ).

The palette constants live in `class_mapping.py` (`SRC_*` variables at the top of the file). If the actual downloaded masks differ from the documented palette, update only those constants.

---

## Files

| File | Purpose |
|------|---------|
| `download_real.py` | CLI fetch script — Kaggle API → HuggingFace fallback → manual instructions |
| `class_mapping.py` | `remap_lunar_landscape_mask()` + `verify_palette()` helpers |
| `README.md` | This file |

---

## How to download

**On the DGX Spark (recommended):**

```bash
# Step 1 — get Kaggle credentials
# Download kaggle.json from https://www.kaggle.com/settings → API → Create New Token
# then copy it to the Spark:
scp ~/Downloads/kaggle.json spark:~/.kaggle/kaggle.json
ssh spark 'chmod 600 ~/.kaggle/kaggle.json'

# Step 2 — run in the training container
ssh spark "docker exec mjlab-dev bash -lc '
  pip install -q kaggle &&
  python eval/data/download_real.py --out /workspace/datasets/lunar_landscape
'"
```

**Without Kaggle credentials** (browser download + manual copy):

```bash
# 1. Download the ZIP from:
#    https://www.kaggle.com/datasets/romainpessia/artificial-lunar-rocky-landscape-dataset
# 2. Copy to Spark:
scp artificial-lunar-rocky-landscape-dataset.zip spark:/workspace/datasets/
# 3. Unzip on Spark:
ssh spark 'unzip /workspace/datasets/artificial-lunar-rocky-landscape-dataset.zip \
    -d /workspace/datasets/lunar_landscape'
```

**Verify the palette** after first download (Task 4 step):

```python
import numpy as np
from PIL import Image
from eval.data.class_mapping import verify_palette

mask = np.array(Image.open("/workspace/datasets/lunar_landscape/images/ground/ground0001.png").convert("RGB"))
print(verify_palette(mask))
# Expected: {(0,0,0): N, (255,0,0): N, (0,255,0): N, (0,0,255): N}
```

---

## Citation

```bibtex
@misc{pessia2019lunar,
  author    = {Romain Pessia and Genya Ishigami},
  title     = {Artificial Lunar Rocky Landscape Dataset},
  year      = {2019},
  publisher = {Kaggle},
  url       = {https://www.kaggle.com/datasets/romainpessia/artificial-lunar-rocky-landscape-dataset}
}
```
