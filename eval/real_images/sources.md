# Real lunar surface photographs — sources & public-domain status

These are the **real** ground-level lunar surface photographs used for the sim-to-real
evaluation in [`docs/reports/04-sim-to-real.md`](../../docs/reports/04-sim-to-real.md).
The model was trained **only** on synthetic renders (chapters 02–03); these images are
its first contact with real lunar pixels.

## Curation criteria

Frames were chosen to **match the training viewpoint**: ground-level, rocky regolith
terrain, horizon, black sky, harsh single-source shadows. We deliberately **avoided**
frames dominated by astronauts, the lander/rover, flags, or Earth — the 3-class model
(`{regolith, rock, sky}`) was never trained on those objects, so testing on them would
be unfair. A few frames contain incidental reseau cross-hairs (the `+` fiducial marks
on Apollo Hasselblad film) and, on one or two, a sliver of equipment or a footpad at the
extreme edge; these are noted where relevant in the report.

The set spans **five missions**, one **color** frame (Apollo 17) and six **black-and-white**
frames, and both **with-sky** (horizon + black sky) and **without-sky** (down-looking,
terrain-filling) compositions — a fair spread of real conditions.

## Public-domain status

All images are NASA Apollo-program photographs. NASA still and motion imagery is generally
**not subject to copyright and is in the public domain** (see NASA media usage guidelines:
<https://www.nasa.gov/nasa-brand-center/images-and-media/>). They were retrieved over plain
HTTPS (no authentication) from the **NASA Image and Video Library** (`images.nasa.gov` /
`images-assets.nasa.gov`). Each file below was downsized to ≤1024 px on the long edge and
re-encoded as JPEG for repository weight; the originals are 1920 px.

## Image manifest

| File | NASA ID | Mission | Description | Detail page |
|------|---------|---------|-------------|-------------|
| `01-apollo11-surface-horizon.jpg` | AS11-40-5881 | Apollo 11 | Lunar surface and horizon — scattered small rocks, black sky | <https://images.nasa.gov/details/as11-40-5881> |
| `02-apollo14-cone-crater-boulder-field.jpg` | AS14-64-9118 | Apollo 14 | Field of boulders on the flank of Cone Crater | <https://images.nasa.gov/details/as14-64-9118> |
| `03-apollo14-large-boulder.jpg` | AS14-68-9448 | Apollo 14 | Large bright boulder on regolith, horizon, black sky | <https://images.nasa.gov/details/as14-68-9448> |
| `04-apollo14-boulder-field-overlook.jpg` | AS14-68-9451 | Apollo 14 | Large foreground boulders overlooking a boulder field | <https://images.nasa.gov/details/as14-68-9451> |
| `05-apollo15-hadley-fresh-crater.jpg` | AS15-82-11082 | Apollo 15 | Rocky "relatively fresh" crater rubble; Hadley massifs on horizon | <https://images.nasa.gov/details/as15-82-11082> |
| `06-apollo16-rock-block.jpg` | AS16-107-17573 | Apollo 16 | Close view of a sampled block on rocky regolith (down-looking, no sky) | <https://images.nasa.gov/details/as16-107-17573> |
| `07-apollo17-camelot-boulders.jpg` | AS17-145-22183 | Apollo 17 | **Color.** Station 5 (Camelot Crater) boulders; massif + black sky | <https://images.nasa.gov/details/as17-145-22183> |

Direct asset URL pattern (public, no auth):
`https://images-assets.nasa.gov/image/<NASA_ID>/<NASA_ID>~large.jpg`

## How they're used

`eval/eval_real.py` letterboxes each photo to the model's 512×512 input (these frames are
all within ~1% of square, so the black bars are negligible), ImageNet-normalizes it (matching
`training/dataset.py`), and runs both the DR-1500 and no-DR checkpoints. There is **no ground
truth** for these images, so the evaluation is strictly **qualitative** — colored overlays,
never a fabricated IoU.
