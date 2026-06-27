"""
class_mapping.py — map Artificial Lunar Rocky Landscape Dataset masks to regolith canonical IDs.

Dataset
-------
  Kaggle: romainpessia/artificial-lunar-rocky-landscape-dataset
  Authors: Romain Pessia & Genya Ishigami, Space Robotics Group, Keio University

Source mask encoding
--------------------
Masks (in the ground/ and clean/ subdirectories) are **RGB images** with exactly four
distinct pixel colors, one per class. Multiple independent analyses of this dataset
(see Sources below) consistently describe the following pure-primary palette:

  | Dataset class   | Mask color | RGB value    | Canonical ID |
  |-----------------|------------|--------------|--------------|
  | Ground/regolith | Black      | (  0,  0,  0)| 0            |
  | Small rocks     | Green      | (  0,255,  0)| 1            |
  | Large rocks     | Blue       | (  0,  0,255)| 1            |
  | Sky             | Red        | (255,  0,  0)| 2            |
  | Any other pixel | —          | —            | 255 (ignore) |

Confidence: HIGH. Sources consulted:
  - ayushdabra/lunar-landscape-images-segmentation (GitHub README)
  - Moeinh77/Rock-Segmentation-Artificial-Lunar-Landscape (GitHub README)
  - LunarX-Alpha/lunar-segmentation (GitHub)
  - Multiple Medium articles (Trivedi, Panchal) describing the same four-color scheme

Our canonical class IDs (must match training config)
-----------------------------------------------------
  regolith : 0
  rock     : 1   ← both small and large rocks collapsed into one hazard class
  sky      : 2
  ignore   : 255

TODO (Task 4, palette confirmation)
------------------------------------
When the dataset is first downloaded to the Spark, call verify_palette() on a sample
of masks to confirm the exact RGB tuples — particularly that:
  a) There are no anti-aliasing / JPEG artifacts producing near-primary colors
     (the masks are synthetic PNGs so this is unlikely, but worth checking).
  b) The ground/ masks use the same palette as the clean/ masks.
  c) real_moon_images/ masks (if any are hand-drawn) may use a different encoding.

If the palette differs from what's documented here, update ONLY the SRC_* constants
at the top of this file — the mapping logic below does not need to change.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Source palette — one place to update if verification reveals different values.
# ---------------------------------------------------------------------------

SRC_GROUND_RGB: tuple[int, int, int] = (0, 0, 0)       # black  → regolith(0)
SRC_ROCK_SMALL_RGB: tuple[int, int, int] = (0, 255, 0)  # green  → rock(1)
SRC_ROCK_LARGE_RGB: tuple[int, int, int] = (0, 0, 255)  # blue   → rock(1)
SRC_SKY_RGB: tuple[int, int, int] = (255, 0, 0)         # red    → sky(2)

# Canonical target IDs — must match regolith training config
TGT_REGOLITH: int = 0
TGT_ROCK: int = 1
TGT_SKY: int = 2
TGT_IGNORE: int = 255


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def remap_lunar_landscape_mask(mask_rgb: np.ndarray) -> np.ndarray:
    """Convert an Artificial Lunar Rocky Landscape Dataset mask to canonical class IDs.

    Parameters
    ----------
    mask_rgb:
        uint8 array of shape (H, W, 3) — RGB segmentation mask from the ground/ or
        clean/ subdirectory.  Load with, e.g.::

            import numpy as np
            from PIL import Image
            mask_rgb = np.array(Image.open(path).convert("RGB"))

        If loading with OpenCV (BGR order), flip channels first::

            import cv2
            mask_bgr = cv2.imread(str(path))
            mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)

    Returns
    -------
    np.ndarray
        uint8 array of shape (H, W) with values in {0, 1, 2, 255}.

        0  = regolith  (dataset "ground" / black background)
        1  = rock      (dataset small rocks + large rocks merged)
        2  = sky       (dataset sky)
        255 = ignore   (unknown/other pixels — should be rare in these masks)

    Notes
    -----
    - Small rocks (green) and large rocks (blue) are collapsed into a single rock(1)
      class to match the three-class regolith training setup.
    - Pixels not matching any known palette entry map to ignore(255). In practice this
      catches any edge artifacts; a clean mask should have very few ignore pixels.
    - Prefer ground/ masks over clean/ for evaluation — ground/ has more coverage
      (clean/ aggressively removes small labeled regions).
    """
    if mask_rgb.ndim != 3 or mask_rgb.shape[2] != 3:
        raise ValueError(
            f"Expected RGB array of shape (H, W, 3); got shape {mask_rgb.shape}"
        )

    h, w = mask_rgb.shape[:2]
    out = np.full((h, w), TGT_IGNORE, dtype=np.uint8)

    r = mask_rgb[..., 0].astype(np.int16)
    g = mask_rgb[..., 1].astype(np.int16)
    b = mask_rgb[..., 2].astype(np.int16)

    # Ground → regolith(0)
    gr, gg, gb = SRC_GROUND_RGB
    out[(r == gr) & (g == gg) & (b == gb)] = TGT_REGOLITH

    # Small rocks → rock(1)
    sr, sg, sb = SRC_ROCK_SMALL_RGB
    out[(r == sr) & (g == sg) & (b == sb)] = TGT_ROCK

    # Large rocks → rock(1)
    lr, lg, lb = SRC_ROCK_LARGE_RGB
    out[(r == lr) & (g == lg) & (b == lb)] = TGT_ROCK

    # Sky → sky(2)
    skr, skg, skb = SRC_SKY_RGB
    out[(r == skr) & (g == skg) & (b == skb)] = TGT_SKY

    return out


def verify_palette(mask_rgb: np.ndarray) -> dict[tuple[int, int, int], int]:
    """Return all unique (R, G, B) values in a mask and their pixel counts.

    Call this on a handful of downloaded masks (before running full evaluation)
    to confirm that the palette constants above match the actual files::

        from PIL import Image
        import numpy as np
        from eval.data.class_mapping import verify_palette

        mask = np.array(Image.open("images/ground/ground0001.png").convert("RGB"))
        print(verify_palette(mask))
        # Expected: {(0,0,0): N, (255,0,0): N, (0,255,0): N, (0,0,255): N}

    If unexpected colors appear, update SRC_* constants at the top of this file.

    Parameters
    ----------
    mask_rgb:
        uint8 array of shape (H, W, 3).

    Returns
    -------
    dict mapping (r, g, b) → pixel count, sorted by descending count.
    """
    pixels = mask_rgb.reshape(-1, 3)
    unique, counts = np.unique(pixels, axis=0, return_counts=True)
    result = {tuple(int(v) for v in rgb): int(cnt)
              for rgb, cnt in zip(unique, counts)}
    return dict(sorted(result.items(), key=lambda kv: -kv[1]))
