"""
build_lunar_stage.py — author a USD stage representing a lunar landing/traverse scene.

Called by the Replicator pipeline (generate_dataset.py) and directly for
scene inspection / render preview.

Stage layout
------------
/World/
  Regolith       — displaced heightfield mesh tagged with USD Semantics class "regolith"
  Rocks          — UsdGeom.PointInstancer scatter of rock meshes, tagged "rock"
  Sun            — UsdLux.DistantLight at /World/Sun
  StarDome       — dome light or sky sphere, tagged "sky"
  RoverCam       — UsdGeom.Camera at /World/RoverCam

Class map (matches training/metrics.py)
---------------------------------------
  regolith: 0
  rock:     1
  sky:      2
"""

from __future__ import annotations

# TODO (Task 1): replace with actual Omniverse / USD imports once the
# Isaac Lab container is set up on the DGX Spark.
# from pxr import Usd, UsdGeom, UsdLux, Semantics


def build_lunar_stage(seed: int) -> "Usd.Stage":
    """Author a lunar scene USD stage and return it.

    Parameters
    ----------
    seed : int
        Random seed for deterministic terrain + rock scatter.
        Passed through to the heightfield displacement and rock placement.

    Returns
    -------
    Usd.Stage
        An in-memory USD stage with the full scene graph:
        - /World/Regolith — displaced regolith heightfield
        - /World/Rocks    — PointInstancer scatter of rock meshes
        - /World/Sun      — DistantLight
        - /World/StarDome — star/space dome
        - /World/RoverCam — camera prim

        All geometry prims carry USD Semantics class labels
        (regolith, rock, sky) so Replicator can emit perfect masks.

    Notes
    -----
    Heavy artifacts (USD assets, textures) live on the DGX Spark at
    /workspace/regolith/scene/assets/ — not in git.
    See docs/reports/01-the-lunar-stage.md for the full walkthrough.
    """
    # TODO (Task 1): implement
    raise NotImplementedError("build_lunar_stage: implement in Task 1 (Isaac setup)")
