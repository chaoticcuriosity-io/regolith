"""
randomizers.py — register NVIDIA Omniverse Replicator randomizations for the lunar scene.

Called once per dataset generation run (generate_dataset.py) before the
Replicator orchestrator starts iterating frames.

Randomization parameters (set in configs/*.yaml, passed in via cfg)
--------------------------------------------------------------------
  sun_elevation_deg   : [min, max] — 5 to 85 recommended
  sun_azimuth_deg     : [0, 360]
  sun_intensity       : [min, max] in nits
  regolith_albedo     : [min, max]
  regolith_texture    : list of texture asset paths
  rock_count          : [min, max] integer
  rock_scale          : [min, max] float
  rock_placement_seed : randomized per frame by Replicator
  camera_pose_delta   : small random offset from /World/RoverCam base pose
  camera_fov_deg      : [min, max]
"""

from __future__ import annotations

# TODO (Task 2): replace with actual Replicator imports
# import omni.replicator.core as rep


def register(cfg) -> None:
    """Register all Replicator randomizations for the lunar scene.

    Parameters
    ----------
    cfg : dict | SimpleNamespace
        Configuration loaded from a YAML in replicator/configs/.
        Expected keys mirror the randomization parameters listed above.

    Side effects
    ------------
    Calls omni.replicator.core.randomizer.register() for each
    randomization parameter. After this function returns, the
    Replicator orchestrator can call rep.orchestrator.run() to
    iterate frames.

    Notes
    -----
    See docs/reports/02-domain-randomization.md for the rationale
    behind each randomization range.
    """
    # TODO (Task 2): implement
    raise NotImplementedError("register: implement in Task 2 (Replicator SDG)")
