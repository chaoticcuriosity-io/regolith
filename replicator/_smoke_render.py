#!/usr/bin/env python3
"""Minimal Omniverse Replicator SDG smoke test for DGX Spark (aarch64, GB10)."""
import os, sys, glob, time

def phase(msg):
    print(">>> PHASE %s  (+%.1fs)" % (msg, time.time() - _T0), flush=True); sys.stderr.flush()

_T0 = time.time()
OUT_DIR = os.environ.get("SMOKE_OUT", "/workspace/out")

# 1. SimulationApp MUST be created before importing omni.replicator.core
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True, "renderer": "RayTracedLighting"})
phase("simapp_ready")

import carb
import omni.replicator.core as rep

# Let omni.replicator.core finish startup (wp.init() + registries + OmniGraph nodes).
for _ in range(10):
    simulation_app.update()
phase("ext_started")

rep.orchestrator.set_capture_on_play(False)        # manual triggering only

# 2. Trivial scene: cube on a plane, one distant light, one camera
with rep.new_layer():
    plane = rep.create.plane(scale=10, semantics=[("class", "floor")])
    cube = rep.create.cube(position=(0, 0, 50), scale=20, semantics=[("class", "cube")])
    light = rep.create.light(light_type="distant", intensity=3000, rotation=(-45, 0, 0))
    camera = rep.create.camera(position=(250, 250, 200), look_at=(0, 0, 0))
    render_product = rep.create.render_product(camera, (512, 512))
phase("scene_built")

# 3. BasicWriter with rgb + semantic_segmentation
writer = rep.WriterRegistry.get("BasicWriter")
writer.initialize(output_dir=OUT_DIR, rgb=True, semantic_segmentation=True,
                  colorize_semantic_segmentation=True)
writer.attach([render_product])
phase("writer_attached")

# 4. Warm up a few frames
for i in range(5):
    simulation_app.update()
phase("warmup_done")

# 5. Capture ONE frame
rep.orchestrator.step(delta_time=0.0, rt_subframes=8)
phase("step_returned")

# 6. Drain: pump updates and POLL for files (first RTX frame is very slow on Spark;
#    this outlasts wait_until_complete()'s internal drain timeout).
deadline = time.time() + 180.0
while time.time() < deadline:
    simulation_app.update()
    if len(glob.glob(os.path.join(OUT_DIR, "**", "*.png"), recursive=True)) >= 2:
        phase("files_present"); break
else:
    phase("drain_timeout")

try:
    rep.orchestrator.wait_until_complete()
except Exception as e:
    print(">>> wait_until_complete raised: %r" % e, flush=True)
phase("drained")

writer.detach()
simulation_app.close()
print("SMOKE_TEST_DONE output_dir=" + OUT_DIR, flush=True)
