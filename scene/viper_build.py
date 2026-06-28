"""
viper_build.py — author a faithful NASA VIPER (Volatiles Investigating Polar
Exploration Rover) lunar rover in Blender 4.4 (headless) and export to USD.

No CC0/public-domain VIPER 3D model exists for download (verified: NASA 3D
Resources, NASA mission pages, Smithsonian Open Access, Sketchfab CC search,
GitHub). The real lunar rovers that do exist (Lunokhod 1, Yutu) are CC-BY on
Sketchfab and need an authenticated download. So this builds VIPER from
primitives to the documented configuration, fully public-domain (self-authored).

VIPER silhouette cues captured:
  - cubic gold/black MLI-blanket body, golf-cart sized (~1.5 x 1.5 x 1 m)
  - 4 corner wheels on prominent suspension legs (NOT 6 like Perseverance)
  - VERTICAL side solar panels rising above the body (VIPER's most distinctive
    feature — vertical because the south-pole sun grazes the horizon)
  - tall FRONT mast with stereo NavCam head + headlights (drives into shadow)
  - top high-gain antenna patch
  - front-mounted TRIDENT drill pointing at the ground
  - NO RTG (VIPER is solar-powered)

Frame convention (to match build_lunar_stage_v3._reference_rover_usd):
  +Z up, FRONT faces -Y (mast + drill at -Y end), wheel contact at local z = 0,
  authored in real meters (use V3_ROVER_SCALE = 1.0). Z-up USD export.

Run:  blender --background --python viper_build.py -- /abs/out/viper.usd
"""
import bpy
import sys
import math
from mathutils import Vector

# ----------------------------------------------------------------------------- #
out_path = "viper.usd"
if "--" in sys.argv:
    out_path = sys.argv[sys.argv.index("--") + 1]

# --- clean scene -------------------------------------------------------------- #
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

# --- materials ---------------------------------------------------------------- #
def mat(name, base, metallic=0.0, rough=0.5, emiss=None, emiss_str=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (base[0], base[1], base[2], 1.0)
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = rough
    if emiss is not None:
        bsdf.inputs["Emission Color"].default_value = (emiss[0], emiss[1], emiss[2], 1.0)
        bsdf.inputs["Emission Strength"].default_value = emiss_str
    return m

M_GOLD   = mat("VIPER_GoldFoil",  (0.66, 0.49, 0.16), metallic=0.92, rough=0.34)
M_BLACK  = mat("VIPER_BlackFoil", (0.045, 0.045, 0.05), metallic=0.35, rough=0.55)
M_TIRE   = mat("VIPER_Wheel",     (0.05, 0.05, 0.055), metallic=0.25, rough=0.82)
M_HUB    = mat("VIPER_Hub",       (0.52, 0.53, 0.57), metallic=0.9,  rough=0.3)
M_SOLAR  = mat("VIPER_SolarCell", (0.03, 0.06, 0.20), metallic=0.6,  rough=0.22,
               emiss=(0.01, 0.03, 0.09), emiss_str=0.4)
M_FRAME  = mat("VIPER_PanelFrame",(0.62, 0.63, 0.66), metallic=0.85, rough=0.32)
M_MAST   = mat("VIPER_MastMetal", (0.55, 0.56, 0.60), metallic=0.85, rough=0.3)
M_LIGHT  = mat("VIPER_Headlight", (1.0, 0.97, 0.88),  metallic=0.0,  rough=0.1,
               emiss=(1.0, 0.95, 0.82), emiss_str=6.0)
M_WHITE  = mat("VIPER_Antenna",   (0.80, 0.80, 0.83), metallic=0.1,  rough=0.4)
M_DRILL  = mat("VIPER_Drill",     (0.30, 0.30, 0.33), metallic=0.8,  rough=0.4)
M_RED    = mat("VIPER_Accent",    (0.55, 0.05, 0.05), metallic=0.2,  rough=0.5)

_objs = []

def _finish(obj, material, bevel=0.0, shade_smooth=False, segments=2):
    obj.data.materials.append(material)
    if bevel > 0.0:
        b = obj.modifiers.new("bev", "BEVEL")
        b.width = bevel
        b.segments = segments
        b.limit_method = 'ANGLE'
    if shade_smooth:
        for poly in obj.data.polygons:
            poly.use_smooth = True
    _objs.append(obj)
    return obj

def box(name, size, loc, material, bevel=0.02, rot=None):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=loc)
    o = bpy.context.active_object
    o.name = name
    o.scale = (size[0] / 2.0, size[1] / 2.0, size[2] / 2.0)
    if rot:
        o.rotation_euler = rot
    return _finish(o, material, bevel=bevel)

def cyl(name, r, depth, loc, material, axis='Z', rot=None, bevel=0.0, verts=48):
    bpy.ops.mesh.primitive_cylinder_add(radius=r, depth=depth, location=loc, vertices=verts)
    o = bpy.context.active_object
    o.name = name
    if axis == 'X':
        o.rotation_euler = (0.0, math.radians(90.0), 0.0)
    elif axis == 'Y':
        o.rotation_euler = (math.radians(90.0), 0.0, 0.0)
    if rot:
        o.rotation_euler = rot
    return _finish(o, material, bevel=bevel, shade_smooth=True)

# ============================================================================= #
# Geometry. +Z up, front = -Y, wheel contact at z = 0, meters.
# ============================================================================= #
GC = 0.36            # ground clearance (belly height)
BODY_W, BODY_L, BODY_H = 1.34, 1.46, 0.92
body_cz = GC + BODY_H / 2.0
body_top = GC + BODY_H

# --- main body: gold MLI blanket box with a black underbody tray -------------- #
box("Body", (BODY_W, BODY_L, BODY_H), (0, 0, body_cz), M_GOLD, bevel=0.04)
box("Belly", (BODY_W * 0.96, BODY_L * 0.96, 0.18), (0, 0, GC + 0.02), M_BLACK, bevel=0.02)
# top deck plate + a couple of instrument greebles
box("Deck", (BODY_W * 0.9, BODY_L * 0.9, 0.06), (0, 0, body_top + 0.03), M_BLACK, bevel=0.01)
box("Avionics", (0.42, 0.5, 0.16), (0.28, 0.34, body_top + 0.12), M_GOLD, bevel=0.02)
box("AvBlack", (0.34, 0.3, 0.14), (-0.30, 0.30, body_top + 0.11), M_BLACK, bevel=0.02)

# --- 4 wheels on suspension legs (corner-mounted) ----------------------------- #
WR, WW = 0.345, 0.17
wx, wy = 0.70, 0.62
for sx in (-1.0, 1.0):
    for sy in (-1.0, 1.0):
        tag = ("F" if sy < 0 else "R") + ("L" if sx < 0 else "R")  # front=-Y
        wcx, wcy = sx * wx, sy * wy
        # tire
        cyl("Wheel_%s" % tag, WR, WW, (wcx, wcy, WR), M_TIRE, axis='X')
        # bright hub disc (mesh-wheel read)
        cyl("Hub_%s" % tag, WR * 0.42, WW + 0.03, (wcx, wcy, WR), M_HUB, axis='X')
        # suspension leg: angled strut from hub up to the body corner
        bx, by, bz = sx * (BODY_W / 2.0 - 0.06), sy * (BODY_L / 2.0 - 0.10), GC + 0.06
        mid = ((wcx + bx) / 2.0, (wcy + by) / 2.0, (WR + bz) / 2.0)
        d = Vector((bx - wcx, by - wcy, bz - WR))
        length = d.length
        # orient a thin box along d
        box("Leg_%s" % tag, (0.07, 0.09, length), mid, M_MAST, bevel=0.01)
        leg = _objs[-1]
        # rotate +Z to point along d
        z = Vector((0, 0, 1))
        axis = z.cross(d.normalized())
        ang = math.acos(max(-1.0, min(1.0, z.dot(d.normalized()))))
        if axis.length > 1e-6:
            leg.rotation_mode = 'AXIS_ANGLE'
            leg.rotation_axis_angle = (ang, axis.x, axis.y, axis.z)
        # small shoulder actuator (cylinder) at the body joint
        cyl("Shldr_%s" % tag, 0.075, 0.16, (bx, by, bz), M_HUB, axis='X')

# --- VERTICAL side solar panels (VIPER signature) ----------------------------- #
PANEL_T, PANEL_L, PANEL_H = 0.035, 1.30, 1.02
panel_cz = body_top + PANEL_H / 2.0 - 0.18      # overlap base into the body sides
for sx, tag in ((-1.0, "L"), (1.0, "R")):
    px = sx * (BODY_W / 2.0 + 0.055)
    # frame (slightly larger, behind the cells)
    box("PanelFrame_%s" % tag, (PANEL_T + 0.02, PANEL_L + 0.04, PANEL_H + 0.04),
        (px, 0.0, panel_cz), M_FRAME, bevel=0.01)
    # solar cell face
    box("Panel_%s" % tag, (PANEL_T, PANEL_L, PANEL_H), (px + sx * 0.015, 0.0, panel_cz),
        M_SOLAR, bevel=0.0)
    # support arm from body to panel
    box("PanelArm_%s" % tag, (0.10, 0.12, 0.10), (sx * (BODY_W / 2.0 - 0.02), 0.0, body_top - 0.06),
        M_MAST, bevel=0.01)

# --- top high-gain antenna patch (tilted) ------------------------------------- #
cyl("Antenna", 0.20, 0.035, (-0.12, 0.36, body_top + 0.30), M_WHITE, axis='Z',
    rot=(math.radians(18.0), 0.0, math.radians(20.0)))
cyl("AntStalk", 0.025, 0.34, (-0.12, 0.30, body_top + 0.16), M_MAST, axis='Z')

# --- FRONT mast: stereo NavCam head + headlights ------------------------------ #
mast_x, mast_y = 0.0, -(BODY_L / 2.0 - 0.18)
mast_base = body_top + 0.02
mast_h = 1.14
mast_top = mast_base + mast_h
cyl("Mast", 0.052, mast_h, (mast_x, mast_y, mast_base + mast_h / 2.0), M_MAST, axis='Z')
# camera head (looks -Y = forward)
box("CamHead", (0.52, 0.17, 0.18), (mast_x, mast_y - 0.04, mast_top + 0.03), M_BLACK, bevel=0.02)
for ex, tg in ((-0.17, "L"), (0.17, "R")):
    cyl("NavCam_%s" % tg, 0.045, 0.07, (mast_x + ex, mast_y - 0.13, mast_top + 0.03),
        M_HUB, axis='Y')
# headlights on the head (emissive) + lower body lights
for ex, tg in ((-0.22, "L"), (0.22, "R")):
    cyl("MastLight_%s" % tg, 0.05, 0.05, (mast_x + ex, mast_y - 0.12, mast_top - 0.05),
        M_LIGHT, axis='Y')
for ex, tg in ((-0.42, "L"), (0.42, "R")):
    cyl("BodyLight_%s" % tg, 0.055, 0.05, (mast_x + ex, -(BODY_L / 2.0 + 0.005), body_cz + 0.18),
        M_LIGHT, axis='Y')
# red accent stripe on the mast head (NASA-ish detail)
box("MastAccent", (0.53, 0.02, 0.03), (mast_x, mast_y - 0.13, mast_top + 0.12), M_RED, bevel=0.0)

# --- front TRIDENT drill (points at the ground) ------------------------------- #
drill_x, drill_y = 0.34, -(BODY_L / 2.0 - 0.04)
cyl("DrillHousing", 0.06, 0.34, (drill_x, drill_y, GC + 0.12), M_MAST, axis='Z')
cyl("DrillBit", 0.028, 0.58, (drill_x, drill_y, 0.31), M_DRILL, axis='Z')  # tip ~z=0.02

# ============================================================================= #
# Join into one mesh, normalize, export USD (Z-up).
# ============================================================================= #
for o in bpy.context.scene.objects:
    o.select_set(o in _objs)
bpy.context.view_layer.objects.active = _objs[0]
bpy.ops.object.join()
rover = bpy.context.active_object
rover.name = "VIPER"

# wrap in an empty named root for a clean prim hierarchy
bpy.ops.object.select_all(action='DESELECT')
rover.select_set(True)
bpy.context.view_layer.objects.active = rover

# report bounds
mn = [1e9, 1e9, 1e9]
mx = [-1e9, -1e9, -1e9]
for v in rover.bound_box:
    wv = rover.matrix_world @ Vector(v)
    for i in range(3):
        mn[i] = min(mn[i], wv[i]); mx[i] = max(mx[i], wv[i])
print(">>> VIPER bounds X[% .3f % .3f] Y[% .3f % .3f] Z[% .3f % .3f]"
      % (mn[0], mx[0], mn[1], mx[1], mn[2], mx[2]))
print(">>> dims (m): W=%.2f L=%.2f H=%.2f" % (mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]))
print(">>> tris=%d" % sum(len(p.vertices) - 2 for p in rover.data.polygons))

bpy.ops.wm.usd_export(
    filepath=out_path,
    selected_objects_only=False,
    export_materials=True,
    evaluation_mode='RENDER',          # bake bevel modifiers
    convert_orientation=False,         # keep Blender Z-up
    export_normals=True,
    root_prim_path="/VIPER",
)
print(">>> exported USD -> %s" % out_path)
