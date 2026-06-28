"""
viper_build.py — author a premium, faithful NASA VIPER (Volatiles Investigating
Polar Exploration Rover) lunar rover in Blender 4.4 (headless) and export to USD.

No CC0/public-domain VIPER 3D model exists for download (verified: NASA 3D
Resources, NASA mission pages, Smithsonian Open Access, Sketchfab CC search,
GitHub). The real lunar rovers that do exist (Lunokhod 1, Yutu) are CC-BY on
Sketchfab and need an authenticated download. So this builds VIPER from
primitives to the documented configuration, fully public-domain (self-authored).

Reference (NASA Science VIPER pages, Planetary Society, Wikipedia):
  - golf-cart sized body (~1.5 x 1.5 m, ~2.45 m tall to the mast top), ~430 kg
  - body wrapped in gold/amber MLI thermal blanket (quilted), black underbody tray
  - 4 corner wheels, 50 cm dia, SPOKED, 2.5 cm grousers, each on an active
    suspension leg with INDEPENDENT steering (4 wheel modules) — NOT 6 wheels
  - VERTICAL solar arrays mounted on the body SIDES (and rear) — VIPER's most
    distinctive feature; vertical because the lunar south-pole sun grazes the
    horizon. Deep-blue cells on a silver grid in a metal frame.
  - tall FRONT mast: stereo NavCam head + NASA's first rover HEADLIGHTS (drives
    into permanently-shadowed craters); low-/high-gain antennas up top
  - front-mounted TRIDENT 1 m rotary-percussive drill, pointing at the ground,
    flanked by the MSolo + NIRVSS + NSS science instruments
  - NO RTG (VIPER is solar-powered)

This v3-viper build is a clear step up from the earlier bare-primitive pass:
spoked-and-grousered wheels, a cell-grid'd 3-sided solar array, quilted MLI
panels, a detailed camera head (lenses + lamps + sun-hood), a fluted TRIDENT
drill, instrument greebles, edge bevels everywhere, and proper PBR materials
(metallic gold foil, deep solar-cell blue, brushed chassis metal, rubber tires).

Frame convention (must match build_lunar_stage_v3._reference_rover_usd):
  +Z up, FRONT faces -Y (mast + drill at -Y end), wheel contact at local z = 0,
  authored in real meters (V3_ROVER_SCALE = 1.0). Z-up USD export.

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
def _set(bsdf, name, val):
    """Set a Principled BSDF input by name if it exists (version-robust)."""
    if name in bsdf.inputs:
        bsdf.inputs[name].default_value = val

def mat(name, base, metallic=0.0, rough=0.5, emiss=None, emiss_str=0.0,
        spec=None, aniso=None):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes.get("Principled BSDF")
    _set(bsdf, "Base Color", (base[0], base[1], base[2], 1.0))
    _set(bsdf, "Metallic", metallic)
    _set(bsdf, "Roughness", rough)
    if spec is not None:
        _set(bsdf, "Specular IOR Level", spec)
    if aniso is not None:
        _set(bsdf, "Anisotropic", aniso)
    if emiss is not None:
        _set(bsdf, "Emission Color", (emiss[0], emiss[1], emiss[2], 1.0))
        _set(bsdf, "Emission Strength", emiss_str)
    return m

# gold MLI foil (warm amber, near-mirror metal w/ slight blanket roughness)
M_GOLD    = mat("VIPER_GoldFoil",   (0.74, 0.54, 0.17), metallic=0.95, rough=0.36, spec=0.6)
M_GOLD_DK = mat("VIPER_GoldSeam",   (0.34, 0.23, 0.07), metallic=0.80, rough=0.50)
M_AMBER   = mat("VIPER_GoldLite",   (0.86, 0.66, 0.27), metallic=0.95, rough=0.30)
M_BLACK   = mat("VIPER_BlackFoil",  (0.040, 0.040, 0.045), metallic=0.35, rough=0.55)
M_TIRE    = mat("VIPER_Tire",       (0.045, 0.045, 0.05), metallic=0.20, rough=0.86)
M_GROUSER = mat("VIPER_Grouser",    (0.13, 0.13, 0.14), metallic=0.65, rough=0.55)
M_HUB     = mat("VIPER_Hub",        (0.79, 0.80, 0.83), metallic=0.95, rough=0.22)
M_SOLAR   = mat("VIPER_SolarCell",  (0.018, 0.035, 0.13), metallic=0.55, rough=0.18,
                spec=0.9, emiss=(0.01, 0.03, 0.10), emiss_str=0.25)
M_GRID    = mat("VIPER_CellGrid",   (0.72, 0.74, 0.78), metallic=0.95, rough=0.28)
M_FRAME   = mat("VIPER_PanelFrame", (0.60, 0.61, 0.65), metallic=0.90, rough=0.30)
M_CHASSIS = mat("VIPER_Chassis",    (0.50, 0.51, 0.55), metallic=0.90, rough=0.34, aniso=0.6)
M_MAST    = mat("VIPER_MastMetal",  (0.56, 0.57, 0.61), metallic=0.88, rough=0.28, aniso=0.4)
M_LENS    = mat("VIPER_Lens",       (0.015, 0.015, 0.02), metallic=0.0,  rough=0.06, spec=1.0)
M_LIGHT   = mat("VIPER_Headlight",  (1.0, 0.97, 0.88),  metallic=0.0,  rough=0.10,
                emiss=(1.0, 0.95, 0.82), emiss_str=7.0)
M_WHITE   = mat("VIPER_Antenna",    (0.82, 0.82, 0.85), metallic=0.10, rough=0.40)
M_DRILL   = mat("VIPER_DrillSteel", (0.30, 0.30, 0.33), metallic=0.88, rough=0.32)
M_GOLDINS = mat("VIPER_InstrGold",  (0.80, 0.60, 0.22), metallic=0.92, rough=0.34)
M_RED     = mat("VIPER_Accent",     (0.58, 0.06, 0.06), metallic=0.20, rough=0.45)

_objs = []

def _finish(obj, material, bevel=0.0, shade_smooth=False, segments=2, angle=40.0):
    obj.data.materials.append(material)
    if bevel > 0.0:
        b = obj.modifiers.new("bev", "BEVEL")
        b.width = bevel
        b.segments = segments
        b.limit_method = 'ANGLE'
        b.angle_limit = math.radians(angle)
    if shade_smooth:
        for poly in obj.data.polygons:
            poly.use_smooth = True
    _objs.append(obj)
    return obj

def box(name, size, loc, material, bevel=0.015, rot=None):
    # base cube has full edge length 1.0; scale by `size` -> real full dims = size.
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=loc)
    o = bpy.context.active_object
    o.name = name
    o.scale = (size[0], size[1], size[2])
    if rot:
        o.rotation_euler = rot
    return _finish(o, material, bevel=bevel)

def cyl(name, r, depth, loc, material, axis='Z', rot=None, bevel=0.006, verts=40):
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

def cone(name, r1, r2, depth, loc, material, axis='Z', verts=28):
    bpy.ops.mesh.primitive_cone_add(radius1=r1, radius2=r2, depth=depth,
                                    location=loc, vertices=verts)
    o = bpy.context.active_object
    o.name = name
    if axis == 'X':
        o.rotation_euler = (0.0, math.radians(90.0), 0.0)
    elif axis == 'Y':
        o.rotation_euler = (math.radians(90.0), 0.0, 0.0)
    return _finish(o, material, bevel=0.0, shade_smooth=True)

def torus(name, major_r, minor_r, loc, material, axis='X', mj=44, mn=12):
    bpy.ops.mesh.primitive_torus_add(location=loc, major_radius=major_r,
                                     minor_radius=minor_r, major_segments=mj,
                                     minor_segments=mn)
    o = bpy.context.active_object
    o.name = name
    if axis == 'X':
        o.rotation_euler = (0.0, math.radians(90.0), 0.0)
    elif axis == 'Y':
        o.rotation_euler = (math.radians(90.0), 0.0, 0.0)
    return _finish(o, material, bevel=0.0, shade_smooth=True)

# ----------------------------------------------------------------------------- #
# Quilted-MLI seam grids + solar cell grids (thin raised detail strips).
# ----------------------------------------------------------------------------- #
def _seam_grid(plane, anchor, u_len, v_len, ncols, nrows, mat, t=0.010,
               proud=0.006, tag="q"):
    """Raised thin grid lines on a flat face. plane in {'X','Y','Z'} = face normal.
    anchor=(cx,cy,cz) centre of the face; u/v span the in-plane axes."""
    cx, cy, cz = anchor
    for i in range(ncols + 1):
        f = -0.5 + i / float(ncols)
        if plane == 'X':       # face normal +/-X; u=Y, v=Z; vertical lines const-Y
            box("%s_v%d" % (tag, i), (0.008, t, v_len), (cx, cy + f * u_len, cz), mat, bevel=0.0)
        elif plane == 'Y':     # u=X, v=Z
            box("%s_v%d" % (tag, i), (t, 0.008, v_len), (cx + f * u_len, cy, cz), mat, bevel=0.0)
        else:                  # plane Z; u=X, v=Y
            box("%s_v%d" % (tag, i), (t, v_len, 0.008), (cx + f * u_len, cy, cz), mat, bevel=0.0)
    for j in range(nrows + 1):
        g = -0.5 + j / float(nrows)
        if plane == 'X':
            box("%s_h%d" % (tag, j), (0.008, u_len, t), (cx, cy, cz + g * v_len), mat, bevel=0.0)
        elif plane == 'Y':
            box("%s_h%d" % (tag, j), (u_len, 0.008, t), (cx, cy, cz + g * v_len), mat, bevel=0.0)
        else:
            box("%s_h%d" % (tag, j), (u_len, t, 0.008), (cx, cy + g * v_len, cz), mat, bevel=0.0)

# ============================================================================= #
# Geometry. +Z up, front = -Y, wheel contact at z = 0, meters.
# ============================================================================= #
GC = 0.36            # ground clearance (belly height)
BODY_W, BODY_L, BODY_H = 1.34, 1.46, 0.92
body_cz = GC + BODY_H / 2.0
body_top = GC + BODY_H

# --- main body: gold MLI blanket box with a black underbody tray -------------- #
box("Body", (BODY_W, BODY_L, BODY_H), (0, 0, body_cz), M_GOLD, bevel=0.05)
box("Belly", (BODY_W * 0.96, BODY_L * 0.96, 0.18), (0, 0, GC + 0.02), M_BLACK, bevel=0.02)
# quilted MLI seams on the exposed gold FRONT face (the sides/back/top are mostly
# hidden behind the solar arrays + deck; the front blanket is the hero-visible one)
_seam_grid('Y', (0, -BODY_L / 2.0 - 0.006, body_top - 0.26), BODY_W * 0.80, 0.50,
           5, 3, M_GOLD_DK, tag="qFront")
# a thin gold MLI margin quilt around the deck edge (where gold shows past the deck)
_seam_grid('Z', (0, 0, body_top + 0.004), BODY_W * 0.96, BODY_L * 0.96,
           6, 5, M_GOLD_DK, tag="qTop")

# top deck plate + avionics greebles
box("Deck", (BODY_W * 0.9, BODY_L * 0.9, 0.06), (0, 0, body_top + 0.03), M_BLACK, bevel=0.01)
box("Avionics", (0.42, 0.5, 0.16), (0.28, 0.34, body_top + 0.12), M_GOLDINS, bevel=0.02)
_seam_grid('Z', (0.28, 0.34, body_top + 0.205), 0.36, 0.44, 3, 3, M_GOLD_DK, tag="qAv")
box("AvBlack", (0.34, 0.3, 0.14), (-0.30, 0.30, body_top + 0.11), M_BLACK, bevel=0.02)
cyl("AvFan", 0.085, 0.05, (-0.30, 0.30, body_top + 0.20), M_HUB, axis='Z', verts=24)
# corner tie-down / lift fittings
for sx in (-1.0, 1.0):
    for sy in (-1.0, 1.0):
        cyl("Tie_%d_%d" % (sx, sy), 0.03, 0.07,
            (sx * (BODY_W / 2.0 - 0.07), sy * (BODY_L / 2.0 - 0.07), body_top + 0.07),
            M_CHASSIS, axis='Z', verts=16)

# --- 4 spoked wheels w/ grousers on steered suspension legs ------------------- #
WR, WW = 0.345, 0.205
wx, wy = 0.70, 0.62

def build_wheel(tag, cx, cy, cz, R, W):
    """Open spoked wheel, spin axis = X. Tire torus + grousers + spokes + hub."""
    minor = W * 0.5
    major = R - minor
    torus("Tire_%s" % tag, major, minor, (cx, cy, cz), M_TIRE, axis='X', mj=48, mn=14)
    torus("WRing_%s" % tag, R * 0.46, 0.02, (cx, cy, cz), M_HUB, axis='X', mj=40, mn=8)
    cyl("Hub_%s" % tag, R * 0.20, W * 1.02, (cx, cy, cz), M_HUB, axis='X', verts=28)
    cyl("HubCap_%s" % tag, R * 0.10, W * 1.16, (cx, cy, cz), M_CHASSIS, axis='X', verts=20)
    # spokes (radiate in the wheel plane; local +Y = radial after X-rotation)
    nsp, s_in, s_out = 10, R * 0.20, R * 0.62
    for k in range(nsp):
        ang = 2.0 * math.pi * k / nsp
        rmid = (s_in + s_out) / 2.0
        b = box("Spoke_%s_%d" % (tag, k), (W * 0.30, (s_out - s_in), 0.024),
                (cx, cy + rmid * math.cos(ang), cz + rmid * math.sin(ang)),
                M_CHASSIS, bevel=0.0)
        b.rotation_euler = (ang, 0.0, 0.0)
    # grousers (axial cleats around the tread; local +Y = radial)
    ngr = 26
    for k in range(ngr):
        ang = 2.0 * math.pi * k / ngr
        rr = R * 0.99
        g = box("Grouser_%s_%d" % (tag, k), (W * 0.92, 0.05, 0.030),
                (cx, cy + rr * math.cos(ang), cz + rr * math.sin(ang)),
                M_GROUSER, bevel=0.004)
        g.rotation_euler = (ang, 0.0, 0.0)

for sx in (-1.0, 1.0):
    for sy in (-1.0, 1.0):
        tag = ("F" if sy < 0 else "R") + ("L" if sx < 0 else "R")  # front = -Y
        wcx, wcy = sx * wx, sy * wy
        build_wheel(tag, wcx, wcy, WR, WR, WW)
        # steering knuckle / drive motor on the inboard wheel face
        cyl("Steer_%s" % tag, 0.075, 0.10, (wcx - sx * (WW / 2.0 + 0.05), wcy, WR),
            M_CHASSIS, axis='X', verts=24)
        # suspension leg: angled strut from hub up to the body corner
        bx, by, bz = sx * (BODY_W / 2.0 - 0.06), sy * (BODY_L / 2.0 - 0.10), GC + 0.06
        mid = ((wcx + bx) / 2.0, (wcy + by) / 2.0, (WR + bz) / 2.0)
        d = Vector((bx - wcx, by - wcy, bz - WR))
        length = d.length
        box("Leg_%s" % tag, (0.075, 0.10, length), mid, M_CHASSIS, bevel=0.012)
        leg = _objs[-1]
        z = Vector((0, 0, 1))
        axis = z.cross(d.normalized())
        ang = math.acos(max(-1.0, min(1.0, z.dot(d.normalized()))))
        if axis.length > 1e-6:
            leg.rotation_mode = 'AXIS_ANGLE'
            leg.rotation_axis_angle = (ang, axis.x, axis.y, axis.z)
        # parallel actuator rod alongside the leg (active suspension read)
        off = Vector((d.y, -d.x, 0.0))
        if off.length > 1e-6:
            off = off.normalized() * 0.09
        rod = box("Strut_%s" % tag, (0.035, 0.05, length * 0.86),
                  (mid[0] + off.x, mid[1] + off.y, mid[2]), M_MAST, bevel=0.0)
        if axis.length > 1e-6:
            rod.rotation_mode = 'AXIS_ANGLE'
            rod.rotation_axis_angle = (ang, axis.x, axis.y, axis.z)
        # shoulder actuator at the body joint
        cyl("Shldr_%s" % tag, 0.075, 0.16, (bx, by, bz), M_HUB, axis='X', verts=24)

# --- VERTICAL solar arrays (VIPER signature) — 3 sides ------------------------ #
PANEL_T, PANEL_L, PANEL_H = 0.040, 1.30, 1.02
panel_cz = body_top + PANEL_H / 2.0 - 0.18      # overlap base into the body sides

def solar_panel_side(tag, sx):
    px = sx * (BODY_W / 2.0 + 0.06)
    box("PanelFrame_%s" % tag, (PANEL_T + 0.02, PANEL_L + 0.05, PANEL_H + 0.05),
        (px, 0.0, panel_cz), M_FRAME, bevel=0.012)
    box("Panel_%s" % tag, (PANEL_T, PANEL_L, PANEL_H), (px + sx * 0.012, 0.0, panel_cz),
        M_SOLAR, bevel=0.0)
    _seam_grid('X', (px + sx * (PANEL_T / 2.0 + 0.022), 0.0, panel_cz),
               PANEL_L * 0.96, PANEL_H * 0.96, 7, 6, M_GRID, t=0.008, tag="cellS_%s" % tag)
    # two support arms body->panel
    for ay in (-0.42, 0.42):
        box("PanelArm_%s_%d" % (tag, int(ay * 10)),
            (0.13, 0.10, 0.12), (sx * (BODY_W / 2.0 - 0.01), ay, body_top - 0.10),
            M_CHASSIS, bevel=0.01)

for sx, tg in ((-1.0, "L"), (1.0, "R")):
    solar_panel_side(tg, sx)

# rear vertical panel (third side, +Y)
RP_W, RP_H = 1.00, 0.84
rp_y = BODY_L / 2.0 + 0.06
rp_z = body_top + RP_H / 2.0 - 0.16
box("PanelFrame_B", (RP_W + 0.05, PANEL_T + 0.02, RP_H + 0.05), (0.0, rp_y, rp_z), M_FRAME, bevel=0.012)
box("Panel_B", (RP_W, PANEL_T, RP_H), (0.0, rp_y + 0.012, rp_z), M_SOLAR, bevel=0.0)
_seam_grid('Y', (0.0, rp_y + PANEL_T / 2.0 + 0.022, rp_z), RP_W * 0.95, RP_H * 0.95,
           6, 5, M_GRID, t=0.008, tag="cellB")
box("PanelArm_B", (0.12, 0.13, 0.10), (0.0, BODY_L / 2.0 - 0.01, body_top - 0.10), M_CHASSIS, bevel=0.01)

# --- antennas: top high-gain dish (tilted) + low-gain whip -------------------- #
cyl("AntStalk", 0.025, 0.34, (-0.12, 0.30, body_top + 0.16), M_MAST, axis='Z', verts=20)
cyl("HGAntenna", 0.20, 0.030, (-0.12, 0.36, body_top + 0.30), M_WHITE, axis='Z',
    rot=(math.radians(18.0), 0.0, math.radians(20.0)), verts=36)
cyl("HGBack", 0.07, 0.05, (-0.12, 0.345, body_top + 0.27), M_CHASSIS, axis='Z',
    rot=(math.radians(18.0), 0.0, math.radians(20.0)), verts=20)
cyl("LGAntenna", 0.010, 0.42, (0.40, 0.30, body_top + 0.30), M_WHITE, axis='Z', verts=12)
cyl("LGTip", 0.022, 0.04, (0.40, 0.30, body_top + 0.50), M_RED, axis='Z', verts=12)

# --- FRONT mast: stereo NavCam head + headlights + sun hood ------------------- #
mast_x, mast_y = 0.0, -(BODY_L / 2.0 - 0.18)
mast_base = body_top + 0.02
mast_h = 1.14
mast_top = mast_base + mast_h
cyl("Mast", 0.052, mast_h, (mast_x, mast_y, mast_base + mast_h / 2.0), M_MAST, axis='Z', verts=28)
cyl("MastCollar", 0.075, 0.10, (mast_x, mast_y, mast_base + 0.06), M_CHASSIS, axis='Z', verts=24)
# pan/tilt camera head (looks -Y = forward)
box("CamHead", (0.54, 0.18, 0.20), (mast_x, mast_y - 0.04, mast_top + 0.03), M_BLACK, bevel=0.025)
box("CamHood", (0.58, 0.06, 0.05), (mast_x, mast_y - 0.15, mast_top + 0.14), M_CHASSIS, bevel=0.01)  # sun visor
for ex, tg in ((-0.17, "L"), (0.17, "R")):
    cyl("NavCam_%s" % tg, 0.050, 0.08, (mast_x + ex, mast_y - 0.13, mast_top + 0.03), M_CHASSIS, axis='Y', verts=24)
    cyl("NavLens_%s" % tg, 0.038, 0.02, (mast_x + ex, mast_y - 0.165, mast_top + 0.03), M_LENS, axis='Y', verts=24)
# headlights on the head (emissive) + reflector cups + lower body lights
for ex, tg in ((-0.24, "L"), (0.24, "R")):
    cone("LampCup_%s" % tg, 0.058, 0.030, 0.05, (mast_x + ex, mast_y - 0.11, mast_top - 0.05), M_HUB, axis='Y')
    cyl("MastLight_%s" % tg, 0.040, 0.03, (mast_x + ex, mast_y - 0.135, mast_top - 0.05), M_LIGHT, axis='Y', verts=20)
for ex, tg in ((-0.42, "L"), (0.42, "R")):
    cyl("BodyLight_%s" % tg, 0.055, 0.05, (mast_x + ex, -(BODY_L / 2.0 + 0.005), body_cz + 0.18), M_LIGHT, axis='Y', verts=20)
# red accent stripe on the mast head (NASA-ish detail)
box("MastAccent", (0.55, 0.02, 0.03), (mast_x, mast_y - 0.135, mast_top + 0.13), M_RED, bevel=0.0)

# --- front TRIDENT drill (1 m, points at the ground) + flutes ----------------- #
drill_x, drill_y = 0.36, -(BODY_L / 2.0 - 0.02)
# deployment rail + carriage
box("DrillRail", (0.09, 0.07, 0.62), (drill_x, drill_y, GC + 0.18), M_CHASSIS, bevel=0.01)
box("DrillMotor", (0.16, 0.16, 0.18), (drill_x, drill_y, GC + 0.10), M_BLACK, bevel=0.02)
cyl("DrillHousing", 0.055, 0.30, (drill_x, drill_y, GC - 0.02), M_MAST, axis='Z', verts=24)
cyl("DrillBit", 0.026, 0.60, (drill_x, drill_y, 0.30), M_DRILL, axis='Z', verts=20)
cone("DrillTip", 0.026, 0.004, 0.06, (drill_x, drill_y, 0.02), M_DRILL, axis='Z')
# helical flute ridges along the auger
for k in range(7):
    zz = 0.06 + k * 0.075
    ang = k * math.radians(55.0)
    fb = box("Flute_%d" % k, (0.07, 0.014, 0.022),
             (drill_x, drill_y, zz), M_DRILL, bevel=0.0)
    fb.rotation_euler = (0.0, 0.0, ang)

# --- science instruments (greebles): NSS, NIRVSS, MSolo ----------------------- #
box("NSS", (0.20, 0.12, 0.12), (-0.34, -(BODY_L / 2.0 + 0.02), GC + 0.16), M_GOLDINS, bevel=0.02)
box("NIRVSS", (0.14, 0.16, 0.14), (0.0, -(BODY_L / 2.0 + 0.03), body_cz - 0.05), M_BLACK, bevel=0.02)
cyl("NIRVSSlens", 0.035, 0.04, (0.0, -(BODY_L / 2.0 + 0.11), body_cz - 0.05), M_LENS, axis='Y', verts=20)
box("MSolo", (0.10, 0.10, 0.16), (drill_x + 0.16, drill_y + 0.02, GC + 0.18), M_HUB, bevel=0.015)

# ============================================================================= #
# Join into one mesh, bake bevels, auto-smooth, normalize, export USD (Z-up).
# ============================================================================= #
# Bake each object's bevel modifier so per-part widths survive the join.
for o in list(_objs):
    if "bev" in [m.name for m in o.modifiers]:
        bpy.ops.object.select_all(action='DESELECT')
        bpy.context.view_layer.objects.active = o
        o.select_set(True)
        try:
            bpy.ops.object.modifier_apply(modifier="bev")
        except RuntimeError:
            try:
                o.modifiers.remove(o.modifiers["bev"])
            except Exception:
                pass

bpy.ops.object.select_all(action='DESELECT')
for o in _objs:
    o.select_set(True)
bpy.context.view_layer.objects.active = _objs[0]
bpy.ops.object.join()
rover = bpy.context.active_object
rover.name = "VIPER"

# angle-based auto-smooth across the whole model (crisp edges, smooth rounds)
bpy.ops.object.select_all(action='DESELECT')
rover.select_set(True)
bpy.context.view_layer.objects.active = rover
try:
    bpy.ops.object.shade_smooth_by_angle(angle=math.radians(33.0))
except Exception as e:
    print(">>> shade_smooth_by_angle unavailable (%s); leaving flat shading" % e)

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
    evaluation_mode='RENDER',
    convert_orientation=False,         # keep Blender Z-up
    export_normals=True,
    root_prim_path="/VIPER",
)
print(">>> exported USD -> %s" % out_path)
