"""v3 placement: put the book at THIS episode's PD-measured grasp point.

The kinematic align proxy has ±3 cm per-episode z error vs the PD-executed tip (arm-config-
dependent droop) — that error is exactly what separated v2's passers (tip at book mid-plane)
from its fails (tip 2-3 cm above the book). Run v2 already measured each episode's true PD tip
trajectory, and the pre-contact tip path is object-independent, so the first-close tip position
is a valid per-episode estimate of where the real book edge was: book edge = tip_x - 2.5 cm
bite, mid-plane at tip_z. Base stays at the v2 parking spot (read from sim_reproduce jsonl).

    conda run -n base python real2sim/teleop/refine_episode.py <uuid>
"""
import json, os, subprocess, sys

import numpy as np

from robot_positions import (
    SOFA_CURRENT_SEAT_TOP_M,
    SOFA_INTERACTION_FACE_SHIFT_XY_M,
    translate_interval,
)

REPO = "/home/aether/workspace/agent_bench"
DEMO = "/home/aether/workspace/genie_sim_demo/genie_sim_demo/demo_sim"
INSTANCE = os.environ.get("INSTANCE", "")   # ""=默认; "_B" 等=并行第二实例, 隔离 scene/cam/traj
SCENE = f"{DEMO}/scene_room{INSTANCE}.usda"
V2 = os.path.join(REPO, "dataset", os.environ.get("REFINE_SRC_JSONL", "sim_reproduce.jsonl.v2"))
MID_BLOCKS_Y = (-1.60, -0.50)


def edge_range_for_mode(fixed_base):
    return translate_interval(
        (-1.02, -0.78) if fixed_base else (-0.99, -0.87),
        SOFA_INTERACTION_FACE_SHIFT_XY_M[0],
    )


EDGE_RANGE = edge_range_for_mode("FIXED_BASE" in os.environ)
# TARGET_PRIM: which sofa prop is grasp-anchored at the PD tip (task 12/13 = Book, task 11 =
# Note); the OTHER one becomes the beside-the-target distractor. Placement geometry (bite,
# 0.125 half-length, riser/cushion offsets) stays at the book-calibrated values for either.
TGT = os.environ.get("TARGET_PRIM", "Book")
DIS = "Note" if TGT == "Book" else "Book"
TP, DP = f"/World/Assets/{TGT}", f"/World/Assets/{DIS}"


def main():
    uid = sys.argv[1]
    mp = f"{REPO}/dataset/{os.environ.get('REFINE_SRC_METRICS', 'sim_metrics_repro')}/{uid[:8]}.npz"
    row = None
    for line in open(V2):
        r = json.loads(line)
        if r["uuid"] == uid:
            row = r
            break
    if not os.path.isfile(mp):
        print(json.dumps({"ok": False, "reason": "no v2 metrics/base for episode"}))
        return 3
    if row is not None and row.get("init", {}).get("ok"):
        base = row["init"]["base"]
    else:
        # episode has no v2 record (skipped back then) but a metrics npz exists from THIS
        # batch's kinematic-init attempt — keep the base where init_episode parked it
        import re as _re
        blk = open(SCENE).read()
        m2 = _re.search(r'def Xform "Robot".*?xformOp:translate = \(([-\d.]+), ([-\d.]+)', blk, _re.S)
        if not m2:
            print(json.dumps({"ok": False, "reason": "no v2 metrics/base for episode"}))
            return 3
        base = [float(m2.group(1)), float(m2.group(2))]
    fb = os.environ.get("FIXED_BASE")
    if fb:
        fx, fy = map(float, fb.split(","))
        dbx, dby = fx - base[0], fy - base[1]
        base = [fx, fy]
    else:
        dbx = dby = 0.0
    m = np.load(mp)
    lg = np.asarray(m["lgrip"], float)
    closed = lg >= 0.9
    if not closed.any():
        print(json.dumps({"ok": False, "reason": "grip never closed in v2 metrics"}))
        return 4
    # the close that starts the LONGEST hold is the real grasp — a first-crossing rule
    # latches onto pre-episode test squeezes (26198524/50071fe7 close for ~1 s at frame ~20
    # while the arm is still tucked at z 0.78)
    runs, s = [], None
    for i, c in enumerate(closed):
        if c and s is None:
            s = i
        elif not c and s is not None:
            runs.append((i - s, s)); s = None
    if s is not None:
        runs.append((len(closed) - s, s))
    t = int(max(runs)[1])
    tx, ty, tz = map(float, m["tip"][t])
    tx += dbx; ty += dby   # the anchor was measured under the source base; tip moves 1:1 with it
    # A promoted historical candidate already records the exact refined PD anchor that
    # produced its result.  Reusing that recorded anchor is more reproducible than
    # re-deriving it from a metrics file that may have since been regenerated.
    pd_tip_override = os.environ.get("PD_TIP_OVERRIDE")
    if pd_tip_override:
        tx, ty, tz = map(float, pd_tip_override.split(","))

    bite = float(os.environ.get("BITE_X", "0.025"))
    ztweak = float(os.environ.get("Z_TWEAK", "0.0"))
    edge = tx - bite
    book_x = edge + 0.125   # book long axis (25 cm) points out; centre is deeper
    book_y = min(max(ty, MID_BLOCKS_Y[0]), MID_BLOCKS_Y[1])
    book_z = tz + ztweak             # mid-plane pinch (+ optional raise)
    RISER_DROP = float(os.environ.get("RISER_DROP", "0.030"))
    cushion_x_offset = float(
        os.environ.get("CUSHION_X_OFFSET", "0.120"))
    if not 0.12 <= cushion_x_offset <= 0.24:
        print(json.dumps({
            "ok": False,
            "reason": "CUSHION_X_OFFSET must be within [0.12, 0.24]",
        }))
        return 6
    distractor_x_offset = float(
        os.environ.get("DISTRACTOR_X_OFFSET", "0.125"))
    distractor_y_offset = float(
        os.environ.get("DISTRACTOR_Y_OFFSET", "0.170"))
    if (
        not 0.06 <= distractor_x_offset <= 0.18
        or not 0.10 <= distractor_y_offset <= 0.24
    ):
        print(json.dumps({
            "ok": False,
            "reason": "distractor offsets are outside reviewed bounds",
        }))
        return 6
    riser_top = book_z - RISER_DROP      # same drop-in convention (target settles onto it)
    riser_h = max(
        0.001,
        riser_top - SOFA_CURRENT_SEAT_TOP_M,
    )   # 0.001 = no pile, book directly on the seat
    if not (EDGE_RANGE[0] < edge < EDGE_RANGE[1]):
        # deflected-arm close points land outside the plausible band; clamp instead of reject —
        # rejecting here starved the ladder's fixed-point rung on exactly the hard episodes
        edge = min(max(edge, EDGE_RANGE[0] + 0.005), EDGE_RANGE[1] - 0.005)
        book_x = edge + 0.125
    if (
        riser_top - SOFA_CURRENT_SEAT_TOP_M > 0.12
        or riser_top - SOFA_CURRENT_SEAT_TOP_M < -0.03
    ):
        riser_top = min(
            max(riser_top, SOFA_CURRENT_SEAT_TOP_M - 0.025),
            SOFA_CURRENT_SEAT_TOP_M + 0.115,
        )
        riser_h = max(
            0.001,
            riser_top - SOFA_CURRENT_SEAT_TOP_M,
        )
        book_z = riser_top + RISER_DROP

    # NOTE_VISION: the note exists/lies where the real f0 RGBD says (phantom-note fix:
    # some real episodes have no notebook at all; ours knocked itself into the distractor gate)
    # (only meaningful when the Note IS the distractor — the query locates the notebook)
    note_line = (
        f"set_t('{DP}', {edge} + {distractor_x_offset}, "
        f"max({book_y} - {distractor_y_offset}, -1.60), 0.43)"
    )
    note_rot = 88
    if os.environ.get("NOTE_VISION") and DIS == "Note":
        import subprocess as _sp
        cache = f"{REPO}/dataset/note_vision/{uid[:8]}.json"
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        if not os.path.isfile(cache):
            CAM = "3e-06,-1.0,-0.0,0.0,0.819156,2e-06,0.573571,0.0,-0.573571,-2e-06,0.819156,0.0,-0.970008,-1.049998,1.03066,1.0"
            _sp.run(f"/home/aether/miniconda3/envs/pgsr/bin/python {REPO}/real2sim/teleop/rgbd_locate_sam3.py "
                    f"/home/aether/workspace/data_collected/record/{uid} 'black notebook' {cache} "
                    f"--cam '{CAM}' --box=-1.15,-0.30,-1.9,-0.4,0.36,0.62 --radius 0.005",
                    shell=True, capture_output=True, timeout=900)
        try:
            nv = json.load(open(cache))
        except Exception:
            nv = {"ok": False}
        if nv.get("ok"):
            nx, ny = nv["centre_xy"]
            note_line = f"set_t('{DP}', {nx}, {ny}, 0.43)"
        else:
            note_line = f"s.GetPrimAtPath('{DP}').SetActive(False)"
    target_yaw = float(os.environ.get("TARGET_YAW", "86"))
    script = f"""import math
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Gf, Sdf
s = Usd.Stage.Open('{SCENE}')
def set_t(path, x, y, z):
    for op in UsdGeom.Xformable(s.GetPrimAtPath(path)).GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            op.Set(Gf.Vec3d(x, y, z)); return
set_t('/World/Robot', {base[0]}, {base[1]}, 0)
rob = s.GetPrimAtPath('/World/Robot')
oa = rob.GetAttribute('xformOp:orient')
if oa and oa.Get() is not None:
    from pxr import Gf as _G
    oa.Set(_G.Quatf(1,0,0,0))
set_t('{TP}', {book_x}, {book_y}, {book_z})
_target = s.GetPrimAtPath('{TP}')
_target.CreateAttribute(
    'physics:kinematicEnabled', Sdf.ValueTypeNames.Bool).Set(False)
for _attribute_name in ('physics:velocity', 'physics:angularVelocity'):
    _attribute = _target.GetAttribute(_attribute_name)
    if _attribute and _attribute.Get() is not None:
        _attribute.Set(Gf.Vec3f(0.0, 0.0, 0.0))
for op in UsdGeom.Xformable(_target).GetOrderedXformOps():
    if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
        op.Set(Gf.Vec3f(0, 0, {target_yaw}))   # preserved promoted-candidate yaw
s.GetPrimAtPath('{DP}').SetActive(True)
for _p in ('{TP}',):
    _pr = s.GetPrimAtPath(_p)
    if _pr and _pr.IsValid() and not _pr.IsActive(): _pr.SetActive(True)
{note_line}
_np = s.GetPrimAtPath('{DP}')
if _np and _np.IsActive():
    for op in UsdGeom.Xformable(_np).GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
            op.Set(Gf.Vec3f(0, 0, 88))
riser = s.GetPrimAtPath('/World/Assets/SeatRiser')
for op in UsdGeom.Xformable(riser).GetOrderedXformOps():
    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
        op.Set(Gf.Vec3d(
            {edge} + 0.06 + 0.12,
            {book_y},
            {SOFA_CURRENT_SEAT_TOP_M} + {riser_h}/2,
        ))
    if op.GetOpType() == UsdGeom.XformOp.TypeScale:
        sc = op.Get(); op.Set(Gf.Vec3f(sc[0], sc[1], {riser_h}))
c = s.GetPrimAtPath('/World/Assets/CushionOnBook')
if not c or not c.IsValid():
    raise RuntimeError('CushionOnBook is missing')
# Family initializers may deactivate every unrelated pick prop. Sofa tasks
# require this secondary object, so restore it before asking USD for a bound.
c.SetActive(True)
UsdPhysics.RigidBodyAPI.Apply(c).CreateRigidBodyEnabledAttr().Set(True)
UsdPhysics.RigidBodyAPI(c).CreateKinematicEnabledAttr().Set(False)
UsdPhysics.MassAPI.Apply(c).CreateMassAttr().Set(0.35)
for _attribute_name in ('physics:velocity', 'physics:angularVelocity'):
    _attribute = c.GetAttribute(_attribute_name)
    if _attribute and _attribute.Get() is not None:
        _attribute.Set(Gf.Vec3f(0.0, 0.0, 0.0))
_cushion_translate = None
_cushion_rotation = None
for op in UsdGeom.Xformable(c).GetOrderedXformOps():
    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
        _cushion_translate = op
        _cushion_translate.Set(Gf.Vec3d(0.0, 0.0, 0.0))
    elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
        # The current shared CushionOnBook asset is authored flat with its
        # thin axis in local Z.  The legacy asset needed an X=90 correction;
        # retaining that correction now stands the 0.46 x 0.36 x 0.10 m
        # pillow on edge and correctly trips the pillow-scale gate below.
        _canonical_quat = Gf.Quatd(1.0, Gf.Vec3d(0.0, 0.0, 0.0))
        op.Set(Gf.Quatf(
            float(_canonical_quat.GetReal()),
            Gf.Vec3f(*_canonical_quat.GetImaginary())))
        _cushion_rotation = op
    elif op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
        op.Set(Gf.Vec3f(0.0, 0.0, 0.0))
        _cushion_rotation = op
if _cushion_translate is None:
    raise RuntimeError('CushionOnBook has no translate op')
if _cushion_rotation is None:
    raise RuntimeError('CushionOnBook has no supported rotation op')
b = UsdGeom.Imageable(c).ComputeWorldBound(0, UsdGeom.Tokens.default_).ComputeAlignedBox()
mn, mx = b.GetMin(), b.GetMax()
cur = [(mn[i]+mx[i])/2 for i in range(3)]
if not all(
        math.isfinite(float(value))
        for value in (*mn, *mx, *cur)
    ):
    raise RuntimeError('CushionOnBook has a non-finite composed bound')
_cushion_size = [float(mx[i] - mn[i]) for i in range(3)]
if not (
    0.35 <= _cushion_size[0] <= 0.60
    and 0.35 <= _cushion_size[1] <= 0.60
    and 0.08 <= _cushion_size[2] <= 0.22
):
    raise RuntimeError(
        f'CushionOnBook size is not pillow-scale: {{_cushion_size}}')
tz2 = ({riser_top} + 0.05) + (cur[2]-mn[2])
dx, dy, dz = ({book_x}+{cushion_x_offset})-cur[0], {book_y}-cur[1], tz2-cur[2]
_cushion_translate.Set(Gf.Vec3d(dx, dy, dz))

# Override only the visible cushion's surface. A stronger ancestor binding
# preserves its reviewed mesh/collision while matching the real blue-grey
# striped secondary object.
_material = UsdShade.Material.Define(
    s, '/World/Looks/Real2SimBlueGrayCushion')
_surface = UsdShade.Shader.Define(
    s, '/World/Looks/Real2SimBlueGrayCushion/Surface')
_surface.CreateIdAttr('UsdPreviewSurface')
_surface.CreateInput(
    'roughness', Sdf.ValueTypeNames.Float).Set(0.82)
_surface.CreateInput(
    'metallic', Sdf.ValueTypeNames.Float).Set(0.0)
_reader = UsdShade.Shader.Define(
    s, '/World/Looks/Real2SimBlueGrayCushion/PrimvarReader')
_reader.CreateIdAttr('UsdPrimvarReader_float2')
_reader.CreateInput(
    'varname', Sdf.ValueTypeNames.Token).Set('st0')
_reader.CreateOutput('result', Sdf.ValueTypeNames.Float2)
_texture = UsdShade.Shader.Define(
    s, '/World/Looks/Real2SimBlueGrayCushion/Texture')
_texture.CreateIdAttr('UsdUVTexture')
_texture.CreateInput(
    'file', Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath('./usdz/cushion_blue_gray_stripes.png'))
_texture.CreateInput(
    'sourceColorSpace', Sdf.ValueTypeNames.Token).Set('sRGB')
_texture.CreateInput(
    'wrapS', Sdf.ValueTypeNames.Token).Set('repeat')
_texture.CreateInput(
    'wrapT', Sdf.ValueTypeNames.Token).Set('repeat')
_texture.CreateInput(
    'st', Sdf.ValueTypeNames.Float2).ConnectToSource(
        _reader.ConnectableAPI(), 'result')
_texture.CreateOutput('rgb', Sdf.ValueTypeNames.Float3)
_surface.CreateInput(
    'diffuseColor', Sdf.ValueTypeNames.Color3f).ConnectToSource(
        _texture.ConnectableAPI(), 'rgb')
_surface.CreateOutput('surface', Sdf.ValueTypeNames.Token)
_material.CreateSurfaceOutput().ConnectToSource(
    _surface.ConnectableAPI(), 'surface')
UsdShade.MaterialBindingAPI.Apply(c).Bind(
    _material,
    UsdShade.Tokens.strongerThanDescendants,
)
_final_bound = UsdGeom.Imageable(
    c).ComputeWorldBound(
        0, UsdGeom.Tokens.default_).ComputeAlignedBox()
_final_min, _final_max = _final_bound.GetMin(), _final_bound.GetMax()
_final_center = [
    float((_final_min[i] + _final_max[i]) / 2)
    for i in range(3)
]
if not all(math.isfinite(value) for value in _final_center):
    raise RuntimeError('final CushionOnBook pose is non-finite')
print('placed')
s.GetRootLayer().Save()
"""
    open("/tmp/_re.py", "w").write(script)
    r2 = subprocess.run("conda run -n mesh311 python /tmp/_re.py", shell=True,
                        capture_output=True, text=True, timeout=120)
    ok = "placed" in r2.stdout
    result = {"ok": ok, "hand": "l", "target": TGT,
                      "track_prim": TP,
                      "pd_tip": [round(tx, 3), round(ty, 3), round(tz, 3)],
                      "base": base, "book": [round(book_x, 3), round(book_y, 3), round(book_z, 3)],
                      "riser_top": round(riser_top, 3),
                      "distractor_offset_m": [
                          round(distractor_x_offset, 3),
                          round(-distractor_y_offset, 3),
                      ],
                      "secondary_object": {
                          "prim": "/World/Assets/CushionOnBook",
                          "active": True,
                          "appearance": "blue_gray_stripes",
                          "dynamic": True,
                          "size_m": [0.449, 0.450, 0.130],
                          "center_offset_from_book_x_m": round(
                              cushion_x_offset, 3),
                      }}
    if not ok:
        result["reason"] = (r2.stdout + r2.stderr)[-1200:]
    print(json.dumps(result))
    return 0 if ok else 7


if __name__ == "__main__":
    sys.exit(main())
