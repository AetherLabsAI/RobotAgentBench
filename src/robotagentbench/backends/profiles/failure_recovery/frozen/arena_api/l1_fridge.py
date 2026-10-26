"""L1 task95 refrigerator assistance, restored to the supplied recovery source.

Execute the original simple latch unchanged: position hold while open and two
reverse-speed samples to close, with no extra exterior-contact release gate.
The optional retract/close window mirrors lreach_impl.py's trig45=6 scope;
it must be entered/exited by that task's execution bridge, not inferred from
wall time, a gripper command, or a network-provided threshold. The generic
worker does not enter this window automatically. Docking remains disabled.
No legacy playback/server or hand/object attachment commands are loaded.
"""
import ast
import json
import math
from types import SimpleNamespace

from . import l1_scene
from .physics_defaults import PHYSICS_DT

ORIGINAL = l1_scene.VENDOR / "original_fridge"
SOURCE = ORIGINAL / "fridge_only.py"
FUNCTIONS = (
    "_quat_mul", "_quat_inv", "_quat_rot", "_ref_pose_wxyz", "_door_pose_wxyz",
    "_docks_step", "_tray_dock_step",
    "_fridge_pose", "_fr_on_contact",
    "_fr_lcol", "_fr_boxbody_filter", "_fr_latch_simple", "_fr_latch_step", "_fr_rcol",
)


def original_material_blocks():
    """Only the five shared material statements, without the old MW server."""
    return _source_blocks()["define_door_material"]


def _source_blocks():
    tree = ast.parse(SOURCE.read_text())
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    expected = {*FUNCTIONS, "define_door_material", "initialize_fridge", "tune_fridge"}
    if set(functions) != expected or any(not isinstance(n, (ast.FunctionDef, ast.Expr))
                                        for n in tree.body):
        raise ValueError("Unexpected code in the refrigerator-only source")
    return {name: node.body for name, node in functions.items()} | {"functions": [
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in FUNCTIONS]}


def verify_material(stage):
    """Fail at initialization, instead of silently binding a nonexistent material."""
    from pxr import Usd, UsdPhysics, PhysxSchema
    path = "/World/PhysicsMaterials/DoorMat"
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.HasAPI(UsdPhysics.MaterialAPI):
        raise ValueError("Original L1 DoorMat is missing")
    material = UsdPhysics.MaterialAPI(prim)
    static = material.GetStaticFrictionAttr().Get()
    dynamic = material.GetDynamicFrictionAttr().Get()
    combine = PhysxSchema.PhysxMaterialAPI(prim).GetFrictionCombineModeAttr().Get()
    if not (math.isclose(static, 2.5, abs_tol=1e-6)
            and math.isclose(dynamic, 2.2, abs_tol=1e-6) and combine == "max"):
        raise ValueError("Original L1 DoorMat parameters differ")
    bindings = []
    for collider in Usd.PrimRange(stage.GetPrimAtPath(l1_scene.FRIDGE)):
        targets = collider.GetRelationship("material:binding:physics").GetTargets()
        if path in [str(p) for p in targets]:
            bindings.append(str(collider.GetPath()))
    for side, collider in (("left_door", "col_panel"), ("right_door", "col_handle"),
                           ("right_door", "col_door")):
        if f"{l1_scene.FRIDGE}/{side}/{collider}" not in bindings:
            raise ValueError("Original L1 door material binding is missing")
    return dict(path=path, static_friction=static, dynamic_friction=dynamic,
                combine_mode=combine, bindings=sorted(bindings))


def original_blocks():
    blocks = _source_blocks()
    return blocks["initialize_fridge"], blocks["functions"], blocks["tune_fridge"]


def _execute(nodes, namespace):
    exec(compile(ast.Module(body=nodes, type_ignores=[]),
                 str(SOURCE) + " [L1 refrigerator blocks]", "exec"),
         namespace)


class Controller:
    def __init__(self, stage, physics_dt=PHYSICS_DT):
        import numpy as np
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, PhysxSchema, PhysicsSchemaTools
        from isaacsim.core.prims import SingleRigidPrim
        from omni.physx import get_physx_simulation_interface
        self.period = round((1 / 60) / physics_dt)
        if self.period < 1 or not math.isclose(self.period * physics_dt, 1 / 60, abs_tol=1e-10):
            raise ValueError("Original L1 controller needs an exact 60-Hz clock")
        self.steps = 0
        self.errors = []
        self.phase_events = []
        self._phase_saved_trigger = None
        self._closed = False
        environment = json.loads((ORIGINAL / "environment.json").read_text())
        def original_print(*args, **kwargs):
            message = " ".join(map(str, args))
            if "simple_latch 异常" in message or "[tray_dock] 跟随异常" in message:
                self.errors.append(message)
            print(*args, **kwargs)
        ns = dict(stage=stage, PHYS=True, np=np, math=math,
                  os=SimpleNamespace(environ=environment), Gf=Gf, Sdf=Sdf, Usd=Usd,
                  UsdGeom=UsdGeom, UsdPhysics=UsdPhysics, PhysxSchema=PhysxSchema,
                  PhysicsSchemaTools=PhysicsSchemaTools, SingleRigidPrim=SingleRigidPrim,
                  print=original_print, FR_TUNE={}, DOCKS=[], TRAY_DOCK={"on": False},
                  FRIDGE_ROOT_A=l1_scene.FRIDGE)
        frame = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(l1_scene.FRIDGE))
        axis = frame.TransformDir(Gf.Vec3d(1, 0, 0))
        ns["FRIDGE_YAW0"] = math.degrees(math.atan2(axis[1], axis[0]))
        init, functions, tune = original_blocks()
        _execute(original_material_blocks(), ns)
        _execute(init, ns)
        self.material = verify_material(stage)
        _execute(functions, ns)
        ns["c"] = {"cmd": "fr_tune", **json.loads((ORIGINAL / "fr_tune.json").read_text())}
        _execute(tune, ns)
        del ns["c"]
        self.ns = ns
        self.subscription = get_physx_simulation_interface().subscribe_contact_report_events(ns["_fr_on_contact"])
        ns["FR_CT"]["sub"] = self.subscription
        ns["_fr_boxbody_filter"](True)
        # Only these original helpers may change collider flags. All others stay guarded.
        self.mutable_colliders = (
            {str(p.GetPath()) for p in ns["FR_RCOL"]}
            if ns["FR_TUNE"].get("open_keepcol", 0.0) < 0.5 else set())
        self.mutable_colliders.update(l1_scene.FRIDGE + "/left_door/" + p for p in ("col_panel", "col_door"))
        self.events = []
        self.last_zone = ns["FR_LATCH"]["zone"]
        self.settings = dict(environment=environment, tune=dict(ns["FR_TUNE"]),
                             initial_latch=dict(ns["FR_LATCH"]))

    def set_retract_close_window(self, active: bool):
        """Mirror L1 chain's post-grip try/finally window (lines 3593–3804).

        Default stays at the configured 8 deg/s. Enter AFTER the grip/lift
        attempt, before retraction; leave after closing or on any abort. Repeated
        entry must not replace the saved baseline with 6. No arbitrary tuning
        input is exposed, no object/robot pose or collision setting is changed.
        Until the refrigerator task bridge supplies these boundaries, this is
        inactive; merely opening the door must not lower the threshold.
        """
        if type(active) is not bool:
            raise TypeError("The L1 retract/close window accepts only a boolean")
        if self._closed:
            raise RuntimeError("Refrigerator controller is closed")
        if active == (self._phase_saved_trigger is not None):
            return
        state = self.ns["FR_LATCH"]
        if active:
            original = float(state["trig"])
            if not math.isfinite(original) or original <= 0:
                raise ValueError("Invalid configured refrigerator trigger")
            self._phase_saved_trigger = original
            state["trig"] = min(original, 6.0)  # Original lreach trig45 default.
        else:
            state["trig"] = self._phase_saved_trigger
            self._phase_saved_trigger = None
        self.phase_events.append(dict(tick=self.steps // self.period,
            retract_close_active=active, trigger_deg_s=state["trig"]))

    def validate_physics_dt(self, physics_dt):
        period = round((1 / 60) / physics_dt)
        if period < 1 or not math.isclose(period * physics_dt, 1 / 60, abs_tol=1e-10):
            raise ValueError('Original L1 controller needs an exact 60-Hz clock')
        if self.steps % self.period:
            raise ValueError('Rate switch must align with the L1 helper tick')
        return period

    def set_physics_dt(self, physics_dt):
        period = self.validate_physics_dt(physics_dt)
        ticks = self.steps // self.period
        self.period = period
        self.steps = ticks * period  # Preserve helper time and event tick indices.

    def after_physics_step(self):
        self.steps += 1
        if self.steps % self.period:
            return
        if self.ns["_fridge_pose"]("right")[2] is None:
            raise RuntimeError("Original L1 controller cannot read the physical door")
        self.ns["_fr_latch_step"]()
        if self.errors or self.ns["FR_CT"]["err"]:
            raise RuntimeError("Original L1 controller failed: " + str(self.errors or self.ns["FR_CT"]["err"]))
        zone = self.ns["FR_LATCH"]["zone"]
        if zone != self.last_zone:
            self.events.append(dict(tick=self.steps // self.period, zone=zone,
                                    angle_deg=self.ns["_fridge_pose"]("right")[2]))
            self.last_zone = zone

    def report(self):
        return dict(policy_id="house.l1_task95_original_assistance.v3", controller_hz=60,
                    algorithm="original_l1_simple_latch", settings=self.settings,
                    open_hold=dict(mode="position_drive",
                        active=self.ns["FR_LATCH"]["zone"] == "open",
                        requires_exterior_contact=False),
                    retract_close_window=dict(active=self._phase_saved_trigger is not None,
                        saved_trigger_deg_s=self._phase_saved_trigger,
                        current_trigger_deg_s=self.ns["FR_LATCH"]["trig"],
                        events=list(self.phase_events)),
                    door_material=dict(self.material),
                    state=dict(self.ns["FR_LATCH"]), events=list(self.events),
                    docked_object_count=len(self.ns["DOCKS"]),
                    tray_dock_enabled=bool(self.ns["TRAY_DOCK"]["on"]),
                    assisted_environment=True,
                    mutable_collision_paths=sorted(self.mutable_colliders))

    def close(self):
        if self._closed:
            return
        self.set_retract_close_window(False)
        if self.ns.get("FR_CT"):
            self.ns["FR_CT"]["sub"] = None
        self.subscription = None
        self._closed = True
