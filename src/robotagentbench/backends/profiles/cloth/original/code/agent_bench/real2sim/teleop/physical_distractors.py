"""Physical clutter contract shared by scene setup, live playback and rendering.

No pose driver or contact controller lives here. The owner's controller continues
to execute actions in PhysX; this module validates bodies and records contacts.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

SCHEMA = "real2sim.physical_distractors.v1"


def static_scene_snapshot(stage):
    """Bind collision supports/layout to physics, independently of appearance."""
    from pxr import UsdGeom, UsdPhysics
    rows = {}
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if path.startswith("/World/Robot/") or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
            continue
        ancestor = prim
        while ancestor and not ancestor.IsPseudoRoot():
            if ancestor.HasAPI(UsdPhysics.RigidBodyAPI):
                break
            ancestor = ancestor.GetParent()
        else:
            ancestor = None
        if ancestor:
            continue
        matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
        rows[path] = [list(map(float, row)) for row in matrix]
    return rows


def require_same_static_scene(stage, expected):
    import numpy as np
    current = static_scene_snapshot(stage)
    if (not expected or set(current) != set(expected) or any(
            not np.allclose(current[p], expected[p], rtol=0, atol=1e-6) for p in expected)):
        raise ValueError("requires_physics_replay: collision support/layout changed after physical capture")


def ensure_collision_geometry(stage, path):
    """Add one convex hull from visible mesh vertices only when none exists.

    Some shelf packages contain separate textured faces with no collider. A hull
    over all faces gives them volume; individual flat-face hulls do not. Existing
    reviewed collision geometry is retained verbatim.
    """
    import numpy as np
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    root = stage.GetPrimAtPath(path)
    if not root or not root.IsActive():
        raise ValueError(f"missing physical prop: {path}")
    children = list(Usd.PrimRange(root, Usd.TraverseInstanceProxies()))
    if any(p.HasAPI(UsdPhysics.CollisionAPI) and p.IsA(UsdGeom.Gprim)
           and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() is not False for p in children):
        return None
    analytic = [p for p in children if any(p.IsA(t) for t in
        (UsdGeom.Cube, UsdGeom.Sphere, UsdGeom.Capsule, UsdGeom.Cylinder, UsdGeom.Cone))
        and UsdGeom.Imageable(p).ComputeVisibility() != "invisible"]
    if analytic:
        for prim in analytic:
            UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
        # Mixed mesh/analytic assets must not silently omit the mesh volume.
        if not any(p.IsA(UsdGeom.Mesh) and UsdGeom.Imageable(p).ComputeVisibility() != "invisible"
                   for p in children):
            return str(analytic[0].GetPath())
    inverse = UsdGeom.Xformable(root).ComputeLocalToWorldTransform(0).GetInverse()
    points, counts, indices = [], [], []
    for prim in children:
        if not prim.IsA(UsdGeom.Mesh) or UsdGeom.Imageable(prim).ComputeVisibility() == "invisible":
            continue
        mesh = UsdGeom.Mesh(prim)
        transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0) * inverse
        offset = len(points)
        points.extend(tuple(transform.Transform(Gf.Vec3d(*p))) for p in mesh.GetPointsAttr().Get())
        counts.extend(mesh.GetFaceVertexCountsAttr().Get())
        indices.extend(int(i) + offset for i in mesh.GetFaceVertexIndicesAttr().Get())
    values = np.asarray(points, dtype=float)
    if (len(points) < 4 or not np.isfinite(values).all()
            or np.linalg.matrix_rank(values - values.mean(axis=0), tol=1e-7) < 3):
        raise ValueError(f"prop has no volumetric mesh for physical collision: {path}")
    hull = UsdGeom.Mesh.Define(stage, path + "/RandomizedCollisionHull")
    hull.CreatePointsAttr(points)
    hull.CreateFaceVertexCountsAttr(counts)
    hull.CreateFaceVertexIndicesAttr(indices)
    hull.CreateVisibilityAttr("invisible")
    UsdPhysics.CollisionAPI.Apply(hull.GetPrim()).CreateCollisionEnabledAttr(True)
    UsdPhysics.MeshCollisionAPI.Apply(hull.GetPrim()).CreateApproximationAttr("convexHull")
    return str(hull.GetPath())


def ensure_scene_distractor_collisions(stage, paths, *, author=True):
    """Ensure every declared distractor is collidable, including visual extras.

    Static decoration may remain static; movable props retain their existing
    body parameters. This does not invent physical replay evidence.
    """
    from pxr import Usd, UsdGeom, UsdPhysics
    rows = []
    for path in sorted(set(map(str, paths))):
        root = stage.GetPrimAtPath(path)
        if not root or not root.IsActive():
            raise ValueError(f"missing declared scene distractor: {path}")
        if author:
            # A disabled rigid body can suppress its descendant collision shapes.
            # Decoration without a live body is static collision geometry.
            for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
                if (prim.HasAPI(UsdPhysics.RigidBodyAPI)
                        and UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get() is False):
                    prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
            ensure_collision_geometry(stage, path)
        shapes = []
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if (prim.HasAPI(UsdPhysics.RigidBodyAPI)
                    and UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get() is False):
                raise ValueError(f"distractor has disabled rigid body: {prim.GetPath()}")
            filtered = prim.GetRelationship("physics:filteredPairs")
            if filtered and any(str(p) == "/World" or str(p).startswith("/World/Robot")
                                for p in filtered.GetTargets()):
                raise ValueError(f"distractor filters robot collision: {prim.GetPath()}")
            if prim.IsA(UsdGeom.Gprim) and prim.HasAPI(UsdPhysics.CollisionAPI):
                if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False:
                    shapes.append(str(prim.GetPath()))
        if not shapes:
            raise ValueError(f"distractor lacks collision geometry: {path}")
        rows.append({"prim_path": path, "collision_paths": shapes})
    return rows


def validate_bodies(stage, paths):
    from pxr import Usd, UsdGeom, UsdPhysics
    paths = list(map(str, paths))
    if not paths or len(paths) != len(set(paths)):
        raise ValueError("physical distractor paths must be nonempty and unique")
    rows = []
    for path in paths:
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsActive():
            raise ValueError(f"physical distractor missing/inactive: {path}")
        body = UsdPhysics.RigidBodyAPI(prim)
        if (not prim.HasAPI(UsdPhysics.RigidBodyAPI)
                or body.GetRigidBodyEnabledAttr().Get() is False
                or body.GetKinematicEnabledAttr().Get() is True):
            raise ValueError(f"distractor must be a dynamic rigid body: {path}")
        ancestor = prim.GetParent()
        while ancestor and not ancestor.IsPseudoRoot():
            if (ancestor.HasAPI(UsdPhysics.RigidBodyAPI)
                    and UsdPhysics.RigidBodyAPI(ancestor).GetRigidBodyEnabledAttr().Get() is not False):
                raise ValueError(f"distractor has an enabled rigid ancestor: {path}")
            ancestor = ancestor.GetParent()
        shapes = []
        for child in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
            filtered = child.GetRelationship("physics:filteredPairs")
            if filtered and any(str(p) == "/World" or str(p).startswith("/World/Robot")
                                for p in filtered.GetTargets()):
                raise ValueError(f"distractor filters robot contact: {child.GetPath()}")
            if child != prim and child.HasAPI(UsdPhysics.RigidBodyAPI):
                nested = UsdPhysics.RigidBodyAPI(child)
                if nested.GetRigidBodyEnabledAttr().Get() is not False:
                    raise ValueError(f"nested distractor rigid body: {child.GetPath()}")
            if (child.HasAPI(UsdPhysics.CollisionAPI)
                    and UsdPhysics.CollisionAPI(child).GetCollisionEnabledAttr().Get() is not False
                    and child.IsA(UsdGeom.Gprim)):
                shapes.append(str(child.GetPath()))
        if not shapes:
            raise ValueError(f"distractor has no enabled collision geometry: {path}")
        rows.append(dict(prim_path=path, dynamic=True, collision_paths=shapes))
    return rows


def read_capture(state):
    """Reject old visual-only samples rather than inventing physical evidence."""
    import numpy as np
    key = "physical_distractor_manifest_json"
    if key not in state:
        raise ValueError("requires_physics_replay: no live distractor capture; randomize before executing actions")
    report = json.loads(str(np.asarray(state[key]).item()))
    if (report.get("schema") != SCHEMA or report.get("physics_rollout_performed") is not True
            or report.get("state_pose_overrides_during_actions") is not False):
        raise ValueError("requires_physics_replay: invalid physical distractor evidence")
    objects = report.get("objects", [])
    paths = [str(o["prim_path"]) for o in objects]
    captured = np.asarray(state["rigid_body_paths"]).astype(str).tolist()
    frames = len(state["frame_index"])
    planned = report.get("plan", {}).get("objects", [])
    if (not paths or len(paths) != len(set(paths)) or not set(paths) <= set(captured)
            or len(captured) != len(set(captured))
            or {p.get("prim_path") for p in planned} != set(paths)
            or report.get("frames") != frames or frames < 1
            or not np.array_equal(state["frame_index"], np.arange(frames))
            or any(o.get("dynamic") is not True or not o.get("collision_paths") for o in objects)):
        raise ValueError("requires_physics_replay: distractor state coverage is incomplete")
    xyz, quat = np.asarray(state["prop_xyz_m"]), np.asarray(state["prop_wxyz"])
    if (xyz.shape != (frames, len(captured), 3) or quat.shape != (frames, len(captured), 4)
            or not np.isfinite(xyz).all() or not np.isfinite(quat).all()
            or np.any(np.linalg.norm(quat, axis=2) < 1e-8)):
        raise ValueError("invalid physical distractor state arrays")
    return report


class ContactRecorder:
    """Observe each physics substep; never write transforms or joint state."""
    def __init__(self, stage, paths, plan=None):
        from pxr import PhysxSchema
        from omni.physx import get_physx_simulation_interface
        self.stage = stage
        self.objects = validate_bodies(stage, paths)
        self.paths = set(map(str, paths))
        self.plan = plan or {}
        self.static_scene = static_scene_snapshot(stage)
        self.frame = -1
        self.events = {}
        self.error = None
        for path in self.paths:
            PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath(path)).CreateThresholdAttr(0.0)
        self.subscription = get_physx_simulation_interface().subscribe_contact_report_events(self._on_contact)

    def _on_contact(self, headers, data):
        from pxr import PhysicsSchemaTools
        try:
            for h in headers:
                a = str(PhysicsSchemaTools.intToSdfPath(h.actor0))
                b = str(PhysicsSchemaTools.intToSdfPath(h.actor1))
                if a not in self.paths and b not in self.paths:
                    continue
                for i in range(h.contact_data_offset, h.contact_data_offset + h.num_contact_data):
                    c = data[i]
                    impulse = [float(v) for v in c.impulse]
                    if not all(math.isfinite(v) for v in impulse):
                        raise ValueError("nonfinite PhysX contact impulse")
                    key = (self.frame, a, b)
                    row = self.events.setdefault(key, dict(frame=self.frame, actor0=a, actor1=b,
                        contact_points=0, impulse_ns=[0., 0., 0.], impulse_norm_sum_ns=0.,
                        minimum_separation_m=float(c.separation)))
                    row["contact_points"] += 1
                    row["impulse_ns"] = [x+y for x,y in zip(row["impulse_ns"], impulse)]
                    row["impulse_norm_sum_ns"] += math.sqrt(sum(v*v for v in impulse))
                    row["minimum_separation_m"] = min(row["minimum_separation_m"], float(c.separation))

        except Exception as exc:
            self.error = str(exc)

    def finish(self, frames, captured_paths):
        if self.error:
            raise RuntimeError("contact capture failed: " + self.error)
        final = validate_bodies(self.stage, sorted(self.paths))
        if not self.paths <= set(captured_paths):
            raise RuntimeError("full state omitted simulated distractors")
        robot_events = [e for e in self.events.values() if e["frame"] >= 0 and
                        any(e[k].startswith("/World/Robot/") for k in ("actor0", "actor1"))]
        return dict(schema=SCHEMA, physics_rollout_performed=True,
            state_pose_overrides_during_actions=False, frames=int(frames),
            objects=final, plan=self.plan, contacts=list(self.events.values()),
            contact_aggregation="frame_actor_pair_over_all_physics_substeps",
            static_collision_transforms=self.static_scene,
            robot_contact_points=sum(e["contact_points"] for e in robot_events),
            robot_contact_observed=bool(robot_events),
            # Zero is valid when the action never reaches any distractor.
            robot_contact_impulse_norm_sum_ns=sum(e["impulse_norm_sum_ns"] for e in robot_events))


class RobotContactRecorder(ContactRecorder):
    """Record every robot/environment and self-contact over all substeps."""
    def __init__(self, stage):
        from pxr import UsdPhysics, PhysxSchema
        from omni.physx import get_physx_simulation_interface
        self.paths = {str(p.GetPath()) for p in stage.Traverse()
                      if str(p.GetPath()).startswith("/World/Robot/")
                      and p.HasAPI(UsdPhysics.RigidBodyAPI)}
        if not self.paths:
            raise ValueError("robot contact capture has no rigid bodies")
        self.frame, self.events, self.error = -1, {}, None
        for path in self.paths:
            PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath(path)).CreateThresholdAttr(0.0)
        self.subscription = get_physx_simulation_interface().subscribe_contact_report_events(self._on_contact)
