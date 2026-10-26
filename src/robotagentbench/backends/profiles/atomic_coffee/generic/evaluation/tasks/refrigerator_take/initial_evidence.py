"""Read-only task 9.1 initial measurements; no scoring, reset or physics writes.

Use after World.reset, sample after each physics step. This module does NOT
register itself, enable contact reports, change thresholds, advance simulation,
or infer stability from a single pose. The existing L1 right-door reporter is
required. Missing current reports are unknown, not proof of zero contact.
"""
import math
from pathlib import Path

from .initial_state import FRIDGE, ROBOT, TARGET, TASK_DIR

BODIES = {"target": TARGET, "left_door": FRIDGE + "/left_door",
          "right_door": FRIDGE + "/right_door"}
SLOT_FLOOR = FRIDGE + "/right_door/col_bin34_floor"


def under(path, root):
    return path == root or path.startswith(root + "/")


def finite_vector(value, length):
    result = [float(x) for x in value]
    if len(result) != length or not all(math.isfinite(x) for x in result):
        raise ValueError("Missing or nonfinite measured vector")
    return result


def measured_pose(body):
    transforms, velocities = body.get_transforms(), body.get_velocities()
    if len(transforms) != 1 or len(velocities) != 1:
        raise ValueError("Expected exactly one measured body")
    pose = finite_vector(transforms[0], 7)
    velocity = finite_vector(velocities[0], 6)
    position, quaternion = pose[:3], [pose[6], *pose[3:6]]
    # Preserve measured quaternion values, but reject invalid zero rotations.
    if sum(x*x for x in quaternion) == 0:
        raise ValueError("Zero measured quaternion")
    return {"position_m": position, "quaternion_wxyz": quaternion,
            "linear_velocity_m_s": velocity[:3],
            "angular_velocity_rad_s": velocity[3:]}


def contact_summary(rows):
    """Summarize CURRENT reports, never carry old impulses into sleeping frames."""
    pairs = {}
    for row in rows:
        a, b = row["colliders"]
        if not all(isinstance(p, str) and p.startswith("/") for p in (a, b)):
            raise ValueError("Invalid collider identity")
        hit = [under(a, TARGET), under(b, TARGET)]
        if not any(hit):
            continue
        if all(hit):
            raise ValueError("Unexpected target self-contact")
        other = b if hit[0] else a
        normal = finite_vector(row["normal"], 3)
        impulse = finite_vector(row["impulse_Ns"], 3)
        separation = float(row["separation_m"])
        if not math.isfinite(separation):
            raise ValueError("Nonfinite separation")
        sign = 1 if hit[0] else -1  # PhysX direction: collider1 -> collider0.
        item = pairs.setdefault(other, {
            "other_collider": other,
            "role": ("slot_floor" if other == SLOT_FLOOR else
                     "robot" if under(other, ROBOT) else
                     "fridge_other" if under(other, FRIDGE) else "other"),
            "points": 0, "loaded_points": 0, "upward_normal_points": 0,
            "impulse_on_target_Ns": [0.0, 0.0, 0.0],
            "minimum_separation_m": separation,
        })
        item["points"] += 1
        loaded = any(x != 0 for x in impulse)
        item["loaded_points"] += int(loaded)
        item["upward_normal_points"] += int(loaded and sign * normal[2] > 0)
        item["impulse_on_target_Ns"] = [
            x + sign*y for x, y in zip(item["impulse_on_target_Ns"], impulse)]
        item["minimum_separation_m"] = min(item["minimum_separation_m"], separation)
    return {
        "pairs": sorted(pairs.values(), key=lambda item: item["other_collider"]),
        "current_target_contact_points": sum(p["points"] for p in pairs.values()),
        "support_verified": False,
        "interpretation": "Instantaneous reported forces only; no settled/holding acceptance.",
    }


class InitialEvidenceReader:
    """Bind existing native rigid-body views; expose no setters or task actions."""
    def __init__(self, stage):
        from pxr import UsdPhysics, PhysxSchema
        self.stage = stage
        self.views = {}
        self.last_time = None
        self.sample_index = 0
        for role, path in BODIES.items():
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.IsActive() or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                raise ValueError("Missing existing rigid body: " + path)
            body = UsdPhysics.RigidBodyAPI(prim)
            if not body.GetRigidBodyEnabledAttr().Get() or body.GetKinematicEnabledAttr().Get():
                raise ValueError("Expected existing dynamic body: " + path)
        reporter = stage.GetPrimAtPath(BODIES["right_door"])
        if not reporter.HasAPI(PhysxSchema.PhysxContactReportAPI):
            raise ValueError("Existing L1 door contact reporting is unavailable")
        self.reporting_threshold = float(PhysxSchema.PhysxContactReportAPI(
            reporter).GetThresholdAttr().Get())
        if not math.isfinite(self.reporting_threshold):
            raise ValueError("Nonfinite reporting threshold")
        # Existing L1 contract; do NOT author or lower this value.
        if self.reporting_threshold != 0.0:
            raise ValueError("Existing L1 contact threshold differs from audited baseline")
        from .grasp_evidence import GraspEvidenceReader
        self.grasp_reader = GraspEvidenceReader(stage)

    def initialize(self, world):
        from omni.physics.tensors import create_simulation_view
        from omni.physx import get_physx_simulation_interface
        from pxr import PhysicsSchemaTools
        from omni.physx.bindings._physx import ContactEventType
        self.world = world
        # Read the worker-owned articulation; never construct/reset another one.
        self.robot = world.scene.get_object('arena_robot')
        self.joint_limits = {}
        from pxr import UsdPhysics
        for name in self.robot.dof_names:
            prim = self.stage.GetPrimAtPath(ROBOT+'/joints/'+name)
            unit = 'rad' if prim.IsA(UsdPhysics.RevoluteJoint) else 'm'
            low = float(prim.GetAttribute('physics:lowerLimit').Get())
            high = float(prim.GetAttribute('physics:upperLimit').Get())
            if unit == 'rad':
                low, high = math.radians(low), math.radians(high)
            self.joint_limits[name] = dict(unit=unit, lower=low, upper=high)
        self.interface = get_physx_simulation_interface()
        self.decode_path = PhysicsSchemaTools.intToSdfPath
        self.event_types = ContactEventType
        self.paths = {}
        # Native tensor views read existing PhysX actors, without high-level
        # prim wrappers that may reset Xform properties or author API schemas.
        self.simulation_view = create_simulation_view("numpy")
        for role, path in BODIES.items():
            view = self.simulation_view.create_rigid_body_view(path)
            if view.count != 1 or list(view.prim_paths) != [path]:
                raise ValueError("Wrong native body binding: " + path)
            self.views[role] = view
        self.grasp_reader.initialize(self.simulation_view, world)

    def _contacts(self):
        report = self.interface.get_contact_report()
        if not isinstance(report, tuple) or len(report) < 2:
            raise RuntimeError("Current-step contact report unavailable")
        headers, data = report[:2]
        rows = []
        robot_external_pairs = []
        lost = 0
        for h in headers:
            paths = []
            for token in (h.collider0, h.collider1):
                key = int(token)
                if key not in self.paths:
                    self.paths[key] = str(self.decode_path(token))
                paths.append(self.paths[key])
            if (sum(under(p, ROBOT) for p in paths)==1
                    and h.type != self.event_types.CONTACT_LOST):
                start, count = int(h.contact_data_offset), int(h.num_contact_data)
                impulses = [data[i].impulse for i in range(start,start+count)]
                robot_external_pairs.append(dict(colliders=paths, points=count,
                    sum_impulse_magnitude_Ns=sum(math.sqrt(v.x*v.x+v.y*v.y+v.z*v.z) for v in impulses)))
            if not any(under(p, TARGET) for p in paths):
                continue
            if h.type == self.event_types.CONTACT_LOST:
                lost += 1
                continue
            if h.type not in (self.event_types.CONTACT_FOUND, self.event_types.CONTACT_PERSIST):
                raise RuntimeError("Unknown contact event")
            start, count = int(h.contact_data_offset), int(h.num_contact_data)
            if start < 0 or count < 0 or start + count > len(data):
                raise RuntimeError("Invalid contact buffer range")
            for index in range(start, start + count):
                point = data[index]
                n, impulse = point.normal, point.impulse
                rows.append({"colliders": paths,
                             "normal": [n.x, n.y, n.z],
                             "impulse_Ns": [impulse.x, impulse.y, impulse.z],
                             "separation_m": float(point.separation)})
        result = contact_summary(rows)
        result.update(target_contact_lost_events=lost,
                      robot_external_pairs=robot_external_pairs,
                      report_header_count=len(headers),
                      no_report_means="unknown; sleeping contacts are not reconstructed")
        return result

    def sample(self):
        if set(self.views) != set(BODIES):
            raise RuntimeError("Reader not initialized")
        now = float(self.world.current_time)
        if not math.isfinite(now) or (self.last_time is not None and now <= self.last_time):
            raise ValueError("Repeated, stale or nonfinite simulation time")
        result = {"schema": "arena.fridge_initial_measurements.v1",
                  "sample_index": self.sample_index, "time_s": now,
                  "source": "isaac_physics",
                  "bodies": {role: measured_pose(view) for role, view in self.views.items()},
                  "contacts": self._contacts(),
                  "contact_reporting_threshold": self.reporting_threshold,
                  "initial_state_accepted": False, "task_success": None}
        if getattr(self, "grasp_reader", None) is not None:
            result["grasp"] = self.grasp_reader.sample(now)
        if getattr(self, 'robot', None) is not None:
            result['robot_joints'] = {name: dict(position=float(q), velocity=float(v),
                                               **self.joint_limits[name])
                for name,q,v in zip(self.robot.dof_names,
                                    self.robot.get_joint_positions(),
                                    self.robot.get_joint_velocities())}
        self.last_time = now
        self.sample_index += 1
        return result

    def close(self):
        # Release only this reader's references before Kit shutdown.
        # Do NOT invalidate/reset the shared tensor API or simulation.
        if getattr(self, "grasp_reader", None) is not None:
            self.grasp_reader.close()
        self.views.clear()
        self.simulation_view = None
        self.interface = None
        self.world = None
        self.robot = None
        self.stage = None
        self.paths = {}


def summarize_snapshot(snapshot):
    """Offline audit of an existing native snapshot, without starting Isaac."""
    from .native_initial_smoke import joint_errors
    import json
    expected = json.loads((TASK_DIR / "sources/initial_frame0.json").read_text())["hold_targets"]
    fields = snapshot["observation"]
    if fields.get("source") != "isaac_physics":
        raise ValueError("Not measured Isaac evidence")
    errors = joint_errors(fields, expected)
    rad = fields["joint_positions_rad"]
    arms = {n: math.degrees(v) for n, v in errors.items() if "_arm_" in n}
    grippers = {n: {"position_error_deg": math.degrees(v),
                    "velocity_rad_s": fields["joint_velocities_rad_s"][n]}
                for n, v in errors.items() if "_gripper_" in n}
    for item in grippers.values():
        finite_vector(item.values(), 2)
    for value in errors.values():
        if not math.isfinite(value):
            raise ValueError("Nonfinite measured error")
    return {"run_id": snapshot["run_id"], "time_s": fields["time_s"],
            "joint_errors_native": errors,
            "angular_joint_errors_deg": {n: math.degrees(v) for n, v in errors.items() if n in rad},
            "arm_max_error_deg": max(map(abs, arms.values())),
            "grippers": grippers, "initial_state_accepted": False,
            "box_support_verified": False,
            "missing": ["live box/door poses and velocities", "box-to-slot contact evidence"],
            "conclusion": "Robot snapshot measured; not a complete initial-state acceptance."}


def main():
    import argparse
    import json
    from .native_initial_smoke import write_json
    parser = argparse.ArgumentParser(description="Audit an existing snapshot; no GPU execution.")
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.snapshot.resolve(strict=True), args.output.resolve()
    if not source.is_relative_to(TASK_DIR) or not output.is_relative_to(TASK_DIR):
        parser.error("Inputs and outputs must remain in this task directory")
    report = summarize_snapshot(json.loads(source.read_text()))
    write_json(output, report)  # Exclusive creation: never overwrite previous evidence.
    print(json.dumps(report, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
