"""Read-only left-pad geometry and pair-force evidence for the L1 grasp probe.

No contact-report API, collision filtering, material, force or state setters.
Geometry is cached in each existing link frame; moving poses come from PhysX.
"""
from collections import deque
import math
import numpy as np
from .initial_state import ROBOT, TARGET
from .observation_geometry import rotation_wxyz

PADS = [ROBOT + "/gripper_l_outer_link4", ROBOT + "/gripper_l_inner_link4"]
JAW = "idx41_gripper_l_outer_joint1"


def pad_faces(points, origins):
    """Original L1 extreme-face vertex mean in a 3 mm band."""
    origins = np.asarray(origins, float)
    if origins.shape != (2, 3) or not np.isfinite(origins).all():
        raise ValueError("Invalid pad origins")
    d = origins[1] - origins[0]
    if np.linalg.norm(d) < 1e-8:
        raise ValueError("Coincident pad links")
    d /= np.linalg.norm(d)
    faces = []
    for pts, direction in zip(points, (d, -d)):
        pts = np.asarray(pts, float)
        if pts.ndim != 2 or pts.shape[1] != 3 or not len(pts) or not np.isfinite(pts).all():
            raise ValueError("Missing pad visual points")
        projection = pts @ direction
        faces.append(pts[projection >= projection.max() - .003].mean(axis=0))
    return np.asarray(faces)


def validate_pair_view(view, pad):
    if view.sensor_count != 1 or view.filter_count != 1:
        raise ValueError("Pair force view must bind exactly one pad and target")
    if list(view.sensor_paths) != [pad]:
        raise ValueError("Wrong pad force sensor")
    # Native versions expose filters either flat or grouped per sensor.
    def flatten(x):
        return [x] if isinstance(x, str) else [p for y in x for p in flatten(y)]
    if flatten(view.filter_paths) != [TARGET]:
        raise ValueError("Wrong force target")


def force_vector(view, dt):
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("Invalid measured physics step")
    forces = np.asarray(view.get_contact_force_matrix(dt), float)
    if forces.shape != (1, 1, 3) or not np.isfinite(forces).all():
        raise ValueError("Unavailable current pad-target force")
    return forces[0, 0].copy()


class GraspEvidenceReader:
    def __init__(self, stage):
        from pxr import Usd, UsdGeom
        self.local_points = []
        cache = UsdGeom.XformCache()
        for path in PADS:
            link = stage.GetPrimAtPath(path)
            visual = stage.GetPrimAtPath(path + "/visuals")
            if not link or not visual:
                raise ValueError("Missing original pad visual: " + path)
            inv_link = cache.GetLocalToWorldTransform(link).GetInverse()
            pts = []
            for prim in Usd.PrimRange(visual, Usd.TraverseInstanceProxies()):
                if prim.IsA(UsdGeom.Mesh):
                    transform = cache.GetLocalToWorldTransform(prim) * inv_link
                    pts.extend([list(transform.Transform(p)) for p in UsdGeom.Mesh(prim).GetPointsAttr().Get()])
            points = np.asarray(pts, float)
            if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
                raise ValueError("Empty pad geometry: " + path)
            self.local_points.append(points)
        self.views = []
        self.forces = []
        self.history = deque(maxlen=16)

    def initialize(self, simulation_view, world):
        self.world = world
        self.robot = world.scene.get_object("arena_robot")
        if self.robot is None or list(self.robot.dof_names).count(JAW) != 1:
            raise ValueError("Missing existing measured jaw joint")
        self.jaw_index = list(self.robot.dof_names).index(JAW)
        for pad in PADS:
            view = simulation_view.create_rigid_body_view(pad)
            if view.count != 1 or list(view.prim_paths) != [pad]:
                raise ValueError("Wrong pad body binding")
            self.views.append(view)
            force = simulation_view.create_rigid_contact_view(pad, [TARGET])
            validate_pair_view(force, pad)
            self.forces.append(force)

    def sample(self, now):
        points, origins = [], []
        for pts, view in zip(self.local_points, self.views):
            pose = np.asarray(view.get_transforms(), float)
            if pose.shape != (1, 7) or not np.isfinite(pose).all():
                raise ValueError("Missing current pad pose")
            p = pose[0, :3]
            xyzw = pose[0, 3:]
            R = rotation_wxyz([xyzw[3], *xyzw[:3]])
            origins.append(p)
            points.append(pts @ R.T + p)
        faces = pad_faces(points, origins)
        dt = float(self.world.get_physics_dt())
        forces = [force_vector(v, dt) for v in self.forces]
        jaw = float(self.robot.get_joint_positions()[self.jaw_index])
        if not math.isfinite(jaw):
            raise ValueError("Nonfinite measured jaw")
        self.history.append([now, jaw])
        return dict(pad_paths=PADS, target_path=TARGET, time_s=now,
                    visual_face_centers_m=faces.tolist(),
                    visual_gap_m=float(np.linalg.norm(faces[1]-faces[0])),
                    midpoint_m=faces.mean(axis=0).tolist(),
                    force_on_pad_from_target_N=[v.tolist() for v in forces],
                    pair_force_norm_N=[float(np.linalg.norm(v)) for v in forces],
                    force_source="native_rigid_contact_view_pair_sum",
                    point_count=None, physics_dt_s=dt,
                    jaw_joint=JAW, jaw_position_rad=jaw,
                    jaw_recent_physics_samples=list(self.history),
                    holding_accepted=None)

    def close(self):
        self.views.clear()
        self.forces.clear()
        self.history.clear()
        self.robot = None
        self.world = None
