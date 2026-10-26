"""Read-only geometry bindings for put; no take-task initial-state assumptions."""
from dataclasses import dataclass
import numpy as np
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.transform import Rotation

from .beef_scene import ROBOT, PLATE, PAYLOAD, MICROWAVE_ROOT

BODY = MICROWAVE_ROOT + "/Asset/body"
HINGE = MICROWAVE_ROOT + "/Asset/door_hinge"
PLATE_BODY = PLATE + "/Asset"
STAND = "/World/Assets/MicrowavePlateStand"
PADS = {h: tuple(ROBOT + f"/gripper_{h}_{part}_link4/collisions/{part}_link4/mesh"
                 for part in ("inner", "outer")) for h in ("l", "r")}
WALL_FACES = {BODY + "/col_" + name: face for name, face in
              (("left", (0, 1)), ("right", (0, 0)), ("back", (1, 0)),
               ("top", (2, 0)), ("bottom", (2, 1)))}


def under(path, root):
    return path == root or path.startswith(root + "/")


def rotation(quaternion_wxyz):
    q = np.asarray(quaternion_wxyz, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1) > 1e-4:
        raise ValueError("Invalid measured unit quaternion")
    return Rotation.from_quat([*q[1:], q[0]])


def transform(points, pose):
    p = np.asarray(pose["position_m"], dtype=float)
    if p.shape != (3,) or not np.isfinite(p).all():
        raise ValueError("Invalid measured position")
    return rotation(pose["quaternion_wxyz"]).apply(points) + p


def inverse(points, pose):
    p = np.asarray(pose["position_m"], dtype=float)
    if p.shape != (3,) or not np.isfinite(p).all():
        raise ValueError("Invalid measured position")
    return rotation(pose["quaternion_wxyz"]).inv().apply(np.asarray(points)-p)


@dataclass(frozen=True)
class Wall:
    path: str
    low: tuple
    high: tuple

    def inner_face(self, point):
        # Existing nearest-box-face convention; not a collision-distance gate.
        axis, high_face = WALL_FACES[self.path]
        distances = [abs(point[i]-bound[i]) for i in range(3) for bound in (self.low, self.high)]
        return distances[axis*2+high_face] <= min(distances)+1e-9


def report_coverage(stage, *, payload=PAYLOAD):
    """Each relevant pair has a reported endpoint. Never enable it implicitly."""
    from pxr import UsdPhysics, PhysxSchema
    rows = []
    # Payload covers all payload/robot and payload/plate pairs; body covers
    # robot/camera versus inner walls. Plate additionally records its support.
    for path in (payload, PLATE_BODY, BODY):
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError("Missing contact-report rigid body: " + path)
        api = PhysxSchema.PhysxContactReportAPI(prim)
        threshold = api.GetThresholdAttr().Get() if api else None
        rows.append(dict(path=path, enabled=bool(api), threshold=threshold,
                         complete=bool(api) and threshold == 0))
    return rows


class PutGeometry:
    def __init__(self, stage, *, payload_variant="beef_box"):
        from pxr import Gf, Usd, UsdGeom, UsdPhysics
        from .payload_assets import validate_payload_geometry
        from .payload_binding import payload_for
        self.payload = payload_for(payload_variant)
        validate_payload_geometry(stage, self.payload, payload_variant)
        if UsdGeom.GetStageUpAxis(stage) != "Z" or UsdGeom.GetStageMetersPerUnit(stage) != 1:
            raise ValueError("Expected metre-authored Z-up scene")
        cache = UsdGeom.XformCache()
        mw = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(MICROWAVE_ROOT))
        # Root must be rigid; the existing 1.25 cavity scale is below it.
        axes = np.asarray(mw)[:3, :3]
        if not np.allclose(axes @ axes.T, np.eye(3), rtol=0, atol=1e-9):
            raise ValueError("Nonrigid microwave measurement frame")
        tf = Gf.Transform(mw)
        q = tf.GetRotation().GetQuat()
        self.microwave_pose = dict(position_m=list(tf.GetTranslation()),
                                  quaternion_wxyz=[q.GetReal(), *q.GetImaginary()])
        hinge = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(HINGE))
        if not hinge:
            raise ValueError("Missing microwave hinge")
        low, high = hinge.GetLowerLimitAttr().Get(), hinge.GetUpperLimitAttr().Get()
        if low < -60 and abs(high) < 1e-6:
            self.opening_sign = -1.
        elif abs(low) < 1e-6 and high > 60:
            self.opening_sign = 1.
        else:
            raise ValueError("Unknown hinge opening convention")

        def cube(path):
            prim = stage.GetPrimAtPath(path)
            if (not prim or not prim.IsA(UsdGeom.Cube) or not prim.HasAPI(UsdPhysics.CollisionAPI)
                    or prim.GetAttribute("physics:collisionEnabled").Get() is False):
                raise ValueError("Missing enabled cavity cube: " + path)
            matrix = cache.GetLocalToWorldTransform(prim) * mw.GetInverse()
            axes = np.asarray(matrix)[:3, :3]
            if (np.any(np.sum(abs(axes)>1e-7, axis=0) != 1)
                    or np.any(np.sum(abs(axes)>1e-7, axis=1) != 1)):
                raise ValueError("Cavity wall is not axis aligned")
            half = UsdGeom.Cube(prim).GetSizeAttr().Get()/2
            v = np.array([matrix.Transform(Gf.Vec3d(x,y,z))
                          for x in (-half,half) for y in (-half,half) for z in (-half,half)])
            return v.min(axis=0), v.max(axis=0)

        self.walls = {p: Wall(p, *map(tuple, cube(p))) for p in WALL_FACES}
        self.front_y = float(cube(MICROWAVE_ROOT + "/Asset/door/col_door")[1][1])
        self.low = np.array([self.walls[BODY+"/col_left"].high[0], self.front_y,
                             self.walls[BODY+"/col_bottom"].high[2]])
        self.high = np.array([self.walls[BODY+"/col_right"].low[0],
                              self.walls[BODY+"/col_back"].low[1], self.walls[BODY+"/col_top"].low[2]])
        if not np.all(self.high > self.low):
            raise ValueError("Invalid cavity dimensions")
        self.vertices, self.extrema = {}, {}
        for role, root in (("payload", self.payload), ("plate", PLATE_BODY)):
            prim = stage.GetPrimAtPath(root)
            if not prim or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                raise ValueError("Missing task rigid body: " + root)
            body_tf = cache.GetLocalToWorldTransform(prim)
            axes = np.asarray(body_tf)[:3, :3]
            scales = np.linalg.norm(axes, axis=1)
            if (np.any(scales <= 0) or not np.allclose(
                    (axes/scales[:,None]) @ (axes/scales[:,None]).T, np.eye(3), atol=1e-7)):
                raise ValueError("Sheared task geometry")
            vertices = []
            for child in Usd.PrimRange(prim):
                if not child.IsA(UsdGeom.Gprim):
                    continue
                imageable = UsdGeom.Imageable(child)
                visible = (imageable.ComputeVisibility() != "invisible"
                           and imageable.ComputePurpose() in ("default", "render"))
                collision = (child.HasAPI(UsdPhysics.CollisionAPI)
                             and child.GetAttribute("physics:collisionEnabled").Get() is not False)
                if not (visible or collision):
                    continue
                if not child.IsA(UsdGeom.Mesh):
                    raise ValueError("Unreviewed task geometry type: " + str(child.GetPath()))
                local = cache.GetLocalToWorldTransform(child) * body_tf.GetInverse()
                vertices.extend(np.asarray(local.Transform(Gf.Vec3d(*v)))*scales
                                for v in child.GetAttribute("points").Get())
            data = np.asarray(vertices)
            if data.ndim != 2 or data.shape[1] != 3 or not np.isfinite(data).all():
                raise ValueError("Missing complete task mesh geometry")
            self.vertices[role] = data
            try:
                self.extrema[role] = data[ConvexHull(data).vertices]
            except QhullError:
                self.extrema[role] = data
        self.footprint = ConvexHull(self.vertices["plate"][:,:2]).equations

    def contents(self, poses):
        result = {}
        for role, v in self.extrema.items():
            points = inverse(transform(v, poses[role]), self.microwave_pose)
            low, high = points.min(axis=0), points.max(axis=0)
            result[role] = dict(inside=bool(np.all(low >= self.low) and np.all(high <= self.high)),
                                low=low.tolist(), high=high.tolist())
        return result

    def com_on_plate(self, poses, com_local):
        com = inverse(transform([com_local], poses["payload"]), poses["plate"])[0]
        # Numerical hull-plane roundoff only; no pose-drift or stability gate.
        return bool(com[2] > 0 and np.all(self.footprint[:,:2] @ com[:2] + self.footprint[:,2] <= 1e-7))
