"""Original full-surface cavity/support measurements without tray geometry."""
import numpy as np
from scipy.spatial import ConvexHull
from evaluation.tasks.microwave_take import initial_state as old
from evaluation.tasks.microwave_take.process_rules import INNER_WALL_FACES, WallBox
from evaluation.tasks.microwave_take.isaac_process import extreme_vertices, transform, inverse


class Geometry:
    def __init__(self, stage, payload_variant):
        from pxr import Usd, UsdGeom, UsdPhysics, Gf
        old.verify_plate_asset()
        old.verify_plate_geometry(stage)
        old.verify_stand_geometry(stage)
        if UsdGeom.GetStageUpAxis(stage) != "Z" or UsdGeom.GetStageMetersPerUnit(stage) != 1:
            raise ValueError("Original metre Z-up geometry required")
        self.payload = old.bindings_for(payload_variant)["payload"]
        self.plate_body = old.PLATE+"/Asset"
        self.body = old.MICROWAVE+"/Asset/body"
        self.target = old.PLATE
        cache = UsdGeom.XformCache()
        body_tf = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(self.body)).RemoveScaleShear()
        inv = body_tf.GetInverse()
        def wall(path):
            p = stage.GetPrimAtPath(path)
            if (not p or not p.IsA(UsdGeom.Cube) or not p.HasAPI(UsdPhysics.CollisionAPI)
                    or p.GetAttribute("physics:collisionEnabled").Get() is False):
                raise ValueError("Original physical cavity cube required: "+path)
            tf = cache.GetLocalToWorldTransform(p)*inv
            axes = np.asarray(tf)[:3,:3]
            if (np.any(np.sum(abs(axes)>1e-7, axis=0) != 1)
                    or np.any(np.sum(abs(axes)>1e-7, axis=1) != 1)):
                raise ValueError("Cavity wall axes changed")
            h = UsdGeom.Cube(p).GetSizeAttr().Get()/2
            points = np.asarray([tf.Transform(Gf.Vec3d(x,y,z))
                                for x in (-h,h) for y in (-h,h) for z in (-h,h)])
            return tuple(points.min(0)), tuple(points.max(0))
        self.walls = tuple(WallBox(path, *wall(path)) for path in INNER_WALL_FACES)
        self.front_y = wall(old.MICROWAVE+"/Asset/door/col_door")[1][1]
        by = {w.collider.rsplit("/",1)[1]: w for w in self.walls}
        self.lo = np.asarray([by['col_left'].maximum_m[0], self.front_y, by['col_bottom'].maximum_m[2]])
        self.hi = np.asarray([by['col_right'].minimum_m[0], by['col_back'].minimum_m[1], by['col_top'].minimum_m[2]])
        if not np.all(self.hi > self.lo):
            raise ValueError("Invalid cavity")
        self.vertices = {}
        for role, path in (("plate", self.plate_body), ("payload", self.payload)):
            p = stage.GetPrimAtPath(path)
            if (not p or not p.HasAPI(UsdPhysics.RigidBodyAPI)
                    or p.GetAttribute("physics:kinematicEnabled").Get() is True):
                raise ValueError("Original dynamic body required: "+path)
            tf = cache.GetLocalToWorldTransform(p).RemoveScaleShear()
            points = []
            for prim in Usd.PrimRange(p):
                if not prim.IsA(UsdGeom.Mesh):
                    continue
                visible = UsdGeom.Imageable(prim).ComputeVisibility() != "invisible"
                colliding = (prim.HasAPI(UsdPhysics.CollisionAPI)
                    and prim.GetAttribute("physics:collisionEnabled").Get() is not False)
                if not (visible or colliding):
                    continue
                local = cache.GetLocalToWorldTransform(prim)*tf.GetInverse()
                points.extend(list(local.Transform(Gf.Vec3d(*v))) for v in prim.GetAttribute("points").Get())
            self.vertices[role] = np.asarray(points)
            if (self.vertices[role].ndim != 2 or self.vertices[role].shape[1] != 3
                    or not np.isfinite(self.vertices[role]).all()):
                raise ValueError("Missing complete target surface")
        self.extrema = {k: extreme_vertices(v) for k,v in self.vertices.items()}
        self.plate_footprint = ConvexHull(self.vertices["plate"][:,:2]).equations

    def contents(self, poses):
        result = {}
        for role, points in self.extrema.items():
            local = inverse(transform(points, poses[role]), poses["microwave"])
            low, high = local.min(0), local.max(0)
            result[role] = dict(front_clearance_m=float(self.front_y-high[1]),
                inside=bool(np.all(low >= self.lo) and np.all(high <= self.hi)))
        return result
