"""Read-only large-meal-tray geometry, evaluated in its CURRENT rigid frame.

The tray can move. Its outer handles/rim are not the usable placement floor.
No pose writes, physics overrides, fixed world height or centre-only containment.
"""
import math
import numpy as np
from evaluation.tasks.shelf_pick.geometry import rotation
from .scoring import PlacementFrame


def under(path, root):
    return path == root or path.startswith(root+'/')


class Geometry:
    def __init__(self, stage, target, destination):
        from pxr import Usd, UsdGeom, UsdPhysics, Gf
        from scipy.spatial import ConvexHull
        if UsdGeom.GetStageUpAxis(stage) != 'Z' or UsdGeom.GetStageMetersPerUnit(stage) != 1:
            raise ValueError('Metre Z-up stage required')
        self.target, self.destination = target, destination
        def body(root):
            prim = stage.GetPrimAtPath(root)
            if not prim: raise ValueError('Missing scene instance '+root)
            bodies = [p for p in Usd.PrimRange(prim) if p.HasAPI(UsdPhysics.RigidBodyAPI)
                      and p.GetAttribute('physics:rigidBodyEnabled').Get() is not False]
            if len(bodies) != 1: raise ValueError('Exactly one physical body required '+root)
            b = bodies[0]
            if b.GetAttribute('physics:kinematicEnabled').Get() is True:
                raise ValueError('Expected unchanged dynamic all_assets body '+root)
            m = UsdGeom.Xformable(b).ComputeLocalToWorldTransform(0).RemoveScaleShear()
            q = m.ExtractRotationQuat()
            return str(b.GetPath()), m, (tuple(m.ExtractTranslation()), (q.GetReal(), *q.GetImaginary()))
        self.target_body, target_tf, self.initial_target_pose = body(target)
        self.tray_body, tray_tf, self.initial_tray_pose = body(destination)
        def points(prim):
            if prim.IsA(UsdGeom.Mesh):
                return UsdGeom.Mesh(prim).GetPointsAttr().Get()
            b = UsdGeom.Boundable(prim)
            extent = UsdGeom.Boundable.ComputeExtentFromPlugins(b, Usd.TimeCode.Default())
            if extent is None or len(extent) != 2: raise ValueError('No measured shape extent')
            box = Gf.Range3d(Gf.Vec3d(*extent[0]), Gf.Vec3d(*extent[1]))
            return [box.GetCorner(i) for i in range(8)]
        def local_points(prim, body_tf):
            tf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)*body_tf.GetInverse()
            return np.asarray([list(tf.Transform(Gf.Vec3d(*p))) for p in points(prim)])
        def colliding(p):
            return p.HasAPI(UsdPhysics.CollisionAPI) and p.GetAttribute('physics:collisionEnabled').Get() is not False
        shapes = [p for p in Usd.PrimRange(stage.GetPrimAtPath(target)) if p.IsA(UsdGeom.Boundable)
                  and (colliding(p) or (UsdGeom.Imageable(p).ComputeVisibility()!='invisible'
                       and UsdGeom.Imageable(p).ComputePurpose() in ('default','render')))]
        if not shapes or not any(colliding(p) for p in shapes): raise ValueError('No target colliders')
        self.vertices = np.vstack([local_points(p,target_tf) for p in shapes])
        floors = [p for p in Usd.PrimRange(stage.GetPrimAtPath(destination))
                  if p.GetName()=='FlatFloor' and p.IsA(UsdGeom.Cube) and colliding(p)]
        if len(floors) != 1: raise ValueError('Expected original all_assets large-tray FlatFloor collider')
        self.floor_path = str(floors[0].GetPath())
        v = local_points(floors[0],tray_tf)
        top = v[np.abs(v[:,2]-v[:,2].max()) < 1e-7]
        if len(top)!=4 or not np.isfinite(self.vertices).all(): raise ValueError('Invalid floor/target geometry')
        self.floor_vertices = top
        self.region = ConvexHull(top[:,:2]).equations

    def measure(self, target_pose, tray_pose):
        tp,tq=target_pose; dp,dq=tray_pose
        tp=np.asarray(tp); dp=np.asarray(dp)
        if tp.shape!=(3,) or dp.shape!=(3,) or not np.isfinite([tp,dp]).all():
            raise ValueError('Finite current native positions required')
        tr,dr=rotation(tq),rotation(dq)
        points=dr.inv().apply(tr.apply(self.vertices)+tp-dp)
        inside=bool(np.all(points[:,:2]@self.region[:,:2].T+self.region[:,2] <= 1e-7))
        pos=tuple(map(float,dr.inv().apply(tp-dp)))
        q=(dr.inv()*tr).as_quat()
        quat=(float(q[3]),*map(float,q[:3]))
        tilt=math.degrees(math.acos(float(np.clip(tr.apply([0,0,1])[2],-1,1))))
        return inside,pos,quat,tilt

    def frame(self, binding, step, target_pose, tray_pose, pairs, floors):
        """pairs must be current nonzero-impulse native Contacts.pairs() output."""
        if binding.target != self.target_body or binding.destination != self.tray_body:
            raise ValueError('Exact measured rigid bodies must match scoring roles')
        support=False; robot=False; ground=False
        for paths,samples in pairs:
            for i in (0,1):
                if not under(paths[i],self.target): continue
                other=paths[1-i]; sign=1 if i==0 else -1
                robot |= under(other,'/World/Robot')
                ground |= any(under(other,p) for p in floors)
                # Existing support-normal predicate; rim/handles are excluded.
                if under(other,self.floor_path):
                    support |= any(sign*n[2]>.5 for n,_,_ in samples)
        inside,pos,quat,tilt=self.measure(target_pose,tray_pose)
        return PlacementFrame(binding,step,step*.05,bool(support),inside,bool(robot),bool(ground),pos,quat,tilt)

    def receipt(self):
        return dict(target_body=self.target_body, tray_body=self.tray_body,
            floor_collider=self.floor_path, floor_vertices_in_tray_m=self.floor_vertices.tolist(),
            target_vertex_count=len(self.vertices), target_extent_in_body_m=np.ptp(self.vertices,axis=0).tolist(),
            containment='all_visible_and_collision_vertices_projected_into_current_tray_floor',
            support='actual_upward_native_contact_with_FlatFloor_not_outer_rim',
            native_contact_stream_connected=False, physics_parameters_changed=False)
