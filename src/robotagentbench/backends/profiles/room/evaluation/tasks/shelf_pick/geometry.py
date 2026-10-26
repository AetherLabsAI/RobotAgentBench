"""Read-only support geometry and conservative robot reach bounds.

The outer reach sphere proves unreachability; it is not an IK feasibility test.
No target is declared lost merely because its centre crosses a shelf edge.
"""
import math
import numpy as np
from .initial_state import TARGET, SUPPORT, ROBOT
from .targets import DEFAULT
from .contact_stream import FLOOR, PADS

BASE_LINK=ROBOT+'/base_link'
HANDS={h:ROBOT+f'/gripper_{h}_base_link' for h in PADS}


def rotation(q):
    from scipy.spatial.transform import Rotation
    q=np.asarray(q,dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1)>1e-4:
        raise ValueError('Invalid measured quaternion')
    return Rotation.from_quat([*q[1:],q[0]])


class Geometry:
    def __init__(self,stage,spec=DEFAULT,stock_roots=None,targets=None,supports=None):
        from pxr import Gf, Usd, UsdGeom, UsdPhysics
        if UsdGeom.GetStageUpAxis(stage)!='Z' or abs(UsdGeom.GetStageMetersPerUnit(stage)-1)>1e-9:
            raise ValueError('Expected metre-scale Z-up stage')
        self.spec=spec
        self.support=spec.support
        self.targets=tuple(targets) if targets else tuple(spec.targets)
        self.stock_roots=tuple(stock_roots) if stock_roots else (spec.stock_root,)
        # A source episode fills both tiers, so every support surface in use is measured.
        self.supports=tuple(dict.fromkeys((self.support,*(supports or ()))))
        self.fixed={}
        bounds={}
        # Collision shells may intentionally be invisible to the renderer.
        cache=UsdGeom.BBoxCache(0,['default','render','proxy','guide'],False,True)
        def colliding(p):
            return bool(p and p.IsActive() and p.HasAPI(UsdPhysics.CollisionAPI) and p.GetAttribute('physics:collisionEnabled').Get() is not False)
        original=stage.GetPrimAtPath(FLOOR)
        generated=stage.GetPrimAtPath('/World/HouseTopologyCandidate/Shell/Floors')
        generated_paths=[str(p.GetPath()) for p in generated.GetChildren() if colliding(p)] if generated else []
        if colliding(original) and generated_paths:raise ValueError('Duplicate original/generated floor collisions')
        self.floor_paths=[FLOOR] if colliding(original) else generated_paths
        if not self.floor_paths:raise ValueError('No active physical floor')
        for path in (*self.supports,*self.floor_paths):
            p=stage.GetPrimAtPath(path)
            if not p or not p.HasAPI(UsdPhysics.CollisionAPI) or p.GetAttribute('physics:collisionEnabled').Get() is False:
                raise ValueError('Missing active colliding surface '+path)
            self.fixed[path]=UsdGeom.Xformable(p).ComputeLocalToWorldTransform(0)
            b=cache.ComputeWorldBound(p).ComputeAlignedRange()
            if b.IsEmpty():raise ValueError('Empty surface geometry')
            bounds[path]=[list(b.GetMin()),list(b.GetMax())]
        self.support_bounds=np.asarray(bounds[self.support])
        self.floor_regions=[np.asarray(bounds[p]) for p in self.floor_paths]
        self.floor_bounds=np.stack([np.min([b[0] for b in self.floor_regions],axis=0),np.max([b[1] for b in self.floor_regions],axis=0)])
        self.top=float(self.support_bounds[1,2])
        self.support_tops={p:float(np.asarray(bounds[p])[1,2]) for p in self.supports}
        def body_points(body, collision_only=False):
            transform=UsdGeom.Xformable(body).ComputeLocalToWorldTransform(0)
            # Native rigid-body poses contain rotation/translation, not USD scale.
            # Retain the existing remote's scale in measured geometry only.
            if spec.external_target and str(body.GetPath()) in self.targets:
                transform=transform.RemoveScaleShear()
            inverse=transform.GetInverse()
            points=[]
            for c in Usd.PrimRange(body):
                shape=UsdGeom.Boundable(c)
                if not shape:continue
                collision=colliding(c)
                visible=UsdGeom.Imageable(c).ComputeVisibility()!='invisible'
                if not collision and (collision_only or not visible):continue
                # Schema-derived extents first: several analytic collision shapes in the
                # shelf assets (e.g. good2grow's cylinders) carry a stale authored extent
                # of +/-1 m that does not match their radius/height, which made measured
                # geometry reach ~0.9 m below the board. PhysX uses the schema attributes,
                # so the measurement follows them too. Nothing about physics changes here.
                extent=UsdGeom.Boundable.ComputeExtentFromPlugins(shape,Usd.TimeCode.Default())
                if extent is None:
                    extent=shape.GetExtentAttr().Get()
                if extent is None or len(extent)!=2:raise ValueError('Missing local geometry extent '+str(c.GetPath()))
                box=Gf.Range3d(Gf.Vec3d(*extent[0]),Gf.Vec3d(*extent[1]))
                relative=UsdGeom.Xformable(c).ComputeLocalToWorldTransform(0)*inverse
                points.extend(list(relative.Transform(box.GetCorner(i))) for i in range(8))
            if not points:raise ValueError('Missing body geometry '+str(body.GetPath()))
            return np.asarray(points)
        self.radius={}
        self.collision_vertices={}
        bodies = [c for root in self.stock_roots for c in stage.GetPrimAtPath(root).GetChildren()]
        if spec.external_target:
            bodies.extend(stage.GetPrimAtPath(path) for path in self.targets)
        for p in bodies:
            if not p.HasAPI(UsdPhysics.RigidBodyAPI):continue
            # Local shape extents avoid pose-dependent world AABB inflation.
            points=body_points(p)
            self.radius[str(p.GetPath())]=float(np.linalg.norm(points,axis=1).max())
            self.collision_vertices[str(p.GetPath())]=body_points(p,collision_only=True)
        for path in self.targets:
            if path not in self.radius:raise ValueError('Missing target radius '+path)
        parents={}
        for p in Usd.PrimRange(stage.GetPrimAtPath(ROBOT)):
            if not any(p.IsA(t) for t in (UsdPhysics.FixedJoint,UsdPhysics.RevoluteJoint,UsdPhysics.PrismaticJoint)):continue
            a=p.GetRelationship('physics:body0').GetTargets();b=p.GetRelationship('physics:body1').GetTargets()
            if len(a)!=1 or len(b)!=1:continue
            child=str(b[0])
            if child in parents:raise ValueError('Ambiguous robot reach chain '+child)
            distance=sum(p.GetAttribute('physics:localPos'+str(i)).Get().GetLength() for i in (0,1))
            if p.IsA(UsdPhysics.PrismaticJoint):
                limits=[p.GetAttribute('physics:'+x+'Limit').Get() for x in ('lower','upper')]
                if not all(math.isfinite(x) for x in limits):raise ValueError('Unbounded prismatic reach')
                distance+=max(abs(x) for x in limits)
            parents[child]=(str(a[0]),distance)
        def bound(path,visited):
            if path==BASE_LINK:return 0.
            if path in visited or path not in parents:raise ValueError('Unresolved robot reach chain '+path)
            parent,length=parents[path]
            return length+bound(parent,visited|{path})
        self.reach_by_pad={}
        for pads in PADS.values():
            for path in pads:
                p=stage.GetPrimAtPath(path)
                extra=float(np.linalg.norm(body_points(p),axis=1).max())
                self.reach_by_pad[path]=bound(path,set())+extra
        self.reach=max(self.reach_by_pad.values())

    def validate_static(self,stage):
        from pxr import UsdGeom
        if any(UsdGeom.Xformable(stage.GetPrimAtPath(p)).ComputeLocalToWorldTransform(0)!=tf for p,tf in self.fixed.items()):
            raise RuntimeError('Scoring surfaces changed during episode')

    def bottom_height(self,path,pose):
        points=rotation(pose['quaternion_wxyz']).apply(self.collision_vertices[path])+pose['position_m']
        return float(points[:,2].min())

    def outside_workspace(self,target_position,base_position,path=None):
        p=np.asarray(target_position);b=np.asarray(base_position);r=self.radius[path or self.spec.target]
        outside_reach=np.linalg.norm(p-b)>self.reach+r
        outside_room=all(any(p[i]+r<region[0,i] or p[i]-r>region[1,i] for i in (0,1)) for region in self.floor_regions)
        return bool(outside_reach or outside_room)

    def description(self):
        return dict(support=self.support,floors=self.floor_paths,floor_regions_m=[b.tolist() for b in self.floor_regions],support_bounds_m=self.support_bounds.tolist(),
                    floor_bounds_m=self.floor_bounds.tolist(),body_radius_m=self.radius,
                    maximum_reach_by_pad_m=self.reach_by_pad,
                    workspace='conservative outer bound from native robot chain and actual room floor; entire object must be outside',
                    workspace_is_ik_feasibility_test=False)
