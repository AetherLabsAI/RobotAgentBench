"""Read-only bindings for the actual unified-room payload and destination."""
import itertools
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from pxr import Gf, Usd, UsdGeom, UsdPhysics
from evaluation.tasks.microwave_put.physical_geometry import (
    PutGeometry, Wall, WALL_FACES, BODY, HINGE, PLATE_BODY, STAND, MICROWAVE_ROOT)
from evaluation.tasks.microwave_put.payload_assets import validate_payload_geometry
from evaluation.tasks.refrigerator_take.random_reset import PARTS
from room_fridge_helper_v1 import FRIDGE

ROBOT='/World/Robot'
DOORS=(FRIDGE+'/left_door',FRIDGE+'/right_door')


def parameters(p):
    if set(p)!={'payload_variant'} or p['payload_variant'] not in ('beef_box','leftover_box'):
        raise ValueError('Explicit Task9 payload_variant: beef_box or leftover_box required')
    return p['payload_variant']


def pose_matrix(m):
    q=m.ExtractRotationQuat().GetNormalized()
    return dict(position_m=list(m.ExtractTranslation()),quaternion_wxyz=[q.GetReal(),*q.GetImaginary()])


def vertices(stage, path):
    cache=UsdGeom.XformCache();root=stage.GetPrimAtPath(path)
    if not root or not root.HasAPI(UsdPhysics.RigidBodyAPI):raise ValueError('Missing rigid body '+path)
    body=cache.GetLocalToWorldTransform(root);axes=np.asarray(body)[:3,:3]
    scale=np.linalg.norm(axes,axis=1)
    if not np.allclose((axes/scale[:,None])@(axes/scale[:,None]).T,np.eye(3),atol=1e-7):raise ValueError('Sheared body')
    data=[]
    for p in Usd.PrimRange(root):
        if not p.IsA(UsdGeom.Gprim):continue
        image=UsdGeom.Imageable(p)
        visible=image.ComputeVisibility()!='invisible' and image.ComputePurpose() in ('default','render')
        collision=p.HasAPI(UsdPhysics.CollisionAPI) and p.GetAttribute('physics:collisionEnabled').Get() is not False
        if not (visible or collision):continue
        if not p.IsA(UsdGeom.Mesh):raise ValueError('Unreviewed box/plate geometry '+str(p.GetPath()))
        m=np.asarray(cache.GetLocalToWorldTransform(p)*body.GetInverse())
        vs=np.asarray(p.GetAttribute('points').Get(),float)
        data.extend((vs@m[:3,:3]+m[3,:3])*scale)
    points=np.asarray(data)
    if points.ndim!=2 or not len(points) or not np.isfinite(points).all():raise ValueError('Incomplete geometry')
    return points


class Geometry(PutGeometry):
    """Original containment/support predicates, binding a supplied real target.

    No remapping USD paths, alias bodies, resets or fixed-world coordinates.
    """
    def __init__(self, stage, target, variant):
        self.payload=target
        self.asset_contract=validate_payload_geometry(stage,target,variant)
        if UsdGeom.GetStageUpAxis(stage)!='Z' or UsdGeom.GetStageMetersPerUnit(stage)!=1:raise ValueError('Expected Z-up metres')
        cache=UsdGeom.XformCache();mw=cache.GetLocalToWorldTransform(stage.GetPrimAtPath(MICROWAVE_ROOT))
        axes=np.asarray(mw)[:3,:3]
        if not np.allclose(axes@axes.T,np.eye(3),atol=1e-9):raise ValueError('Nonrigid microwave root')
        self.microwave_pose=pose_matrix(mw)
        hinge=UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(HINGE))
        if not hinge:raise ValueError('Missing microwave hinge')
        low,high=hinge.GetLowerLimitAttr().Get(),hinge.GetUpperLimitAttr().Get()
        if low < -60 and abs(high)<1e-6:self.opening_sign=-1.
        elif abs(low)<1e-6 and high>60:self.opening_sign=1.
        else:raise ValueError('Unknown microwave opening convention')
        def cube(path):
            p=stage.GetPrimAtPath(path)
            if not p or not p.IsA(UsdGeom.Cube) or not p.HasAPI(UsdPhysics.CollisionAPI) or p.GetAttribute('physics:collisionEnabled').Get() is False:
                raise ValueError('Missing enabled cavity cube '+path)
            m=cache.GetLocalToWorldTransform(p)*mw.GetInverse();a=np.asarray(m)[:3,:3]
            if np.any(np.sum(abs(a)>1e-7,axis=0)!=1) or np.any(np.sum(abs(a)>1e-7,axis=1)!=1):raise ValueError('Nonaxis aligned wall')
            h=UsdGeom.Cube(p).GetSizeAttr().Get()/2
            v=np.asarray([m.Transform(Gf.Vec3d(*xyz)) for xyz in itertools.product((-h,h),repeat=3)])
            return v.min(0),v.max(0)
        self.walls={p:Wall(p,*map(tuple,cube(p))) for p in WALL_FACES}
        self.front_y=float(cube(MICROWAVE_ROOT+'/Asset/door/col_door')[1][1])
        self.low=np.array([self.walls[BODY+'/col_left'].high[0],self.front_y,self.walls[BODY+'/col_bottom'].high[2]])
        self.high=np.array([self.walls[BODY+'/col_right'].low[0],self.walls[BODY+'/col_back'].low[1],self.walls[BODY+'/col_top'].low[2]])
        if not np.all(self.high>self.low):raise ValueError('Invalid cavity dimensions')
        self.vertices={r:vertices(stage,p) for r,p in [('payload',target),('plate',PLATE_BODY)]}
        self.extrema={r:v[ConvexHull(v).vertices] for r,v in self.vertices.items()}
        self.footprint=ConvexHull(self.vertices['plate'][:,:2]).equations


class Scene:
    def __init__(self,stage,manifest,variant):
        self.stage=stage;self.variant=variant
        matches=[r for r in manifest['objects'] if r['id']==variant]
        if len(matches)!=1:raise ValueError('Expected one actual initialized '+variant)
        row=matches[0];self.target=row['path']
        if row.get('initialization_owner')!='unified_scene_fridge_boxes_v1':raise ValueError('Unknown scene-owned box provenance')
        if row.get('slot') not in ('low_out','low_in','up_out','up_in'):raise ValueError('Missing source compartment')
        self.source_parts=PARTS['shelf4' if row['slot'].startswith('up') else 'bin34']
        self.source_supports=tuple(DOORS[1]+'/'+p for p in self.source_parts[:2])
        self.source_paths=tuple(DOORS[1]+'/'+p for p in self.source_parts)
        self.geometry=Geometry(stage,self.target,variant)
        self.joints={}
        for name,path in [('left',FRIDGE+'/left_door_hinge'),('right',FRIDGE+'/right_door_hinge'),('microwave',HINGE)]:
            joint=UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(path))
            if not joint:raise ValueError('Missing hinge '+path)
            bodies=[joint.GetBody0Rel().GetTargets(),joint.GetBody1Rel().GetTargets()]
            if any(len(b)!=1 for b in bodies):raise ValueError('Unbound hinge')
            frames=[]
            for getter in (joint.GetLocalRot0Attr,joint.GetLocalRot1Attr):
                q=getter().Get();frames.append(Rotation.from_quat([*q.GetImaginary(),q.GetReal()]).as_matrix())
            self.joints[name]=(tuple(str(b[0]) for b in bodies),frames,str(joint.GetAxisAttr().Get()))
        cache=UsdGeom.XformCache();inv=cache.GetLocalToWorldTransform(stage.GetPrimAtPath(DOORS[1])).GetInverse();points=[]
        for path in self.source_paths:
            p=stage.GetPrimAtPath(path)
            if not p or not p.IsA(UsdGeom.Cube):raise ValueError('Missing original bin support '+path)
            h=UsdGeom.Cube(p).GetSizeAttr().Get()/2;m=cache.GetLocalToWorldTransform(p)*inv
            points.extend(m.Transform(Gf.Vec3d(*x)) for x in itertools.product((-h,h),repeat=3))
        points=np.asarray(points);self.lower,self.upper=points.min(0),points.max(0)
        self.activation={}
        for phase,path in [('refrigerator',FRIDGE),('microwave',MICROWAVE_ROOT)]:
            b=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render']).ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange()
            if b.IsEmpty():raise ValueError('Missing activation object geometry')
            self.activation[phase]=[list(b.GetMin()),list(b.GetMax())]
