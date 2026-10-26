"""Measured microwave/optional small-plate geometry, no scene writes."""
import numpy as np
from scipy.spatial import ConvexHull
from .geometry import Geometry,under
from .rotary_component.geometry import collect_body_points
from .rotary_component.rotary_start_runtime import rotation
from evaluation.tasks.microwave_take import process_rules as rules

ROOT='/World/Assets/DiningTableMicrowave/Asset'
PLATE='/World/Assets/MicrowaveTakePlate'


def transform(vertices,pose):
    return np.asarray(vertices)@rotation(pose['quaternion_wxyz']).T+pose['position_m']


def inverse(vertices,pose):
    return (np.asarray(vertices)-pose['position_m'])@rotation(pose['quaternion_wxyz'])


class MicrowaveGeometry:
    def __init__(self,stage,meal):
        from pxr import Usd,UsdGeom,UsdPhysics,Gf
        self.body=ROOT+'/body';cache=UsdGeom.XformCache()
        b=stage.GetPrimAtPath(self.body)
        if not b or not b.HasAPI(UsdPhysics.RigidBodyAPI):raise ValueError('Native oven body required')
        world=cache.GetLocalToWorldTransform(b).RemoveScaleShear();inv=world.GetInverse()
        def box(path):
            p=stage.GetPrimAtPath(path)
            if not p or not p.IsA(UsdGeom.Cube) or not p.HasAPI(UsdPhysics.CollisionAPI):
                raise ValueError('Original cavity cube missing '+path)
            if p.GetAttribute('physics:collisionEnabled').Get() is False:raise ValueError('Disabled cavity wall')
            tf=cache.GetLocalToWorldTransform(p)*inv
            axes=np.asarray(tf)[:3,:3]
            if np.any(np.sum(abs(axes)>1e-7,axis=0)!=1) or np.any(np.sum(abs(axes)>1e-7,axis=1)!=1):
                raise ValueError('Cavity must be axis aligned in the native oven frame')
            h=UsdGeom.Cube(p).GetSizeAttr().Get()/2
            pts=np.asarray([tf.Transform(Gf.Vec3d(x,y,z)) for x in (-h,h) for y in (-h,h) for z in (-h,h)])
            return tuple(map(float,pts.min(0))),tuple(map(float,pts.max(0)))
        self.walls=tuple(rules.WallBox(p,*box(p)) for p in rules.INNER_WALL_FACES)
        self.front_y=box(ROOT+'/door/col_door')[1][1]
        walls={w.collider.rsplit('/',1)[1]:w for w in self.walls}
        self.lo=np.asarray([walls['col_left'].maximum_m[0],self.front_y,walls['col_bottom'].maximum_m[2]])
        self.hi=np.asarray([walls['col_right'].minimum_m[0],walls['col_back'].minimum_m[1],walls['col_top'].minimum_m[2]])
        if not np.all(self.hi>self.lo):raise ValueError('Invalid measured oven cavity')
        plate=stage.GetPrimAtPath(PLATE)
        self.plate=Geometry(stage,PLATE,meal.destination) if plate and plate.IsActive() else None
        self.plate_footprint=None
        if self.plate:
            points,_=collect_body_points(stage,self.plate.target_body)
            self.plate_footprint=ConvexHull(points[:,:2]).equations
        # USD's unset COM is (-inf,-inf,-inf): PhysX computes the actual COM.
        # initialize() must bind the native result before any carrying facts.
        self.box_com=None

    def cavity_points(self,vertices,target_pose,oven_pose):
        return inverse(transform(vertices,target_pose),oven_pose)

    def inside(self,vertices,target_pose,oven_pose):
        v=self.cavity_points(vertices,target_pose,oven_pose)
        return bool(np.all(v>=self.lo-1e-7) and np.all(v<=self.hi+1e-7))

    def clearance(self,vertices,target_pose,oven_pose):
        v=self.cavity_points(vertices,target_pose,oven_pose)
        return float(self.front_y-v[:,1].max())

    def plate_on_tray(self,poses,pairs):
        if self.plate is None:return False
        # Any actual plate load on ANY part of the designated large tray is
        # forbidden, not only a plate fully inside its flat floor.
        for paths,samples in pairs:
            for i in (0,1):
                if under(paths[i],PLATE) and under(paths[1-i],self.plate.destination):
                    if any((1 if i==0 else -1)*normal[2]>.5 for normal,_,_ in samples):return True
        return False

    def score_contacts(self,rows,oven_pose):
        if not rows:return ()
        positions=inverse([r['position_m'] for r in rows],oven_pose)
        return tuple(rules.ContactPoint(*r['colliders'],tuple(map(float,p)),
            float(np.linalg.norm(r['impulse_Ns']))) for r,p in zip(rows,positions))


def current_contact_rows(report,decode):
    """Keep all current points, including zero impulse, for wall-event continuity."""
    headers,data=report[:2];rows=[]
    def xyz(v):return [float(v.x),float(v.y),float(v.z)]
    for h in headers:
        n=int(h.num_contact_data);start=int(h.contact_data_offset)
        if n<0 or start<0 or start+n>len(data):raise ValueError('Malformed native contacts')
        for point in data[start:start+n]:
            row=dict(colliders=(str(decode(h.collider0)),str(decode(h.collider1))),
                position_m=xyz(point.position),normal=xyz(point.normal),impulse_Ns=xyz(point.impulse),
                separation_m=float(point.separation))
            if not np.isfinite(row['position_m']+row['normal']+row['impulse_Ns']+[row['separation_m']]).all():
                raise ValueError('Nonfinite native contact point')
            rows.append(row)
    return rows


def support_rows(pairs):
    """Measured load history from sleeping-aware native Contacts, not invented force."""
    return [dict(colliders=paths,normal=list(n),position_m=list(p),impulse_Ns=list(f),
        had_positive_impulse=True) for paths,samples in pairs for n,p,f in samples]
