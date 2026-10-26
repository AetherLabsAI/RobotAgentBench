"""Read-only FK of the recorded fridge first observation; no drive changes."""
import json
import math
from pathlib import Path
SOURCE=Path(__file__).with_name('source_ready_pose.json')

def read():
    value=json.loads(SOURCE.read_text())
    if value['status']!='source_fridge_recorded_initial_posture' or value['source_frame_index']!=0:
        raise ValueError('Expected fridge first observation')
    if len(value['targets_rad'])!=34 or not all(math.isfinite(v) for v in value['targets_rad'].values()):
        raise ValueError('Incomplete fridge joint observation')
    return value

def load(stage):
    import numpy as np
    from pxr import Usd,UsdGeom,UsdPhysics
    from scipy.spatial.transform import Rotation
    targets=read()['targets_rad']
    robot='/World/Robot'
    joints=[];seen=set()
    def frame(p,i):
        m=np.eye(4)
        q=p.GetAttribute('physics:localRot'+str(i)).Get()
        m[:3,:3]=Rotation.from_quat([*q.GetImaginary(),q.GetReal()]).as_matrix()
        m[:3,3]=p.GetAttribute('physics:localPos'+str(i)).Get()
        return m
    for p in Usd.PrimRange(stage.GetPrimAtPath(robot)):
        if not any(p.IsA(t) for t in (UsdPhysics.RevoluteJoint,UsdPhysics.PrismaticJoint,UsdPhysics.FixedJoint)):
            continue
        a,b=[p.GetRelationship('physics:body'+str(i)).GetTargets() for i in (0,1)]
        if len(a)!=1 or len(b)!=1:continue
        motion=np.eye(4)
        if not p.IsA(UsdPhysics.FixedJoint):
            name=p.GetName()
            if name not in targets:raise ValueError('Missing source joint '+name)
            seen.add(name);v=targets[name]
            scale=1 if p.IsA(UsdPhysics.PrismaticJoint) else math.pi/180
            lo,hi=[p.GetAttribute('physics:'+k+'Limit').Get() for k in ('lower','upper')]
            if lo is None or hi is None or not lo*scale<=v<=hi*scale:
                raise ValueError('Source reset violates existing joint limit: '+name)
            axis=np.eye(3)['XYZ'.index(p.GetAttribute('physics:axis').Get())]
            if p.IsA(UsdPhysics.PrismaticJoint):motion[:3,3]=axis*v
            else:motion[:3,:3]=Rotation.from_rotvec(axis*v).as_matrix()
        joints.append((str(a[0]),str(b[0]),frame(p,0)@motion@np.linalg.inv(frame(p,1))))
    if seen!=set(targets):raise ValueError('Source joints do not match room')
    cache=UsdGeom.XformCache()
    links={robot+'/base_link':np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(robot+'/base_link'))).T}
    for _ in joints:
        n=len(links)
        for a,b,delta in joints:
            if a in links and b not in links:links[b]=links[a]@delta
        if len(links)==n:break
    rigid={str(p.GetPath()) for p in Usd.PrimRange(stage.GetPrimAtPath(robot)) if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    if rigid-set(links):raise ValueError('Incomplete source FK')
    return {path.rsplit('/',1)[-1]:[m.T] for path,m in links.items()}
