import json
import numpy as np
from scipy.spatial.transform import Rotation
from pxr import Usd,UsdGeom,UsdPhysics
from audit_microwave_placement_v1 import HERE
from audit_task2_workspace import joint_frame

row=json.loads((HERE/'microwave_selected50_placement_v2.json').read_text())['records'][5]
print(row['source'],flush=True)
s=Usd.Stage.Open(row['source']+'/initial_scene.usda');xf=UsdGeom.XformCache()
def w(p):return np.asarray(xf.GetLocalToWorldTransform(s.GetPrimAtPath(p))).T
rows=[]
for p in Usd.PrimRange(s.GetPrimAtPath('/World/Robot')):
    if not any(p.IsA(t) for t in (UsdPhysics.FixedJoint,UsdPhysics.RevoluteJoint,UsdPhysics.PrismaticJoint)):continue
    rel=[p.GetRelationship('physics:body'+str(i)).GetTargets() for i in (0,1)]
    if len(rel[0])!=1 or len(rel[1])!=1:continue
    a,b=str(rel[0][0]),str(rel[1][0]);f0,f1=joint_frame(p,0),joint_frame(p,1)
    m=np.linalg.inv(w(a)@f0)@w(b)@f1;motion=np.eye(4)
    if not p.IsA(UsdPhysics.FixedJoint):
        axis='XYZ'.index(p.GetAttribute('physics:axis').Get())
        if p.IsA(UsdPhysics.RevoluteJoint):motion[:3,:3]=Rotation.from_rotvec(Rotation.from_matrix(m[:3,:3]).as_rotvec()[axis]*np.eye(3)[axis]).as_matrix()
        else:motion[axis,3]=m[axis,3]
    prediction=w(a)@f0@motion@np.linalg.inv(f1)
    rows.append(dict(joint=p.GetName(),body=b,max_matrix_error=float(np.max(abs(prediction-w(b)))),position_error_m=float(np.linalg.norm(prediction[:3,3]-w(b)[:3,3]))))
print(json.dumps(sorted(rows,key=lambda r:-r['max_matrix_error'])[:15],indent=2),flush=True)
