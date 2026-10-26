"""Read-only actual USD inventory for the integrated room demonstration."""
import json
from pathlib import Path
from pxr import Usd, UsdGeom, UsdPhysics
import numpy as np

ROOT = Path(__file__).resolve().parent
ASSETS = Path('/home/lingjun/robot_agent_arena/api-protocol/all_assets')
SOURCE = '/mnt/agent-bench-data/lingjun/releases/living50-candidate-20260919/layouts/ID_01_LIVING_STRAIGHT.usdc'

def inspect(stage, p):
    cache = UsdGeom.BBoxCache(0, ['default', 'render'])
    r = cache.ComputeWorldBound(p).ComputeAlignedRange()
    x = UsdGeom.XformCache().GetLocalToWorldTransform(p) if p.IsA(UsdGeom.Xformable) else None
    points=[]
    for q in Usd.PrimRange(p):
        if q.IsA(UsdGeom.Mesh) and UsdGeom.Imageable(q).ComputeVisibility() != 'invisible':
            if UsdGeom.Imageable(q).ComputePurpose() not in ('default','render'): continue
            vs=UsdGeom.Mesh(q).GetPointsAttr().Get()
            if vs:
                m=np.array(UsdGeom.XformCache().GetLocalToWorldTransform(q))
                points.extend((np.c_[np.array(vs),np.ones(len(vs))]@m)[:,:3])
    exact=None if not points else [np.min(points,axis=0).tolist(),np.max(points,axis=0).tolist()]
    shelf_z = None
    if 'ShelvingUnit' in str(p.GetPath()) and points:
        z,count=np.unique(np.round(np.array(points)[:,2],4),return_counts=True)
        shelf_z=sorted(zip(z.tolist(),count.tolist()),key=lambda x:-x[1])[:25]
    return dict(path=str(p.GetPath()), type=p.GetTypeName(), exact_bounds=exact, shelf_z=shelf_z,
                bounds=None if r.IsEmpty() else [list(r.GetMin()), list(r.GetMax())],
                matrix=None if x is None else [list(row) for row in x],
                bodies=[str(q.GetPath()) for q in Usd.PrimRange(p) if q.HasAPI(UsdPhysics.RigidBodyAPI)],
                children=[str(q.GetPath()) for q in p.GetChildren()])

stage = Usd.Stage.Open(SOURCE)
report = {'source': SOURCE, 'world': [], 'assets': {}, 'furniture_details': []}
horizontal=[]
for q in Usd.PrimRange(stage.GetPrimAtPath('/World/Assets/ShelvingUnit')):
    if not q.IsA(UsdGeom.Mesh):continue
    mesh=UsdGeom.Mesh(q);vs=mesh.GetPointsAttr().Get()
    if not vs:continue
    m=np.array(UsdGeom.XformCache().GetLocalToWorldTransform(q));pts=(np.c_[np.array(vs),np.ones(len(vs))]@m)[:,:3]
    idx=np.array(mesh.GetFaceVertexIndicesAttr().Get());offset=0
    for n in mesh.GetFaceVertexCountsAttr().Get():
        poly=pts[idx[offset:offset+n]];offset+=n
        if np.ptp(poly[:,2])<1e-5 and n>=3:
            area=abs(np.dot(poly[:,0],np.roll(poly[:,1],1))-np.dot(poly[:,1],np.roll(poly[:,0],1)))/2
            if area>.003:horizontal.append(dict(path=str(q.GetPath()),z=float(poly[0,2]),area=float(area),polygon=poly[:,:2].tolist()))
report['shelf_horizontal_faces']=horizontal
for root in ['/World', '/World/Assets']:
    report['world'].extend(inspect(stage,p) for p in stage.GetPrimAtPath(root).GetChildren())
for p in stage.Traverse():
    path = str(p.GetPath())
    if any(s in path.lower() for s in ('shelf','shelving','coffeetable','microwave')) and len(p.GetPath().GetPrefixes()) < 7:
        report['furniture_details'].append(inspect(stage,p))
for entry in sorted(ASSETS.glob('*/asset.usda')):
    asset = Usd.Stage.Open(str(entry))
    report['assets'][entry.parent.name] = inspect(asset,asset.GetDefaultPrim())
(ROOT/'room_demo_inventory_v1.json').write_text(json.dumps(report,indent=2))
print(json.dumps({'world': [{k:v for k,v in row.items() if k in ('path','bounds','children')} for row in report['world']],
                  'asset_count': len(report['assets'])}))
