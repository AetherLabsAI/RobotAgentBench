"""Read approved assemblies in their standard frame; task-local geometry snapshot."""
import hashlib,json,sys,itertools
from pathlib import Path
import numpy as np
from pxr import Usd,UsdGeom,Gf

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
REG=ROOT/'layout_task_activation_v1/task_scene_registration_v1'
MEMBERS={'coffee':('coffee_machine','coffee_cup_blue_compact','coffee_bag'),
         'fryer':('air_fryer','paper_plate_task2','cake_plain_yellow','cake_frosted_white')}

def zone(p):
    fx,fy=p['front'];x,y=p['origin']
    return Gf.Matrix4d(fy,-fx,0,0,fx,fy,0,0,0,0,1,0,x,y,p['top'],1)

def points(stage,path,reference):
    result=[];cache=UsdGeom.XformCache()
    for p in Usd.PrimRange(stage.GetPrimAtPath(path)):
        im=UsdGeom.Imageable(p)
        if not im or im.ComputeVisibility()=='invisible' or im.ComputePurpose() not in ('default','render'):continue
        if p.IsA(UsdGeom.Mesh):vs=p.GetAttribute('points').Get()
        elif p.IsA(UsdGeom.Cube):
            h=UsdGeom.Cube(p).GetSizeAttr().Get()/2;vs=list(itertools.product((-h,h),repeat=3))
        else:continue
        if vs is not None and len(vs):
            m=np.asarray(cache.GetLocalToWorldTransform(p)*reference)
            result.extend((np.c_[np.asarray(vs),np.ones(len(vs))]@m)[:,:3])
    if not result:raise ValueError('Missing visible geometry '+path)
    return np.asarray(result)

def main():
    scene=REG/'room_demo_v6/room.usda';report=REG/'room_demo_v6/composition.json';plan=REG/'room_preview_plan_seed42.json'
    s=Usd.Stage.Open(str(scene));rows=json.loads(report.read_text())['objects']
    old={p['part_id']:p['placement'] for p in json.loads(plan.read_text())['kitchen']['parts']}
    groups={k:[r for r in rows if r['id'] in ids] for k,ids in MEMBERS.items()}
    groups['microwave']=[dict(id='microwave',path='/World/Assets/DiningTableMicrowave')]
    out={}
    for kind,members in groups.items():
        out[kind]=[]
        for row in members:
            a=points(s,row['path'],zone(old[kind]).GetInverse())
            out[kind].append(dict(id=row['id'],path=row['path'],min=a.min(0).tolist(),max=a.max(0).tolist()))
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (scene,report,plan)}
    target=HERE/'station_geometry_v2.json'
    target.write_text(json.dumps(dict(assemblies=out,source_hashes=hashes,method='all_visible_mesh_vertices_in_approved_standard_frame'),indent=2))
    print(target)

if __name__=='__main__':main()
