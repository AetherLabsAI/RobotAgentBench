"""Read-only actual USD utensil-to-holder audit; no physical acceptance."""
import json
from pathlib import Path
import numpy as np
from pxr import Usd, UsdGeom

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]

def points(stage,path,frame,cache):
    prim=stage.GetPrimAtPath(path)
    if not prim:raise ValueError('Missing prim: '+path)
    pieces=[]
    for p in Usd.PrimRange(prim,Usd.TraverseInstanceProxies()):
        if not p.IsA(UsdGeom.Mesh) or not p.IsActive():continue
        image=UsdGeom.Imageable(p)
        if image.ComputeVisibility()=='invisible' or image.ComputePurpose() not in ('default','render'):continue
        # Collision geometry is separately audited as the cavity walls.
        if '/Collision/' in str(p.GetPath()):continue
        v=np.array(p.GetAttribute('points').Get(),dtype=float)
        if v.size:
            transform=np.array(cache.GetLocalToWorldTransform(p))@frame
            pieces.append((np.c_[v,np.ones(len(v))]@transform)[:,:3])
    if not pieces:raise ValueError('No visual mesh: '+path)
    return np.concatenate(pieces)

def audit(scene):
    stage=Usd.Stage.Open(str(scene));cache=UsdGeom.XformCache()
    base='/World/RoomEpisode/CutleryHolder/Asset/Object'
    holder=stage.GetPrimAtPath(base)
    if not holder:raise ValueError('Actual holder object required')
    frame=np.linalg.inv(np.array(cache.GetLocalToWorldTransform(holder)))
    bbox=UsdGeom.BBoxCache(0,['default','render','proxy','guide'],False,True)
    rows=[]
    for i,name in enumerate(('knife_0','fork_1','fork_2')):
        walls={}
        for side in ('Left','Right','Front','Back','Bottom'):
            prim=stage.GetPrimAtPath(base+f'/Collision/Slot_{i}_{side}')
            if not prim:raise ValueError('Missing cavity wall')
            r=bbox.ComputeRelativeBound(prim,holder).ComputeAlignedRange()
            walls[side]=np.array([r.GetMin(),r.GetMax()])
        lo=np.array([walls['Left'][1,0],walls['Front'][1,1]])
        hi=np.array([walls['Right'][0,0],walls['Back'][0,1]])
        if np.any(lo>=hi):raise ValueError('Invalid cavity axes')
        rim=min(walls[s][1,2] for s in ('Left','Right','Front','Back'))
        path='/World/RoomEpisode/Cutlery_'+name
        v=points(stage,path,frame,cache)
        below=v[v[:,2]<=rim+1e-9]
        margin=lambda q:float(min((q[:,:2]-lo).min(),(hi-q[:,:2]).min()))
        rows.append(dict(path=path,vertices=len(v),vertices_below_rim=len(below),
            cavity_lo_xy=lo.tolist(),cavity_hi_xy=hi.tolist(),rim_z=float(rim),
            all_vertices_margin_m=margin(v),
            below_rim_margin_m=margin(below) if len(below) else None,
            bottom_visual_gap_m=float(v[:,2].min()-walls['Bottom'][1,2]),
            source_one_mm_horizontal_gate_passed=bool(len(below) and margin(v)>=.001)))
    return rows

def main():
    rows=[]
    for seed in (5,10,8):
        manifest=json.loads((ROOT/f'short_tasks/agent_probe_v1/runs/cutlery_station_{seed}_v5/native/manifest.json').read_text())
        rows.append(dict(seed=seed,layout_id=manifest['layout_id'],checks=audit(manifest['scene'])))
    result=dict(native_physics_validated=False,geometry_only=True,rooms=rows)
    path=HERE/'holder_cavity_audit_v1.json'
    if path.exists():raise ValueError('Keep previous evidence')
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':main()
