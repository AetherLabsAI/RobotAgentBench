"""Create a task-local layer from an existing clear_table_trash prepared room.

No shared registry changes. Output is a preparation bundle, NOT an admitted launch.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import shutil
import sys

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parents[2]
sys.path.insert(0,str(SOURCE))
from short_task_ContextReasoning.tasks.select_trash.task import sample

def read(path): return json.loads(Path(path).read_text())
def write(path,data): Path(path).write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def visible_bounds(stage,path):
    import numpy as np
    from pxr import Usd,UsdGeom
    cache=UsdGeom.XformCache();points=[]
    for p in Usd.PrimRange(stage.GetPrimAtPath(path)):
        if not p.IsA(UsdGeom.Mesh):continue
        if UsdGeom.Imageable(p).ComputeVisibility()==UsdGeom.Tokens.invisible:continue
        v=np.asarray(UsdGeom.Mesh(p).GetPointsAttr().Get())
        if len(v):points.append((np.c_[v,np.ones(len(v))]@np.array(cache.GetLocalToWorldTransform(p)))[:,:3])
    if not points:raise ValueError('Missing visible geometry: '+path)
    p=np.concatenate(points)
    return np.array([p.min(0),p.max(0)])

def prepare(launch_path, *, seed, category, count, output):
    import numpy as np
    from pxr import Usd,UsdGeom,Gf,UsdPhysics
    from short_tasks.tasks.clear_table_trash import initial_station,prepare_initial
    launch=read(launch_path)
    if launch['task_name']!='clear_table_trash':raise ValueError('Requires original trash short-task preparation')
    manifest=read(launch['manifest'])
    old=[o for o in manifest['objects'] if o.get('task_id')=='trash_group']
    if not old or any(o.get('source')!='coffee_table.objects' for o in old):
        raise ValueError('Original trash table bindings required')
    out=Path(output).resolve()
    if not out.is_relative_to(HERE) or out==HERE or out.exists():
        raise ValueError('Output must be a fresh directory inside this task')
    catalog=read(SOURCE/'short_task_ContextReasoning/asset_pools/trash_selection_v1/catalog.json')
    plan=sample(catalog,seed=seed,category=category,count=count)
    out.mkdir(parents=True)
    write(out/'selection_private.json',plan)
    node=shutil.which('node') or str(Path.home()/'.local/bin/node')
    if not Path(node).is_file():raise ValueError('Existing Node runtime unavailable')
    placement=json.loads(subprocess.check_output([node,str(HERE/'placement.cjs'),str(SOURCE),str(out/'selection_private.json')],text=True))
    base=Usd.Stage.Open(manifest['scene'])
    delta,table,baseline=initial_station.table_context(base,manifest)
    heights=[visible_bounds(base,o['path'])[0,2]-delta[3,2] for o in old]
    if not np.allclose(heights,heights[0],atol=1e-5):raise ValueError('Source trash bottom heights differ')
    scene=out/'room.usda'
    stage=Usd.Stage.CreateNew(str(scene));stage.GetRootLayer().subLayerPaths=[manifest['scene']]
    UsdGeom.SetStageMetersPerUnit(stage,1);UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
    for o in old:stage.OverridePrim(o['path']).SetActive(False)
    UsdGeom.Xform.Define(stage,'/World/ContextSelection')
    byid={p['instance_id']:p for p in placement['placements']}
    rows=[]
    for a in plan['candidates']:
        entry=Path(a['entry'])
        if not entry.is_relative_to((SOURCE/'all_assets').resolve()):
            raise ValueError('Candidate must come from approved all_assets')
        if sha(entry)!=a['sha256']:raise ValueError('Candidate source changed')
        root=UsdGeom.Xform.Define(stage,a['path'])
        child=stage.DefinePrim(a['path']+'/Asset');child.GetReferences().AddReference(str(entry))
        assert sum(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in Usd.PrimRange(child))==1
        p=byid[a['instance_id']]
        m=Gf.Matrix4d(1);m.SetRotate(Gf.Rotation(Gf.Vec3d(0,0,1),np.degrees(p['yaw'])-90))
        op=root.MakeMatrixXform();op.Set(m)
        bounds=visible_bounds(stage,a['path']);centre=(bounds[0]+bounds[1])/2
        t=Gf.Matrix4d(1);t.SetTranslate(Gf.Vec3d(p['world_center'][0]-centre[0],p['world_center'][1]-centre[1],heights[0]-bounds[0,2]))
        op.Set(m*t*Gf.Matrix4d(delta.tolist()))
        actual=visible_bounds(stage,a['path'])
        rows.append(dict(id=a['all_assets_id'],path=a['path'],entry=str(entry),
            instance_id=a['instance_id'],task_id='select_trash',source='coffee_table.objects',
            visible_bounds_world_m=actual.tolist()))
    stage.GetRootLayer().Save()
    prepared=deepcopy(manifest)
    prepared.update(scene=str(scene),seed=seed,objects=[o for o in manifest['objects'] if o not in old]+rows,
                    tasks=[],runtime_registration_pending=True)
    # The old sampler only uses trash_group to identify the source candidates.
    # The temporary view is never published as a disposal task or scored.
    view=deepcopy(prepared)
    for row in view['objects']:
        if row.get('task_id')=='select_trash':row['task_id']='trash_group'
    spawn=prepare_initial.choose(stage,view,seed=seed)
    prepared['scene_sha256']=sha(scene)
    prepared['source_hashes']={str(l.realPath):sha(l.realPath) for l in stage.GetUsedLayers() if l.realPath}
    prepared['source_hashes'].update(spawn['reference_hashes'])
    prepared['source_hashes'].update({str(entry):sha(entry) for entry in
        (HERE/'task.py',HERE/'prepare.py',HERE/'placement.cjs',Path(launch_path),Path(launch['manifest']))})
    write(out/'manifest_private.json',prepared)
    write(out/'spawn.json',spawn)
    write(out/'placement.json',placement)
    write(out/'ready_pose.json',dict(targets_rad=spawn['targets_rad'],source='clear_table_trash.prepare_initial.choose'))
    write(out/'prompt_public.json',plan['public'])
    bundle=dict(status='task_local_prepared_not_registered',task_name='select_trash',
        scene=str(scene),manifest=str(out/'manifest_private.json'),selection=str(out/'selection_private.json'),
        prompt=str(out/'prompt_public.json'),ready_pose=str(out/'ready_pose.json'),
        initial_base_xyyaw=[*spawn['position'][:2],spawn['yaw_rad']],sensor=launch['sensor'],
        source_trash_launch=str(Path(launch_path).resolve()),actual_layout_id=manifest['layout_id'],
        count=len(rows),category=plan['category'],native_physics_validated=False,
        shared_registration_modified=False)
    write(out/'preparation.json',bundle)
    return bundle

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-launch',required=True)
    ap.add_argument('--seed',required=True,type=int)
    ap.add_argument('--category',choices=['recyclable','residual'])
    ap.add_argument('--count',type=int,choices=[4,5])
    ap.add_argument('--output',required=True)
    args=ap.parse_args()
    print(json.dumps(prepare(args.source_launch,seed=args.seed,category=args.category,count=args.count,output=args.output)))
