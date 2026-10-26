"""Eight-item preview using existing sofa policies. No runtime registration."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import sys
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
from short_tasks.tasks.sofa_background_v1.generation import separated
from short_tasks.tasks.sofa_to_coffee_table import initial_station
from short_tasks.tasks.sofa_to_drawer.native_geometry import shape_points
REG=ROOT/'layout_task_activation_v1/task_scene_registration_v1'
sys.path.insert(0,str(REG))
from randomize_other_living_v1 import world, setworld, center_bottom, seat_height_function
from build_room_demo_v1 import exact, T, R, matrix

IDS=('remote_tv','wallet','glasses_case','power_bank','baseball_cap','umbrella','water_bottle','notebook_green')
BASE=ROOT/'short_tasks/agent_probe_v1/runs/sofa_background_native_20261013/sofa_to_coffee_table/prepared/native'
def read(p):return json.loads(Path(p).read_text())
def write(p,obj):Path(p).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def pack(sizes,lo,hi,obstacles,gap,seed):
    rng=random.Random(f'select-sofa:{seed}')
    # Same limits as the existing sofa-background sampler; no relaxed spacing.
    order=sorted(range(len(sizes)),key=lambda i:-np.prod(sizes[i]))
    for attempt in range(100):
        placed=[]; result={}
        for i in order:
            for trial in range(300):
                q=rng.randrange(4);span=np.asarray(sizes[i])[[1,0] if q%2 else [0,1]]
                if np.any(hi-lo<span):continue
                c=np.array([rng.uniform(a,b) for a,b in zip(lo+span/2,hi-span/2)])
                box=np.array([c-span/2,c+span/2])
                if all(separated(box,b,gap) for b in (*placed,*obstacles)):
                    placed.append(box);result[i]={'center_uv':c.tolist(),'yaw_deg':q*90,'bounds_uv':box.tolist()};break
            else:break
        if len(result)==len(sizes):return [result[i] for i in range(len(sizes))]
    return None

def build(seed, output, *, domain='full_approved', base_launch=None):
    from types import SimpleNamespace
    args=SimpleNamespace(seed=seed,domain=domain)
    out=Path(output).resolve()
    if not out.is_relative_to(HERE/'checks'): raise ValueError('Task-local checks output required')
    out.mkdir(parents=True,exist_ok=False)
    launch=read(base_launch or BASE/'launch.json');original=read(launch['manifest'])
    source=Usd.Stage.Open(original['scene'])
    delta,_,_=initial_station.sofa_context(source,original)
    stage=Usd.Stage.CreateNew(str(out/'room.usda'));stage.GetRootLayer().subLayerPaths=[original['scene']]
    UsdGeom.SetStageMetersPerUnit(stage,1);UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
    UsdGeom.Xform.Define(stage,'/World/ContextSofa')
    removed=[o for o in original['objects'] if o.get('source')=='sofa']
    for o in removed:stage.OverridePrim(o['path']).SetActive(False)
    dynamic=read(ROOT/'openend/tasks/all_objects_integration/room_all_dynamic_v2/composition.json')
    sizes=[];rows=[];physics=[]
    for i,asset in enumerate(IDS):
        if asset=='umbrella': scene=HERE/'assets/umbrella_v1/asset.usda'; old='/Asset'
        else:
            matches=[r for r in dynamic['objects'] if r['id']==asset]
            assert len(matches)==1,(asset,len(matches))
            scene=Path(dynamic['scene']);old=matches[0]['path']
        proto=Usd.Stage.OpenMasked(str(scene),Usd.StagePopulationMask([old]))
        path=f'/World/ContextSofa/item_{i:02d}'
        Sdf.CreatePrimInLayer(stage.GetRootLayer(),path)
        assert Sdf.CopySpec(proto.Flatten(),old,stage.GetRootLayer(),path)
        # Copy any referenced external material into this candidate's namespace.
        all_source=Usd.Stage.Open(str(scene))
        for prim in list(Usd.PrimRange(stage.GetPrimAtPath(path))):
            for rel in prim.GetRelationships():
                if not rel.GetName().startswith('material:binding'):continue
                targets=[]
                for target in rel.GetTargets():
                    if target.HasPrefix(path):targets.append(target);continue
                    src=all_source.GetPrimAtPath(target)
                    if not src:raise ValueError('Unresolved prototype material '+str(target))
                    dst=path+'/ExternalMaterials/'+src.GetName()
                    UsdGeom.Scope.Define(stage,path+'/ExternalMaterials')
                    Sdf.CreatePrimInLayer(stage.GetRootLayer(),dst)
                    assert Sdf.CopySpec(all_source.Flatten(),target,stage.GetRootLayer(),dst)
                    targets.append(Sdf.Path(dst))
                rel.SetTargets(targets)
        bodies=[p for p in Usd.PrimRange(stage.GetPrimAtPath(path)) if p.HasAPI(UsdPhysics.RigidBodyAPI) and p.GetAttribute('physics:rigidBodyEnabled').Get() is not False]
        assert len(bodies)==1 and bodies[0].GetAttribute('physics:kinematicEnabled').Get() is not True,(asset,[(str(p.GetPath()),p.IsActive(),p.IsDefined(),p.GetAppliedSchemas(),p.GetAttribute('physics:kinematicEnabled').Get()) for p in Usd.PrimRange.AllPrims(stage.GetPrimAtPath(path))])
        physics.append({'asset':asset,'body':str(bodies[0].GetPath()),'kinematic':False,'source_scene':str(scene),'source_path':old})
        center_bottom(stage,path,[0,0,0],R(-90))
        points=np.vstack([v for _,v,c,visible in shape_points(stage,path) if c or visible])
        # Full geometry (visible or collision), not only origin or visible mesh.
        span=np.ptp(points,axis=0);sizes.append([float(span[1]),float(span[0])])
        rows.append({'id':asset,'path':path,'source':'sofa','instance_id':f'item_{i:02d}','task_role':'selection_candidate','scoring_target':False})
    policies=ROOT/'openend/tasks/scene_requirements/placement_sources'
    policy=read(policies/'sofa-placement-policy-v1.json');structure=read(policies/'sofa-structural-obstacles-v2.json')
    gap=policy['ordinary_object_padding_xy_m']
    obstacles=[np.array(o['bounds']).reshape(2,2) for o in structure['obstacles']]
    target_policy=read(ROOT/'short_tasks/tasks/sofa_to_coffee_table/position_policy.json')
    lo,hi=np.array(target_policy['footprint_bounds_uv_m'])
    if args.domain=='full_approved':
        lo=np.array([-policy['reference_approximate_width_m']/2+policy['edge_margin_m'],policy['edge_margin_m']])
        hi=np.array([policy['reference_approximate_width_m']/2-policy['edge_margin_m'],policy['reference_depth_m']-policy['edge_margin_m']])
    result=pack(sizes,lo,hi,obstacles,gap,args.seed)
    diagnostic={'domain':args.domain,'bounds_uv_m':[lo.tolist(),hi.tolist()],'seed':args.seed,'count':8,'sizes_uv_m':dict(zip(IDS,sizes)),
        'padding_m':gap,'source_physics':physics,'sample_found':result is not None,'native_physics_validated':False}
    write(out/'placement_audit.json',diagnostic)
    if result is None:
        print(json.dumps(diagnostic));raise RuntimeError('Eight objects did not fit approved domain with existing clearance; no relaxation')
    height=seat_height_function(stage);inverse=np.linalg.inv(delta);actual=[]
    for row,placement in zip(rows,result):
        u,v=placement['center_uv'];path=row['path']
        center_bottom(stage,path,[v-.3503563938,-u-.8710,.453],R(placement['yaw_deg']-90))
        setworld(stage,path,world(stage,path)*matrix(delta))
        box=exact(stage,path);setworld(stage,path,world(stage,path)*T([0,0,height(box)-box[0,2]]))
        points=np.vstack([v for _,v,c,visible in shape_points(stage,path) if c or visible])
        canonical=(np.c_[points,np.ones(len(points))]@inverse)[:,:3]
        uv=np.c_[-canonical[:,1]-.871,canonical[:,0]+.3503563938];bounds=np.array([uv.min(0),uv.max(0)])
        assert np.all(bounds[0]>=lo-1e-8) and np.all(bounds[1]<=hi+1e-8)
        assert all(separated(bounds,b,gap) for b in (*actual,*obstacles)),row['id']
        actual.append(bounds);row.update(placement=placement,visible_bounds_world_m=exact(stage,path).tolist(),verified_complete_bounds_uv_m=bounds.tolist())
    stage.GetRootLayer().Save()
    manifest=deepcopy(original);manifest.update(scene=str(out/'room.usda'),scene_sha256=sha(out/'room.usda'),objects=[o for o in original['objects'] if o not in removed]+rows,tasks=[],seed=args.seed)
    for row in manifest['objects']: row.pop('task_id',None)
    manifest['context_selection']={'count':8,'question_independent_placement':True,'status':'static_initial_only_not_registered','source_launch':str(base_launch or BASE/'launch.json')}
    write(out/'composition.json',manifest)
    write(out/'source_robot_launch.json',launch)
    diagnostic.update(sample_found=True,actual_bounds_verified=True,source_room=original['scene'],scene=str(out/'room.usda'),placements=rows)
    write(out/'placement_audit.json',diagnostic)
    print(json.dumps({'status':'static_scene_prepared','scene':manifest['scene'],'count':8,'domain':args.domain}))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--domain',choices=['middle_two','full_approved'],default='full_approved');args=ap.parse_args()
    build(args.seed,HERE/f'checks/initial_{args.domain}_{args.seed}',domain=args.domain)

if __name__=='__main__':main()
