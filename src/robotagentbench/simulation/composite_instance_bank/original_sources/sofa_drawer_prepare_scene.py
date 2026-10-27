"""Build a task-owned initial scene from an existing unified room, without scores.

The room generator still chooses the layout. This wrapper only initializes the
one selected sofa object and restores the already approved closed drawer slide.
It does not register a new public Short Task or pretend its score rules exist.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def add_existing_drawer(stage):
    from pxr import Usd,UsdGeom,Gf
    from evaluation.tasks.tidy_sofa.initial_state import BASE,SLIDE_LAYER
    from evaluation.tasks.tidy_sofa.specs import COFFEE_TABLE,DRAWER,DRAWER_JOINT,DRAWER_INITIAL_LIP_M
    def world(scene,path):
        return UsdGeom.XformCache().GetLocalToWorldTransform(scene.GetPrimAtPath(path))
    reference=Usd.Stage.OpenMasked(str(BASE),Usd.StagePopulationMask([COFFEE_TABLE]))
    m=world(reference,COFFEE_TABLE).GetInverse()*world(stage,COFFEE_TABLE)
    layer=stage.GetEditTarget().GetLayer()
    layer.subLayerPaths=[str(SLIDE_LAYER)]+list(layer.subLayerPaths)
    joint=stage.GetPrimAtPath(DRAWER_JOINT)
    p=joint.GetAttribute('physics:localPos0').Get();q=joint.GetAttribute('physics:localRot0').Get()
    joint.GetAttribute('physics:localPos0').Set(Gf.Vec3f(*m.Transform(Gf.Vec3d(*p))))
    joint.GetAttribute('physics:localRot0').Set(Gf.Quatf(m.RemoveScaleShear().ExtractRotationQuat()*Gf.Quatd(q.GetReal(),Gf.Vec3d(*q.GetImaginary()))))
    initial=world(stage,DRAWER).ExtractTranslation()
    expected=m.Transform(world(reference,DRAWER).ExtractTranslation())
    if (initial-expected).GetLength()>=.003:
        raise ValueError('Drawer initial transform incompatible with unchanged slide')
    contract=dict(source_to_layout_matrix=[list(r) for r in m],initial_origin_m=list(initial),
        initial_lip_m=DRAWER_INITIAL_LIP_M,slide_source=str(SLIDE_LAYER),slide_sha256=sha(SLIDE_LAYER))
    stage.GetPrimAtPath(DRAWER).SetCustomDataByKey('openend:drawerCalibration',json.dumps(contract))
    return contract


def prepare(manifest_path, *, asset, seed, output):
    from pxr import Usd,UsdGeom
    from short_tasks.scene import _scene_imports,_candidate,verify_manifest
    from .prepare_initial import choose
    if asset not in ('wallet','notebook_green'):
        raise ValueError('Only the two traced Short 5 targets are supported')
    original=json.loads(Path(manifest_path).read_text())
    verify_manifest(original)
    out=Path(output).resolve()
    if not out.is_relative_to(HERE) or out.exists():
        raise ValueError('A fresh output inside this task directory is required')
    # Task-independent unified room as an immutable source, never change it.
    initial_id='sofa_to_drawer_init_'+str(seed)
    if any(initial_id in r['path'] for r in original['objects']):
        raise ValueError('Use an untouched unified-room template, not this reset itself')
    request=dict(schema='openend.scene_requirements.request.v2',seed=seed,
        layout_id=original['layout_id'],tasks=[dict(id=initial_id,
        type='notebook_transfer' if asset=='notebook_green' else 'wallet_transfer',source='sofa')],extras=[])
    with _scene_imports():
        import room_request
        resolved=room_request.resolve(request)
    data=_candidate(request,resolved,original['layout_id'],0)
    if data['plan']['status']!='planar_candidate':
        raise ValueError('Unified source-object placement is infeasible')
    out.mkdir(parents=True)
    stage=Usd.Stage.CreateNew(str(out/'room.usda'))
    stage.GetRootLayer().subLayerPaths=[original['scene']]
    UsdGeom.SetStageMetersPerUnit(stage,1);UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
    # Background/infrastructure is retained. Remove prior task targets only in
    # this fresh initialization layer; a new task must not inherit old ownership.
    removed=[]
    for row in original['objects']:
        if row.get('task_id') and row.get('source'):
            stage.OverridePrim(row['path']).SetActive(False);removed.append(row['path'])
    hashes=deepcopy(original['source_hashes'])
    hashes[original['scene']]=sha(original['scene'])
    reg=ROOT/'layout_task_activation_v1/task_scene_registration_v1'
    old=list(sys.path)
    try:
        sys.path[:0]=[str(ROOT/'openend/tasks/scene_requirements'),str(reg)]
        from build_living import compose
        rows,_,extra,_=compose(stage,data,ROOT,reg,hashes,sha)
        for row in rows:
            if row.get('task_id')==initial_id:
                row['task_id']='sofa_to_drawer'
    finally:
        sys.path[:]=old
    if 'openend_drawer' not in extra:
        if original.get('openend_drawer'):
            extra['openend_drawer']=deepcopy(original['openend_drawer'])
        else:
            extra['openend_drawer']=add_existing_drawer(stage)
    stage.GetRootLayer().Save()
    manifest=deepcopy(original)
    manifest.update(extra)
    manifest.update(seed=seed,scene=str(out/'room.usda'),scene_sha256=sha(out/'room.usda'),
        objects=rows+[r for r in original['objects'] if r['path'] not in removed
            and r['path'] not in {a['path'] for a in rows}],
        tasks=[],registration={},source_hashes=hashes,
        initialization_only=True,public_short_task_registered=False,
        native_physics_validated=False,runtime_started=False)
    for row in manifest['objects']:
        if row.get('task_id')=='sofa_to_drawer':
            row['destination_role']='coffee_table.drawer'
    receipt=choose(stage,manifest,seed=seed)
    manifest['sofa_to_drawer_station_initialization']=receipt
    manifest['source_hashes'].update(receipt['reference_hashes'])
    for name,value in [('manifest.json',manifest),('spawn.json',receipt),
        ('ready_pose.json',dict(targets_rad=receipt['targets_rad'],seed=seed,
            status='source_drawer_return_initialization_requires_native_acceptance'))]:
        (out/name).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
    return dict(output=str(out),asset=asset,layout_id=manifest['layout_id'],
        initialization_prepared=True,public_short_task_registered=False,
        robot_room_geometry_checked=True,native_physics_validated=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--asset',choices=['wallet','notebook_green'],required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    print(json.dumps(prepare(args.manifest,asset=args.asset,seed=args.seed,output=args.output)))
