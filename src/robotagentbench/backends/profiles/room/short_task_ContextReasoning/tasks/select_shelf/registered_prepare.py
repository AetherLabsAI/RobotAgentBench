"""Registered preparation reuses source short-task scenes, not their scoring."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

def read(path): return json.loads(Path(path).read_text())
def write(path,value): Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def admit_prepared(request, scene_directory, source_launch):
    from short_tasks import registry
    from short_tasks.agent_probe_v1.native_owner import admit_launch
    from pxr import Usd, Sdf
    directory=Path(scene_directory).resolve()
    plan=registry.resolve(request); name=plan['task_name']
    if not directory.is_relative_to(registry.task_directory(name).resolve()/'checks'):
        raise ValueError('Prepared scene must remain task-local')
    manifest=read(directory/'composition.json'); launch=read(source_launch)
    role='selection_candidate' if name=='select_sofa' else 'touch_candidate'
    for row in manifest['objects']:
        row.pop('task_id',None)
        if row.get('task_role')==role: row['task_id']=name
    manifest.update(tasks=[dict(id=name,type=name)],seed=request['seed'],
        runtime_registration_pending=False,static_scene_only=False,native_physics_validated=False)
    # Resolve and pin every scene dependency, including copied materials/textures.
    stage=Usd.Stage.Open(manifest['scene'])
    if stage is None: raise ValueError('Scene cannot be opened')
    files={l.realPath for l in stage.GetUsedLayers() if l.realPath}
    builtin_modules=set()
    for prim in stage.Traverse():
        for attr in prim.GetAttributes():
            if attr.GetTypeName() not in (Sdf.ValueTypeNames.Asset,Sdf.ValueTypeNames.AssetArray): continue
            value=attr.Get()
            values=value if attr.GetTypeName()==Sdf.ValueTypeNames.AssetArray else [value]
            for asset in values or ():
                if not asset or not asset.path:continue
                if asset.resolvedPath and Path(asset.resolvedPath).is_file():files.add(asset.resolvedPath)
                elif asset.path in ('OmniPBR.mdl','OmniPBR_Opacity.mdl','OmniGlass.mdl'):
                    builtin_modules.add(asset.path)
                elif attr.GetName()=='info:unreal:sourceAsset' and asset.path.startswith('/Game/'):
                    continue  # Import provenance, not the MDL renderer's shader.
                elif attr.GetName().startswith(('inputs:','info:')):
                    raise ValueError('Unresolved runtime asset: '+str(attr.GetPath())+' '+asset.path)
                # Unreal import provenance is not a runtime file dependency.
    hashes={str(Path(p).resolve()):sha(p) for p in files}
    manifest['runtime_builtin_material_modules']=sorted(builtin_modules)
    for filename in ('composition.json','source_robot_launch.json','initial_receipt.json',
                     'inventory_plan_private.json','public_question.json','placement_audit.json'):
        p=directory/filename
        if p.exists(): hashes[str(p)]=sha(p)
    manifest['source_hashes']=hashes
    ready_path=Path(launch['ready_pose'])
    if name!='select_sofa':
        initial=read(directory/'initial_receipt.json')
        ready=read(ready_path);ready['targets_rad']=initial['joint_positions']
        ready_path=directory/'ready_pose_registered.json';write(ready_path,ready)
        station=initial['station']
        launch['initial_base_xyyaw']=[*station['position_xy'],station['yaw_rad']]
    manifest['source_hashes'][str(ready_path)]=sha(ready_path)
    bound=registry.bind(plan,manifest)
    write(directory/'manifest_registered.json',manifest)
    write(directory/'request_registered.json',request)
    # Only the public question is the policy instruction. No private answer path
    # or target identity is added to the action/observation protocol.
    write(directory/'question_registered.json',plan['public_question'])
    launch.update(task_name=name,request=str(directory/'request_registered.json'),
        manifest=str(directory/'manifest_registered.json'),ready_pose=str(ready_path),
        preparation_receipt=str(directory/'preparation_registered.json'))
    admit_launch(launch,directory/'native_admission_not_created')
    write(directory/'launch_registered.json',launch)
    receipt=dict(status='prepared_not_native_accepted',task_name=name,
        launch=str(directory/'launch_registered.json'),native_admission_passed=True,
        native_physics_validated=False,candidate_count=len(bound['selection']['candidates']))
    write(directory/'preparation_registered.json',receipt)
    return receipt

def source_request(request):
    sofa=request['task_name']=='select_sofa'
    base=dict(schema='openend.short_task.request.v1',
        task_name='sofa_to_coffee_table' if sofa else 'stock_wooden_tray',
        rule_version='openend.short_task.'+('sofa_to_coffee_table' if sofa else 'stock_wooden_tray')+'.v1',
        seed=request['seed'])
    if sofa: base['options']=dict(asset='remote_tv')
    else: base['targets']=[dict(category='sprite_c1',quantity=1)]
    if request.get('layout_id') is not None: base['layout_id']=request['layout_id']
    return base

def prepare_registered(request,ready_pose,sensor_file,output):
    from short_tasks import registry
    from short_tasks.agent_probe_v1.prepare_task import prepare as prepare_base
    plan=registry.resolve(request);registry.verify_sources(plan)
    name=plan['task_name'];out=Path(output).resolve()
    root=registry.task_directory(name).resolve()
    if not out.is_relative_to(root/'checks') or out.exists():
        raise ValueError('Fresh task-local checks directory required')
    out.mkdir(parents=True)
    sofa=name=='select_sofa'
    base=source_request(request)
    prepared=prepare_base(base,ready_pose,sensor_file,out/'base_source')
    if sofa:
        from short_task_ContextReasoning.tasks.select_sofa.prepare_initial import build
        build(request['seed'],out/'selection',base_launch=prepared['launch'])
    else:
        from .prepare_preview import build
        build(plan['public_question']['question_id'],request['seed'],out/'selection',base_launch=prepared['launch'])
    result=admit_prepared(request,out/'selection',prepared['launch'])
    write(out/'preparation.json',result)
    return result
