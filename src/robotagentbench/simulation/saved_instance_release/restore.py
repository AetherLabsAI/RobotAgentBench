"""Restore a selected compact initial state with the original CPU scene owner.

The small release contains sampled state, not per-episode scene snapshots.
This command never samples a new room plan or starts Isaac/physics/a model.
Unsupported task-specific owners are rejected, rather than silently using a
generic scene. Original task code, physical parameters and scoring are retained.
"""
from copy import deepcopy
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import uuid
import re

BANK_SCHEMA='robotagentbench.compact_initial_states.v1'
STATE_SCHEMA='robotagentbench.compact_initial_state.v1'


def _sha(path):
    with Path(path).open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def _json(path,value):
    with Path(path).open('x')as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')


def _read(path):
    def unique(pairs):
        result={}
        for key,value in pairs:
            if key in result:raise ValueError('Duplicate configuration field')
            result[key]=value
        return result
    return json.loads(Path(path).read_text(),object_pairs_hook=unique,
                      parse_constant=lambda value:(_ for _ in()).throw(ValueError('Nonfinite configuration value')))


def _member(root,row):
    relative=row['path'];parts=PurePosixPath(relative)
    if (not isinstance(relative,str)or parts.is_absolute()or'..'in parts.parts
            or parts.as_posix()!=relative or'\\'in relative):
        raise ValueError('Canonical relative saved-state member required')
    path=root/relative
    if path.resolve(strict=True)!=path or not path.is_file()or path.stat().st_size!=row['bytes']or _sha(path)!=row['sha256']:
        raise ValueError('Saved-state member differs from its release index: '+relative)
    return path


def select(release,task,episode):
    """Verify one compact state and its small owner attachments; no SDK import."""
    if not isinstance(task,str)or not re.fullmatch('[A-Za-z0-9_-]{1,100}',task):
        raise ValueError('A published task identifier is required')
    path=Path(release).absolute()
    if path.resolve(strict=True)!=path or not path.is_file():raise ValueError('Canonical compact release file required')
    value=_read(path)
    if value.get('schema')!=BANK_SCHEMA:raise ValueError('A compact initial-state release is required')
    if type(episode)is not int or not 1<=episode<=30:raise ValueError('Select an episode from 1 through 30')
    rows=[r for r in value['instances']if r['task_id']==task and r['episode']==episode]
    if len(rows)!=1:raise ValueError('One exact task and episode must be selected')
    row=rows[0];state_path=_member(path.parent,row['state']);state=_read(state_path)
    if (state.get('schema')!=STATE_SCHEMA or state.get('task_id')!=task
            or state.get('episode')!=episode or state.get('actual_seed')!=row['actual_seed']):
        raise ValueError('Saved state identity differs from its index')
    attachments=[]
    for record in row.get('attachments',[]):
        attachments.append(dict(record,source=str(_member(path.parent,record))))
    if len({a['path']for a in attachments})!=len(attachments):raise ValueError('Duplicate saved owner attachment')
    return dict(release=path,row=row,state=state,state_path=state_path,attachments=attachments)


def _owner(source,state):
    options=state.get('launch_options',{});request=state.get('task_parameters',{})
    name=options.get('task_name',request.get('task_name'))
    if state.get('kind')=='cloth' or state.get('source_profile')=='cloth':
        return source,'','cloth'
    if state.get('source_profile')in('failure_recovery','atomic_coffee'):
        return source,'','specialized'
    recipe=state.get('scene_recipe')
    if not isinstance(recipe,dict)or not all(key in recipe for key in('layout','plan')):
        raise ValueError('An original saved scene recipe is required; no planner/resampling fallback is available')
    if options.get('schema')=='openend.coding_room.launch.v1':
        if name and name.startswith('long_task_'):
            return source/'short_tasks/agent_probe_v1/long_task_eval_setup_v1/generated_rooms','input.json','long'
        if name in('openend','task_discovery'):
            base=source/'short_tasks/agent_probe_v1/cap_v1/native_tools/openend_control_adapter_v1'
            if name=='task_discovery':base=base/'task_discovery_v1/four_scenarios_min10_v1'
            return base/'generated_rooms','input.json','room'
        raise ValueError('Unsupported room-owner identity')
    if options.get('schema')!='openend.short_task.launch.v1':raise ValueError('Unsupported original launch schema')
    if name in('select_drink','select_snack','select_trash','select_sofa'):
        return source/'short_tasks/agent_probe_v1/runs','placement.json','context'
    policy=recipe.get('task_scene_policy',{}).get('task_name')
    if policy in('microwave_heat_take','wipe_table'):
        return source/'short_tasks/tasks'/policy/'execution','placement.json','task'
    if policy in('stock_wooden_tray','stock_wooden_tray_interference'):
        return source/'short_tasks/tasks'/policy/'generated_rooms','placement.json','task'
    if policy:raise ValueError('A task-specific saved composition owner is required')
    return source/'short_tasks/agent_probe_v1/runs','placement.json','ordinary'


def describe(release,runtime,task,episode,output,*,owner_config=None):
    selected=select(release,task,episode)
    base=Path(runtime).resolve(strict=True);metadata=_read(base/'runtime.json')
    if metadata.get('schema')!='robotagentbench.local_installation.v1':raise ValueError('A configured Runtime is required')
    relative=Path(metadata['source_root']).relative_to(metadata['runtime']);source=base/relative
    if source.resolve(strict=True)!=source or not source.is_dir():raise ValueError('Canonical installed source required')
    state=selected['state'];state.setdefault('source_profile',selected['row'].get('source_profile'))
    state.setdefault('kind',selected['row'].get('kind'))
    profile=state.get('source_profile')
    if profile and profile!=metadata['profile']:raise ValueError('Install the saved state\'s matching source profile first')
    destination=Path(output).absolute()
    if destination.resolve()!=destination or destination.exists()or not destination.parent.is_dir():
        raise ValueError('Fresh canonical receipt output with an existing parent required')
    owner,filename,mode=_owner(source,state)
    if owner.resolve()!=owner or not owner.is_relative_to(source):raise ValueError('Dedicated Runtime owner path required')
    return dict(selected=selected,runtime=base,source=source,metadata=metadata,output=destination,
                owner=owner,recipe_filename=filename,mode=mode,owner_config=owner_config)


def _seal_specialized(operator,family,task,launch_path,files,destination):
    """Seal newly derived inputs through the unchanged original owner gates.

    Existing source, environment and startup pins remain byte-for-byte intact.
    Only this selected task's generated input bindings receive their real new
    hashes. The old launch identity stays in the saved-state provenance.
    """
    from robotagentbench.simulation.specialized import config,operator as owner
    rebound=deepcopy(operator);dependencies=rebound['dependencies']
    root='compact_saved_inputs';index=0
    while root in dependencies['roots']:
        index+=1;root='compact_saved_inputs_'+str(index)
    input_root=launch_path.parent
    dependencies['roots'][root]=str(input_root)
    keys=[];launch_key=None
    for index,path in enumerate(dict.fromkeys([launch_path,*files])):
        path=Path(path)
        if not path.is_relative_to(input_root):raise ValueError('Derived specialized inputs must share their fresh owner directory')
        key=root+'.'+str(index)
        dependencies['resources'][key]=dict(root=root,path=path.relative_to(input_root).as_posix(),sha256=_sha(path),kind='asset')
        keys.append(key)
        if path==launch_path:launch_key=key
    task_config=rebound['families'][family]['tasks'][task]
    task_config.update(launch=launch_key,inputs=keys)
    operator_path=destination/'operator.json';_json(operator_path,rebound)
    # read_operator/prepare verify all original profile/source/environment pins,
    # plus the exact new task input set; there is no scorer or admission bypass.
    checked=config.read_operator(operator_path)
    dispatch=owner.prepare(checked,operator_path,family,task,destination/'dispatch.json')
    return dict(owner_config=str(operator_path),owner_config_sha256=_sha(operator_path),
        dispatch=str(dispatch),launch=str(launch_path),launch_sha256=_sha(launch_path),
        native_output=str(Path(task_config['output_parent'])/('compact_'+task+'_'+uuid.uuid4().hex[:12])))


def _rebind_saved_json(value,paths,hashes):
    """Relocate generated inputs, not source identities or physical values."""
    if isinstance(value,list):return [_rebind_saved_json(v,paths,hashes)for v in value]
    if not isinstance(value,dict):return paths.get(value,value)if isinstance(value,str)else value
    result={paths.get(k,k):_rebind_saved_json(v,paths,hashes)for k,v in value.items()}
    for key,old in value.items():
        if isinstance(old,str)and old in hashes:
            if key+'_sha256'in value:result[key+'_sha256']=hashes[old]
        # A source-layer map can include generated layer dependencies. Only an
        # explicitly saved/copied layer gets a new digest; code pins never do.
        if key in hashes and isinstance(old,str)and re.fullmatch('[0-9a-f]{64}',old):
            result[paths[key]]=hashes[key]
    return result


def _write_saved_json_sidecars(sidecars,paths,hashes):
    """Resolve generated file hashes without rejecting harmless path cycles."""
    data={old:_read(row['source'])for old,row in sidecars.items()}
    previous=dict(hashes)
    for _ in range(len(data)+2):
        values={old:_rebind_saved_json(value,paths,previous)for old,value in data.items()}
        calculated={old:hashlib.sha256((json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()).hexdigest()
                    for old,value in values.items()}
        if all(previous.get(old)==pin for old,pin in calculated.items()):break
        previous.update(calculated)
    else:raise ValueError('Original owner JSON has a cyclic content-hash dependency')
    files=[]
    for old,value in values.items():
        target=Path(paths[old]);_json(target,value)
        if _sha(target)!=calculated[old]:raise ValueError('Derived sidecar serialization changed')
        hashes[old]=calculated[old];files.append(target)
    return files


def _specialized_saved(prepared,operator,family,task,original_launch,destination):
    """Restore exact small owner-authored layers and JSON, without sampling."""
    from pxr import Sdf
    selected=prepared['selected'];state=selected['state']
    layers=[a for a in selected['attachments']if a.get('role')in('scene_root','scene_layer')]
    roots=[a for a in layers if a.get('role')=='scene_root']
    initial=state.get('saved_initialization',{}).get('initial')
    if len(roots)!=1 or not isinstance(initial,dict):
        raise ValueError('Saved specialized owner requires its exact small scene_root/scene_layer attachments and original initial JSON; no seed resampling is permitted')
    if any(not isinstance(a.get('original_path'),str)for a in layers):
        raise ValueError('Saved specialized layers require original composition identities')
    work=original_launch.parent/('compact_saved_'+uuid.uuid4().hex[:12]);work.mkdir()
    paths={a['original_path']:str(work/('scene_'+str(i)+'.usda'))for i,a in enumerate(layers)}
    hashes={};files=[]
    for attachment in layers:
        target=Path(paths[attachment['original_path']]);layer=Sdf.Layer.CreateNew(str(target))
        original=Sdf.Layer.FindOrOpen(attachment['source'])
        if original is None:raise ValueError('Saved original specialized layer could not be read')
        layer.TransferContent(original)
        dependencies=list(layer.GetCompositionAssetDependencies())
        for dependency in dependencies:
            absolute=str((Path(attachment['original_path']).parent/dependency).resolve())if not Path(dependency).is_absolute()else dependency
            replacement=paths.get(dependency,paths.get(absolute))
            # TransferContent changes the anchoring directory. An immutable
            # external relative reference must keep its original absolute
            # namespace identity, not be resolved relative to the fresh output.
            if replacement is None and not Path(dependency).is_absolute():replacement=absolute
            if replacement is not None:layer.UpdateCompositionAssetDependency(dependency,replacement)
        layer.Save();hashes[attachment['original_path']]=_sha(target);files.append(target)
    root=Path(paths[roots[0]['original_path']])
    original_initial=state.get('manifest_template',{}).get('initial',state.get('manifest_template',{}).get('initial_file'))
    if not isinstance(original_initial,str):raise ValueError('Original dedicated initial identity is missing')
    paths[original_initial]=str(work/'initial.json')
    sidecars={a['original_path']:a for a in selected['attachments']
        if a.get('role')=='owner_json'and isinstance(a.get('original_path'),str)
        and a['original_path']!=original_initial}
    for index,old in enumerate(sidecars):paths[old]=str(work/('owner_'+str(index)+'.json'))
    files.extend(_write_saved_json_sidecars(sidecars,paths,hashes))
    initial=_rebind_saved_json(initial,paths,hashes)
    initial.update(scene=str(root),scene_sha256=_sha(root))
    _json(work/'initial.json',initial);hashes[original_initial]=_sha(work/'initial.json');files.append(work/'initial.json')
    request=state['task_parameters'];_json(work/'request.json',request);files.append(work/'request.json')
    manifest=_rebind_saved_json(state['manifest_template'],paths,hashes)
    manifest.update(scene=str(root),scene_sha256=_sha(root))
    initial_key='initial_file'if family=='atomic_coffee'else'initial'
    manifest.update({initial_key:str(work/'initial.json'),'initial_sha256':_sha(work/'initial.json')})
    _json(work/'manifest.json',manifest);files.append(work/'manifest.json')
    launch=deepcopy(state['launch_options'])
    launch.update(request=str(work/'request.json'),manifest=str(work/'manifest.json'),scene=str(root))
    for name in('ready_pose','sensors','spawn'):
        original=state.get('saved_initialization',{}).get(name)
        if original is not None:
            target=work/(name+'.json');_json(target,_rebind_saved_json(original,paths,hashes))
            files.append(target);launch[name]=str(target)
    for name in('request','manifest','scene'):
        if name+'_sha256'in launch:launch[name+'_sha256']=_sha(launch[name])
    _json(work/'launch.json',launch)
    return _seal_specialized(operator,family,task,work/'launch.json',files,destination)


def _dedicated(prepared):
    """Prepare saved sidecars using each existing dedicated owner contract."""
    selected=prepared['selected'];state=selected['state'];destination=prepared['output']
    configured=prepared.get('owner_config')
    if configured is None:raise ValueError('The saved task requires its existing dedicated --owner-config')
    configured=Path(configured).absolute()
    if configured.resolve(strict=True)!=configured:raise ValueError('Canonical dedicated operator configuration required')
    task=state['task_id'];result=dict(schema='robotagentbench.restored_compact_initial_state.v1',task_id=task,
        episode=state['episode'],actual_seed=state['actual_seed'],state=str(selected['state_path']),
        state_sha256=_sha(selected['state_path']),owner_config=str(configured),owner_config_sha256=_sha(configured),
        owner_attachments=selected['attachments'],launch=None,native_started=False,model_called=False,
        resampled=False,physics_scoring_modified=False)
    if prepared['mode']=='cloth':
        from robotagentbench.simulation.cloth import config,launcher,randomization
        operator=launcher.read_operator(configured);workspace=launcher.workspace_for(operator)
        if not any(destination.is_relative_to(root)and destination!=root for root in workspace.input_roots):
            raise ValueError('Cloth preparation output must be inside an operator-declared input root')
        launch=launcher.launch_for(operator,task)
        files=[a for a in selected['attachments']if a['path'].endswith('.npz')]
        records=[]
        for attachment in selected['attachments']:
            if attachment['path'].endswith('.json'):
                record=_read(attachment['source'])
                if record.get('schema')==randomization.SCHEMA:records.append(record)
        if len(files)!=1 or len(records)!=1:raise ValueError('Exact saved cloth NPZ and variation record attachments required')
        record=deepcopy(records[0]);randomization.verify_record(record,task=task)
        if record['seed']!=state['actual_seed']or files[0]['sha256']!=record['initial_sha256']:
            raise ValueError('Cloth attachment identity differs from the selected saved instance')
        # Rebind original bytes through the explicit operator. Every replacement
        # must have the same already-declared SHA, never a refreshed source pin.
        roots=launch['owner']['directories'];pins=launch['owner']['hashes']
        catalog=Path(roots['asset_root'])/'code/agent_bench/real2sim/production/house_layouts.json'
        appearance=Path(roots['original_root'])/'code/agent_bench/real2sim/production/livingroom_environment_randomization.py'
        if _sha(catalog)!=record['layout']['catalog_sha256']or _sha(appearance)!=record['appearance']['source_sha256']:
            raise ValueError('Configured cloth layout/appearance bytes differ from saved sources')
        record['layout']['catalog']=str(catalog);record['appearance']['source']=str(appearance)
        known={str(catalog):_sha(catalog),str(appearance):_sha(appearance),**pins};rebound={}
        for original,pin in record['source_hashes'].items():
            candidates=[p for p,h in known.items()if h==pin]
            if original in candidates:chosen=original
            elif candidates:chosen=sorted(candidates)[0]
            else:raise ValueError('Saved cloth dependency is not bound by the dedicated operator')
            workspace.input(chosen);rebound[chosen]=pin
        record['source_hashes']=rebound
        destination.mkdir();initial=destination/'initial.npz';variation=destination/'variation.json'
        with initial.open('xb')as stream:stream.write(Path(files[0]['source']).read_bytes())
        _json(variation,record)
        launch['variation']=dict(record=str(variation),record_sha256=_sha(variation),initial=str(initial),initial_sha256=_sha(initial))
        launch_path=config.prepare(destination/'launch.json',launch,workspace=workspace)
        result.update(launch=str(launch_path),launch_sha256=_sha(launch_path),status='prepared_dedicated_cloth_launch',
                      native_output=str(workspace.output_root/('compact_'+task+'_'+uuid.uuid4().hex[:12])))
    else:
        from robotagentbench.simulation.specialized import config,operator as specialized
        operator=config.read_operator(configured);family=state['source_profile']
        config.selection(operator,family,task)
        # A caller may already have prepared this exact selected instance with
        # its original owner. Reuse it only when its source identity matches.
        supplied=operator['families'][family]['tasks'][task]
        deps=config.dependencies(operator);launch_path=deps.file(supplied['launch'])
        expected=state.get('provenance',{}).get('source_launch_sha256')
        destination.mkdir()
        if expected and _sha(launch_path)==expected:
            dispatch=specialized.prepare(operator,configured,family,task,destination/'dispatch.json')
            result.update(launch=str(launch_path),launch_sha256=expected,dispatch=str(dispatch),status='selected_original_dedicated_launch')
        else:
            result.update(_specialized_saved(prepared,operator,family,task,launch_path,destination),
                          status='prepared_derived_dedicated_launch',source_launch_sha256=expected)
    _json(destination/'selection.json',result)
    return result


def _portable(value,source,runtime,metadata):
    """Resolve saved asset/source identities through the installed Runtime only."""
    if isinstance(value,list):return [_portable(v,source,runtime,metadata)for v in value]
    if isinstance(value,dict):return {k:_portable(v,source,runtime,metadata)for k,v in value.items()}
    if not isinstance(value,str)or not value.startswith('/')or value.startswith('/World'):return value
    if '/all_assets/'in value:
        tail=value.split('/all_assets/',1)[1];parts=PurePosixPath(tail)
        if '..'in parts.parts:raise ValueError('Invalid saved asset identity')
        target=(source/'all_assets'/tail).resolve(strict=True)
        if not target.is_relative_to(runtime):raise ValueError('Saved asset leaves the installed Runtime')
        return str(target)
    # Original compatibility identities remain names, not host installation
    # requirements. Resolve an exact installed suffix where one is declared.
    candidates=[]
    for relative,recorded in metadata.get('resources',{}).items():
        # Bundled profile/library members retain source-relative identities,
        # while saved recipes may still name their original logical prefix.
        for prefix in('profiles/'+metadata['profile']+'/', 'libraries/generic_current/'):
            if relative.startswith(prefix):
                tail=relative[len(prefix):]
                if value.endswith('/'+tail):
                    path=source/tail
                    if path.is_file()and path.resolve(strict=True)==path:candidates.append(path)
        if relative.endswith(value.lstrip('/')):
            path=runtime/relative
            if path.is_file():candidates.append(path.resolve(strict=True))
    if len(set(candidates))==1:return str(candidates[0])
    return value


def _portable_state(state,source,runtime,metadata,mode):
    rebound=_portable(state,source,runtime,metadata)
    if mode=='long':
        # The original Long owner compares this complete recipe with its
        # deterministic seeded recipe, including logical layout identities.
        # The verified Runtime namespace supplies those read-only identities;
        # only execution-side paths outside the recipe are relocated here.
        rebound['scene_recipe']=deepcopy(state['scene_recipe'])
    return rebound


def _refresh_launch_hashes(launch,scene):
    for key in('scene','manifest','request','ready_pose'):
        if key+'_sha256'in launch:
            # Some original short-task launches pin the manifest's scene
            # without duplicating its path as a launch schema field.
            path=launch.get(key,scene if key=='scene'else None)
            if path is None:raise ValueError('Missing launch input for '+key+'_sha256')
            launch[key+'_sha256']=_sha(path)


def _typed(kind,value):
    from pxr import Gf,Vt
    vectors={'double3':Gf.Vec3d,'float3':Gf.Vec3f,'half3':Gf.Vec3h}
    if kind=='matrix4d':
        flat=[n for row in value for n in row]if len(value)==4 and isinstance(value[0],list)else value
        if len(flat)!=16:return (_ for _ in()).throw(ValueError('Expected sixteen matrix elements'))
        return Gf.Matrix4d(*flat)
    if kind in vectors:return vectors[kind](*value)
    if kind=='token[]':return Vt.TokenArray(value)
    if kind in('quatd','quatf','quath'):
        constructor={'quatd':Gf.Quatd,'quatf':Gf.Quatf,'quath':Gf.Quath}[kind]
        if isinstance(value,dict)and set(value)=={'real','imaginary'}:
            vector={'quatd':Gf.Vec3d,'quatf':Gf.Vec3f,'quath':Gf.Vec3h}[kind]
            return constructor(value['real'],vector(*value['imaginary']))
        if isinstance(value,list)and len(value)==2:return constructor(value[0],value[1])
        if isinstance(value,list)and len(value)==4:return constructor(value[0],value[1],value[2],value[3])
        raise ValueError('Original quaternion encoding required')
    if kind in('double','float','half','bool','token','string','int'):return value
    raise ValueError('Unsupported saved initial-state attribute type: '+kind)


def _rebind_generated_stain(state,composed):
    """Relocate the identical seeded stain, never replace its scoring layout."""
    saved=state.get('manifest_template',{}).get('wipe_table',{})
    record=saved.get('randomization')
    if record is None:return state
    current=composed.get('wipe_table',{})
    other=current.get('randomization',{})
    if record.get('version')!='openend.wipe_stain.randomization.v1':
        raise ValueError('Unknown saved stain generator')
    transient={'layout_file','randomization_file','randomization_sha256','randomization'}
    if ({k:v for k,v in saved.items()if k not in transient}!=
            {k:v for k,v in current.items()if k not in transient}):
        raise ValueError('Recomposed stain geometry or scoring layout differs')
    if ({k:v for k,v in record.items()if k not in ('files','source_hashes')}!=
            {k:v for k,v in other.items()if k not in ('files','source_hashes')}):
        raise ValueError('Recomposed stain seed or appearance differs')
    def sources(value):
        rows=value.get('source_hashes',{})
        result={PurePosixPath(p).name:h for p,h in rows.items()}
        if len(result)!=len(rows)or not result:raise ValueError('Ambiguous stain source identities')
        return result
    if sources(record)!=sources(other):raise ValueError('Stain generator sources changed')
    def members(group):
        parent=PurePosixPath(group['layout_file']).parent
        return {PurePosixPath(p).relative_to(parent).as_posix():(p,h)
                for p,h in group['randomization']['files'].items()}
    before,after=members(saved),members(current)
    required={'asset.usda','asset_layout.json','textures/stain.png'}
    if set(before)!=required or set(after)!=required:
        raise ValueError('Incomplete saved stain output closure')
    paths={}
    for relative,(old,pin)in before.items():
        new,newpin=after[relative]
        if pin!=newpin or _sha(new)!=pin:raise ValueError('Recomposed stain contents differ')
        paths[old]=new
    if (_sha(current['randomization_file'])!=current['randomization_sha256'] or
            _read(current['randomization_file'])!=other):
        raise ValueError('Recomposed stain provenance differs')
    paths[saved['randomization_file']]=current['randomization_file']
    result=_rebind_saved_json(state,paths,{})
    result['manifest_template']['wipe_table']=deepcopy(current)
    return result


def _ops(stage,path,attributes,*,joint=False):
    from pxr import Sdf
    prim=stage.GetPrimAtPath(path)
    if not prim:raise ValueError('Saved initial-state prim is absent: '+path)
    for name,record in attributes.items():
        if not(name=='xformOpOrder'or name.startswith('xformOp:')or(joint and name.startswith('state:')and name.endswith(':physics:position'))):
            raise ValueError('Only saved transforms/initial joint positions may be restored')
        typ=Sdf.ValueTypeNames.Find(record['type'])
        if not typ:raise ValueError('Unknown original USD value type')
        attribute=prim.CreateAttribute(name,typ,custom=False)
        # A recorded Get()==None is an effective absence, not an empty array
        # or an invented transform. A strong value block retains that absence
        # if the fresh composed weaker layer happens to provide a default.
        if record['value']is None:attribute.Block()
        else:attribute.Set(_typed(record['type'],record['value']))


def _restore_saved_background(stage,row,state,source_hashes):
    """Copy a saved sofa distractor from its pinned original prototype.

    These props were added after the original scene recipe was composed.
    Reuse the recorded prototype and pose; do not rerun the background sampler.
    """
    from pxr import Sdf,Usd,UsdPhysics
    contract=state.get('manifest_template',{}).get('sofa_background',{})
    if (contract.get('version')!='sofa_background.v1'
            or row.get('id') not in contract.get('assets',[])
            or row.get('role')!='background' or row.get('task_role')!='background'
            or row.get('scoring_target') is not False or row.get('required_by')!=[]
            or row.get('task_id') is not None or row.get('source')!='sofa'
            or row.get('path')!='/World/RoomEpisode/SofaBackground_'+row['id']):
        return None
    scene=row.get('prototype_scene');path=row.get('prototype_path')
    if not isinstance(scene,str) or not isinstance(path,str) or not source_hashes:
        raise ValueError('Saved background requires its pinned original prototype')
    prototype=Usd.Stage.OpenMasked(scene,Usd.StagePopulationMask([path]))
    if prototype is None or not prototype.GetPrimAtPath(path):
        raise ValueError('Saved background prototype is absent: '+str(path))
    pins=set(source_hashes.values());used={}
    for layer in prototype.GetUsedLayers():
        if not layer.realPath:continue
        digest=_sha(layer.realPath)
        if digest not in pins:
            raise ValueError('Background prototype layer is not pinned by this Runtime: '+layer.realPath)
        used[layer.realPath]=digest
    if not used:raise ValueError('Saved background has no pinned source layers')
    destination=stage.GetRootLayer()
    Sdf.CreatePrimInLayer(destination,row['path'])
    if not Sdf.CopySpec(prototype.Flatten(),path,destination,row['path']):
        raise ValueError('Saved background prototype copy failed')
    prim=stage.GetPrimAtPath(row['path'])
    bodies=[p for p in Usd.PrimRange(prim) if p.HasAPI(UsdPhysics.RigidBodyAPI)
            and p.GetAttribute('physics:rigidBodyEnabled').Get() is not False]
    if len(bodies)!=1 or bodies[0].GetAttribute('physics:kinematicEnabled').Get() is True:
        raise ValueError('Saved background is not the original single dynamic body')
    return used


def _restore_scene(scene,state,composed,out,*,source_hashes=None):
    from pxr import Gf,Usd,UsdGeom
    original=Usd.Stage.Open(str(scene))
    if original is None:raise ValueError('Original composed scene cannot be opened')
    stage=Usd.Stage.CreateNew(str(out));stage.GetRootLayer().subLayerPaths=[str(scene)]
    # USD stage metadata is root-owned, not inherited from sublayers. Copy
    # the original owner's units/axes/time metadata, never choose new values.
    metadata={key:original.GetMetadata(key) for key in
              ('upAxis','metersPerUnit','timeCodesPerSecond','framesPerSecond')}
    for key,value in metadata.items():
        if value is not None:stage.SetMetadata(key,value)
    if original.GetDefaultPrim():
        stage.SetDefaultPrim(stage.GetPrimAtPath(original.GetDefaultPrim().GetPath()))
    old={row['path']:row for row in composed.get('objects',[])}
    saved_paths={row['path']for row in state['objects']}
    for path in old:
        if path not in saved_paths and not any(p.startswith(path+'/')for p in saved_paths):
            stage.OverridePrim(path).SetActive(False)
    # A task-local strong layer references the already composed original asset
    # subtree when its wrapper differs; shared layers are never modified.
    background_sources={};background_count=0
    for row in state['objects']:
        path=row['path'];prim=stage.GetPrimAtPath(path)
        if not prim:
            copied=_restore_saved_background(stage,row,state,source_hashes)
            if copied is not None:
                background_sources.update(copied);background_count+=1
                prim=stage.GetPrimAtPath(path)
        if not prim:
            matches=[r for r in old.values()if r.get('id')==row['id']and r.get('task_id')==row.get('task_id')]
            if len(matches)!=1:matches=[r for r in old.values()if r.get('id')==row['id']]
            cup_variant=False
            if not matches and state['task_id']=='prepare_coffee'and row['id'].startswith('coffee_cup_'):
                # The original post-composition coffee reset swaps this one
                # approved asset reference. Retain its composed outer wrapper
                # and use the exact saved style, without sampling another one.
                randomization=state.get('saved_randomization',{})
                style=randomization.get('cup_style',randomization.get('proposal',{}).get('cup_style'))
                if style in('blue_compact','cream_classic','green_tall')and row['id']=='coffee_cup_'+style:
                    matches=[r for r in old.values()if r.get('id','').startswith('coffee_cup_')]
                    cup_variant=True
            if len(matches)!=1:raise ValueError('Missing original composed subtree for '+row['id'])
            prim=stage.DefinePrim(path)
            prim.GetReferences().AddReference(str(scene),matches[0]['path'])
            prim.SetActive(True)
            if cup_variant:
                from pxr import Sdf
                asset=stage.GetPrimAtPath(path+'/Asset')
                if not asset:raise ValueError('Original coffee cup wrapper is absent')
                stage.OverridePrim(asset.GetPath()).GetReferences().SetReferences([Sdf.Reference(row['entry'])])
        if not prim.IsActive():prim.SetActive(True)
    parents=state.get('parent_transform_ops',{})
    for path,ops in sorted(parents.items(),key=lambda row:len(row[0].split('/'))):_ops(stage,path,ops)
    for row in state['objects']:
        if row.get('transform_frame')=='parent':_ops(stage,row['path'],row['authored_transform_ops'])
        elif row.get('transform_frame')=='world':
            values=row['world_matrix_row_major_4x4']
            if len(values)!=16:raise ValueError('Saved world matrix must contain sixteen numbers')
            prim=stage.GetPrimAtPath(row['path']);parent=UsdGeom.XformCache().GetLocalToWorldTransform(prim.GetParent())
            UsdGeom.Xformable(prim).MakeMatrixXform().Set(Gf.Matrix4d(*values)*parent.GetInverse())
        else:raise ValueError('Original transform frame must be explicit')
    for path,ops in state.get('object_subtree_transform_ops',{}).items():_ops(stage,path,ops)
    for path,attributes in state.get('object_initial_joint_state',{}).items():_ops(stage,path,attributes,joint=True)
    stage.GetRootLayer().Save()
    errors=[]
    for row in state['objects']:
        prim=stage.GetPrimAtPath(row['path'])
        if not prim or not prim.IsActive():raise ValueError('Restored target is absent/inactive')
        if row['transform_frame']=='world':
            actual=UsdGeom.XformCache().GetLocalToWorldTransform(prim)
            error=max(abs(actual[i][j]-row['world_matrix_row_major_4x4'][i*4+j])for i in range(4)for j in range(4))
            if error>1e-7:raise ValueError('Restored saved world transform differs')
            errors.append(error)
    return dict(objects=len(state['objects']),maximum_world_matrix_error=max(errors,default=0.),
                saved_world_precision_preserved=True,stage_metadata=metadata,
                restored_background_objects=background_count,background_source_hashes=background_sources)


def _context_layers(prepared,state,base_scene,work):
    """Apply the original selection delta to its exact recomposed base."""
    from pxr import Sdf
    attachments=prepared['selected']['attachments']
    layers=[a for a in attachments if a.get('role')in('scene_root','scene_layer')]
    roots=[a for a in layers if a.get('role')=='scene_root']
    original_base=state.get('saved_base_scene',state.get('saved_initialization',{}).get('saved_base_scene'))
    if len(roots)!=1 or not isinstance(original_base,str):
        raise ValueError('Context restoration requires the original selection layers and exact saved_base_scene identity')
    if any(not isinstance(a.get('original_path'),str)for a in layers):
        raise ValueError('Original selection layer identities are required')
    paths={original_base:str(base_scene)};hashes={original_base:_sha(base_scene)};files=[]
    for index,attachment in enumerate(layers):paths[attachment['original_path']]=str(work/('selection_'+str(index)+'.usda'))
    for attachment in layers:
        target=Path(paths[attachment['original_path']]);layer=Sdf.Layer.CreateNew(str(target))
        original=Sdf.Layer.FindOrOpen(attachment['source'])
        if original is None:raise ValueError('Original selection delta cannot be opened')
        layer.TransferContent(original)
        for dependency in list(layer.GetCompositionAssetDependencies()):
            absolute=dependency if Path(dependency).is_absolute()else str((Path(attachment['original_path']).parent/dependency).resolve())
            replacement=paths.get(absolute)
            if replacement is None:
                replacement=_portable(absolute,prepared['source'],prepared['runtime'],prepared['metadata'])
            layer.UpdateCompositionAssetDependency(dependency,replacement)
        layer.Save();hashes[attachment['original_path']]=_sha(target);files.append(target)
    sidecars=[a for a in attachments if a.get('role')=='owner_json']
    if any(not isinstance(a.get('original_path'),str)for a in sidecars):raise ValueError('Original selection sidecar identities required')
    for index,attachment in enumerate(sidecars):paths[attachment['original_path']]=str(work/('selection_owner_'+str(index)+'.json'))
    files.extend(_write_saved_json_sidecars({a['original_path']:a for a in sidecars},paths,hashes))
    state['manifest_template']=_rebind_saved_json(state['manifest_template'],paths,hashes)
    state['launch_options']=_rebind_saved_json(state['launch_options'],paths,hashes)
    return Path(paths[roots[0]['original_path']]),{str(path):_sha(path)for path in files}


def restore(release,runtime,task,episode,output,*,python_executable=None,owner_config=None):
    """CPU preparation only. Return a real original-owner launch and receipt."""
    prepared=describe(release,runtime,task,episode,output,owner_config=owner_config)
    if prepared['mode']in('cloth','specialized'):return _dedicated(prepared)
    selected=prepared['selected'];state=deepcopy(selected['state']);source=prepared['source'];base=prepared['runtime']
    state=_portable_state(state,source,base,prepared['metadata'],prepared['mode'])
    # Original compose() starts a fresh Python builder. Its import namespace
    # must follow the same explicit source/USD/PhysX/SDK order as this CPU
    # process; changing sys.path alone does not reach that child interpreter.
    inherited=os.environ.get('PYTHONPATH','').split(os.pathsep)
    os.environ['PYTHONPATH']=os.pathsep.join(dict.fromkeys([str(source),*(p for p in inherited if p)]))
    from robotagentbench.simulation.source_tree import NativeSourceTree
    tree=NativeSourceTree(source,prepared['metadata']['source_files'])
    owner=prepared['owner'];owner.mkdir(parents=True,exist_ok=True)
    work=owner/('compact_'+task+'_'+str(episode).zfill(2)+'_'+uuid.uuid4().hex[:12]);work.mkdir()
    _json(work/'restore_intent.json',dict(state_sha256=_sha(selected['state_path']),original_seed=state['actual_seed'],native_started=False,model_started=False))
    with tree.activated():
        scene_module=tree.import_module('short_tasks.scene');recipe=deepcopy(state['scene_recipe'])
        if prepared['mode']=='ordinary'or(prepared['mode']=='context'and not recipe.get('task_scene_policy')):
            composed=scene_module.LocalFriendProvider(python_executable=python_executable).compose(recipe,work/'composed')
        else:
            compose_work=work
            if prepared['mode']=='context':
                policy=recipe.get('task_scene_policy',{}).get('task_name')
                if policy!='stock_wooden_tray':raise ValueError('Unsupported original selection base-composer owner')
                compose_work=source/'short_tasks/tasks'/policy/'generated_rooms'/('compact_context_'+uuid.uuid4().hex[:12])
                compose_work.mkdir(parents=True)
            input_path=compose_work/prepared['recipe_filename'];_json(input_path,recipe)
            if prepared['mode']=='long':_json(work/'long_request.json',state['task_parameters'])
            command=[python_executable or sys.executable,str(source/'openend/tasks/scene_requirements/build_selected_room.py'),str(input_path)]
            result=subprocess.run(command,cwd=str(source/'openend/tasks/scene_requirements'),capture_output=True,text=True,timeout=180)
            if result.returncode:raise RuntimeError('Original saved-recipe composer rejected input: '+result.stderr[-2500:])
            composed=_read(compose_work/'composition.json')
        scene_module.verify_manifest(composed)
    state=_rebind_generated_stain(state,composed)
    scene=Path(composed['scene']);restored=work/'restored.usda'
    extra_hashes={}
    if prepared['mode']=='context':scene,extra_hashes=_context_layers(prepared,state,scene,work)
    # Background prototypes can compose pinned shared-room library layers as
    # well as source-root assets. Their installation pins are retained too.
    prototype_pins={**prepared['metadata']['source_files'],
                    **prepared['metadata'].get('library_files',{})}
    checks=_restore_scene(scene,state,composed,restored,source_hashes=prototype_pins)
    extra_hashes.update(checks['background_source_hashes'])
    request=state['task_parameters'];ready=deepcopy(state['ready_pose'])
    targets=state['robot']['targets_rad']
    if ready.get('targets_rad')!=targets:raise ValueError('Saved ready posture differs from saved robot state')
    _json(work/'request.json',request);_json(work/'ready_pose.json',ready)
    manifest=deepcopy(state['manifest_template']);manifest['scene']=str(restored);manifest['scene_sha256']=_sha(restored)
    manifest['source_hashes']=dict(composed['source_hashes']);manifest['source_hashes'][str(scene)]=_sha(scene)
    manifest['source_hashes'].update(extra_hashes)
    manifest['source_hashes'][str(restored)]=_sha(restored)
    _json(work/'manifest.json',manifest)
    launch=deepcopy(state['launch_options'])
    launch.update(manifest=str(work/'manifest.json'),request=str(work/'request.json'),ready_pose=str(work/'ready_pose.json'),initial_base_xyyaw=state['robot']['initial_base_xyyaw'])
    # These keys are optional for the original generic room schema. If the
    # selected original owner carried them, keep their semantics but bind the
    # newly derived scene, never a stale server episode path.
    if 'scene'in launch:launch['scene']=str(restored)
    _refresh_launch_hashes(launch,restored)
    # Rebind immutable code identities to the installed profile, not to an old
    # server directory or a generated source snapshot.
    if launch['schema']=='openend.coding_room.launch.v1':
        launch['source_hashes']={str(source/relative):pin for relative,pin in prepared['metadata']['source_files'].items()}
        keys=['manifest','request','ready_pose']
        saved=state.get('saved_initialization',{})
        if 'spawn'in saved:
            spawn=deepcopy(saved['spawn']);spawn.update(scene_sha256=manifest['scene_sha256'],initial_base_xyyaw=launch['initial_base_xyyaw'])
            _json(work/'spawn.json',spawn);launch['spawn']=str(work/'spawn.json');keys.append('spawn')
        launch['bindings']={key:_sha(launch[key])for key in keys}
    _json(work/'launch.json',launch)
    public_binding=None
    if state.get('public_input')is not None:
        if launch['schema']=='openend.coding_room.launch.v1':
            public=state['public_input'];public=dict(instruction=public.get('instruction',public.get('prompt')),context=public['context'])
            _json(work/'public_input.json',public)
            public_binding=dict(schema='robotagentbench.public_input_binding.v1',source=str(work/'public_input.json'),sha256=_sha(work/'public_input.json'),launch_sha256=_sha(work/'launch.json'))
            _json(work/'public_input_binding.json',public_binding)
        elif prepared['mode']=='context':
            # The original registered selection owner obtains its question
            # from the unchanged request/question bank, not a new launch key.
            _json(work/'public_input.json',state['public_input'])
        else:raise ValueError('Separate public context requires its original room or selection owner')
    extra=[]
    if request.get('task_name')=='stock_wooden_tray_interference':
        from robotagentbench.simulation.diagnostic_instance_bank.relocation_resources import original_resource_aliases
        extra=original_resource_aliases(base)
    prepared['output'].mkdir()
    result=dict(schema='robotagentbench.restored_compact_initial_state.v1',task_id=task,episode=episode,actual_seed=state['actual_seed'],
        runtime=str(base),source_root=str(source),launch=str(work/'launch.json'),launch_sha256=_sha(work/'launch.json'),
        public_input_binding=str(work/'public_input_binding.json')if public_binding else None,
        native_output=str(work/'native'),readonly_resource_aliases=extra,state_sha256=_sha(selected['state_path']),
        checks=checks,resampled=False,physics_scoring_modified=False,native_started=False,model_called=False)
    _json(prepared['output']/'selection.json',result)
    return result


def _sdk_handoff(site,argv):
    from robotagentbench.example import sdk_paths
    if sys.version_info[:2]!=(3,11):raise ValueError('Use the original CPython3.11 CPU USD environment for restoration')
    site,usd,physx=sdk_paths(site);marker='ROBOTAGENTBENCH_COMPACT_CPU_USD_HANDOFF'
    if os.environ.get(marker)==str(site):
        sys.path[:0]=[str(usd),str(physx),str(site)]
        os.environ['PYTHONPATH']=os.pathsep.join(map(str,(usd,physx,site)));return
    environment=dict(os.environ)
    for key in('PYTHONPATH','PYTHONHOME','LD_PRELOAD','LD_AUDIT'):environment.pop(key,None)
    loader=[str(usd/'bin'),str(physx/'bin'),str(Path(sys.executable).resolve().parent.parent/'lib')]
    environment.update({marker:str(site),'PYTHONDONTWRITEBYTECODE':'1','LD_LIBRARY_PATH':os.pathsep.join(loader),
                        'PXR_AR_DEFAULT_SEARCH_PATH':str(site/'isaacsim/kit/mdl/core/Base')})
    package=str(Path(__file__).resolve().parents[3])
    code='import runpy,sys;sys.path.insert(0,'+repr(package)+');sys.argv=["compact-restore",*sys.argv[1:]];runpy.run_module("robotagentbench.simulation.saved_instance_release.restore",run_name="__main__")'
    os.execve(sys.executable,[sys.executable,'-I','-S','-B','-c',code,*argv],environment)


def main(argv=None):
    argv=sys.argv[1:]if argv is None else argv
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release',type=Path,required=True);parser.add_argument('--runtime',type=Path,required=True)
    parser.add_argument('--task',required=True);parser.add_argument('--episode',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--sdk-site',type=Path)
    parser.add_argument('--owner-config',type=Path,help='Existing trusted cloth or specialized native operator; never a model input')
    parser.add_argument('--execute',action='store_true');args=parser.parse_args(argv)
    ready=describe(args.release,args.runtime,args.task,args.episode,args.output,owner_config=args.owner_config)
    if not args.execute:
        print(json.dumps(dict(task_id=args.task,episode=args.episode,actual_seed=ready['selected']['state']['actual_seed'],
            mode='describe',original_composer_owner=ready['mode'],restoration_started=False,native_started=False,model_called=False),indent=2));return 0
    if args.sdk_site and ready['mode']!='cloth':_sdk_handoff(args.sdk_site,argv)
    print(json.dumps(restore(args.release,args.runtime,args.task,args.episode,args.output,owner_config=args.owner_config),indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
