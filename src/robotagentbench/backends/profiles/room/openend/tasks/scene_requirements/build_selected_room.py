"""202 CPU USD composition. Only writes a fresh directory inside this task."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
REG=ROOT/'layout_task_activation_v1/task_scene_registration_v1'
sys.path[:0]=[str(ROOT),str(ROOT/'api-protocol'),str(REG)]
from build_room_demo_v1 import exact
from initialize_room_tote_v1 import apply as initialize_tote
from descendant_transforms import rigid_delta, audit

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def world(stage,path): return UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(path))
def zone(p):
    fx,fy=p['front'];x,y=p['origin']
    return Gf.Matrix4d(fy,-fx,0,0,fx,fy,0,0,0,0,1,0,x,y,p['top'],1)
def rectangle(bounds):
    a,b=bounds
    return [[a[0],a[1]],[b[0],a[1]],[b[0],b[1]],[a[0],b[1]]]

def fryer_initial_scope(data, validated_long_request=None):
    """Retain the old single task; admit only the validated Long21 recipe."""
    identities=[(t['id'],t['type']) for t in data['request']['tasks']]
    if identities == [('fryer','fryer')]:return True
    if (data.get('openend_episode_policy') != {
            'schema':'openend.task_owned_generation.v1','owner':'long_task_eval_setup_v1'}
            or identities != [('coffee','coffee'),('fryer','fryer')]
            or not isinstance(validated_long_request,dict)
            or validated_long_request.get('document_id') != '21'):
        return False
    parameters=validated_long_request.get('parameters',{})
    return (data['short_task_fryer_initial'].get('variant') == parameters.get('bread_variant')
            and parameters.get('heating_required') is False
            and parameters.get('finish_on_small_plate') is False)

def main():
    source=Path(sys.argv[1]).resolve()
    data=json.loads(source.read_text());folder=source.parent
    task_policy=data.get('task_scene_policy')
    validated_long_request=None
    if data.get('openend_episode_policy') is not None:
        from short_tasks.agent_probe_v1.cap_v1.native_tools.openend_control_adapter_v1.native_entry import validate_generation_input
        validate_generation_input(source, data['openend_episode_policy'])
        if data['openend_episode_policy'] == {'schema':'openend.task_owned_generation.v1','owner':'long_task_eval_setup_v1'}:
            # The validator above pins this sibling and reproduces the exact
            # trusted recipe. No unvalidated JSON can activate the extra scope.
            validated_long_request=json.loads((folder/'long_request.json').read_text())
    elif task_policy is None:
        from short_tasks.tasks.robot_pose_alignment_v1.temporary_outputs import generated
        if source.parent.parent != generated(HERE/'generated_rooms'): raise ValueError('Input must be in fresh task episode')
    elif task_policy.get('task_name') == 'stock_wooden_tray':
        from short_tasks.tasks.stock_wooden_tray.generation import validate_policy,validate_output
        validate_policy(task_policy,data['requirements'].get('shelf_task_requirements',[]))
        validate_output(source,ROOT)
    elif task_policy.get('task_name') == 'microwave_heat_take':
        from short_tasks.tasks.microwave_heat_take.generation import validate_output
        validate_output(source,ROOT,data)
    elif task_policy.get('task_name') == 'wipe_table':
        from short_tasks.tasks.wipe_table.generation import validate_output
        validate_output(source,ROOT,data)
    else:
        from short_tasks.tasks.stock_wooden_tray_interference.generation import validate_policy,validate_output
        validate_policy(task_policy,data['requirements'].get('shelf_task_requirements',[]))
        validate_output(source,ROOT)
    layout=data['layout'];base=Path(layout['scene'])
    if sha(base)!=layout['scene_sha256']:raise ValueError('Layout geometry snapshot is stale')
    if data['plan']['status']!='planar_candidate':raise ValueError('No complete planar solution')
    approved_path=REG/'room_demo_v6/room.usda'
    approved=Usd.Stage.Open(str(approved_path));flat=approved.Flatten()
    report=json.loads((REG/'room_demo_v6/composition.json').read_text())
    oldplan=json.loads((REG/'room_preview_plan_seed42.json').read_text())
    oldparts={p['part_id']:p['placement'] for p in oldplan['kitchen']['parts']}
    source_hashes={str(base):sha(base),str(approved_path):sha(approved_path)}
    snapshot=json.loads((HERE/'station_geometry_v2.json').read_text())
    for path,digest in snapshot['source_hashes'].items():
        if sha(path)!=digest:raise ValueError('Station planning geometry snapshot is stale: '+path)
    out=folder/'room.usda'
    if out.exists():raise ValueError('Existing episode is immutable')
    stage=Usd.Stage.CreateNew(str(out));stage.GetRootLayer().subLayerPaths=[str(base)]
    UsdGeom.SetStageMetersPerUnit(stage,1);UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
    UsdGeom.Xform.Define(stage,'/World/RoomEpisode')
    removed=[]
    if not any(p['id']=='microwave' for p in data['plan']['placements']):
        path='/World/Assets/DiningTableMicrowave'
        if stage.GetPrimAtPath(path):
            stage.OverridePrim(path).SetActive(False);removed.append(path)
    for path in report['removed_episode_only']:
        if stage.GetPrimAtPath(path):stage.OverridePrim(path).SetActive(False);removed.append(path)
    selections={'coffee':('coffee_machine','coffee_cup_blue_compact','coffee_bag'),
                'fryer':('air_fryer','paper_plate_task2','cake_plain_yellow','cake_frosted_white'),
                'tray':('meal_tray_large','cutlery_holder')}
    background=data['requirements'].get('background_requirements')
    background_devices=set(background['devices']) if background is not None else set()
    if background is not None:
        from short_tasks.agent_probe_v1.cap_v1.native_tools.openend_control_adapter_v1.outing_scene_v3 import policy as outing_policy
        source_hashes[str(Path(outing_policy.__file__))]=sha(outing_policy.__file__)
    meal_requests=data['requirements'].get('meal_task_requirements',[])
    cutlery_requests=[t for t in meal_requests if t['type']=='cutlery_to_meal_tray']
    if cutlery_requests:selections['tray']+=tuple(t['asset'] for t in cutlery_requests)
    rows=[];tasks=[];relative_errors={};station_groups=[]
    for p in data['plan']['placements']:
        kind=p['id'];delta=zone(oldparts[kind]).GetInverse()*zone(p)
        if kind=='microwave':
            path='/World/Assets/DiningTableMicrowave'
            # Replace the episode's old placement at the SAME canonical path.
            # No duplicate microwave and no change to its internal mechanism.
            if not Sdf.CopySpec(flat,path,stage.GetRootLayer(),path):raise ValueError('Microwave copy failed')
            UsdGeom.Xformable(stage.GetPrimAtPath(path)).MakeMatrixXform().Set(world(approved,path)*delta*world(stage,str(stage.GetPrimAtPath(path).GetParent().GetPath())).GetInverse())
            for child in Usd.PrimRange(stage.GetPrimAtPath(path)):
                if not UsdGeom.Xformable(child):continue
                cp=str(child.GetPath())
                a=world(stage,cp)*world(stage,path).GetInverse()
                b=world(approved,cp)*world(approved,path).GetInverse()
                error=float(np.abs(np.asarray(a)-np.asarray(b)).max())
                if error>1e-9:raise ValueError('Microwave internal transform changed: '+cp)
                relative_errors[cp]=error
            rows.append(dict(id='microwave',path=path,role='randomized_infrastructure',placement=p,
                             source_scene=str(approved_path),physical_parameters_unchanged=True))
            if kind in background_devices:
                rows[-1].update(task_role='background_device',background_id='background_'+kind)
            station_groups.append(dict(id=kind,members=[path],support=p['support'],placement=p,
                internal_supports=['/World/Assets/MicrowaveTakePlate','/World/Assets/MicrowavePlateStand']))
            continue
        selected=selections[kind][:1] if kind in background_devices else selections[kind]
        members=[r for r in report['objects'] if r['id'] in selected]
        if kind=='tray':
            # Repeated categories own distinct source slots, never duplicate
            # the first fork. Preserve all authored relative transforms.
            from openend.tasks.meal_tray.scene_wiring import select_cutlery_members
            members=select_cutlery_members(members,cutlery_requests)
        if len(members)!=len(selected):raise ValueError('Missing/ambiguous approved assembly')
        primary=next(r['path'] for r in members if r['id']==selections[kind][0])
        for row in members:
            path=row['path'];Sdf.CreatePrimInLayer(stage.GetRootLayer(),path)
            if not Sdf.CopySpec(flat,path,stage.GetRootLayer(),path):raise ValueError('Copy failed '+path)
            UsdGeom.Xformable(stage.GetPrimAtPath(path)).MakeMatrixXform().Set(world(approved,path)*delta)
            new=dict(row,support=p['support'],support_path=p['support'])
            if kind in background_devices:
                new.pop('task_id',None)
                new.update(task_role='background_device',background_id=p['task_id'])
            else:new['task_id']=p['task_id']
            rows.append(new)
            if row.get('entry'):source_hashes[row['entry']]=sha(row['entry'])
        for row in members:
            a=world(stage,row['path'])*world(stage,primary).GetInverse()
            b=world(approved,row['path'])*world(approved,primary).GetInverse()
            error=float(np.abs(np.asarray(a)-np.asarray(b)).max())
            if error>1e-9:raise ValueError('Assembly relative transform changed')
            relative_errors[row['path']]=error
        if kind!='tray' and kind not in background_devices:tasks.append(dict(id=p['task_id'],type=kind,name='咖啡' if kind=='coffee' else '炸锅',
                          target_path=primary,object_paths=[r['path'] for r in members],placement=p,
                          status='scene_bound',score=None,runtime_started=False))
        station_groups.append(dict(id=kind,members=[r['path'] for r in members],support=p['support'],placement=p))
    from openend.tasks.meal_tray.microwave_initialization import compose as compose_microwave
    if task_policy is not None and task_policy.get('task_name') == 'microwave_heat_take':
        from short_tasks.tasks.microwave_heat_take.generation import compose as compose_short_microwave
        microwave_rows=compose_short_microwave(stage,data,rows,tasks,ROOT,source_hashes,sha)
    else:
        microwave_rows=compose_microwave(stage,meal_requests,rows,ROOT,source_hashes,sha,payload_from_fridge=True)
    rows.extend(microwave_rows)
    if microwave_rows:
        # The requested box is inside this assembly, not a foreign obstacle.
        # Its outer bounds still participate in checks against every OTHER
        # station/wall. Actual cavity containment is checked by the native port.
        next(g for g in station_groups if g['id']=='microwave')['members'].extend(
            r['path'] for r in microwave_rows if r['id'] in ('beef_box','leftover_box'))
    from openend.tasks.meal_tray.scene_wiring import attach_tasks
    from openend.tasks.meal_tray.coffee_initialization import compose as compose_coffee_beans
    bean_rows,coffee_initialization=compose_coffee_beans(stage,[r for r in rows if r.get('task_role')!='background_device'],ROOT,source_hashes,sha)
    rows.extend(bean_rows)
    if bean_rows:
        next(g for g in station_groups if g['id']=='coffee')['members'].extend(r['path'] for r in bean_rows)
        next(t for t in tasks if t['type']=='coffee')['object_paths'].extend(r['path'] for r in bean_rows)
    # Furniture-only all_assets shelf. No unrequested stock/targets are copied.
    shelf=next(r for r in report['objects'] if r['id']=='snack_shelf')
    original=Usd.Stage.Open(str(Path(layout['scene']).parent/'ID_01_LIVING_STRAIGHT.usdc'))
    base_stage=Usd.Stage.Open(str(base))
    source_transform_audit=audit(base_stage)
    if layout['id'] in (18,21,22,23):
        from openend.tasks.layout_repair_v1.shelf import canonical_delta
        delta,shelf_transform_evidence=canonical_delta(original,base_stage,'/World/Assets/ShelvingUnit')
    else:
        delta,shelf_transform_evidence=rigid_delta(original,base_stage,'/World/Assets/ShelvingUnit')
    Sdf.CopySpec(flat,shelf['path'],stage.GetRootLayer(),shelf['path'])
    UsdGeom.Xformable(stage.GetPrimAtPath(shelf['path'])).MakeMatrixXform().Set(world(approved,shelf['path'])*delta)
    if layout['id']==23:
        from openend.tasks.layout_repair_v1.shelf_clearance import clear_wall
        from source_geometry import floor_paths
        shelf_transform_evidence['wall_clearance_translation']=clear_wall(
            stage,shelf['path'],lambda path:exact(stage,path),floor_paths(stage))
    infra=[dict(shelf,role='infrastructure'),initialize_tote(stage)]
    from build_living import compose
    living_rows,living_tasks,extra,pending=compose(stage,data,ROOT,REG,source_hashes,sha)
    rows.extend(living_rows);tasks.extend(living_tasks)
    if background is not None and not data['requirements'].get('wipe_task_requirements'):
        # The obsolete dirty cloth is not an unregistered stain objective.
        # Same episode-only removal used by the existing wiping composer;
        # the original physical tabletop remains untouched.
        obsolete={r['path'] for r in rows if r['id']=='tablecloth_dirty'}
        for path in obsolete:stage.OverridePrim(path).SetActive(False)
        removed.extend(sorted(obsolete))
        rows=[r for r in rows if r['path'] not in obsolete]
    from openend.tasks.wipe_table.scene import compose as compose_wiping
    wipe_rows,wipe_tasks,wipe_extra=compose_wiping(stage,data,ROOT,rows,sha)
    if wipe_rows:
        removed_cloth=set(wipe_extra['wipe_table']['removed_cloth_paths'])
        rows=[r for r in rows if r['path'] not in removed_cloth]
        rows.extend(wipe_rows);tasks.extend(wipe_tasks);extra.update(wipe_extra)
        for r in wipe_rows:source_hashes[r['entry']]=sha(r['entry'])
        layout_file=wipe_extra['wipe_table']['layout_file']
        source_hashes[layout_file]=sha(layout_file)
    from build_fridge import compose as compose_fridge
    fridge_rows,fridge_tasks,fridge_extra,fridge_pending=compose_fridge(stage,data,source_hashes,sha)
    rows.extend(fridge_rows);tasks.extend(fridge_tasks);extra.update(fridge_extra);pending.extend(fridge_pending)
    from build_shelf import compose as compose_shelf
    shelf_rows,shelf_tasks,shelf_extra,shelf_pending=compose_shelf(stage,data,ROOT,REG,source_hashes,sha)
    rows.extend(shelf_rows);tasks.extend(shelf_tasks);extra.update(shelf_extra);pending.extend(shelf_pending)
    if layout['id'] in (18,21,22,23):
        from source_geometry import floor_paths
        floors=floor_paths(stage)
        if not floors:raise ValueError('Replacement shelf requires measured floor')
        station_groups.append(dict(id='canonical_shelf',members=[shelf['path']]+[
            r['path'] for r in shelf_rows if r['path'].startswith('/World/RoomEpisode/Shelf_')],
            support=floors[0],internal_supports=floors[1:]))
    from openend.tasks.meal_tray.salad_initialization import compose as compose_salad
    salad_rows=compose_salad(stage,data,rows,source_hashes,sha)
    rows.extend(salad_rows)
    attach_tasks(meal_requests,rows,tasks)
    from openend.tasks.destination_transfers.registration import attach as attach_destinations
    attach_destinations(stage,data['requirements'],rows,tasks)
    fryer_initial = data.get('short_task_fryer_initial')
    if fryer_initial is not None:
        if (set(fryer_initial) != {'schema','variant'} or
                fryer_initial['schema'] != 'short_tasks.fryer_initial.v1' or
                fryer_initial['variant'] not in ('sugar','no_sugar') or
                not fryer_initial_scope(data,validated_long_request)):
            raise ValueError('Invalid short-task fryer initialization request')
        from short_tasks.tasks.bake_bread import prepare_initial as fryer_reset
        placement, = [p for p in data['plan']['placements'] if p['id']=='fryer']
        extra['fryer_initialization'] = fryer_reset.compose(stage,rows,placement,layout,
            seed=data['request']['seed'],variant=fryer_initial['variant'])
        for dependency in (Path(fryer_reset.__file__),fryer_reset.POLICY):
            source_hashes[str(dependency)] = sha(dependency)
    # Re-open the saved stage before extracting evidence; do not draw uncommitted plans.
    stage.GetRootLayer().Save();stage=None
    stage=Usd.Stage.Open(str(out))
    from station_scene_checks import check as check_station_scene
    clearance=check_station_scene(stage,station_groups,lambda path:exact(stage,path),layout=layout)
    (folder/'station_clearance.json').write_text(json.dumps(clearance,indent=2))
    if not clearance['passed']:raise ValueError('Station scene overlap: '+json.dumps(clearance['failures'][:3]))
    composed_transform_audit=audit(stage)
    broken=[]
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Joint):
            for rel in ('physics:body0','physics:body1'):
                for target in prim.GetRelationship(rel).GetTargets():
                    if not stage.GetPrimAtPath(target):broken.append([str(prim.GetPath()),str(target)])
    if broken:raise ValueError('Broken joint references '+str(broken))
    for row in rows+infra:row['visible_bounds_world_m']=exact(stage,row['path']).tolist()
    furniture=[];unpictured=[]
    names={'Sofa':'沙发','CoffeeTable':'茶几','SmallCoffeeTable':'小茶几','TvTable':'电视柜',
           'CoatRack':'衣架','Task5PackStation':'装包柜','TrashBinRed':'普通垃圾桶',
           'TrashBinTeal':'回收桶','TableclothBasket':'收纳篮','DiningTableMicrowave':'微波炉（随机摆位）'}
    for key,label in names.items():
        path='/World/Assets/'+key
        if stage.GetPrimAtPath(path) and stage.GetPrimAtPath(path).IsActive():
            try:bounds=exact(stage,path).tolist()
            except AssertionError:
                unpictured.append(dict(path=path,reason='no visible mesh/cube geometry'));continue
            furniture.append(dict(path=path,name=label,bounds=bounds,polygon=rectangle(bounds)))
    for row in infra:
        furniture.append(dict(path=row['path'],name='货架' if row['id']=='snack_shelf' else '固定开口包',
                              bounds=row['visible_bounds_world_m'],polygon=rectangle(row['visible_bounds_world_m'])))
    for row in rows:row['polygon']=rectangle(row['visible_bounds_world_m'])
    before_changed=[p for p,h in source_hashes.items() if sha(p)!=h]
    if before_changed:raise ValueError('Read-only source changed '+str(before_changed))
    manifest=dict(schema='openend.generated_room.preview.v1',episode_id=folder.name,
        layout_id=layout['id'],seed=data['request']['seed'],layout_selection=data['layout_selection'],
        scene=str(out),scene_sha256=sha(out),source_layout=str(base),source_sha256=sha(base),
        objects=rows+infra,tasks=tasks,removed_episode_only=removed,
        supports={p['id']:dict(path=p['support'],group=p['group'],actual_top_m=p['top'],origin=p['origin'],front=p['front']) for p in data['plan']['placements']},
        source_hashes=source_hashes,source_unchanged=True,relative_transform_errors=relative_errors,
        station_clearance=clearance,placement_policy='random_microwave_joint_station_v2',
        descendant_transform_audit=dict(source=source_transform_audit,composed=composed_transform_audit,shelf_relocation=shelf_transform_evidence),
        scene_generated=True,native_physics_validated=False,runtime_started=False,
        limitations=['Actual composed USD; static rendered overview is not a physical validation',
                     'Planar placement checked; full accessory support, 3D collision and native settling still need validation',
                     'Rendered static composition is not physical task validation'],
        unpictured_furniture=unpictured,
        overview=dict(bounds=layout['bounds'],floors=layout['floors'],surfaces=layout['surfaces'],blockers=layout['blockers'],
                      furniture=furniture,robot=list(world(stage,'/World/Robot').ExtractTranslation())))
    from openend.room import prepare_registration
    manifest.update(extra)
    if background is not None:
        background_rows=[r for r in rows if r.get('task_role') in ('background_device','background_stock')]
        if any(r.get('task_id') for r in background_rows):raise ValueError('Background acquired scoring ownership')
        if {t['id'] for t in tasks}!={t['id'] for t in data['request']['tasks']}:raise ValueError('Outing objectives changed during composition')
        manifest['outing_background']=dict(policy=outing_policy.VERSION,request=background,
            objects=[dict(id=r['id'],path=r['path'],role=r['task_role']) for r in background_rows],
            registered_background_tasks=0,people_dependent=False)
    if coffee_initialization is not None:manifest['coffee_initialization']=coffee_initialization
    runtime_request=dict(tasks=[dict(instance_id=t['id'],type=t['type'],
                          parameters=t.get('parameters',{'bread_variant':'no_sugar'} if t['type']=='fryer' else {'auto_bind':True} if t['type']=='trash_transfer' else {})) for t in tasks],
                          no_progress_timeout_s=120,selection_order='distance_then_front')
    pending_types={p['type'] for p in pending}
    shelf_requests=[t for t in runtime_request['tasks'] if t['type']=='snack_drink_to_tray']
    if len(shelf_requests)>1:
        # Scene generation is independent of runtime admission. Do not bypass
        # the shared factory's current one-kind guard or claim registration.
        from openend.room import validate_request
        try:validate_request(dict(runtime_request,tasks=shelf_requests))
        except ValueError as exc:
            if str(exc)!='Duplicate instance or same actual asset':raise
            pending.append(dict(type='snack_drink_to_tray',reason='多品类目标已生成；共享入口暂不允许同类多实例，尚未注册判分'))
            pending_types.add('snack_drink_to_tray')
    compatible_request=dict(runtime_request,tasks=[t for t in runtime_request['tasks'] if t['type'] not in pending_types])
    if task_policy is not None and task_policy.get('task_name') == 'microwave_heat_take':
        from short_tasks.tasks.microwave_heat_take.generation import registration as short_registration
        manifest['registration']=short_registration(compatible_request,manifest)
    else:
        manifest['registration']=prepare_registration(compatible_request,manifest) if compatible_request['tasks'] else {}
    manifest['registration']['pending']=pending
    manifest['registration']['all_requested_tasks_bound']=not pending
    for t in tasks:
        t['binding_status']='pending_runtime_adapter' if t['type'] in pending_types else 'identity_bound_not_physics_verified'
    manifest['registration']['runtime_started']=False
    (folder/'runtime_request.json').write_text(json.dumps(runtime_request,ensure_ascii=False,indent=2))
    (folder/'composition.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print(json.dumps(dict(scene=str(out),layout=layout['id'],tasks=len(tasks),objects=len(rows),status='composed_static_preview'),ensure_ascii=False))

if __name__=='__main__':main()
