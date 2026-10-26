"""Task-local composition using approved assets and existing placement samplers."""
import json,math
from pathlib import Path
import numpy as np
from pxr import Usd,UsdGeom,UsdPhysics,Sdf,Gf
from build_room_demo_v1 import exact,T,R,matrix
from randomize_other_living_v1 import world,setworld,center_bottom,seat_height_function
from openend.tasks.trash_transfer.scene_semantics import mark_object,bin_roles
from openend.tasks.trash_transfer.paper_variant_usd import apply_variant,plan_variant
from openend.tasks.trash_transfer.paper_variants import annotate_object
from task_choices import BY_TYPE

def compose(stage,data,root,reg,source_hashes,sha):
    plan=data['living_plan'];inventory=data['requirements']['portable_inventory']
    if not inventory['objects']:return [],[],{},[]
    approved_dir=root/'openend/tasks/all_objects_integration/room_all_dynamic_v2'
    report=json.loads((approved_dir/'composition.json').read_text())
    old=Usd.Stage.Open(report['scene']);flat=old.Flatten();source_hashes[report['scene']]=sha(report['scene'])
    baseline=Usd.Stage.Open(str(Path(data['layout']['scene']).parent/'ID_01_LIVING_STRAIGHT.usdc'))
    byid={o['instance_id']:o for o in plan['objects']};rows=[];extra={};pending=[]
    def delta(source,name):
        path='/World/Assets/'+name
        # Layouts may relocate mesh children as well as the furniture root.
        # Fit the unchanged visible furniture, excluding its movable drawer.
        def vertices(s):
            values={};cache=UsdGeom.XformCache()
            for p in Usd.PrimRange(s.GetPrimAtPath(path)):
                if name=='CoffeeTable' and '/black_box_1' in str(p.GetPath()):continue
                if not p.IsA(UsdGeom.Mesh):continue
                v=np.asarray(UsdGeom.Mesh(p).GetPointsAttr().Get())
                if len(v):values[str(p.GetPath())[len(path):]]=(np.c_[v,np.ones(len(v))]@np.asarray(cache.GetLocalToWorldTransform(p)))[:,:3]
            return values
        aa,bb=vertices(source),vertices(stage)
        if not aa or set(aa)!=set(bb) or any(aa[k].shape!=bb[k].shape for k in aa):
            raise ValueError('Furniture geometry differs from approved sampler: '+name)
        a=np.concatenate([aa[k] for k in sorted(aa)]);b=np.concatenate([bb[k] for k in sorted(bb)])
        ac,bc=a.mean(0),b.mean(0);u,_,vt=np.linalg.svd((a-ac).T@(b-bc));rot=u@vt
        translation=bc-ac@rot
        if np.linalg.det(rot)<0 or not np.allclose(rot[2],[0,0,1],atol=1e-7) or np.max(np.abs(a@rot+translation-b))>1e-5:
            raise ValueError('Furniture geometry differs from approved sampler: '+name)
        m=np.eye(4);m[:3,:3]=rot;m[3,:3]=translation
        return matrix(m)
    # Validate only furniture whose canonical placement sampler is used.
    # An unrelated small-table variant must not reject a sofa-only task.
    ds=delta(baseline,'Sofa') if plan['sofa'] else None
    dt=delta(baseline,'CoffeeTable') if plan['coffee_table'] else None
    dm=delta(baseline,'SmallCoffeeTable') if plan['small_table'] else None
    def prototype(asset):
        matches=[r for r in report['objects'] if r['id']==asset]
        if len(matches)!=1:raise ValueError('Missing or ambiguous portable prototype '+asset)
        return matches[0]
    def copy_asset(obj,canonical=False):
        if obj['asset_id']=='umbrella':
            from openend.tasks.umbrella_transfer.scene import copy_target
            path='/World/RoomEpisode/Selected_'+obj['instance_id'].replace('-','_')
            row=copy_target(stage,path,source_hashes,sha)
            value=dict(row,path=path,instance_id=obj['instance_id'],task_id=obj['required_by'][0],source=obj['source'])
            rows.append(value);return path
        row=prototype(obj['asset_id']);path='/World/RoomEpisode/Selected_'+obj['instance_id'].replace('-','_')
        if canonical:
            UsdGeom.Xform.Define(stage,path);child=stage.DefinePrim(path+'/Asset')
            child.GetReferences().AddReference(str(root/'api-protocol/all_assets'/obj['asset_id']/'asset.usda'))
            for q in Usd.PrimRange(child):
                if q.IsA(UsdGeom.Xformable) and UsdGeom.Xformable(q).GetResetXformStack():UsdGeom.Xformable(q).SetResetXformStack(False)
        else:
            Sdf.CreatePrimInLayer(stage.GetRootLayer(),path)
            if not Sdf.CopySpec(flat,row['path'],stage.GetRootLayer(),path):raise ValueError('Asset copy failed')
        value=dict(row,path=path,instance_id=obj['instance_id'],task_id=obj['required_by'][0],source=obj['source'])
        rows.append(value);return path
    if plan['sofa']:
        height=seat_height_function(stage)
        overlay='/World/Assets/SeatOverlay'
        if old.GetPrimAtPath(overlay):
            Sdf.CreatePrimInLayer(stage.GetRootLayer(),overlay);Sdf.CopySpec(flat,overlay,stage.GetRootLayer(),overlay)
            setworld(stage,overlay,world(old,overlay)*delta(old,'Sofa'))
        for p in plan['sofa']:
            obj=byid[p['id']];path=copy_asset(obj)
            center_bottom(stage,path,[-.3503563938+p['y'],-.8710-p['x'],.453],R(p['yaw_deg']-90))
            setworld(stage,path,world(stage,path)*ds)
            b=exact(stage,path);setworld(stage,path,world(stage,path)*T([0,0,height(b)-b[0,2]]))
    for p in plan['small_table']:
        path=copy_asset(p,canonical=True);setworld(stage,path,matrix(np.array(p['placement']['world_delta_column_matrix']).T)*dm)
        # Compare actual corners against the sampled reference footprint transformed with the furniture.
        from openend.tasks.cloth_clear.scene import bounds
        actual=bounds(stage,path,dm)
        expected=np.array(p['placement']['after_geometry_bounds_m'])
        if not np.allclose(actual,expected,atol=1e-5):
            raise ValueError('Small-table footprint transform mismatch')
    # Fixed cloth/tray placement remains furniture-relative; no unused movable props.
    if plan['coffee_table'] or any(t['type']=='cloth_clear' for t in inventory['tasks']):
        for asset in ('tablecloth_dirty','japanese_tray'):
            row=prototype(asset);path=row['path'];Sdf.CopySpec(flat,path,stage.GetRootLayer(),path)
            setworld(stage,path,world(old,path)*delta(old,'CoffeeTable'))
            rows.append(dict(row,role='workspace_support'))
    lib=root/'api-protocol/all_assets'
    paper_flat=None;paper_path=None
    if any(p['assetId']=='paper_ball' for p in plan['coffee_table']):
        approved_paper=root/'openend/tasks/trash_transfer/dynamic_room.IF8mai'
        paper_report=json.loads((approved_paper/'dynamic_report.json').read_text())
        if not paper_report['passed']:raise ValueError('Approved dynamic paper evidence missing')
        paper_path=next(r['body'] for r in paper_report['results'] if 'paper_ball' in r['body'])
        paper_scene=approved_paper/'room.usda'
        paper_stage=Usd.Stage.OpenMasked(str(paper_scene),Usd.StagePopulationMask([paper_path]))
        paper_flat=paper_stage.Flatten();source_hashes[str(paper_scene)]=sha(paper_scene)
    for p in plan['coffee_table']:
        path='/World/RoomEpisode/Selected_'+p['instance_id'].replace('-','_')
        UsdGeom.Xform.Define(stage,path);child=stage.DefinePrim(path+'/Asset')
        entry=lib/p['assetId']/'asset.usda'
        if p['assetId']=='paper_ball':
            # Copy the already approved dynamic prototype, not the old kinematic library default.
            if not Sdf.CopySpec(paper_flat,paper_path,stage.GetRootLayer(),path+'/Asset'):raise ValueError('Dynamic paper copy failed')
            child=stage.GetPrimAtPath(path+'/Asset')
        else:child.GetReferences().AddReference(str(entry))
        for q in Usd.PrimRange(child):
            if q.IsA(UsdGeom.Xformable) and UsdGeom.Xformable(q).GetResetXformStack():UsdGeom.Xformable(q).SetResetXformStack(False)
        variant=plan_variant(p)
        if variant is not None:apply_variant(stage,path+'/Asset',variant)
        center_bottom(stage,path,p['world_center']+[.451],R(math.degrees(p['rollRad']),(1,0,0))*R(math.degrees(p['yaw'])-90))
        setworld(stage,path,world(stage,path)*dt)
        row=dict(id=p['assetId'],path=path,entry=str(entry),instance_id=p['instance_id'],task_id=p['required_by'][0],source=p['source'])
        if p['trash_category']:row=mark_object(row,empty_packaging=p['assetId']=='zbar_t')
        if variant is not None:row=annotate_object(row,variant)
        rows.append(row)
    cloth=[o for o in plan['objects'] if o['source']=='coffee_table.cloth']
    if cloth:
        from openend.tasks.cloth_clear.scene import bounds,DYNAMIC
        from openend.tasks.cloth_clear.placement_pose import initial_pose,check_occupancy,initial_pose_policy
        from cloth_placement import sample_centres
        variants={r['id']:r for r in json.loads(DYNAMIC.read_text())['assets']}
        # Existing dynamic all_assets book, without scaling or physics overrides.
        book_entry=lib/'book_red'/'asset.usda'
        book_stage=Usd.Stage.Open(str(book_entry))
        book_root=book_stage.GetPrimAtPath('/Asset')
        book_bodies=[p for p in Usd.PrimRange(book_root) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        if (len(book_bodies)!=1 or book_bodies[0]!=book_root
                or book_root.GetAttribute('physics:rigidBodyEnabled').Get() is False
                or book_root.GetAttribute('physics:kinematicEnabled').Get() is True):
            raise ValueError('Existing book must be one dynamic /Asset body')
        variants['book_red']=dict(id='book_red',entry=str(book_entry),dynamic=True,
            collision_material_geometry_preserved=True,
            source_layers={l.realPath:sha(l.realPath) for l in book_stage.GetUsedLayers() if l.realPath})
        cp=prototype('tablecloth_dirty')['path'];tp=prototype('japanese_tray')['path'];frame=world(stage,cp);cb=bounds(stage,cp,frame)
        sources=[]
        for o in cloth:
            item=variants[o['asset_id']]
            if not item['dynamic'] or not item['collision_material_geometry_preserved']:raise ValueError('Unverified cloth asset')
            for path,digest in item['source_layers'].items():
                if sha(path)!=digest:raise ValueError('Cloth source changed')
            src=Usd.Stage.Open(item['entry']);rotation,b=initial_pose(src,o['asset_id'])
            sources.append((o,item,rotation,b))
        centres=sample_centres([b[1]-b[0] for _,_,_,b in sources],cb,np.random.default_rng(data['request']['seed']))
        paths=[];placed_boxes=[]
        for (obj,item,rotation,b),centre in zip(sources,centres):
            path='/World/RoomEpisode/Selected_'+obj['instance_id'].replace('-','_');paths.append(path)
            node=UsdGeom.Xform.Define(stage,path);child=stage.DefinePrim(path+'/Asset');child.GetReferences().AddReference(item['entry'])
            setworld(stage,path,rotation*T(centre-b.mean(0))*frame);child.SetCustomDataByKey('roomAssetId',obj['asset_id'])
            actual=bounds(stage,path,frame)
            if not np.allclose(actual,b+(centre-b.mean(0)),atol=1e-7):raise ValueError('Cloth posed footprint mismatch')
            placed_boxes.append(actual)
            rows.append(dict(id=obj['asset_id'],path=path,entry=str(lib/obj['asset_id']/'asset.usda'),dynamic_entry=item['entry'],
                initial_pose_policy=initial_pose_policy(obj['asset_id']),
                initial_bounds_cloth_local_m=actual.tolist(),
                rigid_body_path=path+'/Asset',task_role='cloth_clear_target',source=obj['source'],instance_id=obj['instance_id'],task_id=obj['required_by'][0],native_verified=False))
        check_occupancy(placed_boxes,cb)
        extra['cloth_clear']=dict(version=1,seed=data['request']['seed'],target_paths=paths,asset_pool_ids=sorted(variants),cloth_path=cp,
            table_path='/World/Assets/CoffeeTable',exclusion_paths=[tp],sampler='task_local_book_cloth_extension_4_same_spacing',native_verified=False)
        # Runtime inventory now binds up to four explicit cloth targets.
        # Let the shared registration validator verify the actual target roles.
    drawer_objects=[o for o in plan['objects'] if o['source']=='coffee_table.drawer']
    if len(drawer_objects)>1:raise ValueError('Drawer initial capacity is one object')
    if drawer_objects or any(t['type']=='notebook_transfer' for t in inventory['tasks']):
        # Reuse the previously approved slide, with frame calibration only.
        from evaluation.tasks.tidy_sofa.initial_state import BASE,SLIDE_LAYER
        from evaluation.tasks.tidy_sofa.specs import COFFEE_TABLE,DRAWER,DRAWER_JOINT,DRAWER_INITIAL_LIP_M
        reference=Usd.Stage.OpenMasked(str(BASE),Usd.StagePopulationMask([COFFEE_TABLE]))
        m=world(reference,COFFEE_TABLE).GetInverse()*world(stage,COFFEE_TABLE)
        stage.GetRootLayer().subLayerPaths=[str(SLIDE_LAYER)]+list(stage.GetRootLayer().subLayerPaths)
        joint=stage.GetPrimAtPath(DRAWER_JOINT);p=joint.GetAttribute('physics:localPos0').Get();q=joint.GetAttribute('physics:localRot0').Get()
        joint.GetAttribute('physics:localPos0').Set(Gf.Vec3f(*m.Transform(Gf.Vec3d(*p))))
        joint.GetAttribute('physics:localRot0').Set(Gf.Quatf(m.RemoveScaleShear().ExtractRotationQuat()*Gf.Quatd(q.GetReal(),Gf.Vec3d(*q.GetImaginary()))))
        initial=world(stage,DRAWER).ExtractTranslation();expected=m.Transform(world(reference,DRAWER).ExtractTranslation())
        if (initial-expected).GetLength()>=.003:raise ValueError('Drawer initial transform incompatible')
        contract=dict(source_to_layout_matrix=[list(r) for r in m],initial_origin_m=list(initial),initial_lip_m=DRAWER_INITIAL_LIP_M,
                      slide_source=str(SLIDE_LAYER),slide_sha256=sha(SLIDE_LAYER))
        stage.GetPrimAtPath(DRAWER).SetCustomDataByKey('openend:drawerCalibration',json.dumps(contract));extra['openend_drawer']=contract
    for obj in drawer_objects:
        # Reuse a packaged producer frame-0 target-to-drawer transform only.
        # Never call the old station builder: it also relocates the robot/furniture.
        from evaluation.tasks.find_object.specs import episodes,episode_manifest,EPISODES
        from evaluation.tasks.find_object.initial_state import pose_matrix,asset_root_transform
        from openend.tasks.cloth_clear.scene import bounds
        key={'glasses_case':'4.4','wallet':'4.10'}[obj['asset_id']]
        path=copy_asset(obj,canonical=True)
        asset_transform=asset_root_transform(obj['asset_id'])
        scale=Gf.Matrix4d().SetScale(Gf.Vec3d(*[asset_transform.GetRow3(i).GetLength() for i in range(3)]))
        # Recorded poses are PhysX body frames, NOT USD scaled mesh frames.
        # Multiplying by the full drawer world matrix would scale the target a
        # second time (this release's drawer carries about 1.3x authored scale).
        frame=world(stage,DRAWER).RemoveScaleShear();container=bounds(stage,DRAWER,frame)
        # Only choose among the existing recorded frame-0 poses. Do not clamp,
        # rescale, or invent a pose when a recorded sample does not fit.
        import random
        labels=list(episodes(key));random.Random(data['request']['seed']).shuffle(labels)
        rejected=[]
        for label in labels:
            source=episode_manifest(key,label);target=source['target'];drawer=source['drawers']['black_box_1']
            if target['asset']!=obj['asset_id']:raise ValueError('Drawer source identity mismatch')
            relative=pose_matrix(target['position_m'],target['quaternion_wxyz'])*pose_matrix(drawer['position_m'],drawer['quaternion_wxyz']).GetInverse()
            setworld(stage,path+'/Asset',scale*relative*frame)
            actual=bounds(stage,path,frame)
            # Source wallet poses can lean above the rim; the old initialization
            # has no whole-object-below-rim rule. Do not invent that restriction.
            # Keep the recorded pose, check XY containment and no floor underrun;
            # wall contact/stability remain for real native initial acceptance.
            if np.all(actual[0]>=container[0]) and np.all(actual[1,:2]<=container[1,:2]):break
            rejected.append(dict(episode=label,bounds_m=actual.tolist()))
        else:raise ValueError('No recorded drawer pose fits current geometry: '+json.dumps(dict(asset=obj['asset_id'],container=container.tolist(),rejected=rejected)))
        manifest_path=EPISODES/key/label/'episode.json';source_hashes[str(manifest_path)]=sha(manifest_path)
        rows[-1].update(source_episode=label,source_subtask=key,source_api_id='9-1' if key=='4.4' else '9-2',
            source_pose_policy='seeded_existing_frame0_candidates_fit_checked_no_new_pose',rejected_source_poses=rejected,
            drawer_path=DRAWER,drawer_body_frame_bounds_m=actual.tolist(),
            source_rim_protrusion_m=max(0.,float(actual[1,2]-container[1,2])),
            source_wall_collision_validated=False,native_physics_validated=False)
    if any(t['type']=='trash_transfer' for t in inventory['tasks']):extra.update(trash_semantics_version=1,task_destinations=bin_roles(stage))
    tasks=[]
    for t in inventory['tasks']:
        members=[r for r in rows if r.get('task_id')==t['id']]
        if len(members)!=t['target_count']:raise ValueError('Requested target missing '+t['id'])
        if t['type']=='book_transfer' and len(members)>1:pending.append(dict(type=t['type'],reason='多本书已生成；旧运行时仍需逐实例适配'))
        tasks.append(dict(id=t['id'],type=t['type'],name=BY_TYPE[t['type']]['name'],description=BY_TYPE[t['type']]['description'],
                          target_path=members[0]['path'],object_paths=[r['path'] for r in members],status='scene_bound',score=None,runtime_started=False,
                          source=members[0]['source'],target_count=len(members)))
    for row in rows:
        if row.get('entry'):source_hashes[row['entry']]=sha(row['entry'])
    return rows,tasks,extra,pending
