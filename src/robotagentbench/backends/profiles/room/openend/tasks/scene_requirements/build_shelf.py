"""Original two-tier random stock; explicit task targets separate from background."""
import json,sys
from pathlib import Path
import numpy as np
from pxr import Usd,UsdGeom,Sdf,Gf
from openend.tasks.scene_requirements import shelf_operation_region as operation_region

def compose(stage,data,root,reg,source_hashes,sha):
    requests=data['requirements'].get('shelf_task_requirements',[])+[t for t in data['requirements'].get('meal_task_requirements',[]) if t['type']=='drink_to_meal_tray']
    background=data['requirements'].get('background_requirements',{}).get('shelf_stock',False)
    if not requests and not background:return [],[],{},[]
    if requests:source_hashes[str(operation_region.SOURCE)]=sha(operation_region.SOURCE)
    if background:
        from short_tasks.agent_probe_v1.cap_v1.native_tools.openend_control_adapter_v1.outing_scene_v3.dependencies import activate
        activate(source_hashes,sha)
    sys.path.insert(0,str(Path(__file__).resolve().parent/'cpu_dependencies'))
    from repair_room_placement_v2 import cs,_proxies,_separated,shape_mesh,trimesh,MultiPoint,Rotation
    from shelf_sampler import sample_scene
    from evaluation.tasks.shelf_pick.openended_composite_v1.catalog import PRODUCTS
    from build_room_demo_v1 import exact
    from collections import defaultdict,deque
    byproduct=defaultdict(deque)
    for request in requests:byproduct[request['product']].append(request)
    if not (0 if background else 1)<=len(requests)<=4 or len({r['id'] for r in requests})!=len(requests):raise ValueError('Expected 1–4 distinct shelf task instances, or explicit background-only stock')
    from shelf_inventory_policy import compose as inventory_plan,verify_placed_inventory
    inventory=inventory_plan(dict(seed=data['request']['seed'],required_categories=list(byproduct)))
    from openend.tasks.meal_tray.product_claims import reserve_inventory
    inventory=reserve_inventory(inventory,requests)
    counts={l['support_id']:l['count'] for l in inventory['layers']}
    planned={l['support_id']:[o['product_family'] for o in l['objects']] for l in inventory['layers']}
    pools={sid:list(dict.fromkeys(names)) for sid,names in planned.items()}
    library=root/'api-protocol/all_assets';catalog={};paths={}
    for sid,names in pools.items():
        for i,name in enumerate(names):
            entry=library/name/'asset.usda';source_hashes[str(entry)]=sha(entry)
            asset=Usd.Stage.Open(str(entry));prim=asset.GetDefaultPrim();cache=UsdGeom.XformCache()
            visual=trimesh.util.concatenate([m for _,m in cs.mesh_parts(prim,cache,reference=np.eye(4))])
            proxies=[(b.name,b.mesh.copy()) for b in _proxies(asset,prim,cache)]
            vertices=np.concatenate([visual.vertices]+[m.vertices for _,m in proxies])
            catalog[name]=dict(visual=visual,proxies=proxies,vertices=vertices,
                footprint_xy=np.array(MultiPoint(vertices[:,:2]).convex_hull.exterior.coords[:-1]))
            paths[name]=f'/World/RoomEpisode/Shelf_{sid}_{i}'
    # Work in the shelf frame, so the same sampler handles rotated layouts.
    shelf='/World/RoomEpisode/Shelf';frame=UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(shelf))
    inverse=frame.GetInverse();manifest_path=library/'snack_shelf/manifest.json'
    source_hashes[str(manifest_path)]=sha(manifest_path)
    supports={}
    for tier in counts:
        slab=next(s for s in json.loads(manifest_path.read_text())['support_slab_checks'] if tier.title()+'Overlay' in s['entry_prim'])
        low,high=np.array(slab['min_m']),np.array(slab['max_m'])
        board=shelf+'/Asset'+slab['entry_prim'].removeprefix('/Asset')
        supports[tier]=dict(polygon_xy_m=[[low[0],low[1]],[high[0],low[1]],[high[0],high[1]],[low[0],high[1]]],top_z_m=high[2],prim_path=board)
    low=np.min([np.min(s['polygon_xy_m'],axis=0) for s in supports.values()],axis=0)
    high=np.max([np.max(s['polygon_xy_m'],axis=0) for s in supports.values()],axis=0)
    fixtures=[];cache=UsdGeom.XformCache()
    from pxr import UsdPhysics
    for p in stage.Traverse():
        shape=shape_mesh(p)
        if shape is None:continue
        visible=UsdGeom.Imageable(p).ComputeVisibility()!='invisible' and UsdGeom.Imageable(p).ComputePurpose() not in ('proxy','guide')
        collider=p.HasAPI(UsdPhysics.CollisionAPI) and p.GetAttribute('physics:collisionEnabled').Get() is not False
        if not (visible or collider):continue
        shape.apply_transform(cs.matrix(cache.GetLocalToWorldTransform(p)*inverse))
        body=cs.Body(str(p.GetPath()),shape)
        if body.hi[0]>low[0]-.3 and body.lo[0]<high[0]+.3 and body.hi[1]>low[1]-.3 and body.lo[1]<high[1]+.3:
            fixtures.append(body)
    def pose_item(name,xy,yaw,sid,role):
        c=catalog[name];vertices=c['vertices'];visual=c['visual'];proxies=c['proxies'];path=paths[name]
        rot=Rotation.from_euler('z',yaw,degrees=True)
        xyz=np.r_[xy,supports[sid]['top_z_m']+.0005-rot.apply(vertices)[:,2].min()]
        quat=rot.as_quat()[[3,0,1,2]];tf=cs.pose(xyz,quat);v=cs.points(vertices,tf)
        bodies=[cs.Body(path+'#visible',visual,tf)]+[cs.Body(path+'#proxy'+str(i),m,tf) for i,(_,m) in enumerate(proxies)]
        return dict(product_family=name,prim_path=path,support_id=sid,xyz_m=xyz.tolist(),wxyz=quat.tolist(),yaw_deg=float(yaw),collision_paths=[b.name for b in bodies]),bodies,MultiPoint(v[:,:2]).convex_hull
    task_policy=data.get('task_scene_policy')
    reservation=None
    ordinary_receipt=None
    if task_policy is None:
        placed,metrics=sample_scene(np.random.default_rng(data['request']['seed']),catalog,supports,fixtures,counts,pools,pose_item,_separated,planned_inventory=planned,
            candidate_allowed=operation_region.target_filter(byproduct,supports) if requests else None)
        if requests:
            metrics['operation_region']=dict(policy=operation_region.POLICY,
                max_origin_depth_from_front_m=operation_region.MAX_ORIGIN_DEPTH_M)
    elif task_policy.get('task_name') == 'stock_wooden_tray':
        from short_tasks.tasks.stock_wooden_tray.generation import sample_stock
        placed,metrics,ordinary_receipt=sample_stock(task_policy,requests,data['request']['seed'],
            catalog,supports,fixtures,counts,pools,pose_item,_separated,planned,sample_scene)
        source_hashes.update(ordinary_receipt['source_hashes'])
    else:
        from short_tasks.tasks.stock_wooden_tray_interference.generation import sample_reserved
        placed,metrics,reservation=sample_reserved(task_policy,stage,requests,data['request']['seed'],
            catalog,supports,fixtures,counts,pools,pose_item,_separated,planned,sample_scene)
    # Keep the approved all_assets instance's restored material bindings as well
    # as geometry: raw cross-root asset references can lose authored GripMat.
    approved_report=json.loads((reg/'room_demo_v6/composition.json').read_text())
    approved=Usd.Stage.Open(str(reg/'room_demo_v6/room.usda'));flat=approved.Flatten()
    source_hashes[str(reg/'room_demo_v6/room.usda')]=sha(reg/'room_demo_v6/room.usda')
    rows=[];used={};placements=[];target_paths={}
    for pose,_,_ in placed:
        name=pose['product_family'];sid=pose['support_id'];path=pose['prim_path']
        used[name]=used.get(name,0)+1
        instance_id=f'shelf/{sid}/{name}/{used[name]:02d}'
        placements.append(dict(pose,instance_id=instance_id))
        prototype=next(o for o in approved_report['objects'] if o['id']==name)
        Sdf.CreatePrimInLayer(stage.GetRootLayer(),path)
        if not Sdf.CopySpec(flat,prototype['path'],stage.GetRootLayer(),path):raise ValueError('Shelf prototype copy failed')
        for p in Usd.PrimRange(stage.GetPrimAtPath(path)):
            rel=p.GetRelationship('material:binding:physics')
            for target in rel.GetTargets() if rel else ():
                if not stage.GetPrimAtPath(target):
                    if not approved.GetPrimAtPath(target):raise ValueError('Missing authored physics material '+str(target))
                    Sdf.CreatePrimInLayer(stage.GetRootLayer(),target)
                    Sdf.CopySpec(flat,target,stage.GetRootLayer(),target)
        UsdGeom.Xformable(stage.GetPrimAtPath(path)).MakeMatrixXform().Set(Gf.Matrix4d(*cs.pose(pose['xyz_m'],pose['wxyz']).T.reshape(-1).tolist())*frame)
        selected=bool(byproduct.get(name))
        row=dict(id=name,path=path,entry=str(library/name/'asset.usda'),instance_id=instance_id,source='snack_shelf',task_role='requested_full_product' if selected else 'background_stock',is_garbage=False)
        if selected:
            request=byproduct[name].popleft();target_paths[request['id']]=path;row['task_id']=request['id']
            row['task_group']=request['type']
        rows.append(row)
    inventory_check=verify_placed_inventory(inventory,placements)
    if set(target_paths)!={r['id'] for r in requests}:raise ValueError('Requested shelf target missing')
    from openend.tasks.trash_transfer.scene_semantics import mark_object
    rows=[mark_object(r,empty_packaging=False) for r in rows]
    # Reuse the confirmed narrow-edge tray location, not a new centre/size guess.
    report=json.loads((root/'openend/tasks/all_objects_integration/room_all_dynamic_v2/composition.json').read_text())
    tray=next(o for o in report['objects'] if o['id']=='japanese_tray');tp=tray['path']
    wood_requested=any(r['type']=='snack_drink_to_tray' for r in requests)
    for anchor in (('japanese_tray','tablecloth_dirty') if wood_requested else ()):
        row=next(o for o in report['objects'] if o['id']==anchor);ap=row['path']
        if stage.GetPrimAtPath(ap):continue
        old=Usd.Stage.Open(report['scene']);flat=old.Flatten();source_hashes[report['scene']]=sha(report['scene'])
        Sdf.CreatePrimInLayer(stage.GetRootLayer(),ap);Sdf.CopySpec(flat,ap,stage.GetRootLayer(),ap)
        table='/World/Assets/CoffeeTable';c=UsdGeom.XformCache()
        from descendant_transforms import rigid_delta
        delta,_=rigid_delta(old,stage,table,exclude=('/black_box_1',))
        UsdGeom.Xformable(stage.GetPrimAtPath(ap)).MakeMatrixXform().Set(c.GetLocalToWorldTransform(old.GetPrimAtPath(ap))*delta)
        rows.append(dict(row,role='workspace_support'))
    shelf_rows=[dict(id='snack_shelf',path=shelf),dict(tray),*rows]
    from openend.tasks.scene_requirements.shelf_binding import SceneBinding
    # Dedupe the already present tray, never publish two role aliases.
    unique={r['path']:r for r in shelf_rows}
    tasks=[];binding_receipts=[]
    for request in requests:
        product=request['product']
        if request['type']=='drink_to_meal_tray':continue  # task-owned large-tray binder below
        binding=SceneBinding(stage,{'objects':list(unique.values())},product,'scene_check',request['id'])
        target_path=target_paths[request['id']]
        binding_receipts.append(dict(task_id=request['id'],product=product,target_bodies=list(binding.identity.candidates),stock_bodies=len(binding.stock)))
        tasks.append(dict(id=request['id'],type='snack_drink_to_tray',name='货架取'+PRODUCTS[product].label+'放木托盘',source='snack_shelf',parameters={'product':product},target_path=target_path,object_paths=[target_path],status='scene_bound',runtime_started=False))
    extra=dict(shelf_initialization=dict(products=list(byproduct),seed=data['request']['seed'],inventory_plan=inventory,inventory_check=inventory_check,placements_in_shelf_frame=placements,metrics=metrics,unrequested_stock_spawned=True,background_stock_count=len(placed)-len(tasks),scoring_target_paths=target_paths,binding_receipts=binding_receipts,source_sampler='original_scene_first_visible360_v1_with_required_inventory',native_physics_validated=False))
    if ordinary_receipt is not None:
        extra['shelf_initialization']['placement_policy']=ordinary_receipt
        extra['shelf_initialization']['source_sampler']=task_policy['version']
        extra['task_scene_policy']=dict(task_policy)
    if reservation is not None:
        # Bind to the native identity after the unchanged asset copier runs.
        reservation['target_path']=binding.identity.candidates[0]
        reservation['target_wrapper']=target_paths[requests[0]['id']]
        extra.update(environment_reservations=[reservation],task_scene_policy=task_policy)
    return rows,tasks,extra,[]
