"""Frozen robot joints, fresh target XY and fresh backgrounds at every startup.

The 2–5 cm constraint and original product/support bounds define the target
domain. Objects are not selected from a precomputed scene/position pool.
"""
import argparse
from copy import deepcopy
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import numpy as np
from ..common import HERE,CURRENT,owned,digest,write_json,physics,finish
from .catalog import APIS,generator,select
from .constraint import propose,POLICY

@dataclass
class Box:
    name:str
    points:object
    @property
    def lo(self):return self.points.min(0)
    @property
    def hi(self):return self.points.max(0)

def separated(a,b,gap):
    # Conservative full body bounds: reject uncertainty, never certify a mesh.
    return bool(np.any(a.hi+gap<=b.lo) or np.any(b.hi+gap<=a.lo))

def local_geometry(stage,path):
    from pxr import Usd,UsdGeom,UsdPhysics
    from ..poses import world
    inverse=np.linalg.inv(world(stage,path));cache=UsdGeom.XformCache()
    visible=[];collision=[]
    for prim in Usd.PrimRange(stage.GetPrimAtPath(path),Usd.TraverseInstanceProxies()):
        shape=UsdGeom.Boundable(prim)
        if not shape:continue
        colliding=prim.HasAPI(UsdPhysics.CollisionAPI) and prim.GetAttribute('physics:collisionEnabled').Get() is not False
        showing=UsdGeom.Imageable(prim).ComputeVisibility()!='invisible'
        if not colliding and not showing:continue
        extent=UsdGeom.Boundable.ComputeExtentFromPlugins(shape,Usd.TimeCode.Default())
        if extent is None:extent=shape.GetExtentAttr().Get()
        if extent is None:raise ValueError('Missing original geometry: '+str(prim.GetPath()))
        corners=np.array([[extent[0 if bits&(1<<i)==0 else 1][i] for i in range(3)]+[1.] for bits in range(8)])
        matrix=np.asarray(cache.GetLocalToWorldTransform(prim)).T
        points=(inverse@matrix@corners.T).T[:,:3]
        if showing:visible.append(points)
        if colliding:collision.append(points)
    if not visible or not collision:raise ValueError('Original visible+colliding asset required')
    return np.concatenate(visible+collision),np.concatenate(collision)

class Context:
    def __init__(self):
        from pxr import Usd
        from evaluation.tasks.shelf_pick import initial_state,layout_sampling as ls
        self.ls=ls;self.source=initial_state
        self.bank_path=HERE/'shelf_continuous_v2/robot_bank.json';self.bank=json.loads(self.bank_path.read_text())
        if self.bank['schema']!='failure_recovery.robot_only_bank.v2' or self.bank['contains_object_positions'] or self.bank['contains_layouts']:raise ValueError('Robot-only frozen bank required')
        self.ranges_path=CURRENT/'short_task_robustness/tasks/stock_wooden_tray_interference/source_product_ranges.json'
        self.ranges=json.loads(self.ranges_path.read_text())
        self.measure=Usd.Stage.Open(str(initial_state.BASE));self.measure.SetEditTarget(self.measure.GetSessionLayer())
        self.catalog,self.supports=ls.scene_geometry(self.measure)
        self.local={name:local_geometry(self.measure,row['prim_path']) for name,row in self.catalog.items()}
        self.source_files=[self.bank_path,self.ranges_path,Path(ls.__file__),Path(initial_state.__file__)]

def build(seed,output,*,api=None,profile_id=None,context=None):
    from pxr import Usd,UsdGeom
    from scipy.spatial.transform import Rotation
    from shapely.geometry import Polygon
    from evaluation.tasks.shelf_pick.targets import spec
    from evaluation.tasks.shelf_pick.episode_randomization import perturb
    from ..poses import place,trace_pose,world
    from ..procedural_grasp import Kinematics
    from ..frozen_initials_v1.cup import hand_geometry
    from ..shelf_checkpoint import Clearance,target_hand
    from ..frozen_initials_v3.geometry import Gate
    from ..visual_preload import apply
    from openend.tasks.scene_requirements import shelf_operation_region
    from . import packing
    started=time.monotonic();c=context or Context();out=owned(output,fresh=True);out.mkdir(parents=True)
    api,profile=select(c.bank,seed,api,profile_id);identity=spec(api);product=identity.target.rsplit('/',1)[-1]
    # Keep only the pose's source station reference. Do NOT copy its objects.
    reference=dict(schema='arena.shelf_pick.initial_state.v2',api_id=api,
        robot=deepcopy(profile['base_reference']),joint_positions=deepcopy(profile['robot_initialization']['positions']))
    station,station_receipt=perturb(reference,seed)
    # perturb()'s arm draws are deliberately not applied to frozen joints.
    initial=deepcopy(profile['robot_initialization']);hand=profile['active_hand']
    key,value,rng,_=c.ls.seed_chain(api,seed,1)
    raw_counts={tier:int(rng.integers(*c.ls.COUNT_RANGE)) for tier in c.ls.LAYER_ORDER}
    counts={tier:min(raw_counts[tier],len(c.ls.LAYERS[tier][2])) for tier in c.ls.LAYER_ORDER}
    planned={}
    for tier,n in counts.items():
        names=list(c.ls.LAYERS[tier][2]);rng.shuffle(names)
        if tier=='tier3':names=[product]+[name for name in names if name!=product]
        planned[tier]=names[:n]
    active={c.catalog[name]['prim_path'] for names in planned.values() for name in names}
    stage=Usd.Stage.CreateNew(str(out/'initial.usda'));stage.GetRootLayer().subLayerPaths=[os.path.relpath(c.source.BASE,out)]
    baseline=c.measure
    for k in ('metersPerUnit','upAxis','timeCodesPerSecond','framesPerSecond'):
        if baseline.HasAuthoredMetadata(k):stage.SetMetadata(k,baseline.GetMetadata(k))
    stage.SetDefaultPrim(stage.GetPrimAtPath('/World'))
    for root in c.source.STOCK_ROOTS:
        for child in stage.GetPrimAtPath(root).GetAllChildren():child.SetActive(str(child.GetPath()) in active)
    before=physics(stage)
    place(stage,c.source.ROBOT,trace_pose(dict(position=station['robot']['position_m'],quaternion=station['robot']['quaternion_wxyz'])))
    model=Kinematics(stage,hand,initial['positions']);tips,frames=hand_geometry(stage,model,initial['positions'])
    raw_clearance=Clearance(stage,identity.target,[],hand)
    robot_boxes=[Box(path,(frames[body]@corners.T).T[:,:3]) for path,body,corners in raw_clearance.robot]
    supports={tier:dict(polygon_xy_m=[row['rect_xy_m'][0],
        [row['rect_xy_m'][1][0],row['rect_xy_m'][0][1]],row['rect_xy_m'][1],
        [row['rect_xy_m'][0][0],row['rect_xy_m'][1][1]]],top_z_m=row['top_z_m'],prim_path=row['prim_path']) for tier,row in c.supports.items()}
    catalog={name:dict(vertices=points,footprint_xy=c.catalog[name]['footprint_xy_m']) for name,(points,_) in c.local.items()}
    def pose_item(name,xy,yaw,tier,role):
        rot=Rotation.from_euler('z',yaw,degrees=True);vertices,collision=c.local[name]
        z=supports[tier]['top_z_m']+c.ls.SUPPORT_CLEARANCE_M-rot.apply(vertices)[:,2].min()
        xyz=np.r_[xy,z];points=rot.apply(vertices)+xyz
        row=dict(product_family=name,prim_path=c.catalog[name]['prim_path'],support_id=tier,
            xyz_m=xyz.tolist(),wxyz=rot.as_quat()[[3,0,1,2]].tolist(),yaw_deg=float(yaw),collision_paths=[])
        return row,[Box(row['prim_path'],points)],Polygon(points[:,:2]).convex_hull
    bounds=c.ranges['products'][product]
    support=c.supports['tier3'];mid_x=float(np.mean(np.asarray(support['rect_xy_m'])[:,0]));front=float(support['rect_xy_m'][0][1])
    depth=shelf_operation_region.cap_bounds(bounds['inward_m'])
    source_xy=np.array([[mid_x+bounds['lateral_m'][0],front+depth[0]],
                        [mid_x+bounds['lateral_m'][1],front+depth[1]]])
    target=None;draws=[];target_check=None
    # The existing source proposal budget, not an episode/seed retry.
    for attempt in range(c.ls.PROPOSALS):
        yaw=float(rng.uniform(*bounds['yaw_deg']));rot=Rotation.from_euler('z',yaw,degrees=True)
        visual,collider=c.local[product];v=rot.apply(visual);p=rot.apply(collider)
        z=support['top_z_m']+c.ls.SUPPORT_CLEARANCE_M-v[:,2].min()
        xy,request=propose(tips,p,v[:,:2],model.base[:3,:3],z,source_xy,support['rect_xy_m'],rng)
        request.update(proposal=attempt,yaw_deg=yaw)
        if xy is None:draws.append(request);continue
        item=pose_item(product,xy,yaw,'tier3','target');row,bodies,fp=item
        if any(not separated(bodies[0],body,0.) for body in robot_boxes):
            draws.append(dict(request,reason='robot_geometry_overlap'));continue
        measured_side=target_hand(dict(position_m=row['xyz_m']),station['robot'])[0]
        if measured_side!=hand:draws.append(dict(request,reason='wrong_active_hand_side'));continue
        if not Polygon(supports['tier3']['polygon_xy_m']).covers(fp):draws.append(dict(request,reason='support_rejected'));continue
        target=item;target_check=request;draws.append(request);break
    if target is None:
        stage.GetRootLayer().Save();write_json(out/'rejection.json',dict(seed=seed,api=api,profile_id=profile['id'],draws=draws,ik_calls=0,
            scene_fallback=False,seed_retried=False,robot_pose_modified=False))
        raise ValueError('No target in source/support/2–5cm intersection for the selected frozen pose')
    # Existing ordinary-task trapezoid dimensions, rigidly expressed in this shelf.
    ordinary_path=CURRENT/'short_tasks/tasks/stock_wooden_tray/placement_policy.json';ordinary=json.loads(ordinary_path.read_text())
    x0,y0,x1,y1=target[2].bounds
    lane=Polygon([(x0-ordinary['front_side_margin_m'],front),(x1+ordinary['front_side_margin_m'],front),
                  (x1+ordinary['target_side_margin_m'],y1),(x0-ordinary['target_side_margin_m'],y1)])
    fixtures=[]
    from arena_api.scene_obstacles import inventory
    for fixture in inventory(stage):
        if fixture['enabled'] and fixture['category']=='shelf' and fixture['type']=='Cube':
            b=UsdGeom.BBoxCache(0,['default','render','proxy','guide'],False,True).ComputeWorldBound(stage.GetPrimAtPath(fixture['path'])).ComputeAlignedRange()
            lo,hi=np.array(b.GetMin()),np.array(b.GetMax())
            corners=np.array([[lo[i] if bits&(1<<i)==0 else hi[i] for i in range(3)] for bits in range(8)])
            fixtures.append(Box(fixture['path'],corners))
    def allowed(item,placed):
        row,bodies,fp=item
        if row['support_id']=='tier3' and lane.intersection(fp.envelope).area>1e-12:return False
        return not any(not separated(a,b,0.) for a in bodies for b in robot_boxes)
    placed,metrics=packing.sample_scene(rng,catalog,supports,fixtures,counts,
        {tier:list(c.ls.LAYERS[tier][2]) for tier in counts},pose_item,separated,
        fixed_target=target,planned_inventory=planned,candidate_allowed=allowed)
    snapshot=dict(schema='arena.shelf_pick.initial_state.v2',api_id=api,target_path=identity.target,target_tier='tier3',
        target=dict(position_m=target[0]['xyz_m'],quaternion_wxyz=target[0]['wxyz']),robot=station['robot'],
        joint_positions=deepcopy(initial['positions']),source=dict(origin=POLICY,training_data_read=False),items=[])
    for row,_,_ in placed:
        pose=dict(position_m=row['xyz_m'],quaternion_wxyz=row['wxyz'])
        place(stage,row['prim_path'],trace_pose(dict(position=pose['position_m'],quaternion=pose['quaternion_wxyz'])))
        if row['prim_path']!=identity.target:snapshot['items'].append(dict(pose,prim_path=row['prim_path'],support=supports[row['support_id']]['prim_path']))
    if len({row[0]['prim_path'] for row in placed})!=sum(counts.values()):raise ValueError('Inventory count or uniqueness changed')
    record=dict(variant='shelf_drink',source_api=api,seed=seed,target=identity.target,support=identity.support,
        neighbors=[i['prim_path'] for i in snapshot['items']],active_hand=hand,robot_initialization=initial,
        gripper_closure=deepcopy(profile['gripper_closure']),frozen_profile_id=profile['id'],source_robot_base=station['robot'],
        instruction=identity.instruction,maximum_score=4,completion_policy='original_pick_success_only',
        initialization_policy=POLICY,preparation_accepted=True,source_checkpoint=None,source_motion_read=False,
        generation_reads_recorded_trajectory=False,runtime_training_dataset_required=False,online_ik_calls=0,
        joint_coordinates_frozen=True,whole_scene_frozen=False,target_xy_continuously_sampled=True,backgrounds_freshly_sampled=True,
        source_pose_provenance=profile['training_provenance'],seed_retried=False,randomization=dict(
            product=product,target_draw=target_check,target_proposals=draws,background_packing=metrics,
            raw_layer_counts=raw_counts,layer_counts=counts,source_instance_ceilings={tier:len(c.ls.LAYERS[tier][2]) for tier in counts},
            robot_base_perturbation=station_receipt,arm_jitter_applied=False,seed_key=key,
            grasp_lane_world_xy=list(map(list,lane.exterior.coords))),code_sources={})
    check=Gate(stage,record).check(initial,hand);record['procedural_grasp']=check
    if not check['accepted_cpu']:
        stage.GetRootLayer().Save();write_json(out/'rejection.json',dict(seed=seed,api=api,profile_id=profile['id'],check=check))
        raise ValueError('Final original FK/camera/collision gate rejected generated scene: '+str(check['reason']))
    for path in [Path(__file__),Path(packing.__file__),Path(sys.modules[__package__+'.constraint'].__file__),
                 Path(sys.modules[__package__+'.catalog'].__file__),ordinary_path,*c.source_files,
                 HERE/'shelf_continuous_v2/adapter.py',HERE/'shelf_continuous_v2/runtime.py',
                 HERE/'shelf_continuous_v2/registration.py',HERE/'shelf_continuous_v2/owner.py']:
        record['code_sources'][str(path)]=digest(path)
    write_json(out/'source_snapshot.json',snapshot)
    value=apply(stage,out,finish(stage,out,record,before));value['generation_wall_seconds']=time.monotonic()-started
    write_json(out/'initial.json',value)
    return value

def dependency_guard(event,args):
    if event=='open' and args and isinstance(args[0],(str,bytes)):
        path=os.fsdecode(args[0])
        if '/training_dataset/' in path or path.endswith(('.npz','.parquet')):raise RuntimeError('No runtime trajectory/training reads')

def no_optimizer(frame,event,arg):
    if event=='call' and ((frame.f_code.co_name=='solve' and frame.f_code.co_filename.endswith('procedural_grasp.py')) or
                         (frame.f_code.co_name=='least_squares' and 'scipy' in frame.f_code.co_filename)):
        raise RuntimeError('No online IK')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--output',required=True)
    p.add_argument('--api',choices=APIS);p.add_argument('--diagnostic-profile');a=p.parse_args()
    sys.addaudithook(dependency_guard);sys.setprofile(no_optimizer)
    try:v=build(a.seed,a.output,api=a.api,profile_id=a.diagnostic_profile)
    finally:sys.setprofile(None)
    write_json(owned(a.output)/'dependency_audit.json',dict(optimizer_calls=0,trajectory_reads=0))
    print(json.dumps(dict(api=v['source_api'],profile=v['frozen_profile_id'],seconds=v['generation_wall_seconds'])))

