"""Data-range cutlery robot reset; read-only room FK geometry screening."""
import hashlib
import math
from pathlib import Path
import numpy as np
from . import initial_station
from short_tasks.tasks.retrieve_from_drawer import prepare_initial as geometry
from short_tasks.tasks.retrieve_from_drawer.prepare_initial import corners, fk

ROBOT='/World/Robot'


class NoFeasibleInitial(ValueError):
    pass


def check_room_candidate(manifest, *, seed):
    import copy
    from pxr import Usd
    from . import environment_initial as environment
    stage = Usd.Stage.Open(manifest['scene'])
    if not stage:
        raise ValueError('Cannot open actual composed cutlery room')
    # Read-only screening using the exact deterministic environment draw later
    # authored by prepare_manifest. Failed geometry participates in layout fallback.
    stage.SetEditTarget(stage.GetSessionLayer())
    value=copy.deepcopy(manifest)
    receipt=environment.apply(stage,value,seed)
    try:
        receipt['geometry']=environment.validate(stage,value)
    except environment.NoFeasibleEnvironment as exc:
        raise NoFeasibleInitial(str(exc)) from exc
    value['cutlery_environment_initialization']=receipt
    value['supports']['tray']['origin']=receipt['actual_tray_origin_xy']
    return choose(stage, value, seed=seed)


def choose(stage, manifest, *, seed, count=100):
    from pxr import Usd,UsdGeom,UsdPhysics
    from scipy.spatial.transform import Rotation
    from arena_api.robot_profiles import FOLLOWERS, GRIPPER_INTERNAL
    from openend.tasks.scene_requirements.source_geometry import floor_paths
    from openend.tasks.scene_requirements.station_scene_checks import intersects,triangles_hit_box
    from openend.tasks.layout_repair_v1.spawn_v2 import inside
    targets_owned=[o['path'] for o in manifest['objects']
                   if o.get('id') in ('meal_knife','meal_fork') and o.get('meal_task_id')]
    if not targets_owned or len(set(targets_owned))!=len(targets_owned):
        raise ValueError('Unique scene-owned cutlery targets required')
    delta,table,baseline=initial_station.station_context(stage,manifest)
    floors,polygons=floor_paths(stage),manifest.get('overview',{}).get('floors',[])
    if not floors or not polygons:raise ValueError('Measured floors required')
    cache=UsdGeom.XformCache()
    bbox=UsdGeom.BBoxCache(0,['default','render','proxy','guide'],False,True)
    robot_parts,obstacles=[],[]
    for p in Usd.PrimRange(stage.GetPseudoRoot(),Usd.TraverseInstanceProxies()):
        if not p.IsActive() or not p.HasAPI(UsdPhysics.CollisionAPI) or p.GetAttribute('physics:collisionEnabled').Get() is False:continue
        path=str(p.GetPath());r=bbox.ComputeWorldBound(p).ComputeAlignedRange()
        box=np.array([r.GetMin(),r.GetMax()])
        if not np.isfinite(box).all() or np.any(box[0]>box[1]):raise ValueError('Invalid collider bounds: '+path)
        if path.startswith(ROBOT+'/'):
            body=p
            while body and not body.IsPseudoRoot() and not body.HasAPI(UsdPhysics.RigidBodyAPI):body=body.GetParent()
            if not body or body.IsPseudoRoot():raise ValueError('Collider without robot body')
            local=np.c_[corners(box),np.ones(8)]@np.linalg.inv(np.array(cache.GetLocalToWorldTransform(body)))
            robot_parts.append((path,str(body.GetPath()),local));continue
        if any(path==f or path.startswith(f+'/') for f in floors):continue
        tri=None
        if p.IsA(UsdGeom.Mesh):
            vs=p.GetAttribute('points').Get();ids=list(p.GetAttribute('faceVertexIndices').Get() or [])
            faces=[];offset=0
            for n in p.GetAttribute('faceVertexCounts').Get() or []:
                face=ids[offset:offset+n];offset+=n
                faces.extend((face[0],face[i],face[i+1]) for i in range(1,n-1))
            if vs is not None and len(vs) and faces:
                vs=(np.c_[np.array(vs),np.ones(len(vs))]@np.array(cache.GetLocalToWorldTransform(p)))[:,:3]
                tri=vs[np.array(faces)]
        obstacles.append((path,box,tri))
    if not robot_parts:raise ValueError('No robot collider geometry')
    # Source reset names only commanded axes plus paired fingers. Preserve the
    # room's authored internal finger coordinates for this static FK estimate;
    # do not invent zero resets or add these axes to the native ready command.
    preserved={}
    source_names=set(next(initial_station.sample_candidates(seed,delta,table,1))['targets_rad'])
    for name in set(GRIPPER_INTERNAL)-source_names-set(FOLLOWERS):
        p=stage.GetPrimAtPath(ROBOT+'/joints/'+name)
        value=p.GetAttribute('state:angular:physics:position').Get() if p else None
        if value is None or not math.isfinite(value):
            raise ValueError('Missing authored internal finger coordinate: '+name)
        preserved[name]=math.radians(value)
    rejections=[]
    for proposal in initial_station.sample_candidates(seed,delta,table,count):
        targets=dict(proposal['targets_rad'])
        for inner,outer in FOLLOWERS.items():targets[inner]=-targets[outer]
        x,y,yaw=proposal['initial_base_xyyaw']
        base=np.eye(4);base[:3,:3]=Rotation.from_euler('z',yaw).as_matrix();base[:3,3]=[x,y,0.]
        try:links=fk(stage,{**preserved,**targets},base)
        except ValueError as exc:
            if not str(exc).startswith('Reset outside unchanged joint limits:'):raise
            rejections.append(dict(candidate_index=proposal['candidate_index'],reason=str(exc)));continue
        reason=None
        for path,body,local in robot_parts:
            v=(local@links[body].T)[:,:3];box=np.array([v.min(0),v.max(0)])
            perimeter=[(a,b) for a in np.linspace(box[0,0],box[1,0],11) for b in (box[0,1],box[1,1])]
            perimeter += [(a,b) for b in np.linspace(box[0,1],box[1,1],11) for a in (box[0,0],box[1,0])]
            if not all(any(inside(v,p) for p in polygons) for v in perimeter):
                reason='robot_outside_floor:'+path;break
            hit=next((p for p,b,t in obstacles if intersects(box,b) and (t is None or triangles_hit_box(t,box))),None)
            if hit:reason='robot_room_overlap:'+path+':'+hit;break
        if reason:
            rejections.append(dict(candidate_index=proposal['candidate_index'],reason=reason));continue
        refs=[Path(__file__),Path(initial_station.__file__),initial_station.POLICY,
              Path(geometry.__file__),*map(Path,table['reference_files'])]
        return dict(position=[x,y,0.],yaw_rad=yaw,targets_rad=targets,candidate=proposal,
            target=dict(paths=targets_owned,source='cutlery_holder',reference_files=list(map(str,refs))),
            method='source_cutlery_station_actual_room_fk',
            station_delta=delta.tolist(),station_context=table,
            fk_preserved_internal_joint_coordinates_rad=preserved,
            internal_joint_reset_overrides=False,
            candidates_checked=proposal['candidate_index']+1,rejections=rejections,
            target_transforms_unchanged=True,robot_room_geometry_checked=True,
            robot_self_collision_validated=False,native_physics_validated=False,physics_certified=False,
            reference_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in refs})
    raise NoFeasibleInitial('Cutlery station candidates exhausted without widening: '+str(rejections))


def apply_target(stage, receipt):
    # Shared preparer calls this hook; there is deliberately no target reset.
    if receipt.get('target_transforms_unchanged') is not True:
        raise ValueError('Cutlery targets must remain scene-owned')


def refresh_manifest(stage, manifest, receipt):
    manifest['cutlery_station_initialization']=receipt
