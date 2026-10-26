"""Read-only actual-pose room checks, reusing existing joint-frame FK."""
import hashlib
from pathlib import Path
import numpy as np
from . import initial_station


def choose(stage, manifest, seed, context, delta):
    from pxr import Usd, UsdGeom, UsdPhysics
    from scipy.spatial.transform import Rotation
    from arena_api.robot_profiles import FOLLOWERS, GRIPPER_INTERNAL
    from short_tasks.tasks.retrieve_from_drawer import prepare_initial as geometry
    from openend.tasks.scene_requirements.source_geometry import floor_paths
    from openend.tasks.scene_requirements.station_scene_checks import intersects, triangles_hit_box
    from openend.tasks.layout_repair_v1.spawn_v2 import inside
    from short_tasks.agent_probe_v1.target_spawn import NoFeasibleTaskFront
    robot='/World/Robot'
    floors=floor_paths(stage)
    polygons=manifest.get('overview',{}).get('floors',[])
    if not floors or not polygons:raise ValueError('Measured floors required')
    cache=UsdGeom.XformCache()
    bbox=UsdGeom.BBoxCache(0,['default','render','proxy','guide'],False,True)
    parts,obstacles=[],[]
    for p in Usd.PrimRange(stage.GetPseudoRoot(),Usd.TraverseInstanceProxies()):
        if not p.IsActive() or not p.HasAPI(UsdPhysics.CollisionAPI) or p.GetAttribute('physics:collisionEnabled').Get() is False:continue
        path=str(p.GetPath());r=bbox.ComputeWorldBound(p).ComputeAlignedRange()
        box=np.array([r.GetMin(),r.GetMax()])
        if not np.isfinite(box).all() or np.any(box[0]>box[1]):raise ValueError('Invalid collider: '+path)
        if path.startswith(robot+'/'):
            body=p
            while body and not body.IsPseudoRoot() and not body.HasAPI(UsdPhysics.RigidBodyAPI):body=body.GetParent()
            if not body or body.IsPseudoRoot():raise ValueError('Robot collider lacks rigid body')
            local=np.c_[geometry.corners(box),np.ones(8)]@np.linalg.inv(np.array(cache.GetLocalToWorldTransform(body)))
            parts.append((path,str(body.GetPath()),local));continue
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
    if not parts:raise ValueError('No robot colliders')
    rejections=[]
    # The 20-D recording does not contain passive gripper coordinates. For the
    # geometry check only, recover those from the authored joint frames. Native
    # reset still receives only the recorded 20 coordinates plus its followers.
    passive={}
    recorded=set(initial_station.posture_policy()['ranges'])|set(FOLLOWERS)
    for p in Usd.PrimRange(stage.GetPrimAtPath(robot)):
        if not p.IsA(UsdPhysics.RevoluteJoint) or p.GetName() in recorded:continue
        name=p.GetName()
        if name not in GRIPPER_INTERNAL:raise ValueError('Unknown unrecorded coordinate: '+name)
        frames=[]
        for i in (0,1):
            body=p.GetRelationship('physics:body'+str(i)).GetTargets()
            if len(body)!=1:raise ValueError('Expected gripper joint bodies')
            q=p.GetAttribute('physics:localRot'+str(i)).Get()
            local=np.eye(4);local[:3,:3]=Rotation.from_quat([*q.GetImaginary(),q.GetReal()]).as_matrix()
            local[:3,3]=p.GetAttribute('physics:localPos'+str(i)).Get()
            frames.append(np.array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(body[0]))).T@local)
        motion=np.linalg.inv(frames[0])@frames[1]
        passive[name]=float(Rotation.from_matrix(motion[:3,:3]).as_rotvec()['XYZ'.index(p.GetAttribute('physics:axis').Get())])
    for proposal in initial_station.sample_reset_candidates(seed,delta,context,count=100):
        targets=dict(passive,**proposal['targets_rad'])
        for inner,outer in FOLLOWERS.items():targets[inner]=-targets[outer]
        x,y,yaw=proposal['initial_base_xyyaw']
        base=np.eye(4);base[:3,:3]=Rotation.from_euler('z',yaw).as_matrix();base[:3,3]=[x,y,0.]
        try:links=geometry.fk(stage,targets,base)
        except ValueError as exc:
            if not str(exc).startswith('Reset outside unchanged joint limits:'):raise
            rejections.append(dict(candidate_index=proposal['candidate_index'],reason=str(exc)));continue
        reason=None
        for path,body,local in parts:
            vs=(local@links[body].T)[:,:3];box=np.array([vs.min(0),vs.max(0)])
            perimeter=[(a,b) for a in np.linspace(box[0,0],box[1,0],11) for b in (box[0,1],box[1,1])]
            perimeter += [(a,b) for b in np.linspace(box[0,1],box[1,1],11) for a in (box[0,0],box[1,0])]
            if not all(any(inside(v,p) for p in polygons) for v in perimeter):reason='robot_outside_floor:'+path;break
            hit=next((p for p,b,t in obstacles if intersects(box,b) and (t is None or triangles_hit_box(t,box))),None)
            if hit:reason='robot_room_overlap:'+path+':'+hit;break
        if reason:
            rejections.append(dict(candidate_index=proposal['candidate_index'],reason=reason));continue
        refs=[Path(__file__),Path(initial_station.__file__),initial_station.POLICY,initial_station.POSTURE_POLICY,Path(geometry.__file__),*map(Path,context['reference_files'])]
        return dict(position=[x,y,0.],yaw_rad=yaw,targets_rad=dict(proposal['targets_rad']),candidate=proposal,
            target=dict(paths=[context['target_path']],source='sofa',reference_files=list(map(str,refs))),
            method='sofa_remote_dataset_posture_actual_room_fk',posture_source=proposal['posture_source'],
            candidates_checked=proposal['candidate_index']+1,rejections=rejections,
            target_transforms_unchanged=True,robot_room_geometry_checked=True,
            robot_self_collision_validated=False,native_physics_validated=False,physics_certified=False,
            reference_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in refs})
    raise NoFeasibleTaskFront('Remote data posture candidates exhausted without widening: '+str(rejections))
