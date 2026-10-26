"""Continuous observed-data robot reset in the actual tray-support frame."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
POLICY=HERE/'dataset_robot_ranges.json'
SOURCE=HERE/'robot_station_source.json'


def sample_candidates(seed, delta, context, count=100):
    if type(seed) is not int or not 1 <= seed <= 4294967295:
        raise ValueError('Positive uint32 seed required')
    if type(count) is not int or count < 1:raise ValueError('Positive count required')
    delta=np.asarray(delta,dtype=float)
    if (delta.shape!=(4,4) or not np.isfinite(delta).all()
        or not np.allclose(delta[:,3],[0,0,0,1],atol=1e-8)
        or not np.allclose(delta[:3,:3]@delta[:3,:3].T,np.eye(3),atol=1e-8)
        or not np.isclose(np.linalg.det(delta[:3,:3]),1,atol=1e-8)
        or not np.allclose(delta[2,:3],[0,0,1],atol=1e-8)
        or abs(delta[3,2])>1e-8):
        raise ValueError('Upright rigid floor-preserving station frame required')
    policy=json.loads(POLICY.read_text())['ranges']
    canonical=json.loads(SOURCE.read_text())['canonical_tray_xyz']
    rng=np.random.default_rng(int.from_bytes(hashlib.sha256(
        f'cutlery-station-observed-ranges-v1:{seed}'.encode()).digest(),'big'))
    for i in range(count):
        x=float(rng.uniform(*policy['canonical_base_x']))
        y=float(rng.uniform(*policy['canonical_base_y']))
        angle=float(rng.uniform(*policy['canonical_yaw_deg']))
        base=np.array([x-canonical[0],y-canonical[1],0.,1.])@delta
        inward=np.array([1.,0.,0.])@delta[:3,:3]
        yaw=math.atan2(inward[1],inward[0])+math.radians(angle)
        yaw=math.atan2(math.sin(yaw),math.cos(yaw))
        targets={name:float(rng.uniform(*interval)) for name,interval in policy.items()
                 if name.startswith('idx')}
        yield dict(candidate_index=i,sampling='continuous_cutlery_observed_ranges_v1',
            initial_base_xyyaw=[float(base[0]),float(base[1]),yaw],
            canonical_base_xy=[x,y],yaw_deviation_deg=angle,
            tray_forward_distance_m=canonical[0]-x,
            tray_lateral_offset_m=y-canonical[1],targets_rad=targets,
            joints_sampling='continuous independent observed source joint ranges; requires actual FK gate')


def station_context(stage,manifest):
    from pxr import UsdGeom
    support=manifest['supports']['tray']
    front=np.asarray(support['front'],dtype=float)
    if front.shape!=(2,) or not np.isfinite(front).all() or np.linalg.norm(front)<1e-8:
        raise ValueError('Actual tray support front required')
    front/=np.linalg.norm(front);inward=-front;left=np.array([-inward[1],inward[0]])
    path='/World/RoomEpisode/MealTray'
    prim=stage.GetPrimAtPath(path)
    holder=stage.GetPrimAtPath('/World/RoomEpisode/CutleryHolder')
    support_prim=stage.GetPrimAtPath(support['path'])
    if not all(p and p.IsActive() for p in (prim,holder,support_prim)):
        raise ValueError('Actual tray, holder and support must exist')
    cache=UsdGeom.XformCache()
    tray=np.asarray(cache.GetLocalToWorldTransform(prim))
    hm=np.asarray(cache.GetLocalToWorldTransform(holder))
    measured_origin=tray[3,:2]
    if not np.allclose(measured_origin,support['origin'],atol=1e-5,rtol=0):
        raise ValueError('Declared station origin differs from measured tray')
    # Tray jitter must not drag the independently sampled robot station with it.
    origin=np.asarray(manifest.get('cutlery_environment_initialization',{}).get(
        'nominal_station_origin_xy',measured_origin),dtype=float)
    holder_depth=float(np.dot(hm[3,:2]-origin,inward))
    if holder_depth<=0:raise ValueError('Holder is not behind tray in station frame')
    delta=np.eye(4);delta[0,:2]=inward;delta[1,:2]=left;delta[3,:2]=origin
    refs=[str(POLICY),str(SOURCE),str(Path(__file__))]
    return delta,dict(tray_path=path,support_path=support['path'],
        origin_xy=origin.tolist(),outward_front=front.tolist(),
        actual_surface_height_m=support['actual_top_m'],holder_depth_m=holder_depth,
        reference_files=refs),SOURCE
