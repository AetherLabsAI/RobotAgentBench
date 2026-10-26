"""Scene-owned tote placement at the authored packing-cabinet spawn frame.

Fixed CPU open-container initialization. No soft-body startup is required.
Task scoring and full robot manipulation are separate from asset validation.
"""
import json
from pathlib import Path
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

LIB=Path('/home/lingjun/robot_agent_arena/api-protocol/all_assets')
STATION='/World/Assets/Task5PackStation'
SPAWN=STATION+'/ToteSpawn'
TOTE=SPAWN+'/Tote'


def apply(stage):
    spawn=stage.GetPrimAtPath(SPAWN)
    if not spawn or not spawn.IsActive():raise ValueError('Missing scene-owned tote spawn')
    old=stage.GetPrimAtPath(TOTE)
    if old:
        if old.GetCustomDataByKey('roomAssetId')!='tote_bag':raise ValueError('Occupied tote spawn')
    else:
        if spawn.GetChildren():raise ValueError('Spawn contains unrecognized scene objects')
        old=stage.DefinePrim(TOTE,'Xform')
        old.SetCustomDataByKey('roomAssetId','tote_bag')
    old.GetReferences().SetReferences([Sdf.Reference(str(LIB/'tote_bag/asset_fixed_cpu_v1.usda'))])
    old.SetCustomDataByKey('physicsStartupStatus','fixed_open_cpu_v1')
    sim=stage.GetPrimAtPath(TOTE+'/Simulation')
    assert sim.HasAPI(UsdPhysics.CollisionAPI) and not old.HasAPI(UsdPhysics.RigidBodyAPI)
    assert sim.GetAttribute('physics:approximation').Get()=='none'
    cache=UsdGeom.XformCache();inv=cache.GetLocalToWorldTransform(spawn).GetInverse()
    bounds={}
    for suffix in ('Simulation','Visual/Geometry'):
        p=stage.GetPrimAtPath(TOTE+'/'+suffix);assert p and p.IsA(UsdGeom.Mesh)
        v=np.asarray(p.GetAttribute('points').Get(),float)
        m=np.asarray(cache.GetLocalToWorldTransform(p)*inv)
        world=(np.c_[v,np.ones(len(v))]@m)[:,:3]
        bounds[suffix]=[world.min(0).tolist(),world.max(0).tolist()]
    # Existing layout contract, not a newly introduced clearance threshold.
    half=spawn.GetCustomDataByKey('task5_tote_spawn:top_half_extents_m')
    half=json.loads(half) if isinstance(half,str) else list(half)
    assert len(half)==2 and all(np.isfinite(half)) and min(half)>0
    body=np.asarray(bounds['Simulation']);visual=np.asarray(bounds['Visual/Geometry'])
    assert np.all(body[0,:2]>=-np.asarray(half)+.01) and np.all(body[1,:2]<=np.asarray(half)-.01),bounds
    assert np.all(visual[0,:2]>=-np.asarray(half)-.04) and np.all(visual[1,:2]<=np.asarray(half)+.04),bounds
    return dict(id='tote_bag',path=TOTE,entry=str(LIB/'tote_bag/asset_fixed_cpu_v1.usda'),
        spawn_path=SPAWN,support_path=STATION+'/Collider',initialization_owner='unified_scene',
        placement='existing_spawn_no_added_randomization',spawn_local_bounds_m=bounds,
        body_edge_clearance_passed=True,handle_overhang_passed=True,
        physics_status='fixed_open_cpu_v1',native_physics_validated=False,
        native_support_probe_verified=True,
        native_validation_scope='isolated_240Hz_real_glasses_drop_60Hz_settled_not_robot_task',
        contents_initialized=False,task_scoring_enabled=False)
