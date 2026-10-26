"""Natural native settling diagnostic: no pose/velocity/physics overrides."""
import json
import sys
import traceback
from pathlib import Path

ROOT=Path('/home/lingjun/robot_agent_arena/layout_task_activation_v1/task_scene_registration_v1')
sys.path[:0]=[str(ROOT),str(ROOT.parents[1]/'api-protocol')]
import numpy as np
import omni.kit.app
import omni.physx
import omni.physics.tensors
from pxr import Usd, UsdGeom, UsdUtils
from initialize_room_fridge_boxes_v1 import apply, ASSETS, DOOR
from multi_task_room_v2 import MultiTaskRoom
from coffee_hotwake_v1 import NativeSceneClock
from room_fridge_helper_v1 import FRIDGE

out=Path('/pilot'); report=dict(status='started',native_settling_only=True,full_task_success=False,
    runtime_pose_writes=0,runtime_velocity_writes=0,physics_parameter_overrides=0)
owner=sim=None;time=0.
try:
    manifest=json.loads((ROOT/'coffee_initialized_rooms_v1/layout03/composition.json').read_text())
    paths=[FRIDGE,'/World/Background/Shell/Floor','/World/PhysicsScene','/World/PhysicsMaterials']
    paths.extend(v[2] for v in ASSETS.values())
    stage=Usd.Stage.OpenMasked(manifest['scene'],Usd.StagePopulationMask(paths))
    stage.SetEditTarget(stage.GetSessionLayer())
    rows=apply(stage,seed=int(manifest['seed']));report['placements']=rows
    request=json.loads((ROOT/'multi_task_episode_v1.json').read_text());request['tasks']=[]
    sensor=json.loads((ROOT.parent/'hotwake_probe_inputs.json').read_text())['sensor']
    owner=MultiTaskRoom(stage,manifest,out/'room','fridge-box-natural-settle',request=request,sensor=sensor)
    scene=stage.GetPrimAtPath('/World/PhysicsScene')
    assert scene.GetAttribute('physxScene:timeStepsPerSecond').Get()==60
    assert scene.GetAttribute('physxScene:enableGPUDynamics').Get() is False
    sim=omni.physx.get_physx_simulation_interface()
    sid=UsdUtils.StageCache.Get().Insert(stage).ToLongInt();assert sim.attach_stage(sid)
    sim.simulate(1/60,time);sim.fetch_results();time+=1/60
    view=omni.physics.tensors.create_simulation_view('numpy',sid);view.set_subspace_roots('/')
    door=view.create_rigid_body_view(DOOR)
    bodies={r['key']:view.create_rigid_body_view(r['path']) for r in rows}
    assert door.count==1 and all(v.count==1 for v in bodies.values())
    def snapshot():return door.get_transforms().copy().tobytes()+door.get_velocities().copy().tobytes()
    clock=NativeSceneClock(stage,sim,lambda:time,snapshot);owner.initialize(clock,view)
    helpers=dict(events=[],docked_object_count=0,tray_dock_enabled=False)
    def state():
        return {k:dict(transform_xyzw=v.get_transforms()[0].tolist(),velocity=v.get_velocities()[0].tolist()) for k,v in bodies.items()}
    report['initial']=state();report['samples']=[]
    for step in range(540):
        dt=clock.get_physics_dt();sim.simulate(dt,time);sim.fetch_results();time+=dt
        owner.after_physics(helpers)
        if step%60==59:report['samples'].append(dict(time_s=time,boxes=state()))
    report.update(status='diagnostic_completed',final=state(),simulation_time_s=time,
                  door_helper=owner.environment_helpers()['refrigerator'])
except Exception:
    report.update(status='error',error=traceback.format_exc())
finally:
    if owner:owner.close()
    if sim:sim.detach_stage()
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print('FRIDGE_BOX_SETTLE',report['status'],report.get('error',''),flush=True)
    omni.kit.app.get_app().post_quit(1 if report['status']=='error' else 0)
