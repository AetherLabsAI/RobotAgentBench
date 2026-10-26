"""Short native host test. No action trajectory and no claimed manipulation."""
import sys,json,traceback,hashlib,argparse
from pathlib import Path
ROOT=Path('/home/lingjun/robot_agent_arena/layout_task_activation_v1/task_scene_registration_v1')
sys.path[:0]=[str(ROOT),'/home/lingjun/robot_agent_arena/api-protocol']
import numpy as np
import omni.kit.app,omni.physx,omni.physics.tensors
from pxr import Sdf,Usd,UsdGeom,UsdPhysics,UsdUtils
from multi_task_room_v2 import MultiTaskRoom
from coffee_hotwake_v1 import NativeSceneClock
from room_fridge_helper_v1 import FRIDGE
from fridge_to_microwave_v1.support import apply
from fridge_to_microwave_v1.scene import ROBOT,MICROWAVE_ROOT,PLATE_BODY,STAND
out=Path('/pilot');result=dict(status='started',full_trajectory_verified=False,runtime_pose_writes=0,
    scope='native selected task adapter, original scene assets; not continuous manipulation')
owner=sim=None;time=0.
try:
    m=json.loads((ROOT/'coffee_initialized_rooms_v1/layout03/composition.json').read_text())
    layer=Sdf.Layer.CreateAnonymous();layer.subLayerPaths=[m['scene']]
    masks=[FRIDGE,ROBOT,MICROWAVE_ROOT,PLATE_BODY.rsplit('/',1)[0],STAND,
           '/World/PhysicsScene','/World/PhysicsMaterials','/World/MicrowaveStandLooks','/World/Background/Shell/Floor']
    masks.extend(r['path'] for r in m['objects'] if r['id'] in ('beef_box','salad_box','leftover_box'))
    s=Usd.Stage.OpenMasked(layer,Usd.StagePopulationMask(masks))
    UsdGeom.SetStageMetersPerUnit(s,1);UsdGeom.SetStageUpAxis(s,UsdGeom.Tokens.z)
    if not s.GetPrimAtPath(PLATE_BODY):apply(s)
    s.SetEditTarget(s.GetSessionLayer())
    hashes={l.realPath:hashlib.sha256(Path(l.realPath).read_bytes()).hexdigest() for l in s.GetUsedLayers() if l.realPath}
    req=json.loads((ROOT/'multi_task_episode_task9_v1.json').read_text())
    req['tasks']=[r for r in req['tasks'] if r['type']=='fridge_to_microwave']
    parser=argparse.ArgumentParser();parser.add_argument('--payload-variant',choices=['beef_box','leftover_box'])
    args,_=parser.parse_known_args()
    if args.payload_variant:req['tasks'][0]['parameters']['payload_variant']=args.payload_variant
    sensor=json.loads((ROOT.parent/'hotwake_probe_inputs.json').read_text())['sensor']
    owner=MultiTaskRoom(s,m,out/'room','task9-native-binding',request=req,sensor=sensor)
    sim=omni.physx.get_physx_simulation_interface();sid=UsdUtils.StageCache.Get().Insert(s).ToLongInt()
    assert sim.attach_stage(sid)
    sim.simulate(1/60,time);sim.fetch_results();time+=1/60
    view=omni.physics.tensors.create_simulation_view('numpy',sid);view.set_subspace_roots('/')
    paths=[str(p.GetPath()) for p in s.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    bodies=view.create_rigid_body_view(paths)
    def snapshot():return bodies.get_transforms().copy().tobytes()+bodies.get_velocities().copy().tobytes()
    clock=NativeSceneClock(s,sim,lambda:time,snapshot);owner.initialize(clock,view)
    helpers=dict(events=[],docked_object_count=0,tray_dock_enabled=False)
    for _ in range(240):
        dt=clock.get_physics_dt();sim.simulate(dt,time);sim.fetch_results();time+=dt;owner.after_physics(helpers)
    owner.accept_initial();port=owner.ports['fridge_microwave']
    # Isolate reader/scorer contract. Selection geometry is a diagnostic input,
    # not a robot-motion claim. No robot pose or velocity is written.
    lo,hi=port.activation_geometry()[0];center=np.mean([lo,hi],axis=0)
    selection_pose=dict(xy=[float(center[0]-.5),float(center[1])],yaw=0)
    records=[]
    for _ in range(20):
        plan=owner.begin_control(selection_pose,command_vw=[0,0],base_velocity=[0]*6)
        assert plan['current_task']=='fridge_microwave' and plan['physics_hz']==60 and not plan['switch_pending'],plan
        for _ in range(plan['substeps']):
            sim.simulate(plan['physics_dt_s'],time);sim.fetch_results();time+=plan['physics_dt_s'];owner.after_physics(helpers)
        records.append(owner.finish_control(completed_substeps=plan['substeps']))
    assert port.index==20 and port.score['score']==0 and not port.score['success'],port.score
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    result.update(status='passed',control_hz=20,selected_controls=port.index,score=port.score,
                  native_details=port.source.latest['details'],clock_transitions=owner.transitions,
                  source_layers_unchanged=True,selection_pose_source='contract_fixture_not_native_navigation',
                  simulation_time_s=time)
except Exception:result.update(status='failed',error=traceback.format_exc())
finally:
    try:
        if owner:owner.close()
        if sim:sim.detach_stage()
    except Exception:result['cleanup_error']=traceback.format_exc();result['status']='failed'
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print('TASK9_NATIVE',result['status'],result.get('error',''),flush=True)
    omni.kit.app.get_app().post_quit(0 if result['status']=='passed' else 1)
