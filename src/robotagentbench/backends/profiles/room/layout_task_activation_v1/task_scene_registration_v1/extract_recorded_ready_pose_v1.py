"""Copy an existing website replay's measured ready pose, without optimization."""
import json,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'room_demo_v6'))
from action22 import JOINT_NAMES
run=Path('/home/lingjun/robot_agent_arena/api-protocol/evaluation/tasks/coffee_machine/artifacts/batch-20260930-r4/1-1/source-run-3')
trace=run/'run/episode/trace.json';result=run/'run/report.json'
d=json.loads(trace.read_text());o=d['initial_observation'];r=json.loads(result.read_text())
assert d['episode_config']['task_parameters']['source_episode']=='042_right_239.7deg'
assert o['source']=='isaac_physics' and o['step_index']==0 and r['verdict']=='success'
q={**o['joint_positions_rad'],**o['joint_positions_m']}
out=ROOT/'recorded_ready_pose_042_v1.json';assert not out.exists()
report=dict(status='existing_recorded_ready_pose_no_generated_joint_angles',api='1-1',episode='042_right_239.7deg',
 step_index=0,time_s=o['time_s'],targets_rad={n:q[n] for n in JOINT_NAMES},
 units='radians except idx01_body_joint1 in meters',source=str(trace),
 source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (trace,result)},
 provenance='initial measured settled observation of successful public replay; not invented IK',
 new_room_collision_validated=False)
out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
