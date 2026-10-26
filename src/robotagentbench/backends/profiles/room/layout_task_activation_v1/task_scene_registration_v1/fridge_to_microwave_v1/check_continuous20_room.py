"""Read-only source/current-room compatibility, not a physical rollout."""
import json
from pathlib import Path
import numpy as np
from pxr import Usd,UsdGeom
from fridge_to_microwave_v1.l1_trial_inputs import audit,ROOT,MW

base=ROOT/'fridge_to_microwave_v1/l1_continuous20_v1'
manifest=json.loads((ROOT/'coffee_initialized_rooms_v1/layout03/composition.json').read_text())
stage=Usd.Stage.Open(manifest['scene'])
raw,_,old=audit(stage,manifest)
job=json.loads((base/'source/job.json').read_text())
source=np.load(base/'source/recording_60hz.npz')
converted=np.load(base/'converted/actions_20hz.npz')
from scipy.spatial.transform import Rotation
def column_pose(p):
    m=np.eye(4);m[:3,:3]=Rotation.from_quat(p[3:7]).as_matrix();m[:3,3]=p[:3];return m
source_to_room=np.asarray(old['source_to_room_matrix']).T@np.linalg.inv(np.asarray(job['fridge_world_from_source']))
cache=UsdGeom.XformCache()
def actual(path):return np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path))).T
mw_initial=np.load(MW/'initial_state.npz');body_names=mw_initial['rigid_body_paths'].tolist()
source_mw=np.asarray(job['microwave_world_from_source'])@column_pose(mw_initial['body_pose_xyzw'][body_names.index('/World/Assets/DiningTableMicrowave/Asset/body')])
predicted_mw=source_to_room@source_mw
current_mw=actual('/World/Assets/DiningTableMicrowave/Asset/body')
predicted_box=source_to_room@np.asarray(job['initial_box'])
current_box=actual('/World/Assets/MicrowaveBeefbox')
report=dict(status='preflight_only',control_hz=20,actions=len(converted['action']),
    source_to_room_matrix=source_to_room.tolist(),
    predicted_microwave_position_m=predicted_mw[:3,3].tolist(),
    current_microwave_position_m=current_mw[:3,3].tolist(),
    microwave_position_difference_m=float(np.linalg.norm(predicted_mw[:3,3]-current_mw[:3,3])),
    microwave_rotation_difference_deg=float(np.rad2deg(Rotation.from_matrix(predicted_mw[:3,:3]@current_mw[:3,:3].T).magnitude())),
    target_initial_position_difference_m=float(np.linalg.norm(predicted_box[:3,3]-current_box[:3,3])),
    full_physical_replay_started=False,task_success=None,
    scope='Same layout number is not proof of same current randomized scene; no scene changes made')
with (base/'room_compatibility.json').open('x') as f:json.dump(report,f,indent=2)
print(json.dumps(report))
