"""Read-only tracking of physically selected material and actual jaw closure."""
from pathlib import Path
import json,time,numpy as np
p=Path(__file__).parent;n='continuous_full_closure_removal_0179';ep=p/'runs'/n;log=p/(n+'.log');last=-1;selected=None
while not(ep/'trial.json').exists():
 try:
  if selected is None:
   for line in log.read_text().splitlines():
    if '"actual_open_mouth_two_layer_selection"'not in line:continue
    event=json.loads(line)
    if event.get('decision')=='grasp':selected=event['selected'];break
  if selected is not None:
   with np.load(ep/'live_observation.npz')as z:
    frame=int(z['frame'])
    if frame!=last:
     ids=selected['pair_indices'];xyz=z['cloth_q'][ids];initial=np.asarray(selected['pair_xyz']);j=dict(zip(z['joint_names'],map(float,z['joint_position'])));old=z['cloth_q'][z['cloth_id']==0];row=dict(frame=frame,source_frame=int(z['source_frame']),phase=str(z['phase']),pair_indices=ids,selected_material_lift_m=(xyz[:,2]-initial[:,2]).tolist(),selected_material_xyz=xyz.tolist(),jaw_actual_rad={k:j[k]for k in ['idx71_gripper_r_inner_joint1','idx81_gripper_r_outer_joint1']},old_cloth_z_quantiles=np.percentile(old[:,2],[0,5,50,95,100]).tolist(),force_confirmed=False,read_only=True)
     with(ep/'selected_patch_progress.jsonl').open('a')as f:f.write(json.dumps(row)+'\n')
     last=frame
 except(FileNotFoundError,ValueError,KeyError):pass
 time.sleep(5)
