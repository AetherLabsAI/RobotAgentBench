"""Read-only all-material native pad geometry for physical recovery actions."""
from pathlib import Path
import sys,json,numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from bimanual_lay_plan_v30 import LayPlanner,DATA
from cooked_closure_guard import CookedClosureGuard
p=Path(__file__).parent;ep=p/'runs'/sys.argv[1];stage=sys.argv[2];z=np.load(ep/('observed_'+stage+'.npz'));q=z['cloth_q'];uv=z['uv'];a=dict(zip(z['joint_names'],map(float,z['joint_position'])));pl=LayPlanner();poses=pl.robot.fk(a);names=list(poses);bodies=np.array([np.r_[t[:3,3],Rotation.from_matrix(t[:3,:3]).as_quat()]for t in poses.values()]);cad=np.load(DATA/'robot_native_collision.npz');result=dict(stage=stage,read_only=True,force_confirmed=False,hands={})
for side in ['l','r']:
 dist=[]
 for kind in ['inner','outer']:
  n='gripper_'+side+'_'+kind+'_link4';t=poses[n];dist.append(cKDTree(cad['points_'+n]).query((q-t[:3,3])@t[:3,:3])[0])
 near=np.flatnonzero((dist[0]<.005)&(dist[1]<.005));g=CookedClosureGuard(ep/('native_pad_cooked_convexes.json'if side=='r'else'native_left_pad_cooked_convexes.json'));alias=names if side=='r'else[n.replace('gripper_l_','gripper_r_')if n.startswith('gripper_l_')else'unused_'+n for n in names];valid=[int(i)for i in near if g.check(q[i],bodies,alias)['geometry_allows_lift']];row=dict(bilateral_proximity=len(near),cooked_between=len(valid),indices=valid,uv_mean=np.mean(uv[valid],axis=0).tolist()if valid else None,uv_bounds=[uv[valid].min(0).tolist(),uv[valid].max(0).tolist()]if valid else None,jaw_angles=[a[n]for n in a if ('_gripper_'+side+'_')in n and n.endswith('joint1')],gap_axis=poses['gripper_'+side+'_base_link'][:3,1].tolist());result['hands'][side]=row
out=ep/('recovery_grip_'+stage+'.json');out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
