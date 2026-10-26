"""Offline native-geometry audit; never changes simulation state."""
import json,sys,re
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
R=Path(__file__).resolve().parent
REPO=Path('/home/aether/workspace/agent_bench')
sys.path[:0]=[str(R/'src'),str(REPO/'experiments/physics_fluid'),str(REPO)]
from robot_furniture_audit import primitive_samples,inside_primitive
from real2sim.teleop.task6_drawer_sequence_contract import FAMILY4_STANDARD_ARM_RAD
from tablecloth_robot_kinematics import Robot
META=REPO/'.codex-tmp/tablecloth-action-loop-20260905/robot_native_collision.json'
m=json.loads(META.read_text())
shapes=[s for s in m['nonhand_collision_shapes'] if s.get('primitive')]
samples=[primitive_samples(s['primitive'],10) for s in shapes]
adj={frozenset((j['parent'],j['child'])) for j in m['joints']}
# Preserve asset-authored filtered collision pairs, including shoulder mounting overlaps.
asset_text=Path(m['asset']).read_text()
for match in re.finditer(r'over "([^"/]+)"[^{}]*\{(?:(?!\n    over ).)*?rel physics:filteredPairs = (\[[^\]]*\]|<[^>]*>)',asset_text,re.S):
 for target in re.findall(r'</G01/([^>]+)>',match.group(2)):
  adj.add(frozenset((match.group(1),target)))
def audit(poses):
 rows=[]
 for i,a in enumerate(shapes):
  if a['link'] not in poses:continue
  A=poses[a['link']];points=samples[i]@A[:3,:3].T+A[:3,3]
  for b in shapes:
   if a['link']==b['link'] or b['link'] not in poses or frozenset((a['link'],b['link'])) in adj:continue
   if not ('arm_' in a['link'] or 'arm_' in b['link']):continue
   B=poses[b['link']];depth=float(inside_primitive(b['primitive'],(points-B[:3,3])@B[:3,:3]).max())
   if depth>.002:rows.append({'source':a['link'],'target':b['link'],'penetration_mm':round(depth*1000,2)})
 return sorted(rows,key=lambda x:-x['penetration_mm'])
def main():
 trial=R/'trials/explore_810_grasp_diagonal_yawscan';z=np.load(trial/'full_state.npz');names=z['joint_names'].tolist();bn=z['body_names'].tolist();out={'scope':'native primitive surface sampling; excludes directly connected links; not complete mesh collision certification','frames':[]}
 by={j['name']:j for j in m['joints']}
 for f in [0,140,160,200,240,273]:
  poses={}
  for n,x in zip(bn,z['body_q'][f]):
   T=np.eye(4);T[:3,:3]=Rotation.from_quat(x[3:]).as_matrix();T[:3,3]=x[:3];poses[n]=T
  row={'frame':f,'phase':str(z['phase'][f]),'collisions':audit(poses),'arms':{}}
  for sd,home in [('L',FAMILY4_STANDARD_ARM_RAD[:7]),('R',FAMILY4_STANDARD_ARM_RAD[7:])]:
   ns=Robot.arm_names(sd);q=z['joint_position'][f,[names.index(n) for n in ns]];lim=np.array([by[n]['limits'] for n in ns]);row['arms'][sd]={'degrees':np.rad2deg(q).round(2).tolist(),'standard_wait_degrees':np.rad2deg(home).round(2).tolist(),'limits_deg':np.rad2deg(lim).round(2).tolist(),'max_limit_violation_deg':float(np.rad2deg(np.maximum(lim[:,0]-q,q-lim[:,1])).clip(0).max())}
  out['frames'].append(row)
 out['native_filtered_pairs']=[sorted(x) for x in adj]
 out['finding']='IK enforces native +/-180 degree limits, but no self collision or natural wait posture objective. Bad right pose is already present at checkpoint start.'
 (R/'arm_posture_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
 for row in out['frames']:print(row['frame'],row['collisions'][:4])
if __name__=='__main__':main()
