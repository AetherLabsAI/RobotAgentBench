"""Keep the same live physical episode available for measured corrective joint actions.
Waiting only pauses physics for planning. Never restores cloth/robot state.
"""
from bimanual_lay_extension_v30 import LayExtension as BaseExtension
from bimanual_lay_plan_v30 import DATA
from pathlib import Path
import numpy as np,json,time,cv2
from scipy.spatial import cKDTree
class LayExtension(BaseExtension):
 def __init__(self,output,cloth_ids):
  super().__init__(output,cloth_ids);self.repair_index=0;self.repair_waiting=False
 def flat_readback(self,actual,cloth,uv):
  # Same canonical tabletop, plus geometric area/height/release checks. Diagnostic only.
  p=cloth[self.ids];on=(abs(p[:,2]-.451)<.012)&(p[:,0]>=-2.35)&(p[:,0]<=-1.574649)&(p[:,1]>=-.7210436)&(p[:,1]<=.4789564)
  # Topology is taken from this episode's initial geometry metadata, never a state snapshot.
  if not hasattr(self,'repair_faces'):
   ref=np.load(Path(__file__).parent/'runs/round12_diagonal_aligned_pads_0153/full_state.npz');faces=ref['faces'];self.repair_faces=faces[np.all(ref['cloth_id'][faces]==1,axis=1)];ref.close()
  f=self.repair_faces;tri=cloth[f];u=uv[f];rest=float(abs((u[:,1,0]-u[:,0,0])*(u[:,2,1]-u[:,0,1])-(u[:,1,1]-u[:,0,1])*(u[:,2,0]-u[:,0,0])).sum()/2);lo=p[:,:2].min(0)-.002;size=np.ceil((p[:,:2].max(0)-lo+.002)/.001).astype(int)+1
  if max(size)>10000:raise RuntimeError('Unbounded cloth geometry in final readback')
  mask=np.zeros((size[1],size[0]),np.uint8)
  for poly in np.round((tri[:,:,:2]-lo)/.001).astype(np.int32):cv2.fillConvexPoly(mask,poly,1)
  n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);normal=float(abs(n[:,2]).sum()/np.linalg.norm(n,axis=1).sum());gap={};poses=self.planner.robot.fk(actual)
  for side in ['l','r']:
   pts=[]
   for kind,local in zip(['inner','outer'],self.pad_points[side]):
    t=poses['gripper_'+side+'_'+kind+'_link4'];pts.append(local@t[:3,:3].T+t[:3,3])
   gap[side]=float(cKDTree(np.concatenate(pts)).query(p)[0].min())
  row=dict(table_node_fraction=float(on.mean()),projected_union_fraction=float(mask.sum()*1e-6/rest),horizontal_normal_fraction=normal,z_p05_p95=np.quantile(p[:,2],[.05,.95]).tolist(),minimum_pad_distances=gap,owner_approved=False,production_count=0)
  row['flat_and_clear_diagnostic']=bool(row['table_node_fraction']>=.95 and row['projected_union_fraction']>=.95 and normal>=.95 and row['z_p05_p95'][1]-row['z_p05_p95'][0]<=.015 and min(gap.values())>.005)
  return row
 def append_at_end(self,command,actual,cloth,uv,recent):
  if self.stage not in ['settled','awaiting_physical_repair']:return super().append_at_end(command,actual,cloth,uv,recent)
  original_stage=self.stage;self.stage='settled';row=self.flat_readback(actual,cloth,uv);self.events.append(dict(stage='strict_flat_and_release_readback',repair_index=self.repair_index,metrics=row,object_state_writes=0))
  if row['flat_and_clear_diagnostic']:
   np.savez_compressed(self.output/'observed_final_settled.npz',cloth_q=cloth,uv=uv,joint_names=np.array(list(actual)),joint_position=np.array(list(actual.values())),command_joint_names=np.array(list(command)),joint_target=np.array(list(command.values())))
   return super().append_at_end(command,actual,cloth,uv,recent)
  self.stage='awaiting_physical_repair';self.complete=False;self.repair_waiting=True
  for suffix in ['awaiting_physical_repair','repair_'+str(self.repair_index).zfill(3)]:
   np.savez_compressed(self.output/('observed_'+suffix+'.npz'),cloth_q=cloth,uv=uv,joint_names=np.array(list(actual)),joint_position=np.array(list(actual.values())),command_joint_names=np.array(list(command)),joint_target=np.array(list(command.values())))
  if original_stage=='settled':
   np.savez_compressed(self.output/'observed_settled.npz',cloth_q=cloth,uv=uv,joint_names=np.array(list(actual)),joint_position=np.array(list(actual.values())),command_joint_names=np.array(list(command)),joint_target=np.array(list(command.values())))
  request=self.output/('repair_request_'+str(self.repair_index).zfill(3)+'.json');self.save();(self.output/'repair_wait.json').write_text(json.dumps(dict(waiting=True,physics_paused_for_planning=True,request_path=str(request),metrics=row,object_state_restores=0),indent=2));print(json.dumps({'awaiting_live_physical_repair':str(request),'metrics':row}),flush=True)
  while not request.exists():time.sleep(.5)
  data=json.loads(request.read_text())
  if data.get('kind')=='finish_failed':
   self.failure='Measured flat/release check failed; '+str(data.get('reason',''));self.stage='failed';self.repair_waiting=False;self.save();return []
  if data.get('kind')!='joint_targets':raise RuntimeError('Live repair accepts only validated native joint targets')
  rows=[];previous=command;allowed={n for n in command if '_arm_'in n or '_gripper_'in n};names=self.planner.robot.by_name
  for i,item in enumerate(data['actions']):
   updates=item['targets'];assert set(updates)<=allowed,'Repair cannot command base/torso or object state';goal=self.planner.robot.calibrate_loops(previous|{n:float(x)for n,x in updates.items()})
   assert all(np.isfinite(list(goal.values())))
   for n in updates:
    if n in names:
     low,high=names[n]['limits'];assert low-1e-6<=goal[n]<=high+1e-6,(n,goal[n],'native limit')
    if '_arm_'in n:assert abs(goal[n]-previous[n])<=.7/20+1e-5,(n,'native action-rate bound')
   if i%3==0 or i==len(data['actions'])-1:
    poses=self.planner.robot.fk(goal)
    assert max(self.planner.collision(poses,s).max()for s in ['L','R'])<=.004,'Unchanged native collision gate rejected corrective action'
   rows.append((goal,str(item.get('phase','LR_live_physical_repair'))));previous=goal
  assert rows;self.repair_index+=1;self.repair_waiting=False;self.events.append(dict(stage='validated_live_joint_repair',request=str(request),frames=len(rows),physics_state_writes=0));self.save();(self.output/'repair_wait.json').write_text(json.dumps(dict(waiting=False,accepted_request=str(request),frames=len(rows),object_state_restores=0),indent=2));return rows
