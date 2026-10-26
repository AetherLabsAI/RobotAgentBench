"""Native-filter-aware arm self collision screening added after measured148 failure.
Uses conservative convex face separation; no collision filter/physics changes.
"""
from bimanual_lay_plan_v18 import LayPlanner as BasePlanner,DATA
from pathlib import Path
from scipy.spatial import ConvexHull
import numpy as np,json
class LayPlanner(BasePlanner):
 def __init__(self):
  super().__init__();meta=json.loads((DATA/'robot_native_collision.json').read_text());filters=json.loads((Path(__file__).parent/'native_collision_filters_readonly.json').read_text());skip=set()
  for r in filters['filtered_pairs']:
   for b in r['filtered']:skip.add(frozenset([r['path'].split('/')[-1],b.split('/')[-1]]))
  for j in filters['joints']:
   if j['enabled']and not j['collision_enabled']and j['body0']and j['body1']:skip.add(frozenset([j['body0'][0].split('/')[-1],j['body1'][0].split('/')[-1]]))
  self.self_hulls={}
  for n,pts in self.shapes.items():
   if not n.startswith(('arm_','gripper_')):continue
   h=ConvexHull(pts);eq=np.unique(np.round(h.equations,10),axis=0);self.self_hulls[n]=(pts[h.vertices],eq)
  self.self_pairs={}
  for side in ['L','R']:
   names=[n for n in self.self_hulls if n.startswith(('arm_'+side.lower()+'_','gripper_'+side.lower()+'_'))];pairs=[]
   for i,a in enumerate(names):
    for b in names[i+1:]:
     if a.startswith('gripper_')and b.startswith('gripper_'):continue
     if frozenset([a,b])in skip:continue
     pairs.append((a,b))
   self.self_pairs[side]=pairs
 def self_clearances(self,poses,side):
  world={}
  for n,(pts,eq)in self.self_hulls.items():
   if not n.startswith(('arm_'+side.lower()+'_','gripper_'+side.lower()+'_')):continue
   t=poses[n];q=pts@t[:3,:3].T+t[:3,3];normal=eq[:,:3]@t[:3,:3].T;world[n]=(q,np.c_[normal,eq[:,3]-normal@t[:3,3]])
  values=[]
  for a,b in self.self_pairs[side]:
   qa,ea=world[a];qb,eb=world[b];aabb=max(float((qa.min(0)-qb.max(0)).max()),float((qb.min(0)-qa.max(0)).max()))
   if aabb>.012:values.append(aabb);continue
   # Any separating face proves separation. Missing edge-cross axes can only
   # reject a safe pose, never accept an intersecting pair.
   gap=max(float((qa@eb[:,:3].T+eb[:,3]).min(axis=0).max()),float((qb@ea[:,:3].T+ea[:,3]).min(axis=0).max()));values.append(gap)
  return np.asarray(values)
 def collision(self,poses,side):
  return np.r_[super().collision(poses,side),np.maximum(0,.006-self.self_clearances(poses,side))]
 def clear_left_torso(self,v,cloth):
  from scipy.spatial import cKDTree
  self.cloth_tree=cKDTree(cloth);names=self.robot.arm_names('L');failures=[]
  try:
   # Straighten elbow before changing aperture. This increases the measured
   # inner-pad/upper-arm clearance instead of translating a tightly bent arm.
   for joint,delta in [(3,.4),(3,.2),(1,-.4),(0,.2),(3,.6)]:
    goal=v.copy();goal[names[joint]]+=delta
    try:
     rows=self.interpolate(v,goal,'L',120,'L_unfold_elbow_before_aperture');narrow=self.robot.calibrate_loops(goal|{'idx31_gripper_l_inner_joint1':-.35,'idx41_gripper_l_outer_joint1':.35});self.interpolate(goal,narrow,'L',40,'offline_aperture_clearance_check')
     self.last_torso_clearance_plan=dict(joint=names[joint],delta_rad=delta,frames=len(rows),keeps_original_aperture=True,native_collision_surfaces=True,self_collision_checked=True,object_state_writes=0);return rows
    except RuntimeError as e:failures.append(str(e))
   raise RuntimeError(('No self/torso/cloth-cleared staging move',failures))
  finally:self.cloth_tree=None
