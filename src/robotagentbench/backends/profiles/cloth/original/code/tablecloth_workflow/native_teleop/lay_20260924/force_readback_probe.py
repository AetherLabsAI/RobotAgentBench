"""Read-only native force instrumentation; never treats explicit effort as PD effort."""
from pathlib import Path
import json,numpy as np
class ForceReadback:
 def __init__(self,out,view,stage,robot,names,body_names,cloth_views):
  from pxr import Usd,UsdPhysics
  self.out=Path(out);self.view=view;self.rows=[];self.disabled={};self.names=names;self.body_names=body_names
  self.methods=['get_dof_projected_joint_forces','get_dof_actuation_forces','get_link_incoming_joint_force','get_link_accelerations','get_link_velocities','get_link_transforms','get_dof_positions','get_dof_velocities','get_dof_position_targets']
  self.static={}
  for n in ['get_dof_stiffnesses','get_dof_dampings','get_dof_max_forces','get_masses','get_coms']:
   try:self.static[n]=getattr(view,n)().detach().cpu().numpy()[0].copy()
   except Exception as e:self.disabled[n]=repr(e)
  graph=[]
  for prim in Usd.PrimRange(robot):
   if not prim.IsA(UsdPhysics.Joint):continue
   j=UsdPhysics.Joint(prim);q=j.GetLocalRot1Attr().Get();graph.append(dict(name=prim.GetName(),type=prim.GetTypeName(),body0=[str(x)for x in j.GetBody0Rel().GetTargets()],body1=[str(x)for x in j.GetBody1Rel().GetTargets()],local_rot1_xyzw=list(q.GetImaginary())+[q.GetReal()],excluded=prim.GetAttribute('physics:excludeFromArticulation').Get(),enabled=j.GetJointEnabledAttr().Get()))
  self.body_physics={n:{key:stage.GetPrimAtPath('/World/Robot/'+n).GetAttribute(key).Get() for key in ['physxRigidBody:disableGravity','physxRigidBody:linearDamping','physxRigidBody:angularDamping']} for n in body_names}
  self.meta=dict(joint_names=names,body_names=body_names,joint_graph=graph,disabled=self.disabled,cloth_api_methods=[n for n in dir(cloth_views[0].view) if any(x in n for x in ['force','stress','contact'])],meaning={'get_dof_actuation_forces':'Explicit applied effort only, independent of implicit PD motor force; zero does not mean motor off.','get_dof_projected_joint_forces':'Measured joint wrench projected on motion axis, not isolated cloth pressure.','get_link_incoming_joint_force':'Measured six-axis joint reaction in child joint frame, not isolated cloth pressure.','cloth_contact_force':'Unsupported by installed native deformable contact-report interface; no fabricated zero reading.'},official_limitations='https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/guides/current_limitations.html')
 def sample(self,frame,substep,source,phase,rigid,pair):
  d=dict(frame=np.int32(frame),substep=np.int32(substep),source_frame=np.int32(source),phase=str(phase),rigid_net_force=rigid.copy(),pad_pair_force=pair.copy())
  for n in self.methods:
   if n in self.disabled:continue
   try:d[n]=getattr(self.view,n)().detach().cpu().numpy()[0].copy()
   except Exception as e:self.disabled[n]=repr(e)
  self.rows.append(d)
 def save(self):
  if self.rows:
   keys=set.intersection(*(set(r)for r in self.rows));data={k:np.asarray([r[k] for r in self.rows])for k in keys};data.update({'static_'+k:v for k,v in self.static.items()});np.savez_compressed(self.out/'native_force_readback.npz',**data)
  self.meta['body_physics']=self.body_physics;self.meta['disabled']=self.disabled;self.meta['substep_sample_count']=len(self.rows);self.meta['available_readbacks']=sorted(set.intersection(*(set(r)for r in self.rows))) if self.rows else [];(self.out/'native_force_readback_metadata.json').write_text(json.dumps(self.meta,indent=2)+'\n')
