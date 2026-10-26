"""Task8 session preparation through the existing House randomization owners."""
from pathlib import Path
import copy,json
import numpy as np

def _attribute_value(attribute):
 from pxr import Sdf
 value=attribute.Get()
 return str(value.resolvedPath or value.path) if isinstance(value,Sdf.AssetPath) else str(value)

def _world(stage,path):
 from pxr import UsdGeom
 return np.asarray(UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(0)).T

def open_scene(task_id,seed,layout_id):
 from pxr import Usd
 import session_scene
 name=f'task8_{task_id.replace(".","_")}_{seed}'
 session_scene.open_session(name,[],task_ids=[task_id],layout=layout_id)
 return Usd.Stage.Open(str(session_scene.CANONICAL)),name

def select_layout(stage, task, plan):
 """Reject initial base overlaps within the existing 50-layout domain.

 Use geometry-derived extents: native cached extent attributes can be stale.
 This preflight does not replace live full-body or manipulation validation.
 """
 from pxr import Usd,UsdGeom,UsdPhysics
 from real2sim.production import house_layout,house_topology,house_topology_visual as topology
 requested,catalog=topology.candidate_for_id(plan['layout_id'])
 base=stage.GetPrimAtPath('/World/Robot/base_link/collisions')
 if not base:raise RuntimeError('Native base collision geometry is missing')
 points=[]
 for prim in Usd.PrimRange(base,Usd.TraverseInstanceProxies()):
  if not prim.HasAPI(UsdPhysics.CollisionAPI):continue
  if prim.GetAttribute('physics:collisionEnabled').Get() is False:continue
  extent=UsdGeom.Boundable.ComputeExtentFromPlugins(UsdGeom.Boundable(prim),0)
  if extent is None:raise RuntimeError('Cannot compute native collider extent: '+str(prim.GetPath()))
  low,high=np.asarray(extent)
  if not np.isfinite([low,high]).all() or not np.all(high>=low):raise RuntimeError('Invalid native collider extent')
  corners=np.array([[x,y,z] for x in (low[0],high[0]) for y in (low[1],high[1]) for z in (low[2],high[2])])
  matrix=np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)).T
  points.extend((corners@matrix[:3,:3].T+matrix[:3,3])[:,:2])
 if not points:raise RuntimeError('Cannot read native base footprint')
 points=np.asarray(points);frame=task['runtime']['environment']['layout_frame']
 order=[requested]+[topology.candidate_for_ordinal(i,catalog)[0] for i in house_layout.task_layout_permutation(task['key'],seed=int(plan['seed']))]
 seen=set();attempts=[]
 for candidate in order:
  if candidate['id'] in seen:continue
  seen.add(candidate['id'])
  t=house_layout.transform_for_frame(topology.topology_layout(candidate,catalog['interaction_modules']),frame)
  angle=np.deg2rad(t.yaw_deg);rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
  pivot=np.asarray(t.pivot_xy_m)
  transformed=(points-pivot)@rotation.T+pivot+t.translate_xy_m
  low=transformed.min(0);high=transformed.max(0);blockers=[]
  for module,pose in candidate['module_poses'].items():
   x,y,w,h=house_topology._fixture_rect(catalog['interaction_modules'][module],pose)
   if np.all(high>np.array([x,y])-.015) and np.all(low<np.array([x+w,y+h])+.015):blockers.append(module)
  attempts.append({'layout_id':candidate['id'],'base_footprint':[low.tolist(),high.tolist()],'blocking_modules':blockers})
  if not blockers:
   return candidate,catalog,{'requested_layout_id':requested['id'],'resolved_layout_id':candidate['id'],'attempts':attempts,'selection':'existing_50_layout_domain_rejection_sampling','margin_m':.015,'live_full_body_validation_required':True}
 raise RuntimeError('No existing House layout admits the initialized native base: '+str(attempts))


def _prepare_clutter_initialization(stage,task,targets,cloth_path,plan,out,prepare_clutter,session_scene):
 scene=Path(stage.GetRootLayer().realPath)
 # Resolve a legal clutter draw before any physics step. The delivered owner
 # sampler and its full ranges stay unchanged; only geometrically invalid
 # initialization plans are rejected, never failed physical trajectories.
 initial_layer=stage.GetRootLayer().ExportToString()
 initial_session=json.loads(session_scene.SIDECAR.read_text())
 clutter_rng=np.random.default_rng([int(plan['seed']),8,20260915])
 clutter_attempts=[]
 for attempt in range(32):
  clutter_seed=int(plan['seed']) if attempt==0 else int(clutter_rng.integers(0,2**31-1))
  if attempt:
   stage.GetRootLayer().ImportFromString(initial_layer)
   stage.GetRootLayer().Save()
   session_scene._atomic_sidecar(initial_session)
  try:
   clutter=prepare_clutter(scene=scene,task_key=task['key'],target=targets[0],seed=clutter_seed,protected=[*targets[1:],cloth_path])
   clutter_attempts.append({'attempt':attempt,'owner_seed':clutter_seed,'accepted':True})
   break
  except ValueError as error:
   if not str(error).startswith('sofa prop outside middle two seats:'):raise
   clutter_attempts.append({'attempt':attempt,'owner_seed':clutter_seed,'accepted':False,'reason':str(error)})
   (out/'clutter_initialization_sampling.json').write_text(json.dumps({'episode_seed':int(plan['seed']),'attempts':clutter_attempts,'physics_started':False},indent=2))
 else:
  raise RuntimeError('No legal owner clutter initialization after 32 geometric rejections')
 (out/'clutter_initialization_sampling.json').write_text(json.dumps({'episode_seed':int(plan['seed']),'attempts':clutter_attempts,'physics_started':False},indent=2))
 return clutter


def prepare(stage,task,targets,cloth_path,plan,out):
 from pxr import Usd,UsdPhysics,UsdGeom
 import session_scene,house_topology_visual as topology,appearance
 from real2sim.production.prepare_physical_distractors import prepare as prepare_clutter
 scene=Path(stage.GetRootLayer().realPath)
 stage.GetRootLayer().Save()
 before=_world(stage,'/World/Robot')
 clutter=_prepare_clutter_initialization(stage,task,targets,cloth_path,plan,out,prepare_clutter,session_scene)
 frames={row['prim_path']:row['surface'] for row in clutter['objects']}
 frame=task['runtime']['environment']['layout_frame']
 frames.update({p:frame for p in [*targets,cloth_path]})
 candidate,catalog,selection=select_layout(stage,task,plan)
 resolved={**plan,'requested_layout_id':plan['layout_id'],'layout_id':candidate['id'],'layout_selection':selection}
 (out/'resolved_episode_plan.json').write_text(json.dumps(resolved,indent=2))
 sidecar=json.loads(session_scene.SIDECAR.read_text())
 if sidecar.get('status')!='open' or sidecar.get('task_ids')!=[task['id']]:
  raise RuntimeError('Layout selection requires the current single-task initialization session')
 sidecar.update(layout=candidate['id'],layout_selection=selection)
 session_scene._atomic_sidecar(sidecar)
 layout=topology.apply_topology_overlay(scene,candidate,catalog['interaction_modules'],task_frame=frame,robot_path='/World/Robot',extra_prim_frames=frames,physics_overlay=True)
 visual=copy.deepcopy(plan['appearance'])
 visual['topology_visual']=topology.build_visual_plan(candidate,catalog['interaction_modules'],visual)
 appearance_result=appearance.apply_plan(scene,visual)
 stage=Usd.Stage.Open(str(scene))
 # Decor placement is done by the shared appearance owner. Existing native
 # visible geometry must have physical collisions in this live episode too.
 from physical_distractors import ensure_scene_distractor_collisions,validate_bodies
 from visual_scene_geometry import visible_bounds
 decor=[p for group in topology.DECOR_SLOT_ROOTS.values() for p in group if stage.GetPrimAtPath(p) and stage.GetPrimAtPath(p).IsActive() and visible_bounds(stage,p) is not None]
 decor_physics=ensure_scene_distractor_collisions(stage,decor)
 stage.GetRootLayer().Save()
 transform=_world(stage,'/World/Robot')@np.linalg.inv(before)
 anchors={}
 for row in layout['module_scene_overlay']['authored_prims']:
  path=row.get('path',row.get('prim_path'))
  if not path:continue
  prim=stage.GetPrimAtPath(path)
  if not prim or not prim.IsActive() or path=='/World/Robot' or prim.HasAPI(UsdPhysics.RigidBodyAPI):continue
  anchors[path]=_world(stage,path).tolist()
 attrs={'appearance':{},'lighting':{}}
 for prim in stage.Traverse():
  if str(prim.GetPath()).startswith('/World/Robot'):continue
  light=prim.GetTypeName().endswith('Light')
  shader=prim.GetTypeName()=='Shader'
  if not (light or shader):continue
  for attr in prim.GetAttributes():
   if attr.GetName().startswith('inputs:') and attr.Get() is not None:
    attrs['lighting' if light else 'appearance'][str(attr.GetPath())]=_attribute_value(attr)
 bodies=clutter['applied_prim_paths']
 state={'scene':str(scene),'frame_to_world':transform.tolist(),'anchors':anchors,'attributes':attrs,'distractor_paths':bodies,'clutter':clutter,'layout':layout,'layout_selection':selection,'appearance':appearance_result,'decor_physics':decor_physics}
 (out/'environment_preparation.json').write_text(json.dumps(state,indent=2,default=str))
 return stage,state

def readback(stage,prepared):
 from pxr import Usd,UsdPhysics,UsdGeom
 axes={}
 for kind in ['appearance','lighting']:
  expected=prepared['attributes'][kind]
  actual={p:_attribute_value(stage.GetAttributeAtPath(p)) for p in expected}
  axes[kind]={'verified':bool(expected) and actual==expected,'readback':actual}
 actual_anchors={p:_world(stage,p).tolist() for p in prepared['anchors']}
 axes['applicable_scene_layout']={'verified':bool(actual_anchors) and all(np.allclose(actual_anchors[p],expected,rtol=0,atol=1e-5) for p,expected in prepared['anchors'].items()),'readback':actual_anchors,'layout_id':prepared['layout']['topology_id'],'candidate_status':'2d_candidate_until_live_validation'}
 actual=[]
 for path in prepared['distractor_paths']:
  prim=stage.GetPrimAtPath(path)
  colliders=[str(p.GetPath()) for p in Usd.PrimRange(prim) if p.HasAPI(UsdPhysics.CollisionAPI) and p.GetAttribute('physics:collisionEnabled').Get() is not False]
  matrix=_world(stage,path)
  actual.append({'path':path,'world_transform':matrix.tolist(),'colliders':colliders,'valid':bool(colliders) and bool(np.isfinite(matrix).all())})
 axes['physical_distractors']={'verified':bool(actual) and all(p['valid'] for p in actual),'readback':actual}
 return axes


def is_base_support_contact(hit, floor_paths):
 """Recognize shallow base support on authored floors, never arm contacts."""
 return (hit.get('kind')=='world'
         and hit.get('a','').startswith('/World/Robot/base_link/')
         and hit.get('b') in floor_paths
         and 0<=hit.get('depth_m',float('inf'))<=.005)
