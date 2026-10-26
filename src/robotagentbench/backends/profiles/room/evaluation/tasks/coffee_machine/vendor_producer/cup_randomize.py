"""Deterministic XY/yaw proposals and Cartesian retargeting of the successful placement."""
import copy,hashlib,json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scene_pose import machine_delta,INITIAL_SCENE,COLLISION_DIRECTORY
from cup_variants import SPECS,spec,handle_shift,asset_path
from preflight import create_model,configure,interpolate
B=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def context():
 files=[B/'placement_template.json',B/'randomization.json',B.parent/'initial_state.json',B.parent/'robot_alignment.py',B/'kinematics.py',B/'robot_model.json',B/'contact_policy.py',B/'run_selected.py',B/'randomize.py',B/'preflight.py',B/'certificate.py',B/'posture_policy.py',B/'collision_model.py',B/COLLISION_DIRECTORY/'collision_scene.json.gz',B/COLLISION_DIRECTORY/'gripper_calibration.json']
 files += [B.parent/INITIAL_SCENE,B.parent/'assets/bag/coffee_bag_original_green_v1.usda',B/'scene_pose.py',B/'cup_variants.py',B/'cup_variants.json']+[asset_path(name) for name in SPECS]
 return {str(p.relative_to(B.parent)):sha(p) for p in files}
def pose_id(offset,yaw_deg=0.,cup_style='cream_classic',machine_offset=(0,0,0),machine_yaw_deg=0.,bag_offset=(0,0,0),bag_yaw_deg=0.):return hashlib.sha256(json.dumps([[round(float(v),9) for v in [*offset,yaw_deg,*machine_offset,machine_yaw_deg,*bag_offset,bag_yaw_deg]],cup_style]).encode()).hexdigest()[:16]
def proposals(seed,config,count):
 rng=np.random.default_rng(seed)
 for _ in range(count):yield np.r_[rng.uniform(-np.array(config['xy_half_range_m']),np.array(config['xy_half_range_m'])),0.]

def pose_proposals(seed,config,count):
 rng=np.random.default_rng(seed)
 for _ in range(count):
  offset=np.r_[rng.uniform(-np.array(config['xy_half_range_m']),np.array(config['xy_half_range_m'])),0.]
  yield offset,float(rng.uniform(-config['yaw_half_range_deg'],config['yaw_half_range_deg']))

def large_xy(rng, half_range, minimum=0.):
 bounds=np.asarray(half_range,dtype=float)
 if minimum<0 or minimum>=np.linalg.norm(bounds):raise ValueError('Invalid minimum XY displacement')
 for _ in range(10000):
  v=rng.uniform(-bounds,bounds)
  if np.linalg.norm(v)>=minimum:return np.r_[v,0.]
 raise RuntimeError('XY proposal sampling exhausted')
def large_yaw(rng, maximum, minimum=0.):
 if not 0<=minimum<=maximum:raise ValueError('Invalid minimum yaw')
 if minimum==0:return float(rng.uniform(-maximum,maximum))
 return float(rng.choice([-1,1])*rng.uniform(minimum,maximum))

def scene_proposals(seed,config,count):
 rng=np.random.default_rng(seed)
 for _ in range(count):
  offset=large_xy(rng,config['xy_half_range_m'],config.get('xy_min_radius_m',0.))
  yaw=large_yaw(rng,config['yaw_half_range_deg'],config.get('yaw_min_abs_deg',0.))
  machine=large_xy(rng,config['machine_xy_half_range_m'],config.get('machine_xy_min_radius_m',0.))
  style=config['cup_styles'][int(rng.integers(len(config['cup_styles'])))]
  machine_yaw=large_yaw(rng,config['machine_yaw_half_range_deg'],config.get('machine_yaw_min_abs_deg',0.))
  bag=np.r_[rng.uniform(-np.array(config['bag_xy_half_range_m']),np.array(config['bag_xy_half_range_m'])),0.]
  bag_yaw=float(rng.uniform(*config['bag_yaw_range_deg']))
  yield offset,yaw,style,machine,machine_yaw,bag,bag_yaw

def retarget_tcp(tcp,rotation,pivot,offset,yaw_deg,weight):
 r=Rotation.from_euler('z',yaw_deg*weight,degrees=True).as_matrix()
 return pivot+r@(np.asarray(tcp)-pivot)+weight*offset,r@rotation

def retarget(template,offset,seed,config,yaw_deg=0.,cup_style='cream_classic',machine_offset=(0,0,0),machine_yaw_deg=0.,bag_offset=(0,0,0),bag_yaw_deg=0.):
 offset=np.asarray(offset,dtype=float)
 if offset.shape!=(3,) or not np.isfinite(offset).all() or abs(offset[2])>1e-12 or np.any(abs(offset[:2])>np.array(config['xy_half_range_m'])+1e-12):raise ValueError('Offset must be finite XY within configured proposal limits; Z stays fixed')
 if not np.isfinite(yaw_deg) or abs(yaw_deg)>config.get('yaw_half_range_deg',0.)+1e-12:raise ValueError('Yaw must be finite and within configured limits')
 machine_offset=np.asarray(machine_offset,dtype=float);spec(cup_style)
 if machine_offset.shape!=(3,) or not np.isfinite(machine_offset).all() or abs(machine_offset[2])>1e-12 or np.any(abs(machine_offset[:2])>np.array(config.get('machine_xy_half_range_m',[0,0]))+1e-12):raise ValueError('Machine XY offset outside configured bounds')
 if not np.isfinite(machine_yaw_deg) or abs(machine_yaw_deg)>config.get('machine_yaw_half_range_deg',0.)+1e-12:raise ValueError('Machine yaw outside configured bounds')
 bag_offset=np.asarray(bag_offset,dtype=float)
 if bag_offset.shape!=(3,) or not np.isfinite(bag_offset).all() or abs(bag_offset[2])>1e-12 or np.any(abs(bag_offset[:2])>np.array(config.get('bag_xy_half_range_m',[0,0]))+1e-12):raise ValueError('Bag XY offset outside configured bounds')
 yaw_min,yaw_max=config.get('bag_yaw_range_deg',[0,0])
 if not np.isfinite(bag_yaw_deg) or not yaw_min-1e-12<=bag_yaw_deg<=yaw_max+1e-12:raise ValueError('Bag yaw outside configured bounds')
 plan=copy.deepcopy(template);plan.update(bag_offset_m=bag_offset.tolist(),bag_yaw_deg=float(bag_yaw_deg));plan['machine_yaw_deg']=float(machine_yaw_deg);plan['cup_style']=cup_style;plan['machine_offset_m']=machine_offset.tolist();m,c=create_model(plan)
 nominal_cup=c.initial_cup.copy();nominal_grasp=c.grasp_tcp.copy()
 handle_local=nominal_cup[:3,:3].T@(np.array(template['geometry']['handle_center'])-nominal_cup[:3,3]);grasp_shift_local=handle_shift(cup_style,handle_local)
 delta=machine_delta(machine_offset,machine_yaw_deg)
 for key in ('cup_target','outlet'):plan['geometry'][key]=(delta[:3,:3]@np.array(plan['geometry'][key])+delta[:3,3]).tolist()
 for key in ('front','machine_front'):
  if key in plan['geometry']:plan['geometry'][key]=(delta[:3,:3]@np.array(plan['geometry'][key])).tolist()
 yaw=Rotation.from_euler('z',yaw_deg,degrees=True).as_matrix()
 c.initial_cup[:3,3]+=offset;c.initial_cup[:3,:3]=yaw@nominal_cup[:3,:3];c.goal=np.array(plan['geometry']['cup_target'])
 initial=plan['initial_targets'];m.defaults.update(initial)
 c.set_state(plan['initial'],initial['idx01_body_joint1'],initial['idx02_body_joint2'],initial['idx81_gripper_r_outer_joint1'])
 table=next(i for i,s in enumerate(c.shapes) if s['path']=='/World/Background/Shell/CounterTop_17')
 cup=[i for i,s in enumerate(c.shapes) if s['body']=='/World/CoffeeTaskPreview/Cup']
 margin=config['table_edge_margin_m']
 if np.any(c.low[cup,:2].min(0)<c.low[table,:2]+margin) or np.any(c.high[cup,:2].max(0)>c.high[table,:2]-margin):raise ValueError('Entire cup footprint must remain inside table margin')
 layout=c.check_machine_layout(margin)+c.check_bag_layout(margin)
 if layout:raise ValueError('Machine layout collision: '+str(layout[:2]))
 _,collisions=c.check('initial')
 if collisions:raise ValueError('Initial scene collision: '+str(collisions[:2]))
 plan['geometry']['cup_origin']=c.initial_cup[:3,3].tolist()
 plan['geometry']['handle_center']=(nominal_cup[:3,3]+offset+yaw@(np.array(template['geometry']['handle_center'])-nominal_cup[:3,3]+nominal_cup[:3,:3]@grasp_shift_local)).tolist()
 plan['geometry']['cup_rotation']=c.initial_cup[:3,:3].tolist()
 plan['geometry']['cup_target_rotation']=(delta[:3,:3]@nominal_cup[:3,:3]).tolist()
 plan['randomization']=dict(seed=int(seed),offset_m=offset.tolist(),cup_xyz=plan['geometry']['cup_origin'],yaw_delta_deg=float(yaw_deg),pose_id=pose_id(offset,yaw_deg,cup_style,machine_offset,machine_yaw_deg,bag_offset,bag_yaw_deg),cup_style=cup_style,machine_offset_m=machine_offset.tolist(),machine_yaw_deg=float(machine_yaw_deg),bag_offset_m=bag_offset.tolist(),bag_yaw_deg=float(bag_yaw_deg),height_policy=config['z_mode'])
 plan['cup_start_xyz']=plan['geometry']['cup_origin'];plan['cup_start_yaw_deg']=float(yaw_deg)
 from head_visibility import require
 require(plan)
 counts={phase:sum(r['phase']==phase for r in template['steps']) for phase in {r['phase'] for r in template['steps']}}
 progress={};rows=[];correction=np.zeros(7)
 for row0 in template['steps']:
  row=dict(row0);row['bodylift']=initial['idx01_body_joint1'];row['waist']=initial['idx02_body_joint2'];phase=row['phase'];progress[phase]=progress.get(phase,0)+1;u=progress[phase]/counts[phase]
  if phase=='move_above_handle':weight=10*u**3-15*u**4+6*u**5
  elif phase in {'lower_pregrasp','approach_handle','close','lift','lift_check'}:weight=1.
  elif phase=='transfer_to_front':weight=1-(10*u**3-15*u**4+6*u**5)
  else:weight=0.
  nominal_tcp=np.eye(4);nominal_tcp[:3,:3]=row['rotation'];nominal_tcp[:3,3]=row['tcp']
  pivot=(nominal_tcp@np.linalg.inv(nominal_grasp)@nominal_cup)[:3,3] if phase in {'lift','lift_check','transfer_to_front'} else nominal_cup[:3,3]
  nominal_cup_rotation=(nominal_tcp@np.linalg.inv(nominal_grasp)@nominal_cup)[:3,:3] if phase in {'lift','lift_check','transfer_to_front','insert','lower_to_tray','release','withdraw','settle'} else nominal_cup[:3,:3]
  style_weight=10*u**3-15*u**4+6*u**5 if phase=='move_above_handle' else (0. if phase=='ready' else 1.)
  style_tcp=np.array(row['tcp'])+style_weight*(nominal_cup_rotation@grasp_shift_local)
  target,rot=retarget_tcp(style_tcp,np.array(row['rotation']),pivot,offset,yaw_deg,weight)
  machine_weight=1-weight if phase=='transfer_to_front' else (1. if phase in {'insert','lower_to_tray','release','withdraw','settle'} else 0.)
  md=machine_delta(machine_offset,machine_yaw_deg,machine_weight)
  target=md[:3,:3]@target+md[:3,3];rot=md[:3,:3]@rot;nominal=np.array(row['q'])
  m.defaults.update(idx01_body_joint1=row['bodylift'],idx02_body_joint2=row['waist'])
  if np.linalg.norm(m.tcp(nominal)[:3,3]-target)+np.linalg.norm(Rotation.from_matrix(rot@m.tcp(nominal)[:3,:3].T).as_rotvec())>1e-10:
   start=np.clip(nominal+correction,m.lo+1e-7,m.hi-1e-7)
   def residual(q):
    t=m.tcp(q);return np.r_[(t[:3,3]-target)*15,Rotation.from_matrix(rot@t[:3,:3].T).as_rotvec(),(q-start)*.0002]
   opt=least_squares(residual,start,bounds=(m.lo,m.hi),max_nfev=45);q=opt.x;t=m.tcp(q)
   pe=np.linalg.norm(t[:3,3]-target);re=np.linalg.norm(Rotation.from_matrix(rot@t[:3,:3].T).as_rotvec())
   if pe>.0003 or re>.003:raise ValueError(f'Grasp retarget unreachable: {phase}, position {pe}, rotation {re}')
  else:q=nominal;t=m.tcp(q)
  correction=q-nominal;row.update(q=q.tolist(),tcp=t[:3,3].tolist(),rotation=t[:3,:3].tolist())
  if rows:
   old=rows[-1];n=max(1,int(np.ceil(np.max(abs(q-old['q']))/.025)))
   for f in np.linspace(0,1,n+1)[1:-1]:
    r=interpolate(old,row,f);m.defaults.update(idx01_body_joint1=r['bodylift'],idx02_body_joint2=r['waist']);tf=m.tcp(r['q']);r.update(tcp=tf[:3,3].tolist(),rotation=tf[:3,:3].tolist());rows.append(r)
  rows.append(row)
 plan['steps']=rows
 from posture_policy import validate_fixed_posture
 validate_fixed_posture(plan)
 return plan

def valid_entries(bank,ctx):
 entries=[]
 for entry in bank.get('entries',[]):
  if entry.get('context')!=ctx or entry.get('status','validated')!='validated':continue
  evidence=Path(entry['result_path'])
  if not evidence.is_file() or sha(evidence)!=entry['result_sha256']:continue
  result=json.loads(evidence.read_text())
  if result.get('passed') and not result.get('continued_from') and result.get('randomization',{}).get('pose_id')==entry['pose_id']:entries.append(entry)
 return sorted(entries,key=lambda e:e['pose_id'])

def promote(bank_path,plan,result_path,ctx):
 result=json.loads(Path(result_path).read_text())
 if not result.get('passed') or result.get('continued_from'):raise ValueError('Only an uninterrupted physically successful placement may enter the validated pool')
 if result.get('randomization')!=plan['randomization']:raise ValueError('Result does not belong to this sampled start')
 bank=json.loads(bank_path.read_text()) if bank_path.exists() else {'version':1,'entries':[]}
 r=plan['randomization'];entry=dict(status='validated',pose_id=r['pose_id'],offset_m=r['offset_m'],cup_xyz=r['cup_xyz'],yaw_delta_deg=r.get('yaw_delta_deg',0.),cup_style=r.get('cup_style','cream_classic'),machine_offset_m=r.get('machine_offset_m',[0,0,0]),machine_yaw_deg=r.get('machine_yaw_deg',0.),bag_offset_m=r.get('bag_offset_m',[0,0,0]),bag_yaw_deg=r.get('bag_yaw_deg',0.),context=ctx,result_path=str(Path(result_path).resolve()),result_sha256=sha(result_path))
 bank['entries']=[e for e in bank['entries'] if not(e['pose_id']==entry['pose_id'] and e['context']==ctx)]+[entry]
 tmp=bank_path.with_suffix('.tmp');tmp.write_text(json.dumps(bank,indent=2));tmp.replace(bank_path)
 return entry


def record_failure(bank_path,plan,result_path,ctx):
    """A later physical failure removes this start from the active pool."""
    if not bank_path.exists():return
    bank=json.loads(bank_path.read_text())
    for entry in bank['entries']:
        if entry['pose_id']==plan['randomization']['pose_id'] and entry['context']==ctx:
            entry.update(status='quarantined',failed_result_path=str(Path(result_path).resolve()),failed_result_sha256=sha(result_path))
    tmp=bank_path.with_suffix('.tmp');tmp.write_text(json.dumps(bank,indent=2));tmp.replace(bank_path)
