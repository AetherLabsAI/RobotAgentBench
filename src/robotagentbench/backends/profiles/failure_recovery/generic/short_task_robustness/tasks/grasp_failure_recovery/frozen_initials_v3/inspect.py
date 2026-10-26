"""Offline exact-row reset checks, including original protected self-collision."""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
from ..common import HERE,owned,write_json,digest,audit_reset

SOURCES={
    'shelf_drink':'procedural_v1/shelf_adopt303_cpu/initial/initial.json',
    'coffee_bag':'procedural_v1/bag_adopt301_cpu/initial/initial.json',
    'sofa':'expansion_v2/candidates/sofa_empty401_operating_v8/initial.json',
}


def reset(stage,ready,row):
    from ..procedural_grasp import complete_closed_reset
    issued=deepcopy(ready);state=row['measured_state']
    names=[n for n in state if '_arm_' in n or '_body_' in n or '_head_' in n]
    if len(names)!=18:raise ValueError('Exact 18 source arm/body/head channels required')
    for name in names:issued['positions'][name]=state[name];issued['hold_targets'][name]=state[name]
    return complete_closed_reset(stage,issued,row['active_hand'])


def inspect(variant,inventory,output):
    from pxr import Usd,UsdGeom
    from ..poses import world
    from ..procedural_grasp import Kinematics,collider_points
    from ..frozen_initials_v1.cup import hand_geometry
    from ..expansion_v2.empty_grasp import envelope_distance
    from ..expansion_v2.camera_projection_complete import project_target
    from ..shelf_checkpoint import Clearance
    from short_tasks.agent_probe_v1.collision_guard.stage_model import from_stage
    from short_tasks.agent_probe_v1.collision_guard.reactive_gate import ReactiveGate
    from ..expansion_v2.operating_station import RoomGate
    from short_tasks.tasks.robot_pose_alignment_v1.geometry import checker
    inventory=owned(inventory);data=json.loads(inventory.read_text())
    out=owned(output,fresh=True);out.mkdir(parents=True)
    source=HERE/SOURCES[variant];record=json.loads(source.read_text())
    stage=Usd.Stage.Open(record['scene']);stage.SetEditTarget(stage.GetSessionLayer())
    target=record['target'];points=np.concatenate([r[3] for r in collider_points(stage,target)])
    base=world(stage,'/World/Robot/base_link');basis=base[:3,:3]
    center=(points.min(0)+points.max(0))/2;target_local=np.linalg.inv(base)@np.r_[center,1.]
    side='l' if target_local[1]>0 else 'r'
    protected=ReactiveGate(from_stage(stage));models={};clearances={};room=None
    if variant=='shelf_drink':obstacles=record['neighbors']
    elif variant=='coffee_bag':
        from evaluation.tasks.coffee_machine.specs import MACHINE,CUP,COUNTER
        obstacles=[MACHINE,CUP,COUNTER]
    else:obstacles=[];room=RoomGate(stage,record['original_launch'] and json.loads(Path(record['original_launch']['manifest']).read_text()))
    rows=[];counts=Counter()
    for row in data['candidates']:
        hand=row['active_hand'];reason=None;result=dict(id=row['id'],episode_index=row['episode_index'],frame=row['frame'],hand=hand)
        try:
            initial,closure=reset(stage,record['robot_initialization'],row);audit_reset(stage,initial)
            model=models.setdefault(hand,Kinematics(stage,hand,initial['positions']))
            links=protected.model.fk(initial['positions'],base);scan=protected.scan(links)
            if scan['overlap']:reason='protected_self_overlap'
            tips,frames=hand_geometry(stage,model,initial['positions'])
            distance=envelope_distance(points,tips,basis)
            gaps={str(sign):float((tips@(basis[:,1]*sign)).min()-(points@(basis[:,1]*sign)).max()) for sign in (-1,1)}
            signs=[s for s in (-1,1) if .02<=gaps[str(s)]<=.05]
            result.update(conservative_3d_envelope_distance_m=distance,signed_lateral_gaps_m=gaps,
                fingertip_z_m=[float(tips[:,2].min()),float(tips[:,2].max())],
                target_z_m=[float(points[:,2].min()),float(points[:,2].max())],supported_miss_signs=signs,
                protected_overlap=bool(scan['overlap']),protected_contacts=[c['bodies'] for c in scan.get('contacts',[])])
            if reason is None and hand!=side:reason='target_side_hand_mismatch'
            if reason is None and (not signs or not .02<=distance<=.05):reason='fixed_target_gap_not_2_5cm'
            if reason is None:
                projections=project_target(stage,frames,target);result['original_camera_projection']=projections
                if not any(x['whole_target_envelope_inside_original_frustum'] for x in projections.values()):reason='target_not_wholly_in_original_camera'
            if reason is None:
                if variant=='sofa':
                    manifest=json.loads(Path(record['original_launch']['manifest']).read_text())
                    reason=room.check(model,initial['positions'])
                    if reason is None:reason=checker(stage,manifest,record['initial_base_xyyaw'])({'targets_rad':initial['positions']})
                else:
                    if hand not in clearances:clearances[hand]=Clearance(stage,target,obstacles,hand)
                    result['clearance']=clearances[hand].evaluate(initial['positions'])
                    if result['clearance']['overlaps']:reason='original_collider_clearance_rejected'
            result.update(accepted_cpu=reason is None,reason=reason,active_arm_copied_exactly=True,
                robot_initialization=initial,gripper_closure=closure,source_row=row['source_row'])
        except ValueError as exc:result.update(accepted_cpu=False,reason='native_reset_rejected: '+str(exc))
        rows.append(result);counts['accepted' if result['accepted_cpu'] else result['reason']]+=1
        if len(rows)%50==0:
            write_json(out/'progress.json',dict(examined=len(rows),counts=dict(counts)));print(variant,len(rows),dict(counts),flush=True)
    write_json(out/'inspection.json',dict(variant=variant,source_initial=str(source),source_initial_sha256=digest(source),
        inventory=str(inventory),inventory_sha256=digest(inventory),rows=rows,counts=dict(counts),
        target_unchanged=True,robot_station_unchanged=True,online_ik_calls=0,native_physics_tested=False,
        protected_source_shape_and_pair_rules_unchanged=True))
    print(json.dumps(dict(variant=variant,counts=dict(counts))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=tuple(SOURCES),required=True)
    p.add_argument('--inventory',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    inspect(a.variant,a.inventory,a.output)
