"""Offline FK feasibility inventory. No IK, joint clipping, or source pose mirroring."""
import argparse
import json
from copy import deepcopy
from pathlib import Path
import numpy as np
from ..common import HERE,owned,write_json,audit_reset,digest
from ..frozen_initials_v1.bank import load
from ..frozen_initials_v1.cup import context,base_matrix,hand_geometry
from ..procedural_grasp import Kinematics,complete_closed_reset


def initialization(c,row):
    # Preserve recorded arm/body/head states, but use the current native linkage
    # reset for the inactive gripper. No missing passive states are fabricated.
    state=row['measured_state'];names=list(state)[:18]
    issued=dict(positions=dict(c['ready']['targets_rad']),hold_targets=dict(c['ready']['targets_rad']))
    issued['positions'].update({n:state[n] for n in names})
    issued['hold_targets'].update({n:state[n] for n in names})
    result,receipt=complete_closed_reset(c['stage'],issued,row['active_hand'])
    audit_reset(c['stage'],result)
    return result,receipt


def inspect(base_key,inventory,output):
    from short_tasks.agent_probe_v1.native_owner import initialize_base_pose
    out=owned(output,fresh=True);out.mkdir(parents=True);c=context(base_key,out)
    bank=load(HERE/'frozen_initials_v1/configurations/bank_v1.json')
    templates={p['active_hand']:p for p in bank['cases']['table_waste_empty_grasp']['profiles']}
    data=json.loads(Path(inventory).read_text());rows=[]
    for row in data['candidates']:
        hand=row['active_hand'];p=templates[hand];base=c['delta'].T@np.asarray(p['base_canonical_column'])
        yaw=float(np.arctan2(base[1,0],base[0,0]));initialize_base_pose(c['stage'],[*base[:2,3],yaw])
        try:
            reset,closure=initialization(c,row);model=Kinematics(c['stage'],hand,reset['positions'])
            tips,frames=hand_geometry(c['stage'],model,reset['positions'])
            z=[float(tips[:,2].min()),float(tips[:,2].max())];cupz=[float(c['cup'][:,2].min()),float(c['cup'][:,2].max())]
            separation=max(0.,z[0]-cupz[1],cupz[0]-z[1])
            q=[reset['positions'][n] for n in model.names]
            rows.append(dict(id=row['id'],active_hand=hand,episode_index=row['episode_index'],frame=row['frame'],
                benchmark_task_id=row['benchmark_task_id'],fingertip_z_m=z,cup_z_m=cupz,
                vertical_envelope_separation_m=separation,arm_q_rad=q,tool_world=frames[model.tool].tolist(),
                reset_accepted=True,source_row=row['source_row'],robot_initialization=reset,gripper_closure=closure,
                base_canonical_column=p['base_canonical_column'],supported_miss_signs=p['supported_miss_signs']))
        except ValueError as exc:rows.append(dict(id=row['id'],reset_accepted=False,error=str(exc)))
    write_json(out/'inspection.json',dict(inventory=str(inventory),inventory_sha256=digest(inventory),rows=rows,
        online_ik_calls=0,source_tree='full_only',physical_tested=False))
    print(json.dumps(dict(rows=len(rows),reset_accepted=sum(r['reset_accepted'] for r in rows),
        vertical_overlap_by_hand={h:sum(r.get('reset_accepted') and r.get('active_hand')==h and r['vertical_envelope_separation_m']==0 for r in rows) for h in ('l','r')})))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base-key',required=True);p.add_argument('--inventory',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();inspect(a.base_key,a.inventory,a.output)
