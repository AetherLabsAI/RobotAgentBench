"""Explain source-grid incompatibility; do not alter poses or acceptance."""
import json
from collections import Counter
import numpy as np
from ..common import HERE,owned,write_json
from .geometry import Gate,existing


if __name__=='__main__':
    from ..frozen_initials_v1.cup import base_matrix,hand_geometry
    from ..expansion_v2.empty_grasp import envelope_distance
    out=owned(HERE/'frozen_initials_v3/verification/sofa_grid_diagnostic_v1',fresh=True);out.mkdir(parents=True)
    stage,record,_=existing('sofa');gate=Gate(stage,record)
    inventory=json.loads((HERE/'frozen_initials_v3/mine/sofa_training58_v1/inventory.json').read_text())
    counts=Counter();matches=[];closest=None
    for row in inventory['candidates']:
        initial,_=gate.source_initial(row);hand=row['active_hand'];model=gate.model(hand,initial['positions'],gate.base)
        tips,_=hand_geometry(stage,model,initial['positions']);local=(np.c_[tips,np.ones(len(tips))]@np.linalg.inv(gate.base).T)[:,:3]
        for station in gate.bases:
            counts['station_pairs']+=1;base=base_matrix(station['initial_base_xyyaw']);basis=base[:3,:3]
            moved=(np.c_[local,np.ones(len(local))]@base.T)[:,:3]
            center=(gate.points.min(0)+gate.points.max(0))/2;side=(np.linalg.inv(base)@np.r_[center,1.])[1]
            if ('l' if side>0 else 'r')!=hand:counts['wrong_hand_side']+=1;continue
            gaps=[float((moved@(basis[:,1]*s)).min()-(gate.points@(basis[:,1]*s)).max()) for s in (-1,1)]
            distance=envelope_distance(gate.points,moved,basis)
            if closest is None or distance<closest['distance_m']:closest=dict(profile=row['id'],distance_m=distance,gaps=gaps,station=station)
            if not any(.02<=g<=.05 for g in gaps):counts['lateral_gap_rejected']+=1;continue
            if not .02<=distance<=.05:counts['3d_gap_rejected']+=1;continue
            result=gate.check(initial,hand,base=base);counts[result['reason'] or 'accepted']+=1
            if len(matches)<20:matches.append(dict(profile=row['id'],station=station,result=result))
    value=dict(grid_candidates=len(gate.bases),station_domain=gate.station_domain,counts=dict(counts),closest=closest,matches=matches,
        task_definition_changed=False,original_geometry_rules_unchanged=True)
    write_json(out/'diagnostic.json',value);print(json.dumps(value),flush=True)
