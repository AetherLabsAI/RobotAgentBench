"""Read-only CPU replay of the original protected-pair guard, not physics."""
import json
import math
from pathlib import Path
import numpy as np
from ..common import HERE,owned,write_json,digest
from ..frozen_initials_v1.bank import load
from ..frozen_initials_v1.cup import context


def main():
    from short_tasks.agent_probe_v1.native_owner import initialize_base_pose
    from short_tasks.agent_probe_v1.collision_guard.stage_model import from_stage
    from short_tasks.agent_probe_v1.collision_guard.reactive_gate import ReactiveGate
    from ..poses import world
    out=owned(HERE/'frozen_initials_v2/verification/protected_pairs_v1',fresh=True);out.mkdir(parents=True)
    c=context('tipped_cup_base401_r3',out);stage=c['stage'];stage.SetEditTarget(stage.GetSessionLayer())
    bankpath=HERE/'frozen_initials_v2/compile/cup_training_adapt_v5/bank_v2.json';bank=load(bankpath)
    gate=ReactiveGate(from_stage(stage));rows=[]
    for p in bank['cases']['table_waste_empty_grasp']['profiles']:
        base=c['delta'].T@np.asarray(p['base_canonical_column'])
        initialize_base_pose(stage,[float(base[0,3]),float(base[1,3]),math.atan2(base[1,0],base[0,0])])
        links=gate.model.fk(p['robot_initialization']['positions'],world(stage,'/World/Robot/base_link'))
        scan=gate.scan(links);rows.append(dict(profile=p['id'],hand=p['active_hand'],scan=scan))
    report=dict(bank_sha256=digest(bankpath),rows=rows,profiles=len(rows),
        no_protected_overlap=sum(not r['scan']['overlap'] for r in rows),
        protected_overlap=sum(r['scan']['overlap'] for r in rows),
        native_physics_tested=False,acceptance_rules_changed=False,immutable_bank_modified=False)
    write_json(out/'report.json',report);print(json.dumps({k:v for k,v in report.items() if k!='rows'}))


if __name__=='__main__':main()
