"""One-time config extraction: no object layouts in the runtime posture bank."""
from copy import deepcopy
import json
from ..common import HERE,digest,write_json
from ..catalog import generator

if __name__=='__main__':
    cfg=HERE/'procedural_config.json'
    source=HERE/'frozen_initials_v3/banks/remaining_exact_training_v1/bank.json'
    bank=json.loads(source.read_text());layouts=json.loads(cfg.read_text())['shelf']['layouts'];profiles=[]
    for row in bank['cases']['shelf_drink']['profiles']:
        index=generator('shelf_drink',row['fit']['source_seed'],'source-layout').randrange(len(layouts))
        profiles.append(dict(id=row['id'],active_hand=row['active_hand'],
            robot_initialization=deepcopy(row['robot_initialization']),gripper_closure=deepcopy(row['gripper_closure']),
            base_reference=deepcopy(layouts[index]['robot']),training_provenance=deepcopy(row['training_provenance'])))
    path=HERE/'shelf_continuous_v2/robot_bank.json'
    if path.exists():raise ValueError('Do not overwrite frozen robot bank')
    write_json(path,dict(schema='failure_recovery.robot_only_bank.v2',cases=dict(shelf_drink=dict(profiles=profiles)),
        offline_provenance_sha256=dict(old_joint_bank=digest(source),old_base_config=digest(cfg)),
        contains_object_positions=False,contains_layouts=False,online_ik=False))
    print(json.dumps(dict(profiles=len(profiles),runtime_object_table=False)))
