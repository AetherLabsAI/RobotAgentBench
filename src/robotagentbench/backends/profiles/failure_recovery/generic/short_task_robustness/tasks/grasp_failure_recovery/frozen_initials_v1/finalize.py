"""Resume export of already solved poses; no solver and no repeated IK."""
import json
from pathlib import Path
from ..common import HERE,owned,write_json
from .bank import SCHEMA,CASES,checksum,validate
from .compile import INPUTS,extract
from .cup import base_matrix


def main():
    import numpy as np
    from pxr import Usd
    from short_tasks.tasks.clear_table_trash.initial_station import table_context
    result=dict(schema=SCHEMA,cases={name:dict(profiles=[],online_ik_fallback=False) for name in CASES},
        source_tree='full_only',runtime_joint_interpolation=False,scope='frozen_initials_not_new_scoring')
    for variant,(source,report) in INPUTS.items():result['cases'][variant]['profiles'].append(extract(variant,source,report))
    for profile,source,sign in (
        (result['cases']['table_waste_empty_grasp']['profiles'][0],INPUTS['table_waste_empty_grasp'][0],1),
        (extract('table_waste_empty_grasp','frozen_initials_v1/offline/cup_left_v1/initial.json',None),
            'frozen_initials_v1/offline/cup_left_v1/initial.json',-1)):
        record=json.loads((HERE/source).read_text());manifest=json.loads(Path(record['original_launch']['manifest']).read_text())
        delta,_,_=table_context(Usd.Stage.Open(record['scene']),manifest)
        profile.update(base_canonical_column=(np.linalg.inv(delta.T)@base_matrix(record['initial_base_xyyaw'])).tolist(),
            supported_miss_signs=[sign],asset_id='paper_cup',anchor='canonical_actual_coffee_table_geometry')
        if sign==-1:result['cases']['table_waste_empty_grasp']['profiles'].append(profile)
    result['cases']['table_waste_tipped_cup'].update(status='blocked_missing_native_supported_rest_and_failure_history',
        candidate='expansion_v2/candidates/tipped_side401_cpu_r3/initial.json')
    for name,case in result['cases'].items():
        case['hand_coverage']=sorted({p['active_hand'] for p in case['profiles']})
        case['fast_scene_generator_connected']=name=='table_waste_empty_grasp'
    result['content_sha256']=checksum(result);validate(result)
    target=owned(HERE/'frozen_initials_v1/configurations/bank_v1.json',fresh=True)
    write_json(target,result)
    print(json.dumps({name:dict(profiles=len(c['profiles']),hands=c['hand_coverage']) for name,c in result['cases'].items()}))


if __name__=='__main__':main()
