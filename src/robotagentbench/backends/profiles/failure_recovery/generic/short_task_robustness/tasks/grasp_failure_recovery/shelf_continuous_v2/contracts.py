"""CPU registration and sampler-regression audit in the full source container."""
import argparse
import ast
import json
from pathlib import Path
from ..common import HERE,CURRENT,owned,digest,write_json
from .registration import registration_for

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    output=owned(a.output,fresh=True);output.parent.mkdir(parents=True,exist_ok=True)
    rows=[]
    for path in sorted((HERE/'shelf_continuous_v2/initials').glob('*v3/initial.json')):
        registration,receipt=registration_for(path)
        if not receipt['inherited_fields_unchanged']:raise ValueError('Source registration changed')
        rows.append(dict(initial=path.parent.name,source_api=registration.task_id,
            instruction=registration.instruction,source_budgets_and_execution_unchanged=True))
    original=CURRENT/'openend/tasks/scene_requirements/shelf_sampler.py'
    copied=HERE/'shelf_continuous_v2/packing.py'
    def function(path):return next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='sample_scene')
    same=ast.dump(function(original),include_attributes=False)==ast.dump(function(copied),include_attributes=False)
    if not same:raise ValueError('Source packing algorithm or budgets changed')
    write_json(output,dict(registrations=rows,all_original_fields_unchanged=True,packing_function_ast_identical=True,
        source_hashes={str(original):digest(original),str(copied):digest(copied)},
        global_registration_written=False,model_calls=0,native_physics_test=False))
    print(json.dumps(dict(registrations_passed=len(rows),packing_unchanged=same)))
