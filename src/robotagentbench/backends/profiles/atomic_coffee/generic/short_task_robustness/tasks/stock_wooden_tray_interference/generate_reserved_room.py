"""Task-owned full-room CPU generation; native validation is a separate step."""
import argparse
from copy import deepcopy
import importlib
import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
SCENE=ROOT/'openend/tasks/scene_requirements'
sys.path[:0]=[str(ROOT),str(SCENE)]


def generate(request, output, robot_route_reference=None):
    from short_tasks.scene import verify_scene_sources
    from short_tasks.tasks.stock_wooden_tray_interference.generator_integrity import verify_reviewed_generators
    from short_tasks.tasks.stock_wooden_tray_interference.adapter import resolve
    from short_tasks.tasks.stock_wooden_tray_interference.generation import policy_for_plan,validate_output
    from short_tasks.tasks.stock_wooden_tray_interference.prepare_scene import prepare_reserved
    from pxr import Usd
    plan=resolve(request)
    output=Path(output).resolve()
    validate_output(output/'placement.json',ROOT)
    if output.exists():raise ValueError('Episode output already exists')
    shared_integrity=verify_scene_sources()  # Never refresh or disable shared pins.
    generator_integrity=verify_reviewed_generators(SCENE)
    resolved=importlib.import_module('room_request').resolve(plan['scene_request'])
    layouts=json.loads((SCENE/'placement_sources/random-layouts-v1.json').read_text())['layouts']
    output.mkdir(parents=True)
    attempts=[]
    for index,layout_id in enumerate(resolved['layout_candidates']):
        data=deepcopy(resolved)
        data['layout']=next(row for row in layouts if row['id']==layout_id)
        data['request']['layout_id']=layout_id
        data['requirements']=importlib.import_module('planner_v2').compile_request(data['request'])
        data['layout_selection'].update(selected=layout_id,rerolls=index)
        request_path=output/'candidate.json';request_path.write_text(json.dumps(data,indent=2)+'\n')
        result=subprocess.run(['node',str(SCENE/'room_plan.cjs'),str(request_path)],
                              capture_output=True,text=True,timeout=30)
        if result.returncode not in (0,2):raise RuntimeError('Room planner failed: '+result.stderr)
        data=json.loads(result.stdout)
        attempts.append(dict(layout_id=layout_id,status=data['plan']['status']))
        if data['plan']['status']=='planar_candidate':break
    else:raise ValueError('Existing seeded layout candidate list exhausted')
    data['task_scene_policy']=policy_for_plan(plan)
    if robot_route_reference is not None:
        reference=Path(robot_route_reference).resolve()
        if not reference.is_relative_to(HERE) or not reference.is_file():
            raise ValueError('Task-owned robot route reference required')
        data['task_scene_policy']['robot_route_reference']=str(reference)
    placement=output/'placement.json';placement.write_text(json.dumps(data,indent=2)+'\n')
    with (output/'build.log').open('w') as log:
        subprocess.run([sys.executable,'-B',str(SCENE/'build_selected_room.py'),str(placement)],
                       stdout=log,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
    verify_scene_sources()
    verify_reviewed_generators(SCENE)
    manifest_path=output/'composition.json';manifest=json.loads(manifest_path.read_text())
    stage=Usd.Stage.Open(manifest['scene'])
    frozen=deepcopy(manifest['environment_reservations'])
    if not frozen or any('source_product_range' not in r for r in frozen):
        raise ValueError('New rooms require source product range evidence')
    prepared=prepare_reserved(plan,stage,manifest)
    if prepared['environment_reservations']!=frozen:raise ValueError('Frozen route changed during preparation')
    if prepared['interference_background_exclusion']['excluded_paths']:
        raise ValueError('Reserve-first generation unexpectedly retained duplicate target stock')
    prepared['public_short_task']=dict(task_name=plan['task_name'],rule_version=plan['rule_version'],
        instruction=plan['instruction'],native_available=False)
    manifest_path.write_text(json.dumps(prepared,indent=2)+'\n')
    (output/'task_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    receipt=dict(scene_generated=True,native_physics_validated=False,gpu_started=False,
        task_name=plan['task_name'],attempts=attempts,reservation_frozen=True,
        runtime_available=False,manifest=str(manifest_path),source_product_range=frozen[0]['source_product_range'],
        shared_integrity=shared_integrity,generator_integrity=generator_integrity,
        relocation_distance=frozen[0]['relocation_distance'])
    (output/'generation_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--request',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--robot-route-reference',type=Path)
    args=ap.parse_args()
    print(json.dumps(generate(json.loads(args.request.read_text()),args.output,args.robot_route_reference),indent=2))


if __name__=='__main__':main()
