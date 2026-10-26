import json,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from multi_task_room_v2 import validate_request,FACTORIES
HERE=Path(__file__).resolve().parent
request=json.loads((ROOT/'multi_task_episode_task9_v1.json').read_text());validate_request(request)
assert len(request['tasks'])==4
deployment=json.loads((HERE/'deployment_receipt.json').read_text())
for p,h in deployment['files'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h
bindings=json.loads((HERE/'binding_check.json').read_text());assert bindings['status']=='passed' and len(bindings['bindings'])==6
for path in deployment['manifests']:
    d=json.loads(Path(path).read_text())
    for asset in ('refrigerator','beef_box','leftover_box','salad_box','microwave_support_plate','microwave_plate_stand'):
        assert sum(o['id']==asset for o in d['objects'])==1
tests={}
for log,n in [('rule_tests.log',68),('box_tests.log',8)]:
    text=(HERE/log).read_text();assert f'Ran {n} tests' in text and text.rstrip().endswith('OK');tests[log]=n
native={}
for name in ('native.QgwRzD','native.NLNCgS'):
    p=HERE/name/'result.json';r=json.loads(p.read_text())
    assert r['status']=='passed' and r['source_layers_unchanged']
    assert r['selected_controls']==20 and r['score']['score']==0 and not r['score']['success']
    native[r['score']['payload_variant']]=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
        physics_hz=60,control_hz=20,selected_controls=20,simulation_time_s=r['simulation_time_s'],
        native_evidence=True,selection_pose_source=r['selection_pose_source'])
assert set(native)=={'beef_box','leftover_box'}
receipt=dict(status='connected_to_unified_room_host',registration_scope=deployment['scope'],
    factories=sorted(FACTORIES),request=str(ROOT/'multi_task_episode_task9_v1.json'),tests=tests,
    cpu_geometry_bindings=6,native=native,phase_handoff='rule_and_host_contract_tested_not_native_action_verified',
    full_trajectory_verified=False,all_four_tasks_native_rollout_verified=False,
    files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
           for p in sorted(HERE.glob('*.py'))})
(HERE/'connection_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
