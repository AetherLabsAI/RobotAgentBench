"""Evidence-only audit; never promote a CPU scene to native acceptance."""
from collections import Counter
import hashlib
import json
from pathlib import Path
from .. import sim2_runner as io

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def translated(path):
    path=Path(path)
    if path.is_relative_to(io.GUEST):return io.HOST/path.relative_to(io.GUEST)
    if path.is_relative_to(Path(io.CURRENT)):
        return io.PROJECT/'short_tasks/agent_probe_v1/deployment_generic_20261010/source'/path.relative_to(Path(io.CURRENT))
    raise ValueError('Unexpected source path: '+str(path))

if __name__=='__main__':
    root=io.HOST/'shelf_continuous_v2';bank=json.loads((root/'robot_bank.json').read_text())
    profiles={p['id']:p for p in bank['cases']['shelf_drink']['profiles']};rows=[]
    for p in sorted((root/'initials').glob('*v3/initial.json')):
        d=json.loads(p.read_text());snapshot=json.loads((p.parent/'source_snapshot.json').read_text())
        q=d['robot_initialization'];checks=dict(joints_unchanged=q==profiles[d['frozen_profile_id']]['robot_initialization'],
            gap_valid=.02<=d['procedural_grasp']['conservative_3d_envelope_distance_m']<=.05,
            cpu_gate=d['procedural_grasp']['accepted_cpu'],scene_sha256=sha(translated(d['scene']))==d['scene_sha256'],
            code_hashes=all(sha(translated(path))==expected for path,expected in d['code_sources'].items()),
            no_online_ik=d['online_ik_calls']==0,continuous_target=d['target_xy_continuously_sampled'],fresh_background=d['backgrounds_freshly_sampled'])
        initial_exit=root/'commands'/p.parent.name/'exit.json'
        checks['prepare_exit0']=initial_exit.is_file() and json.loads(initial_exit.read_text())['returncode']==0
        n=root/'native'/p.parent.name/'native_report.json';e=root/'native_commands'/p.parent.name/'exit.json'
        native=json.loads(n.read_text()) if n.is_file() else {}
        passed=(e.is_file() and json.loads(e.read_text())['returncode']==0 and native.get('status')=='completed' and not native.get('cleanup_errors')
                and native.get('original_native_initial_accepted') is True and native.get('initial_score')==0)
        rows.append(dict(key=p.parent.name,source_api=d['source_api'],profile=d['frozen_profile_id'],cpu_pass=all(checks.values()),checks=checks,
            target_position_m=snapshot['target']['position_m'],gap_m=d['procedural_grasp']['conservative_3d_envelope_distance_m'],
            background_sha256=hashlib.sha256(json.dumps(snapshot['items'],sort_keys=True).encode()).hexdigest(),
            joints_sha256=hashlib.sha256(json.dumps(q,sort_keys=True).encode()).hexdigest(),native_pass=passed,
            native_status=native.get('status','not_started'),native_error=native.get('error')))
    original=json.loads((io.HOST/'public_component_sources.json').read_text())
    intact=all(sha(io.HOST/name)==expected for name,expected in original['task_sources'].items())
    rejected=[]
    for path in sorted((root/'initials').glob('*v3/rejection.json')):
        rejected.append(dict(key=path.parent.name,evidence=str(path),reason='explicit_initialization_rejection',
            counts_as_model_zero=False,old_failure_record_retained=True))
    summary=dict(schema='failure_recovery.continuous_shelf.validation.v2',cpu_pass=sum(r['cpu_pass'] for r in rows),
        actual_initial_count=len(rows),product_types=sorted({r['source_api'] for r in rows if r['cpu_pass']}),
        actual_target_positions=len({tuple(r['target_position_m']) for r in rows if r['cpu_pass']}),
        actual_background_layouts=len({r['background_sha256'] for r in rows if r['cpu_pass']}),
        native_pass=sum(r['native_pass'] for r in rows),rows=rows,original_published_task_files_unchanged=intact,
        model_recovery_tested=False,global_registration_switched=False,real_50_initials_validated=False,
        rejected_initial_requests=rejected,arbitrary_seed_guaranteed_valid=False)
    io.save(root/'validation_summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'}))
