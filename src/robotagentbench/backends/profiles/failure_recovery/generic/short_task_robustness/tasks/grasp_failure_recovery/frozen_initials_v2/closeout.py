"""Freeze code and observed CPU evidence without changing public source pins."""
import json
import shutil
import statistics
from pathlib import Path
from ..common import HERE,owned,digest,write_json
from ..frozen_initials_v1.bank import load


def main():
    root=HERE/'frozen_initials_v2';bankpath=root/'compile/cup_training_adapt_v5/bank_v2.json';bank=load(bankpath)
    profiles=bank['cases']['table_waste_empty_grasp']['profiles']
    batches=[root/'batch'/key/'summary.json' for key in ('cup_profiles13_first_v2','cup_profiles13_second_v2')]
    rows=[];calls=[]
    for path in batches:
        result=json.loads(path.read_text());rows+=result['rows'];calls+=result['optimizer_calls']
        if not result['training_dataset_mounted'] is False:raise ValueError('Runtime must not mount training dataset')
    if calls or len(rows)!=26 or not all(r['accepted'] for r in rows):raise ValueError('Actual CPU generation incomplete')
    counts={p['id']:sum(r['profile']==p['id'] for r in rows) for p in profiles}
    if any(v!=2 for v in counts.values()):raise ValueError('Every frozen posture needs two actual random scenes')
    pins=json.loads((HERE/'public_component_sources.json').read_text())['task_sources']
    drift=[p for p,h in pins.items() if digest(HERE/p)!=h]
    if drift:raise ValueError('Published source pins changed; do not rewrite them')
    smoke_path=HERE/'expansion_v2/candidates/cup_default_training13_smoke_v2/initial.json';smoke=json.loads(smoke_path.read_text())
    launch=json.loads((HERE/'expansion_v2/commands/cup_default_training13_smoke_v2/command.json').read_text())
    if launch['implementation']!='grasp_failure_recovery.frozen_initials_v2.cup' or not smoke['preparation_accepted'] or smoke['online_ik_calls']!=0:
        raise ValueError('Task-local default dispatch not verified')
    tests=root/'verification/rules_v2/result.json';test_result=json.loads(tests.read_text())
    if not test_result['successful']:raise ValueError('Rule tests failed')
    out=owned(root/'versions/training_v2',fresh=True);out.mkdir(parents=True);hashes={}
    paths=[*root.glob('*.py'),root/'README.md',bankpath,HERE/'expansion_v2/candidate_runner.py']
    for path in paths:
        rel=path.relative_to(HERE);dst=out/rel;dst.parent.mkdir(parents=True,exist_ok=True)
        if path.is_symlink():raise ValueError('Snapshot must not be an external link')
        shutil.copy2(path,dst);hashes[str(rel)]=digest(dst)
    times=[r['generation_wall_seconds'] for r in rows]
    value=dict(profiles=len(profiles),hands={h:sum(p['active_hand']==h for p in profiles) for h in ('l','r')},
        exact_training_profiles=sum(p.get('training_provenance',{}).get('active_arm_copied_exactly') is True for p in profiles),
        training_seeded_adapted_profiles=sum(p.get('training_provenance',{}).get('training_seeded_offline_ik') is True for p in profiles),
        retained_v1_profiles=sum(not p.get('training_provenance') for p in profiles),
        bank_sha256=digest(bankpath),bank_bytes=bankpath.stat().st_size,source_snapshot=hashes,
        published_v1_source_count=len(pins),published_v1_source_drift=drift,shared_entries_edited=False,
        cpu_generations=len(rows),cpu_accepted=len(rows),profile_counts=counts,optimizer_calls=calls,
        distinct_cup_poses=len({json.dumps(r['target_world']) for r in rows}),
        timing_seconds=dict(min=min(times),median=statistics.median(times),max=max(times)),
        actual_batch_summaries={str(p.relative_to(HERE)):digest(p) for p in batches},
        default_task_local_dispatch='v2',default_smoke_sha256=digest(smoke_path),
        unit_tests=test_result['tests_run'],unit_receipt_sha256=digest(tests),
        training_dataset_required_at_runtime=False,fresh_whole_rooms_generated=False,
        native_physics_tested=False,rendered_images_verified=False,recovery_rollout_tested=False,model_calls=0)
    write_json(out/'audit.json',value);print(json.dumps({k:v for k,v in value.items() if k not in ('source_snapshot','profile_counts')}))


if __name__=='__main__':main()
