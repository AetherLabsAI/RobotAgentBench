"""Independent clock/selection-isolation audit; not full manipulation proof."""
import json,sys,hashlib
from pathlib import Path

def audit(path):
    root=Path(path);r=json.loads((root/'navigation_result.json').read_text())
    assert r['status']=='native_snack_room_activation_passed',r.get('error')
    rows=r['rows'];previous={k:0 for k in ('coffee','fryer','snack')}
    selected=0;start=rows[0]['time_s']-.05;rates=set()
    for i,row in enumerate(rows):
        p=row['result']['plan'];indices=row['result']['selected_control_steps']
        assert p['frame']==i and p['control_hz']==20
        assert p['substeps']==p['physics_hz']//20
        assert abs(row['time_s']-(start+(i+1)*.05))<1e-6
        rates.add(p['physics_hz'])
        for k,n in indices.items():
            delta=n-previous[k];assert delta in (0,1)
            if delta:
                assert k==p['current_task']=='snack'
                assert p['physics_hz']==240 and not p['switch_pending']
                selected+=1
            previous[k]=n
        assert row['result']['outcome']['total_score']==0
    assert rates=={60,240} and selected>0
    trace=[json.loads(s) for s in (root/'room_evidence/snack/selected_scores.jsonl').read_text().splitlines()]
    assert len(trace)==selected
    for i,row in enumerate(trace,1):
        s=row['score']
        assert s['native_scoring_samples']==12*i
        assert not s['initial_state_gate_enabled'] and not s['outside_workspace_failure_enabled']
        assert s['score']==0 and s['phase']=='pick' and not s['success']
    assert r['initial_acceptance']['snack']['status']=='not_checked_user_disabled'
    assert not r['contacts'] and not r['self_contacts']
    assert r['source_layers_unchanged'] and r['scorer_instances_retained']
    assert r['rollout_pose_velocity_writes']==0
    assert sum(t['to_task']=='snack' for t in r['task_switches'])>=2
    for t in r['transitions']:assert t['physical_state_unchanged']
    return dict(status='passed',control_frames=len(rows),selected_control_frames=selected,
                snack_native_samples=selected*12,physics_hz=sorted(rates),control_hz=20,
                full_manipulation_verified=False,initial_gate=False,outside_failure=False,
                contacts=0,score=0,scope='native clock and score isolation; no full pick/place action',
                source_report_sha256=hashlib.sha256((root/'navigation_result.json').read_bytes()).hexdigest())

if __name__=='__main__':
    result=audit(sys.argv[1])
    if len(sys.argv)>2:Path(sys.argv[2]).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
