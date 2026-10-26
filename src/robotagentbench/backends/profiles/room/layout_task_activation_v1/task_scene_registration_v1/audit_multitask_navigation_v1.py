"""Read-only independent replay of saved measured selection and clock evidence."""
import json,sys,math
from pathlib import Path
from multi_task_dispatch_v1 import MultiTaskDispatcher,TaskRegistration

def audit(path):
    root=Path(path);report=json.loads((root/'navigation_result.json').read_text())
    registration=json.loads((root/'room_evidence/registration.json').read_text())
    assert report['status']=='native_multitask_navigation_passed'
    regs=[TaskRegistration(**r) for r in registration['geometry']]
    d=MultiTaskDispatcher(regs,sector_radius=registration['sensor']['range'],sector_half_angle=registration['sensor']['halfAngle'])
    rows=report['rows'];start=rows[0]['time_s']-.05
    d.start(accepted_scene=True,simulation_time_s=start)
    prior={r.instance_id:0 for r in regs};selected={k:0 for k in prior}
    for row in rows:
        plan=row['result']['plan'];out=row['result']['outcome']
        assert abs(row['time_s']-d.last_time-.05)<1e-6
        expected=d.select(row['robot'],simulation_time_s=d.last_time)
        assert expected['current_task']==plan['current_task']
        assert len(expected['candidates'])==len(plan['candidates'])
        for a,b in zip(expected['candidates'],plan['candidates']):
            assert a['instance_id']==b['instance_id'] and a['physics_hz']==b['physics_hz']
            for field in ('distance_m','heading_error_deg'):
                assert abs(a[field]-b[field])<1e-10,(field,a[field],b[field])
        assert plan['substeps']==plan['physics_hz']//20
        assert abs(row['time_s']-(start+(plan['frame']+1)*.05))<1e-6
        key=plan['current_task'];indices=row['result']['selected_control_steps']
        for k,index in indices.items():
            delta=index-prior[k]
            assert delta in (0,1)
            if delta:
                assert k==key and not plan['switch_pending'] and plan['physics_hz']==240
                selected[k]+=1
            prior[k]=index
        d.complete_physics_frame(simulation_time_s=row['time_s'])
        d.finish_frame()
        assert out['total_score']==0
    assert d.ended and abs(d.last_time-start-120)<1e-6
    assert all(n>0 for n in selected.values())
    assert not report['contacts'] and not report['self_contacts']
    assert report['rollout_pose_velocity_writes']==0 and report['scorer_instances_retained']
    for transition in report['transitions']:assert transition['physical_state_unchanged']
    return dict(status='passed',control_frames=len(rows),simulation_elapsed_s=d.last_time-start,
        selected_scoring_frames=selected,physics_transitions=report['transitions'],
        task_switches=report['task_switches'],score_instances_retained=True,
        robot_penetration_events=0,full_manipulation_verified=False,
        scope='single layout native navigation and dispatch, not full-room asset collision certification')

if __name__=='__main__':print(json.dumps(audit(sys.argv[1]),indent=2))
