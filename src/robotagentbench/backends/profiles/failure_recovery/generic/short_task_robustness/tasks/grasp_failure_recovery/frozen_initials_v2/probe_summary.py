"""Summarize observed probes, including rejected examples; no source edits."""
import json
import math
from pathlib import Path
from ..common import HERE,owned,write_json,digest

NAMES=('cup_left_train5_native_v1','cup_right_adapt4_native_v1',
       'cup_left_train5_native_v2','cup_right_adapt4_native_v2','cup_right_adapt13_native_sim1_v2')


def main():
    root=HERE/'frozen_initials_v2';out=owned(root/'verification/native_probe_review_v1',fresh=True);out.mkdir(parents=True)
    rows=[]
    for name in NAMES:
        path=root/'probes'/name/'native_report.json';r=json.loads(path.read_text())
        exit_path=root/'probe_commands'/name/'exit.json'
        if not exit_path.exists():raise ValueError('Wait for real cleanup/exit: '+name)
        ex=json.loads(exit_path.read_text());row=dict(run=name,status=r['status'],profile=r['profile'],error=r.get('error'),
            source_report_sha256=digest(path),exit=ex,cleanup_errors=r['cleanup_errors'],
            hold_control_steps=r.get('hold_control_steps'),original_initial_acceptance=r.get('original_initial_acceptance'),
            initial_scores=r.get('initial_result',{}).get('scores'),final_scores=r.get('final_result',{}).get('scores'))
        measured=r.get('measurements',{})
        if 'settled' in measured and 'after_hold' in measured:
            a=measured['settled']['target_transforms'][0];b=measured['after_hold']['target_transforms'][0]
            row['cup_translation_during_hold_m']=math.sqrt(sum((a[i]-b[i])**2 for i in range(3)))
            hand='l' if 'left' in name else 'r';n='idx41_gripper_l_outer_joint1' if hand=='l' else 'idx81_gripper_r_outer_joint1'
            row['active_gripper_primary_after_hold_rad']=measured['after_hold']['gripper_coordinates_rad'][n]
        rows.append(row)
    self_path=root/'verification/protected_pairs_v1/report.json';scans=json.loads(self_path.read_text())
    pins=json.loads((HERE/'public_component_sources.json').read_text())['task_sources']
    drift=[p for p,h in pins.items() if digest(HERE/p)!=h]
    value=dict(rows=rows,successful_native_hold_samples=sum(r['status']=='completed' and r['exit']['returncode']==0 and not r['cleanup_errors'] for r in rows),
        protected_pair_cpu_scan=dict(profiles=scans['profiles'],without_overlap=scans['no_protected_overlap'],with_overlap=scans['protected_overlap'],
            rejected_profiles=[r['profile'] for r in scans['rows'] if r['scan']['overlap']],source_sha256=digest(self_path)),
        all_13_native_accepted=False,bank_edited=False,default_profile_pool_filtered=False,
        bank_sha256=digest(root/'compile/cup_training_adapt_v5/bank_v2.json'),published_source_drift=drift,
        source_parameters_changed=False,model_calls=0,recovery_policy_tested=False)
    write_json(out/'summary.json',value);print(json.dumps(value))
    if drift:raise ValueError('Published source drift')


if __name__=='__main__':main()
