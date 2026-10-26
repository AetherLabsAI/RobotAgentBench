#!/usr/bin/env python3
"""1.4 取杯子 (coffee_retrieve_mixed_original11_20260926): one row per episode in data_list.json
(+ compact rows for directories that are present but not part of the batch of 50)."""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa

TASK = '1.4'
DS = os.path.join(ROOT, 'coffee_retrieve_mixed_original11_20260926')
EXPECTED = ['pose_proposal.json', 'queue.json', 'plan.json', 'executed_plan.json', 'preflight_certificate.json', 'result.json',
            'status.json', 'liquid_report.json', 'runner.py', 'head.mp4', 'left.mp4', 'right.mp4', 'camera_manifest.json',
            'capture_events.jsonl', 'initial_arm_fixture.json', 'appearance.json']
STD_LEFT = [-0.9566962718963623, 0.5711908936500549, 0.273301899433136, -1.16750967502594, 1.2575517892837524, 1.3616235256195068, -0.16790390014648438]
STD_RIGHT = [1.073685884475708, -0.6114768385887146, -0.280689537525177, 1.285286784172058, -0.7304344177246094, -1.481042742729187, 0.18751047551631927]
L_KEYS = [f'idx2{i}_arm_l_joint{i}' for i in range(1, 8)]
R_KEYS = [f'idx6{i}_arm_r_joint{i}' for i in range(1, 8)]


def build(d, eid, ms, rec, legacy_runs):
    files = sorted(os.listdir(d)) if os.path.isdir(d) else []
    missing = [f for f in EXPECTED if f not in files]
    plan = load_opt(f'{d}/plan.json') or {}
    ex = load_opt(f'{d}/executed_plan.json') or {}
    res = load_opt(f'{d}/result.json') or {}
    cert = load_opt(f'{d}/preflight_certificate.json') or {}
    status = load_opt(f'{d}/status.json') or {}
    liquid = load_opt(f'{d}/liquid_report.json') or {}
    queue = load_opt(f'{d}/queue.json')
    pprop = load_opt(f'{d}/pose_proposal.json')
    seed_plan = load_opt(f'{d}/seed_plan.json')
    fixture = load_opt(f'{d}/initial_arm_fixture.json')
    repair_metrics = load_opt(f'{d}/repair_metrics.json')
    cam = load_opt(f'{d}/camera_manifest.json') or {}
    trace = res.get('trace') or []
    t0 = trace[0] if trace else {}
    tl = trace[-1] if trace else {}
    steps = plan.get('steps') or []
    s0 = steps[0] if steps else {}
    arm = plan.get('active_arm') or (ms or {}).get('arm') or (queue or {}).get('arm')
    std = STD_LEFT if arm == 'left' else STD_RIGHT
    ajp = t0.get('all_joint_positions') or {}
    keys = L_KEYS if arm == 'left' else R_KEYS
    active_q0 = [ajp.get(k) for k in keys] if ajp else None
    other_keys = R_KEYS if arm == 'left' else L_KEYS
    other_std = STD_RIGHT if arm == 'left' else STD_LEFT
    other_q0 = [ajp.get(k) for k in other_keys] if ajp else None
    m_yaw = machine_yaw_from_matrix(t0['machine_rotation'])[0] if t0.get('machine_rotation') else None
    bag_heading = local_z_heading(t0['bag_rotation'])[0] if t0.get('bag_rotation') else None
    handle_yaw = heading_deg(t0['handle_axis']) % 360 if t0.get('handle_axis') else None
    mxyz = t0.get('machine')
    rpr = plan.get('retrieval_pose_randomization') or {}
    rnd = plan.get('randomization') or {}
    checks = res.get('checks') or {}
    ts = [x.get('t') for x in trace]
    dts = sorted(set(round(ts[i + 1] - ts[i], 6) for i in range(len(ts) - 1))) if len(ts) > 1 else []
    fc, goal = res.get('final_cup'), res.get('goal')
    ic = res.get('initial_cup')
    ctarget = (plan.get('geometry') or {}).get('cup_target')
    ph = phases_of(steps)
    vids = {k: os.path.join(d, f'{k}.mp4') for k in ('head', 'left', 'right')} if rec is None else rec['videos']
    ipr = plan.get('initial_prefix_repair') or {}
    row = {
        'task': TASK, 'episode_id': eid, 'dir': d,
        'title': (rec or {}).get('title'), 'data_list_id': (rec or {}).get('id'),
        'source_status': (rec or {}).get('source_status'), 'source_exists_on_202': (rec or {}).get('source_exists_on_202'),
        'source_path': (rec or {}).get('source_path'),
        'in_data_list': rec is not None, 'in_dataset_manifest': ms is not None,
        'legacy_initial': eid in legacy_runs,
        'initial_state_status': (ms or {}).get('initial_state_status'),
        'missing_files': missing, 'files': files,
        'seed': {'plan_randomization': rnd.get('seed'), 'retrieval_pose_randomization': rpr.get('seed'),
                 'retrieval_pose_proposal_index': rpr.get('proposal_index'),
                 'pose_proposal_json': pprop if pprop is None else {k: v for k, v in pprop.items() if len(str(v)) < 300},
                 'seed_plan_json_keys': sorted(seed_plan.keys()) if isinstance(seed_plan, dict) else None,
                 'result_randomization': (res.get('randomization') or {}).get('seed')},
        'arm': {'plan': plan.get('active_arm'), 'manifest': (ms or {}).get('arm'), 'queue': (queue or {}).get('arm'),
                'preflight_model_directory': cert.get('model_directory'), 'resolved': arm},
        'group': {'manifest': (ms or {}).get('group'), 'plan_perturbation_group': plan.get('perturbation_group'),
                  'plan_randomization_bounds': plan.get('randomization_bounds')},
        'cup_style': {'plan': plan.get('cup_style'), 'manifest': (ms or {}).get('cup_style'),
                      'preflight_cup_asset': (cert.get('cup_asset') or {}).get('name')},
        'machine': {'plan_offset_m': plan.get('machine_offset_m'), 'plan_yaw_deg': plan.get('machine_yaw_deg'),
                    'manifest_offset_m': (ms or {}).get('machine_offset_m'), 'manifest_yaw_deg': (ms or {}).get('machine_yaw_deg'),
                    'trace0_xyz': mxyz, 'trace0_yaw_deg': m_yaw,
                    'offset_from_nominal_m': [mxyz[i] - NOMINAL_MACHINE[i] for i in range(3)] if mxyz else None,
                    'retrieval_source_machine_pose': plan.get('retrieval_source_machine_pose')},
        'cup': {'initial_handle_world_yaw_deg_manifest': (ms or {}).get('initial_handle_world_yaw_deg'),
                'trace0_handle_yaw_deg': handle_yaw,
                'placing_handle_world_yaw_deg': rnd.get('handle_world_yaw_deg'),
                'placing_cup_start_xyz': plan.get('cup_start_xyz'), 'placing_cup_start_yaw_deg': plan.get('cup_start_yaw_deg'),
                'actual_initial_cup_manifest': (ms or {}).get('actual_initial_cup'), 'actual_final_cup_manifest': (ms or {}).get('actual_final_cup'),
                'result_initial_cup': ic, 'result_final_cup': fc, 'goal': goal,
                'plan_cup_target_on_tray': ctarget,
                'initial_cup_to_tray_target_xy_m': dist(ic, ctarget, 2) if ic and ctarget else None,
                'initial_cup_rel_machine_xy': [ic[0] - mxyz[0], ic[1] - mxyz[1]] if ic and mxyz else None,
                'removal_goal': plan.get('removal_goal'), 'chest_tcp_goal': plan.get('chest_tcp_goal')},
        'bag': {'plan_offset_m': plan.get('bag_offset_m'), 'plan_yaw_deg': plan.get('bag_yaw_deg'),
                'trace0_xyz': t0.get('bag'), 'trace0_localZ_heading_deg': bag_heading},
        'liquid': {'initial': liquid.get('initial'), 'final': liquid.get('final'), 'method': liquid.get('method'),
                   'manifest_equal': ((ms or {}).get('liquid_report') == liquid) if ms else None,
                   'plan_liquid_initialization': plan.get('liquid_initialization')},
        'robot': {'root_xyz': plan.get('root_xyz'), 'yaw': plan.get('yaw'),
                  'initial_targets_hash': sha_obj(plan.get('initial_targets')) if plan.get('initial_targets') else None,
                  'initial_targets': plan.get('initial_targets'),
                  'plan_initial_q': plan.get('initial'),
                  'plan_initial_q_is_standard': (max(abs(a - b) for a, b in zip(plan['initial'], std)) < 1e-6) if plan.get('initial') else None,
                  'plan_initial_q_max_abs_diff_from_standard_rad': max(abs(a - b) for a, b in zip(plan['initial'], std)) if plan.get('initial') else None,
                  'step0_q': s0.get('q'), 'step0_opening': s0.get('opening'), 'step0_tcp': s0.get('tcp'), 'step0_phase': s0.get('phase'),
                  'trace0_active_arm_q': active_q0,
                  'trace0_active_arm_max_abs_diff_from_standard_rad': max(abs(a - b) for a, b in zip(active_q0, std)) if active_q0 and None not in active_q0 else None,
                  'trace0_other_arm_max_abs_diff_from_standard_rad': max(abs(a - b) for a, b in zip(other_q0, other_std)) if other_q0 and None not in other_q0 else None,
                  'trace0_opening': t0.get('opening'), 'trace0_tcp': t0.get('tcp'),
                  'trace0_bodylift': t0.get('bodylift'), 'trace0_waist': t0.get('waist'),
                  'trace0_tcp_to_initial_cup_m': dist(t0['tcp'], ic, 3) if t0.get('tcp') and ic else None,
                  'trace0_all_joint_positions': ajp or None,
                  'initial_arm_fixture_json': fixture,
                  'plan_initial_active_arm_retarget': plan.get('initial_active_arm_retarget'),
                  'plan_initial_prefix_repair': {k: ipr.get(k) for k in ('added_commands', 'route_nodes', 'collision_queries', 'preserve_original_scene_and_suffix', 'source')} if ipr else None,
                  'plan_approach_shortcut_repair': plan.get('approach_shortcut_repair'),
                  'plan_local_route_repair': plan.get('local_route_repair'),
                  'initial_state_reference': plan.get('initial_state_reference'),
                  'initial_manifest_sha256': plan.get('initial_manifest_sha256')},
        'plan': {'n_steps': len(steps), 'phases': ph, 'first_phase_len': ph[0][1] if ph else None,
                 'step_keys': sorted(s0.keys()) if s0 else None, 'q_len': len(s0['q']) if 'q' in s0 else None,
                 'keys': sorted(plan.keys()), 'task': plan.get('task'), 'scope': plan.get('scope'),
                 'continued_from': plan.get('continued_from'),
                 'executed_equals_plan': (ex == plan) if ex and plan else None,
                 'executed_n_steps': len(ex.get('steps') or []),
                 'executed_steps_equal': (ex.get('steps') == plan.get('steps')) if ex and plan else None,
                 'keys_only_in_executed': sorted(set(ex) - set(plan)), 'keys_only_in_plan': sorted(set(plan) - set(ex)),
                 'keys_with_different_values': sorted(k for k in set(ex) & set(plan) if ex[k] != plan[k]),
                 'head_visibility_passed': (plan.get('head_visibility') or {}).get('passed')},
        'preflight': {'passed': cert.get('passed'), 'checked_samples': cert.get('checked_samples'),
                      'n_violations': len(cert.get('violations') or []), 'clearance_m': cert.get('clearance_m'),
                      'mode': cert.get('mode'), 'coverage_complete': (cert.get('coverage') or {}).get('complete'),
                      'keys': sorted(cert.keys())},
        'result': {'passed': res.get('passed'), 'failure': res.get('failure'), 'checks': checks,
                   'n_checks': len(checks), 'all_checks_true': all(v is True for v in checks.values()) if checks else None,
                   'false_checks': [k for k, v in checks.items() if v is not True],
                   'status_json_status': status.get('status'), 'status_json_failure': status.get('failure'),
                   'status_checks_equal': (status.get('checks') == checks) if status else None,
                   'manifest_physical_checks_equal': ((ms or {}).get('physical_checks') == checks) if ms else None,
                   'manifest_evidence_checks': (ms or {}).get('evidence_checks'),
                   'manifest_initial_repair_audit_passed': ((ms or {}).get('initial_repair_audit') or {}).get('passed'),
                   'manifest_initial_repair_audit_checks': ((ms or {}).get('initial_repair_audit') or {}).get('checks'),
                   'manifest_initial_repair_added_commands': ((ms or {}).get('initial_repair_audit') or {}).get('added_commands'),
                   'final_to_goal_xy_m': dist(fc, goal, 2) if fc and goal else None,
                   'final_to_goal_xyz_m': dist(fc, goal, 3) if fc and goal else None,
                   'cup_displacement_m': dist(fc, ic, 3) if fc and ic else None,
                   'cup_lift_dz_m': (fc[2] - ic[2]) if fc and ic else None,
                   'final_tcp_to_chest_goal_m': dist(tl['tcp'], plan['chest_tcp_goal'], 3) if tl.get('tcp') and plan.get('chest_tcp_goal') else None,
                   'initial_yaw_error_deg': res.get('initial_yaw_error_deg'),
                   'max_bilateral_contact_frames': res.get('max_bilateral_contact_frames'),
                   'fk_error': res.get('fk_error'),
                   'n_unexpected_contacts': len(res.get('unexpected_contacts') or []),
                   'trace_len': len(trace), 'trace_t_first': ts[0] if ts else None, 'trace_t_last': ts[-1] if ts else None,
                   'trace_dt_values': dts, 'trace_keys': sorted(t0.keys()),
                   'trace_has_all_joint_positions': 'all_joint_positions' in t0,
                   'n_joints_recorded': len(ajp),
                   'execution_scope': res.get('execution_scope'), 'keys': sorted(res.keys())},
        'manifest_extra': {k: (ms or {}).get(k) for k in ('original_selected_id', 'original_source', 'supersedes', 'user_decision', 'repair_metrics', 'source')},
        'repair_metrics_json': repair_metrics,
        'queue_json': queue,
        'appearance': None,
        'appearance_note': 'appearance.json absent in dataset dir; runner.py applies apply_appearance(seed=plan.randomization.seed) and writes appearance.json to the run output dir (not copied to 202)',
        'appearance_seed_used_by_runner': rnd.get('seed', 0),
        'camera_resolution': {k: v.get('resolution') for k, v in cam.items()} if cam else None,
        'capture_events': capture_events_info(f'{d}/capture_events.jsonl'),
        'video': video_block(vids),
        'sha256_manifest': {k: (((ms or {}).get('videos') or {}).get(k) or {}).get('sha256') for k in ('head', 'left', 'right')},
        'manifest_video_meta': {k: {kk: vv for kk, vv in ((((ms or {}).get('videos') or {}).get(k)) or {}).items() if kk != 'sha256'} for k in ('head', 'left', 'right')},
    }
    return row


def main():
    recs = records(TASK)
    manifest = load(os.path.join(DS, 'dataset_manifest.json'))
    msamples = {s['id']: s for s in manifest['samples']}
    legacy = manifest.get('legacy_initial_runs') or []
    dirs = sorted(x for x in os.listdir(DS) if os.path.isdir(os.path.join(DS, x)))
    vids = []
    for x in dirs:
        vids += [os.path.join(DS, x, f'{k}.mp4') for k in ('head', 'left', 'right')]
    ffprobe_many(vids)
    rows = []
    for r in recs:
        d = episode_dir(r)
        eid = os.path.basename(d)
        rows.append(build(d, eid, msamples.get(eid), r, legacy))
    listed = [r['episode_id'] for r in rows]
    extra_rows = [build(os.path.join(DS, x), x, msamples.get(x), None, legacy) for x in dirs if x not in listed]
    extra = {
        'dataset_dir': DS, 'n_rows': len(rows), 'n_dirs': len(dirs),
        'dirs_not_in_data_list': [x for x in dirs if x not in listed],
        'data_list_not_in_dirs': [x for x in listed if x not in dirs],
        'manifest_not_in_data_list': [x for x in msamples if x not in listed],
        'data_list_not_in_manifest': [x for x in listed if x not in msamples],
        'manifest_top': {k: v for k, v in manifest.items() if k != 'samples'},
        'legacy_initial_runs': legacy,
        'extra_dir_rows': extra_rows,
        'data_list_legacy_status_titles': [r['title'] for r in recs if '旧起点' in r['source_status']],
    }
    save_rows(TASK, rows, extra)
    print('rows', len(rows), 'extra dirs', extra['dirs_not_in_data_list'])


if __name__ == '__main__':
    main()
