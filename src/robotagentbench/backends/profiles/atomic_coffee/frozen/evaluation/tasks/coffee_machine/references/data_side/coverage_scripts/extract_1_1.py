#!/usr/bin/env python3
"""1.1 放杯子 (coffee_cup_left15_right35_20260925): one row per episode in data_list.json."""
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa

TASK = '1.1'
DS = os.path.join(ROOT, 'coffee_cup_left15_right35_20260925')
EXPECTED = ['appearance.json', 'proposal.json', 'plan.json', 'executed_plan.json', 'preflight_certificate.json',
            'source.json', 'result.json', 'camera_manifest.json', 'capture_events.jsonl', 'head.mp4', 'left.mp4', 'right.mp4']


def main():
    recs = records(TASK)
    manifest = load(os.path.join(DS, 'dataset_manifest.json'))
    msamples = {s['sample_id']: s for s in manifest['samples']}
    rows, flats = [], []
    vids = {}
    for r in recs:
        for k, p in r['videos'].items():
            vids[p] = None
    ffprobe_many(list(vids))
    for r in recs:
        d = episode_dir(r)
        eid = os.path.basename(d)
        ms = msamples.get(eid)
        files = sorted(os.listdir(d)) if os.path.isdir(d) else []
        missing = [f for f in EXPECTED if f not in files]
        prop = load_opt(f'{d}/proposal.json') or {}
        plan = load_opt(f'{d}/plan.json') or {}
        ex = load_opt(f'{d}/executed_plan.json') or {}
        res = load_opt(f'{d}/result.json') or {}
        cert = load_opt(f'{d}/preflight_certificate.json') or {}
        src = load_opt(f'{d}/source.json') or {}
        app = load_opt(f'{d}/appearance.json')
        cam = load_opt(f'{d}/camera_manifest.json') or {}
        trace = res.get('trace') or []
        t0 = trace[0] if trace else {}
        tl = trace[-1] if trace else {}
        m_yaw = m_conv = None
        if t0.get('machine_rotation'):
            m_yaw, _ = machine_yaw_from_matrix(t0['machine_rotation'])
        bag_heading = None
        if t0.get('bag_rotation'):
            bag_heading, _ = local_z_heading(t0['bag_rotation'])
        handle_yaw_trace = heading_deg(t0['handle_axis']) % 360 if t0.get('handle_axis') else None
        mxyz = t0.get('machine')
        m = re.match(r'(\d+)_(left|right)_([\d.]+)deg', eid)
        name = (ms or {}).get('name') or src.get('name')
        seed_from_name = None
        mm = re.search(r'_(\d{4,})_', name or '')
        if mm:
            seed_from_name = int(mm.group(1))
        steps = plan.get('steps') or []
        ts = [x.get('t') for x in trace]
        dts = sorted(set(round(ts[i + 1] - ts[i], 6) for i in range(len(ts) - 1))) if len(ts) > 1 else []
        goal = res.get('goal')
        fc = res.get('final_cup')
        checks = res.get('checks') or {}
        row = {
            'task': TASK, 'episode_id': eid, 'dir': d, 'title': r['title'], 'data_list_id': r['id'],
            'source_status': r['source_status'], 'source_exists_on_202': r['source_exists_on_202'],
            'source_path': r['source_path'], 'source_run_name': name,
            'in_dataset_manifest': ms is not None,
            'missing_files': missing, 'files': files,
            'seed': {'proposal': prop.get('seed'), 'plan_randomization': (plan.get('randomization') or {}).get('seed'),
                     'result_randomization': (res.get('randomization') or {}).get('seed'),
                     'appearance': (app or {}).get('seed'), 'from_source_name': seed_from_name},
            'arm': {'dir_name': m.group(2) if m else None, 'manifest': (ms or {}).get('arm'), 'proposal': prop.get('active_arm'),
                    'plan': plan.get('active_arm')},
            'group': (ms or {}).get('group'),
            'handle_world_yaw_deg': {'proposal': prop.get('handle_world_yaw_deg'), 'manifest_angle': (ms or {}).get('angle'),
                                     'trace0_handle_axis': handle_yaw_trace, 'dir_name': float(m.group(3)) if m else None},
            'angle_bin': prop.get('angle_bin'),
            'plan_randomization_raw': plan.get('randomization'),
            'plan_extra_keys': {k: plan[k] for k in plan if k not in ('steps', 'initial_targets', 'geometry', 'head_visibility', 'initial', 'randomization') and len(str(plan[k])) < 400},
            'cup_xy_offset_m': prop.get('cup_xy_offset_m'),
            'proposal_keys': sorted(prop.keys()),
            'proposal_extra': {k: v for k, v in prop.items() if k not in ('seed', 'handle_world_yaw_deg', 'cup_xy_offset_m', 'angle_bin', 'active_arm')},
            'cup_style': {'plan': plan.get('cup_style'), 'manifest': (ms or {}).get('style'), 'preflight_cup_asset': (cert.get('cup_asset') or {}).get('name')},
            'cup_start_xyz': {'plan': plan.get('cup_start_xyz'), 'manifest': (ms or {}).get('cup_xyz'), 'result_initial_cup': res.get('initial_cup'), 'trace0': t0.get('cup')},
            'cup_start_yaw_deg_plan': plan.get('cup_start_yaw_deg'),
            'machine': {
                'plan_offset_m': plan.get('machine_offset_m'), 'plan_yaw_deg': plan.get('machine_yaw_deg'),
                'manifest_xyz': (ms or {}).get('machine_xyz'), 'trace0_xyz': mxyz,
                'trace0_yaw_deg': m_yaw,
                'offset_from_nominal_m': [mxyz[i] - NOMINAL_MACHINE[i] for i in range(3)] if mxyz else None,
                'machine_front_plan': (plan.get('geometry') or {}).get('machine_front'),
            },
            'bag': {'plan_offset_m': plan.get('bag_offset_m'), 'plan_yaw_deg': plan.get('bag_yaw_deg'),
                    'trace0_xyz': t0.get('bag'), 'trace0_localZ_heading_deg': bag_heading},
            'robot': {'root_xyz': plan.get('root_xyz'), 'yaw': plan.get('yaw'),
                      'initial_targets_hash': sha_obj(plan.get('initial_targets')) if plan.get('initial_targets') else None,
                      'initial_targets_keys': sorted((plan.get('initial_targets') or {}).keys()),
                      'plan_initial_q': plan.get('initial'),
                      'trace0_all_joint_positions': t0.get('all_joint_positions'),
                      'trace0_tcp': t0.get('tcp'),
                      'initial_state_reference': plan.get('initial_state_reference'),
                      'initial_manifest_sha256': plan.get('initial_manifest_sha256')},
            'plan': {'n_steps': len(steps), 'phases': phases_of(steps), 'step_keys': sorted(steps[0].keys()) if steps else None,
                     'q_len': len(steps[0]['q']) if steps and 'q' in steps[0] else None,
                     'keys': sorted(plan.keys()),
                     'executed_equals_plan': (ex == plan) if ex and plan else None,
                     'executed_n_steps': len(ex.get('steps') or []),
                     'executed_steps_equal': (ex.get('steps') == plan.get('steps')) if ex and plan else None,
                     'keys_only_in_executed': sorted(set(ex) - set(plan)), 'keys_only_in_plan': sorted(set(plan) - set(ex)),
                     'head_visibility_passed': (plan.get('head_visibility') or {}).get('passed'),
                     'cup_target': (plan.get('geometry') or {}).get('cup_target')},
            'preflight': {'passed': cert.get('passed'), 'checked_samples': cert.get('checked_samples'),
                          'n_violations': len(cert.get('violations') or []), 'clearance_m': cert.get('clearance_m'),
                          'mode': cert.get('mode'), 'model_directory': cert.get('model_directory'),
                          'coverage_complete': (cert.get('coverage') or {}).get('complete'),
                          'n_layout_violations': len(cert.get('layout_violations') or []),
                          'keys': sorted(cert.keys())},
            'result': {'passed': res.get('passed'), 'failure': res.get('failure'), 'checks': checks,
                       'n_checks': len(checks), 'all_checks_true': all(v is True for v in checks.values()) if checks else None,
                       'false_checks': [k for k, v in checks.items() if v is not True],
                       'manifest_checks_count': (ms or {}).get('checks'),
                       'initial_cup': res.get('initial_cup'), 'final_cup': fc, 'goal': goal,
                       'final_to_goal_xy_m': dist(fc, goal, 2) if fc and goal else None,
                       'final_to_goal_xyz_m': dist(fc, goal, 3) if fc and goal else None,
                       'final_minus_goal_z_m': (fc[2] - goal[2]) if fc and goal else None,
                       'initial_yaw_error_deg': res.get('initial_yaw_error_deg'),
                       'max_bilateral_contact_frames': res.get('max_bilateral_contact_frames'),
                       'fk_error': res.get('fk_error'),
                       'n_unexpected_contacts': len(res.get('unexpected_contacts') or []),
                       'n_contact_pairs': len(res.get('contact_pairs') or {}),
                       'trace_len': len(trace), 'trace_t_first': ts[0] if ts else None, 'trace_t_last': ts[-1] if ts else None,
                       'trace_dt_values': dts, 'trace_keys': sorted(t0.keys()),
                       'trace_has_all_joint_positions': 'all_joint_positions' in t0,
                       'n_joints_recorded': len(t0.get('all_joint_positions') or {}),
                       'trace_last_cup': tl.get('cup'), 'trace_last_machine': tl.get('machine'),
                       'execution_scope': res.get('execution_scope'), 'continued_from': res.get('continued_from'),
                       'keys': sorted(res.keys())},
            'appearance': appearance_compact(app),
            'appearance_keys': sorted(app.keys()) if app else None,
            'camera_resolution': {k: v.get('resolution') for k, v in cam.items()} if cam else None,
            'capture_events': capture_events_info(f'{d}/capture_events.jsonl'),
            'video': video_block(r['videos']),
            'sha256_manifest': {k: ((ms or {}).get('files') or {}).get(k, {}).get('sha256') for k in ('head', 'left', 'right')},
            'manifest_video_meta': {k: ((ms or {}).get('files') or {}).get(k, {}).get('video') for k in ('head', 'left', 'right')},
            'identity_sha256': (ms or {}).get('identity_sha256'),
        }
        rows.append(row)
        if app:
            flats.append(flatten(app))
    varying, constant = varying_keys(flats)
    dirs = sorted(x for x in os.listdir(DS) if os.path.isdir(os.path.join(DS, x)))
    listed = [r['episode_id'] for r in rows]
    extra = {
        'dataset_dir': DS, 'n_rows': len(rows), 'n_dirs': len(dirs),
        'dirs_not_in_data_list': [x for x in dirs if x not in listed],
        'data_list_not_in_dirs': [x for x in listed if x not in dirs],
        'manifest_not_in_data_list': [x for x in msamples if x not in listed],
        'data_list_not_in_manifest': [x for x in listed if x not in msamples],
        'manifest_summary': manifest.get('summary'),
        'appearance_varying': varying, 'appearance_constant_keys': sorted(constant.keys()),
        'appearance_constant_values': {k: (v if not isinstance(v, str) or len(v) < 200 else v[:200] + '...') for k, v in constant.items()},
        'n_appearance_files': len(flats),
    }
    save_rows(TASK, rows, extra)
    print('rows', len(rows), 'extra dirs', extra['dirs_not_in_data_list'])


if __name__ == '__main__':
    main()
