#!/usr/bin/env python3
"""1.3 按按钮 (coffee_button_v2_20260925): one row per episode in data_list.json."""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa

TASK = '1.3'
DS = os.path.join(ROOT, 'coffee_button_v2_20260925')
EXPECTED = ['appearance.json', 'proposal.json', 'initialization.json', 'button_plan.json', 'button_certificate.json',
            'horizontal_route.json', 'motion/head.mp4', 'motion/left.mp4', 'motion/right.mp4', 'motion/result.json',
            'motion/capture_events.jsonl', 'motion/videos_ready.json']
VERIF = os.path.join(ROOT, 'coffee_updates_0925_verification.json')


def pose(m):
    if not m:
        return None
    hz, trans = local_z_heading(m)
    hx, _ = local_x_heading(m)
    return {'xyz': trans, 'localZ_heading_deg': hz, 'localX_heading_deg': hx, 'tilt_deg': tilt_deg(m)}


def main():
    recs = records(TASK)
    manifest = load(os.path.join(DS, 'manifest.json'))
    mruns = {s['run']: s for s in manifest['episodes']}
    verif = {v['file']: v for v in load(VERIF)} if os.path.exists(VERIF) else {}
    vids = []
    for r in recs:
        vids += list(r['videos'].values())
    ffprobe_many(vids)
    rows, flats, flats_init = [], [], []
    for r in recs:
        d = episode_dir(r, parent=True)
        eid = os.path.basename(d)
        ms = mruns.get(eid)
        missing = [f for f in EXPECTED if not os.path.exists(os.path.join(d, f))]
        files = sorted(os.listdir(d)) + ['motion/' + f for f in sorted(os.listdir(os.path.join(d, 'motion')))] if os.path.isdir(d) else []
        prop = load_opt(f'{d}/proposal.json') or {}
        init = load_opt(f'{d}/initialization.json') or {}
        app = load_opt(f'{d}/appearance.json')
        plan = load_opt(f'{d}/button_plan.json') or {}
        cert = load_opt(f'{d}/button_certificate.json') or {}
        route = load_opt(f'{d}/horizontal_route.json') or {}
        res = load_opt(f'{d}/motion/result.json') or {}
        vready = load_opt(f'{d}/motion/videos_ready.json') or {}
        mw = init.get('machine_world')
        m_yaw, m_xyz = machine_yaw_from_matrix(mw) if mw else (None, None)
        trace = res.get('trace') or []
        t0 = trace[0] if trace else {}
        steps = plan.get('steps') or []
        bgr = init.get('bag_geometry_revision') or {}
        checks = res.get('checks') or {}
        events = res.get('events') or []
        disp = [(t.get('status') or {}).get('button_displacement_m') for t in trace]
        disp = [x for x in disp if x is not None]
        pressed_steps = sum(1 for t in trace if (t.get('status') or {}).get('button_pressed'))
        disp_steps = sum(1 for t in trace if (t.get('status') or {}).get('dispensing'))
        parts = [t.get('particles') for t in trace if t.get('particles') is not None]
        bean_counts = sorted(set((t.get('status') or {}).get('bean_count') for t in trace))
        raw_bean_counts = sorted(set((t.get('status') or {}).get('raw_bean_count') for t in trace))
        ts = [t.get('time_s') for t in trace]
        dts = sorted(set(round(ts[i + 1] - ts[i], 6) for i in range(len(ts) - 1))) if len(ts) > 1 else []
        mm = re.search(r'seed(\d+)', eid)
        travel = [s.get('button_travel') for s in steps if s.get('button_travel') is not None]
        row = {
            'task': TASK, 'episode_id': eid, 'dir': d, 'title': r['title'], 'data_list_id': r['id'],
            'source_status': r['source_status'], 'source_exists_on_202': r['source_exists_on_202'],
            'source_path': r['source_path'], 'in_dataset_manifest': ms is not None,
            'missing_files': missing, 'files': files,
            'seed': {'dir_name': int(mm.group(1)) if mm else None, 'proposal': prop.get('seed'), 'manifest': (ms or {}).get('seed'),
                     'initialization': init.get('seed'), 'initialization_randomization': (init.get('randomization') or {}).get('seed'),
                     'appearance': (app or {}).get('seed'), 'initialization_appearance': (init.get('appearance') or {}).get('seed')},
            'group': None,
            'arm': 'right' if plan.get('initial') and abs(plan['initial'][0] - (plan.get('initial_targets') or {}).get('idx61_arm_r_joint1', 1e9)) < 1e-9 else ('left' if plan.get('initial') and abs(plan['initial'][0] - (plan.get('initial_targets') or {}).get('idx21_arm_l_joint1', 1e9)) < 1e-9 else None),
            'proposal': {k: prop.get(k) for k in ('machine_dx_m', 'machine_dy_m', 'machine_dyaw_deg')},
            'proposal_config': prop.get('config'),
            'proposal_equal_manifest': ({k: prop.get(k) for k in ('seed', 'machine_dx_m', 'machine_dy_m', 'machine_dyaw_deg')} == (ms or {}).get('proposal')) if ms else None,
            'proposal_equal_initialization_randomization': ({k: prop.get(k) for k in ('seed', 'machine_dx_m', 'machine_dy_m', 'machine_dyaw_deg')} == init.get('randomization')),
            'randomization_config': init.get('randomization_config'),
            'bag_cycle_initialization': init.get('bag_cycle_initialization'),
            'machine': {'world_xyz': m_xyz, 'world_yaw_deg': m_yaw,
                        'offset_from_nominal_m': [m_xyz[i] - NOMINAL_MACHINE[i] for i in range(3)] if m_xyz else None,
                        'implied_reference_xyz': [m_xyz[0] - prop.get('machine_dx_m', 0), m_xyz[1] - prop.get('machine_dy_m', 0), m_xyz[2]] if m_xyz and prop else None,
                        'implied_reference_yaw_deg': norm_deg(m_yaw - prop.get('machine_dyaw_deg', 0)) if m_yaw is not None and prop else None},
            'button_world': init.get('button_world'), 'button_up': init.get('button_up'),
            'bag': {'world': pose(init.get('bag_world')),
                    'geometry': {k: bgr.get(k) for k in ('revision', 'axis', 'width_m', 'depth_m', 'height_m', 'scale_x', 'scale_z')}},
            'cup': {'world': pose(init.get('cup_world')), 'placement_policy': init.get('cup_placement_policy'),
                    'cup_style_in_initialization': (init.get('randomization') or {}).get('cup_style', init.get('cup_style'))},
            'beans': {'bean_paths': len(init.get('bean_paths') or []), 'bag_geometry_bean_count': bgr.get('bean_count'),
                      'bag_geometry_bean_scale': bgr.get('bean_scale'),
                      'beans_initially_in_hopper': init.get('beans_initially_in_hopper'),
                      'beans_initially_in_bag': init.get('beans_initially_in_bag'),
                      'trace_bean_count_values': bean_counts, 'trace_raw_bean_count_values': raw_bean_counts,
                      'bean_assets': [os.path.basename(x) for x in init.get('bean_assets') or []]},
            'machine_hopper_revision': init.get('machine_hopper_revision'),
            'reference_initialization': init.get('reference_initialization'),
            'resize_source_initial': init.get('resize_source_initial'),
            'head_visibility_passed': (init.get('head_visibility') or {}).get('passed'),
            'robot': {'root_xyz': plan.get('root_xyz'), 'yaw': plan.get('yaw'),
                      'initial_targets_hash': sha_obj(plan.get('initial_targets')) if plan.get('initial_targets') else None,
                      'robot_targets_hash': sha_obj(init.get('robot_targets')) if init.get('robot_targets') else None,
                      'robot_targets_n': len(init.get('robot_targets') or {}),
                      'robot_targets': init.get('robot_targets'),
                      'plan_initial_q': plan.get('initial')},
            'plan': {'n_steps': len(steps), 'phases': phases_of(steps), 'step_keys': sorted(steps[0].keys()) if steps else None,
                     'q_len': len(steps[0]['q']) if steps and 'q' in steps[0] else None,
                     'physics_hz': plan.get('physics_hz'), 'control_hz': plan.get('control_hz'),
                     'two_press_wait_seconds': plan.get('two_press_wait_seconds'),
                     'horizontal_height': plan.get('horizontal_height'),
                     'button_travel_min': min(travel) if travel else None, 'button_travel_max': max(travel) if travel else None,
                     'task': plan.get('task'), 'scope': plan.get('scope'), 'keys': sorted(plan.keys())},
            'route': {'yaw': route.get('yaw'), 'n_nodes': len(route.get('nodes') or []), 'opening': route.get('opening'),
                      'vertical_travel': route.get('vertical_travel'), 'horizontal_height': route.get('horizontal_height'),
                      'keys': sorted(route.keys())},
            'certificate': {'passed': cert.get('passed'), 'samples': cert.get('samples'), 'commands': cert.get('commands'),
                            'n_fingerprints': len(cert.get('fingerprints') or {}), 'keys': sorted(cert.keys())},
            'result': {'passed': res.get('passed'), 'failure': res.get('failure'), 'checks': checks,
                       'n_checks': len(checks), 'all_checks_true': all(v is True for v in checks.values()) if checks else None,
                       'false_checks': [k for k, v in checks.items() if v is not True],
                       'manifest_status': (ms or {}).get('status'), 'manifest_failure': (ms or {}).get('failure'),
                       'manifest_checks_equal': ((ms or {}).get('checks') == checks) if ms else None,
                       'events': [{k: e.get(k) for k in ('event', 'time_s', 'step', 'button_displacement_m', 'button_pressed', 'dispensing', 'bean_count', 'raw_bean_count')} for e in events],
                       'manifest_events_equal': ((ms or {}).get('events') == events) if ms else None,
                       'n_events': len(events),
                       'press_displacement_min_m': min(disp) if disp else None,
                       'press_displacement_max_m': max(disp) if disp else None,
                       'press_displacement_first_m': disp[0] if disp else None,
                       'pressed_control_steps': pressed_steps, 'dispensing_control_steps': disp_steps,
                       'particles_final': res.get('particles'), 'particles_trace_max': max(parts) if parts else None,
                       'n_button_contacts': len(res.get('button_contacts') or []),
                       'n_unexpected': len(res.get('unexpected') or []),
                       'simulation_seconds': res.get('simulation_seconds'), 'video_capture_fps': res.get('video_capture_fps'),
                       'appearance_only_changes': res.get('appearance_only_changes'),
                       'trace_len': len(trace), 'trace_keys': sorted(t0.keys()), 'trace_dt_values': dts,
                       'trace_t_first': ts[0] if ts else None, 'trace_t_last': ts[-1] if ts else None,
                       'trace_has_joint_positions': any(k in t0 for k in ('all_joint_positions', 'q', 'actual_arm')),
                       'keys': sorted(res.keys()), 'scope': res.get('scope')},
            'initialization_checks': init.get('checks'), 'initialization_passed': init.get('passed'),
            'initialization_keys': sorted(init.keys()),
            'appearance': appearance_compact(app),
            'appearance_json_equals_initialization_appearance': (app == init.get('appearance')) if app is not None and init.get('appearance') is not None else None,
            'appearance_keys': sorted(app.keys()) if app else None,
            'capture_events': capture_events_info(f'{d}/motion/capture_events.jsonl'),
            'videos_ready': vready,
            'manifest': {k: (ms or {}).get(k) for k in ('source_run', 'status', 'gpu', 'duration', 'started', 'ended', 'source')},
            'video': video_block(r['videos']),
            'sha256_manifest': {k: (verif.get(p) or {}).get('sha256') for k, p in r['videos'].items()},
            'sha256_manifest_source': 'coffee_updates_0925_verification.json' if verif else None,
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
        'manifest_not_in_data_list': [x for x in mruns if x not in listed],
        'data_list_not_in_manifest': [x for x in listed if x not in mruns],
        'manifest_top': {k: v for k, v in manifest.items() if k != 'episodes'},
        'appearance_varying': varying, 'appearance_constant_keys': sorted(constant.keys()),
        'appearance_constant_values': {k: (v if not isinstance(v, str) or len(v) < 200 else v[:200] + '...') for k, v in constant.items()},
        'n_appearance_files': len(flats),
    }
    save_rows(TASK, rows, extra)
    print('rows', len(rows), 'extra dirs', extra['dirs_not_in_data_list'])


if __name__ == '__main__':
    main()
