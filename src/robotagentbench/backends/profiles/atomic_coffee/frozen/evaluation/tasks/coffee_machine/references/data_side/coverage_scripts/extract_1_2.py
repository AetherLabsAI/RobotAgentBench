#!/usr/bin/env python3
"""1.2 倾倒咖啡豆 (coffee_pour_final50_20260924): one row per episode in data_list.json."""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa

TASK = '1.2'
DS = os.path.join(ROOT, 'coffee_pour_final50_20260924')
EXPECTED = ['initialization.json', 'plan.json', 'result.json', 'video_manifest.json', 'initial.png', 'initial_left.png',
            'initial_right.png', 'pour_complete.mp4', 'pour_complete_left.mp4', 'pour_complete_right.mp4']


def pose(m):
    if not m:
        return None
    hz, trans = local_z_heading(m)
    hx, _ = local_x_heading(m)
    return {'xyz': trans, 'localZ_heading_deg': hz, 'localX_heading_deg': hx, 'tilt_deg': tilt_deg(m)}


def main():
    recs = records(TASK)
    manifest = load(os.path.join(DS, 'manifest.json'))
    mruns = {s['run']: s for s in manifest}
    vids = []
    for r in recs:
        vids += list(r['videos'].values())
    ffprobe_many(vids)
    rows, flats = [], []
    for r in recs:
        d = episode_dir(r)
        eid = os.path.basename(d)
        ms = mruns.get(eid)
        files = sorted(os.listdir(d)) if os.path.isdir(d) else []
        missing = [f for f in EXPECTED if f not in files]
        init = load_opt(f'{d}/initialization.json') or {}
        plan = load_opt(f'{d}/plan.json') or {}
        res = load_opt(f'{d}/result.json') or {}
        vm = load_opt(f'{d}/video_manifest.json') or {}
        # original run directory (first local_artifact that is not under the dataset)
        orig_motion = next((p for p in r['local_artifacts'] if not p.startswith(DS)), None)
        orig_ep = os.path.dirname(orig_motion) if orig_motion else None
        orig_run = os.path.dirname(orig_ep) if orig_ep else None
        batch = load_opt(os.path.join(orig_run, 'batch.json')) if orig_run else None
        orig_app = load_opt(os.path.join(orig_ep, 'initial', 'appearance.json')) if orig_ep else None
        orig_init = load_opt(os.path.join(orig_ep, 'initial', 'initialization.json')) if orig_ep else None
        plan_source = 'dataset/plan.json' if plan else None
        orig_exec = load_opt(os.path.join(orig_motion, 'executed_plan.json')) if orig_motion else None
        orig_plan = load_opt(os.path.join(orig_motion, 'plan.json')) if orig_motion else None
        orig_cert = load_opt(os.path.join(orig_motion, 'certificate.json')) if orig_motion else None
        if not plan and orig_exec:
            plan = orig_exec
            plan_source = 'original_run/motion/executed_plan.json (dataset plan.json absent)'
        app = init.get('appearance')
        rnd = init.get('randomization') or {}
        mw = init.get('machine_world')
        m_yaw, m_xyz = machine_yaw_from_matrix(mw) if mw else (None, None)
        trace = res.get('trace') or []
        t0 = trace[0] if trace else {}
        steps = plan.get('steps') or []
        bgr = init.get('bag_geometry_revision') or {}
        checks = res.get('checks') or {}
        mt = plan.get('motion_timing') or {}
        sg = plan.get('side_grasp') or {}
        seed_dir = None
        mm = re.search(r'seed_(\d+)', eid)
        if mm:
            seed_dir = int(mm.group(1))
        vmf = vm.get('frames') or []
        ep_index = None
        mm = re.search(r'episode_(\d+)$', eid)
        if mm:
            ep_index = int(mm.group(1))
        batch_result = None
        if batch:
            for br in batch.get('results') or []:
                if br.get('episode') == ep_index:
                    batch_result = {k: br.get(k) for k in ('episode', 'seed', 'status', 'phase', 'failure', 'in_hopper', 'in_bag', 'outside')}
        row = {
            'task': TASK, 'episode_id': eid, 'dir': d, 'title': r['title'], 'data_list_id': r['id'],
            'source_status': r['source_status'], 'source_exists_on_202': r['source_exists_on_202'],
            'source_path': r['source_path'], 'original_episode_dir': orig_ep,
            'original_episode_dir_exists': bool(orig_ep and os.path.isdir(orig_ep)),
            'original_episode_dir_files': sorted(os.listdir(orig_ep)) if orig_ep and os.path.isdir(orig_ep) else None,
            'original_initial_dir_files': sorted(os.listdir(os.path.join(orig_ep, 'initial'))) if orig_ep and os.path.isdir(os.path.join(orig_ep, 'initial')) else None,
            'original_motion_json_files': sorted(f for f in os.listdir(orig_motion) if f.endswith(('.json', '.jsonl', '.py', '.usda', '.mp4'))) if orig_motion and os.path.isdir(orig_motion) else None,
            'in_dataset_manifest': ms is not None,
            'missing_files': missing, 'files': files,
            'group': {'manifest': (ms or {}).get('group'), 'title': r['title'].split(' · ')[1] if r['title'].count(' · ') >= 2 else None,
                      'batch_prefix': re.sub(r'(_seed_\d+)?_episode_\d+$', '', eid)},
            'seed': {'dir_name': seed_dir, 'initialization': init.get('seed'), 'plan': plan.get('seed'), 'result': res.get('seed'),
                     'appearance': (app or {}).get('seed'), 'batch_json_seed': (batch or {}).get('seed'),
                     'batch_json_episode_seed': (batch_result or {}).get('seed')},
            'batch_json': {'random_ranges': (batch or {}).get('random_ranges'), 'bag_width_m': (batch or {}).get('bag_width_m'),
                           'bag_depth_m': (batch or {}).get('bag_depth_m'), 'episodes': (batch or {}).get('episodes'),
                           'attempted': (batch or {}).get('attempted'), 'physical_successes': (batch or {}).get('physical_successes'),
                           'reference_layout': (batch or {}).get('reference_layout'), 'scope': (batch or {}).get('scope'),
                           'episode_result': batch_result} if batch else None,
            'bag_cycle_initialization': init.get('bag_cycle_initialization'),
            'arm': 'right' if plan.get('initial') and plan.get('initial_targets') and abs(plan['initial'][0] - plan['initial_targets'].get('idx61_arm_r_joint1', 1e9)) < 1e-9 else ('left' if plan.get('initial') and abs(plan['initial'][0] - (plan.get('initial_targets') or {}).get('idx21_arm_l_joint1', 1e9)) < 1e-9 else None),
            'randomization': rnd,
            'randomization_equal_init_plan_result_manifest': (rnd == plan.get('randomization') == res.get('randomization') == (ms or {}).get('randomization')),
            'randomization_equal_init_result_manifest': (rnd == res.get('randomization') == (ms or {}).get('randomization')),
            'cup_style': rnd.get('cup_style'),
            'machine': {'offset_m': rnd.get('machine_offset_m'), 'yaw_deg': rnd.get('machine_yaw_deg'),
                        'world_xyz': m_xyz, 'world_yaw_deg': m_yaw,
                        'offset_from_nominal_m': [m_xyz[i] - NOMINAL_MACHINE[i] for i in range(3)] if m_xyz else None},
            'bag': {'offset_m': rnd.get('bag_offset_m'), 'yaw_deg': rnd.get('bag_yaw_deg'),
                    'world': pose(init.get('bag_world')), 'result_initial': pose((res.get('initial') or {}).get('bag')),
                    'result_final': pose((res.get('final') or {}).get('bag')),
                    'geometry': {k: bgr.get(k) for k in ('revision', 'axis', 'width_m', 'depth_m', 'height_m', 'original_width_m', 'original_depth_m', 'scale_x', 'scale_z', 'normal_wall_thickness_m')},
                    'collision_variant_files': sorted(((plan.get('bag_collision_variant') or {}).get('sha256') or {}).keys())},
            'cup': {'world': pose(init.get('cup_world')), 'result_initial': pose((res.get('initial') or {}).get('cup')),
                    'result_final': pose((res.get('final') or {}).get('cup')),
                    'placement_policy': init.get('cup_placement_policy')},
            'beans': {'bean_paths': len(init.get('bean_paths') or []), 'bag_geometry_bean_count': bgr.get('bean_count'),
                      'bag_geometry_bean_scale': bgr.get('bean_scale'), 'plan_bean_count': plan.get('bean_count'),
                      'plan_bean_geometry_scale': plan.get('bean_geometry_scale'), 'bean_assets': [os.path.basename(x) for x in init.get('bean_assets') or []],
                      'result_initial': {k: (res.get('initial') or {}).get(k) for k in ('in_bag', 'in_hopper', 'outside')},
                      'result_final': {k: (res.get('final') or {}).get(k) for k in ('in_bag', 'in_hopper', 'outside')},
                      'bean_outcome': res.get('bean_outcome'), 'manifest_bean_outcome': (ms or {}).get('bean_outcome'),
                      'strict_pour_quality': res.get('strict_pour_quality')},
            'machine_hopper_revision': init.get('machine_hopper_revision'),
            'robot': {'root_xyz': plan.get('root_xyz'), 'yaw': plan.get('yaw'),
                      'initial_targets_hash': sha_obj(plan.get('initial_targets')) if plan.get('initial_targets') else None,
                      'robot_targets_hash': sha_obj(init.get('robot_targets')) if init.get('robot_targets') else None,
                      'robot_targets_n': len(init.get('robot_targets') or {}),
                      'robot_targets': init.get('robot_targets'),
                      'plan_initial_q': plan.get('initial'),
                      'trace0_actual_arm': t0.get('actual_arm'), 'trace0_bodylift': t0.get('bodylift'), 'trace0_waist': t0.get('waist')},
            'original_run': {'executed_plan_present': orig_exec is not None, 'plan_present': orig_plan is not None,
                             'executed_plan_equals_dataset_plan': (orig_exec == plan) if orig_exec is not None else None,
                             'executed_plan_n_steps': len((orig_exec or {}).get('steps') or []) if orig_exec else None,
                             'certificate_present': orig_cert is not None,
                             'certificate_passed': (orig_cert or {}).get('passed'),
                             'certificate_execution_step_count': (orig_cert or {}).get('execution_step_count'),
                             'capture_events': capture_events_info(os.path.join(orig_motion, 'capture_events.jsonl')) if orig_motion else None,
                             'has_physics_ticks': os.path.exists(os.path.join(orig_motion, 'physics_ticks.jsonl.gz')) if orig_motion else None,
                             'has_runtime_usda': os.path.exists(os.path.join(orig_motion, 'runtime.usda')) if orig_motion else None,
                             'has_initial_usda': os.path.exists(os.path.join(orig_ep, 'initial', 'initial.usda')) if orig_ep else None},
            'plan_source': plan_source,
            'plan': {'n_steps': len(steps), 'phases': phases_of(steps), 'step_keys': sorted(steps[0].keys()) if steps else None,
                     'q_len': len(steps[0]['q']) if steps and 'q' in steps[0] else None,
                     'left_q_constant': len(set(tuple(s.get('left_q') or []) for s in steps)) == 1 if steps else None,
                     'control_hz': mt.get('control_hz'), 'motion_timing_policy': mt.get('policy'),
                     'candidate_seconds': mt.get('candidate_seconds'), 'original_seconds': mt.get('original_seconds'),
                     'video_capture_control_stride': plan.get('video_capture_control_stride'),
                     'video_capture_fps': plan.get('video_capture_fps'),
                     'pour_tilt_duration_s': plan.get('pour_tilt_duration_s'),
                     'intermediate_hold_removal': {k: (plan.get('intermediate_hold_removal') or {}).get(k) for k in ('original_commands', 'commands', 'removed_seconds')},
                     'side_grasp': {k: sg.get(k) for k in ('policy', 'height_m', 'angle_deg', 'side', 'roll_deg', 'depth_m', 'center_shift_m', 'loaded_opening', 'hold_opening')},
                     'keys': sorted(plan.keys()), 'stage': plan.get('stage'), 'scope': plan.get('scope')},
            'result': {'passed': res.get('passed'), 'failure': res.get('failure'), 'checks': checks,
                       'n_checks': len(checks), 'all_checks_true': all(v is True for v in checks.values()) if checks else None,
                       'false_checks': [k for k, v in checks.items() if v is not True],
                       'manifest_passed': (ms or {}).get('passed'),
                       'manifest_checks_equal': ((ms or {}).get('checks') == checks) if ms else None,
                       'bag_return_position_error_m': (res.get('bag_return') or {}).get('position_error_m'),
                       'bag_return_rotation_error_rad': (res.get('bag_return') or {}).get('rotation_error_rad'),
                       'lift_position_drift_m': res.get('lift_position_drift_m'),
                       'lift_rotation_drift_rad': res.get('lift_rotation_drift_rad'),
                       'hold_position_drift_m': res.get('hold_position_drift_m'),
                       'max_bilateral_contact_frames': res.get('max_bilateral_contact_frames'),
                       'recovered_contact_events': res.get('recovered_contact_events'),
                       'recorded_overlap_events': res.get('recorded_overlap_events'),
                       'spill_contact_events': res.get('spill_contact_events'),
                       'n_unexpected_contacts': len(res.get('unexpected_contacts') or []),
                       'motion_completed': res.get('motion_completed'), 'end_to_end': res.get('end_to_end'),
                       'acceptance_scope': res.get('acceptance_scope'), 'contact_acceptance': res.get('contact_acceptance'),
                       'trace_len': len(trace), 'trace_keys': sorted(t0.keys()),
                       'trace_last_step': trace[-1].get('step') if trace else None,
                       'trace_has_all_joint_positions': 'all_joint_positions' in t0,
                       'trace_joint_fields': [k for k in ('actual_arm', 'gripper_joints', 'bodylift', 'waist', 'opening') if k in t0],
                       'initial_keys': sorted((res.get('initial') or {}).keys()),
                       'keys': sorted(res.keys())},
            'initialization_checks': init.get('checks'), 'initialization_passed': init.get('passed'),
            'initialization_keys': sorted(init.keys()),
            'appearance': appearance_compact(app),
            'appearance_in_initialization_equals_original_appearance_json': (app == orig_app) if (app is not None and orig_app is not None) else None,
            'initialization_equals_original_initialization_json': (init == orig_init) if (init and orig_init is not None) else None,
            'video_manifest': {'control_stride': vm.get('control_stride'), 'simulation_seconds': vm.get('simulation_seconds'),
                               'actual_render_fps': vm.get('actual_render_fps'), 'synchronized': vm.get('synchronized'),
                               'result_passed': vm.get('result_passed'), 'n_frames': len(vmf),
                               'last_frame_sim_time': vmf[-1].get('simulation_time') if vmf else None,
                               'n_frames_per_view': {k: len(v.get('frames') or []) for k, v in (vm.get('views') or {}).items()},
                               'source': vm.get('source')},
            'manifest_duration': (ms or {}).get('duration'), 'manifest_view_durations': (ms or {}).get('view_durations'),
            'video': video_block(r['videos']),
            'sha256_manifest': None,
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
        'manifest_sample_keys': sorted(manifest[0].keys()) if manifest else None,
        'appearance_varying': varying, 'appearance_constant_keys': sorted(constant.keys()),
        'appearance_constant_values': {k: (v if not isinstance(v, str) or len(v) < 200 else v[:200] + '...') for k, v in constant.items()},
        'n_appearance_files': len(flats),
    }
    save_rows(TASK, rows, extra)
    print('rows', len(rows), 'extra dirs', extra['dirs_not_in_data_list'])


if __name__ == '__main__':
    main()
