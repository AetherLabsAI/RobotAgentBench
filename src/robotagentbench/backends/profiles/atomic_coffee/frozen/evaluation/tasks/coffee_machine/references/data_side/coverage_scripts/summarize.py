#!/usr/bin/env python3
"""Merge per-task rows into coverage_rows.json and compute the statistics used in coverage_summary.md.
Writes coverage_work/stats.json (all numbers) and coverage_work/summary_tables.md (auto-generated tables).
The hand-written narrative in coverage_summary.md quotes these numbers."""
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT_DIR, SCRATCH, NOMINAL_MACHINE  # noqa

C = collections.Counter
NOMINAL_BAG = (2.5659632, -0.1072122, 0.92)   # bag position in every 1.1 trace whose plan has bag_offset_m == 0
NOMINAL_BAG_HEADING = -8.0                      # local-Z heading of the bag at bag_yaw_deg == 0


def L(task):
    return json.load(open(os.path.join(OUT_DIR, f'rows_{task}.json')))


def X(task):
    return json.load(open(os.path.join(OUT_DIR, f'extra_{task}.json')))


def st(vals, nd=4):
    v = [float(x) for x in vals if x is not None]
    if not v:
        return None
    mean = sum(v) / len(v)
    sd = (sum((x - mean) ** 2 for x in v) / len(v)) ** 0.5
    return {'n': len(v), 'min': round(min(v), nd), 'max': round(max(v), nd), 'mean': round(mean, nd), 'sd': round(sd, nd)}


def cnt(vals):
    c = C('null' if v is None else (v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)) for v in vals)
    return dict(sorted(c.items(), key=lambda kv: (-kv[1], kv[0])))


def fmt_stat(s, unit=''):
    if s is None:
        return 'n/a'
    return f"n={s['n']} min={s['min']} max={s['max']} mean={s['mean']} sd={s['sd']}{unit}"


def fmt_cnt(c):
    return ', '.join(f'{k}: {v}' for k, v in c.items())


def table(headers, rows):
    out = ['| ' + ' | '.join(headers) + ' |', '|' + '|'.join('---' for _ in headers) + '|']
    for r in rows:
        out.append('| ' + ' | '.join('' if x is None else str(x) for x in r) + ' |')
    return '\n'.join(out)


def r4(x, n=4):
    return None if x is None else round(float(x), n)


def video_stats(rows):
    out = {}
    for k in ('head', 'left', 'right'):
        out[k] = {
            'resolution': cnt([f"{r['video'][k].get('width')}x{r['video'][k].get('height')}" for r in rows]),
            'fps': cnt([r['video'][k].get('r_frame_rate') for r in rows]),
            'codec': cnt([f"{r['video'][k].get('codec')}/{r['video'][k].get('pix_fmt')}" for r in rows]),
            'nb_frames': st([r['video'][k].get('nb_frames') for r in rows], 1),
            'duration_s': st([r['video'][k].get('duration_s') for r in rows], 3),
            'bytes': st([r['video'][k].get('bytes') for r in rows], 0),
            'missing': [r['episode_id'] for r in rows if not r['video'][k].get('exists')],
        }
    out['three_views_same_frame_count'] = cnt([len({r['video'][k].get('nb_frames') for k in ('head', 'left', 'right')}) == 1 for r in rows])
    return out


def appearance_stats(rows):
    apps = [r['appearance'] for r in rows if r.get('appearance')]
    if not apps:
        return None
    ce_i = [x for a in apps for x in a['ceiling']['intensity_scale']]
    ce_t = [x for a in apps for x in a['ceiling']['color_temperature_k']]
    da_i = [x for a in apps for x in a['daylight']['intensity_scale']]
    da_t = [x for a in apps for x in a['daylight']['color_temperature_k']]
    mats = collections.defaultdict(list)
    for a in apps:
        for k, v in a['materials'].items():
            mats[k].append(v)
    plants = C()
    for a in apps:
        for slot, p in (a.get('plants') or {}).items():
            plants[p] += 1
    return {
        'n': len(apps),
        'policy': cnt([a['policy'] for a in apps]), 'scope': cnt([a['scope'] for a in apps]),
        'seed_equals_episode_seed': None,
        'palette': cnt([a['palette'] for a in apps]),
        'lighting_scheme': cnt([a['lighting_scheme'] for a in apps]),
        'environment_style': cnt([a['environment_style'] for a in apps]),
        'floor_texture_asset': cnt([a['floor_texture_asset'] for a in apps]),
        'palette_env_floor_combos': cnt([f"{a['palette']} / {a['environment_style']} / {a['floor_texture_asset']}" for a in apps]),
        'central_table_option': cnt([a['central_table_option'] for a in apps]),
        'stove_rule': cnt([a['stove_rule'] for a in apps]),
        'stove_lower': cnt([a['stove_lower'] for a in apps]),
        'stove_upper': cnt([a['stove_upper'] for a in apps]),
        'furniture_looks': {k: cnt(v) for k, v in mats.items()},
        'ceiling_lights_per_episode': cnt([a['ceiling']['n'] for a in apps]),
        'daylight_lights_per_episode': cnt([a['daylight']['n'] for a in apps]),
        'ceiling_intensity_scale': st(ce_i, 5), 'ceiling_color_temperature_k': st(ce_t, 1),
        'daylight_intensity_scale': st(da_i, 5), 'daylight_color_temperature_k': st(da_t, 1),
        'ceiling_distinct_temperature_per_episode': cnt([len(a['ceiling']['color_temperature_k']) for a in apps]),
        'planned_artwork': cnt([a['artwork'] for a in apps]),
        'planned_lamp_mode': cnt([a['lamp_mode'] for a in apps]),
        'planned_lamp': cnt([a['lamp'] for a in apps]),
        'planned_plants_distinct_species': len(plants),
        'harmony_status': cnt([a['harmony_status'] for a in apps]),
        'source_snapshot_sha256': cnt([a['source_snapshot_sha256'] for a in apps]),
        'distinct_full_appearance_records': len({json.dumps({k: v for k, v in a.items() if k not in ('seed', 'gen_seed')}, sort_keys=True) for a in apps}),
    }


def dup(vals):
    return {str(k): v for k, v in C(vals).items() if v > 1 and k is not None}


# ------------------------------------------------------------------ 1.1
def s11():
    rows = L('1.1')
    ex = X('1.1')
    A = [r for r in rows if 'proposal.json' not in r['missing_files']]
    B = [r for r in rows if 'proposal.json' in r['missing_files']]
    S = {'n': len(rows), 'extra': {k: ex[k] for k in ('n_dirs', 'dirs_not_in_data_list', 'data_list_not_in_dirs', 'manifest_not_in_data_list', 'data_list_not_in_manifest', 'manifest_summary')}}
    S['arm'] = cnt([r['arm']['manifest'] for r in rows])
    S['group'] = cnt([r['group'] for r in rows])
    S['style'] = cnt([r['cup_style']['plan'] for r in rows])
    S['arm_x_style'] = cnt([f"{r['arm']['manifest']}/{r['cup_style']['plan']}" for r in rows])
    S['arm_x_group'] = cnt([f"{r['arm']['manifest']}/{r['group']}" for r in rows])
    S['missing_files'] = cnt([','.join(r['missing_files']) or 'none' for r in rows])
    S['population'] = {'A_with_proposal_json': [r['episode_id'] for r in A], 'B_without_proposal_json': [r['episode_id'] for r in B]}
    S['source_run_prefix'] = cnt([(r['source_run_name'] or '').rsplit('_seed_', 1)[0] if '_seed_' in (r['source_run_name'] or '') else (r['source_run_name'] or '')[:17] for r in rows])

    def block(rr):
        return {
            'n': len(rr),
            'arm': cnt([r['arm']['manifest'] for r in rr]), 'style': cnt([r['cup_style']['plan'] for r in rr]), 'group': cnt([r['group'] for r in rr]),
            'handle_world_yaw_deg': st([r['handle_world_yaw_deg']['trace0_handle_axis'] for r in rr], 2),
            'cup_start_x': st([r['cup_start_xyz']['result_initial_cup'][0] for r in rr]),
            'cup_start_y': st([r['cup_start_xyz']['result_initial_cup'][1] for r in rr]),
            'cup_start_z': st([r['cup_start_xyz']['result_initial_cup'][2] for r in rr], 5),
            'machine_dx': st([r['machine']['offset_from_nominal_m'][0] for r in rr]),
            'machine_dy': st([r['machine']['offset_from_nominal_m'][1] for r in rr]),
            'machine_dz': st([r['machine']['offset_from_nominal_m'][2] for r in rr]),
            'machine_yaw_deg': st([r['machine']['trace0_yaw_deg'] for r in rr], 3),
            'bag_dx': st([r['bag']['trace0_xyz'][0] - NOMINAL_BAG[0] for r in rr]),
            'bag_dy': st([r['bag']['trace0_xyz'][1] - NOMINAL_BAG[1] for r in rr]),
            'bag_yaw_deg': st([r['bag']['plan_yaw_deg'] for r in rr], 3),
            'final_to_goal_xy_m': st([r['result']['final_to_goal_xy_m'] for r in rr], 5),
            'final_to_goal_xyz_m': st([r['result']['final_to_goal_xyz_m'] for r in rr], 5),
            'final_minus_goal_z_m': st([r['result']['final_minus_goal_z_m'] for r in rr], 5),
            'plan_steps': st([r['plan']['n_steps'] for r in rr], 1),
            'video_duration_s': st([r['video']['head']['duration_s'] for r in rr], 2),
            'max_bilateral_contact_frames': cnt([r['result']['max_bilateral_contact_frames'] for r in rr]),
        }
    S['all'] = block(rows)
    S['A'] = block(A)
    S['B'] = block(B)
    S['left'] = block([r for r in rows if r['arm']['manifest'] == 'left'])
    S['right'] = block([r for r in rows if r['arm']['manifest'] == 'right'])
    S['A_cup_xy_offset_x'] = st([r['cup_xy_offset_m'][0] for r in A])
    S['A_cup_xy_offset_y'] = st([r['cup_xy_offset_m'][1] for r in A])
    S['A_angle_bin'] = cnt([r['angle_bin'] for r in A])
    S['A_studies'] = {r['episode_id']: (r['plan_randomization_raw'] or {}).get('study') for r in A if (r['plan_randomization_raw'] or {}).get('study')}
    S['B_offset_m_x'] = st([r['plan_randomization_raw']['offset_m'][0] for r in B])
    S['B_offset_m_y'] = st([r['plan_randomization_raw']['offset_m'][1] for r in B])
    S['B_yaw_delta_deg'] = st([r['plan_randomization_raw']['yaw_delta_deg'] for r in B], 2)
    S['B_randomization_keys'] = cnt([sorted((r['plan_randomization_raw'] or {}).keys()) for r in B])
    S['A_randomization_keys'] = cnt([sorted((r['plan_randomization_raw'] or {}).keys()) for r in A])
    S['seed_sources_agree'] = cnt([len({v for k, v in r['seed'].items() if v is not None and k != 'from_source_name'}) == 1 for r in rows])
    S['duplicate_seeds'] = dup([r['seed']['plan_randomization'] for r in rows])
    S['seeds_A'] = sorted(r['seed']['plan_randomization'] for r in A)
    S['seeds_B'] = sorted(r['seed']['plan_randomization'] for r in B)
    S['robot'] = {'root_xyz': cnt([r['robot']['root_xyz'] for r in rows]), 'yaw': cnt([r['robot']['yaw'] for r in rows]),
                  'initial_targets_hash': cnt([r['robot']['initial_targets_hash'] for r in rows]),
                  'initial_state_reference': cnt([r['robot']['initial_state_reference'] for r in rows]),
                  'initial_manifest_sha256': cnt([r['robot']['initial_manifest_sha256'] for r in rows]),
                  'trace0_joint_spread_max_rad': max(max(r['robot']['trace0_all_joint_positions'][k] for r in rows) - min(r['robot']['trace0_all_joint_positions'][k] for r in rows) for k in rows[0]['robot']['trace0_all_joint_positions']),
                  'initial_targets': json.load(open(rows[0]['dir'] + '/plan.json'))['initial_targets']}
    S['checks'] = {'passed': cnt([r['result']['passed'] for r in rows]), 'n_checks': cnt([r['result']['n_checks'] for r in rows]),
                   'all_true': cnt([r['result']['all_checks_true'] for r in rows]), 'failure': cnt([r['result']['failure'] for r in rows]),
                   'names_16': list(B[0]['result']['checks'].keys()) if B else None,
                   'extra_in_17': sorted(set(A[0]['result']['checks']) - set(B[0]['result']['checks'])) if A and B else None,
                   'n_checks_by_arm': cnt([f"{r['arm']['manifest']}:{r['result']['n_checks']}" for r in rows]),
                   'manifest_checks_count_equal': cnt([r['result']['manifest_checks_count'] == r['result']['n_checks'] for r in rows]),
                   'unexpected_contacts': cnt([r['result']['n_unexpected_contacts'] for r in rows]),
                   'initial_yaw_error_deg': st([r['result']['initial_yaw_error_deg'] for r in rows], 6),
                   'fk_error': st([r['result']['fk_error'] for r in rows], 9)}
    S['preflight'] = {'passed': cnt([r['preflight']['passed'] for r in rows]), 'violations': cnt([r['preflight']['n_violations'] for r in rows]),
                      'clearance_m': cnt([r['preflight']['clearance_m'] for r in rows]), 'mode': cnt([r['preflight']['mode'] for r in rows]),
                      'model_directory': cnt([r['preflight']['model_directory'] for r in rows]),
                      'checked_samples': st([r['preflight']['checked_samples'] for r in rows], 1)}
    S['plan'] = {'executed_equals_plan': cnt([r['plan']['executed_equals_plan'] for r in rows]),
                 'step_keys': cnt([r['plan']['step_keys'] for r in rows]), 'q_len': cnt([r['plan']['q_len'] for r in rows]),
                 'phase_sequences': cnt([[p[0] for p in r['plan']['phases']] for r in rows]),
                 'trace_len_equals_plan_steps': cnt([r['result']['trace_len'] == r['plan']['n_steps'] for r in rows]),
                 'trace_dt': cnt([r['result']['trace_dt_values'] for r in rows]),
                 'n_joints_in_trace': cnt([r['result']['n_joints_recorded'] for r in rows]),
                 'frames_minus_steps': cnt([r['video']['head']['nb_frames'] - r['plan']['n_steps'] for r in rows])}
    S['capture_events'] = {'n_events': st([r['capture_events']['n_events'] for r in rows], 1),
                           'first_sim_time_s': cnt([round(r['capture_events']['first_sim_time_s'], 3) for r in rows]),
                           'last_sim_time_s': st([r['capture_events']['last_sim_time_s'] for r in rows], 2)}
    S['video'] = video_stats(rows)
    S['sha256_recorded'] = cnt([all(r['sha256_manifest'].values()) for r in rows])
    S['appearance'] = appearance_stats(rows)
    S['appearance']['seed_equals_episode_seed'] = cnt([r['appearance']['seed'] == r['seed']['plan_randomization'] for r in rows])
    S['appearance_varying_key_count'] = len(ex['appearance_varying'])
    S['appearance_constant_key_count'] = len(ex['appearance_constant_keys'])
    tab = table(['episode', 'source run', 'seed', 'arm', 'group', 'style', 'handle yaw°', 'cup x', 'cup y', 'mach dx', 'mach dy', 'mach yaw°', 'bag dx', 'bag dy', 'bag yaw°', 'palette', 'lighting', 'steps', 'video s', 'final→goal xy mm', 'checks'],
                [[r['episode_id'], r['source_run_name'], r['seed']['plan_randomization'], r['arm']['manifest'], r['group'], r['cup_style']['plan'],
                  r4(r['handle_world_yaw_deg']['trace0_handle_axis'], 2), r4(r['cup_start_xyz']['result_initial_cup'][0]), r4(r['cup_start_xyz']['result_initial_cup'][1]),
                  r4(r['machine']['offset_from_nominal_m'][0]), r4(r['machine']['offset_from_nominal_m'][1]), r4(r['machine']['trace0_yaw_deg'], 2),
                  r4(r['bag']['plan_offset_m'][0]), r4(r['bag']['plan_offset_m'][1]), r4(r['bag']['plan_yaw_deg'], 2),
                  r['appearance']['palette'], r['appearance']['lighting_scheme'], r['plan']['n_steps'], r['video']['head']['duration_s'],
                  r4(r['result']['final_to_goal_xy_m'] * 1000, 2), f"{sum(1 for v in r['result']['checks'].values() if v is True)}/{r['result']['n_checks']}"] for r in rows])
    return S, tab, rows


# ------------------------------------------------------------------ 1.2
def s12():
    rows = L('1.2')
    ex = X('1.2')
    S = {'n': len(rows), 'extra': {k: ex[k] for k in ('n_dirs', 'dirs_not_in_data_list', 'data_list_not_in_dirs', 'manifest_not_in_data_list', 'data_list_not_in_manifest')}}
    S['group'] = cnt([r['group']['manifest'] for r in rows])
    S['batch_prefix'] = cnt([r['group']['batch_prefix'] for r in rows])
    S['group_x_batch_prefix_x_batchjson_ranges'] = cnt([f"{r['group']['manifest']} | {r['group']['batch_prefix']} | {json.dumps((r['batch_json'] or {}).get('random_ranges'))}" for r in rows])
    S['initialization_bag_cycle_ranges'] = cnt([(r['bag_cycle_initialization'] or {}).get('ranges') for r in rows])
    S['arm'] = cnt([r['arm'] for r in rows])
    S['style'] = cnt([r['cup_style'] for r in rows])
    S['group_x_style'] = cnt([f"{r['group']['manifest']}/{r['cup_style']}" for r in rows])
    S['missing_files'] = cnt([','.join(r['missing_files']) or 'none' for r in rows])
    S['plan_source'] = cnt([r['plan_source'] for r in rows])

    def block(rr):
        return {
            'n': len(rr), 'style': cnt([r['cup_style'] for r in rr]),
            'machine_dx': st([r['machine']['offset_m'][0] for r in rr]), 'machine_dy': st([r['machine']['offset_m'][1] for r in rr]),
            'machine_dz': st([r['machine']['offset_from_nominal_m'][2] for r in rr]),
            'machine_yaw_deg': st([r['machine']['yaw_deg'] for r in rr], 3),
            'machine_world_x': st([r['machine']['world_xyz'][0] for r in rr]), 'machine_world_y': st([r['machine']['world_xyz'][1] for r in rr]),
            'bag_dx': st([r['bag']['offset_m'][0] for r in rr]), 'bag_dy': st([r['bag']['offset_m'][1] for r in rr]),
            'bag_yaw_deg': st([r['bag']['yaw_deg'] for r in rr], 3),
            'bag_world_x': st([r['bag']['world']['xyz'][0] for r in rr]), 'bag_world_y': st([r['bag']['world']['xyz'][1] for r in rr]),
            'bag_world_z': st([r['bag']['world']['xyz'][2] for r in rr], 5),
            'cup_world_x': st([r['cup']['world']['xyz'][0] for r in rr]), 'cup_world_y': st([r['cup']['world']['xyz'][1] for r in rr]),
            'cup_world_z': st([r['cup']['world']['xyz'][2] for r in rr], 5),
            'beans_in_hopper': st([r['beans']['bean_outcome']['in_hopper'] for r in rr], 2),
            'beans_outside': st([r['beans']['bean_outcome']['outside'] for r in rr], 2),
            'beans_in_bag_final': st([r['beans']['bean_outcome']['in_bag'] for r in rr], 2),
            'bean_outcome_counts': cnt([f"{r['beans']['bean_outcome']['in_hopper']}/{r['beans']['bean_outcome']['in_bag']}/{r['beans']['bean_outcome']['outside']}" for r in rr]),
            'plan_steps': st([r['plan']['n_steps'] for r in rr], 1),
            'video_duration_s': st([r['video']['head']['duration_s'] for r in rr], 2),
        }
    S['all'] = block(rows)
    for g in S['group']:
        S['group:' + g] = block([r for r in rows if r['group']['manifest'] == g])
    for b in S['batch_prefix']:
        S['batch:' + b] = {k: v for k, v in block([r for r in rows if r['group']['batch_prefix'] == b]).items() if k in ('n', 'machine_dx', 'machine_dy', 'machine_yaw_deg', 'bag_dx', 'bag_dy', 'bag_yaw_deg')}
    S['world_matches_offsets'] = {
        'machine_xyz_max_abs_err_m': max(max(abs(r['machine']['offset_from_nominal_m'][i] - r['machine']['offset_m'][i]) for i in range(3)) for r in rows),
        'machine_yaw_max_abs_err_deg': max(abs(r['machine']['world_yaw_deg'] - r['machine']['yaw_deg']) for r in rows),
        'bag_xy_max_abs_err_m': max(max(abs(r['bag']['world']['xyz'][i] - NOMINAL_BAG[i] - r['bag']['offset_m'][i]) for i in range(2)) for r in rows),
        'bag_heading_minus_yaw_deg': cnt([round(r['bag']['world']['localZ_heading_deg'] - r['bag']['yaw_deg'], 2) for r in rows]),
    }
    S['randomization_equal_init_result_manifest'] = cnt([r['randomization_equal_init_result_manifest'] for r in rows])
    S['seed_sources_agree'] = cnt([len({v for k, v in r['seed'].items() if v is not None and k in ('initialization', 'plan', 'result', 'appearance', 'batch_json_episode_seed')}) == 1 for r in rows])
    S['seed_dirname_mismatch'] = {r['episode_id']: r['seed'] for r in rows if r['seed']['dir_name'] != r['seed']['initialization']}
    S['duplicate_seeds'] = dup([r['seed']['initialization'] for r in rows])
    S['seeds'] = sorted(r['seed']['initialization'] for r in rows)
    S['bag_geometry'] = cnt([r['bag']['geometry'] for r in rows])
    S['beans'] = {'bean_paths': cnt([r['beans']['bean_paths'] for r in rows]), 'bean_count': cnt([r['beans']['bag_geometry_bean_count'] for r in rows]),
                  'bean_scale': cnt([r['beans']['bag_geometry_bean_scale'] for r in rows]), 'plan_bean_count': cnt([r['beans']['plan_bean_count'] for r in rows]),
                  'plan_bean_geometry_scale': cnt([r['beans']['plan_bean_geometry_scale'] for r in rows]),
                  'bean_assets': cnt([r['beans']['bean_assets'] for r in rows]), 'initial_in_bag': cnt([r['beans']['result_initial']['in_bag'] for r in rows]),
                  'strict_pour_quality_all_beans_in_hopper': cnt([(r['beans']['strict_pour_quality'] or {}).get('all_beans_in_hopper') for r in rows]),
                  'manifest_bean_outcome_equal': cnt([r['beans']['manifest_bean_outcome'] == r['beans']['bean_outcome'] for r in rows])}
    S['machine_hopper_revision'] = cnt([r['machine_hopper_revision'] for r in rows])
    S['cup_placement_policy'] = cnt([r['cup']['placement_policy'] for r in rows])
    S['robot'] = {'root_xyz': cnt([r['robot']['root_xyz'] for r in rows]), 'yaw': cnt([r['robot']['yaw'] for r in rows]),
                  'initial_targets_hash': cnt([r['robot']['initial_targets_hash'] for r in rows]),
                  'robot_targets_hash(initialization.json, 34 joints)': cnt([r['robot']['robot_targets_hash'] for r in rows])}
    S['checks'] = {'passed': cnt([r['result']['passed'] for r in rows]), 'n_checks': cnt([r['result']['n_checks'] for r in rows]),
                   'all_true': cnt([r['result']['all_checks_true'] for r in rows]), 'failure': cnt([r['result']['failure'] for r in rows]),
                   'names': list(rows[0]['result']['checks'].keys()),
                   'manifest_checks_equal': cnt([r['result']['manifest_checks_equal'] for r in rows]),
                   'initialization_checks': cnt([r['initialization_checks'] for r in rows]),
                   'acceptance_scope': cnt([r['result']['acceptance_scope'] for r in rows]),
                   'contact_acceptance': cnt([r['result']['contact_acceptance'] for r in rows]),
                   'unexpected_contacts': cnt([r['result']['n_unexpected_contacts'] for r in rows]),
                   'bag_return_position_error_m': st([r['result']['bag_return_position_error_m'] for r in rows], 5),
                   'bag_return_rotation_error_rad': st([r['result']['bag_return_rotation_error_rad'] for r in rows], 5),
                   'lift_position_drift_m': st([r['result']['lift_position_drift_m'] for r in rows], 5),
                   'lift_rotation_drift_rad': st([r['result']['lift_rotation_drift_rad'] for r in rows], 5),
                   'max_bilateral_contact_frames': st([r['result']['max_bilateral_contact_frames'] for r in rows], 1),
                   'recorded_overlap_events': st([r['result']['recorded_overlap_events'] for r in rows], 1),
                   'recovered_contact_events': st([r['result']['recovered_contact_events'] for r in rows], 1),
                   'spill_contact_events': st([r['result']['spill_contact_events'] for r in rows], 1)}
    S['plan'] = {'control_hz': cnt([r['plan']['control_hz'] for r in rows]), 'q_len': cnt([r['plan']['q_len'] for r in rows]),
                 'step_keys': cnt([r['plan']['step_keys'] for r in rows]), 'left_q_constant_within_episode': cnt([r['plan']['left_q_constant'] for r in rows]),
                 'phase_sequences': cnt([[p[0] for p in r['plan']['phases']] for r in rows]),
                 'trace_len_equals_plan_steps': cnt([r['result']['trace_len'] == r['plan']['n_steps'] for r in rows]),
                 'video_capture_fps': cnt([r['plan']['video_capture_fps'] for r in rows]),
                 'video_capture_control_stride': cnt([r['plan']['video_capture_control_stride'] for r in rows]),
                 'side_grasp_height_angle_depth': cnt([[r['plan']['side_grasp'].get(k) for k in ('policy', 'height_m', 'angle_deg', 'side', 'roll_deg', 'depth_m')] for r in rows]),
                 'trace_joint_fields': cnt([r['result']['trace_joint_fields'] for r in rows]),
                 'trace_has_all_joint_positions': cnt([r['result']['trace_has_all_joint_positions'] for r in rows])}
    S['original_run'] = {'dir_exists': cnt([r['original_episode_dir_exists'] for r in rows]),
                         'executed_plan_equals_dataset_plan': cnt([r['original_run']['executed_plan_equals_dataset_plan'] for r in rows]),
                         'certificate_passed': cnt([r['original_run']['certificate_passed'] for r in rows]),
                         'has_physics_ticks': cnt([r['original_run']['has_physics_ticks'] for r in rows]),
                         'has_runtime_usda': cnt([r['original_run']['has_runtime_usda'] for r in rows]),
                         'has_initial_usda': cnt([r['original_run']['has_initial_usda'] for r in rows]),
                         'appearance_json_equals_initialization_appearance': cnt([r['appearance_in_initialization_equals_original_appearance_json'] for r in rows]),
                         'initialization_json_equal': cnt([r['initialization_equals_original_initialization_json'] for r in rows]),
                         'roots': cnt([r['original_episode_dir'].split('/coffee_task_workbench_20260922/')[0] for r in rows]),
                         'batch_json_attempted_vs_success': cnt([f"{(r['batch_json'] or {}).get('attempted')}/{(r['batch_json'] or {}).get('physical_successes')}" for r in rows])}
    S['video_manifest'] = {'actual_render_fps': cnt([r['video_manifest']['actual_render_fps'] for r in rows]),
                           'synchronized': cnt([r['video_manifest']['synchronized'] for r in rows]),
                           'n_frames_equals_ffprobe': cnt([r['video_manifest']['n_frames'] == r['video']['head']['nb_frames'] for r in rows]),
                           'simulation_seconds': st([r['video_manifest']['simulation_seconds'] for r in rows], 2),
                           'sim_seconds_equals_steps_over_20': cnt([abs(r['video_manifest']['simulation_seconds'] - r['plan']['n_steps'] / 20.0) < 1e-6 for r in rows])}
    S['video'] = video_stats(rows)
    S['sha256_recorded'] = 'no sha256 for the videos in manifest.json / video_manifest.json'
    S['appearance'] = appearance_stats(rows)
    S['appearance']['seed_equals_episode_seed'] = cnt([r['appearance']['seed'] == r['seed']['initialization'] for r in rows])
    S['appearance_varying_key_count'] = len(ex['appearance_varying'])
    S['appearance_constant_key_count'] = len(ex['appearance_constant_keys'])
    tab = table(['episode', 'seed', 'group', 'style', 'mach dx', 'mach dy', 'mach yaw°', 'bag dx', 'bag dy', 'bag yaw°', 'palette', 'lighting', 'steps', 'video s', 'frames', 'hopper/bag/out', 'bag return mm', 'checks'],
                [[r['episode_id'], r['seed']['initialization'], r['group']['manifest'], r['cup_style'],
                  r4(r['machine']['offset_m'][0]), r4(r['machine']['offset_m'][1]), r4(r['machine']['yaw_deg'], 2),
                  r4(r['bag']['offset_m'][0]), r4(r['bag']['offset_m'][1]), r4(r['bag']['yaw_deg'], 2),
                  r['appearance']['palette'], r['appearance']['lighting_scheme'], r['plan']['n_steps'], r['video']['head']['duration_s'], r['video']['head']['nb_frames'],
                  f"{r['beans']['bean_outcome']['in_hopper']}/{r['beans']['bean_outcome']['in_bag']}/{r['beans']['bean_outcome']['outside']}",
                  r4(r['result']['bag_return_position_error_m'] * 1000, 2),
                  f"{sum(1 for v in r['result']['checks'].values() if v is True)}/{r['result']['n_checks']}"] for r in rows])
    return S, tab, rows


# ------------------------------------------------------------------ 1.3
def s13():
    rows = L('1.3')
    ex = X('1.3')
    S = {'n': len(rows), 'extra': {k: ex[k] for k in ('n_dirs', 'dirs_not_in_data_list', 'data_list_not_in_dirs', 'manifest_not_in_data_list', 'data_list_not_in_manifest', 'manifest_top')}}
    S['arm'] = cnt([r['arm'] for r in rows])
    S['group'] = 'no group label recorded (single population)'
    S['missing_files'] = cnt([','.join(r['missing_files']) or 'none' for r in rows])
    S['proposal_config'] = cnt([r['proposal_config'] for r in rows])
    S['initialization_bag_cycle_ranges'] = cnt([(r['bag_cycle_initialization'] or {}).get('ranges') for r in rows])
    S['proposal'] = {'machine_dx_m': st([r['proposal']['machine_dx_m'] for r in rows]), 'machine_dy_m': st([r['proposal']['machine_dy_m'] for r in rows]),
                     'machine_dyaw_deg': st([r['proposal']['machine_dyaw_deg'] for r in rows], 3),
                     'equal_manifest': cnt([r['proposal_equal_manifest'] for r in rows]),
                     'equal_initialization_randomization': cnt([r['proposal_equal_initialization_randomization'] for r in rows])}
    S['machine_world'] = {'x': st([r['machine']['world_xyz'][0] for r in rows]), 'y': st([r['machine']['world_xyz'][1] for r in rows]),
                          'z': st([r['machine']['world_xyz'][2] for r in rows], 5),
                          'dx_from_nominal': st([r['machine']['offset_from_nominal_m'][0] for r in rows]),
                          'dy_from_nominal': st([r['machine']['offset_from_nominal_m'][1] for r in rows]),
                          'yaw_from_nominal_deg': st([r['machine']['world_yaw_deg'] for r in rows], 3),
                          'implied_reference_x': st([r['machine']['implied_reference_xyz'][0] for r in rows], 6),
                          'implied_reference_y': st([r['machine']['implied_reference_xyz'][1] for r in rows], 6),
                          'implied_reference_yaw_deg': st([r['machine']['implied_reference_yaw_deg'] for r in rows], 5)}
    S['button_world'] = {'x': st([r['button_world'][0] for r in rows]), 'y': st([r['button_world'][1] for r in rows]), 'z': st([r['button_world'][2] for r in rows], 5)}
    S['bag_world'] = {'xyz': cnt([[round(v, 5) for v in r['bag']['world']['xyz']] for r in rows]),
                      'yaw_deg(localZ heading + 8)': cnt([round(r['bag']['world']['localZ_heading_deg'] - NOMINAL_BAG_HEADING, 3) for r in rows]),
                      'offset_from_nominal_bag': [round(rows[0]['bag']['world']['xyz'][i] - NOMINAL_BAG[i], 5) for i in range(3)],
                      'geometry': cnt([r['bag']['geometry'] for r in rows])}
    S['cup_world'] = {'x': st([r['cup']['world']['xyz'][0] for r in rows]), 'y': st([r['cup']['world']['xyz'][1] for r in rows]),
                      'z': st([r['cup']['world']['xyz'][2] for r in rows], 5),
                      'style_recorded': cnt([r['cup']['cup_style_in_initialization'] for r in rows]),
                      'placement_policy': cnt([r['cup']['placement_policy'] for r in rows])}
    S['beans'] = {'bean_paths': cnt([r['beans']['bean_paths'] for r in rows]), 'bean_count': cnt([r['beans']['bag_geometry_bean_count'] for r in rows]),
                  'bean_scale': cnt([r['beans']['bag_geometry_bean_scale'] for r in rows]),
                  'beans_initially_in_hopper': cnt([r['beans']['beans_initially_in_hopper'] for r in rows]),
                  'beans_initially_in_bag': cnt([r['beans']['beans_initially_in_bag'] for r in rows]),
                  'trace_bean_count_values': cnt([r['beans']['trace_bean_count_values'] for r in rows])}
    S['reference_initialization'] = cnt([r['reference_initialization'] for r in rows])
    S['resize_source_initial'] = cnt([r['resize_source_initial'] for r in rows])
    S['seed_sources_agree'] = cnt([len(set(r['seed'].values())) == 1 for r in rows])
    S['duplicate_seeds'] = dup([r['seed']['proposal'] for r in rows])
    S['seeds'] = sorted(r['seed']['proposal'] for r in rows)
    S['robot'] = {'root_xyz': cnt([r['robot']['root_xyz'] for r in rows]), 'yaw': cnt([r['robot']['yaw'] for r in rows]),
                  'initial_targets_hash': cnt([r['robot']['initial_targets_hash'] for r in rows]),
                  'robot_targets_hash(initialization.json, 34 joints)': cnt([r['robot']['robot_targets_hash'] for r in rows])}
    S['plan'] = {'n_steps': cnt([r['plan']['n_steps'] for r in rows]), 'physics_hz': cnt([r['plan']['physics_hz'] for r in rows]),
                 'control_hz': cnt([r['plan']['control_hz'] for r in rows]), 'two_press_wait_seconds': cnt([r['plan']['two_press_wait_seconds'] for r in rows]),
                 'q_len': cnt([r['plan']['q_len'] for r in rows]), 'step_keys': cnt([r['plan']['step_keys'] for r in rows]),
                 'phases': cnt([r['plan']['phases'] for r in rows]),
                 'button_travel_min': st([r['plan']['button_travel_min'] for r in rows], 5), 'button_travel_max': st([r['plan']['button_travel_max'] for r in rows], 5),
                 'horizontal_height': st([r['plan']['horizontal_height'] for r in rows], 5)}
    S['route'] = {'yaw': cnt([r['route']['yaw'] for r in rows]), 'n_nodes': cnt([r['route']['n_nodes'] for r in rows]),
                  'vertical_travel': st([r['route']['vertical_travel'] for r in rows], 5)}
    S['certificate'] = {'passed': cnt([r['certificate']['passed'] for r in rows]), 'samples': st([r['certificate']['samples'] for r in rows], 1),
                        'commands': cnt([r['certificate']['commands'] for r in rows])}
    S['checks'] = {'passed': cnt([r['result']['passed'] for r in rows]), 'n_checks': cnt([r['result']['n_checks'] for r in rows]),
                   'all_true': cnt([r['result']['all_checks_true'] for r in rows]), 'failure': cnt([r['result']['failure'] for r in rows]),
                   'names': list(rows[0]['result']['checks'].keys()),
                   'manifest_status': cnt([r['result']['manifest_status'] for r in rows]),
                   'manifest_checks_equal': cnt([r['result']['manifest_checks_equal'] for r in rows]),
                   'manifest_events_equal': cnt([r['result']['manifest_events_equal'] for r in rows]),
                   'events': cnt([[e['event'] for e in r['result']['events']] for r in rows]),
                   'event_started_time_s': st([r['result']['events'][0]['time_s'] for r in rows], 3),
                   'event_stopped_time_s': st([r['result']['events'][1]['time_s'] for r in rows], 3),
                   'event_started_displacement_m': st([r['result']['events'][0]['button_displacement_m'] for r in rows], 6),
                   'event_stopped_displacement_m': st([r['result']['events'][1]['button_displacement_m'] for r in rows], 6),
                   'event_bean_count': cnt([[e['bean_count'] for e in r['result']['events']] for r in rows]),
                   'press_displacement_min_m(deepest)': st([r['result']['press_displacement_min_m'] for r in rows], 6),
                   'press_displacement_max_m(rest)': st([r['result']['press_displacement_max_m'] for r in rows], 6),
                   'pressed_control_steps': st([r['result']['pressed_control_steps'] for r in rows], 1),
                   'dispensing_control_steps': st([r['result']['dispensing_control_steps'] for r in rows], 1),
                   'particles_final': st([r['result']['particles_final'] for r in rows], 1),
                   'particles_final_counts': cnt([r['result']['particles_final'] for r in rows]),
                   'button_contacts': st([r['result']['n_button_contacts'] for r in rows], 1),
                   'unexpected': cnt([r['result']['n_unexpected'] for r in rows]),
                   'simulation_seconds': cnt([round(r['result']['simulation_seconds'], 3) for r in rows]),
                   'trace_keys': cnt([r['result']['trace_keys'] for r in rows]),
                   'trace_has_joint_positions': cnt([r['result']['trace_has_joint_positions'] for r in rows]),
                   'liquid_in_cup': 'not recorded (only emitted particle count)'}
    S['manifest'] = {'gpu': cnt([r['manifest']['gpu'] for r in rows]), 'duration': cnt([r['manifest']['duration'] for r in rows]),
                     'source_run_differs_from_run': cnt([r['manifest']['source_run'] != r['episode_id'] for r in rows])}
    S['capture_events'] = cnt([r['capture_events'] for r in rows])
    S['videos_ready'] = cnt([r['videos_ready'] for r in rows])
    S['video'] = video_stats(rows)
    S['sha256_recorded'] = {'source': 'coffee_updates_0925_verification.json (datasets root)', 'all_three_present': cnt([all(r['sha256_manifest'].values()) for r in rows])}
    S['appearance'] = appearance_stats(rows)
    S['appearance']['seed_equals_episode_seed'] = cnt([r['appearance']['seed'] == r['seed']['proposal'] for r in rows])
    S['appearance_json_equals_initialization_appearance'] = cnt([r['appearance_json_equals_initialization_appearance'] for r in rows])
    S['appearance_varying_key_count'] = len(ex['appearance_varying'])
    S['appearance_constant_key_count'] = len(ex['appearance_constant_keys'])
    tab = table(['episode', 'seed', 'prop dx', 'prop dy', 'prop dyaw°', 'mach dx vs nominal', 'mach dy vs nominal', 'mach yaw° vs nominal', 'palette', 'lighting', 'route yaw', 'steps', 'video s', 'frames', 'press min mm', 't start s', 't stop s', 'particles', 'beans', 'checks'],
                [[r['episode_id'], r['seed']['proposal'], r4(r['proposal']['machine_dx_m']), r4(r['proposal']['machine_dy_m']), r4(r['proposal']['machine_dyaw_deg'], 2),
                  r4(r['machine']['offset_from_nominal_m'][0]), r4(r['machine']['offset_from_nominal_m'][1]), r4(r['machine']['world_yaw_deg'], 2),
                  r['appearance']['palette'], r['appearance']['lighting_scheme'], r['route']['yaw'], r['plan']['n_steps'], r['video']['head']['duration_s'], r['video']['head']['nb_frames'],
                  r4(r['result']['press_displacement_min_m'] * 1000, 3), r4(r['result']['events'][0]['time_s'], 2), r4(r['result']['events'][1]['time_s'], 2),
                  r['result']['particles_final'], r['beans']['trace_bean_count_values'],
                  f"{sum(1 for v in r['result']['checks'].values() if v is True)}/{r['result']['n_checks']}"] for r in rows])
    return S, tab, rows


# ------------------------------------------------------------------ 1.4
def s14():
    rows = L('1.4')
    ex = X('1.4')
    S = {'n': len(rows), 'extra': {k: ex[k] for k in ('n_dirs', 'dirs_not_in_data_list', 'data_list_not_in_dirs', 'manifest_not_in_data_list', 'data_list_not_in_manifest', 'legacy_initial_runs')}}
    S['manifest_top'] = {k: v for k, v in ex['manifest_top'].items() if k not in ('coverage',)}
    S['manifest_coverage_claim'] = ex['manifest_top'].get('coverage')
    S['arm'] = cnt([r['arm']['resolved'] for r in rows])
    S['arm_recorded_in_plan'] = cnt([r['arm']['plan'] for r in rows])
    S['group'] = cnt([r['group']['manifest'] for r in rows])
    S['style'] = cnt([r['cup_style']['plan'] for r in rows])
    S['arm_x_style'] = cnt([f"{r['arm']['resolved']}/{r['cup_style']['plan']}" for r in rows])
    S['arm_x_group'] = cnt([f"{r['arm']['resolved']}/{r['group']['manifest']}" for r in rows])
    S['group_x_plan_bounds'] = cnt([f"{r['group']['manifest']} | perturbation_group={r['group']['plan_perturbation_group']} | bounds={json.dumps(r['group']['plan_randomization_bounds'])}" for r in rows])
    S['group_x_has_retrieval_pose_randomization'] = cnt([f"{r['group']['manifest']} | retrieval_pose_randomization={'yes' if r['seed']['retrieval_pose_randomization'] is not None else 'no'}" for r in rows])
    S['initial_state_status'] = cnt([r['initial_state_status'] for r in rows])
    S['missing_files'] = cnt([','.join(r['missing_files']) or 'none' for r in rows])

    def block(rr):
        return {
            'n': len(rr), 'arm': cnt([r['arm']['resolved'] for r in rr]), 'style': cnt([r['cup_style']['plan'] for r in rr]), 'group': cnt([r['group']['manifest'] for r in rr]),
            'machine_dx': st([r['machine']['plan_offset_m'][0] for r in rr]), 'machine_dy': st([r['machine']['plan_offset_m'][1] for r in rr]),
            'machine_dz': st([r['machine']['offset_from_nominal_m'][2] for r in rr]),
            'machine_yaw_deg': st([r['machine']['plan_yaw_deg'] for r in rr], 3),
            'initial_handle_world_yaw_deg': st([r['cup']['trace0_handle_yaw_deg'] for r in rr], 2),
            'initial_cup_x': st([r['cup']['result_initial_cup'][0] for r in rr]), 'initial_cup_y': st([r['cup']['result_initial_cup'][1] for r in rr]),
            'initial_cup_z': st([r['cup']['result_initial_cup'][2] for r in rr], 5),
            'initial_cup_to_tray_target_xy_m': st([r['cup']['initial_cup_to_tray_target_xy_m'] for r in rr], 5),
            'bag_dx': st([r['bag']['plan_offset_m'][0] for r in rr]), 'bag_dy': st([r['bag']['plan_offset_m'][1] for r in rr]),
            'bag_yaw_deg': st([r['bag']['plan_yaw_deg'] for r in rr], 2),
            'bag_at_nominal': sum(1 for r in rr if not any(r['bag']['plan_offset_m']) and not r['bag']['plan_yaw_deg']),
            'final_to_goal_xyz_m': st([r['result']['final_to_goal_xyz_m'] for r in rr], 5),
            'cup_lift_dz_m': st([r['result']['cup_lift_dz_m'] for r in rr], 4),
            'cup_displacement_m': st([r['result']['cup_displacement_m'] for r in rr], 4),
            'final_tcp_to_chest_goal_m': st([r['result']['final_tcp_to_chest_goal_m'] for r in rr], 5),
            'final_cup_x': st([r['cup']['result_final_cup'][0] for r in rr]), 'final_cup_y': st([r['cup']['result_final_cup'][1] for r in rr]),
            'final_cup_z': st([r['cup']['result_final_cup'][2] for r in rr]),
            'plan_steps': st([r['plan']['n_steps'] for r in rr], 1),
            'first_phase(removal_ready)_steps': st([r['plan']['first_phase_len'] for r in rr], 1),
            'video_duration_s': st([r['video']['head']['duration_s'] for r in rr], 2),
            'trace0_gripper_opening_rad': st([r['robot']['trace0_opening'] for r in rr], 3),
            'trace0_tcp_to_cup_m': st([r['robot']['trace0_tcp_to_initial_cup_m'] for r in rr], 3),
            'trace0_active_arm_max_joint_diff_from_standard_rad': st([r['robot']['trace0_active_arm_max_abs_diff_from_standard_rad'] for r in rr], 4),
            'trace0_other_arm_max_joint_diff_from_standard_rad': st([r['robot']['trace0_other_arm_max_abs_diff_from_standard_rad'] for r in rr], 6),
            'plan_initial_q_is_standard': cnt([r['robot']['plan_initial_q_is_standard'] for r in rr]),
            'has_initial_prefix_repair': cnt([r['robot']['plan_initial_prefix_repair'] is not None for r in rr]),
            'has_initial_arm_fixture_json': cnt([r['robot']['initial_arm_fixture_json'] is not None for r in rr]),
            'initial_targets_hash': cnt([r['robot']['initial_targets_hash'] for r in rr]),
            'max_bilateral_contact_frames': st([r['result']['max_bilateral_contact_frames'] for r in rr], 1),
        }
    S['all'] = block(rows)
    S['left'] = block([r for r in rows if r['arm']['resolved'] == 'left'])
    S['right'] = block([r for r in rows if r['arm']['resolved'] == 'right'])
    S['regular'] = block([r for r in rows if r['group']['manifest'] == 'regular'])
    S['large'] = block([r for r in rows if r['group']['manifest'] == 'large'])
    S['legacy8'] = block([r for r in rows if r['legacy_initial']])
    S['full42'] = block([r for r in rows if not r['legacy_initial']])
    S['legacy_detail'] = [{
        'episode': r['episode_id'], 'arm': r['arm']['resolved'], 'group': r['group']['manifest'], 'style': r['cup_style']['plan'],
        'supersedes': r['manifest_extra']['supersedes'], 'user_decision': r['manifest_extra']['user_decision'],
        'plan_steps': r['plan']['n_steps'], 'removal_ready_steps': r['plan']['first_phase_len'],
        'trace0_opening_rad': r4(r['robot']['trace0_opening'], 3), 'trace0_tcp_to_cup_m': r4(r['robot']['trace0_tcp_to_initial_cup_m'], 3),
        'active_arm_max_joint_diff_from_standard_rad': r4(r['robot']['trace0_active_arm_max_abs_diff_from_standard_rad'], 3),
        'has_initial_arm_fixture_json': r['robot']['initial_arm_fixture_json'] is not None,
        'has_initial_active_arm_retarget': r['robot']['plan_initial_active_arm_retarget'] is not None,
        'evidence_checks_all_true': (all(r['result']['manifest_evidence_checks'].values()) if r['result']['manifest_evidence_checks'] else None),
        'has_initial_repair_audit': r['result']['manifest_initial_repair_audit_passed'] is not None,
        'video_s': r['video']['head']['duration_s']} for r in rows if r['legacy_initial']]
    S['consistency'] = {
        'plan_vs_manifest_machine_offset_max_abs': max(max(abs(a - b) for a, b in zip(r['machine']['plan_offset_m'], r['machine']['manifest_offset_m'])) for r in rows),
        'plan_vs_trace_machine_offset_max_abs': max(max(abs(a - b) for a, b in zip(r['machine']['plan_offset_m'], r['machine']['offset_from_nominal_m'])) for r in rows),
        'plan_vs_trace_machine_yaw_max_abs_deg': max(abs(r['machine']['plan_yaw_deg'] - r['machine']['trace0_yaw_deg']) for r in rows),
        'manifest_vs_trace_handle_yaw_max_abs_deg': max(abs(r['cup']['initial_handle_world_yaw_deg_manifest'] - r['cup']['trace0_handle_yaw_deg']) for r in rows),
        'manifest_vs_result_initial_cup_max_abs_m': max(max(abs(a - b) for a, b in zip(r['cup']['actual_initial_cup_manifest'], r['cup']['result_initial_cup'])) for r in rows),
        'manifest_vs_result_final_cup_mismatch_gt_0.01mm': {r['episode_id']: round(max(abs(a - b) for a, b in zip(r['cup']['actual_final_cup_manifest'], r['cup']['result_final_cup'])), 5) for r in rows if max(abs(a - b) for a, b in zip(r['cup']['actual_final_cup_manifest'], r['cup']['result_final_cup'])) > 1e-5},
        'bag_trace_heading_minus_plan_yaw_deg': cnt([round(r['bag']['trace0_localZ_heading_deg'] - r['bag']['plan_yaw_deg'], 1) for r in rows]),
    }
    S['seeds'] = {'placing_seed(plan.randomization.seed)_duplicates': dup([r['seed']['plan_randomization'] for r in rows]),
                  'retrieval_pose_seed_present': cnt([r['seed']['retrieval_pose_randomization'] is not None for r in rows]),
                  'retrieval_pose_seed_duplicates': dup([r['seed']['retrieval_pose_randomization'] for r in rows]),
                  'continued_from_duplicates': dup([(r['plan']['continued_from'] or '').split('/')[-1] for r in rows]),
                  'pose_proposal_json_present': [r['episode_id'] for r in rows if r['seed']['pose_proposal_json']]}
    S['liquid'] = {'initial_final': cnt([[r['liquid']['initial'], r['liquid']['final']] for r in rows]), 'method': cnt([r['liquid']['method'] for r in rows]),
                   'manifest_equal': cnt([r['liquid']['manifest_equal'] for r in rows])}
    S['robot'] = {'root_xyz': cnt([r['robot']['root_xyz'] for r in rows]), 'yaw': cnt([r['robot']['yaw'] for r in rows]),
                  'initial_state_reference': cnt([r['robot']['initial_state_reference'] for r in rows]),
                  'initial_manifest_sha256': cnt([r['robot']['initial_manifest_sha256'] for r in rows]),
                  'trace0_bodylift': st([r['robot']['trace0_bodylift'] for r in rows], 6), 'trace0_waist': st([r['robot']['trace0_waist'] for r in rows], 6)}
    S['checks'] = {'passed': cnt([r['result']['passed'] for r in rows]), 'n_checks': cnt([r['result']['n_checks'] for r in rows]),
                   'all_true': cnt([r['result']['all_checks_true'] for r in rows]), 'failure': cnt([r['result']['failure'] for r in rows]),
                   'names': list(rows[0]['result']['checks'].keys()),
                   'status_json': cnt([r['result']['status_json_status'] for r in rows]),
                   'status_checks_equal': cnt([r['result']['status_checks_equal'] for r in rows]),
                   'manifest_physical_checks_equal': cnt([r['result']['manifest_physical_checks_equal'] for r in rows]),
                   'unexpected_contacts': cnt([r['result']['n_unexpected_contacts'] for r in rows]),
                   'manifest_initial_repair_audit': cnt([f"legacy={r['legacy_initial']} audit_passed={r['result']['manifest_initial_repair_audit_passed']} evidence_checks={'yes' if r['result']['manifest_evidence_checks'] else 'no'} repair_metrics={'yes' if r['repair_metrics_json'] else 'no'}" for r in rows]),
                   'initial_yaw_error_deg': st([r['result']['initial_yaw_error_deg'] for r in rows], 6)}
    S['preflight'] = {'passed': cnt([r['preflight']['passed'] for r in rows]), 'violations': cnt([r['preflight']['n_violations'] for r in rows]),
                      'clearance_m': cnt([r['preflight']['clearance_m'] for r in rows]),
                      'checked_samples': st([r['preflight']['checked_samples'] for r in rows], 1)}
    S['plan'] = {'executed_equals_plan': cnt([r['plan']['executed_equals_plan'] for r in rows]),
                 'executed_steps_equal': cnt([r['plan']['executed_steps_equal'] for r in rows]),
                 'keys_with_different_values': cnt([r['plan']['keys_with_different_values'] for r in rows]),
                 'step_keys': cnt([r['plan']['step_keys'] for r in rows]), 'q_len': cnt([r['plan']['q_len'] for r in rows]),
                 'phase_sequences': cnt([[p[0] for p in r['plan']['phases']] for r in rows]),
                 'trace_len_equals_plan_steps': cnt([r['result']['trace_len'] == r['plan']['n_steps'] for r in rows]),
                 'trace_dt': cnt([r['result']['trace_dt_values'] for r in rows]),
                 'n_joints_in_trace': cnt([r['result']['n_joints_recorded'] for r in rows]),
                 'frames_minus_steps': cnt([r['video']['head']['nb_frames'] - r['plan']['n_steps'] for r in rows]),
                 'repairs': cnt([f"prefix_repair={'yes' if r['robot']['plan_initial_prefix_repair'] else 'no'} shortcut={'yes' if r['robot']['plan_approach_shortcut_repair'] else 'no'} local_route={'yes' if r['robot']['plan_local_route_repair'] else 'no'}" for r in rows])}
    S['capture_events'] = {'n_events': st([r['capture_events']['n_events'] for r in rows], 1),
                           'first': cnt([[r['capture_events']['first_phase'], r['capture_events']['first_step'], round(r['capture_events']['first_sim_time_s'], 3)] for r in rows])}
    S['video'] = video_stats(rows)
    S['sha256_recorded'] = cnt([all(r['sha256_manifest'].values()) for r in rows])
    S['appearance'] = None
    S['appearance_note'] = rows[0]['appearance_note']
    S['appearance_seed_used_by_runner'] = {'n_distinct': len({r['appearance_seed_used_by_runner'] for r in rows}), 'duplicates': dup([r['appearance_seed_used_by_runner'] for r in rows])}
    S['extra_dirs'] = [{
        'dir': r['episode_id'], 'arm': r['arm']['resolved'], 'style': r['cup_style']['plan'], 'plan_steps': r['plan']['n_steps'],
        'removal_ready_steps': r['plan']['first_phase_len'], 'plan_initial_q_is_standard': r['robot']['plan_initial_q_is_standard'],
        'trace0_opening_rad': r4(r['robot']['trace0_opening'], 3), 'result_passed': r['result']['passed'], 'all_checks_true': r['result']['all_checks_true'],
        'video_s': r['video']['head']['duration_s'], 'machine_offset': [r4(x) for x in r['machine']['plan_offset_m']], 'machine_yaw_deg': r4(r['machine']['plan_yaw_deg'], 3),
        'handle_yaw_deg': r4(r['cup']['trace0_handle_yaw_deg'], 2),
        'batch_member_with_same_original_selected_id': next((q['episode_id'] for q in rows if q['manifest_extra']['original_selected_id'] == r['episode_id']), None)} for r in ex['extra_dir_rows']]
    tab = table(['episode', 'start', 'arm', 'group', 'style', 'placing seed', 'retrieval seed', 'mach dx', 'mach dy', 'mach yaw°', 'handle yaw°', 'cup x', 'cup y', 'bag dx', 'bag dy', 'bag yaw°', 'steps', 'ready steps', 'grip open0', 'video s', 'final→goal mm', 'liquid in/total', 'checks'],
                [[r['episode_id'], '旧起点' if r['legacy_initial'] else 'full', r['arm']['resolved'], r['group']['manifest'], r['cup_style']['plan'],
                  r['seed']['plan_randomization'], r['seed']['retrieval_pose_randomization'],
                  r4(r['machine']['plan_offset_m'][0]), r4(r['machine']['plan_offset_m'][1]), r4(r['machine']['plan_yaw_deg'], 2),
                  r4(r['cup']['trace0_handle_yaw_deg'], 2), r4(r['cup']['result_initial_cup'][0]), r4(r['cup']['result_initial_cup'][1]),
                  r4(r['bag']['plan_offset_m'][0]), r4(r['bag']['plan_offset_m'][1]), r4(r['bag']['plan_yaw_deg'], 2),
                  r['plan']['n_steps'], r['plan']['first_phase_len'], r4(r['robot']['trace0_opening'], 3), r['video']['head']['duration_s'],
                  r4(r['result']['final_to_goal_xyz_m'] * 1000, 2), f"{r['liquid']['final']['inside']}/{r['liquid']['final']['count']}",
                  f"{sum(1 for v in r['result']['checks'].values() if v is True)}/{r['result']['n_checks']}"] for r in rows])
    return S, tab, rows


def slim(row):
    """Drop bulky fields from the per-episode rows written to coverage_rows.json."""
    r = json.loads(json.dumps(row))
    r.pop('files', None)
    for k in ('original_episode_dir_files', 'original_initial_dir_files', 'original_motion_json_files'):
        r.pop(k, None)
    return r


def main():
    stats, tabs, allrows = {}, {}, {}
    for task, fn in (('1.1', s11), ('1.2', s12), ('1.3', s13), ('1.4', s14)):
        S, tab, rows = fn()
        stats[task] = S
        tabs[task] = tab
        allrows[task] = [slim(r) for r in rows]
    with open(os.path.join(SCRATCH, 'coverage_rows.json'), 'w') as f:
        json.dump(allrows, f, ensure_ascii=False, indent=1)
    with open(os.path.join(OUT_DIR, 'stats.json'), 'w') as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    with open(os.path.join(OUT_DIR, 'summary_tables.md'), 'w') as f:
        for t in tabs:
            f.write(f'\n### {t} per-episode table\n\n{tabs[t]}\n')
    print('rows', {k: len(v) for k, v in allrows.items()})
    print('coverage_rows.json bytes', os.path.getsize(os.path.join(SCRATCH, 'coverage_rows.json')))


if __name__ == '__main__':
    main()
