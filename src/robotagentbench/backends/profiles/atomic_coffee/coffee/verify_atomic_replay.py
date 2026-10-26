"""Recompute the actual trigger from native evidence; not full task scoring."""
import gzip
import hashlib
import json
import math
from bisect import bisect_left
from pathlib import Path
import subprocess
from types import SimpleNamespace
import numpy as np
from pxr import Usd
from evaluation.tasks.coffee_machine.specs import BAG
from .build_initial import HERE
from .geometry import bag_mouth_local
from .moving_frame import mouth_above_opening, pose_matrix


def verify(output, scene):
    output, scene = Path(output).resolve(), Path(scene).resolve()
    if not output.is_relative_to(HERE) or not scene.is_relative_to(HERE):
        raise ValueError('Task-owned evidence required')
    report = json.loads((output / 'report.json').read_text())
    trace = json.loads((output / 'trace.json').read_text())
    times = json.loads((output / 'video_times.json').read_text())
    actions = json.loads((output / 'adapted_source_actions.json').read_text())
    events = json.loads((output / 'trigger_events.json').read_text())
    accepted = json.loads((output / 'replay_initial_acceptance.json').read_text())
    start = events['first_start']
    checks = dict(native_completed=report['status'] == 'atomic_source_replay_completed',
        clean_shutdown=report.get('native_shutdown_complete') is True and report['cleanup_errors'] == [],
        source_layers_unchanged=report['source_layers_unchanged'] is True,
        source_plan_unchanged=hashlib.sha256(Path(actions['source_plan_path']).read_bytes()).hexdigest() == actions['source_plan_sha256'],
        actual_source_initial_accepted=accepted['accepted'] is True,
        all_source_actions_executed=report['source_rows_executed'] == len(actions['actions']) == len(trace),
        no_resampling_or_appended_actions=actions['generated_rows'] == 0 and actions['resampled'] is False and actions['appended_hold_rows'] == 0,
        no_mechanism_trigger_bypass=report['production_trigger_bypassed_for_unscored_mechanism_test'] is False,
        one_real_trigger=report['trigger_count'] == 1 and start is not None,
        no_target_pose_commands=all(report[k] == 0 for k in ('machine_pose_commands','cup_pose_commands','bag_pose_commands')),
        absent_at_initial=report['human_initial_absence']['visible'] is False and report['human_initial_absence']['collision_enabled'] is False,
        all_control_frames_timed=len(times) == len(trace) + 1 and all(abs(b-a-.05) < 1e-6 for a,b in zip(times,times[1:])),
        explicitly_adapted_unscored_diagnostic=report['original_unmodified_replay'] is False and report['scored_episode'] is False and report['model_run'] is False)
    recomputed = None
    intervention_distance = None
    pour_window = {p: dict(samples=0, opening_overlap_samples=0,
                          tilted_at_least_45_samples=0, tilted_at_least_45_overlap_samples=0,
                          overlap_tilt_range_deg=None) for p in
                   ('source_tilt_bag', 'source_untilt_bag')}
    if start is not None:
        wanted = start['time_s']
        row = None
        appeared = next((e.get('appearance_time_s') for e in events['actor_events'] if e.get('appearance_time_s') is not None), None)
        retired = next((e['time_s'] for e in events['actor_events'] if e.get('phase') == 'retired'), None)
        samples = {}
        data = json.loads((output / 'physical_evidence/detection-profile.json').read_text())
        hopper = SimpleNamespace(**data['machine_contract']['hopper'])
        stage = Usd.Stage.Open(str(scene))
        mouth_local = bag_mouth_local(stage)
        control_times = [r['time_s'] for r in trace]
        with gzip.open(output / 'physical_evidence/physical_facts.jsonl.gz', 'rt') as stream:
            for line in stream:
                value = json.loads(line)
                if value['time_s'] == wanted:
                    row = value
                index = min(bisect_left(control_times, value['time_s']), len(trace)-1)
                phase = trace[index]['phase']
                # The push necessarily starts while the mouth is above the
                # hopper. Test the subsequent pour, never count that required
                # trigger overlap as an intervention failure.
                if phase in pour_window and value['time_s'] >= wanted + 1.:
                    bag = value['poses'][BAG]
                    tf = pose_matrix(bag['position_m'], bag['quaternion_wxyz'])
                    mouth = tf[:3,:3] @ mouth_local + tf[:3,3]
                    overlaps = mouth_above_opening(mouth, value['measured_machine_world_matrix'], hopper)
                    pour_window[phase]['samples'] += 1
                    pour_window[phase]['opening_overlap_samples'] += int(overlaps)
                    tilt = value['tilt_deg'][BAG]
                    pour_window[phase]['tilted_at_least_45_samples'] += int(tilt >= 45.)
                    pour_window[phase]['tilted_at_least_45_overlap_samples'] += int(overlaps and tilt >= 45.)
                    if overlaps:
                        old = pour_window[phase]['overlap_tilt_range_deg']
                        pour_window[phase]['overlap_tilt_range_deg'] = [min(tilt, old[0]), max(tilt, old[1])] if old else [tilt, tilt]
                for label, target_time in (('appearance', appeared), ('retirement', retired)):
                    if target_time is not None and (label not in samples or
                            abs(value['time_s'] - target_time) < abs(samples[label]['time_s'] - target_time)):
                        samples[label] = value
        if len(samples) == 2:
            a = samples['appearance']['poses']['/World/Task75/Machine/Body']['position_m'][:2]
            b = samples['retirement']['poses']['/World/Task75/Machine/Body']['position_m'][:2]
            intervention_distance = math.dist(a, b)
        checks['trigger_has_same_substep_native_evidence'] = row is not None
        if row is not None:
            bag = row['poses'][BAG]
            tf = pose_matrix(bag['position_m'], bag['quaternion_wxyz'])
            mouth = tf[:3,:3] @ mouth_local + tf[:3,3]
            above = mouth_above_opening(mouth, row['measured_machine_world_matrix'], hopper)
            recomputed = dict(time_s=wanted, holding_bag=row['contacts'][BAG]['grasped'],
                              mouth_above_opening=above, tilt_deg=row['tilt_deg'][BAG])
            checks['current_grasp_at_trigger'] = recomputed['holding_bag'] is True
            checks['current_mouth_above_real_hopper'] = bool(above)
            checks['current_tilt_at_least_45'] = recomputed['tilt_deg'] >= 45.
    contacts = [c for r in trace for c in r['palm_machine_contacts']]
    mechanism = dict(
        actual_lower_case_palm_contact=any(any(p.endswith(('/Collision/Left','/Collision/Right','/Collision/Back')) for p in c['paths']) for c in contacts),
        no_hopper_or_head_palm_contact=not any(any('/Collision/Hopper' in p or p.endswith('/Collision/Head') for p in c['paths']) for c in contacts),
        actual_intervention_displacement_5_to_10cm=intervention_distance is not None and .05 <= intervention_distance <= .10,
        actor_retired=report['human_retired'] is True,
        no_observed_robot_human_contact=not report['robot_human_contact_observed'],
        final_machine_supported=bool(trace[-1]['machine_counter_contacts']),
        final_cup_supported='tray' in trace[-1]['cup_contacts']['supports'])
    video = output / 'atomic_trigger_replay.mp4'
    media = json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
        '-show_entries','stream=width,height,nb_frames','-of','json',str(video)], text=True))['streams'][0]
    resolution = tuple(report.get('capture_resolution', (640, 480)))
    checks['video_matches_control_clock'] = (int(media['nb_frames']) == len(times)
        and resolution in ((640, 480), (1440, 1080))
        and (media['width'], media['height']) == resolution)
    pour_clear = all(v['samples'] > 0 and v['opening_overlap_samples'] == 0 for v in pour_window.values())
    policy = (start or {}).get('admission', {}).get('proposal', {})
    admission = (start or {}).get('admission', {})
    attempts = admission.get('direction_attempts', [])
    x = policy.get('mouth_projection_local_x_m', 0)
    expected_preferred = -1 if x > 0 else 1 if x < 0 else None
    fallback = admission.get('opposite_side_fallback_used') is True
    depth_fallback = admission.get('front_back_fallback_used') is True
    selected_sign_valid = (policy.get('side') in ('front', 'back') and policy.get('machine_axis') == 'depth'
                          if depth_fallback else expected_preferred is None or policy.get('sign') == expected_preferred * (-1 if fallback else 1))
    policy_checks = dict(
        ten_cm_planned_travel=policy.get('distance_m') == .10,
        direction_from_current_native_mouth=(policy.get('direction_source') ==
            ('front_back_after_lateral_routes_blocked' if depth_fallback else
             'opposite_side_after_preferred_route_blocked' if fallback else 'current_native_bag_mouth_projection')),
        no_future_action_control=policy.get('future_source_actions_used') is False,
        selected_side_matches_projection_and_fallback=selected_sign_valid,
        checked_opposite_only_after_blocked_preferred=(len(attempts) == 2 and attempts[0]['admitted'] is False
            and attempts[1]['admitted'] is True and attempts[1]['side'] == policy.get('side') if fallback else True),
        checked_front_back_only_after_blocked_lateral=(len(attempts) in (3, 4)
            and {a['side'] for a in attempts[:2]} == {'left', 'right'}
            and all(a['admitted'] is False for a in attempts[:-1])
            and attempts[-1]['admitted'] is True and attempts[-1]['side'] == policy.get('side') if depth_fallback else True),
        no_wait_after_first_eligible=events.get('first_eligible', {}).get('time_s') == (start or {}).get('time_s'))
    result = dict(real_trigger_checks_passed=all(checks.values()), checks=checks,
        recomputed_trigger=recomputed, mechanism_checks=mechanism,
        capture_resolution=list(resolution),
        hd_presentation_verified=(all(checks.values()) and resolution == (1440, 1080)
            and report.get('capture_is_native_resolution') is True
            and report.get('camera_pose_and_lens_unchanged') is True),
        subsequent_source_pour_stays_off_hopper=pour_clear,
        subsequent_pour_native_samples=pour_window,
        farther_push_policy_checks=policy_checks,
        requested_farther_push_verified=all(checks.values()) and all(mechanism.values()) and all(policy_checks.values()) and pour_clear,
        whole_episode_machine_displacement_m=report['measured_machine_horizontal_displacement_m'],
        machine_displacement_during_intervention_m=intervention_distance,
        source_episode=actions['source_episode'], source_rows=len(actions['actions']),
        changed_open_commands=actions['changed_commands_count'], original_unmodified_replay=False,
        full_task_scoring_verified=False, unified_native_adapter_verified=False, model_verified=False,
        artifacts_sha256={n:hashlib.sha256((output/n).read_bytes()).hexdigest() for n in
            ('report.json','trace.json','trigger_events.json','adapted_source_actions.json','video_times.json','atomic_trigger_replay.mp4')})
    (output / 'atomic_trigger_verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    import sys
    r = verify(sys.argv[1], sys.argv[2])
    raise SystemExit(0 if r['requested_farther_push_verified'] else 1)
