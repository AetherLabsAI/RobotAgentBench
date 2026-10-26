"""Recompute isolated mechanical evidence; never validates a full task episode."""
import hashlib
import json
import math
from pathlib import Path
import subprocess
from .push_motion import DISTANCE_M


def verify(output):
    here = Path(__file__).resolve().parent
    output = Path(output).resolve()
    if output == here or not output.is_relative_to(here):
        raise ValueError('Task-owned native output required')
    report = json.loads((output / 'report.json').read_text())
    trace = json.loads((output / 'trace.json').read_text())
    times = json.loads((output / 'video_times.json').read_text())
    initial = report['initial_machine_pose'][:2]
    final = report['final_machine_pose'][:2]
    distance = math.dist(initial, final)
    post = [row for row in trace if row['phase'] == 'post_exit']
    retirement = report.get('human_retirement', {})
    checks = dict(
        native_completed=report['status'] == 'mechanical_preview_completed',
        shutdown_complete=report.get('native_shutdown_complete') is True,
        cleanup_clean=report.get('cleanup_errors') == [],
        source_layers_unchanged=report.get('source_layers_unchanged') is True,
        cpu_physics=report['physical_backend'] == 'cpu_physx',
        real_palm_contact=any(row['palm_machine_contacts'] for row in trace),
        actual_distance_in_approved_range=DISTANCE_M[0] <= distance <= DISTANCE_M[1],
        no_machine_cup_bag_pose_commands=all(report[key] == 0 for key in
            ('machine_pose_commands', 'cup_pose_commands', 'bag_pose_commands')),
        machine_supported_after_exit=bool(post) and all(row['machine_counter_contacts'] for row in post),
        cup_tray_supported_after_exit=bool(post) and all('tray' in row['cup_contacts']['supports'] for row in post),
        human_disappeared=(report['human_retired'] is True and retirement.get('visible') is False
                           and retirement.get('collision_enabled') is False),
        no_observed_robot_human_contact=not any(row['robot_human_contacts'] for row in trace),
        original_20hz_robot_hold_commands=report['robot_control_commands_continued_20hz'] is True,
        explicitly_not_a_task_trigger_run=(report['task_trigger_exercised'] is False
                                          and report['production_trigger_bypassed_for_unscored_mechanism_test'] is True),
        unscored_no_model=report['scored_episode'] is False and report['model_run'] is False,
        video_clock_monotone=all(b > a for a, b in zip(times, times[1:])),
    )
    fit = report.get('route_admission', {}).get('palm_fit', {})
    if fit.get('contact_region') in ('lower_main_case_side_wall', 'lower_main_case_wall'):
        contacts = [contact for row in trace for contact in row['palm_machine_contacts']]
        checks['fingers_horizontal'] = fit.get('fingers_horizontal') is True
        checks['native_palm_on_lower_case'] = any(
            any(path in ('/World/Task75/Machine/Body/Collision/Left',
                         '/World/Task75/Machine/Body/Collision/Right',
                         '/World/Task75/Machine/Body/Collision/Back') for path in contact['paths'])
            for contact in contacts)
        checks['no_palm_on_hopper_or_head'] = not any(
            any('/Collision/Hopper' in path or path.endswith('/Collision/Head')
                for path in contact['paths']) for contact in contacts)
    media = output / 'mechanical_preview.mp4'
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,nb_frames', '-of', 'json', str(media)], text=True))['streams'][0]
    checks['video_matches_frame_evidence'] = (int(probe['nb_frames']) == len(times) == report['frame_count']
                                               and probe['width'] == 640 and probe['height'] == 480)
    value = dict(mechanical_checks_passed=all(checks.values()), checks=checks,
                 measured_horizontal_displacement_m=distance,
                 maximum_machine_tilt_deg=report['maximum_machine_tilt_deg'],
                 maximum_cup_tilt_deg=report['maximum_cup_tilt_deg'],
                 full_task_accepted=False, actual_45_degree_trigger_verified=False,
                 scoring_verified=False, model_verified=False,
                 robot_movement_scope='original reset hold only, not an adaptive/continuous manipulation trajectory',
                 source_artifact_sha256={name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                     for name in ('report.json', 'trace.json', 'video_times.json', 'mechanical_preview.mp4')})
    (output / 'mechanical_verification.json').write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(value, indent=2))
    return value


if __name__ == '__main__':
    import sys
    result = verify(sys.argv[1])
    raise SystemExit(0 if result['mechanical_checks_passed'] else 1)
