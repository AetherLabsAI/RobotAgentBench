"""Check bag/combined probe artifacts honestly; never infer full-task scoring."""
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent


def verify(output):
    output = Path(output).resolve(strict=True)
    if not output.is_relative_to(HERE):
        raise ValueError('Task-local artifacts required')
    report = json.loads((output / 'report.json').read_text())
    actors = json.loads((output / 'actor_events.json').read_text())
    times = json.loads((output / 'video_times.json').read_text())
    trace = json.loads((output / 'trace.json').read_text())
    started = [event for event in actors['bag'] if event.get('kind') == 'started']
    acquired = [event for event in actors['bag'] if event.get('kind') == 'acquired']
    completed = [event for event in actors['bag'] if event.get('kind') == 'completed']
    media = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,nb_frames,r_frame_rate', '-of', 'json',
        str(output / 'actor_probe.mp4')], text=True))['streams'][0]
    completion = report.get('bag_completion_evidence') or {}
    checks = dict(
        native_run_completed=report['status'] == 'approach_probe_completed',
        exactly_one_native_proximity_trigger=len(started) == 1,
        recorded_distance_at_most_10cm=(len(started) == 1 and started[0]['threshold_m'] == .10
            and min(started[0]['distances_m'].values()) <= .10),
        opposed_native_contact_acquisition_recorded=len(acquired) == 1,
        completed_once_after_acquisition=(len(completed) == len(acquired) == 1
            and acquired[0]['time_s'] < completed[0]['time_s']),
        actual_release_and_support=completion.get('released') is True and completion.get('supported') is True
            and completion.get('constraint_active') is False,
        completion_in_training_domain=completion.get('in_training_domain') is True,
        moved_away_from_native_jaw=(completion.get('direction') or {}).get('allowed') is True,
        hand_retired=report.get('bag_human_retired') is True,
        source_unchanged=report.get('source_layers_unchanged') is True,
        robot_commands_continued=report.get('robot_continued') is True,
        no_target_pose_commands=report['bag_pose_commands'] == report['machine_pose_commands'] == 0,
        cleanup_complete=report.get('native_shutdown_complete') is True and not report['cleanup_errors'],
        video_clock_consistent=(int(media['nb_frames']) == len(times) == len(trace) + 1
            and (media['width'], media['height']) == (640, 480) and media['r_frame_rate'] == '20/1'
            and all(abs(b - a - .05) < 1e-6 for a, b in zip(times, times[1:]))),
    )
    result = dict(component_intervention_verified=all(checks.values()), checks=checks,
        final_in_training_domain=report.get('final_in_training_domain'),
        complete_preview_verified=all(checks.values()) and report.get('final_in_training_domain') is True,
        full_pour_verified=False, full_task_scoring_verified=False, model_verified=False,
        proximity_evidence_scope='recorded distances from native trigger reader; native jaw transforms not exported for independent geometric recomputation',
        source_rows_executed=report.get('rows_executed'),
        actual_relocation_m=(completion.get('direction') or {}).get('horizontal_displacement_m'),
        bag_trigger_time_s=report.get('bag_trigger_time_s'),
        completed_components=report.get('completed_components'),
        machine_triggered=report.get('machine_triggered'),
        combined_full_trigger_verified=report.get('combined_full_trigger_verified', False))
    (output / 'probe_verification.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    import sys
    value = verify(sys.argv[1])
    print(json.dumps(value, indent=2))
    raise SystemExit(0 if value['component_intervention_verified'] else 1)
