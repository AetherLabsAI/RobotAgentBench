"""Acceptance of relocation preview only, never of full atomic pouring."""
import json
from pathlib import Path
from .trigger_rules import TRIGGER_DISTANCE_M
from .training_domain import contains_origin
from evaluation.tasks.coffee_machine.api_runtime import INITIAL_BAG_TILT_DEG

HERE=Path(__file__).resolve().parent


def validate(directory):
    directory=Path(directory).resolve()
    if not directory.is_relative_to(HERE):raise ValueError('Task-owned preview required')
    report=json.loads((directory/'report.json').read_text())
    if report.get('status')!='preview_completed':
        result=dict(accepted=False,scope='unscored relocation diagnostic',
                    checks=dict(native_execution_completed=False),
                    reason=report.get('error',report.get('status')),
                    full_atomic_pour_verified=False,model_verified=False)
        (directory/'relocation_acceptance.json').write_text(json.dumps(result,indent=2)+'\n')
        return result
    rows=json.loads((directory/'trace.json').read_text())
    acquisition=report.get('acquisition',{})
    retired=report.get('human_retirement',{})
    trigger=report.get('trigger',{})
    done=[r for r in rows if r['phase']=='done']
    checks=dict(
        native_shutdown=report.get('native_shutdown_complete') is True,
        no_cleanup_errors=report.get('cleanup_errors')==[],
        actual_10cm_trigger=trigger.get('threshold_m')==TRIGGER_DISTANCE_M
            and trigger.get('start') is True and min(trigger['distances_m'].values())<=TRIGGER_DISTANCE_M,
        real_opposed_acquisition=acquisition.get('hand_target_contact') is True
            and 'thumb' in acquisition.get('native_digits',[]) and any(
                d in acquisition.get('native_digits',[]) for d in ('index','middle','ring','pinky')),
        acquired=report.get('acquired') is True,
        released=report.get('released') is True,
        human_disappeared=report.get('human_retired') is True
            and retired.get('visible') is False and retired.get('collision_enabled') is False,
        post_exit_physics_recorded=bool(done) and rows[-1]['time_s']>done[0]['time_s'],
        native_counter_support=report.get('final_supported') is True,
        upright=report['final_tilt_deg']<=INITIAL_BAG_TILT_DEG,
        nonzero_actual_relocation=report['measured_relocation_m']>0,
        actual_endpoint_in_original_training_domain=report.get('final_in_training_domain') is True
            and contains_origin(report['final_target_pose'][:3],report['training_domain']),
        original_100_beans_retained=report['initial_beans_in_bag']==report['beans_in_bag_final']==100,
        source_robot_commands_continued=report.get('source_robot_commands_continued') is True,
        no_runtime_bag_pose_writes=report.get('bag_pose_writes')==0,
        source_layers_unchanged=report.get('source_layers_unchanged') is True,
        scoped_gripper_filter=report['human_collision_policy']['gripper_pairs_filtered'] is True
            and report['human_collision_policy']['other_robot_links_filtered'] is False,
        no_other_human_robot_contact=report.get('robot_human_contacts')==[],
        actual_jaw_direction_admitted=trigger.get('direction_check',{}).get('allowed') is True,
        actual_final_direction_nonapproaching=report.get('final_direction_check',{}).get('allowed') is True,
    )
    result=dict(accepted=all(checks.values()),checks=checks,scope='one unscored source-layout relocation preview',
        full_atomic_pour_verified=False,model_verified=False,
        trigger_distance_m=TRIGGER_DISTANCE_M,measured_relocation_m=report['measured_relocation_m'],
        post_retirement_physics_s=rows[-1]['time_s']-done[0]['time_s'] if done else 0.)
    (directory/'relocation_acceptance.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path)
    a=p.parse_args();result=validate(a.directory);print(json.dumps(result))
    raise SystemExit(0 if result['accepted'] else 1)
