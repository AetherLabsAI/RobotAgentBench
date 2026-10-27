"""Read and verify original task-owner closure receipts, never task scores.

Only a trusted operator invokes this after the worker exits. A closed ACK alone
is insufficient. Each family retains its existing receipt semantics; absent
generic fields are not fabricated for the original atomic/recovery owners.
"""
import argparse
import json
from pathlib import Path

FILES = {
    'room': ('native_report.json',),
    'atomic_coffee': ('native_report.json',),
    'failure_recovery': ('native_report.json',),
    'cloth': ('source_owner_report.json', 'formal_owner_report.json'),
}
ATOMIC = frozenset(('coffee_bag_interference', 'coffee_machine_push_interference',
                    'coffee_bag_then_machine_interference', 'coffee_button_lighting_interference'))
CLOTH = frozenset(('spread_tablecloth', 'collect_tablecloth', 'long_task_18'))
SHELF_APIS = frozenset('4-' + str(index) for index in range(1, 9))
RECOVERY = {
    'failure_recovery_shelf_drink': ('shelf_drink', None),
    'failure_recovery_coffee_bag': ('coffee_bag', None),
    'failure_recovery_sofa': ('sofa', 'sofa_to_coffee_table'),
    'failure_recovery_table_waste_empty_grasp': ('table_waste_empty_grasp', 'clear_table_trash'),
    'failure_recovery_table_waste_tipped_cup': ('table_waste_tipped_cup', 'clear_table_trash'),
}


def collect(family, native_output):
    """Fixed filenames only, no simulator imports, writes or recursive reads."""
    if family not in FILES:
        raise ValueError('Unknown original owner family')
    root = Path(native_output)
    if (not root.is_absolute() or root.resolve(strict=True) != root or not root.is_dir()
            or root in (Path(root.anchor), Path.home())):
        raise ValueError('Canonical task-owned output directory required')
    files = {}
    for name in FILES[family]:
        path = root/name
        if path.resolve(strict=True) != path or not path.is_file() or path.stat().st_size > 3*1024*1024:
            raise ValueError('Missing, indirect or oversized original receipt: '+name)
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError('Original receipt must be a JSON object')
        files[name] = value
    return dict(native_output=str(root), files=files)


def _clean(report):
    return (isinstance(report, dict) and report.get('cleanup_errors') == []
            and not any(report.get(key) for key in ('error', 'cleanup_error', 'traceback')))


def _identity(result, task, episode):
    return (isinstance(result, dict) and result.get('task_name') == task
            and result.get('episode_id') == episode)


def _lifecycle(value, api, episode):
    # Original TaskLifecycle._snapshot, not a generic result wrapper.
    return (isinstance(value, dict) and value.get('task_id') == api
            and value.get('episode_id') == episode and value.get('valid') is True
            and value.get('finalized') is True)


def verify(family, task_id, episode_id, bundle, *, closed_ack, worker_exit_code):
    """Fail closed on cross-run receipts. True means closure, not task success."""
    if (not isinstance(family, str) or family not in FILES or closed_ack is not True or type(worker_exit_code) is not int
            or worker_exit_code != 0 or not isinstance(task_id, str) or not task_id
            or not isinstance(episode_id, str) or not episode_id or not isinstance(bundle, dict)
            or set(bundle) != {'native_output', 'files'}):
        return False
    output = bundle['native_output']
    if not isinstance(output, str) or not Path(output).is_absolute() or '..' in Path(output).parts:
        return False
    files = bundle['files']
    if not isinstance(files, dict) or set(files) != set(FILES[family]):
        return False
    if family == 'cloth':
        from robotagentbench.simulation.cloth.supervisor import closure_accepted
        native, owner = files['source_owner_report.json'], files['formal_owner_report.json']
        if (task_id not in CLOTH or not isinstance(native, dict) or not isinstance(owner, dict)
                or not isinstance(native.get('final_result'), dict)
                or type(owner.get('physical_returncode')) is not int):
            return False
        return bool(native.get('schema') == 'robotagentbench.original_cloth_owner.v18'
            and native.get('episode_id') == episode_id
            and closure_accepted(native, owner.get('physical_returncode'), closed_reply=True, task_name=task_id)
            and _clean(owner) and owner.get('schema') == 'robotagentbench.formal_cloth_owner.v1'
            and owner.get('status') == 'native_session_closed' and owner.get('task_name') == task_id
            and owner.get('closed_ack_sent') is True and owner.get('lease_released') is True
            and owner.get('native_cleanup_accepted') is True and owner.get('input_forward_errors') == []
            and _identity(owner.get('final_result'), task_id, episode_id)
            and owner['final_result'] == native['final_result'])
    native = files['native_report.json']
    if not _clean(native) or native.get('status') != 'completed':
        return False
    if family == 'room':
        from robotagentbench.simulation.shutdown import shutdown_succeeded
        return bool(shutdown_succeeded(native) and _identity(native.get('task_result'), task_id, episode_id))
    if family == 'atomic_coffee':
        result = native.get('result', {})
        return bool(task_id in ATOMIC and native.get('task_name') == task_id
            and native.get('native_shutdown_complete') is True
            and _identity(result, task_id, episode_id)
            and result.get('schema') == 'short_tasks.atomic_coffee.result.v1'
            and result.get('status') == 'completed' and result.get('evaluation_valid') is True
            and _lifecycle(result.get('source_evaluation'),
                           '1-3' if task_id == 'coffee_button_lighting_interference' else '1-2', episode_id))
    if task_id not in RECOVERY or native.get('original_native_initial_accepted') is not True:
        return False
    variant, original_task = RECOVERY[task_id]
    if native.get('variant') != variant:
        return False
    if original_task is None:
        result = native.get('task_result', {})
        if not isinstance(result, dict):
            return False
        evaluation = result.get('evaluation')
        final = native.get('final_lifecycle_evaluation', {})
        api = native.get('source_api')
        allowed_apis = SHELF_APIS if variant == 'shelf_drink' else frozenset(('1-2',))
        return bool(native.get('task_name') == task_id and native.get('source_layers_unchanged') is True
            and result.get('status') == 'completed' and result.get('evaluation_valid') is True
            and isinstance(api, str) and api in allowed_apis
            and _lifecycle(evaluation, api, episode_id) and final == evaluation)
    result = native.get('closed_result', {})
    # The original room owner keeps its base-task result name. The explicit
    # variant binds it to the selected recovery without altering that result.
    # SingleTaskEpisode.finish emits no evaluation_valid field. Successful owner
    # admission, its exact finished-result schema, source re-verification, clean
    # exit and delayed closed ACK are the original room-recovery guarantees.
    return bool(_identity(result, original_task, episode_id)
        and result.get('schema') == 'openend.short_task.result.draft.v1'
        and result.get('status') == 'finished'
        and native.get('task_local_robot22_connected') is True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', choices=sorted(FILES), required=True)
    parser.add_argument('--native-output', type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(collect(args.family, args.native_output), allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
