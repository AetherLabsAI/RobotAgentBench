"""Explicit raw-worker Codex mode; preparation and native lifecycle stay owned.

Uses the same official App Server, sequential turn loop and Worker transport as
the coding example. It does not build a sandbox, prepare a scene, expose TCP/IK,
retry a model/robot request, or replace a native task's cleanup requirements.
"""
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import subprocess


FAMILIES = frozenset(('room', 'atomic_coffee', 'failure_recovery', 'cloth'))
RECEIPT_FIELDS = frozenset(('family', 'task_id', 'native_output', 'read_argv'))
CONTINUE = ('The native environment has not terminated. Inspect the current observations and '
            'continue with one safe Arena tool. A prose summary is not task completion. '
            'Use done with an honest reason to stop.')


def validate_settings(value):
    """Pure validation; do not invoke the configured worker or receipt reader."""
    if value.get('worker_mode') != 'raw22' or value.get('prepare_argv') != []:
        raise ValueError('Raw mode requires an already prepared worker; no preparation command')
    receipt = value.get('native_receipt')
    if not isinstance(receipt, dict) or set(receipt) != RECEIPT_FIELDS:
        raise ValueError('Raw mode requires an explicit native receipt binding')
    if not isinstance(receipt['family'], str) or receipt['family'] not in FAMILIES:
        raise ValueError('Unknown native receipt family')
    if not isinstance(receipt['task_id'], str) or not receipt['task_id'].strip():
        raise ValueError('Explicit native task identity required')
    output = receipt['native_output']
    if not isinstance(output, str):
        raise ValueError('Canonical native output directory required')
    path = Path(output)
    if (not path.is_absolute() or path.resolve() != path or path == Path(path.anchor)
            or path == Path.home() or path == Path(value['output'])):
        raise ValueError('Separate canonical task-owned native output directory required')
    argv = receipt['read_argv']
    if (not isinstance(argv, list) or not argv
            or any(not isinstance(part, str) or not part or '\0' in part for part in argv)):
        raise ValueError('Explicit trusted receipt-reader argv required, never model-provided shell')
    return deepcopy(receipt)


def _save(path, value):
    temporary = path.with_suffix(path.suffix + '.next')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def _owner_result(report, native):
    """Operator-only result fields; do not infer success from done/closed."""
    if not isinstance(native, dict):
        return
    report['owner_status'] = native.get('status') if isinstance(native.get('status'), str) else None
    for name in ('evaluation_valid', 'success'):
        report[name] = native[name] if type(native.get(name)) is bool else None
    for name in ('scoring_status', 'termination_reason'):
        report[name] = native[name] if isinstance(native.get(name), str) else None
    if type(native.get('score')) in (int, float) and math.isfinite(native['score']):
        report['score'] = native['score']
    scores = native.get('scores')
    if isinstance(scores, dict) and all(type(x) in (int, float) and math.isfinite(x) for x in scores.values()):
        report['owner_scores'] = scores
    if type(native.get('success')) is bool:
        report['owner_success'] = native['success']


def _verify_receipts(binding, bridge, closure, output):
    # This command belongs only to the trusted operator. It is never visible to
    # App Server and must collect fixed task-family filenames after actual exit.
    from robotagentbench.model_examples.native_receipts import verify
    collected = subprocess.run(binding['read_argv'], capture_output=True, text=True,
                               check=True, timeout=60, shell=False)
    if len(collected.stdout.encode('utf-8')) > 4 * 1024 * 1024:
        raise ValueError('Oversized native receipt bundle')
    bundle = json.loads(collected.stdout)
    if not isinstance(bundle, dict) or bundle.get('native_output') != binding['native_output']:
        raise ValueError('Native receipt output differs from the selected worker output')
    _save(output / 'native_receipts.private.json', bundle)
    accepted = verify(binding['family'], binding['task_id'], bridge.observed['episode_id'], bundle,
                      closed_ack=closure['closed_ack'], worker_exit_code=closure['transport_exit_code'])
    return closure['worker_closed'] is True and accepted is True


def run(value):
    """Run only on explicit --execute; unknown physical outcomes remain held."""
    from .app_server import AppServer
    from .protocol import Worker
    from .raw_bridge import RawRobotBridge, text_item, tool_specs
    from .session import run_turn

    binding = validate_settings(value)
    output = Path(value['output'])
    output.mkdir(parents=True, exist_ok=False)
    os.chmod(output, 0o700)
    report = dict(status='initializing', worker_mode='raw22', model=value['model'], effort=value['effort'],
                  model_responses=0, actions_executed=0, evaluation_valid=None, score=None,
                  native_cleanup_verified=False, automatic_retries=False)
    worker = server = bridge = None

    def checkpoint():
        if bridge is not None:
            report.update(model_responses=bridge.model_responses, actions_executed=bridge.observed['step'],
                          in_flight_action=bridge.pending is not None)
        _save(output / 'operator_result.json', report)

    def close_worker():
        nonlocal worker
        closure = bridge.close()  # No automatic repeat after an uncertain close.
        worker = None  # close() returned only after an actual transport process exit.
        report['worker_close'] = closure
        report['worker_exit_code'] = closure['transport_exit_code']
        report['native_cleanup_verified'] = _verify_receipts(binding, bridge, closure, output)
        report['receipt_binding'] = {key: binding[key] for key in ('family', 'task_id', 'native_output')}
        report['receipt_binding']['episode_id'] = bridge.observed['episode_id']
        return report['native_cleanup_verified']

    checkpoint()
    try:
        version = subprocess.run([value['codex_executable'], '--version'], capture_output=True,
                                 text=True, check=True).stdout.strip()
        if version != value['expected_cli_version']:
            raise ValueError('CLI version differs from the explicitly selected deployment')
    except BaseException as error:
        report.update(status='invalid_before_start', error_type=type(error).__name__, evaluation_valid=False)
        checkpoint()
        raise
    report['cli_version'] = version
    with (output / 'worker_stderr.log').open('x') as stderr:
        try:
            worker = Worker(value['worker_argv'], stderr)
            bridge = RawRobotBridge(worker, output / 'observations',
                                    max_model_responses=value['max_model_responses'])
            server = AppServer(value['codex_executable'])
            started = server.start_thread(value['model'], bridge.instructions, tool_specs())
            thread, items = started['thread']['id'], bridge.initial_items()
            report['status'] = 'model_running'
            checkpoint()
            while True:
                result = run_turn(server, thread, bridge, input_items=items, effort=value['effort'])
                checkpoint()
                if bridge.pending is not None:
                    raise RuntimeError('Uncertain robot action: no replay or replacement')
                if bridge.completed or bridge.abandoned:
                    break
                if result['status'] != 'turn_completed_environment_held':
                    raise RuntimeError('Model turn stopped without a native result')
                items = [text_item(CONTINUE)]
            closing_server, server = server, None
            closing_server.close()
            clean = close_worker()
            report.update(status='closed' if clean else 'cleanup_unverified',
                          model_done_is_not_success=True,
                          outcome='native_ended' if bridge.completed else 'model_done')
            if clean:
                _owner_result(report, bridge.owner_result)
            checkpoint()
            return 0 if clean else 1
        except BaseException as error:
            report.update(status='invalid_or_held', error_type=type(error).__name__,
                          score=None, evaluation_valid=False)
            checkpoint()
            raise
        finally:
            if server is not None:
                try:
                    server.close()
                except Exception as error:
                    report['model_cleanup_error_type'] = type(error).__name__
            if worker is not None:
                if bridge is None or bridge.pending is not None or bridge.close_attempted:
                    # Do not send close over an unknown action/closure outcome.
                    # The existing independent owner/watchdog remains responsible.
                    report['status'] = 'uncertain_worker_held_for_operator'
                    checkpoint()
                    worker.process.wait()
                else:
                    try:
                        close_worker()
                    except Exception as error:
                        report['cleanup_error_type'] = type(error).__name__
                        if worker is not None:
                            report['status'] = 'uncertain_worker_held_for_operator'
                            checkpoint()
                            worker.process.wait()
            checkpoint()
