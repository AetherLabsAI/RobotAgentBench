"""Portable API-action recording, with explicit reset compatibility.

Export a unified-runner trace; replay with ``arena_api.evaluate --source trajectory``.
This tool only packages already-issued commands. It cannot convert legacy USD/NPZ
replays with different resets or timing, nor guarantee task success on replay.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from . import bimanual as b, policy_execution as pe, robot_profiles as rp
from .policy_contract import digest
from .schemas import Randomization

SCHEMA = 'arena.recorded_actions.v1'
SCHEMA20 = 'arena.recorded_actions.v2'


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate trajectory key: ' + key)
        result[key] = value
    return result


def load_json(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=_unique)


def validate(data):
    if not isinstance(data, dict) or data.get('schema_version') not in (SCHEMA, SCHEMA20):
        raise ValueError('Expected arena.recorded_actions.v1; legacy replay manifests need explicit reset/timing adaptation')
    if set(data) != {'schema_version', 'execution_profile', 'reset', 'actions', 'provenance', 'content_sha256'}:
        raise ValueError('Unexpected or missing trajectory fields')
    digest(data)  # Finite JSON only.
    if data['content_sha256'] != digest({k: v for k, v in data.items() if k != 'content_sha256'}):
        raise ValueError('Trajectory content SHA256 mismatch')
    profile = rp.JOINT16 if data['schema_version'] == SCHEMA else rp.JOINT20
    if data['execution_profile'] != pe.spec(profile):
        raise ValueError('Trajectory control rate/units/joints/profile mismatch; no automatic resampling')
    reset = data['reset']
    state_key, tolerance_key = (('joint_positions_rad', 'joint_tolerance_rad') if profile == rp.JOINT16
                                else ('joint_positions_native', 'joint_tolerances_native'))
    if set(reset) - {'task_parameters'} != {'task_id', 'seed', 'randomization', 'task_contract',
                                            state_key, tolerance_key}:
        raise ValueError('Incomplete trajectory reset contract')
    if type(reset['seed']) is not int or not 0 <= reset['seed'] < 2**32:
        raise ValueError('Invalid trajectory seed')
    if Randomization.model_validate(reset['randomization']).model_dump() != reset['randomization']:
        raise ValueError('Randomization flags must be explicit')
    if not isinstance(reset['task_contract'], dict) or not reset['task_contract']:
        raise ValueError('Missing task-provided reset compatibility contract')
    pe._values(reset[state_key], 'recorded initial state', profile)
    tolerance = reset[tolerance_key]
    if profile == rp.JOINT16:
        if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance) or not 0 < tolerance <= .02:
            raise ValueError('Initial joint tolerance must be positive and at most 0.02 rad')
    else:
        rp.values(tolerance, profile, "explicit per-joint tolerances in native units")
        if any(v <= 0 for v in tolerance.values()):
            raise ValueError("Native tolerances must be explicitly positive")
    from .task_registry import default_registry
    registration = default_registry().get(reset['task_id'])
    registration.resolve_parameters(reset.get('task_parameters', {}))
    actions = data['actions']
    if not isinstance(actions, list) or not 1 <= len(actions) <= registration.max_steps:
        raise ValueError('Invalid recorded action count')
    for action in actions:
        pe._values(action, 'recorded targets', profile)
    if not isinstance(data['provenance'], dict) or not data['provenance']:
        raise ValueError('Missing recording provenance')
    return data


def load_trajectory(path):
    path = Path(path)
    raw = path.read_bytes()
    data = json.loads(raw, object_pairs_hook=_unique)
    return validate(data), hashlib.sha256(raw).hexdigest()


def export_trace(trace_path, output, joint_tolerance_rad=.005, *, joint_tolerances_native=None):
    trace_path, output = Path(trace_path), Path(output)
    trace = load_json(trace_path)
    if (trace.get('error') or not trace.get('closed_loop_transport_pass') or
            not trace.get('cleanup') or any(v != 'closed' for v in trace['cleanup'].values())):
        raise ValueError('Only a completed, acknowledged API execution trace may be exported')
    config, obs = trace['episode_config'], trace['initial_observation']
    if (obs['step_index'] != 0 or obs['source'] != 'isaac_physics' or
            obs['episode_id'] != trace['episode_id']):
        raise ValueError('Missing initial measured API observation')
    profile = rp.get(trace['execution_profile']['robot_profile']).id
    if trace['execution_profile'] != pe.spec(profile) or config.get('robot_profile') != profile:
        raise ValueError("Trace reset/execution robot profile mismatch")
    keys = pe.evidence_keys(profile)
    if profile == rp.JOINT20:
        rp.values(joint_tolerances_native, profile, "explicit per-joint replay tolerances")
    actions = []
    previous_time = obs['time_s']
    for i, step in enumerate(trace['steps']):
        request, after, execution = step['request'], step['observation'], step['execution']
        # Existing trajectory schemas cannot preserve a mixed-rate schedule.
        # Never silently export one as a fixed-240-Hz replay.
        if (request.get('physics_hz') not in (None, 240)
                or execution.get('physics_rate', {}).get('physics_hz', 240) != 240):
            raise ValueError('Mixed-rate traces require a rate-aware replay format; retain the source trace')
        if (request['expected_step_index'] != i or after['step_index'] != i+1 or
                after['episode_id'] != obs['episode_id'] or
                execution[keys['requested_targets_rad']] != request['action']['targets'] or
                execution['applied_substeps'] != pe.SUBSTEPS or
                abs(after['time_s'] - previous_time - b.CONTROL_DT) > 1e-6):
            raise ValueError('Missing or reordered acknowledged trajectory commands')
        actions.append(dict(request['action']['targets']))
        previous_time = after['time_s']
    data = dict(schema_version=SCHEMA if profile == rp.JOINT16 else SCHEMA20, execution_profile=trace['execution_profile'],
        reset=dict(task_id=config['task_id'], seed=config['seed'], randomization=config['randomization'],
            task_parameters=config.get('task_parameters', {}),
            task_contract=trace['initial_metrics'].get('replay_reset_contract'),
            joint_positions_rad={n: obs['joint_positions_rad'][n] for n in b.JOINTS},
            joint_tolerance_rad=joint_tolerance_rad), actions=actions,
        provenance=dict(source_trace_sha256=hashlib.sha256(trace_path.read_bytes()).hexdigest(),
            source_episode_id=trace['episode_id'], source_kind=trace['action_source'],
            interpretation='Exact issued API targets; not a promise of identical contact outcomes or pixels'))
    if profile == rp.JOINT20:
        data['reset'].pop('joint_positions_rad')
        data['reset'].pop('joint_tolerance_rad')
        data['reset']['joint_positions_native'] = {
            n: obs['joint_positions_' + u][n] for n,u in rp.get(profile).units.items()}
        data['reset']['joint_tolerances_native'] = dict(joint_tolerances_native)
    data['content_sha256'] = digest(data)
    validate(data)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        stream.write(json.dumps(data, indent=2, allow_nan=False) + '\n')
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-trace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--joint-tolerances-native', type=Path,
                        help='Required for joint20: JSON map of per-joint rad/m replay tolerances')
    args = parser.parse_args()
    data = export_trace(args.from_trace, args.output,
                        joint_tolerances_native=load_json(args.joint_tolerances_native)
                        if args.joint_tolerances_native else None)
    print(json.dumps(dict(output=str(args.output), control_steps=len(data['actions']),
                          content_sha256=data['content_sha256']), indent=2))


if __name__ == '__main__':
    main()
