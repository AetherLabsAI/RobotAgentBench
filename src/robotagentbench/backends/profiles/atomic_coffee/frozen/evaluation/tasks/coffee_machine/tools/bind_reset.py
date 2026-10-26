"""Bind a converted source trajectory to the accepted initial-state probe of the SAME source episode.

Never invents a reset contract: `task_contract` is copied verbatim from the `replay_reset_contract` that
`api_runtime.validate_initial` wrote during the probe; the public replay checks it again before any motion.
Only the layout a trajectory was recorded on may be bound: a seeded random layout is refused.
"""
import argparse
import hashlib
import json
from pathlib import Path

from arena_api import trajectory, robot_profiles as rp, policy_execution as pe
from arena_api.policy_contract import digest
from arena_api.schemas import Randomization
from evaluation.tasks.coffee_machine.process_rules import RULE_VERSION
from evaluation.tasks.coffee_machine.registration import registration
from evaluation.tasks.coffee_machine.specs import TASK, EVALUATOR_ID, RESET_SCHEMA
from evaluation.tasks.coffee_machine.tools.convert_trajectory import SCHEMA as CANDIDATE_SCHEMA


def bind(candidate, probe):
    candidate, probe = Path(candidate), Path(probe)
    data = json.loads(candidate.read_text())
    if data.get('schema') != CANDIDATE_SCHEMA:
        raise ValueError('Wrong converted candidate schema')
    api, episode = data['task_id'], data['source_episode']
    if registration(api).evaluator_id != EVALUATOR_ID:
        raise ValueError('Registered evaluator is not ' + EVALUATOR_ID)
    manifest = json.loads((probe / 'manifest.json').read_text())
    if manifest.get('status') != 'completed' or manifest.get('source_unchanged') is not True:
        raise ValueError('A completed stable-version physical probe is required')
    if manifest.get('task_id') != api or manifest.get('role') != 'source':
        raise ValueError('The source-layout probe of the same API is required')
    parameters = manifest['task_parameters']
    if parameters != dict(source_episode=episode, random_layout=False):
        raise ValueError('Probe and trajectory belong to different source episodes')
    metadata = json.loads((probe / 'initial-metadata.json').read_text())
    acceptance = metadata.get('task_initial_acceptance', {})
    if acceptance.get('accepted') is not True or not all(acceptance['checks'].values()):
        raise ValueError('Missing accepted initial-state evidence')
    contract = metadata.get('replay_reset_contract') or {}
    if (contract.get('schema'), contract.get('task_id'), contract.get('source_episode'), contract.get('rule_version')) != (RESET_SCHEMA, api, episode, RULE_VERSION):
        raise ValueError('Probe carries no usable reset contract for this source episode')
    if contract.get('layout_randomization') is not None or contract.get('role') != 'source':
        raise ValueError('A recorded trajectory is only replayed on the layout it was recorded on')
    flags = Randomization.model_validate(contract.get('randomization', {})).model_dump()
    if any(flags.values()):
        raise ValueError('House randomization is not part of a portable replay')
    if data['execution_profile'] != pe.spec(rp.JOINT20):
        raise ValueError('Public controller changed after conversion')
    checkpoint = json.loads((probe / 'measured_checkpoint.json').read_text())
    initial = {n: float(checkpoint['joint_positions_native'][n]) for n in rp.get(rp.JOINT20).joints}
    # matching tolerances only; they neither relax task success nor move joints
    tolerances = {n: .001 if unit == 'm' else .005 for n, unit in rp.get(rp.JOINT20).units.items()}
    limits = metadata['action_limits_native']
    for row in data['actions']:
        pe.validate(row, limits, rp.JOINT20)
    result = dict(
        schema_version=trajectory.SCHEMA20, execution_profile=data['execution_profile'],
        reset=dict(task_id=api, seed=manifest['seed'], randomization=flags, task_parameters=parameters, task_contract=contract,
                   joint_positions_native=initial, joint_tolerances_native=tolerances),
        actions=data['actions'],
        provenance=dict(source_kind='explicit_recorded_command_conversion', rule_version=RULE_VERSION, evaluator_id=EVALUATOR_ID,
                        candidate_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(), probe_episode_id=manifest['episode_id'],
                        probe_manifest_sha256=hashlib.sha256((probe / 'manifest.json').read_bytes()).hexdigest(),
                        probe_metadata_sha256=hashlib.sha256((probe / 'initial-metadata.json').read_bytes()).hexdigest(),
                        source=data['source'], adaptation=data['adaptation'], physical_success_not_established=True))
    result['content_sha256'] = digest(result)
    return trajectory.validate(result)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--probe', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(argv)
    out = a.output.resolve()
    if out.exists() or not out.is_relative_to((TASK / 'artifacts').resolve()):
        raise ValueError('New task-owned output required')
    result = bind(a.candidate, a.probe)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as f:
        json.dump(result, f, indent=1, allow_nan=False)
    print(json.dumps(dict(output=str(out), actions=len(result['actions']), rule_version=RULE_VERSION, physical_success=False)))


if __name__ == '__main__':
    main()
