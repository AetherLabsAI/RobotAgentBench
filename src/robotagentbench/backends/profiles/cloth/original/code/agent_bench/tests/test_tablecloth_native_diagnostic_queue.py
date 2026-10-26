import json
import subprocess

import pytest

from experiments.physics_fluid.tablecloth_native_diagnostic_queue import run


def plan(root):
    limits = root/'limits.json'
    limits.write_text('{"max_edge_strain":1.15,"max_particle_speed_m_s":5}')
    cases = [dict(id=f'case_{i}', output=str(root/f'episode_{i}'), command=[
        'docker','run','--name',f'owned_{i}','--diagnostic-canonical',
        '--output',str(root/f'episode_{i}')]) for i in range(2)]
    path = root/'plan.json'
    path.write_text(json.dumps(dict(diagnostic_only=True, cases=cases,
        owner_limits=str(limits), simulation_timeout_s=1, review_timeout_s=1)))
    return path


def test_timeout_stops_only_owned_container_and_stops_queue(tmp_path, monkeypatch):
    path = plan(tmp_path)
    calls = []

    def execute(command, **kwargs):
        calls.append(command)
        if command[:2] == ['docker','inspect']:
            return subprocess.CompletedProcess(command, 1)
        if command[:2] == ['docker','run']:
            raise subprocess.TimeoutExpired(command, 1)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, 'run', execute)
    result = run(path)
    assert len(result) == 1 and not result[0]['qualified']
    assert calls[-1] == ['docker','stop','--timeout','10','owned_0']
    assert len([c for c in calls if c[:2] == ['docker','run']]) == 1
    assert json.loads((tmp_path/'QUEUE_STATUS.json').read_text())['phase'] == 'stopped_after_timeout'


def test_detached_commands_are_rejected_before_execution(tmp_path):
    path = plan(tmp_path)
    value = json.loads(path.read_text())
    value['cases'][1]['command'].insert(2, '-d')
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='foreground'):
        run(path)


def test_completed_attempts_are_preserved_on_resume(tmp_path, monkeypatch):
    path = plan(tmp_path)
    for i in range(2):
        (tmp_path/f'case_{i}_queue_result.json').write_text(json.dumps(dict(id=f'case_{i}',qualified=False)))

    def unexpected(*args, **kwargs):
        raise AssertionError('Completed attempts must not rerun')

    monkeypatch.setattr(subprocess, 'run', unexpected)
    assert len(run(path)) == 2
    assert len(json.loads((tmp_path/'QUEUE_RESULTS.json').read_text())['results']) == 2
