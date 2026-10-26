import json
import sys
import subprocess
from pathlib import Path

import numpy as np
import pytest

from experiments.physics_fluid.tablecloth_native_batch import (
    prepare, run_diagnostics, validate_native_capture, solver_log_errors,
)


def capture(root, *, state_clock_offset=.05, failure=None):
    root.mkdir(exist_ok=True)
    (root/'trial.json').write_text(json.dumps(dict(fps=20,frames=2,failure=failure,
        post_start_pose_or_velocity_writes=False,attachment_count=0,robot_parameters_unchanged=True)))
    np.savez(root/'action.npz',joint_names=['j1'],joint_target=[[.1],[.2]],time_s=[0.,.05])
    np.savez(root/'full_state.npz',joint_names=['j1'],joint_position=[[.08],[.16]],
        joint_velocity=[[0.],[.1]],time_s=np.array([0.,.05])+state_clock_offset,
        cloth_q=np.zeros((2,3,3)),cloth_qd=np.zeros((2,3,3)))


def test_native_capture_keeps_actual_tracking_error(tmp_path):
    capture(tmp_path)
    result = validate_native_capture(tmp_path)
    assert result['capture_valid']
    assert result['metrics']['tracking_error_max_native_units'] == pytest.approx(.04)
    assert not result['eligible_for_production']


def test_wrong_clock_and_incomplete_capture_rejected(tmp_path):
    capture(tmp_path,state_clock_offset=0.,failure='incomplete')
    result = validate_native_capture(tmp_path)
    assert not result['capture_valid']
    assert 'invalid_measured_state_clock' in result['reasons']
    assert 'simulation_incomplete_or_failed' in result['reasons']


def test_plan_has_fifty_per_workflow_and_no_qualification(tmp_path):
    result = prepare(tmp_path/'batch',2026101100,50)
    assert result['target'] == {'remove':50,'spread':50}
    assert len(result['rows']) == 100
    assert len({r['seed'] for r in result['rows']}) == 100
    for workflow in ('remove','spread'):
        rows = [r for r in result['rows'] if r['workflow']==workflow]
        assert len({r['layout_id'] for r in rows}) == 50
        assert all(not r['eligible_for_production'] for r in rows)


def test_failed_process_preserved_and_retry_explicit(tmp_path):
    root = tmp_path/'batch'
    prepare(root,1000001,1)
    command = [sys.executable,'-c','raise SystemExit(3)', '--diagnostic-canonical','{output}']
    commands = tmp_path/'commands.json'
    commands.write_text(json.dumps(dict(remove=command,spread=command)))
    first = run_diagnostics(root,commands,1)
    assert first[0]['returncode'] == 3 and not first[0]['qualified']
    second = run_diagnostics(root,commands,1)
    assert first[0]['request_id'] != second[0]['request_id']
    assert run_diagnostics(root,commands,1) == []
    third = run_diagnostics(root,commands,1,retry_failed=True)
    assert third[0]['request_id'] == first[0]['request_id']
    assert len(list((root/'attempts'/first[0]['request_id']).glob('attempt_*'))) == 2


def test_detached_process_cannot_masquerade_as_finished(tmp_path):
    root = tmp_path/'batch'
    prepare(root,1000001,1)
    commands = tmp_path/'commands.json'
    command = ['docker','run','-d','--diagnostic-canonical','{output}']
    commands.write_text(json.dumps(dict(remove=command,spread=command)))
    with pytest.raises(ValueError,match='foreground'):
        run_diagnostics(root,commands,1)


def test_cli_propagates_failed_attempt(tmp_path):
    root = tmp_path/'batch'
    prepare(root,1000001,1)
    commands = tmp_path/'commands.json'
    command = [sys.executable,'-c','raise SystemExit(3)','--diagnostic-canonical','{output}']
    commands.write_text(json.dumps(dict(remove=command,spread=command)))
    script = Path(__file__).resolve().parents[1]/'experiments/physics_fluid/tablecloth_native_batch.py'
    result = subprocess.run([sys.executable,str(script),'run-diagnostic','--root',str(root),
        '--commands',str(commands),'--limit','1'],capture_output=True,text=True)
    assert result.returncode == 1
    assert 'process_exit:3' in result.stdout


def test_runtime_contact_overflow_is_not_a_successful_exit(tmp_path):
    log = tmp_path/'physics.log'
    error = '[Error] [omni.physx.plugin] PhysX error: Deformable surface contact buffer overflow detected'
    log.write_text('normal frame\n'+error+'\n'+error+'\n[Warning] [omni.usd] material binding warning\n')
    assert solver_log_errors(log) == ['PhysX error: Deformable surface contact buffer overflow detected']


def test_zero_exit_with_complete_capture_and_solver_error_is_rejected(tmp_path):
    reference = tmp_path/'reference'
    capture(reference)
    assert validate_native_capture(reference)['capture_valid']
    root = tmp_path/'batch'
    prepare(root,1000001,1)
    commands = tmp_path/'commands.json'
    script = ('import shutil,sys; shutil.copytree(sys.argv[1],sys.argv[2]); '
              'print("[Error] [omni.physx.plugin] Deformable surface contact buffer overflow")')
    command = [sys.executable,'-c',script,str(reference),'{output}','--diagnostic-canonical']
    commands.write_text(json.dumps(dict(remove=command,spread=command)))
    result = run_diagnostics(root,commands,1)[0]
    assert result['returncode'] == 0
    assert not result['capture']['capture_valid']
    assert result['capture']['reasons'] == ['physx_runtime_error']
    assert result['solver_errors'] and not result['qualified']


def test_missing_executable_leaves_failed_attempt_record(tmp_path):
    root = tmp_path/'batch'
    prepare(root,1000001,1)
    commands = tmp_path/'commands.json'
    command = [str(tmp_path/'missing'),'--diagnostic-canonical','{output}']
    commands.write_text(json.dumps(dict(remove=command,spread=command)))
    result = run_diagnostics(root,commands,1)
    assert result[0]['returncode'] == 127 and not result[0]['qualified']
