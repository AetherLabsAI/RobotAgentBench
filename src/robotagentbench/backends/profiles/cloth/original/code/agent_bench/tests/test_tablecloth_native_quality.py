import json
import numpy as np
import pytest

from experiments.physics_fluid.tablecloth_native_quality import measure


def episode(root):
    reference = root/'reference'
    reference.mkdir()
    (reference/'trial.json').write_text('{}')
    (root/'trial.json').write_text(json.dumps(dict(reference=str(reference),
        particle_mass_source='authored_total_and_rest_area_estimates')))
    uv = np.array([[0.,0.],[1.,0.],[0.,1.]])
    q = np.tile(np.c_[uv,np.zeros(3)],(2,1,1))
    np.savez(root/'full_state.npz',cloth_q=q,cloth_qd=q*0,uv=uv,faces=[[0,1,2]],
        cloth_id=[0,0,0],phase=['hold','hold'],particle_mass=[1.,1.,1.],
        rigid_contact_force_peak=np.zeros((2,1,3)))


def test_missing_owner_limits_reports_measurements_without_approval(tmp_path):
    episode(tmp_path)
    result = measure(tmp_path)
    assert result['peak_edge_length_over_rest'] == pytest.approx(1.)
    assert not result['quality_pass']
    assert result['reasons'] == ['owner_quality_limits_missing']
    assert result['positive_solver_masses'] is None


def test_explicit_limits_check_quality_without_claiming_task_success(tmp_path):
    episode(tmp_path)
    result = measure(tmp_path,dict(max_edge_strain=1.01,max_particle_speed_m_s=1.))
    assert result['quality_pass'] and not result['task_success_verified']
    assert not result['eligible_for_production']
    with pytest.raises(ValueError,match='positive finite'):
        measure(tmp_path,dict(max_edge_strain=float('nan'),max_particle_speed_m_s=1.))


def test_intermediate_physics_step_failure_cannot_hide_between_saved_frames(tmp_path):
    episode(tmp_path)
    trial = json.loads((tmp_path/'trial.json').read_text())
    trial.update(fps=20, physics_clock=dict(actual_hz=60))
    (tmp_path/'trial.json').write_text(json.dumps(trial))
    with np.load(tmp_path/'full_state.npz') as z:
        q = np.repeat(z['cloth_q'], 3, axis=0)
    q[1, 1, 0] = 1.2
    values = dict(cloth_q=q, cloth_qd=q*0, time_s=np.arange(1,7)/60,
                  cloth_id=[0,0,0], all_post_reset_steps_recorded=True)
    np.savez(tmp_path/'substep_observations.npz', **values)
    limits = dict(max_edge_strain=1.15, max_particle_speed_m_s=5.)
    result = measure(tmp_path, limits, require_all_steps=True)
    assert not result['quality_pass']
    assert result['frames'] == 2 and result['samples'] == 6
    assert result['first_failed_sample'] == 1 and result['first_failed_frame'] == 0
    assert result['all_post_reset_steps_checked']
    values['time_s'] = np.arange(6)/60
    np.savez(tmp_path/'substep_observations.npz', **values)
    with pytest.raises(ValueError, match='Incomplete or inconsistent'):
        measure(tmp_path, limits)


def test_full_step_requirement_rejects_action_rate_only(tmp_path):
    episode(tmp_path)
    with pytest.raises(ValueError, match='All physics steps are required'):
        measure(tmp_path, require_all_steps=True)
