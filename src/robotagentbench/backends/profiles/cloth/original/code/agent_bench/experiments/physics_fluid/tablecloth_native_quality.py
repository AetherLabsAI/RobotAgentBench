"""Measure native cloth diagnostics against the existing owner's quality limits.

This is a physical-quality check, not a replacement for any Sheet success gate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def measure(episode, quality_limits=None, *, require_all_steps=False):
    episode = Path(episode)
    trial = json.loads((episode/'trial.json').read_text())
    limits_source = 'explicit_owner_limits' if quality_limits is not None else 'reference_policy'
    if quality_limits is None:
        reference_path = Path(trial['reference'])/'trial.json'
        reference = json.loads(reference_path.read_text()) if reference_path.is_file() else {}
        quality_limits = reference.get('policy', {}).get('quality_guard')
    limits = quality_limits
    missing_limits = limits is None
    if not missing_limits:
        if not isinstance(limits, dict) or any(
                key not in limits or isinstance(limits[key], bool)
                or not np.isfinite(limits[key]) or limits[key] <= 0
                for key in ('max_edge_strain', 'max_particle_speed_m_s')):
            raise ValueError('Owner quality limits must contain positive finite strain and speed bounds')
    with np.load(episode/'full_state.npz', allow_pickle=False) as z:
        q, qd, uv, faces = z['cloth_q'], z['cloth_qd'], z['uv'], z['faces']
        frame_count = len(q)
        frame_indices = np.arange(frame_count)
        all_steps = False
        step_path = episode/'substep_observations.npz'
        if step_path.is_file():
            with np.load(step_path, allow_pickle=False) as steps:
                if bool(steps['all_post_reset_steps_recorded']):
                    hz = float(trial['physics_clock']['actual_hz'])
                    fps = float(trial['fps'])
                    stride = int(round(hz/fps))
                    times = steps['time_s']
                    expected = np.arange(1, frame_count*stride+1)/hz
                    if (stride < 1 or not np.isclose(hz, fps*stride)
                            or times.shape != expected.shape
                            or not np.allclose(times, expected, atol=1e-7, rtol=0)
                            or steps['cloth_q'].shape != (len(expected), *q.shape[1:])
                            or steps['cloth_qd'].shape != steps['cloth_q'].shape
                            or not np.array_equal(steps['cloth_id'], z['cloth_id'])
                            or not np.allclose(steps['cloth_q'][stride-1::stride], q, atol=1e-7, rtol=0)
                            or not np.allclose(steps['cloth_qd'][stride-1::stride], qd, atol=1e-7, rtol=0)):
                        raise ValueError('Incomplete or inconsistent all-physics-step capture')
                    q, qd = steps['cloth_q'], steps['cloth_qd']
                    frame_indices = np.arange(len(q))//stride
                    all_steps = True
        if require_all_steps and not all_steps:
            raise ValueError('All physics steps are required; action-rate frames are insufficient')
        if (q.ndim != 3 or not len(q) or q.shape[-1] != 3 or qd.shape != q.shape
                or uv.shape != (q.shape[1], 2) or faces.ndim != 2 or faces.shape[1] != 3
                or not len(faces) or faces.dtype.kind not in 'iu'
                or faces.min() < 0 or faces.max() >= q.shape[1]
                or not np.isfinite(uv).all()):
            raise ValueError('Invalid native state dimensions or topology')
        edges = np.unique(np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]],
                                                  faces[:, [2, 0]]]), axis=1), axis=0)
        rest = np.linalg.norm(uv[edges[:, 1]]-uv[edges[:, 0]], axis=1)
        if not np.isfinite(rest).all() or np.any(rest <= 0):
            raise ValueError('Degenerate material edge')
        peak_stretch, peak_speed, failures = [], [], []
        for frame, (points, velocities) in enumerate(zip(q, qd)):
            stretch = np.linalg.norm(points[edges[:, 1]]-points[edges[:, 0]], axis=1)/rest
            speed = np.linalg.norm(velocities, axis=1)
            peak_stretch.append(float(stretch.max()))
            peak_speed.append(float(speed.max()))
            if (not np.isfinite(points).all() or not np.isfinite(velocities).all()
                    or (not missing_limits and (stretch.max() > limits['max_edge_strain']
                    or speed.max() > limits['max_particle_speed_m_s']))):
                failures.append(frame)
        first = failures[0] if failures else None
        per_cloth = {}
        for cid in np.unique(z['cloth_id']):
            selected = z['cloth_id'] == cid
            local_edges = edges[np.all(selected[edges], axis=1)]
            local_rest = np.linalg.norm(uv[local_edges[:, 1]]-uv[local_edges[:, 0]], axis=1)
            values = [float((np.linalg.norm(points[local_edges[:, 1]]-points[local_edges[:, 0]], axis=1)/local_rest).max())
                      for points in q]
            bad = (np.flatnonzero(np.asarray(values) > limits['max_edge_strain'])
                   if not missing_limits else np.asarray([], dtype=int))
            per_cloth[str(int(cid))] = dict(peak_edge_length_over_rest=max(values),
                first_failed_frame=int(frame_indices[bad[0]]) if len(bad) else None,
                initial_bounds=[q[0, selected].min(0).tolist(), q[0, selected].max(0).tolist()],
                final_bounds=[q[-1, selected].min(0).tolist(), q[-1, selected].max(0).tolist()])
        return dict(frames=frame_count, samples=len(q), quality_limits=limits,
                    quality_limits_source=limits_source if not missing_limits else None,
                    quality_pass=not failures and not missing_limits,
                    reasons=(['owner_quality_limits_missing'] if missing_limits else [])
                        + (['physical_quality_failed'] if failures else []),
                    coverage=('all recorded post-reset physics steps' if all_steps else
                              'saved action-rate frames; substeps require a separate audit'),
                    all_post_reset_steps_checked=all_steps,
                    initialization_warmup_checked=False,
                    first_failed_sample=first,
                    first_failed_frame=int(frame_indices[first]) if first is not None else None,
                    first_failed_phase=str(z['phase'][frame_indices[first]]) if first is not None else None,
                    peak_edge_length_over_rest=max(peak_stretch),
                    peak_particle_speed_m_s=max(peak_speed),
                    per_cloth=per_cloth,
                    positive_solver_masses=(bool(np.all(z['particle_mass'] > 0))
                        if trial.get('particle_mass_source', 'native_GPU_tensors') == 'native_GPU_tensors' else None),
                    mass_record_source=trial.get('particle_mass_source', 'native_GPU_tensors'),
                    rigid_force_signal_observed=bool(np.any(z['rigid_contact_force_peak'] != 0)),
                    rigid_force_note='Zero signals do not prove absence of collisions or a supported grasp',
                    task_success_verified=False, eligible_for_production=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('episode', type=Path)
    p.add_argument('--owner-limits', type=Path, help='Delivered owner quality_guard JSON; no default thresholds')
    p.add_argument('--output', type=Path, help='Optional writable report path')
    p.add_argument('--require-all-steps', action='store_true',
                   help='Reject missing/incomplete physics-step captures; complete captures are always preferred')
    args = p.parse_args()
    limits = json.loads(args.owner_limits.read_text()) if args.owner_limits else None
    result = measure(args.episode, limits, require_all_steps=args.require_all_steps)
    (args.output or args.episode/'physical_quality.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['quality_pass'] else 1)


if __name__ == '__main__':
    main()
