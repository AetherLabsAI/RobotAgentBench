"""Failure-only operator diagnostics from the meal-tray reader's cached data.

No native reader, detector, contact reducer, clock or physics update is called.
This sidecar is not an observation, an acceptance override or scoring evidence.
"""
from collections import deque
import json
import math
from pathlib import Path

TARGET = '/World/RoomEpisode/MealTray'
NAME = 'initial_failure_diagnostic.json'
MAX_PAIRS = 128
MAX_SAMPLES = 16
SETTLING_SAMPLES = 240


def _number(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('Finite cached diagnostic value required')
    return value


def _vector(values, size):
    result = [_number(value) for value in values]
    if len(result) != size:
        raise ValueError('Original cached vector shape required')
    return result


def _paths(values):
    if not all(isinstance(value, str) for value in values):
        raise ValueError('Cached native prim paths required')
    return list(values)


def _row(row):
    flags = {name: row[name] for name in ('held', 'robot_contact', 'on_floor')}
    if any(type(value) is not bool for value in flags.values()):
        raise ValueError('Original cached Boolean flags required')
    pose = vars(row['pose'])
    return dict(time_s=_number(row['time_s']), physics_dt_s=_number(row['physics_dt_s']),
                **flags, supports=sorted(_paths(row['supports'])),
                pose=dict(position_m=_vector(pose['position_m'], 3),
                          quaternion_wxyz=_vector(pose['quaternion_wxyz'], 4)))


class MealTraySettlingCache:
    """Bounded copies of rows already produced by the unchanged settling loop.

    No file is written here. Collection errors disable this diagnostic only;
    neither a diagnostic exception nor malformed cached data may affect native
    execution. The first unobserved startup step is deliberately not invented.
    """

    def __init__(self, session):
        self.status = 'not_applicable'
        self.rows = []
        self.initial_pose = None
        self.error_type = None
        self.port = None
        try:
            ports = vars(session).get('ports')
            if not isinstance(ports, dict) or 'meal_tray' not in ports:
                return
            port = vars(ports['meal_tray'])
            geometry = vars(port['geometry'])
            if geometry.get('target') != TARGET:
                return
            position, quaternion = geometry['initial_pose']
            self.initial_pose = dict(position_m=_vector(position, 3),
                                     quaternion_wxyz=_vector(quaternion, 4))
            self.port = port
            self.status = 'collecting'
        except BaseException as error:
            self.status = 'unavailable'
            self.error_type = type(error).__name__

    def capture(self):
        """Call once after the original after_physics; do not query the scene."""
        if self.status != 'collecting':
            return
        try:
            if len(self.rows) >= SETTLING_SAMPLES:
                raise ValueError('Unexpected additional settling sample')
            history = self.port['history']
            if not isinstance(history, deque) or not history:
                raise ValueError('Existing settling history cache required')
            row = _row(history[-1])
            if abs(row['physics_dt_s'] - 1/60) > 1e-10:
                raise ValueError('Original 60 Hz settling cache required')
            if self.rows and abs(row['time_s'] - self.rows[-1]['time_s'] - 1/60) > 1e-6:
                raise ValueError('Nonconsecutive original settling cache')
            self.rows.append(row)
        except BaseException as error:
            self.status = 'unavailable'
            self.error_type = type(error).__name__

    def evidence(self):
        status = self.status
        if status == 'collecting':
            status = 'complete' if len(self.rows) == SETTLING_SAMPLES else 'partial'
        return dict(status=status, cache_only=True, native_state_modified=False,
                    expected_sample_count=SETTLING_SAMPLES,
                    recorded_sample_count=len(self.rows),
                    initial_pose=self.initial_pose,
                    initial_pose_source='existing_geometry_cache_before_physics_attach',
                    first_startup_physics_step_observed=False,
                    error_type=self.error_type, rows=self.rows)


def meal_tray_cache(session):
    """Copy only existing cache attributes; never ask the simulator for data."""
    ports = vars(session).get('ports')
    if not isinstance(ports, dict) or 'meal_tray' not in ports:
        return None
    port = vars(ports['meal_tray'])
    geometry = vars(port['geometry'])
    if geometry.get('target') != TARGET:
        return None
    history = port['history']
    if not isinstance(history, deque) or not history:
        raise ValueError('Existing meal-tray history cache required')
    # Match the original initial_window sample count, without calling the
    # clock or changing/re-evaluating any admission predicate.
    window = _number(vars(port['config'])['initial_history_s'])
    dt = _number(history[-1]['physics_dt_s'])
    if window <= 0 or not any(abs(dt - value) < 1e-10 for value in (1/60, 1/240)):
        raise ValueError('Original cached initial window required')
    count = math.ceil(window / dt - 1e-9) + 1
    rows = [_row(row) for row in list(history)[-count:]]
    active = vars(port['contacts'])['active']
    if not isinstance(active, dict):
        raise ValueError('Original cached contact dictionary required')
    pairs = []
    relevant = 0
    for paths, _stage, _actors, samples in active.values():
        paths = _paths(paths)
        if (not samples or not any(path == TARGET or path.startswith(TARGET + '/') for path in paths)):
            continue
        relevant += 1
        if len(pairs) >= MAX_PAIRS:
            continue
        pairs.append(dict(paths=paths, cached_sample_count=len(samples),
            samples_truncated=len(samples) > MAX_SAMPLES,
            samples=[dict(normal=_vector(normal, 3), position_m=_vector(position, 3),
                          impulse=_vector(impulse, 3))
                     for normal, position, impulse in samples[:MAX_SAMPLES]]))
    return dict(schema='robotagentbench.native_initial_failure_cache.v1',
                purpose='operator_diagnostic_only', instance_id='meal_tray',
                target=TARGET, body=geometry['body'], robot_root=port['robot'],
                floor_roots=_paths(geometry['floors']),
                cache_only=True, native_state_modified=False,
                initial_history_s=window, physics_dt_s=dt,
                requested_sample_count=count, available_history_samples=len(history),
                recorded_sample_count=len(rows), rows=rows,
                any_in_window={name: any(row[name] for row in rows)
                               for name in ('held', 'robot_contact', 'on_floor')},
                latest_target_contact_pairs=pairs,
                latest_target_pair_count=relevant, contact_pairs_truncated=relevant > MAX_PAIRS)


def accept_initial_with_diagnostics(session, output, report, *, settling=None):
    """Preserve the original return/exception; diagnostics cannot mask either."""
    try:
        return session.accept_initial()
    except BaseException:
        try:
            evidence = meal_tray_cache(session)
            if evidence is None:
                report['initial_failure_diagnostic'] = dict(status='not_applicable')
            else:
                if settling is not None:
                    evidence['settling'] = settling.evidence()
                # Serialize before creating the fresh output, so malformed
                # cached data cannot leave a misleading successful sidecar.
                raw = json.dumps(evidence, indent=2, allow_nan=False) + '\n'
                path = Path(output) / NAME
                with path.open('x', encoding='utf-8') as stream:
                    stream.write(raw)
                report['initial_failure_diagnostic'] = dict(status='written', path=str(path),
                    cache_only=True, native_state_modified=False)
        except BaseException as diagnostic_error:
            report['initial_failure_diagnostic'] = dict(status='unavailable',
                error_type=type(diagnostic_error).__name__)
        raise
