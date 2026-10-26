"""Connect source50 fridge station and recorded arms; preserve box scene hook."""
import hashlib
from pathlib import Path
from . import initial_station, initial_posture
from short_tasks.tasks.stock_wooden_tray import station_geometry
from short_tasks.tasks.retrieve_from_drawer import prepare_initial as fk_source


class NoFeasibleInitial(ValueError):
    pass


def choose(stage, manifest, *, seed, count=256):
    frame = initial_station.fridge_frame(stage, manifest)
    targets = initial_posture.read()['targets_rad']
    measured = station_geometry.geometry(stage, manifest)
    rejections = []
    for candidate in initial_station.sample_candidates(frame, seed, count):
        reason = station_geometry.rejection(stage, measured, targets,
            candidate['position_xy'], candidate['yaw_rad'])
        if reason:
            rejections.append(dict(candidate_index=candidate['candidate_index'], reason=reason))
            continue
        candidate['sampling'] = 'existing_fridge_source50_continuous_ranges_recorded_arms'
        refs = [Path(__file__), Path(initial_station.__file__), Path(initial_posture.__file__),
                initial_posture.SOURCE, Path(station_geometry.__file__), Path(fk_source.__file__)]
        return dict(position=[*candidate['position_xy'], 0.], yaw_rad=candidate['yaw_rad'],
            targets_rad=targets, candidate=candidate, frame=frame,
            target=dict(path=frame['fridge_path'], reference_files=list(map(str, refs))),
            method='source_fridge_station_recorded_arms_actual_room_fk',
            rejections=rejections, candidates_checked=candidate['candidate_index'] + 1,
            object_transforms_unchanged=True, robot_room_geometry_checked=True,
            robot_self_collision_validated=False, native_physics_validated=False,
            reference_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in refs})
    raise NoFeasibleInitial('Fridge station range exhausted without widening: ' + str(rejections))


def apply_target(stage, receipt):
    if receipt.get('object_transforms_unchanged') is not True:
        raise ValueError('Fridge boxes must remain owned by the existing scene hook')


def refresh_manifest(stage, manifest, receipt):
    manifest['fridge_station_initialization'] = receipt
