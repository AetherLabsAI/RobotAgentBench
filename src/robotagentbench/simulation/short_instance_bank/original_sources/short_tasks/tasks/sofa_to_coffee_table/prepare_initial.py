"""Approved station rectangle + measured generic folded posture, actual-room checks."""
import hashlib
from pathlib import Path
from . import initial_station
from short_tasks.agent_probe_v1.target_spawn import NoFeasibleTaskFront as NoFeasibleInitial


def choose(stage, manifest, *, seed, sensor):
    from short_tasks.agent_probe_v1 import target_spawn
    from openend.tasks.layout_repair_v1 import ready_envelope
    delta,context,_=initial_station.sofa_context(stage,manifest)
    if context['asset']=='remote_tv':
        from . import posture_geometry
        receipt=posture_geometry.choose(stage,manifest,seed,context,delta)
        receipt['reference_hashes'][str(Path(__file__))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        return receipt
    target=target_spawn.target_geometry(stage,manifest,'sofa_to_coffee_table')
    rows=initial_station.sample_reset_candidates(seed,delta,context,count=100)
    receipt=target_spawn.choose(stage,manifest,'sofa_to_coffee_table',sensor,candidate_rows=rows,target=target)
    receipt.update(method='approved_sofa_table_station_measured_folded_envelope',
        targets_rad=receipt['candidate']['targets_rad'],target_transforms_unchanged=True,
        native_physics_validated=False,robot_room_geometry_checked=True,
        posture_source='approved_generic_folded_ready_not_dataset_fit')
    refs=[Path(__file__),Path(initial_station.__file__),initial_station.POLICY,
        Path(target_spawn.__file__),Path(ready_envelope.__file__),
        ready_envelope.REFERENCE/'navigation_result.json',ready_envelope.REFERENCE/'native_poses.npz',
        *map(Path,context['reference_files'])]
    receipt['reference_hashes']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in refs}
    return receipt


def check_room_candidate(manifest, *, seed, sensor):
    from pxr import Usd
    return choose(Usd.Stage.Open(manifest['scene']),manifest,seed=seed,sensor=sensor)


def refresh_manifest(stage, manifest, receipt):
    manifest['sofa_table_station_initialization']=receipt
    if 'sofa_table_source_initialization' in manifest:
        manifest['sofa_table_source_initialization']['posture']=receipt['posture_source']
