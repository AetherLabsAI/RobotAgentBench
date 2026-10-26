"""Read-only binding to one scene-owned remote. No USD writes or reset hooks.

Geometry is for recording/localization only: neither a bounding box nor proximity
proves holding, support, placement or task success.
"""
from dataclasses import asdict, dataclass
import math


def vector(values, size):
    values = tuple(values)
    if len(values) != size or any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
        raise ValueError('Finite measured vector required')
    return values


@dataclass(frozen=True)
class Pose:
    position_m: tuple
    quaternion_wxyz: tuple

    def __post_init__(self):
        object.__setattr__(self, 'position_m', vector(self.position_m, 3))
        q = vector(self.quaternion_wxyz, 4)
        if not math.isclose(sum(v*v for v in q), 1, abs_tol=1e-6):
            raise ValueError('Unit quaternion required; do not silently normalize evidence')
        object.__setattr__(self, 'quaternion_wxyz', q)


@dataclass(frozen=True)
class Snapshot:
    episode_id: str
    target_path: str
    sample_index: int
    simulation_time_s: float
    pose: Pose


class RemoteBinding:
    """One instance per episode; trusted reader supplies current measured poses."""
    def __init__(self, manifest, *, episode_id, source_region, source_furniture_path,
                 destination_path, initial, initial_supports=None):
        matches = [o for o in manifest['objects'] if o['id'] == 'remote_tv']
        if len(matches) != 1:
            raise ValueError('Scene must supply exactly one remote_tv; never spawn a replacement')
        if not isinstance(source_region, str) or not source_region.strip():
            raise ValueError('Source description required; not a location whitelist')
        self.target_path = matches[0]['path']
        paths = (self.target_path, source_furniture_path, destination_path)
        if any(not isinstance(p, str) or not p.startswith('/') for p in paths) or self.target_path in paths[1:]:
            raise ValueError('Resolved support/destination paths distinct from target required')
        self.initial_supports = frozenset((source_furniture_path,)) if initial_supports is None else frozenset(initial_supports)
        if not self.initial_supports or any(not isinstance(p,str) or not p.startswith('/') or p==self.target_path for p in self.initial_supports):
            raise ValueError('Nonempty measured initial support identities required')
        if not isinstance(episode_id, str) or not episode_id.strip():
            raise ValueError('Episode identity required')
        self.episode_id = episode_id
        self.source_region = source_region
        self.source_furniture_path = source_furniture_path
        self.destination_path = destination_path
        self.asset_entry = matches[0]['entry']
        self._validate(initial)
        if initial.sample_index != 0:
            raise ValueError('Initial snapshot index must be zero')
        self.initial = self.current = initial

    def _validate(self, sample):
        if not isinstance(sample, Snapshot) or not isinstance(sample.pose, Pose):
            raise ValueError('Typed measured snapshot required')
        if sample.episode_id != self.episode_id or sample.target_path != self.target_path:
            raise ValueError('Cross-episode or wrong-object evidence')
        if type(sample.sample_index) is not int or sample.sample_index < 0:
            raise ValueError('Invalid sample index')
        t = sample.simulation_time_s
        if type(t) not in (int, float) or not math.isfinite(t) or t < 0:
            raise ValueError('Invalid simulation time')

    def observe(self, sample):
        self._validate(sample)
        if sample.sample_index != self.current.sample_index + 1 or sample.simulation_time_s <= self.current.simulation_time_s:
            raise ValueError('Duplicate, stale or missing observation')
        self.current = sample
        return self.receipt()

    def receipt(self):
        return dict(schema='openend.remote_transfer.binding.v1', episode_id=self.episode_id,
                    target_path=self.target_path, asset_id='remote_tv', asset_entry=self.asset_entry,
                    source_region=self.source_region, source_furniture_path=self.source_furniture_path,
                    initial_supports=sorted(self.initial_supports), source_is_task_constraint=False,
                    destination_path=self.destination_path, initial=asdict(self.initial), current=asdict(self.current),
                    initialization_owner='unified_scene', scene_writes=0,
                    source_support_verified=False, destination_zone_verified=False,
                    scoring_status='not_connected_rule_alignment_required',
                    physical_validation=False)


def inspect_scene(manifest, *, episode_id, source_region, source_furniture_path, destination_path):
    """Inspect composed USD without a simulator or any authored scene changes."""
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.Open(manifest['scene'])
    if stage is None:
        raise ValueError('Cannot open unified scene')
    rows = [o for o in manifest['objects'] if o['id'] == 'remote_tv']
    if len(rows) != 1:
        raise ValueError('Missing or duplicate remote in scene inventory')
    target = rows[0]['path']
    paths = (target, source_furniture_path, destination_path)
    for path in paths:
        p = stage.GetPrimAtPath(path)
        if not p or not p.IsActive() or not p.IsLoaded():
            raise ValueError('Missing/inactive/unloaded scene binding: '+path)
    before = {l.identifier: l.ExportToString() for l in stage.GetUsedLayers()}
    cache = UsdGeom.XformCache()
    matrix = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(target))
    rotation = matrix.ExtractRotationQuat()
    pose = Pose(tuple(matrix.ExtractTranslation()), (rotation.GetReal(), *rotation.GetImaginary()))
    initial = Snapshot(episode_id, target, 0, 0.0, pose)
    bound = RemoteBinding(manifest, episode_id=episode_id, source_region=source_region,
                          source_furniture_path=source_furniture_path,
                          destination_path=destination_path, initial=initial)
    receipt = bound.receipt()
    receipt['evidence_source'] = 'composed_USD_static_not_native_physics'
    receipt['initial']['simulation_time_s'] = None  # static USD has no simulation clock
    receipt['current']['simulation_time_s'] = None
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render'], useExtentsHint=False)
    receipt['geometry'] = {}
    for path in paths:
        prim = stage.GetPrimAtPath(path)
        box = bbox.ComputeWorldBound(prim).ComputeAlignedRange()
        if box.IsEmpty():
            raise ValueError('Empty geometry: '+path)
        receipt['geometry'][path] = dict(world_aabb_m=[list(box.GetMin()), list(box.GetMax())],
            local_to_world_row_matrix=[list(r) for r in cache.GetLocalToWorldTransform(prim)])
    receipt['layers_unchanged'] = all(l.ExportToString() == before[l.identifier] for l in stage.GetUsedLayers())
    if not receipt['layers_unchanged']:
        raise RuntimeError('Read-only binding mutated scene')
    return receipt
