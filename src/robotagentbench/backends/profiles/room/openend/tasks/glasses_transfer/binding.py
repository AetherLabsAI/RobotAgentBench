"""Task-local read-only scene identity binding; not a physical reader or scorer."""
from dataclasses import asdict
import math

from openend.tasks.remote_transfer.binding import Pose, Snapshot
from openend.tasks.remote_transfer.selection import ContactSelection


class GlassesBinding:
    def __init__(self, manifest, *, episode_id, source_path, bag_path,
                 cabinet_path, initial, initial_supports=None):
        rows = [o for o in manifest['objects'] if o['id'] == 'glasses_case']
        if len(rows) != 1:
            raise ValueError('Scene must own exactly one glasses_case')
        self.target = rows[0]['path']
        paths = (self.target, source_path, bag_path, cabinet_path)
        if any(not isinstance(p, str) or not p.startswith('/') for p in paths):
            raise ValueError('Explicit resolved scene paths required; never guess the bag')
        if self.target in paths[1:] or bag_path == cabinet_path:
            raise ValueError('Target, bag and cabinet must remain distinct')
        self.initial_supports = frozenset((source_path,)) if initial_supports is None else frozenset(initial_supports)
        if not self.initial_supports or any(not isinstance(p,str) or not p.startswith('/') or p==self.target for p in self.initial_supports):
            raise ValueError('Nonempty measured initial support identities required')
        if not isinstance(episode_id, str) or not episode_id.strip():
            raise ValueError('Episode identity required')
        self.episode_id = episode_id
        self.source, self.bag, self.cabinet = source_path, bag_path, cabinet_path
        self._validate(initial)
        if initial.sample_index != 0:
            raise ValueError('Initial snapshot must be index zero')
        self.initial = self.current = initial
        self.selection = ContactSelection()

    def _validate(self, sample):
        if not isinstance(sample, Snapshot) or not isinstance(sample.pose, Pose):
            raise ValueError('Measured typed snapshot required')
        if sample.episode_id != self.episode_id or sample.target_path != self.target:
            raise ValueError('Wrong episode or target')
        if type(sample.sample_index) is not int or sample.sample_index < 0:
            raise ValueError('Invalid index')
        t = sample.simulation_time_s
        if type(t) not in (int, float) or not math.isfinite(t) or t < 0:
            raise ValueError('Invalid simulation time')

    def observe(self, sample):
        self._validate(sample)
        if (sample.sample_index != self.current.sample_index + 1
                or sample.simulation_time_s <= self.current.simulation_time_s):
            raise ValueError('Missing, duplicate or stale sample')
        self.current = sample

    def receipt(self):
        return dict(target=self.target, source=self.source, bag=self.bag,
            initial_supports=sorted(self.initial_supports),source_is_task_constraint=False,
            cabinet=self.cabinet, initial=asdict(self.initial), current=asdict(self.current),
            initialization_owner='unified_scene', scene_writes=0,
            destination_semantics='inside_bag_not_on_cabinet',
            bag_on_cabinet_verified=False, destination_physics_verified=False,
            scoring_status='not_connected', registered=False)
