"""Additive contact credit; no replacement of native operation predicates."""
from copy import deepcopy
import math

RULE_VERSION = 'openend.additive_target_contact_1pt.v2'
KINDS = frozenset(('meal_tray_transfer', 'microwave_box_to_meal_tray',
    'coffee_to_meal_tray', 'bread_to_meal_tray', 'cutlery_to_meal_tray',
    'snack_drink_to_tray', 'fridge_to_microwave', 'trash_transfer',
    'powerbank_transfer', 'water_bottle_transfer', 'book_transfer', 'cap_transfer',
    'notebook_transfer', 'paper_cup_transfer', 'wallet_transfer', 'glasses_transfer',
    'remote_transfer', 'coffee', 'fryer', 'salad_to_meal_tray',
    'drink_to_meal_tray', 'wipe_table', 'umbrella_transfer'))


def finite(x):
    return type(x) in (int, float) and math.isfinite(x)


def ancestors(path):
    if (not isinstance(path, str) or not path.startswith('/World/')
            or any(p in ('', '.', '..') for p in path.split('/')[1:])):
        raise ValueError('Canonical absolute prim path required')
    yield path
    while '/' in path[1:]:
        path = path.rsplit('/', 1)[0]
        yield path


class ContactLedger:
    """Indexes genuine, already-collected finger contacts for one episode.

    No physics query, grasp/closure inference or change to task activation.
    Prerequisite eligibility is supplied by the native room at each substep.
    """
    def __init__(self, *, episode_id, kinds, bindings, fingers):
        if not isinstance(episode_id, str) or not episode_id.strip():
            raise ValueError('Episode required')
        if not kinds or set(kinds) != set(bindings) or set(kinds.values()) - KINDS:
            raise ValueError('Supported registered task bindings required')
        self.episode_id, self.kinds = episode_id, dict(kinds)
        self.bindings = {k: tuple(v) for k, v in bindings.items()}
        self.fingers = frozenset(fingers)
        if not self.fingers:
            raise ValueError('Native finger roots required')
        for root in self.fingers:
            tuple(ancestors(root))
            if not root.startswith('/World/Robot/gripper_'):
                raise ValueError('Only native finger bodies, not arms/palms')
        self.index = {}
        for key, roots in self.bindings.items():
            if not roots:
                raise ValueError('Empty target binding')
            for root in roots:
                tuple(ancestors(root))
                if root == '/World/Robot' or root.startswith('/World/Robot/'):
                    raise ValueError('Robot cannot be a task target')
                self.index.setdefault(root, set()).add(key)
        self.origin = self.last_time = None
        self.sample_index = 0
        self.events = {}
        self.ended = False

    def arm(self, time_s):
        if self.origin is not None or not finite(time_s) or time_s < 0:
            raise ValueError('Arm once after accepted initialization')
        self.origin = self.last_time = time_s

    def observe(self, pairs, *, time_s, eligible):
        if self.origin is None or self.ended:
            raise RuntimeError('Live accepted episode required')
        if not finite(time_s) or time_s <= self.last_time:
            raise ValueError('Fresh native substep required')
        if set(eligible) != set(self.kinds) or any(type(v) is not bool for v in eligible.values()):
            raise ValueError('Explicit prerequisite eligibility required')
        touched = {}
        for paths, samples in pairs:
            if len(paths) != 2:
                raise ValueError('Native collider pair required')
            chains = [tuple(ancestors(p)) for p in paths]
            if not samples:
                continue
            hands = [bool(self.fingers.intersection(chain)) for chain in chains]
            if hands[0] == hands[1]:
                continue
            other = 1 if hands[0] else 0
            for root in chains[other]:
                for key in self.index.get(root, ()):
                    if eligible[key]:
                        touched.setdefault(key, dict(target_root=root, collider_paths=list(paths)))
        self.last_time = time_s
        self.sample_index += 1
        new = []
        for key in sorted(touched):
            if key not in self.events:
                event = dict(episode_id=self.episode_id, instance_id=key,
                    task_type=self.kinds[key], time_s=time_s,
                    sample_index=self.sample_index, points=1,
                    source='existing_native_physx_finger_contact_stream', **touched[key])
                self.events[key] = event
                new.append(deepcopy(event))
        return new

    def augment(self, key, native):
        if key not in self.kinds or not isinstance(native, dict) or 'contact_credit' in native:
            raise ValueError('One original native result per registered instance required')
        score, maximum = native.get('score'), native.get('maximum_score')
        if not finite(score) or not finite(maximum) or maximum <= 0 or score < 0 or score > maximum:
            raise ValueError('Finite native score and original maximum required')
        result = deepcopy(native)
        points = 1 if key in self.events else 0
        # Retain native deductions and aggregate clamping. No original grasp
        # points are removed. Composites without raw totals retain their own
        # component clamping, plus the independent contact point.
        if 'raw_score' in native:
            if not finite(native['raw_score']) or abs(score - max(0, native['raw_score'])) > 1e-8:
                raise ValueError('Unexpected native raw-score contract')
            result['raw_score'] = native['raw_score'] + points
            result['score'] = max(0, result['raw_score'])
        else:
            result['score'] = score + points
        result['maximum_score'] = maximum + 1
        result['contact_credit'] = dict(rule_version=RULE_VERSION, points=points,
            maximum_score=1, source_native_score=score,
            source_native_maximum_score=maximum,
            source_native_raw_score=native.get('raw_score'),
            source_grasp_points_removed=0, event=deepcopy(self.events.get(key)),
            completion_and_native_penalties_unchanged=True)
        return result

    def finalize(self):
        self.ended = True
        return dict(rule_version=RULE_VERSION, episode_id=self.episode_id,
            bindings=deepcopy(self.bindings), events=deepcopy(self.events))
