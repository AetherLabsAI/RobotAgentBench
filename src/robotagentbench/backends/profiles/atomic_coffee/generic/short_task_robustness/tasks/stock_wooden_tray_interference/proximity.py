"""One-shot, task-owned 25 cm pre-contact trigger. No scene writes."""
import math
import warnings

TRIGGER_DISTANCE_M = 0.25  # Explicit user decision; not a policy action option.


def distance_to_box(point, bounds):
    if len(point) != 3 or len(bounds) != 2 or any(len(row) != 3 for row in bounds):
        raise ValueError('Three-dimensional point and box required')
    values = [*point, *bounds[0], *bounds[1]]
    if not all(math.isfinite(v) for v in values):
        raise ValueError('Nonfinite proximity geometry')
    if any(a > b for a, b in zip(*bounds)):
        raise ValueError('Inverted target box')
    return math.sqrt(sum(max(lo - p, 0., p - hi) ** 2
                         for p, lo, hi in zip(point, *bounds)))


class ProximityTrigger:
    def __init__(self, target_path):
        if not isinstance(target_path, str) or not target_path:
            raise ValueError('Exact target identity required')
        self.target_path = target_path
        self.last_time = None
        self.fired = False
        self.robot_touched = False
        self.latest = None
        self.contact_gap_steps = 0

    def observe(self, facts, *, route_clear):
        now = facts['simulation_time_s']
        if not math.isfinite(now) or now < 0 or (self.last_time is not None and now <= self.last_time):
            raise ValueError('Fresh increasing native timestamp required')
        if facts['target_path'] != self.target_path:
            raise ValueError('Proximity target identity changed')
        for key in ('robot_contacting_target', 'robot_holding_target', 'target_supported', 'contacts_complete'):
            if type(facts[key]) is not bool:
                raise ValueError('Explicit boolean evidence required: '+key)
        if type(route_clear) is not bool:
            raise ValueError('Explicit route admission required')
        centers = facts['gripper_centers_target_local_m']
        if set(centers) != {'l', 'r'}:
            raise ValueError('Both gripper centers required')
        distances = {hand: distance_to_box(p, facts['target_bounds_local_m'])
                     for hand, p in centers.items()}
        self.last_time = now
        if not facts['contacts_complete']:
            self.contact_gap_steps += 1
            if self.contact_gap_steps == 1:
                warnings.warn('Incomplete trigger contacts; continuing but not '
                              'authorizing interference from missing evidence.', RuntimeWarning)
            self.latest = dict(simulation_time_s=now, target_path=self.target_path,
                threshold_m=TRIGGER_DISTANCE_M, distances_m=distances,
                triggering_hand=None, start=False, fired=self.fired,
                reason='waiting_for_contact_evidence')
            return dict(self.latest)
        self.robot_touched |= facts['robot_contacting_target'] or facts['robot_holding_target']
        near = min(distances.values()) <= TRIGGER_DISTANCE_M
        eligible = (not self.fired and not self.robot_touched and near
                    and facts['target_supported'] and route_clear)
        reason = ('already_triggered' if self.fired else
                  'robot_already_contacted' if self.robot_touched else
                  'waiting_for_proximity' if not near else
                  'target_not_supported' if not facts['target_supported'] else
                  'reserved_route_blocked' if not route_clear else 'triggered')
        if eligible:
            self.fired = True
        self.latest = dict(simulation_time_s=now, target_path=self.target_path,
            threshold_m=TRIGGER_DISTANCE_M, distances_m=distances,
            triggering_hand=min(distances, key=distances.get) if eligible else None,
            start=eligible, fired=self.fired, reason=reason)
        return dict(self.latest)


class NativeTriggerFacts:
    """Read jaw midpoints and target OBB from existing CPU PhysX evidence.

    Center = midpoint of the existing inner/outer link4 pad origins, not base,
    camera, wrist, commands or estimated FK. Box follows the native target pose.
    """
    def __init__(self, source, simulation_view):
        import numpy as np
        from evaluation.tasks.shelf_pick.contact_stream import PADS
        from evaluation.tasks.coffee_machine.openended_composite_v1.native_source import _BoundBodies
        self.source = source
        self.target = source.active
        if tuple(source.binding.identity.candidates) != (self.target,):
            raise ValueError('One exact target required')
        vertices = np.asarray(source.binding.workspace.collision_vertices[self.target], dtype=float)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices) or not np.isfinite(vertices).all():
            raise ValueError('Missing measured target geometry')
        self.bounds = [vertices.min(0).tolist(), vertices.max(0).tolist()]
        self.pads = {h: [_BoundBodies(simulation_view, [p]) for p in paths]
                     for h, paths in PADS.items()}

    def read(self):
        import numpy as np
        from evaluation.tasks.shelf_pick.geometry import rotation
        source = self.source
        row = source.latest
        if row is None or row['time_s'] != float(source.world.current_time) or source.active != self.target:
            raise RuntimeError('Fresh exact-target native observation required')
        pose = row['poses'][self.target]
        inverse = rotation(pose['quaternion_wxyz']).inv()
        centers = {}
        for hand, pads in self.pads.items():
            points = np.asarray([v.get_world_pose()[0] for v in pads], dtype=float)
            if points.shape != (2, 3) or not np.isfinite(points).all():
                raise RuntimeError('Invalid native finger-pad positions')
            centers[hand] = inverse.apply(points.mean(0) - pose['position_m']).tolist()
        # Room readers may continue through a reported contact gap. Never turn
        # that unknown interval into authorization to steal the target.
        return dict(simulation_time_s=row['time_s'], target_path=self.target,
            target_bounds_local_m=self.bounds, gripper_centers_target_local_m=centers,
            robot_contacting_target=self.target in row['contacts']['robot_contacts'],
            robot_holding_target=bool(row['grasped'][self.target]),
            target_supported=bool(row['contacts']['target_supported']),
            contacts_complete=row['contacts'].get('contact_evidence', {}).get('complete') is True)
