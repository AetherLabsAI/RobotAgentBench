"""Approved indirect placement evidence; never claims measured soft force.

No new distance, speed, force or angle thresholds. Live collision-geometry
witnesses classify insertion. Conservative collider AABB separation plus the
rigid contact stream establishes release and excludes external supports.
This is a task predicate, not a complete mesh-intersection certificate.
"""
from dataclasses import dataclass
import math
import numpy as np
from .process_rules import PlacementFacts, TARGET, DESTINATION


def separated(a, b):
    """Strict AABB separation. Touching/overlapping is NOT proven separation."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if (a.shape != (2, 3) or b.shape != (2, 3)
            or not np.isfinite(a).all() or not np.isfinite(b).all()
            or np.any(a[0] > a[1]) or np.any(b[0] > b[1])):
        raise ValueError('Invalid current collider bounds')
    return bool(np.any(a[1] < b[0]) or np.any(b[1] < a[0]))


def insertion(diagnostic):
    """Reject above-rim bridging; allow the umbrella's upper end to protrude.

    Origin AND lowest collider vertex must be in the current deformed cavity,
    origin strictly below the lowest rim, and all below-rim collider-centre
    witnesses must be inside. Missing/ambiguous points fail closed.
    """
    cavity = diagnostic['cavity']
    points = np.asarray(diagnostic['witness_world_positions'], float)
    inside = cavity['witness_inside']
    rim = np.asarray(cavity['opening_min'], float)
    names = diagnostic['witness_names']
    if (len(points) < 3 or points.shape != (len(inside), 3)
            or len(names) != len(points) or names[:2] !=
            ['target_origin', 'lowest_authored_collider_vertex']
            or rim.shape != (3,) or not np.isfinite(points).all()
            or not np.isfinite(rim).all()
            or any(type(v) is not bool for v in inside)):
        raise ValueError('Incomplete/ambiguous cavity evidence')
    centre_below = bool(points[0, 2] < rim[2])
    lower = points[:, 2] < rim[2]
    entered = bool(inside[0] and inside[1] and centre_below
                   and all(v for v, used in zip(inside, lower) if used))
    return entered, not centre_below


@dataclass(frozen=True)
class PhysicalSample:
    index: int
    time_s: float
    inside: bool
    rim_only: bool
    robot_clear: bool
    robot_contact: bool
    bilateral_grasp: bool
    external_clear: bool

    @property
    def released(self):
        return self.robot_clear and not self.robot_contact

    @property
    def placed(self):
        return (self.inside and not self.rim_only and self.released
                and not self.bilateral_grasp and self.external_clear)


class GeometryEvidenceAccumulator:
    """All-physics-step continuity; no endpoint-only continuous-time inference."""
    def __init__(self, target=TARGET):
        self.target = target
        self.last = None
        self.previous_control_placed = False
        self.interval_placed = True
        self.last_collected_index = None
        self.control_index = -1

    def push(self, sample):
        if (not isinstance(sample, PhysicalSample) or type(sample.index) is not int
                or sample.index < 0 or not math.isfinite(sample.time_s)
                or sample.time_s < 0):
            raise ValueError('Invalid physical sample')
        for name in ('inside', 'rim_only', 'robot_clear', 'robot_contact',
                     'bilateral_grasp', 'external_clear'):
            if type(getattr(sample, name)) is not bool:
                raise ValueError('Missing physical predicate: '+name)
        if sample.bilateral_grasp and (sample.robot_clear or not sample.robot_contact):
            raise ValueError('Contradictory held/separation evidence')
        if self.last is not None and (sample.index != self.last.index+1
                                      or sample.time_s <= self.last.time_s):
            raise ValueError('Missing/stale physics step')
        self.last = sample
        self.interval_placed = self.interval_placed and sample.placed

    def collect(self):
        s = self.last
        if s is None or self.last_collected_index == s.index:
            raise ValueError('No new physical interval')
        if self.control_index == -1 and not s.bilateral_grasp:
            raise ValueError('Initial physical bilateral hold has not been validated')
        self.control_index += 1
        uninterrupted = bool(self.previous_control_placed and self.interval_placed)
        # supported_by_basket is explicitly INDIRECT under the approved rule.
        # An overlapping external AABB means unknown, not measured support;
        # conservatively withhold the proxy, without reporting a false force.
        facts = PlacementFacts(self.control_index, s.time_s, self.target, DESTINATION,
            s.inside, s.inside and not s.rim_only and s.external_clear,
            s.released, s.bilateral_grasp, False, s.rim_only, uninterrupted)
        self.previous_control_placed = facts.placed
        self.interval_placed = True
        self.last_collected_index = s.index
        return facts
