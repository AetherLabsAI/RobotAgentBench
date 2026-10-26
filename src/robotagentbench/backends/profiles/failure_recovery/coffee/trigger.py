"""One-shot trigger from fully validated CURRENT physical evidence."""
import math
import warnings
from dataclasses import dataclass

MINIMUM_TILT_DEG = 45.0


@dataclass(frozen=True)
class TriggerFacts:
    time_s: float
    holding_bag: bool
    mouth_above_opening: bool
    bag_tilt_deg: float
    contacts_complete: bool


class PourApproachTrigger:
    def __init__(self):
        self.armed = False
        self.fired = False
        self.last_time = None
        self.grasp_seen = False
        self.contact_gap_steps = 0

    def arm(self):
        if self.armed:
            raise RuntimeError('Initial admission arms this trigger once')
        self.armed = True

    def observe(self, facts, *, route_clear):
        if not self.armed:
            raise RuntimeError('Initial physical admission required')
        if not isinstance(facts, TriggerFacts):
            raise ValueError('Native current-frame trigger facts required')
        for key in ('holding_bag', 'mouth_above_opening', 'contacts_complete'):
            if type(getattr(facts, key)) is not bool:
                raise ValueError('Explicit boolean trigger evidence required: ' + key)
        if type(route_clear) is not bool:
            raise ValueError('Explicit route admission required')
        for value in (facts.time_s, facts.bag_tilt_deg):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError('Finite time and tilt evidence required')
        if not 0 <= facts.bag_tilt_deg <= 180 or facts.time_s < 0:
            raise ValueError('Trigger evidence out of range')
        if self.last_time is not None and facts.time_s <= self.last_time:
            raise ValueError('Repeated or stale physical trigger frame')
        if not facts.contacts_complete:
            self.last_time=facts.time_s
            self.contact_gap_steps+=1
            if self.contact_gap_steps==1:
                warnings.warn('Missing native trigger contact evidence; continuing '
                              'without authorizing the push.',RuntimeWarning)
            return dict(start=False,reason='waiting_for_contact_evidence',time_s=facts.time_s,
                grasp_seen=self.grasp_seen)
        self.last_time = facts.time_s
        self.grasp_seen |= facts.holding_bag
        if self.fired:
            reason = 'already_triggered'
        elif not facts.holding_bag:
            reason = 'bag_not_currently_held'
        elif not facts.mouth_above_opening:
            reason = 'bag_mouth_not_above_hopper'
        elif facts.bag_tilt_deg < MINIMUM_TILT_DEG:
            reason = 'bag_not_yet_tilted'
        elif not route_clear:
            reason = 'push_route_blocked'
        else:
            reason = 'pour_approach_triggered'
            self.fired = True
        return dict(start=reason == 'pour_approach_triggered', reason=reason,
                    time_s=facts.time_s, grasp_seen=self.grasp_seen)
