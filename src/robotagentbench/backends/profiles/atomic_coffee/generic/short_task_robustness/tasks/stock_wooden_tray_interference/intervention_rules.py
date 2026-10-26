"""Task-local rule primitives; NOT an Isaac adapter or motion executor.

Inputs must be derived by a trusted native adapter, never by a policy action.
Contact input is a COMPLETE current-contact snapshot for each physical step.
"""
from dataclasses import dataclass
from enum import Enum
from math import isfinite
import warnings


class Phase(str, Enum):
    WAITING = 'waiting'
    APPROACH = 'approach'
    CARRY = 'carry'
    RELEASE = 'release'
    RETREAT = 'retreat'
    DONE = 'done'


class HumanCollisionLedger:
    """One charge per continuous robot/human contact episode, max three."""

    def __init__(self, robot_bodies, human_bodies):
        self.robot = frozenset(robot_bodies)
        self.human = frozenset(human_bodies)
        if not self.robot or not self.human or self.robot & self.human:
            raise ValueError('Require nonempty, disjoint native body IDs')
        if any(not isinstance(x, str) or not x for x in self.robot | self.human):
            raise ValueError('Empty native IDs are invalid evidence')
        self.last_step = -1
        self.last_time = -1.0
        self.active = False
        self.contact_episodes = 0
        self.gap_steps = 0

    def observe(self, step, sim_time, pairs, *, complete):
        if type(complete) is not bool:
            raise ValueError('Explicit contact completeness required')
        if not isinstance(step, int) or step < 0 or step <= self.last_step:
            raise ValueError('Physical step must increase')
        if not isfinite(sim_time) or sim_time < 0 or sim_time <= self.last_time:
            raise ValueError('Native simulation time must increase')
        current = False
        for a, b in pairs:
            if not isinstance(a, str) or not isinstance(b, str) or not a or not b:
                raise ValueError('Unresolved body ID; do not treat as no contact')
            current |= ((a in self.robot and b in self.human)
                        or (b in self.robot and a in self.human))
        if current and not self.active:
            self.contact_episodes += 1
        # A missing report cannot certify separation or re-arm a collision
        # charge. Positive current contact can still be observed in a partial
        # report; no-contact is accepted only from a complete report.
        if complete or current:
            self.active = current
        if not complete:
            self.gap_steps += 1
            if self.gap_steps == 1:
                warnings.warn('Partial human contact evidence; continuing without '
                              'inferring separation or duplicate collision charges.', RuntimeWarning)
        self.last_step, self.last_time = step, sim_time
        return self.deduction

    @property
    def deduction(self):
        return min(self.contact_episodes, 3)


@dataclass(frozen=True)
class EvidenceRoute:
    # These flags are integration obligations, not an implemented score mask.
    suppress_robot_pickup_evidence: bool
    continue_human_collision_monitor: bool = True


class InterventionAudit:
    """Audits one same-instance relocation; performs NO scene writes."""

    def __init__(self, target_path, shelf_level_path):
        if not target_path or not shelf_level_path:
            raise ValueError('Exact target and shelf-level identity required')
        self.target = target_path
        self.shelf_level = shelf_level_path
        self.phase = Phase.WAITING

    def trigger(self, *, robot_holding, robot_contacting):
        if self.phase != Phase.WAITING:
            return False
        if robot_holding or robot_contacting:
            return False  # Do not steal an object already in robot contact.
        self.phase = Phase.APPROACH
        return True

    def advance(self, next_phase, *, target_path, shelf_level_path,
                human_released=False, native_supported=False,
                destination_valid=False, actor_clear=False):
        phases = list(Phase)
        next_phase = Phase(next_phase)
        if self.phase in (Phase.WAITING, Phase.DONE):
            raise ValueError('Trigger first; exactly one intervention allowed')
        if phases.index(next_phase) != phases.index(self.phase) + 1:
            raise ValueError('Intervention phases cannot be skipped')
        if target_path != self.target or shelf_level_path != self.shelf_level:
            raise ValueError('Target instance and shelf level must not change')
        if next_phase == Phase.RETREAT and not (
                human_released and native_supported and destination_valid):
            raise ValueError('Release needs measured support at a valid destination')
        if next_phase == Phase.DONE and not actor_clear:
            raise ValueError('Actor has not exited')
        self.phase = next_phase

    @property
    def evidence_route(self):
        # Conservative until the human has exited: no human-generated pick credit.
        return EvidenceRoute(self.phase != Phase.DONE)

    def outcome(self, *, base_score, base_success, collision_ledger):
        if type(base_score) not in (int, float) or not isfinite(base_score) or not 0 <= base_score <= 7:
            raise ValueError('Expected original single-item score, maximum 7')
        exercised = self.phase == Phase.DONE
        return {
            'base_score': base_score,
            'human_collision_deduction': collision_ledger.deduction,
            'score': max(0, (base_score if exercised else 0) - collision_ledger.deduction),
            'base_success': bool(base_success),
            'intervention_exercised': exercised,
            'robust_trial_success': bool(base_success and exercised),
            'native_integration_verified': False,
        }
