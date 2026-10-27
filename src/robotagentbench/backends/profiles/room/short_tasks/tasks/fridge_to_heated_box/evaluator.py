"""Read-only fridge/put scoring, with no knob or lamp stage (v3).

The room owns the clock. This evaluator never advances physics, sets door/box
state, creates a handoff, or trusts a client completion flag.
The old put result supplies historical process credit; the final success
predicate is checked against fresh facts for the same box on every observation.
"""
from copy import deepcopy
import math

from layout_task_activation_v1.task_scene_registration_v1.fridge_to_microwave_v1.composite import Composite
from evaluation.tasks.microwave_put.process_rules import PutFacts
from evaluation.tasks.microwave_put.openended_fridge_to_microwave_v1.fridge_rules import STAGE_POINTS
from .adapter import RULE_VERSION
from .sources import verify_sources


class FridgeHeatingEvaluator:
    def __init__(self, composite):
        self.source_receipt = verify_sources()
        if not isinstance(composite, Composite):
            raise ValueError("Actual upstream fridge/put composite required")
        self.composite = composite
        self.last_time = None
        self.last_facts = None
        self.ended = False

    def observe_current(self, facts, time_s):
        if self.ended:
            raise RuntimeError("Task 07 already settled")
        if not isinstance(facts, PutFacts):
            raise ValueError("Typed native current put facts required")
        if (facts.payload_variant, facts.payload_instance) != (self.composite.variant, self.composite.target_prim):
            raise ValueError("Different food-box instance or variant")
        for key in ("door_open", "door_closed", "payload_inside", "payload_supported",
                    "held_by_robot", "held_by_right", "released"):
            if type(getattr(facts, key)) is not bool:
                raise ValueError("Complete measured current boolean required: " + key)
        if (facts.held_by_right and not facts.held_by_robot
                or facts.released and facts.held_by_robot
                or facts.door_open and facts.door_closed):
            raise ValueError("Conflicting measured current holding or door facts")
        if type(time_s) not in (int, float) or not math.isfinite(time_s) or time_s < 0:
            raise ValueError("Finite actual simulation time required")
        if self.last_time is not None and (time_s < self.last_time
                or time_s == self.last_time and self.last_facts is not None):
            raise ValueError("Fresh ordered current evidence required")
        if self.composite.last is None or time_s < self.composite.last.time_s:
            raise ValueError("Current evidence cannot precede the measured component baseline")
        self.last_facts = facts
        self.last_time = float(time_s)
        return self.summary()

    def pause(self):
        self.last_facts = None
        self.composite.pause()

    def finalize(self, reason="ended"):
        self.composite.finalize()
        self.ended = True
        self.termination_reason = reason
        return self.summary()

    def summary(self):
        old = self.composite.summary()
        fridge, put = old["refrigerator"], old["microwave"]
        stages = []
        gross = deductions = 0
        fridge_total = 0
        points = dict(STAGE_POINTS)
        if fridge:
            fridge_earned = sum(points[name] for name in fridge["stages"])
            gross += fridge_earned
            deductions += fridge["collision_deduction"]
            # v2: user removed the terminal incomplete-hold penalty only.
            # Recompute from earned points, rather than adding 3 to an already
            # floor-clipped legacy score (which could invent unearned points).
            fridge_total = max(0, fridge_earned - fridge["collision_deduction"])
            stages += [dict(id="fridge." + name, points=points[name], complete=True) for name in fridge["stages"]]
        if put:
            earned = put["details"]["earned"]
            gross += sum(earned.values())
            deductions += put["details"]["inner_wall_collision_penalty"]
            stages += [dict(id="microwave." + name, points=value, complete=True) for name, value in earned.items()]
        facts = self.last_facts
        current_inside = bool(facts and facts.payload_inside)
        current_valid = bool(facts and facts.payload_inside and facts.payload_supported
            and facts.released and not facts.held_by_robot and facts.door_closed)
        process_complete = bool(self.composite.success)
        success = bool(process_complete and current_valid)
        # Preserve independent floors and the unchanged put scoring.
        total = fridge_total + (put["score"] if put else 0)
        effective_deductions = gross - total
        return dict(task_name="fridge_to_heated_box", rule_version=RULE_VERSION,
            payload_variant=self.composite.variant, target_prim=self.composite.target_prim,
            score=total, maximum_score=18,
            score_channels=dict(process=gross, current_final=0,
                deductions=effective_deductions, total=total, maximum=18),
            inherited_deductions=deductions, stages=stages, success=success,
            settlement_policy="no_fridge_incomplete_hold_penalty",
            legacy_hold_failure_deduction_not_applied=(fridge["hold_failure_deduction"] if fridge else 0),
            process_complete=process_complete, current_inside_microwave=current_inside,
            current_box_valid=current_valid, ended=self.ended,
            fridge_put=deepcopy(old),
            source_receipt=deepcopy(self.source_receipt),
            no_scene_pose_writes=True, native_physics_verified=False)
