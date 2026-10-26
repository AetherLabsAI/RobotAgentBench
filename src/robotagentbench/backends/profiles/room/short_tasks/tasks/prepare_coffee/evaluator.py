"""Coffee preparation over the friend's unchanged measured scoring components.

The scene owner supplies one immutable Binding and dense measured Frame values.
This module cannot create/reset assets, advance physics, or manufacture device
events. Four source phases award at most 21 process points, including retrieval.
Brewing alone is not task success; the robot must take out and hold the cup.
"""

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path

from evaluation.tasks.coffee_machine.facts import Facts, TRAY, COUNTER
from evaluation.tasks.coffee_machine.openended_composite_v1.composite import (
    Binding, Frame, PHASES, UnresolvedSequence,
)
from evaluation.tasks.coffee_machine.process_rules import (
    CLOCK_TOLERANCE_S, CONTROL_DT_S, RULE_VERSION as COMPONENT_RULE_VERSION,
    SPECS, ML_PER_PARTICLE,
)
from openend.tasks.components import CoffeeProcess
from openend.tasks.scores import CoffeeTask, interrupt_hold


SOURCE_COMMIT = "ef47a69776d709047c35293c28cc0505d6c2b6f9"
RULE_VERSION = "openend.short_task.prepare_coffee.v2"
PREPARATION_PHASES = ("1-1", "1-2", "1-3")
MAXIMUM_SCORE = sum(SPECS[phase].maximum_score for phase in PHASES)
PLACEMENT_POLICY = "receiving_tray_spout_alignment_v1"
BEAN_PROGRESS_POLICY = "hopper_progress_90_full_v1"
BEANS_FOR_FULL_POINTS = 90
BAG_RETURN_POLICY = "independent_stable_bag_return_v1"
BEAN_PHASE_POLICY = "stable_return_no_bean_count_gate_v1"


def verify_component_sources():
    """Reject silent component drift before attaching one episode's evaluator."""
    receipt = json.loads(Path(__file__).with_name("component_sources.json").read_text())
    root = Path(__file__).resolve().parents[3]
    changed = [relative for relative, digest in receipt["files_sha256"].items()
               if not (root / relative).is_file() or
               hashlib.sha256((root / relative).read_bytes()).hexdigest() != digest]
    if changed:
        raise ValueError("Pinned coffee components changed: " + ", ".join(changed))
    return receipt


class _RecoverableCoffeeProcess(CoffeeProcess):
    """Document policy: measured recoverable faults block rather than terminate.

On a fault frame source milestones do not run. Existing once-only deductions
still run, and the unfinished hold is discarded. No new penalty is invented.
The source's timing-only limits remain disabled by CoffeeProcess.
"""

    def __init__(self, api):
        super().__init__(api)
        self.current_condition = None
        self._sequence_block = None
        self._prestart_emitted = 0

    def observe(self, facts, *, sequence_block=None):
        self._previous_condition = self.current_condition
        self.current_condition = None
        self._sequence_block = sequence_block
        return super().observe(facts)

    def _failure_now(self, facts):
        return super()._failure_now(facts) or self._sequence_block

    def _award(self, facts, stage):
        if self.api == '1-2' and stage == 2:
            return self._award_bean_progress(facts)
        if self.api == '1-2' and stage == 3:
            # The inherited advance requests this immediately when enough beans
            # are present. Award only after its measured stability hold below.
            return
        return super()._award(facts, stage)

    def _award_bean_progress(self, facts):
        if 1 not in self._awards:
            return
        maximum = self.spec.stage_points[2]
        points = maximum * min(facts.beans_in_hopper, BEANS_FOR_FULL_POINTS) / BEANS_FOR_FULL_POINTS
        previous = self._awards.get(2, {}).get('points', 0.)
        if points <= previous:
            return
        # Process credit is a high watermark, not repeatable rewards for the
        # same beans leaving/re-entering. Keep physical counts unchanged.
        self._awards[2] = dict(points=points, step=facts.step, time_s=float(facts.time_s))
        self._events.append(dict(kind='bean_progress', stage=2, points=points-previous,
                                 cumulative_points=points, beans_in_hopper=facts.beans_in_hopper,
                                 full_credit_beans=BEANS_FOR_FULL_POINTS,
                                 step=facts.step, time_s=float(facts.time_s)))

    def _pour_beans(self, facts, emitted_before):
        # Bean quantity earns proportional points only; it cannot lock later
        # operations. Retain measured pickup, release/support/tilt and hold.
        bag = facts.bag
        back = (COUNTER in bag.supports and bag.released
                and bag.tilt_deg <= self.spec.thresholds['bag_upright_tilt_deg'])
        if self._stage == 1 and bag.grasped:
            self._award(facts, 1)
        if 1 not in self._awards:
            self._charge()
            return None
        self._award_bean_progress(facts)
        if back:
            self._enter(facts, 3)
        elif bag.grasped:
            self._enter(facts, 2)
        elif bag.released and bag.supports:
            self._enter(facts, 1)
        else:
            self._enter(facts, 2)
        self._charge(holding=back)
        if self._hold(facts, back, bag.position):
            super()._award(facts, 3)
            return 'success'
        return None

    def _settle(self, facts):
        if self.api != '1-2':
            return super()._settle(facts)
        # Legacy settlement otherwise reinstates the removed bean-count failure.
        # Keep measured counts and proportional credit, with no bean deductions.
        if not self._settled:
            self._settled = True
            self._beans_band = 'quantity_for_points_only'

    def summary(self):
        result = super().summary()
        if self.api == '1-2':
            points = self._awards.get(2, {}).get('points', 0.)
            result['stage_points']['2'].update(points=points,
                maximum_points=self.spec.stage_points[2],
                fully_awarded=points >= self.spec.stage_points[2])
            result['bean_progress_policy'] = BEAN_PROGRESS_POLICY
            result['full_credit_beans'] = BEANS_FOR_FULL_POINTS
            result['bag_return_policy'] = BAG_RETURN_POLICY
            result['bean_phase_policy'] = BEAN_PHASE_POLICY
        return result

    def _place_cup(self, facts, _):
        # Task-local, user-approved receiving-region rule. Keep raw anchor
        # distance as telemetry; do not fabricate or rewrite measured facts.
        cup, placement = facts.cup, facts.placement
        held = cup.grasped
        placed = (TRAY in cup.supports and cup.released
                  and cup.tilt_deg <= self.spec.thresholds['cup_upright_tilt_deg']
                  and not placement.outside_bay and placement.outlet_over_opening)
        if self._stage == 1 and held:
            self._award(facts, 1)
        if 1 in self._awards:
            if placed:
                self._award(facts, 2)
                self._enter(facts, 3)
            elif held:
                self._enter(facts, 2)
            elif self._stage == 1:
                pass
            elif cup.robot_contact:
                self._enter(facts, 2)
            elif cup.supports:
                self._enter(facts, 1)
        self._charge()
        return 'success' if self._hold(facts, 1 in self._awards and placed, cup.position) else None

    def _press_button(self, facts, emitted_before):
        # Only a new robot start in phase 1-3 can credit brewing. Keep all raw
        # device/volume counters intact; this baseline limits reward only.
        if 1 not in self._awards:
            valid_start = any(name == 'started' and robot for name, robot in
                              zip(facts.button.events, facts.button.event_by_robot))
            if valid_start:
                if emitted_before is not None:
                    self._prestart_emitted = emitted_before
            else:
                self._prestart_emitted = facts.liquid.emitted
        return super()._press_button(facts, emitted_before)

    def _hold(self, facts, condition, position):
        if self.api == '1-3':
            # Conservatively assume early liquid may still be in the cup.
            # It cannot satisfy the existing minimum-volume reward threshold.
            credited_in_cup = max(0, facts.liquid.in_cup - self._prestart_emitted)
            condition = (condition and credited_in_cup * ML_PER_PARTICLE >=
                         self.spec.thresholds['cup_liquid_min_ml'])
        return super()._hold(facts, condition, position)

    def _finish(self, facts, *, success=False, failure=None):
        recoverable = ("cup_on_ground", "bag_on_ground", "cup_toppled",
                       "liquid_spilled", "cup_overfilled",
                       "cup_tilt_exceeded", "device_operated_before_brew")
        if failure in recoverable:
            self.current_condition = failure
            self._window = []
            self._charge_deductions(facts)
            if failure != 'device_operated_before_brew' or self._previous_condition != failure:
                self._events.append(dict(kind="recoverable_condition", reason=failure,
                                         stage=self._stage, step=facts.step,
                                         time_s=float(facts.time_s)))
            return
        return super()._finish(facts, success=success, failure=failure)


class _PreparationCoffeeTask(CoffeeTask):
    """CoffeeTask's read-only phase composition with task-local process policy."""

    def _validate_global(self, frame):
        try:
            super()._validate_global(frame)
        except UnresolvedSequence:
            # Source raises only after identity, clock and counter-integrity
            # validation. Do not swallow malformed evidence or invalid reset.
            # The retrieval-restart policy is deliberately unchanged.
            if self.last is None or self.index >= 2:
                raise

    def observe(self, frame):
        self._validate_global(frame)
        facts = frame.facts
        entering = self.process is None
        baseline = (self.last or facts) if entering else self.baseline
        start = facts.step if entering else self.phase_start
        process = (_RecoverableCoffeeProcess(self.phase) if entering else
                   deepcopy(self.process, {id(self.process.spec): self.process.spec}))
        if entering and self.phase == '1-3':
            process._prestart_emitted = baseline.liquid.emitted
        liquid = facts.liquid
        if self.phase == "1-4":
            # Measured brew-to-retrieval boundary, not a scene or counter reset.
            liquid = replace(liquid, initial_in_cup=baseline.liquid.in_cup,
                             spilled=facts.liquid.spilled-baseline.liquid.spilled)
        local = replace(
            facts, step=facts.step-start, time_s=(facts.step-start)*CONTROL_DT_S, liquid=liquid,
            cup=replace(facts.cup, displacement_m=math.dist(facts.cup.position, baseline.cup.position)),
            bag=replace(facts.bag, displacement_m=math.dist(facts.bag.position, baseline.bag.position)),
        )
        sequence_block = ('device_operated_before_brew' if self.index < 2 and
                          (facts.button.events or facts.button.dispensing) else None)
        process.observe(local, sequence_block=sequence_block)
        self.process, self.baseline, self.phase_start, self.last = process, baseline, start, facts
        if process.failure:
            self.failure = dict(phase=self.phase, reason=process.failure)
            self.ended = True
        elif process.success:
            process.finalize("success")
            self.completed.append(dict(api=self.phase, start_step=start,
                                       end_step=facts.step, result=process.summary()))
            self.transitions.append(dict(from_phase=self.phase, step=facts.step,
                                         scene_reset=False, liquid_in_cup=facts.liquid.in_cup,
                                         beans_in_hopper=facts.beans_in_hopper))
            self.index += 1
            self.process = None
            self.success = self.index == len(PHASES)
            self.ended = self.success
        return self.summary()


class PrepareCoffeeEvaluator:
    """A single bound task; consumes facts, never controls or repairs the world.

``success`` is current evidence; completed phases and points are historical.
The host owns termination and its 120-second progress timeout. Before host
finalization, losing the retrieved cup revokes success without replaying points.
"""

    def __init__(self, binding: Binding, *, evidence_source="external_facts"):
        if not isinstance(binding, Binding):
            raise ValueError("A unique scene-owned coffee Binding is required")
        if evidence_source not in ("external_facts", "native_physx", "cpu_synthetic_fixture"):
            raise ValueError("Explicit supported evidence source required")
        self.source_receipt = verify_component_sources()
        self.binding = binding
        self.component = _PreparationCoffeeTask(binding)
        self.component.source = evidence_source
        self.evidence_source = evidence_source
        self.last = None
        self.success = False
        self.termination = None
        self.current_result = self._empty_current_result()

    @staticmethod
    def _empty_current_result():
        return dict(preparation_complete=False, extraction_complete=False,
                    cup_retrieval_valid=False, result_valid=False, valid=False)

    @property
    def phase(self):
        return self.component.phase

    @property
    def reader_phase(self):
        return "1-4" if self.component.success else self.component.phase

    @property
    def facts_phase(self):
        return self.reader_phase

    @property
    def process(self):
        return self.component.process

    @property
    def ended(self):
        return self.termination is not None

    @property
    def preparation_complete(self):
        return all(any(row["api"] == phase for row in self.component.completed)
                   for phase in PREPARATION_PHASES)

    def _validate(self, frame):
        if self.ended:
            raise ValueError("The episode is finalized; cannot consume further evidence")
        if not isinstance(frame, Frame) or frame.binding != self.binding:
            raise ValueError("Wrong episode, task or scene-owned object binding")
        facts = frame.facts
        if not isinstance(facts, Facts):
            raise ValueError("Typed coffee Facts required")
        expected = 0 if self.last is None else self.last.step + 1
        if type(facts.step) is not int or facts.step != expected:
            raise ValueError("Repeated, missing or out-of-order 20 Hz coffee frame")
        if not math.isfinite(facts.time_s) or abs(facts.time_s - facts.step * CONTROL_DT_S) > CLOCK_TOLERANCE_S:
            raise ValueError("Coffee evidence must retain the dense 20 Hz clock")
        # Reuse source fact validation without scoring or a new component phase.
        CoffeeProcess("1-3")._validate(replace(facts, step=0, time_s=0.))
        if facts.liquid.mode != "visual_cpu_v1":
            raise ValueError("Short coffee preparation requires visual_cpu_v1 evidence")
        if self.evidence_source == "native_physx" and any(
                not hasattr(obj, "following_confirmed") for obj in (facts.cup, facts.bag)):
            raise ValueError("Native coffee requires the position-follow reader, not bilateral contact grasping")
        if facts.beans_in_hopper > facts.beans_total:
            raise ValueError("Measured hopper count exceeds the bound bean inventory")
        if self.last is not None:
            previous = self.last
            if facts.beans_total != previous.beans_total:
                raise ValueError("Bound bean inventory changed during the episode")
            if facts.liquid.mode != previous.liquid.mode:
                raise ValueError("Coffee evidence mode changed during the episode")
            if facts.liquid.emitted < previous.liquid.emitted or facts.liquid.spilled < previous.liquid.spilled:
                raise ValueError("Cumulative coffee counters cannot reset")

    def _current(self, facts):
        cup, placement, liquid, button = facts.cup, facts.placement, facts.liquid, facts.button
        take = SPECS["1-4"].thresholds
        brew = SPECS["1-3"].thresholds
        cup_valid = (cup.grasped and placement.outside_bay and not cup.ground_contact
                     and cup.tilt_deg <= take["cup_upright_tilt_deg"])
        baseline = self.component.baseline
        spill_valid = (baseline is not None and
                       (liquid.spilled-baseline.liquid.spilled)*100 <=
                       take["spill_percent"]*baseline.liquid.in_cup)
        result_valid = (not button.dispensing
                        and not button.events
                        and -button.displacement_m <= brew["button_released_depth_m"]
                        and spill_valid)
        return dict(preparation_complete=self.preparation_complete,
                    extraction_complete=self.component.success,
                    cup_retrieval_valid=bool(cup_valid), result_valid=bool(result_valid),
                    valid=bool(self.component.success and cup_valid and result_valid))

    def observe(self, frame: Frame):
        self._validate(frame)
        if not self.component.ended:
            self.component.observe(frame)
        elif not self.component.success:
            raise ValueError("Source coffee process has terminated without success")
        self.last = frame.facts
        self.current_result = self._current(self.last)
        self.success = self.current_result["valid"]
        if self.component.failure:
            self.termination = "failure"
        return self.summary()

    def pause(self):
        # Clears only an unfinished source hold; awarded points are retained.
        if not self.ended and not self.component.success:
            interrupt_hold(self.component)

    def finalize(self, reason):
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Episode termination reason required")
        if self.termination is not None:
            if reason != self.termination:
                raise ValueError("Contradictory repeated finalization")
            return self.summary()
        if self.last is None:
            raise ValueError("Cannot finalize coffee preparation without measured evidence")
        if reason == "success" and not self.success:
            raise ValueError("No current measured coffee success")
        self.component.finalize(reason)
        self.termination = reason
        # A host must use success for a successful episode rather than silently
        # label a currently valid result as timeout or source exhaustion.
        self.success = bool(reason == "success" and self.current_result["valid"])
        return self.summary()

    def summary(self):
        raw = self.component.summary()
        phases = [row["result"] for row in raw["completed"]]
        if raw["active"] is not None:
            phases.append(raw["active"])
        points = sum(row["points_awarded"] for row in phases)
        deductions = sum(row["deduction_total"] for row in phases)
        failure = deepcopy(self.component.failure)
        return deepcopy(dict(
            schema="openend.short_task.evaluator.v1", task_name="prepare_coffee",
            task_instance_id=self.binding.task_instance_id, episode_id=self.binding.episode_id,
            rule_version=RULE_VERSION, component_rule_version=COMPONENT_RULE_VERSION,
            source_commit=SOURCE_COMMIT, phase=self.phase, reader_phase=self.reader_phase,
            component_sources_verified=True,
            process_complete=self.component.success, success=self.success,
            ended=self.ended, termination=self.termination, failure=failure,
            score=max(0., float(points - deductions)), maximum_score=MAXIMUM_SCORE,
            scores=dict(process=float(points), current_final=0., deductions=float(deductions),
                        total=max(0., float(points - deductions)), maximum=MAXIMUM_SCORE),
            current_result=self.current_result, phase_results=phases,
            handoff=None, extraction_scored=True, scene_reset=False,
            initialization_owner="unified_scene", stage_time_limits_enforced=False,
            total_timeout_policy="single_episode_net_progress_120s",
            recoverable_condition=getattr(self.process, "current_condition", None),
            inherited_physical_failure_policy="block_milestones_preserve_original_deductions",
            early_device_policy="block_while_active_no_reset_no_retroactive_brew_credit",
            placement_policy=PLACEMENT_POLICY,
            bean_progress_policy=BEAN_PROGRESS_POLICY,
            bean_phase_policy=BEAN_PHASE_POLICY,
            brew_uncredited_emitted=(getattr(self.process, '_prestart_emitted', 0)
                                     if self.phase == '1-3' else None),
            evidence_source=self.evidence_source,
            physics_verified=False,
            evidence_step=self.last.step if self.last is not None else None,
            evidence_time_s=self.last.time_s if self.last is not None else None,
        ))
