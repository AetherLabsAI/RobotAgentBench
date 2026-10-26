"""Task-local composition; no actions, USD writes or physics overrides.

Only a trusted native reader may supply these facts. Source strings are format
checks, not authentication. This component is not yet a physical task port.
"""
from copy import deepcopy
from dataclasses import dataclass, replace
import math

from evaluation.tasks.microwave_take.process_rules import (
    ProcessFrame, ProcessScorer, HOLD_FAILURE_PENALTY, bindings_for,
)
from evaluation.tasks.microwave_take.success import check_success
from short_tasks.tasks.interaction_progress_v1.scoring import (
    Progress, Sample, APPROACH_POINTS, TOUCH_POINTS,
)

KNOB = "/World/Assets/DiningTableMicrowave/Asset/StartKnob"
MAXIMUM = 14.0


@dataclass(frozen=True)
class HeatingEvidence:
    episode_id: str
    step_index: int
    time_s: float
    state: str
    actual_lamp_intensity: float
    runtime_lamp_on: bool
    observed_running_light: bool
    return_home_confirmed: bool
    completed: bool
    events: tuple
    evidence_valid: bool
    source: str = "native_rotary_device"


class HeatTakeRules:
    """Gate the unchanged original plate scorer on a real completed cycle.

Extraction gets its own local step origin at the measured lamp-off boundary.
Actual timestamps/geometry remain unchanged; no prior task is marked complete.
    Approved points: knob approach 1, knob contact 1, complete cycle 2, and the
    original plate extraction 10. Original component clipping is preserved.
"""

    def __init__(self, episode_id, physics_dt_s, walls, *,
                 initial_state_accepted, payload_variant="beef_box"):
        if type(physics_dt_s) not in (int, float) or not any(
                abs(physics_dt_s-1/hz) < 1e-10 for hz in (60, 240)):
            raise ValueError("Existing selected-workspace physical clock required")
        self.episode_id = episode_id
        self.dt = physics_dt_s
        self.walls = walls
        self.bindings = bindings_for(payload_variant)
        self.accepted = initial_state_accepted
        self._preheat = self._new_extraction()
        self._interaction = Progress(episode_id, (KNOB,))
        self._control_substeps = round(.05/self.dt)
        if APPROACH_POINTS != .5 or TOUCH_POINTS != 1.:
            raise ValueError("Existing interaction contract changed")
        self._extraction = None
        self._unlock_step = None
        self._last_step = -1
        self._origin = None
        self._last_time = None
        self._seen_lit = False
        self._trigger_time = None
        self._finish_time = None
        self._fault = False
        self._state = None
        self._error = None
        self._finalized = False

    def _new_extraction(self):
        return ProcessScorer(self.episode_id, self.dt, self.walls,
            initial_state_accepted=self.accepted,
            hold_failure_penalty=HOLD_FAILURE_PENALTY, bindings=self.bindings)

    def configure_clock(self, physics_dt_s):
        """Adapt evidence indexing to an actual selector-owned rate segment.

        No simulator setting is changed. Original stage and hold timestamps
        remain real simulation time; only the source's constant-rate index
        origin is rebased at the measured segment boundary.
        """
        if type(physics_dt_s) not in (float, int) or not any(
                abs(physics_dt_s-1/hz) < 1e-10 for hz in (60, 240)):
            raise ValueError("Unsupported actual workspace timestep")
        if self._last_time is not None and abs((self._last_time-self._origin)*20-round((self._last_time-self._origin)*20)) > 1e-6:
            raise ValueError("Clock segments may change only at control boundaries")
        self.dt = physics_dt_s
        self._control_substeps = round(.05/self.dt)
        for scorer in (self._preheat, self._extraction):
            if scorer is not None:
                scorer.dt = physics_dt_s
                if scorer.last_time is not None:
                    scorer.start_time = scorer.last_time-scorer.last_step*physics_dt_s

    def pause(self):
        if self._extraction is not None:
            self._extraction.hold_start = None
            self._extraction.hold_elapsed = 0.
            self._extraction.hold_satisfied = False

    def observe(self, frame, heating, interaction=None):
        if self._finalized or self._error:
            raise RuntimeError("Closed or invalid task evidence")
        other = deepcopy(self)
        try:
            other._observe(frame, heating, interaction)
        except (ValueError, TypeError, KeyError) as exc:
            self._error = str(exc)
            raise ValueError(self._error) from exc
        self.__dict__.update(other.__dict__)
        return self.summary()

    def _observe(self, frame, heating, interaction):
        if not isinstance(frame, ProcessFrame) or not isinstance(heating, HeatingEvidence):
            raise ValueError("Current physical plate and heating evidence required")
        if (heating.source != "native_rotary_device" or heating.evidence_valid is not True
                or heating.episode_id != self.episode_id
                or heating.step_index != frame.step_index
                or type(frame.step_index) is not int
                or frame.step_index != self._last_step+1
                or frame.episode_id != self.episode_id
                or heating.time_s != frame.time_s
                or type(frame.time_s) not in (int, float)
                or not math.isfinite(frame.time_s)):
            raise ValueError("Mismatched, missing or stale native evidence")
        if self._origin is None:
            self._origin = frame.time_s
        if self._last_time is not None and abs(frame.time_s-self._last_time-self.dt) > 1e-6:
            raise ValueError("Missing physical substep")
        control_time = (frame.time_s-self._origin)*20
        boundary = abs(control_time-round(control_time)) < 1e-6
        if boundary:
            if not isinstance(interaction, Sample) or interaction.active is not True:
                raise ValueError("Current 20 Hz knob interaction sample required")
            if frame.step_index == 0 and (interaction.left.touching or interaction.right.touching):
                raise ValueError("Initial knob must not already be touched")
            self._interaction.observe(self.episode_id,
                round(control_time), frame.time_s,
                {KNOB: interaction}, baseline=frame.step_index == 0)
        elif interaction is not None:
            raise ValueError("Interaction facts belong to actual 20 Hz boundaries")
        self._validate_heating(heating)
        # Before the gate, reuse the original validation, including its accepted
        # closed/unheld/loaded initial frame. Never expose its pre-gate awards.
        if self._extraction is None:
            self._preheat.observe(frame)
        if heating.completed and self._extraction is None:
            self._unlock_step = frame.step_index
            self._extraction = self._new_extraction()
        if self._extraction is not None:
            if not heating.completed or self._fault:
                raise ValueError("Completed heating evidence was revoked")
            local = replace(frame, step_index=frame.step_index-self._unlock_step)
            self._extraction.observe(local)
        self._last_step = frame.step_index
        self._last_time = frame.time_s

    def _validate_heating(self, h):
        flags = (h.runtime_lamp_on, h.observed_running_light,
                 h.return_home_confirmed, h.completed)
        if any(type(v) is not bool for v in flags):
            raise ValueError("Actual native heating booleans required")
        intensity = h.actual_lamp_intensity
        if (type(intensity) not in (int, float) or not math.isfinite(intensity)
                or intensity < 0 or h.runtime_lamp_on != (intensity > 0)
                or h.state not in ("idle", "running", "done", "fault")
                or h.runtime_lamp_on != (h.state == "running")
                or not isinstance(h.events, tuple)):
            raise ValueError("Invalid measured lamp lifecycle")
        if self._last_step == -1 and (h.state != "idle" or intensity != 0
                or h.events or any(flags)):
            raise ValueError("Heating must start idle and physically uncompleted")
        if self._fault and h.state != "fault":
            raise ValueError("A faulted cycle cannot silently restart")
        transitions = {None: ("idle",), "idle": ("idle", "running", "fault"),
                       "running": ("running", "done", "fault"),
                       "done": ("done",), "fault": ("fault",)}
        if h.state not in transitions[self._state]:
            raise ValueError("Out-of-order native cycle state")
        self._state = h.state
        self._fault |= h.state == "fault"
        self._seen_lit |= h.state == "running" and intensity > 0
        if h.observed_running_light != self._seen_lit:
            raise ValueError("Running light was not observed in this episode")
        trigger = finish = None
        for event in h.events:
            if not isinstance(event, dict) or not isinstance(event.get("event"), str):
                raise ValueError("Malformed native mechanism event")
            t = event.get("seconds")
            if type(t) not in (int, float) or not math.isfinite(t) or not self._origin <= t <= h.time_s:
                raise ValueError("Stale or future mechanism event")
            if event["event"] == "rotation_contact_trigger":
                if trigger is not None:
                    raise ValueError("Repeated startup trigger")
                trigger = t
            if event["event"] == "cycle_finished_lamp_off":
                if finish is not None:
                    raise ValueError("Repeated cycle finish")
                finish = t
        if self._trigger_time is not None and trigger != self._trigger_time:
            raise ValueError("Startup history changed")
        if self._finish_time is not None and finish != self._finish_time:
            raise ValueError("Completion history changed")
        self._trigger_time, self._finish_time = trigger, finish
        if h.state == "running" and (intensity <= 0 or trigger is None or finish is not None):
            raise ValueError("Unobserved or inconsistent running cycle")
        expected = bool(h.state == "done" and not self._fault and self._seen_lit
            and trigger is not None and finish is not None and trigger < finish
            and h.return_home_confirmed and intensity == 0)
        if h.completed != expected or h.state == "done" and not expected:
            raise ValueError("No verified contact/light/return-home completion")

    def summary(self):
        extraction = self._extraction.summary() if self._extraction else None
        valid = self._error is None and self._last_step >= 0
        success = bool(valid and not self._fault and extraction and check_success(extraction))
        interaction = self._interaction.summary()
        item = interaction["items"][0]
        approach, contact = item["approach"]/APPROACH_POINTS, item["touch"]
        cycle = 2. if self._unlock_step is not None else 0.
        extraction_score = extraction["process_score"] if extraction else 0.
        stages = dict(approach=approach, contact=contact, illumination_cycle=cycle)
        stages.update({k: v["points"] if v else 0.
            for k, v in (extraction["stages"].items() if extraction else
                         dict.fromkeys(("open", "grasp", "take_out", "payload_retained", "close")).items())})
        deductions = (extraction["collision_deduction"]+extraction["hold_deduction"]) if extraction else 0.
        return dict(schema="arena.short_microwave_heat_take.rules.v1",
            episode_id=self.episode_id, last_step=self._last_step,
            evidence_valid=valid, error=self._error, heating_fault=self._fault,
            heating_completed=self._unlock_step is not None,
            extraction_origin_step=self._unlock_step, extraction=deepcopy(extraction),
            success=success if valid else None, finalized=self._finalized,
            score=approach+contact+cycle+extraction_score if valid else None,
            maximum_score=MAXIMUM, scoring_status="scoring" if valid else "invalid_or_missing_evidence",
            stage_points=stages, deducted_points=deductions,
            arithmetic="interaction_plus_cycle_plus_original_clipped_extraction",
            interaction=interaction,
            native_runtime_connected=False, physical_validation="not_run")

    def finalize(self):
        if self._error or self._last_step < 0:
            raise ValueError("Cannot settle invalid or missing physical evidence")
        if not self._finalized:
            if self._extraction is not None:
                self._extraction.finalize()
            self._finalized = True
        return self.summary()
