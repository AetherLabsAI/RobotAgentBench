"""Bind the existing measured rotary runtime to ALL downstream microwave credit.

No model-supplied completion flag, command-angle trigger, timer-only completion,
or assumed lamp state. Room integration must advance this device every native
substep, including while its scorer is inactive. This is not a room factory.
The existing runtime owns light/return mechanics; this link adds no drives.
"""
from dataclasses import replace
import math

from .final_state import Binding, PlacementFrame
from .rotary_component.rotary_return_runtime import RotaryReturnRuntime


class IlluminationGate:
    def __init__(self, binding, runtime, *, initial_time_s):
        if not isinstance(binding, Binding) or binding.kind != 'microwave_box_to_meal_tray':
            raise ValueError('Exact microwave task binding required')
        if not isinstance(runtime, RotaryReturnRuntime):
            raise ValueError('Measured rotary-return runtime required, not a completion flag')
        if runtime.cycle.state != 'idle' or runtime.cycle.events or runtime.cycle.last_time != -1:
            raise ValueError('Fresh episode-owned rotary runtime required')
        if runtime.lamp is None or not runtime.cycle.require_return_home:
            raise ValueError('Actual lamp attribute and measured return-home required')
        if type(initial_time_s) not in (float, int) or not math.isfinite(initial_time_s) or initial_time_s < 0:
            raise ValueError('Finite initial simulation time required')
        self.binding = binding
        self.runtime = runtime
        self.origin = self.last_time = float(initial_time_s)
        self.seen_running_lit = False
        self.completed = False
        self.failed = False
        self.last_result = None
        self._check_light(False)

    def _check_light(self, expected_on):
        value = self.runtime.lamp.Get()
        if type(value) not in (float, int) or not math.isfinite(value):
            raise ValueError('Actual lamp intensity missing or invalid')
        expected = self.runtime.on_intensity if expected_on else 0.
        if not math.isclose(value, expected, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError('Authored lamp state does not match measured rotary lifecycle')
        return float(value)

    def advance(self, now, physics_dt_s, *, body_q, knob_q, door_angle_deg,
                native_contacts, measured_gripper_open, minimum_finger_clearance_m):
        if self.failed:
            raise RuntimeError('Invalid illumination evidence; episode cannot resume scoring')
        try:
            if (type(now) not in (float, int) or not math.isfinite(now)
                    or type(physics_dt_s) not in (float, int)
                    or not any(abs(physics_dt_s-1/hz)<1e-10 for hz in (60,240))
                    or abs(now-self.last_time-physics_dt_s)>1e-6):
                raise ValueError('Every actual workspace physics substep is required')
            if type(measured_gripper_open) is not bool or not isinstance(native_contacts, (tuple,list)):
                raise ValueError('Measured gripper opening and native contact records required')
            # Check before update too: do not erase an unexplained external light
            # change and then pass it off as valid task evidence.
            self._check_light(self.runtime.cycle.lamp_on)
            result = self.runtime.step(now,body_q,knob_q,door_angle_deg,native_contacts,
                measured_gripper_open=measured_gripper_open,
                minimum_finger_clearance_m=minimum_finger_clearance_m)
            intensity = self._check_light(result['lamp_on'])
            self.seen_running_lit |= result['state']=='running' and intensity>0
            if result['state']=='done' and not self.completed:
                events=self.runtime.cycle.events
                triggered=any(e['event']=='rotation_contact_trigger' for e in events)
                finished=any(e['event']=='cycle_finished_lamp_off' for e in events)
                self.completed=bool(triggered and finished and self.seen_running_lit
                    and result['return_home_confirmed'] and intensity==0.)
                if not self.completed:
                    raise ValueError('Completed cycle lacks observed contact/light/return-home evidence')
            if result['state']=='fault':
                self.completed=False
            self.last_time=float(now)
            self.last_result=dict(result,actual_lamp_intensity=intensity)
            return self.summary()
        except Exception:
            self.failed=True
            self.completed=False
            raise

    def bind_frame(self, frame):
        if self.failed:
            raise RuntimeError('Cannot score invalid illumination evidence')
        if not isinstance(frame,PlacementFrame) or frame.binding!=self.binding:
            raise ValueError('Wrong illumination episode, task or target')
        if abs(frame.time_s-(self.last_time-self.origin))>1e-6:
            raise ValueError('Illumination must match this current control frame')
        # The caller cannot unlock credit by setting illumination_completed=True.
        return replace(frame,illumination_completed=self.completed)

    def summary(self):
        return dict(state=self.runtime.cycle.state,completed=self.completed,
            downstream_scoring_enabled=self.completed and not self.failed,
            observed_running_light=self.seen_running_lit,
            last_simulation_time_s=self.last_time,evidence_valid=not self.failed,
            runtime_registered=False,native_physics_verified=False)
