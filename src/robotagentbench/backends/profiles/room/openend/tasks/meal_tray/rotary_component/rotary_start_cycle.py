"""Pure, measured-angle startup state machine. No timer-only fake start."""
from dataclasses import dataclass, field
import math


@dataclass
class RotaryStartCycle:
    duration_seconds: float = 3.
    trigger_angle_deg: float = 30.
    contact_debounce_seconds: float = .05
    max_start_angle_deg: float = 5.
    require_return_home: bool = False
    state: str = 'idle'
    started_at: float | None = None
    ended_at: float | None = None
    contact_since: float | None = None
    contact_start_angle: float | None = None
    last_time: float = -1.
    events: list = field(default_factory=list)

    def __post_init__(self):
        if not (0 < self.duration_seconds <= 30 and 0 < self.trigger_angle_deg < 70
                and self.contact_debounce_seconds > 0
                and 0 <= self.max_start_angle_deg < self.trigger_angle_deg):
            raise ValueError('Invalid startup configuration')

    @property
    def lamp_on(self):
        return self.state == 'running'

    @property
    def may_open(self):
        return self.state == 'done'

    def step(self, now, knob_angle_deg, door_angle_deg, native_gripper_contact,
             *, return_home_confirmed=False):
        if (not all(math.isfinite(x) for x in [now, knob_angle_deg, door_angle_deg])
                or now < self.last_time):
            raise ValueError('Invalid measured startup input')
        self.last_time = now
        closed = abs(door_angle_deg) <= 2.
        if self.state == 'running' and not closed:
            self.state = 'fault'
            self.events.append(dict(event='door_opened_during_cycle', seconds=now))
        if self.state == 'idle':
            if native_gripper_contact and closed:
                if self.contact_since is None and 0 <= knob_angle_deg <= self.max_start_angle_deg:
                    self.contact_since = now
                    self.contact_start_angle = knob_angle_deg
            else:
                self.contact_since = None
                self.contact_start_angle = None
            if (self.contact_since is not None and knob_angle_deg >= self.trigger_angle_deg
                    and now-self.contact_since >= self.contact_debounce_seconds-1e-9):
                self.state = 'running'; self.started_at = now
                self.events.append(dict(event='rotation_contact_trigger', seconds=now,
                                        measured_angle_deg=knob_angle_deg,
                                        contact_start_angle_deg=self.contact_start_angle,
                                        measured_contact_rotation_deg=knob_angle_deg-self.contact_start_angle))
        if (self.state == 'running' and now-self.started_at >= self.duration_seconds-1e-9
                and (not self.require_return_home or return_home_confirmed)):
            self.state = 'done'; self.ended_at = now
            self.events.append(dict(event='cycle_finished_lamp_off', seconds=now))
        return dict(state=self.state, lamp_on=self.lamp_on, may_open=self.may_open)

    def assert_may_open(self):
        if not self.may_open:
            raise RuntimeError('Opening requires an actually triggered, completed startup cycle')
