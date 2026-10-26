"""Coffee demand -> shared scene clock, preserving a fixed 20 Hz control tick.

Consumes the existing multi-activation.js result (sector, near wake, hysteresis,
held/moving protections). It never substitutes a centre-distance wake test.
One PhysX scene has one timestep: an active coffee raises that scene to 240 Hz,
not just the machine. No independent per-object solver is claimed.
"""
import math


def demand(service, activation):
    if type(activation.get('active')) is not bool or activation.get('state') not in (
            'ACTIVE', 'PINNED', 'PRELOAD', 'DORMANT'):
        raise ValueError('Bound, valid multi-activation result required')
    if activation['active'] != (activation['state'] in ('ACTIVE', 'PINNED')):
        raise ValueError('Inconsistent activation state')
    reasons = []
    if activation['active']: reasons.append('existing_region_active')
    if service.consumer is not None and not service.consumer.process.ended:
        reasons.append('unfinished_score_keeps_original_clock_and_budget')
    if service.controller.dispensing or service.animation.counts()['airborne']:
        reasons.append('device_or_inflight_coffee_active')
    return dict(hz=240 if reasons else 60, reasons=reasons)


class NativeSceneClock:
    """Port for the existing explicit sim.simulate(dt,t)/fetch_results runner.

snapshot must return immutable bytes of ALL measured native poses/velocities;
read_time returns the runner's accumulated simulation time, never steps/hz.
This class does not step, reset, save, move bodies or alter renderer settings.
"""
    def __init__(self, stage, simulation, read_time, snapshot):
        self.stage, self.simulation = stage, simulation
        self.read_time, self.snapshot = read_time, snapshot
        self.attribute = stage.GetPrimAtPath('/World/PhysicsScene').GetAttribute('physxScene:timeStepsPerSecond')
        if not self.attribute or self.hz not in (60, 240):
            raise ValueError('Expected existing 60/240 Hz scene')

    @property
    def hz(self): return self.attribute.Get()

    @property
    def current_time(self): return float(self.read_time())

    def get_physics_dt(self): return 1/self.hz

    def set_hz(self, hz):
        if hz not in (60, 240): raise ValueError('Unsupported clock')
        from pxr import Usd
        with Usd.EditContext(self.stage, self.stage.GetSessionLayer()):
            self.attribute.Set(hz)
        self.simulation.flush_changes()


class CoffeeHotwakeCycle:
    """One begin/finish pair per action; no task reset on sector re-entry.

The scene owner holds the validated action for the returned 3 or 12 substeps,
calls its existing sim.simulate(dt, accumulated_time), and updates real time.
It calls startup.after_physics after each fetch. Native-source setup/initial
acceptance remain the host's responsibility; wake never awards task points.
"""
    def __init__(self, startup, clock):
        if startup.service is None or startup.closed:
            raise ValueError('Prepare the persistent device first')
        self.startup, self.clock = startup, clock
        self.pending = None
        self.frame = 0
        self.transitions = []
        self.faulted = False

    def begin_control(self, activation, *, command_vw, base_velocity, other_required_hz=60):
        if self.faulted or self.pending is not None or self.startup.closed:
            raise RuntimeError('Previous frame incomplete or session closed/faulted')
        if other_required_hz not in (60, 240):
            raise ValueError('Other region has an unsupported or unconfirmed rate')
        if len(command_vw) != 2 or len(base_velocity) != 6 or not all(
                math.isfinite(x) for x in (*command_vw, *base_velocity)):
            raise ValueError('Measured native base velocity and actual command required')
        wanted = demand(self.startup.service, activation)
        target = max(wanted['hz'], other_required_hz)
        current = self.clock.hz
        if current not in (60, 240): raise RuntimeError('Unexpected scene clock')
        # Reuse the previous native stopped-switch probe's limits, not new tuning.
        stopped = (max(map(abs, command_vw)) < .001
                   and math.sqrt(sum(v*v for v in base_velocity[:3])) < .005
                   and math.sqrt(sum(v*v for v in base_velocity[3:])) < .01)
        transition = None
        if target != current and stopped:
            snapshot, t = self.clock.snapshot(), self.clock.current_time
            if not isinstance(snapshot, bytes): raise ValueError('Owned immutable native snapshot required')
            self.faulted = True  # a failed setter/readback requires host stop, never reset to hide it
            self.clock.set_hz(target)
            if self.clock.hz != target or self.clock.current_time != t or self.clock.snapshot() != snapshot:
                raise RuntimeError('Clock switch changed physical state/time or failed readback')
            self.faulted = False
            transition = dict(frame=self.frame, from_hz=current, to_hz=target, time_s=t,
                              physical_state_unchanged=True)
            self.transitions.append(transition)
        hz = self.clock.hz
        # Never feed active 240 Hz scoring from a temporarily deferred 60 Hz switch.
        scoring = self.startup.service.consumer
        if scoring is not None and not scoring.process.ended and hz != 240:
            raise RuntimeError('Cannot advance active coffee scoring at navigation rate')
        result = dict(frame=self.frame, physics_hz=hz, physics_dt_s=1/hz,
                      substeps=hz//20, control_hz=20, desired_hz=target,
                      switch_pending=hz != target, coffee_rate_ready=hz == 240,
                      reasons=wanted['reasons'], transition=transition)
        self.pending = (self.clock.current_time, result)
        return dict(result)

    def finish_control(self, *, completed_substeps):
        if self.pending is None or self.faulted: raise RuntimeError('No valid control frame')
        start, plan = self.pending
        if (completed_substeps != plan['substeps'] or self.clock.hz != plan['physics_hz']
                or abs(self.clock.current_time-start-.05) > 1e-6):
            self.faulted = True
            raise RuntimeError('Missing substeps, mid-action switch or broken 20 Hz clock')
        self.pending = None
        self.frame += 1
        return dict(control_frame=self.frame-1, simulation_time_s=self.clock.current_time,
                    physics_hz=self.clock.hz, control_hz=20)
