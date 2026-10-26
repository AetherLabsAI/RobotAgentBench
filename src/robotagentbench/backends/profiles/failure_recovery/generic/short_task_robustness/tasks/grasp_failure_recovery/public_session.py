"""Existing robot22 SessionHost facade over the original TaskLifecycle.

No new scoring, timers, workspace switching, physical writes or policy input.
The original atomic/shelf lifecycle owns the simulation-time deadline.
"""
from arena_api import policy_execution as pe


class Clock:
    def __init__(self, world):
        self.world = world
        self.hz = round(1/pe.PHYSICS_DT)
        if abs(self.hz*pe.PHYSICS_DT-1) > 1e-12:
            raise ValueError('Original physics clock is not integral')

    @property
    def current_time(self):
        return float(self.world.current_time)


class RecoverySession:
    def __init__(self, lifecycle, world, instruction):
        if lifecycle.phase != 'running' or lifecycle.clock.steps != 0:
            raise ValueError('Attach only after the original initial acceptance, before actions')
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError('Explicit public task instruction required')
        self.lifecycle = lifecycle
        self.clock = Clock(world)
        self.plan = dict(instruction=instruction)
        self.episode_id = lifecycle.context.episode_id
        self.started = True
        self.closed = self.faulted = self.paused = False
        self.pending = None

    @property
    def control_index(self):
        return self.lifecycle.clock.steps

    def begin_control(self, base_pose, *, command_vw, base_velocity):
        if self.closed or self.faulted or self.pending is not None or self.lifecycle.phase != 'running':
            raise RuntimeError('No control may begin in this session state')
        self.pending = dict(start=self.clock.current_time, substeps=0)
        return dict(physics_hz=self.clock.hz, substeps=pe.SUBSTEPS)

    def after_physics(self, helper_report):
        if self.pending is None:
            raise RuntimeError('Unrequested physical advance')
        self.lifecycle.after_physics(helper_report)
        self.pending['substeps'] += 1

    def finish_control(self, *, completed_substeps):
        if self.pending is None or completed_substeps != pe.SUBSTEPS or \
                self.pending['substeps'] != completed_substeps:
            raise RuntimeError('Incomplete original control interval')
        expected = self.pending['start']+pe.SUBSTEPS*pe.PHYSICS_DT
        if abs(self.clock.current_time-expected) > 1e-6:
            raise RuntimeError('Native advance differs from the original control clock')
        self.lifecycle.after_control(self.clock.current_time)
        self.pending = None
        return self.result()

    def _fault(self, exception):
        self.faulted = True
        self.lifecycle.close(completed=False)

    def result(self):
        value = self.lifecycle.evaluation() or {}
        return dict(status='error' if self.faulted else 'completed' if
            self.closed or self.lifecycle.phase != 'running' else 'running',
            score=value.get('score'), maximum_score=value.get('maximum_score'),
            success=value.get('success'), evaluation_valid=value.get('valid') is True,
            scoring_status='valid' if value.get('valid') is True else 'invalid',
            reason=value.get('reason'), step=self.control_index,
            evaluation=value)

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.lifecycle.close(completed=not self.faulted and self.pending is None)
