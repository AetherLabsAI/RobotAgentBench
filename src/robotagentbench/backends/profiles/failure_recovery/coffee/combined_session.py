"""Two existing native actors, one read-only state stream and one action clock.

Owner: construct before reset, bind after reset, arm after native initial
acceptance, command_before_step, advance ONCE, update reader ONCE, observe.
There is no scorer, robot controller, simulation stepping or global registry.
"""
from .variants import BAG_MOVE, MACHINE_PUSH, InterferenceSequence


class CombinedNativeSession:
    def __init__(self, stage, initial):
        from .native_bag import NativeBagRelocation
        from .native_push import NativePalmPush
        if initial.get('variant') != 'bag_then_machine' or 'relocation_pair' not in initial:
            raise ValueError('Use an explicitly composed combined initial, not a machine-only replay')
        self.sequence = InterferenceSequence('bag_then_machine')
        self.bag = NativeBagRelocation(stage, initial)
        try:
            self.push = NativePalmPush(stage, initial)
        except BaseException:
            self.bag.close()
            raise
        self.reader = None
        self.armed = False
        self.last_observed_time = None
        self.events = []

    def bind(self, simulation_view, reader):
        if self.reader is not None:
            raise RuntimeError('Bind combined native session only once')
        self.bag.bind(simulation_view, reader)
        self.push.bind(simulation_view, reader)
        self.reader = reader

    def arm(self, *, robot_native_poses):
        if self.armed or self.reader is None:
            raise RuntimeError('Bind and validate native initial before arming once')
        self.bag.arm(robot_native_poses=robot_native_poses)
        self.push.arm()
        self.armed = True

    @property
    def robot_process_scoring_enabled(self):
        return self.sequence.robot_process_scoring_enabled

    def command_before_step(self, time_s):
        if not self.armed:
            raise RuntimeError('Cannot command before native initial acceptance')
        if self.sequence.invalid_reason is not None:
            raise RuntimeError('Invalid intervention cannot resume: ' + self.sequence.invalid_reason)
        if float(time_s) != float(self.reader.world.current_time):
            raise RuntimeError('Use the owner current native physical clock')
        active = self.sequence.active
        if active == BAG_MOVE:
            return self.bag.command_before_step(time_s)
        if active == MACHINE_PUSH:
            return self.push.command_before_step(time_s)
        return {'phase': 'absent'}

    def observe(self, *, robot_native_poses):
        if not self.armed:
            raise RuntimeError('Cannot observe before native initial acceptance')
        if self.sequence.invalid_reason is not None:
            raise RuntimeError('Invalid intervention cannot resume: ' + self.sequence.invalid_reason)
        now = float(self.reader.world.current_time)
        if (self.reader.latest is None or self.reader.latest['time_s'] != now
                or self.reader.last_time != now
                or (self.last_observed_time is not None and now <= self.last_observed_time)):
            raise RuntimeError('Exactly one fresh native observation per substep required')
        self.last_observed_time = now
        component = self.sequence.active or self.sequence.eligible_component
        if component is None:
            return dict(start=False, reason='interference_sequence_complete', time_s=now)
        actor = self.bag if component == BAG_MOVE else self.push
        try:
            event = actor.observe(robot_native_poses=robot_native_poses)
            if event.get('start'):
                self.sequence.begin(component, simulation_time_s=now)
                self.events.append(dict(kind='started', component=component, time_s=now))
            if actor.completed and actor.retired:
                if component == BAG_MOVE and self.bag.completion_evidence is None:
                    raise RuntimeError('Missing native relocation completion evidence')
                self.sequence.complete(component, simulation_time_s=now,
                    native_completion_confirmed=True, actor_retired=True)
                self.events.append(dict(kind='completed', component=component, time_s=now))
            return dict(event, component=component)
        except Exception as error:
            self.sequence.invalidate(component, simulation_time_s=now, reason=str(error))
            raise

    def close(self):
        errors = []
        for actor in (self.bag, self.push):
            try:
                actor.close()
            except Exception as error:
                errors.append(str(error))
        if errors:
            raise RuntimeError('Combined actor cleanup failed: ' + '; '.join(errors))
