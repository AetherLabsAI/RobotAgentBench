"""Glue for an already admitted native owner; no setup/parameter overrides.

Uses the existing NativeAdapter and mobile controller, not a second executor.
Callbacks are supplied by trusted owner code, never by model input. The owner
must freeze physics while waiting for tools and capture all three cameras at
the current control boundary without an extra physics tick.
"""
from copy import deepcopy

from robotagentbench.interfaces import robot_profiles as rp
from robotagentbench.interfaces.action_contract import CAMERAS, public_base


class NativeDriver:
    def __init__(self, session, adapter, mobile_controller, *, measured_joints,
                 apply_native_targets, simulate_fetch, capture_png):
        if adapter.profile.id != rp.MOBILE22:
            raise ValueError('The native owner must explicitly initialize the 22D adapter')
        if mobile_controller.previous != {'base_forward_m_s': 0., 'base_yaw_rate_rad_s': 0.}:
            raise ValueError('Cannot attach to an already commanded moving base')
        self.session, self.adapter, self.mobile = session, adapter, mobile_controller
        self.read_joints = measured_joints
        self.apply_targets = apply_native_targets
        self.simulate_fetch = simulate_fetch
        self.capture_png = capture_png
        self.limits = {n: list(adapter.limits[n]) for n in rp.get(rp.MOBILE22).order}

    def validate_targets(self, targets):
        self.adapter.expand(targets)  # Includes paired fingers, no physical write.

    def base_state(self):
        return public_base(self.mobile.observe())

    def observe(self):
        now = self.session.clock.current_time
        step = self.session.control_index
        joints = {name: float(value) for name, value in self.read_joints().items()}
        base = self.base_state()
        images = self.capture_png()  # {head,left,right: base64 PNG}; render-only.
        if self.session.clock.current_time != now or self.session.control_index != step:
            raise RuntimeError('Camera capture changed the physics/control clock')
        return dict(episode_id=self.session.episode_id, step=step, time_s=float(now),
                    instruction=self.session.plan['instruction'], joints=joints, base_state=base,
                    image_step=step, images={name: images[name] for name in CAMERAS})

    def apply_action(self, targets, velocity):
        native = self.adapter.expand(targets)
        self.mobile.apply(velocity)
        self.apply_targets(deepcopy(native))  # Existing drives, not q/qvel restoration.

    def advance(self, dt):
        before = self.session.clock.current_time
        if abs(dt-1/self.session.clock.hz) > 1e-12:
            raise ValueError('Driver timestep differs from the selected workspace clock')
        # Trusted environment actors only; no actor controls in MOBILE22 actions.
        hook = getattr(self.session, 'before_physics', None)
        if hook is not None:
            hook(dt)
            if self.session.clock.current_time != before:
                raise RuntimeError('Environment actor hook must not step physics')
        report = self.simulate_fetch(dt)
        if abs(self.session.clock.current_time-before-dt) > 1e-6:
            raise RuntimeError('simulate/fetch must advance the shared clock exactly once')
        if not isinstance(report, dict):
            raise RuntimeError('Native helper evidence is required, not invented by the bridge')
        return report
