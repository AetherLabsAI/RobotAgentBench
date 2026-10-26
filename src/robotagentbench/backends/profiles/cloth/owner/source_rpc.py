"""Existing SessionHost transport over the original source's physical owner.

No robot/world construction or autonomous repair actions. A trusted, separately
validated current-frame capture callback is REQUIRED; no blank/stale images.
This module alone does not certify camera synchronization or register a worker.
"""
import math

from robotagentbench.interfaces import robot_profiles as rp
from robotagentbench.interfaces.action_contract import public_base
from robotagentbench.runtime.action_host import InputRejected, SessionHost


class SourceDriver:
    def __init__(self, *, owner, controls, capture_png):
        if not callable(capture_png) or owner.session is None:
            raise ValueError('Accepted native owner and independently verified capture required')
        self.owner, self.controls, self.capture = owner, controls, capture_png
        self.limits = {name: list(controls.adapter.limits[name])
                       for name in rp.get(rp.MOBILE22).order}
        self.pending_targets = None
        self._image_key, self._image_payload = None, None

    def validate_targets(self, targets):
        self.controls.adapter.expand(targets)  # Read-only, entire chunk checked first.

    def base_state(self):
        row = self.owner.current
        index = row['body_names'].index('base_link')
        pose, velocity = row['body_pose'][index], row['body_velocity'][index]
        x, y, z, w = map(float, pose[3:])
        yaw = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
        return public_base(dict(position_world_m=pose[:3].tolist(),
            orientation_world_xyzw=pose[3:].tolist(),
            linear_velocity_world_m_s=velocity[:3].tolist(),
            angular_velocity_world_rad_s=velocity[3:].tolist(),
            yaw_world_unwrapped_rad=yaw))

    def observe(self):
        owner, session = self.owner, self.owner.session
        before = owner._clock()
        frame = owner.current
        if before[0] != frame['physics_step']:
            raise RuntimeError('Stale native observation boundary')
        joints = dict(zip(owner.native['names'], map(float, frame['joints'])))
        base = self.base_state()
        key = (owner.episode_id, frame['physics_step'], session.control_index)
        # Multiple protocol reads at one frozen physical boundary share only
        # that boundary's image. Never reuse it after a physical control step.
        if self._image_key != key:
            images = self.capture(frame)
        else:
            images = dict(self._image_payload)
        after = owner._snapshot()
        import numpy as np
        if (owner._clock() != before
                or any(not np.array_equal(after[k], frame[k])
                       for k in ('positions', 'velocities', 'joints', 'joint_velocities',
                                 'body_pose', 'body_velocity'))):
            raise RuntimeError('Capture mutated the source physical state or clock')
        self._image_key, self._image_payload = key, dict(images)
        return dict(episode_id=owner.episode_id, step=session.control_index,
            time_s=session.clock.current_time, instruction=session.plan['instruction'],
            joints=joints, base_state=base, image_step=session.control_index, images=images)

    def apply_action(self, targets, velocity):
        self.pending_targets = self.controls.public_targets(targets, velocity)

    def advance(self, dt):
        if dt != 1/60 or self.pending_targets is None:
            raise ValueError('One original 60Hz substep with admitted native targets required')
        # Original source reissues exactly these drive targets every substep.
        self.owner._advance(self.pending_targets)
        callback = getattr(self, 'diagnostic_trace', None)
        if callback is not None:
            callback(self.owner.current)
        return dict(events=[], docked_object_count=0, tray_dock_enabled=False)


class FixedSourceHost(SessionHost):
    def _validate(self, request):
        checked = super()._validate(request)
        try:
            for _, velocity in checked:
                self.driver.controls.reject_moving_base(velocity)
        except ValueError:
            raise InputRejected('This original native owner has a fixed base; both base commands must be zero') from None
        return checked

    def ready(self):
        result = super().ready()
        result['native_owner_constraints'] = dict(
            fixed_base=True, required_base_commands=[0., 0.],
            articulated_commands='20 named native physical-unit channels',
            original_source_substeps_per_control=3)
        return result


def bind_source_host(*, owner, controls, capture_png):
    driver = SourceDriver(owner=owner, controls=controls, capture_png=capture_png)
    return FixedSourceHost(owner.session, driver)


def session_handler(*, capture_png, input_stream=None, output_stream=None):
    """Trusted dependency injection; never import a capture module from an RPC."""
    if not callable(capture_png):
        raise ValueError('No formal native RPC without verified current-frame camera capture')
    def handle(owner, controls):
        from robotagentbench.runtime.action_host import serve
        host = bind_source_host(owner=owner, controls=controls, capture_png=capture_png)
        serve(host, input_stream=input_stream, output_stream=output_stream)
    return handle


def serve_original_owner(owner, controls, *, input_stream, output_stream):
    """Trusted operator streams, same common JSONL protocol and native owner.

    No scene/reset/module/path can be supplied through those action messages.
    The original initializer remains responsible for the physical lifecycle.
    """
    if owner.cameras is None or not owner.isolated_camera_regression:
        raise ValueError('Admitted isolated native observation owner required')
    from robotagentbench.runtime.action_host import serve
    host = bind_source_host(owner=owner, controls=controls,
        capture_png=lambda frame: owner.cameras.capture(owner))
    serve(host, input_stream=input_stream, output_stream=output_stream)
    return host
