"""Pure mobile action contract: no simulator, task, scene or Openend imports.

Limits mirror room_demo_v6/action22.py; this HTTP executor controls at 20 Hz.
Joint targets remain native positions; base commands are velocities, not joints.
"""
import math

NAMES = ('base_forward_m_s', 'base_yaw_rate_rad_s')
CONTROL_DT = .05
ZERO = dict(zip(NAMES, (0., 0.)))


def split_action(action):
    """Split one physical-unit 22D vector; do not normalize grippers or clip."""
    from . import public_profiles as rp
    if not isinstance(action, (list, tuple)) or len(action) != 22:
        raise ValueError('Expected exactly 22 physical-unit action values')
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in action):
        raise ValueError('Expected finite numeric action values')
    return dict(zip(rp.get(rp.MOBILE22).order, action[:20])), dict(zip(NAMES, action[20:]))


def specification():
    return dict(required=True, fields=list(NAMES), units=['m/s', 'rad/s'],
                frame='robot_forward_and_world_z', max_abs=[.18, .25],
                max_acceleration=[.12, .30], control_dt_s=CONTROL_DT,
                simultaneous_translation_rotation=False,
                executor='L1_planar_constraints_not_wheels',
                mode_switch='stop command before switching straight/turn',
                joint_action_dimensions=20, base_action_dimensions=2)


def validate_profile(profile, velocity):
    from . import public_profiles as rp
    if profile == rp.MOBILE22:
        if velocity is None:
            raise ValueError('22D profile requires explicit base_velocity, including zero commands')
    elif velocity is not None:
        raise ValueError('base_velocity requires the explicit 22D robot profile; never silently dropped')


def validate(velocity, previous):
    for label, values in (('base_velocity', velocity), ('previous base_velocity', previous)):
        if not isinstance(values, dict) or set(values) != set(NAMES):
            raise ValueError(label + ': expected forward m/s and yaw rad/s')
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
               for v in values.values()):
            raise ValueError(label + ': non-finite/non-numeric velocity')
    v, w = (velocity[n] for n in NAMES)
    if abs(v) > .18 + 2e-7 or abs(w) > .25 + 2e-7:
        raise ValueError('Base speed exceeds existing navigation limits')
    if abs(v) > 1e-8 and abs(w) > 1e-8:
        raise ValueError('Base supports straight motion OR in-place turning, not both')
    for name, acceleration in zip(NAMES, (.12, .30)):
        if abs(velocity[name] - previous[name]) > acceleration * CONTROL_DT + 2e-7:
            raise ValueError('Base acceleration exceeds existing navigation limit: ' + name)
    # The original planar executor reanchors constraints only after stopping.
    if ((abs(v) > 1e-8 and abs(previous[NAMES[1]]) > 1e-8) or
            (abs(w) > 1e-8 and abs(previous[NAMES[0]]) > 1e-8)):
        raise ValueError('Send a zero base command before switching straight/turn modes')
    return {n: float(velocity[n]) for n in NAMES}


class StatePacker:
    def __init__(self):
        self.last_yaw = None
        self.unwrapped = 0.

    def pack(self, pose, velocity):
        if len(pose) != 7 or len(velocity) != 6 or not all(math.isfinite(float(x)) for x in (*pose, *velocity)):
            raise ValueError('Expected finite measured base pose xyz+xyzw and 6D velocity')
        x, y, z, w = map(float, pose[3:])
        if abs(x*x+y*y+z*z+w*w-1) > .01:
            raise ValueError('Invalid measured base quaternion')
        yaw = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
        if self.last_yaw is None:
            self.unwrapped = yaw
        else:
            delta = yaw-self.last_yaw
            self.unwrapped += math.atan2(math.sin(delta), math.cos(delta))
        self.last_yaw = yaw
        return dict(position_world_m=list(map(float, pose[:3])),
                    orientation_world_xyzw=list(map(float, pose[3:])),
                    yaw_world_unwrapped_rad=self.unwrapped,
                    linear_velocity_world_m_s=list(map(float, velocity[:3])),
                    angular_velocity_world_rad_s=list(map(float, velocity[3:])))
