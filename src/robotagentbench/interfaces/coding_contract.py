"""Public robot-only contract. No task/scene/Isaac or privileged reader imports."""
import base64
from copy import deepcopy
import math
import struct

from . import mobile_base, robot_profiles as rp
from .observation_extras import public_extras

ORDER = list(rp.get(rp.MOBILE22).order) + list(mobile_base.NAMES)
MAX_TOOL_STEPS = 600  # Existing GPT trial bound, not a task time limit.
CAMERAS = ('head', 'left', 'right')
BASE_FIELDS = dict(position_world_m=3, orientation_world_xyzw=4,
                   linear_velocity_world_m_s=3, angular_velocity_world_rad_s=3)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def specification(limits):
    if set(limits) != set(ORDER[:20]):
        raise ValueError('Require measured native command limits')
    for pair in limits.values():
        if len(pair) != 2 or not all(map(finite, pair)) or pair[0] >= pair[1]:
            raise ValueError('Invalid native limits')
    return dict(schema='short_tasks.gpt.robot22.v1', robot_profile=rp.MOBILE22,
                control_hz=20, action_order=ORDER, joint_units=rp.NATIVE_UNITS,
                joint_limits=deepcopy(limits), base=mobile_base.specification(),
                max_tool_steps=MAX_TOOL_STEPS, camera_order=list(CAMERAS))


def public_base(source):
    base = {}
    for key, size in BASE_FIELDS.items():
        values = source.get(key)
        if not isinstance(values, (list, tuple)) or len(values) != size or not all(map(finite, values)):
            raise ValueError('Missing measured base state')
        base[key] = list(values)
    yaw = source.get('yaw_world_unwrapped_rad')
    if not finite(yaw) or abs(sum(x*x for x in base['orientation_world_xyzw'])-1) > .01:
        raise ValueError('Invalid measured base orientation')
    base['yaw_world_unwrapped_rad'] = yaw
    return base


def public_robot(row):
    """Recursively whitelist robot state; never forward arbitrary nested metadata."""
    if not isinstance(row, dict) or not isinstance(row.get('episode_id'), str) or not row['episode_id']:
        raise ValueError('Missing measured episode')
    if type(row.get('step')) is not int or row['step'] < 0 or not finite(row.get('time_s')):
        raise ValueError('Missing measured frame identity')
    if not isinstance(row.get('instruction'), str):
        raise ValueError('Missing public instruction')
    joints = row.get('joints', {})
    if set(joints) != set(rp.NATIVE_JOINTS) or not all(map(finite, joints.values())):
        raise ValueError('Need finite named native34 state, without unit conversion')
    base = public_base(row.get('base_state', {}))
    return dict(episode_id=row['episode_id'], step=row['step'], time_s=row['time_s'],
                instruction=row['instruction'], joints=dict(joints), base_state=base,
                **public_extras(row, include_navigation=False))


def public_observation(row):
    # Share the current public whitelist and coherent camera profiles. No resize,
    # padding, pose changes, or permissive per-camera fallback in this adapter.
    from .action_contract import public_observation as current_observation
    return current_observation(row)
