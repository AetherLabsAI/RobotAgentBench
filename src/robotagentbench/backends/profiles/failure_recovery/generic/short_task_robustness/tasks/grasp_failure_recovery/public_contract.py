"""Public robot-only contract. No task/scene/Isaac or privileged reader imports."""
import base64
from copy import deepcopy
import math
import struct

from . import public_base as mobile_base, public_profiles as rp

ORDER = list(rp.get(rp.MOBILE22).order) + list(mobile_base.NAMES)
MAX_TOOL_STEPS = 600  # Existing GPT trial bound, not a task time limit.
CAMERAS = ('head', 'left', 'right')
# Standard native render products; no transport resizing or legacy profile.
CAMERA_SIZE_PROFILES = (((1280,800), (848,480), (848,480)),)
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


def public_observation(row):
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
    if row.get('image_step') != row['step']:
        raise ValueError('Images and robot state are not from the same control frame')
    images = {}
    image_sizes = []
    for camera in CAMERAS:
        value = row.get('images', {}).get(camera)
        if not isinstance(value, str) or len(value) > 8*1024*1024:
            raise ValueError('Missing/oversized camera image')
        raw = base64.b64decode(value, validate=True)
        if len(raw) < 33 or raw[:8] != b'\x89PNG\r\n\x1a\n' or raw[12:16] != b'IHDR':
            raise ValueError('Require native PNG cameras')
        image_sizes.append(struct.unpack('>II', raw[16:24]))
        images[camera] = value
    if tuple(image_sizes) not in CAMERA_SIZE_PROFILES:
        raise ValueError('Require native head 1280x800 / wrist 848x480 PNG cameras')
    return dict(episode_id=row['episode_id'], step=row['step'], time_s=row['time_s'],
                instruction=row['instruction'], joints=dict(joints), base_state=base,
                image_step=row['step'], images=images)


def function(name, description, properties):
    return dict(type='function', function=dict(name=name, description=description,
        parameters=dict(type='object', properties=properties,
                        required=list(properties), additionalProperties=False)))


TOOLS = [
    function('execute_actions', 'Execute explicit physical-unit 22D rows at 20 Hz, then observe. No interpolation or IK.',
        {'actions': dict(type='array', minItems=1, maxItems=MAX_TOOL_STEPS,
            items=dict(type='array', minItems=22, maxItems=22, items=dict(type='number')))}),
    function('observe', 'Read the current three cameras and robot state; advances no physics.', {}),
    function('done', 'Stop this attempt; only the environment decides task success.',
             {'reason': dict(type='string')}),
]

SYSTEM = '''You control a G1 robot in one Short Task using only public cameras and measured robot state.
Use the supplied robot contract: 22 physical-unit channels, 20 Hz, each row is exactly 0.05 simulation seconds.
The first 20 are absolute joint targets: arms, body, head, grippers; body joint1 is meters,
other joints are radians. Grippers are NATIVE radians, NOT normalized 0..1.
Last two channels are forward velocity m/s and yaw velocity rad/s. Read their speed/acceleration
and mode-change limits. Never silently drop them. All rows are checked before the first is sent.
Do not request simultaneous translation/rotation. Decelerate explicitly before changing modes.
Observe is read-only: use explicit valid hold rows to let brewing/holding time advance.
There is no privileged object pose, contact, score, stage, expert trajectory, hidden scene access,
automatic grasp, obstacle avoidance or teleportation. Infer targets from images. This raw-action
tool is not the old move_to/execute_python sandbox; it performs no IK and cannot execute Python.
Keep motions small and inspect feedback. Base and arm motion are native drives, not guaranteed safe paths.
Exactly one tool call per reply. Summarize visible results briefly in Chinese without private reasoning.
Calling done reports your assessment and ends the attempt, never awards points.
'''
