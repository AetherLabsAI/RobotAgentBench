"""Robot-only convenience commands; no maps, scene queries or autonomous search.

All output is ordinary MOBILE22 input to the existing SessionHost. Distances
describe integrated commanded velocity, not guaranteed measured arrival.
"""
import math
from arena_api import mobile_base
from .robot_contract import ORDER, MAX_TOOL_STEPS

TOOLS = frozenset(('move_base', 'turn_base', 'look', 'wait'))
HEAD = ('idx11_head_joint1', 'idx12_head_joint2')


def number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(name + ' must be finite')
    return float(value)


def stopped_profile(amount, speed, acceleration, dt, budget):
    """Discrete symmetric accelerate/cruise/decelerate pulse ending at zero."""
    amount = number(amount, 'displacement')
    speed = number(speed, 'speed')
    if speed <= 0:
        raise ValueError('Speed must be positive')
    if amount == 0:
        return [0.]
    for count in range(2, budget + 1):
        profile = [min((i + 1)*acceleration*dt,
                       (count - 1 - i)*acceleration*dt, speed)
                   for i in range(count)]
        area = sum(profile)*dt
        if area >= abs(amount):
            scale = amount/area
            return [v*scale for v in profile]
    raise ValueError('Requested displacement exceeds the per-tool step budget; request a shorter move')


def compile_actions(request, previous, previous_base, contract):
    if contract['control_hz'] != 20 or contract['action_order'] != ORDER:
        raise ValueError('Current MOBILE22 contract required')
    spec = mobile_base.specification()
    if contract['base'] != spec:
        raise ValueError('Base contract differs from the existing executor')
    if any(previous_base.values()):
        raise ValueError('Base must be commanded stopped before a convenience motion')
    if set(previous) != set(ORDER[:20]):
        raise ValueError('All retained joint commands are required')
    dt = spec['control_dt_s']
    cap = min(contract['max_tool_steps'], MAX_TOOL_STEPS)
    op = request['op']
    if op in ('move_base', 'turn_base'):
        index = 0 if op == 'move_base' else 1
        displacement = request['distance_m'] if index == 0 else request['angle_rad']
        speed = number(request['speed_m_s'] if index == 0 else request['speed_rad_s'], 'speed')
        if speed > spec['max_abs'][index]:
            raise ValueError('Speed exceeds the existing base limit')
        values = stopped_profile(displacement, speed, spec['max_acceleration'][index], dt, cap)
        rows = []
        for value in values:
            base = [0., 0.]
            base[index] = value
            rows.append([previous[n] for n in ORDER[:20]] + base)
    elif op in ('look', 'wait'):
        seconds = number(request['duration_s'], 'duration_s')
        if seconds <= 0 or seconds > cap*dt:
            raise ValueError('Duration must be positive and within the per-tool budget')
        count = math.ceil(seconds/dt - 1e-12)
        goal = dict(previous)
        if op == 'look':
            for name, field in zip(HEAD, ('yaw_rad', 'pitch_rad')):
                goal[name] = number(request[field], field)
        rows = [[previous[n] + (goal[n]-previous[n])*(i+1)/count
                 for n in ORDER[:20]] + [0., 0.] for i in range(count)]
    else:
        raise ValueError('Unknown exploration tool')
    last_base = dict(previous_base)
    for row in rows:
        for name, value in zip(ORDER[:20], row[:20]):
            value = number(value, name)
            if not contract['joint_limits'][name][0] <= value <= contract['joint_limits'][name][1]:
                raise ValueError('Joint command exceeds native limits')
        last_base = mobile_base.validate(dict(zip(mobile_base.NAMES, row[20:])), last_base)
    return rows


def measured_motion(before, after, request, planned, executed):
    a, b = before['base_state'], after['base_state']
    dx, dy = [b['position_world_m'][i]-a['position_world_m'][i] for i in (0, 1)]
    yaw = a['yaw_world_unwrapped_rad']
    forward = dx*math.cos(yaw) + dy*math.sin(yaw)
    turn = b['yaw_world_unwrapped_rad']-yaw
    result = dict(tool=request['op'], command_complete=executed == planned,
                  planned_steps=planned, executed_steps=executed,
                  measured_forward_m=forward, measured_lateral_m=-dx*math.sin(yaw)+dy*math.cos(yaw),
                  measured_yaw_change_rad=turn, navigation_success_not_inferred=True,
                  environment_collision_check=False)
    if request['op'] == 'move_base':
        result['remaining_distance_m'] = request['distance_m']-forward
    if request['op'] == 'turn_base':
        result['remaining_angle_rad'] = request['angle_rad']-turn
    if request['op'] == 'look':
        result['head_error_rad'] = {n: request[f]-after['joints'][n]
                                   for n, f in zip(HEAD, ('yaw_rad', 'pitch_rad'))}
    return result
