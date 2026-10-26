"""Task 10.2 initial calibration candidates. Not a registered reset driver."""
import math
from numbers import Real

TASK_ID = '10.2'
INSTRUCTION = '将装有牛排盒的盘子放到餐桌上，然后松开夹爪并撤手。'
DESTINATION = '/World/Background/Shell/CounterTop_17'
ROBOT = '/World/Robot'
SOURCE_SUPPORT_TOP_M = .45


def hold_targets(plan):
    """Mirror existing jaw coordinates; do not add or enable a passive drive."""
    result = {n: v['value'] for n, v in plan['joint_targets'].items()}
    aperture = plan['settle_gripper_target_rad']
    if not math.isfinite(aperture) or not 0 <= aperture <= .62:
        raise ValueError('Invalid hold opening')
    result['idx81_gripper_r_outer_joint1'] = aperture
    result['idx71_gripper_r_inner_joint1'] = -aperture
    result['idx31_gripper_l_inner_joint1'] = -result['idx41_gripper_l_outer_joint1']
    return result


def height_candidate(source_initial, table_top_m, lift_limits_m):
    """Adjust only the pre-episode vertical prismatic joint, never base Z.

    This preserves the recorded arm angles in kinematics if the lift's world
    axis is vertical. The USD preparer must verify that axis. It says nothing
    about contact stability, current plate geometry, XY station, or reachability.
    """
    values = (table_top_m, *lift_limits_m)
    if (len(lift_limits_m) != 2 or any(isinstance(v, bool) or not isinstance(v, Real)
            or not math.isfinite(v) for v in values) or lift_limits_m[0] >= lift_limits_m[1]):
        raise ValueError('Finite table height and ordered native lift limits required')
    old = source_initial['stationary_joint_targets']['idx01_body_joint1']['value']
    target = old + table_top_m - SOURCE_SUPPORT_TOP_M
    if not lift_limits_m[0] <= target <= lift_limits_m[1]:
        raise ValueError('Dining height exceeds native lift range; retargeting is required')
    fixed = {k: dict(v) for k, v in source_initial['stationary_joint_targets'].items()}
    fixed['idx01_body_joint1']['value'] = target
    return dict(task_id=TASK_ID, instruction=INSTRUCTION, status='calibration_candidate_only',
        destination=DESTINATION, table_top_m=float(table_top_m),
        source_support_top_m=SOURCE_SUPPORT_TOP_M,
        lift_offset_m=float(table_top_m - SOURCE_SUPPORT_TOP_M),
        stationary_joint_targets=fixed, arm_targets_rad=dict(source_initial['arm_targets_rad']),
        base_z_m=0.0, robot_station=None, plate_in_hand=None, payload_in_plate=None,
        gripper_measured_reset_position=None,
        physics_validated=False, runtime_reset_connected=False,
        required_before_execution=['bind a clear dining region and a physically valid robot station',
            'calibrate the current Z-up v12 plate against physical finger pads',
            'calibrate box-on-plate support using current beef asset',
            'validate actual initial aperture and sustained bilateral grasp',
            'validate release tail and full dining-table physical replay'],
        unchanged=['base ground height', 'recorded arm angles', 'assets and collision policy',
                   'drive gains, force limits, mass and friction', '10.1 and common API'])
