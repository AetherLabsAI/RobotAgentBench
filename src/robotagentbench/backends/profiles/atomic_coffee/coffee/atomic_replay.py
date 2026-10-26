"""Unscored source-action trigger test using the unchanged joint20 executor.

Only the user-approved source opening correction occurs in replay_actions.
Reuse the original producer's 20 active drives, with public plan_interval and
direct_gripper_substep validation. This OLD source asset has passive inner
fingers, not the newer NativeAdapter's explicit Mimic contract. Do not install
drives or pretend this diagnostic accepts that newer complete API contract.
No action pause, object pose command or trigger spoofing.
"""
import gzip
import json
import subprocess
import numpy as np

DIAGNOSTIC_RESOLUTION = (640, 480)
HD_PRESENTATION_RESOLUTION = (1440, 1080)


def capture_resolution(hd_presentation=False):
    """User-requested HD delivery only; same 4:3 camera and control clock."""
    if type(hd_presentation) is not bool:
        raise ValueError('Explicit presentation flag required')
    return HD_PRESENTATION_RESOLUTION if hd_presentation else DIAGNOSTIC_RESOLUTION


class StreamingVideo:
    """Native 4:3/20fps capture; no resizing or duplicated raw PNG sequence."""
    def __init__(self, path, resolution=DIAGNOSTIC_RESOLUTION):
        if resolution not in (DIAGNOSTIC_RESOLUTION, HD_PRESENTATION_RESOLUTION):
            raise ValueError('Only diagnostic or requested native HD presentation supported')
        self.resolution = resolution
        width, height = resolution
        quality = ['-crf', '18'] if resolution == HD_PRESENTATION_RESOLUTION else []
        self.process = subprocess.Popen(['ffmpeg', '-nostdin', '-v', 'error', '-f', 'rawvideo',
            '-pix_fmt', 'rgb24', '-s', f'{width}x{height}', '-r', '20', '-i', '-', '-an',
            '-c:v', 'libx264', *quality, '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path)], stdin=subprocess.PIPE)
        self.frames = 0
        self.closed = False

    def write(self, image):
        width, height = self.resolution
        if image.shape != (height, width, 3) or image.dtype != np.uint8:
            raise ValueError('Native RGB shape/dtype must match encoder; no upscaling')
        self.process.stdin.write(np.ascontiguousarray(image).tobytes())
        self.frames += 1

    def close(self):
        if not self.closed:
            self.closed = True
            self.process.stdin.close()
            if self.process.wait() != 0:
                raise RuntimeError('Streaming preview encoder failed')


def compress_reader_output(reader):
    # Storage format only; every original substep fact is still recorded.
    reader.trace.close()
    reader.trace = gzip.open(reader.output / 'physical_facts.jsonl.gz', 'xt')


def validate_initial(reader):
    """Original 1-2 initial acceptance thresholds, with extra dynamic support."""
    from evaluation.tasks.coffee_machine.specs import CUP, BAG
    from evaluation.tasks.coffee_machine.api_runtime import INITIAL_TILT_DEG, INITIAL_BAG_TILT_DEG, INITIAL_CUP_XY_M
    rows = list(reader.history)
    now = reader.latest
    placement = reader.placement(np.asarray(now['poses'][CUP]['position_m']), now['poses'][CUP]['quaternion_wxyz'])
    span = lambda obj: float(np.ptp([r['poses'][obj]['position_m'] for r in rows], axis=0).max())
    checks = dict(history_complete=len(rows) == reader.history.maxlen,
        objects_at_rest=max(span(CUP), span(BAG)) <= .002,
        robot_touches_nothing=not any(r['contacts'][o]['robot'] for r in rows for o in (CUP, BAG))
            and not any(r['robot_machine'] or r['robot_on_button'] for r in rows),
        wrist_cameras_clear=not any(r['camera_contacts'] for r in rows),
        nothing_on_ground=not any(r['contacts'][o]['ground'] for r in rows for o in (CUP, BAG)),
        cup_upright=now['tilt_deg'][CUP] <= INITIAL_TILT_DEG,
        bag_upright=now['tilt_deg'][BAG] <= INITIAL_BAG_TILT_DEG,
        bag_on_counter='counter' in now['contacts'][BAG]['supports'],
        cup_on_tray='tray' in now['contacts'][CUP]['supports'],
        cup_on_placement=placement.xy_error_m <= INITIAL_CUP_XY_M and placement.outlet_over_opening,
        beans_in_bag=reader.beans_in_bag() == 100,
        hopper_empty=now['device']['bean_count'] == 0)
    if not all(checks.values()):
        raise RuntimeError('Atomic source initial rejected: ' + repr(checks))
    return dict(accepted=True, checks=checks, original_thresholds=True, time_s=now['time_s'])


def run(*, stage, robot, reader, session, native_robot, initial, candidate,
        advance, capture, physical_row, checkpoint, report, trace, save):
    from pxr import UsdPhysics
    from isaacsim.core.utils.types import ArticulationAction
    from arena_api import robot_profiles as rp, policy_execution as pe
    from arena_api.bimanual import CONTROL_DT
    profile = rp.get(rp.JOINT20)
    names = robot.dof_names
    units, limits, driven = {}, {}, []
    for name in names:
        prim = stage.GetPrimAtPath('/World/Robot/joints/' + name)
        if prim.IsA(UsdPhysics.RevoluteJoint):
            units[name] = 'rad'
        elif prim.IsA(UsdPhysics.PrismaticJoint):
            units[name] = 'm'
        else:
            raise RuntimeError('Unmapped source joint ' + name)
        if any(s.startswith('PhysicsDriveAPI:') for s in prim.GetAppliedSchemas()):
            driven.append(name)
        lo, hi = (prim.GetAttribute('physics:' + k).Get() for k in ('lowerLimit', 'upperLimit'))
        if lo is not None and hi is not None:
            limits[name] = [float(np.radians(v)) if units[name] == 'rad' else float(v) for v in (lo, hi)]
    def measured():
        values = dict(zip(names, robot.get_joint_positions().tolist()))
        return {n: values[n] for n in profile.joints}
    previous, action_limits = pe.build_limits(measured(), limits, profile.id)
    defaults = dict(zip(names, robot.get_joint_positions().tolist()))
    defaults.update(initial['robot_initialization']['hold_targets'])
    if set(driven) != set(profile.joints) or set(driven) != set(initial['robot_initialization']['hold_targets']):
        raise RuntimeError('Source producer executor requires exactly its original 20 driven joints')
    indices = np.array([names.index(n) for n in driven])
    # Validate ALL adapted targets before issuing any action.
    for action in candidate['actions']:
        pe.validate(action, action_limits, profile.id)
    reader.begin_episode()
    session.arm()
    report.update(status='atomic_source_replay', original_unmodified_replay=False,
                  production_trigger_bypassed_for_unscored_mechanism_test=False,
                  robot_source_reset_hold_only=False, task_trigger_exercised=True,
                  source_action_rows=len(candidate['actions']), changed_open_commands=candidate['changed_commands_count'])
    report.update(action_executor='original_atomic_producer_20_active_drives',
                  unified_native_adapter_verified=False, passive_finger_mechanism_unchanged=True)
    checkpoint()
    capture()
    reasons = {}
    eligible_count = 0
    first_eligible = first_start = None
    events = []
    baseline = reader.latest['poses']['/World/Task75/Machine/Body']['position_m']
    for index, action in enumerate(candidate['actions']):
        plan = pe.plan_interval(action, measured(), previous, action_limits, profile.id)
        for interpolated in plan['interpolated_substep_targets_rad']:
            command, changes = pe.direct_gripper_substep(action, interpolated, measured(), action_limits, profile.id)
            if changes or command != action:
                raise RuntimeError('Native direct-target execution rewrote an adapted command')
            robot.apply_action(ArticulationAction(joint_positions=np.array([command[n] for n in driven]),
                                                 joint_indices=indices))
            session.command_before_step(float(reader.world.current_time))
            advance()
            poses = {'/World/Robot/' + n: m for n, m in zip(native_robot.shared_metatype.link_names,
                _robot_matrices(native_robot.get_link_transforms()[0]))}
            facts = reader.trigger_facts()
            eligible = facts.holding_bag and facts.mouth_above_opening and facts.bag_tilt_deg >= 45.
            eligible_count += int(eligible)
            if eligible and first_eligible is None:
                first_eligible = dict(time_s=facts.time_s, action_step=index, tilt_deg=facts.bag_tilt_deg,
                                      holding_bag=facts.holding_bag, mouth_above_opening=facts.mouth_above_opening)
            event = session.observe(robot_native_poses=poses)
            reason = event['reason']
            reasons[reason] = reasons.get(reason, 0) + 1
            if event.get('start'):
                if first_start is not None:
                    raise RuntimeError('More than one physical trigger fired')
                first_start = dict(event, action_step=index, current_facts=vars(facts))
                events.append(first_start)
            if event.get('admission', {}).get('reason') not in (None, 'not_yet_eligible'):
                if not events or events[-1].get('admission') != event['admission']:
                    events.append(dict(event, action_step=index))
            previous = dict(command)
        row = physical_row('source_' + str(candidate['phases'][index]))
        row.update(action_step=index, trigger_facts=vars(facts), trigger_reason=reason)
        trace.append(row)
        reader.frame()  # original control-frame evidence; no scorer success injection
        capture()
        if index % 20 == 0:
            report.update(control_step=index, current_source_phase=candidate['phases'][index],
                          eligible_substeps=eligible_count, trigger_count=int(first_start is not None),
                          trigger_reason_counts=reasons)
            checkpoint()
    final = reader.latest['poses']['/World/Task75/Machine/Body']['position_m']
    save('trigger_events.json', dict(events=events, actor_events=session.events,
        first_eligible=first_eligible, first_start=first_start, reason_counts=reasons))
    report.update(status='atomic_source_replay_completed', source_rows_executed=len(candidate['actions']),
                  first_eligible=first_eligible, first_trigger=first_start,
                  trigger_count=int(first_start is not None), eligible_substeps=eligible_count,
                  trigger_reason_counts=reasons, human_retired=session.retired,
                  measured_machine_horizontal_displacement_m=float(np.linalg.norm(np.array(final[:2]) - baseline[:2])),
                  palm_contact_observed=any(r['palm_machine_contacts'] for r in trace),
                  robot_human_contact_observed=any(r['robot_human_contacts'] for r in trace),
                  final=trace[-1], full_task_verified=False, scored_episode=False)


def _robot_matrices(rows):
    from short_task_robustness.tasks.stock_wooden_tray_interference.human_asset import pose_matrix
    return [pose_matrix(row[:3], row[3:]).T for row in rows]
