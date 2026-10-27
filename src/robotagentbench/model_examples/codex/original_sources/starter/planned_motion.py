"""Candidate v6: preflight a rate-feasible Cartesian path, then ordered playout.

Inspired by Inspect Robots pre_check and YAM joint-space CollisionApprover.
G1 adaptation: subdivide Cartesian progress instead of clipping each IK joint.
Uses only robot model + observations. Not a trusted safety boundary or a global
planner; a rejected straight path requires a different user/policy waypoint.
"""
from dataclasses import dataclass
from collections.abc import Mapping
import math
import numpy as np
from move_to import (TrackingMoveToController, MoveResult, LABELS, vector,
                     pose_values, values_pose, errors)


@dataclass
class PlannedResult(MoveResult):
    planned_steps: int = 0
    subdivisions: int = 0
    collision: dict | None = None
    requested_step_budget: int = 0
    step_budget: int = 0
    budget_exhausted: bool = False
    remaining_position_m: float = 0.0
    remaining_angle_rad: float = 0.0
    next_action: str = 'inspect_feedback'
    guidance: str = ''


def plan_path(ik, config, seed, start, goal, reference, labels, *, guard):
    """No motion side effects. Every accepted full IK solution is independently checked."""
    limits = config.validate(ik)
    speeds = np.array([config.linear_speed]*3 + [config.angular_speed]*3 +
                      ([config.gripper_speed] if len(labels) == 7 else []))
    ranges = np.array([config.eef_bounds[n][1]-config.eef_bounds[n][0] for n in LABELS] +
                      ([1.] if len(labels) == 7 else []))
    per_step = np.minimum(speeds/config.control_hz, ranges*.05)
    distance = np.abs(goal-start)
    if np.any((per_step == 0) & (distance > 1e-12)):
        raise ValueError('motion along a fixed dimension')
    ratio = float(np.max(np.divide(distance, per_step, out=np.zeros_like(distance), where=per_step>0)))
    increment = 1./max(1., ratio/(1.-1e-6))
    # Editable per-motion budget; mirrors the broker's published per-tool ceiling.
    # No hidden ten-second cap. The default live configuration remains 200.
    budget = min(config.max_steps, 600)
    q = seed.copy(); fraction = 0.; rows = []; subdivisions = 0
    arm_static = np.max(distance[:6]) < 1e-12
    hit = guard.check(dict(zip(ik.names, q)))
    if hit:
        return [], 'initial_self_clearance', subdivisions, hit
    for _ in range(budget):
        jump = min(increment, 1.-fraction)
        failure = 'ik_no_solution'
        for retry in range(13):
            proposed = min(1., fraction+jump)
            waypoint = start+(goal-start)*proposed
            desired = values_pose(waypoint[:6], reference)
            if arm_static:
                candidate = q.copy()
            else:
                answer = ik.reference_solve(desired, q)
                candidate = np.asarray(answer.q, dtype=float)
                if candidate.shape != seed.shape:
                    raise ValueError('invalid IK joint shape')
                if answer.reason in ('qp_no_solution', 'nonfinite_solution') or not np.isfinite(candidate).all():
                    failure = 'ik_no_finite_solution'; candidate = None
            if candidate is not None:
                if not np.allclose(candidate[ik.frozen], seed[ik.frozen], atol=1e-8, rtol=0):
                    raise ValueError('IK changed a non-commanded joint')
                if np.any(candidate[ik.indices] < ik.low[ik.indices]) or np.any(candidate[ik.indices] > ik.high[ik.indices]):
                    failure = 'ik_joint_bounds'
                elif np.any(np.abs(candidate[ik.indices]-q[ik.indices]) > limits+1e-12):
                    failure = 'ik_joint_speed'
                else:
                    pe, ae = errors(ik.fk(candidate), desired)
                    if pe > min(config.position_tolerance, ik.position_tolerance) or ae > min(config.angle_tolerance, ik.angle_tolerance):
                        failure = 'ik_residual'
                    else:
                        break
            candidate = None
            if retry < 12:
                jump *= .5; subdivisions += 1
        if candidate is None:
            return [], failure, subdivisions, None  # reject the entire unsent chunk
        hit = guard.sweep(q, candidate)
        if hit:
            return [], 'predicted_self_collision', subdivisions, hit
        rows.append((candidate.copy(), waypoint.copy(), proposed))
        q = candidate.copy(); fraction = proposed
        if fraction >= 1.-1e-12:
            return rows, 'planned_goal', subdivisions, None
    return rows, 'planned_partial_chunk', subdivisions, None


class MoveToController(TrackingMoveToController):
    def __init__(self, port, ik, *, config, guard=None):
        super().__init__(port, ik, config=config)
        if guard is None:
            from self_collision import PrimitiveSelfGuard
            guard = PrimitiveSelfGuard()
        if hasattr(guard, 'names') and tuple(guard.names) != tuple(ik.names):
            raise ValueError('collision model joint order differs from IK model')
        self.guard = guard

    def move_to(self, targets, *, mode='replace'):
        if mode not in ('replace', 'continue'):
            raise ValueError('mode must be replace or continue')
        ik, cfg = self.ik, self.config
        cfg.validate(ik)
        latest = self.port.observe(); measured = ik.read_q(latest)
        self._start(latest, measured)
        actual_start = pose_values(ik.fk(measured), self.reference)
        labels = LABELS
        if latest.gripper is not None:
            if not np.isfinite(latest.gripper) or not 0 <= latest.gripper <= 1:
                raise ValueError('invalid measured gripper aperture')
            labels += ('gripper',); actual_start = np.append(actual_start, latest.gripper)
        updates = dict(targets) if isinstance(targets, Mapping) else dict(zip(LABELS, pose_values(targets, self.reference)))
        if set(updates)-set(labels):
            raise ValueError('unknown target dimension or gripper observation unavailable')
        target = actual_start.copy()
        low = np.array([cfg.eef_bounds[n][0] for n in LABELS]+([0.] if len(labels)==7 else []))
        high = np.array([cfg.eef_bounds[n][1] for n in LABELS]+([1.] if len(labels)==7 else []))
        for name, value in updates.items():
            i = labels.index(name); target[i] = vector([value], (1,), name)[0]
            if not low[i] <= target[i] <= high[i]:
                raise ValueError('target outside published Cartesian bounds: '+name)
        if mode == 'continue':
            if self.goal is None or self.goal.shape != target.shape:
                raise ValueError('no previous goal in this episode; use replace')
            if any(abs(target[labels.index(n)]-self.goal[labels.index(n)]) > 1e-12 for n in updates):
                raise ValueError('continue cannot change the goal; use replace')
            target, updates = self.goal.copy(), self.goal_updates.copy()
        if np.any(target < low) or np.any(target > high):
            raise ValueError('goal outside current published bounds')
        grip = None
        if 'gripper' in updates:
            grip = float(latest.gripper if self.previous_gripper is None else self.previous_gripper)
        seed = measured.copy(); seed[ik.indices] = self.previous
        start = pose_values(ik.fk(seed), self.reference)
        if len(labels) == 7:
            start = np.append(start, latest.gripper if grip is None else grip)
            if grip is None: target[6] = start[6]
        self.goal, self.goal_updates = target.copy(), updates.copy()
        desired = values_pose(target[:6], self.reference)
        executed = 0; progress = 0.; rows = []; subdivisions = 0; hit = None

        def result(reason, rejected=False):
            actual = ik.fk(measured).copy(); pe, ae = errors(actual, desired)
            command = measured.copy(); command[ik.indices] = self.previous
            cp, ca = errors(actual, ik.fk(command))
            ge = None if grip is None else abs(latest.gripper-target[6])
            arm_reached = pe <= cfg.position_tolerance and ae <= cfg.angle_tolerance
            complete = grip is None or abs(grip-target[6]) <= 1e-12
            finished = arm_reached and complete and not rejected and reason == 'command_complete_not_reached'
            exhausted = reason == 'step_budget_exhausted'
            guidance = (
                f'已达到本次 {min(cfg.max_steps,600)} 步上限；距最终目标 {pe*1000:.2f} mm、'
                f'{math.degrees(ae):.2f}°，与已下发目标的位置差 {cp*1000:.2f} mm。'
                '这次未因跟踪超限停止；可用 continue 继续同一目标，但后续路径仍须重新检查。'
                if exhausted else
                '命令已发完，但实际尚未满足到位精度；这不是步数上限。请检查反馈再决定是否继续。'
                if reason == 'command_complete_not_reached' and not finished else
                '运动执行完成，不代表抓取或任务成功。' if finished else
                '请依据拒绝或跟踪停止原因检查路径；增大步数预算不能解决碰撞或跟踪超限。')
            # Execution completion is not a grasp or task-success assertion.
            return PlannedResult(status='rejected' if rejected else 'finish' if finished else 'partial',
                reason='motion_execution_finished' if finished else reason, steps_executed=executed,
                position_error_m=pe, angle_error_rad=ae, actual_pose=actual, gripper_error=ge,
                progress_fraction=progress, arm_reached=bool(arm_reached),
                gripper_target_reached=None if ge is None else bool(ge<=cfg.gripper_tolerance),
                execution_mode=mode, command_position_error_m=cp, command_angle_error_rad=ca,
                max_joint_tracking_error=float(np.max(np.abs(self.previous-measured[ik.indices]))),
                gripper_command=grip, gripper_command_complete=None if grip is None else bool(complete),
                planned_steps=len(rows), subdivisions=subdivisions, collision=hit,
                requested_step_budget=cfg.max_steps, step_budget=min(cfg.max_steps,600),
                budget_exhausted=exhausted, remaining_position_m=pe, remaining_angle_rad=ae,
                next_action='continue_same_goal_with_recheck' if exhausted else 'inspect_feedback',
                guidance=guidance)

        tracking = cfg.joint_values(cfg.resync_thresholds, ik, 'resync_thresholds')
        if np.any(np.abs(self.previous-measured[ik.indices]) > tracking):
            return result('tracking_error_before_plan', True)
        rows, reason, subdivisions, hit = plan_path(ik, cfg, seed, start, target, self.reference, labels, guard=self.guard)
        if not rows: return result(reason, True)
        for candidate, waypoint, fraction in rows:
            if np.any(np.abs(self.previous-measured[ik.indices]) > tracking):
                return result('tracking_error_during_playout')
            # Recheck using current observation, including the non-commanded arm.
            current_candidate = measured.copy(); current_candidate[ik.indices] = candidate[ik.indices]
            hit = self.guard.sweep(measured, current_candidate)
            if hit: return result('observed_self_clearance', executed == 0)
            kwargs = dict(episode_id=latest.episode_id, expected_step=latest.step)
            next_grip = None if grip is None else float(waypoint[6])
            if next_grip is not None: kwargs['gripper'] = next_grip
            updated = self.port.command_joints(dict(zip(ik.arm_joints,map(float,candidate[ik.indices]))), **kwargs)
            if updated.episode_id != latest.episode_id or updated.step != latest.step+1:
                raise RuntimeError('invalid bridge response: episode/step changed')
            self.previous = candidate[ik.indices].copy()
            if next_grip is not None: grip = self.previous_gripper = next_grip
            latest = updated; self.last_step = updated.step; measured = ik.read_q(updated)
            executed += 1; progress = fraction
        if np.any(np.abs(self.previous-measured[ik.indices]) > tracking):
            return result('tracking_error_during_playout')
        return result('step_budget_exhausted' if reason == 'planned_partial_chunk'
                      else 'command_complete_not_reached')


def hold_checked(self, steps=60):
    """Hold existing native targets, with fresh feedback and unchanged safety limits."""
    if not isinstance(steps, int) or not 1 <= steps <= 600:
        raise ValueError('hold steps must be 1..600')
    ik, cfg = self.ik, self.config
    cfg.validate(ik)
    latest = self.port.observe()
    measured = ik.read_q(latest)
    self._start(latest, measured)
    thresholds = cfg.joint_values(cfg.resync_thresholds, ik, 'resync_thresholds')
    initial_pose = ik.fk(measured).copy()
    for i in range(steps):
        if np.any(np.abs(self.previous-measured[ik.indices]) > thresholds):
            return dict(reason='hold_tracking_stop', steps_executed=i)
        command = measured.copy()
        command[ik.indices] = self.previous
        if np.any(command[ik.indices] < ik.low[ik.indices]) or np.any(command[ik.indices] > ik.high[ik.indices]):
            return dict(reason='hold_joint_bounds', steps_executed=i)
        hit = self.guard.sweep(measured, command)
        if hit:
            return dict(reason='hold_self_clearance', steps_executed=i, collision=hit)
        # No candidate movement: commanded arm joints and gripper stay unchanged.
        kwargs = dict(episode_id=latest.episode_id, expected_step=latest.step)
        if self.previous_gripper is not None:
            kwargs['gripper'] = self.previous_gripper
        updated = self.port.command_joints(dict(zip(ik.arm_joints, map(float,self.previous))), **kwargs)
        if updated.episode_id != latest.episode_id or updated.step != latest.step+1:
            raise RuntimeError('invalid hold bridge response')
        latest = updated
        self.last_step = latest.step
        measured = ik.read_q(latest)
    command = measured.copy(); command[ik.indices] = self.previous
    pe, ae = errors(ik.fk(measured), ik.fk(command))
    drift, angle_drift = errors(ik.fk(measured), initial_pose)
    stopped = np.any(np.abs(self.previous-measured[ik.indices]) > thresholds)
    return dict(reason='hold_tracking_stop' if stopped else 'hold_completed', steps_executed=steps,
                command_position_error_m=pe, command_angle_error_rad=ae,
                measured_drift_m=drift, measured_angle_drift_rad=angle_drift,
                gripper_measured=latest.gripper, gripper_command=self.previous_gripper)

MoveToController.hold_checked = hold_checked


def withdraw_to_measured(self, dry_run=True):
    """Withdraw lagging target, never reset command history or widen thresholds."""
    ik, cfg = self.ik, self.config
    limits = cfg.validate(ik)
    latest = self.port.observe(); measured = ik.read_q(latest)
    self._start(latest, measured)
    threshold = cfg.joint_values(cfg.resync_thresholds, ik, 'resync_thresholds')
    anchor = measured[ik.indices].copy()
    previous = self.previous.copy()
    count = max(1, int(np.ceil(np.max(np.abs(anchor-previous)/limits))))
    rows = [previous+(anchor-previous)*(i/count) for i in range(1,count+1)]
    q = measured.copy(); q[ik.indices] = previous
    for row in rows:
        candidate = measured.copy(); candidate[ik.indices] = row
        if np.any(np.abs(row-q[ik.indices]) > limits+1e-12): return {'reason':'withdraw_speed_rejected'}
        if np.any(np.abs(row-measured[ik.indices]) > threshold): return {'reason':'withdraw_tracking_rejected'}
        if np.any(row<ik.low[ik.indices]) or np.any(row>ik.high[ik.indices]): return {'reason':'withdraw_bounds_rejected'}
        hit = self.guard.sweep(q,candidate)
        if hit: return {'reason':'withdraw_collision_rejected','collision':hit}
        q = candidate
    if dry_run: return {'reason':'withdraw_preflight_ok','steps':count,'withdraw_distance_m':float(np.linalg.norm(ik.fk(measured)[:3,3]-ik.fk(np.array([previous[list(ik.indices).index(i)] if i in ik.indices else measured[i] for i in range(len(measured))]))[:3,3]))}
    for row in rows:
        measured = ik.read_q(latest)
        candidate = measured.copy(); candidate[ik.indices] = row
        if np.any(np.abs(row-measured[ik.indices])>threshold): return {'reason':'withdraw_tracking_stop'}
        hit=self.guard.sweep(measured,candidate)
        if hit:return {'reason':'withdraw_collision_stop','collision':hit}
        kwargs=dict(episode_id=latest.episode_id,expected_step=latest.step)
        if self.previous_gripper is not None:kwargs['gripper']=self.previous_gripper
        updated=self.port.command_joints(dict(zip(ik.arm_joints,map(float,row))),**kwargs)
        if updated.episode_id!=latest.episode_id or updated.step!=latest.step+1:raise RuntimeError('bridge freshness')
        self.previous=row.copy(); self.last_step=updated.step; latest=updated
    self.goal=None; self.goal_updates=None
    return {'reason':'withdraw_completed','steps':count}

MoveToController.withdraw_to_measured = withdraw_to_measured
