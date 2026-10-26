"""Continuous Sheet 8.10 canvas control with physical initial perturbations.

Robot control, graph batching and read-only continuity observation are separate
layers. Material and contact gates stay fixed for each rollout. No real episode,
snapshot branch, or untracked experiment script is needed to run this module.
"""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import warp as wp
from tablecloth_canvas_actions import CanvasActions
from tablecloth_rule_sim import smooth
from tablecloth_capture_quality import CollisionGuard
from real2sim.production.task_randomization.benchmark_8_10.inject_robot_initial_state import apply_initial_perturbation


class HumanCapture(CanvasActions):

    def frame(self, phase, targets, grips):
        if phase not in {'L_enter_tip', 'L_close_tip'}:
            return super().frame(phase, targets, grips)
        cfg = self.policy['left_grasp']
        previous = self.hands.values.copy()
        orientation = Rotation.from_quat(cfg['quaternion_xyzw']).as_matrix() if cfg.get('fixed_wrist_orientation') else None
        self.hands.command(targets, grips, {**self.orientations, 'L': orientation})
        names = self.robot.arm_names('L')
        if phase == 'L_enter_tip':
            vmax = cfg.get('joint_speed_rad_s', 0.55)
            amax = cfg.get('joint_accel_rad_s2', 1.2)
            prevv = getattr(self, 'left_velocity', np.zeros(len(names)))
            desired = np.array([(self.hands.values[n] - previous[n]) / cfg.get('response_seconds', 0.12) for n in names])
            desired *= min(1.0, vmax / max(np.max(abs(desired)), 1e-12))
            velocity = prevv + np.clip(desired - prevv, -amax * self.dt, amax * self.dt)
            self.left_velocity = velocity
        else:
            velocity = np.array([(self.hands.values[n] - previous[n]) / self.dt for n in names])
            velocity *= min(1.0, self.args.max_joint_speed / max(np.max(abs(velocity)), 1e-12))
        for n, v in zip(names, velocity):
            self.hands.values[n] = previous[n] + self.dt * v
        self.hands.apply_grips()
        actual = self.hands.centres()
        self.advance(phase, actual, {'L': 0.0, 'R': 0.0}, targets, True)
        self.capture_servo_samples.append(dict(frame_index=len(self.frames) - 1, phase=phase, planner_goal_error_m=float(np.linalg.norm(actual['L'] - targets['L'])), maximum_joint_speed_rad_s=float(np.max(abs(velocity)))))

    def tip_ready(self, corner):
        point = self.a.particle_q.numpy()[corner]
        distances, vectors = self.hands.pad_surfaces(point[None], 'L')
        cosine = float(np.dot(vectors[0, 0], vectors[0, 1]) / max(np.linalg.norm(vectors[0, 0]) * np.linalg.norm(vectors[0, 1]), 1e-12))
        ready = bool((distances[0] >= -0.0005).all() and cosine < self.policy['left_grasp'].get('alignment_cosine_max', -0.5) and (distances[0].sum() < 0.035))
        if ready:
            self.events.append(dict(time=self.t, event='left_tip_aligned', particle=corner, opposing_surface_distances_m=distances[0].tolist(), opposing_contact_cosine=cosine, role='closure_planner_gate_only; actual_physical_pinch_gate_unchanged'))
        return ready

    def capture_left_tip(self, corner):
        cfg = self.policy['left_grasp']
        if cfg.get('settle_before_approach_seconds', 0.0):
            self.move('let_corner_settle', cfg['settle_before_approach_seconds'])
        print('LEFT_APPROACH_INITIAL', json.dumps(dict(time=self.t, corner=self.a.particle_q.numpy()[corner].tolist(), velocity=self.a.particle_qd.numpy()[corner].tolist())), flush=True)
        start = self.hands.centres()['L'].copy()
        n = max(1, round(cfg.get('approach_seconds', 3.0) * self.args.fps))
        self.left_velocity = np.zeros(7)
        ready = False
        for k in range(n):
            u = smooth((k + 1) / n)
            goals = self.hands.centres()
            goal = self.left_goal(corner)
            stand_off = np.asarray(cfg.get('approach_standoff_m', [0.0, 0.0, 0.0]))
            goals['L'] = start * (1 - u) + goal * u + stand_off * np.sin(np.pi * u)
            self.frame('L_enter_tip', goals, self.hands.grip.copy())
            if k * self.dt >= 1.0 and self.tip_ready(corner):
                ready = True
                break
        for _ in range(round(4.0 * self.args.fps)):
            if ready or self.tip_ready(corner):
                break
            goals = self.hands.centres()
            goals['L'] = self.left_goal(corner)
            self.frame('L_enter_tip', goals, self.hands.grip.copy())
        else:
            raise RuntimeError('Natural left approach missed the measured jaw alignment gate')
        self.pending_grasps['L'] = corner
        g0 = self.hands.grip.copy()
        for k in range(max(3, round(cfg.get('closure_attempt_window_seconds', 0.25) * self.args.fps))):
            u = smooth(min((k + 1) / (cfg.get('close_seconds', 0.45) * self.args.fps), 1.0))
            grips = {**g0, 'L': g0['L'] + (0.965 - g0['L']) * u}
            goals = self.hands.centres()
            goals['L'] = self.left_goal(corner, grip=grips['L'])
            self.frame('L_close_tip', goals, grips)
            if 'L' in self.clamps:
                break
        self.pending_grasps.pop('L', None)
        if 'L' not in self.clamps:
            raise RuntimeError('Natural left closure missed the physical contact gate')

    def save(self, failure=None):
        super().save(failure)
        p = self.output / 'trial.json'
        if p.exists():
            m = json.loads(p.read_text())
            m.update(owner_action_entrypoint=str(Path(__file__).resolve()), dynamic_capture_control='Minimum-jerk moving-corner reach with exterior arc and left joint velocity/acceleration limits; faster measured-contact closure; exact values in policy', trial_role='continuous_canvas_spread_candidate', completed_benchmark_trajectory=False)
            p.write_text(json.dumps(m, indent=2) + '\n')

class PerturbedCapture(HumanCapture):

    def __init__(self, args):
        super().__init__(args)
        self.injection = apply_initial_perturbation(self, args.injection_data)
        self.initial_nominal_joint = np.asarray(self.injection['nominal_initial_joint_position'])
        self.nominal_base = np.asarray(self.injection['nominal_base_transform'])
        self.initial_joint = np.asarray(self.injection['actual_initial_joint_position'])
        self.initial_body = self.body_transforms().copy()
        self.base_xyyaw = np.asarray(self.injection['actual_base_xyyaw_m_deg'])
        (self.output / 'initial_perturbation.json').write_text(json.dumps(self.injection, indent=2) + '\n')

    def close_right_tip(self, corner):
        if self.policy.get('right_grasp', {}).get('legacy_close', False):
            return super().close_right_tip(corner)
        self.pending_grasps['R'] = corner
        g0 = self.hands.grip.copy()
        cfg = self.policy.get('right_grasp', {})
        n = max(1, round(cfg.get('close_seconds', 0.8) * self.args.fps))
        for k in range(n):
            u = smooth((k + 1) / n)
            grips = {**g0, 'R': g0['R'] + (0.965 - g0['R']) * u}
            values = self.hands.values.copy()
            oldgrip = self.hands.grip.copy()
            self.hands.grip.update(grips)
            self.hands.apply_grips()
            bite = self.hands.shallow_grasp_offset('R', self.args.grasp_depth)
            self.hands.values = values
            self.hands.grip = oldgrip
            goals = self.hands.centres()
            goals['R'] = self.a.particle_q.numpy()[corner] + bite
            self.frame('R_close_tip', goals, grips)
            if 'R' in self.clamps:
                break
        self.pending_grasps.pop('R', None)
        if 'R' not in self.clamps:
            raise RuntimeError('Right closure missed unchanged physical pad gate')
        self.move('R_finish_close', 0.2, grips={'R': 0.96})

    def save(self, failure=None):
        super().save(failure)
        if not self.frames:
            return
        for filename in ['full_state.npz', 'action.npz']:
            path = self.output / filename
            data = dict(np.load(path))
            data['base_xyyaw_m_deg'] = np.tile(self.base_xyyaw, (len(self.frames), 1))
            data['initial_joint_position'] = self.initial_joint if filename == 'full_state.npz' else np.asarray(json.dumps(self.initial_joint.tolist()))
            data['initial_base_xyyaw_m_deg'] = self.base_xyyaw if filename == 'full_state.npz' else np.asarray(json.dumps(self.base_xyyaw.tolist()))
            if filename == 'full_state.npz':
                data['initial_body_q'] = self.initial_body
            np.savez_compressed(path, **data)
        p = self.output / 'trial.json'
        m = json.loads(p.read_text())
        m.update(initial_perturbation=self.injection, owner_action_entrypoint=str(Path(__file__).resolve()),  diagnostic_only=bool(self.args.pickup_only), target_identity='clean_tablecloth')
        p.write_text(json.dumps(m, indent=2) + '\n')

@wp.kernel
def set_alpha(alpha: wp.array(dtype=wp.float32), value: float):
    alpha[0] = value

class FastCapture(PerturbedCapture):

    def safe_constant_clamps(self, start, stop):
        if self.pending_grasps:
            return False
        for side in self.clamps:
            times = np.arange(0, self.args.substeps + 1, 2) / self.args.substeps
            points = []
            rotations = []
            for half in ['inner', 'outer']:
                link = f'gripper_{side.lower()}_{half}_link4'
                i = self.body_ids[link]
                a, b = (start[i], stop[i])
                points.append(a[:3] * (1 - times[:, None]) + b[:3] * times[:, None])
                rotations.append(Slerp([0, 1], Rotation.from_quat([a[3:], b[3:]]))(times).as_matrix())
            delta = points[0] - points[1]
            axis = delta / np.linalg.norm(delta, axis=1)[:, None]
            extents = []
            for half, rot in zip(['inner', 'outer'], rotations):
                vertices = self.hands.calibration[f'points_gripper_{side.lower()}_{half}_link4']
                localaxis = np.einsum('nij,ni->nj', rot, axis)
                projection = vertices @ localaxis.T
                extents.append(projection.min(0) if half == 'inner' else projection.max(0))
            gap = np.einsum('ni,ni->n', delta, axis) + extents[0] - extents[1]
            if np.min(gap) < 5e-05 or np.max(gap) > 0.0039:
                return False
        return True

    def save(self, failure=None):
        super().save(failure)
        p = self.output / 'trial.json'
        if p.exists():
            m = json.loads(p.read_text())
            m.update(owner_action_entrypoint=str(Path(__file__).resolve()), graph_batching=dict(fast_frames=getattr(self, 'fast_frames', 0), canonical_callback_frames=getattr(self, 'slow_frames', 0), same_two_substep_solver_graph=True, callback_release_gap_certification='every original boundary; revert near threshold', physical_parameters_changed=False))
            p.write_text(json.dumps(m, indent=2) + '\n')

class GuardedCapture(FastCapture):

    def __init__(self, args):
        super().__init__(args)

        names = list(self.body_ids)
        for item in self.robot_body_shapes:
            if item['link'] not in names:
                names.append(item['link'])
        self.quality_guard = CollisionGuard(names, self.geometry, self.robot.metadata, self.hands.calibration)

    def advance(self, phase, *args, **kwargs):
        super().advance(phase, *args, **kwargs)
        f = self.frames[-1]
        q = f['q']
        ratio = np.linalg.norm(q[self.edges[:, 0]] - q[self.edges[:, 1]], axis=1) / self.rest_lengths
        problem = self.quality_guard.examine(f['body'], q)
        if problem is None and (ratio.max() > 1.1 or np.quantile(ratio, 0.99) > 1.04):
            problem = dict(kind='material_strain', maximum=float(ratio.max()), p99=float(np.quantile(ratio, 0.99)))
        speed = float(np.linalg.norm(f['qd'], axis=1).max())
        if problem is None and speed > 5.0:
            problem = dict(kind='particle_velocity', maximum_m_s=speed)
        if problem is not None:
            self.events.append(dict(event='online_quality_abort', frame_index=len(self.frames) - 1, time=self.t, phase=phase, **problem))
            raise RuntimeError('Physical quality rejection: ' + json.dumps(problem))

    def move(self, phase, *args, **kwargs):
        if phase != 'open_clear_hands':
            return super().move(phase, *args, **kwargs)
        start = self.hands.values.copy()
        grips = self.hands.grip.copy()
        target = dict(zip(self.robot.names, self.initial_joint))
        names = self.robot.arm_names('L') + self.robot.arm_names('R')
        duration = 4.0
        n = round(duration * self.args.fps)
        self.orientations.clear()
        for k in range(n):
            u = smooth((k + 1) / n)
            for name in names:
                self.hands.values[name] = start[name] * (1 - u) + target[name] * u
            self.hands.grip = {s: grips[s] * (1 - u) for s in grips}
            self.hands.apply_grips()
            eef = self.hands.centres()
            self.advance(phase, eef, {'L': 0.0, 'R': 0.0}, eef, False)

    def save(self, failure=None):
        super().save(failure)
        p = self.output / 'trial.json'
        if p.exists():
            m = json.loads(p.read_text())
            m.update(owner_action_entrypoint=str(Path(__file__).resolve()), home_control='Four-second minimum-jerk return to measured initial arm joints after release and vertical clearance')
            p.write_text(json.dumps(m, indent=2) + '\n')

@wp.kernel
def set_relative_alpha(alpha: wp.array(dtype=wp.float32), base: wp.array(dtype=wp.float32), offset: float):
    alpha[0] = base[0] + offset

class ChunkedCapture(GuardedCapture):

    def advance(self, phase, targets, errors, planner_targets, retimed):
        body = self.body_transforms()
        if not self.safe_constant_clamps(self.previous_body, body):
            self.slow_frames = getattr(self, 'slow_frames', 0) + 1
            return super().advance(phase, targets, errors, planner_targets, retimed)
        self.start.assign(self.previous_body)
        self.stop.assign(body)
        self.previous_body = body.copy()
        if self.graph is None:
            self.setup_graph()
        if not hasattr(self, 'whole_graph'):
            self.chunk_base = wp.zeros(1, dtype=wp.float32)
            from warp._src import context
            with wp.ScopedCapture() as capture:
                for sub in range(0, 16, 2):
                    wp.launch(set_relative_alpha, dim=1, inputs=[self.alpha_offset, self.chunk_base, float(sub / self.args.substeps)])
                    if not context.runtime.core.wp_cuda_graph_insert_child_graph(self.graph.device.context, self.graph.device.stream.cuda_stream, self.graph.graph):
                        raise RuntimeError(context.runtime.get_error_string())
            self.whole_graph = capture.graph
        for begin in range(0, self.args.substeps, 16):
            self.chunk_base.assign(np.array([begin / self.args.substeps], dtype=np.float32))
            wp.capture_launch(self.whole_graph)
        self.previous_grip = self.hands.grip.copy()
        points = self.a.particle_q.numpy()
        if not np.isfinite(points).all():
            raise RuntimeError('Non-finite cloth state')
        self.frames.append(dict(phase=phase, q=points, qd=self.a.particle_qd.numpy(), joint=np.array([self.hands.values.get(n, 0.0) for n in self.robot.names]), body=body.copy(), grip=[self.hands.grip[s] for s in ['L', 'R']], eef=np.array([self.hands.centres()[s] for s in ['L', 'R']]), target=np.array([targets[s] for s in ['L', 'R']]), planner_target=np.array([planner_targets[s] for s in ['L', 'R']]), retimed=retimed, ik=np.array([errors[s] for s in ['L', 'R']]), owner=self.owner.copy()))
        self.t += self.dt
        self.fast_frames = getattr(self, 'fast_frames', 0) + 1
        if len(self.frames) % 20 == 0:
            ratio = np.linalg.norm(points[self.edges[:, 0]] - points[self.edges[:, 1]], axis=1) / self.rest_lengths
            print(json.dumps(dict(frame=len(self.frames), time=round(self.t, 2), phase=phase, min_z=float(points[:, 2].min()), max_z=float(points[:, 2].max()), strain_p99=float(np.quantile(ratio, 0.99)), strain_max=float(ratio.max()), particle_speed_max=float(np.linalg.norm(self.frames[-1]['qd'], axis=1).max()), ik=errors, clamps={s: len(i) for s, i in self.clamps.items()}, batched_graph=True)), flush=True)
        if max(errors.values()) > self.args.max_ik_error:
            raise RuntimeError(f'Unreachable EEF target: {errors}')
        f = self.frames[-1]
        q = f['q']
        ratio = np.linalg.norm(q[self.edges[:, 0]] - q[self.edges[:, 1]], axis=1) / self.rest_lengths
        problem = self.quality_guard.examine(f['body'], q)
        if problem is None and (ratio.max() > 1.1 or np.quantile(ratio, 0.99) > 1.04):
            problem = dict(kind='material_strain', maximum=float(ratio.max()), p99=float(np.quantile(ratio, 0.99)))
        speed = float(np.linalg.norm(f['qd'], axis=1).max())
        if problem is None and speed > 5.0:
            problem = dict(kind='particle_velocity', maximum_m_s=speed)
        if problem is not None:
            self.events.append(dict(event='online_quality_abort', frame_index=len(self.frames) - 1, time=self.t, phase=phase, **problem))
            raise RuntimeError('Physical quality rejection: ' + json.dumps(problem))

    def save(self, failure=None):
        super().save(failure)
        p = self.output / 'trial.json'
        if p.exists():
            m = json.loads(p.read_text())
            m.update(owner_action_entrypoint=str(Path(__file__).resolve()), solver_graph_chunk_substeps=16)
            p.write_text(json.dumps(m, indent=2) + '\n')

class ReadyWaypointCapture(ChunkedCapture):

    def move(self, phase, *args, **kwargs):
        if phase != 'single_corner_lift':
            return super().move(phase, *args, **kwargs)
        waypoint = self.policy['left_ready_joint_waypoint']
        names = self.robot.arm_names('L')
        if names != waypoint['joint_names']:
            raise ValueError('Left waypoint joint order mismatch')
        self.ready_start = np.array([self.hands.values[n] for n in names])
        self.ready_end = np.asarray(waypoint['values_rad'], dtype=float)
        self.ready_steps = round(self.policy['single_corner_lift']['seconds'] * self.args.fps)
        self.ready_step = 0
        self.events.append(dict(time=self.t, event='smooth_left_ready_waypoint', source=waypoint, control='Minimum-jerk joint interpolation during right lift'))
        try:
            return super().move(phase, *args, **kwargs)
        finally:
            self.orientations.pop('L', None)

    def frame(self, phase, targets, grips):
        if phase != 'single_corner_lift':
            return super().frame(phase, targets, grips)
        original_command = self.hands.command
        self.ready_step += 1
        u = smooth(min(1.0, self.ready_step / self.ready_steps))
        left = self.ready_start * (1 - u) + self.ready_end * u

        def issue_right_and_left_waypoint(goals, jaw, orientations=None):
            self.hands.grip.update(jaw)
            self.hands.apply_grips()
            self.hands.values, error = self.robot.solve(self.hands.values, 'R', np.asarray(goals['R']), self.hands.mouth_local('R')[0], None if orientations is None else orientations.get('R'))
            for name, value in zip(self.robot.arm_names('L'), left):
                lo, hi = self.robot.by_name[name]['limits']
                if not lo < value < hi:
                    raise ValueError('Left ready waypoint violates a robot joint limit')
                self.hands.values[name] = float(value)
            goals['L'] = self.hands.centres()['L']
            return {'L': 0.0, 'R': error}
        self.hands.command = issue_right_and_left_waypoint
        try:
            return super().frame(phase, targets, grips)
        finally:
            self.hands.command = original_command

@wp.kernel
def record_pair_displacement(before: wp.array(dtype=wp.vec3), after: wp.array(dtype=wp.vec3), maximum: wp.array(dtype=wp.float32), calls: wp.array(dtype=wp.int32)):
    i = wp.tid()
    maximum[i] = wp.max(maximum[i], wp.length(after[i] - before[i]))
    if i == 0:
        calls[0] += 1

class ResolvedCapture(ReadyWaypointCapture):

    def setup_graph(self):
        super().setup_graph()
        original = self.graph
        self.pair_before = wp.empty_like(self.a.particle_q)
        self.pair_maximum = wp.zeros(self.n, dtype=wp.float32)
        self.pair_calls = wp.zeros(1, dtype=wp.int32)
        self.pair_records = []
        from warp._src import context
        with wp.ScopedCapture() as capture:
            wp.copy(self.pair_before, self.a.particle_q)
            if not context.runtime.core.wp_cuda_graph_insert_child_graph(original.device.context, original.device.stream.cuda_stream, original.graph):
                raise RuntimeError(context.runtime.get_error_string())
            wp.launch(record_pair_displacement, dim=self.n, inputs=[self.pair_before, self.a.particle_q, self.pair_maximum, self.pair_calls])
        self.graph = capture.graph
        self.original_physics_graph = original

    def advance(self, *args, **kwargs):
        before = len(self.frames)
        if hasattr(self, 'pair_maximum'):
            self.pair_maximum.zero_()
            self.pair_calls.zero_()
        try:
            result = super().advance(*args, **kwargs)
        finally:
            if len(self.frames) > before and hasattr(self, 'pair_maximum'):
                values = self.pair_maximum.numpy()
                self.pair_records.append(dict(frame_index=len(self.frames) - 1, maximum_two_substep_displacement_m=float(values.max()), maximum_particle=int(values.argmax()), observed_pair_count=int(self.pair_calls.numpy()[0])))
        return result

    def save(self, failure=None):
        super().save(failure)
        if not self.frames:
            return
        records = getattr(self, 'pair_records', [])
        evidence = dict(schema='tablecloth.resolved_motion_observation.v1', measurement='Maximum actual particle displacement inside each original two-substep solver graph; read-only instrumentation', pair_dt_s=2 * self.dt / self.args.substeps, expected_pairs_per_saved_frame=self.args.substeps // 2, physical_solver_or_material_changed=False, frames=records)
        (self.output / 'resolved_motion.json').write_text(json.dumps(evidence, indent=2) + '\n')
        path = self.output / 'trial.json'
        meta = json.loads(path.read_text())
        meta['owner_action_entrypoint'] = str(Path(__file__).resolve())
        meta['motion_resolution_evidence'] = 'resolved_motion.json'
        path.write_text(json.dumps(meta, indent=2) + '\n')

class SteadyRightResolvedCapture(ResolvedCapture):

    def frame(self, phase, targets, grips):
        if phase not in {'let_corner_settle', 'L_enter_tip', 'L_close_tip', 'two_corner_hold'}:
            return super().frame(phase, targets, grips)
        original = self.hands.command
        frozen = {n: self.hands.values[n] for n in self.robot.arm_names('R')}

        def command(goals, jaw, orientations=None):
            errors = original(goals, jaw, orientations)
            self.hands.values.update(frozen)
            goals['R'] = self.hands.centres()['R']
            errors['R'] = 0.0
            return errors
        self.hands.command = command
        try:
            return super().frame(phase, targets, grips)
        finally:
            self.hands.command = original

    def save(self, failure=None):
        super().save(failure)
        p = self.output / 'trial.json'
        if p.exists():
            d = json.loads(p.read_text())
            d.update(owner_action_entrypoint=str(Path(__file__).resolve()), right_hold_control='Hold actual reached right-arm joints while the left hand approaches; eliminate redundant idle IK corrections')
            p.write_text(json.dumps(d, indent=2) + '\n')

class StrictSteadyCapture(SteadyRightResolvedCapture):

    def advance(self, *args, **kwargs):
        result = super().advance(*args, **kwargs)
        row = self.pair_records[-1]
        if row['observed_pair_count'] != self.args.substeps // 2 or row['maximum_two_substep_displacement_m'] > 0.005:
            event = dict(event='online_quality_abort', time=self.t, phase=self.frames[-1]['phase'], kind='resolved_particle_motion', **row)
            self.events.append(event)
            raise RuntimeError('Physical quality rejection: ' + json.dumps(event))
        return result

    def close_right_tip(self, corner):
        cfg = self.policy.get('right_grasp', {})
        if 'depth_m' not in cfg:
            return super().close_right_tip(corner)
        self.pending_grasps['R'] = corner
        g0 = self.hands.grip.copy()
        n = max(1, round(cfg.get('close_seconds', 0.8) * self.args.fps))
        for k in range(n):
            u = smooth((k + 1) / n)
            grips = {**g0, 'R': g0['R'] + (0.965 - g0['R']) * u}
            values = self.hands.values.copy()
            oldgrip = self.hands.grip.copy()
            self.hands.grip.update(grips)
            self.hands.apply_grips()
            bite = self.hands.shallow_grasp_offset('R', cfg['depth_m'])
            self.hands.values = values
            self.hands.grip = oldgrip
            goals = self.hands.centres()
            goals['R'] = self.a.particle_q.numpy()[corner] + bite
            self.frame('R_close_tip', goals, grips)
            if 'R' in self.clamps:
                break
        self.pending_grasps.pop('R', None)
        if 'R' not in self.clamps:
            raise RuntimeError('Right closure missed unchanged physical pad gate')
        self.move('R_finish_close', 0.2, grips={'R': 0.96})

    def save(self, failure=None):
        super().save(failure)
        p = self.output / 'trial.json'
        if p.exists():
            d = json.loads(p.read_text())
            d.update(owner_action_entrypoint=str(Path(__file__).resolve()), online_resolved_motion_bound_m=0.005, right_bite_control='Policy right_grasp.depth_m, default canonical4mm; opposing physical pad gate unchanged')
            p.write_text(json.dumps(d, indent=2) + '\n')

def contact_cosine(vectors):
    v = np.asarray(vectors)
    return float(np.dot(v[0], v[1]) / max(np.linalg.norm(v[0]) * np.linalg.norm(v[1]), 1e-12))

class OpposedCapture(StrictSteadyCapture):

    def clamp(self, side, target_id, **kwargs):
        if side in self.clamps:
            return super().clamp(side, target_id, **kwargs)
        poses = self.actual_hand_poses(side)
        point = self.a.particle_q.numpy()[[target_id]]
        d, v = self.hands.pad_surfaces(point, side, poses=poses)
        cosine = contact_cosine(v[0])
        if not np.isfinite(cosine) or cosine >= -0.55:
            near = bool((d[0] >= -0.0005).all() and (d[0] <= self.args.radius + 0.0015).all())
            key = (side, len(self.frames))
            seen = getattr(self, 'opposition_wait_seen', set())
            if near and key not in seen:
                seen.add(key)
                self.opposition_wait_seen = seen
                self.events.append(dict(event='pinch_waiting_opposition', side=side, frame_index=len(self.frames), time=kwargs.get('event_time', self.t), actual_contact_cosine=cosine, required_cosine_below=-0.55, role='Withhold pinch activation; original physical pad gate remains unchanged'))
            return False
        return super().clamp(side, target_id, **kwargs)

    def save(self, failure=None):
        super().save(failure)
        p = self.output / 'trial.json'
        if p.exists():
            d = json.loads(p.read_text())
            d.update(owner_action_entrypoint=str(Path(__file__).resolve()), pinch_activation_control={'measured_at': 'actual original solver contact boundary', 'maximum_allowed_opposing_contact_cosine': -0.55, 'original_pad_gap_distance_gate_unchanged': True, 'downstream_independent_required_cosine': -0.5})
            p.write_text(json.dumps(d, indent=2) + '\n')

class BoundedCloseCapture(OpposedCapture):

    def frame(self, phase, *args, **kwargs):
        if phase != 'L_close_tip':
            return super().frame(phase, *args, **kwargs)
        original = self.args.max_joint_speed
        bound = float(self.policy['left_grasp'].get('closing_joint_speed_rad_s', 0.5))
        self.args.max_joint_speed = min(original, bound)
        try:
            return super().frame(phase, *args, **kwargs)
        finally:
            self.args.max_joint_speed = original

    def save(self, failure=None):
        super().save(failure)
        p = self.output / 'trial.json'
        if p.exists():
            t = json.loads(p.read_text())
            t.update(owner_action_entrypoint=str(Path(__file__).resolve()), left_closing_joint_speed_bound_rad_s=float(self.policy['left_grasp'].get('closing_joint_speed_rad_s', 0.5)), left_closing_control='Cap actual left-arm joint step during final jaw closure; original jaw timing, geometry gate, material and solver unchanged')
            p.write_text(json.dumps(t, indent=2) + '\n')
