"""Contact-gated canvas pickup, same-edge interception, and table spreading.

Material and initial scene state are explicit, fixed inputs. All corrections
change robot commands; free cloth positions are never prescribed by the policy.
A completed action sequence remains a candidate for Sheet-owner review.
"""
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from tablecloth_rule_sim import Simulation, main, smooth


class CanvasActions(Simulation):
    def __init__(self, args):
        super().__init__(args)
        self.capture_servo_samples = []

    def frame(self, phase, targets, grips):
        if phase not in {'L_enter_tip', 'L_close_tip', 'L_retry_retract'}:
            return super().frame(phase, targets, grips)
        # Replan after every physical frame. Completing an old IK command via
        # inserted frames can make a moving hem escape before the jaws close.
        previous = self.hands.values.copy()
        self.hands.command(targets, grips, {**self.orientations, 'L': None})
        names = self.robot.arm_names('L') + self.robot.arm_names('R')
        delta = max(abs(self.hands.values[n] - previous[n]) for n in names)
        fraction = min(1., self.args.max_joint_speed * self.dt / max(delta, 1e-12))
        for name in names:
            self.hands.values[name] = previous[name] + fraction * (self.hands.values[name] - previous[name])
        self.hands.apply_grips()
        issued = self.hands.centres()
        # Actions are the issued joint positions and their FK. The unconverged
        # moving-corner goal is separately retained as planner_eef_goal.
        self.advance(phase, issued, {'L': 0., 'R': 0.}, targets, True)
        self.capture_servo_samples.append(dict(
            frame_index=len(self.frames) - 1, phase=phase,
            planner_goal_error_m=float(np.linalg.norm(issued['L'] - targets['L'])),
            joint_step_fraction=float(fraction)))

    def run_policy(self):
        if self.args.mode != 'spread':
            raise ValueError('Canvas action producer requires spread mode')
        plan = self.policy['tip_grasp']
        corner = int(np.linalg.norm(self.uv - np.asarray(plan['material_uv']), axis=1).argmin())
        rotation = Rotation.from_quat(plan['quaternion_xyzw']).as_matrix()
        palm = rotation @ self.hands.mouth_local('R')[3]
        self.move('settle', self.args.settle)
        # Keep a small stand-off during the verified right approach; insertion
        # finishes while closing. Left interception below uses one bite offset.
        _, _, _, local_palm, _, _, bounds = self.hands.mouth_local('R')
        stand_off = -rotation @ local_palm * (bounds[1, 0] + self.args.grasp_depth - bounds[1].mean())
        self.move('R_above_tip', 2.5, grips={'R': .9},
                  follow={'R': (corner, stand_off + palm * .12)}, orientations={'R': rotation})
        self.move('R_enter_tip', 1., grips={'R': .9}, follow={'R': (corner, self.hands.shallow_grasp_offset('R', self.args.grasp_depth))})
        self.close_right_tip(corner)
        point = self.hands.centres()['R']
        self.move('clear_basket_rim', 1.2, {'R': point + [0, 0, .16]})
        held_uv = self.uv[self.clamps['R'][0]].copy()
        partner_uv = np.array([held_uv[0], self.size[1] - held_uv[1]])
        partner = int(np.linalg.norm(self.uv - partner_uv, axis=1).argmin())
        self.events.append(dict(time=self.t, event='same_edge_partner_target',
                                held_material_uv=held_uv.tolist(),
                                partner_material_uv=self.uv[partner].tolist(), particle=partner))
        left_rotation = Rotation.from_quat(self.policy['left_grasp']['quaternion_xyzw']).as_matrix()
        self.orientations.pop('R', None)
        self.move('single_corner_lift', 4.,
                  {'R': [-2.12, -.82, 1.30], 'L': [-2.23, -.46, .79]},
                  grips={'L': .9}, orientations={'L': left_rotation})
        self.capture_left_tip(partner)
        self.orientations.clear()
        self.move('two_corner_hold', .5)
        if self.policy.get('development_stage') == 'two_corner_capture':
            return
        span = float(np.linalg.norm(self.uv[self.clamps['L'][0]] - self.uv[self.clamps['R'][0]])) * .95
        self.events.append(dict(time=self.t, event='held_edge_spacing', actual_material_span_m=span/.95,
                                commanded_span_m=span, control='EEF offsets from actual grasp points'))
        self.held_edge('unfold_same_edge', 3., [-2.23, -.50, 1.08], span)
        self.held_edge('centre_over_table', 2., [-2.23, -.20, 1.03], span)
        self.held_edge('fling_forward', 1.2, [-2.05, -.20, .83], span)
        self.held_edge('lay_near_edge', 1.8, [-2.43, -.20, .50], span)
        self.move('release_cloth', .3, grips={'L': .9, 'R': .9})
        self.move('release_drop', .4)
        centres = self.hands.centres()
        self.move('withdraw_hands', 1., {s: centres[s] + [0, 0, .15] for s in centres})
        self.move('open_clear_hands', .4, grips={'L': 0., 'R': 0.})
        self.move('settle_on_table', 3.)

    def close_right_tip(self, corner):
        self.pending_grasps['R'] = corner
        grips0 = self.hands.grip.copy()
        targets = self.hands.centres()
        frames = max(1, round(.8 * self.args.fps))
        for k in range(frames):
            fraction = smooth((k + 1) / frames)
            goals = {hand: point.copy() for hand, point in targets.items()}
            goals['R'] = self.a.particle_q.numpy()[corner] + self.hands.shallow_grasp_offset('R', self.args.grasp_depth)
            grips = {**grips0, 'R': grips0['R'] * (1 - fraction) + .965 * fraction}
            self.frame('R_close_tip', goals, grips)
            if 'R' in self.clamps:
                break
        self.pending_grasps.pop('R', None)
        if 'R' not in self.clamps:
            raise RuntimeError('R tip closure missed the physical jaw contact gate')
        self.move('R_finish_close', .2, grips={'R': .96})

    def left_goal(self, corner, grip=None):
        point = self.a.particle_q.numpy()[corner]
        prediction = self.a.particle_qd.numpy()[corner] * self.dt * self.policy.get('left_grasp', {}).get('prediction_fraction', 1.)
        prediction *= min(1., .012 / max(np.linalg.norm(prediction), 1e-12))
        depth = self.policy.get('left_grasp', {}).get('depth_m', self.args.grasp_depth)
        values, grips = self.hands.values.copy(), self.hands.grip.copy()
        try:
            if grip is not None:
                self.hands.grip['L'] = grip
                self.hands.apply_grips()
            bite = self.hands.shallow_grasp_offset('L', depth)
        finally:
            self.hands.values, self.hands.grip = values, grips
        return point + prediction + bite

    def capture_left_tip(self, corner):
        start = self.hands.centres()['L'].copy()
        seconds = self.policy.get('left_grasp', {}).get('approach_seconds', .3)
        steps = max(1, round(seconds * self.args.fps))
        for k in range(steps):
            fraction = smooth((k + 1) / steps)
            goals = self.hands.centres()
            goals['L'] = start * (1 - fraction) + self.left_goal(corner) * fraction
            self.frame('L_enter_tip', goals, self.hands.grip.copy())
        # Basket folds change the free corner's swing. Close on measured
        # alignment, rather than after the short nominal approach duration.
        for _ in range(max(1, round(3. * self.args.fps))):
            point = self.a.particle_q.numpy()[corner]
            distances, vectors = self.hands.pad_surfaces(point[None], 'L')
            cosine = float(np.dot(vectors[0, 0], vectors[0, 1]) / max(
                np.linalg.norm(vectors[0, 0]) * np.linalg.norm(vectors[0, 1]), 1e-12))
            error = float(np.linalg.norm(self.hands.centres()['L'] - self.left_goal(corner)))
            if (distances[0] >= -.0005).all() and cosine < -.8 and distances[0].sum() < .06:
                self.events.append(dict(time=self.t, event='left_tip_aligned',
                                        particle=corner, predicted_goal_error_m=error,
                                        opposing_surface_distances_m=distances[0].tolist(),
                                        opposing_contact_cosine=cosine))
                break
            goals = self.hands.centres()
            goals['L'] = self.left_goal(corner)
            self.frame('L_enter_tip', goals, self.hands.grip.copy())
        else:
            raise RuntimeError('L tip did not enter the open jaw footprint within the approach window')
        self.pending_grasps['L'] = corner
        grips0 = self.hands.grip.copy()
        closing_seconds = self.policy.get('left_grasp', {}).get('close_seconds', .1)
        for k in range(max(2, round(.6 * self.args.fps))):
            goals = self.hands.centres()
            fraction = smooth(min((k + 1) / max(1, round(closing_seconds * self.args.fps)), 1.))
            grips = {**grips0, 'L': grips0['L'] + (.965 - grips0['L']) * fraction}
            goals['L'] = self.left_goal(corner, grip=grips['L'])
            self.frame('L_close_tip', goals, grips)
            if 'L' in self.clamps:
                break
        self.pending_grasps.pop('L', None)
        if 'L' not in self.clamps:
            raise RuntimeError('L tip closure missed the physical jaw contact gate')

    def held_edge(self, phase, seconds, centre, span):
        override = self.policy.get(phase, {})
        centre = np.asarray(override.get('centre', centre))
        span = override.get('span', span)
        seconds = override.get('seconds', seconds)
        ids = {side: int(self.clamps[side][0]) for side in ('L', 'R')}
        q = self.a.particle_q.numpy()
        start = {side: q[index].copy() for side, index in ids.items()}
        desired = {side: centre + [0, (1 if side == 'L' else -1) * span / 2, 0] for side in ids}
        frames = max(1, round(seconds * self.args.fps))
        print(f'[edge] {phase}: {frames} frames', flush=True)
        for k in range(frames):
            fraction = smooth((k + 1) / frames)
            q, eef = self.a.particle_q.numpy(), self.hands.centres()
            goals = {side: start[side] * (1 - fraction) + desired[side] * fraction + eef[side] - q[ids[side]] for side in ids}
            self.frame(phase, goals, self.hands.grip.copy())

    def save(self, failure=None):
        super().save(failure)
        path = self.output / 'trial.json'
        if not path.exists():
            return
        metadata = json.loads(path.read_text())
        metadata.update(trial_role='continuous_canvas_spread_candidate',
                        owner_action_entrypoint=str(Path(__file__).resolve()),
                        completed_action_sequence=bool(failure is None and self.frames and
                                                       self.frames[-1]['phase'] == 'settle_on_table' and not self.clamps),
                        completed_benchmark_trajectory=False,
                        dynamic_capture_control='20 Hz joint-velocity-limited feedback with one-step cloth-corner prediction; EEF targets are FK of issued joints, planner_eef_goal retains moving target',
                        capture_servo_samples=self.capture_servo_samples)
        path.write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main(simulation_class=CanvasActions)
