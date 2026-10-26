"""Read live cloth position between physical brush passes, then retarget the pinch."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

REPO = Path('/home/aether/workspace/agent_bench')
sys.path.insert(0, str(REPO / 'experiments/physics_fluid'))
from tablecloth_robot_kinematics import Robot

ROBOT_META = REPO / '.codex-tmp/tablecloth-action-loop-20260905/robot_native_collision.json'


class AdaptiveBrush:
    def __init__(self, action, uv, cloth_ids, stage, robot_base, target_overhang_m=.04, follow_brush_corner=False):
        from pxr import Gf, UsdGeom

        self.follow_brush_corner = bool(follow_brush_corner)
        self.initial_corner = None
        self.repeat_shift = None
        self.repeat_start = None
        self.repeat_end = None
        self.repeat_prior_arm = None
        self.target = float(target_overhang_m)
        if not .04 <= self.target <= .15:
            raise ValueError('Brush target overhang outside diagnostic range')
        phases = action['phase'].tolist()
        self.clear_ends = [i for i in range(len(phases)-1)
                           if 'friction_clear' in phases[i] and phases[i+1] != phases[i]]
        self.grasp_source = phases.index('L_match_reference_entry_start')
        if len(self.clear_ends) < 2 or self.grasp_source <= self.clear_ends[-1]:
            raise ValueError('Adaptive command needs at least two complete brushes before pinch')
        old = np.flatnonzero(cloth_ids == 0)
        corner_uv = [float(uv[old, 0].min()), float(uv[old, 1].max())]
        self.corner = int(old[np.linalg.norm(uv[old] - corner_uv, axis=1).argmin()])
        table = stage.GetPrimAtPath('/World/Assets/CoffeeTable/Mesh')
        if not table or not table.IsA(UsdGeom.Mesh):
            raise RuntimeError('Shared House table mesh missing')
        # The authored mesh has a rotated/scaled local coordinate system. Use
        # its actual world vertices projected into the shared robot/task X axis.
        self.axis = np.asarray(robot_base, dtype=float)[:3, 0]
        self.axis /= np.linalg.norm(self.axis)
        self.lateral_axis = np.asarray(robot_base, dtype=float)[:3, 1]
        self.lateral_axis /= np.linalg.norm(self.lateral_axis)
        transform = UsdGeom.XformCache().GetLocalToWorldTransform(table)
        points = UsdGeom.Mesh(table).GetPointsAttr().Get()
        if not points:
            raise RuntimeError('Shared House table mesh has no vertices')
        world = np.asarray([transform.Transform(Gf.Vec3d(p)) for p in points], dtype=float)
        self.edge_projection = float((world @ self.axis).min())
        self.robot = Robot(json.loads(ROBOT_META.read_text()), [0, 0, 0])
        self.robot.base = np.asarray(robot_base, dtype=float)
        self.names = action['joint_names'].tolist()
        self.arm = Robot.arm_names('L')
        self.reference_corner = (None if "grasp_reference_corner" not in action else np.asarray(action["grasp_reference_corner"],dtype=float))
        if self.reference_corner is not None and (self.reference_corner.shape != (3,) or not np.isfinite(self.reference_corner).all()):
            raise ValueError("Invalid grasp reference corner")
        self.first_corner = None
        self.grasp_shift = None
        self.grasp_start_frame = None
        self.prior_arm = None
        self.trace = []
        self.failure = None

    def on_frame(self, source_frame, physical_frame, cloth):
        if physical_frame == 0:
            self.initial_corner = cloth[self.corner].astype(float).copy()
        if source_frame not in self.clear_ends:
            return source_frame + 1
        corner = cloth[self.corner].astype(float)
        overhang = self.edge_projection - float(corner @ self.axis)
        pass_index = self.clear_ends.index(source_frame) + 1
        self.trace.append(dict(physical_frame=physical_frame, command_frame=source_frame,
                               pass_index=pass_index, corner_xyz_m=corner.tolist(),
                               overhang_m=overhang, threshold_m=self.target,
                               decision=("grasp" if overhang >= self.target else "repeat" if source_frame != self.clear_ends[-1] else "stop_insufficient")))
        if self.first_corner is None:
            self.first_corner = corner.copy()
        if overhang >= self.target:
            self.grasp_shift = corner - (self.first_corner if self.reference_corner is None else self.reference_corner)
            self.grasp_start_frame = physical_frame + 1
            print(json.dumps(dict(adaptive_brush='grasp', pass_index=pass_index,
                                  overhang_m=overhang, shift_m=self.grasp_shift.tolist())), flush=True)
            return self.grasp_source
        if source_frame == self.clear_ends[-1]:
            self.failure = f'Brush limit reached: overhang {overhang:.4f} < {self.target:.4f} m'
            print(json.dumps(dict(adaptive_brush='failed', pass_index=pass_index,
                                  overhang_m=overhang)), flush=True)
            return None
        if getattr(self, 'follow_brush_corner', False) and self.initial_corner is not None:
            lateral = float((corner-self.initial_corner) @ self.lateral_axis)
            self.repeat_shift = self.lateral_axis*lateral
            self.repeat_start = source_frame+1
            self.repeat_end = self.clear_ends[pass_index]
            self.repeat_prior_arm = None
            self.trace[-1]['next_brush_lateral_shift_m'] = self.repeat_shift.tolist()
        print(json.dumps(dict(adaptive_brush='repeat', pass_index=pass_index,
                              overhang_m=overhang)), flush=True)
        return source_frame + 1

    def command(self, source_frame, physical_frame, template):
        if self.grasp_start_frame is None or source_frame < self.grasp_source:
            if self.repeat_shift is None or not self.repeat_start <= source_frame <= self.repeat_end:
                return template
            # Shift only robot targets along the edge to follow the displaced corner.
            # Ramp during raised return; restore after the fingers have lifted clear.
            k=source_frame-self.repeat_start
            u=float(np.clip(k/29.,0,1)); blend=u*u*(3-2*u)
            if source_frame>self.repeat_end-9:
                u=float((self.repeat_end-source_frame)/9.); blend=u*u*(3-2*u)
            values=dict(zip(self.names,map(float,template)))
            pose=self.robot.fk(values)['gripper_l_base_link']; seed=values.copy()
            if self.repeat_prior_arm is not None:seed.update(self.repeat_prior_arm)
            solved,error=self.robot.solve(seed,'L',pose[:3,3]+blend*self.repeat_shift,np.zeros(3),pose[:3,:3],max_steps=500)
            actual=self.robot.fk(solved)['gripper_l_base_link']
            angle=float(np.degrees(np.arccos(np.clip((np.trace(pose[:3,:3]@actual[:3,:3].T)-1)/2,-1,1))))
            rate=max(abs(solved[n]-seed[n])*20 for n in self.arm)
            if error>.002 or angle>2 or rate>1.5:raise RuntimeError(f'Live repeat-brush IK failed: frame={physical_frame}, error={error}, angle={angle}, rate={rate}')
            result=template.copy()
            for name in self.arm:result[self.names.index(name)]=solved[name]
            self.repeat_prior_arm={n:solved[n] for n in self.arm}
            return result
        result = template.copy()
        if self.prior_arm is None:
            self.prior_arm = {name: float(template[self.names.index(name)]) for name in self.arm}
        if np.linalg.norm(self.grasp_shift) < 1e-6:
            return result
        values = {name: float(value) for name, value in zip(self.names, template)}
        original = self.robot.fk(values)['gripper_l_base_link']
        u = np.clip((physical_frame-self.grasp_start_frame)/50, 0., 1.)
        blend = u*u*(3-2*u)
        goal = original[:3, 3] + blend*self.grasp_shift
        seed = values.copy()
        seed.update(self.prior_arm)
        solved, error = self.robot.solve(seed, 'L', goal, np.zeros(3),
                                         original[:3, :3], max_steps=180)
        rate = max(abs(solved[name]-self.prior_arm[name])*20 for name in self.arm)
        if error > .002 or rate > 1.5:
            raise RuntimeError(f'Live grasp retarget IK failed: frame={physical_frame}, '
                               f'error={error:.5f}, rate={rate:.3f}')
        for name in self.arm:
            result[self.names.index(name)] = solved[name]
        self.prior_arm = {name: solved[name] for name in self.arm}
        return result

    def report(self):
        return dict(grasp_reference_corner_m=None if self.reference_corner is None else self.reference_corner.tolist(), follow_repeat_brush_corner=self.follow_brush_corner, policy="measure_after_each_brush_grasp_if_at_least_threshold",
                    distance_definition="signed horizontal projection of grasp corner beyond near table edge in robot task X",
                    target_overhang_m=self.target, table_edge_projection_m=self.edge_projection,
                    outward_task_axis_world=self.axis.tolist(),
                    corner_particle_index=self.corner, pass_readbacks=self.trace,
                    grasp_shift_m=None if self.grasp_shift is None else self.grasp_shift.tolist(),
                    grasp_start_frame=self.grasp_start_frame, failure=self.failure,
                    physical_state_splicing=False, cloth_pose_or_velocity_writes=False)
