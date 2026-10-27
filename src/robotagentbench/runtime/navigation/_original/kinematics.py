"""Kinematic model of the G1 omnipicker used by the exploration backend.

Pure numpy. The model JSON comes from tools/extract_robot_model.py, which reads the
joint frames, limits and velocity limits from the scene USD. No physics is simulated:
joints follow their position targets under the USD velocity limits and the base
integrates the commanded forward / yaw-rate velocities.
"""
import json
import math

import numpy as np

BASE_LINK = 'base_link'
AXES = {'X': 0, 'Y': 1, 'Z': 2}

# g1_omnipicker_jointpos20_basevw_v1 (same order as the VLA datasets)
ARM_L = [f'idx{21 + i}_arm_l_joint{i + 1}' for i in range(7)]
ARM_R = [f'idx{61 + i}_arm_r_joint{i + 1}' for i in range(7)]
JOINT_NAMES = ARM_L + ARM_R + [
    'idx01_body_joint1', 'idx02_body_joint2', 'idx11_head_joint1', 'idx12_head_joint2',
    'idx41_gripper_l_outer_joint1', 'idx81_gripper_r_outer_joint1']
ACTION_NAMES = JOINT_NAMES + ['base_forward_m_s', 'base_yaw_rate_rad_s']
STATE_NAMES = JOINT_NAMES + ['base_x_world_m', 'base_y_world_m', 'base_yaw_world_unwrapped_rad']
# The inner gripper joint mirrors the commanded outer joint (same rule as the native room host).
MIMIC = {'idx31_gripper_l_inner_joint1': ('idx41_gripper_l_outer_joint1', -1.0),
         'idx71_gripper_r_inner_joint1': ('idx81_gripper_r_outer_joint1', -1.0)}


def joint_motion(kind, axis, q):
    m = np.eye(4)
    k = AXES[axis]
    if kind == 'prismatic':
        m[k, 3] = q
        return m
    c, s = math.cos(q), math.sin(q)
    i, j = [a for a in range(3) if a != k]
    # right-handed rotation about axis k
    if k == 0:
        m[1, 1], m[1, 2], m[2, 1], m[2, 2] = c, -s, s, c
    elif k == 1:
        m[0, 0], m[0, 2], m[2, 0], m[2, 2] = c, s, -s, c
    else:
        m[0, 0], m[0, 1], m[1, 0], m[1, 1] = c, -s, s, c
    return m


def planar(x, y, yaw, z=0.0):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0, x], [s, c, 0, y], [0, 0, 1, z], [0, 0, 0, 1]], dtype=float)


class RobotModel:
    def __init__(self, model):
        if isinstance(model, str):
            with open(model) as f:
                model = json.load(f)
        self.raw = model
        root_world = np.array(model['root_world'])
        self.authored_links = {k: root_world @ np.array(v) for k, v in model['links'].items()}
        base = self.authored_links[BASE_LINK]
        self.base_z = float(base[2, 3])
        self.authored_base = (float(base[0, 3]), float(base[1, 3]), math.atan2(base[1, 0], base[0, 0]))
        inv_base = np.linalg.inv(base)
        # tree joints: one parent per child, loop-closure (excluded) joints are skipped
        self.tree = []
        seen = {BASE_LINK}
        pending = [j for j in model['joints'] if j['parent'] and j['child']
                   and not j['exclude_from_articulation'] and j['type'] in ('fixed', 'revolute', 'prismatic')]
        while pending:
            progressed = False
            for j in list(pending):
                if j['parent'] in seen and j['child'] not in seen:
                    self.tree.append(j)
                    seen.add(j['child'])
                    pending.remove(j)
                    progressed = True
                elif j['child'] in seen:
                    pending.remove(j)
            if not progressed:
                break
        self.movable = {j['name']: j for j in self.tree if j['type'] != 'fixed'}
        self.limits = {n: (j['lower'], j['upper']) for n, j in self.movable.items()}
        self.max_velocity = {n: j.get('max_velocity') for n, j in self.movable.items()}
        # joint values at which the links were authored, measured from the authored link poses
        self.authored_q = {}
        self.residual = {}
        rel = {k: inv_base @ v for k, v in self.authored_links.items()}
        for j in self.tree:
            f0, f1 = np.array(j['frame0']), np.array(j['frame1'])
            motion = np.linalg.inv(rel[j['parent']] @ f0) @ rel[j['child']] @ f1
            if j['type'] == 'fixed':
                q, ideal = 0.0, np.eye(4)
            elif j['type'] == 'prismatic':
                q = float(motion[AXES[j['axis']], 3])
                ideal = joint_motion('prismatic', j['axis'], q)
            else:
                k = AXES[j['axis']]
                a, b = [x for x in range(3) if x != k]
                q = math.atan2(motion[b, a], motion[a, a]) if k != 1 else math.atan2(motion[a, b], motion[a, a])
                ideal = joint_motion('revolute', j['axis'], q)
            self.residual[j['name']] = float(np.abs(motion - ideal).max())
            if j['type'] != 'fixed':
                self.authored_q[j['name']] = q
        # links that are not reached through the tree keep their authored pose relative to a tree link
        self.static_rel = {}
        for name, pose in rel.items():
            if name not in seen:
                self.static_rel[name] = pose  # relative to base_link (only gripper loop links fall here)
        self.cameras = {k: dict(v, local=np.array(v['local'])) for k, v in model['cameras'].items()}

    def default_q(self):
        return dict(self.authored_q)

    def expand(self, q20):
        """20 commanded joints -> all movable joints (mimic joints follow, the rest stay authored)."""
        q = dict(self.authored_q)
        q.update(q20)
        for name, (source, gain) in MIMIC.items():
            if name in q and source in q:
                lo, hi = self.limits[name]
                q[name] = min(hi, max(lo, gain * q[source]))
        return q

    def fk(self, q, base=None):
        """Link poses in the world for joint values q (dict) and base pose (x, y, yaw)."""
        full = self.expand(q)
        x, y, yaw = self.authored_base if base is None else base
        poses = {BASE_LINK: planar(x, y, yaw, self.base_z)}
        for j in self.tree:
            motion = np.eye(4) if j['type'] == 'fixed' else joint_motion(j['type'], j['axis'], full[j['name']])
            poses[j['child']] = poses[j['parent']] @ np.array(j['frame0']) @ motion @ np.linalg.inv(np.array(j['frame1']))
        for name, rel in self.static_rel.items():
            poses[name] = poses[BASE_LINK] @ rel
        return poses

    def head_camera(self, q, base):
        """World pose of the head camera using only the base -> head chain (fast path for every tick)."""
        if not hasattr(self, '_head_chain'):
            by_child = {j['child']: j for j in self.tree}
            chain, link = [], self.cameras['Head_Camera']['link']
            while link != BASE_LINK:
                chain.append(by_child[link])
                link = by_child[link]['parent']
            self._head_chain = [(j, np.array(j['frame0']), np.linalg.inv(np.array(j['frame1']))) for j in reversed(chain)]
        pose = planar(base[0], base[1], base[2], self.base_z)
        for j, f0, f1inv in self._head_chain:
            motion = np.eye(4) if j['type'] == 'fixed' else joint_motion(j['type'], j['axis'], q.get(j['name'], self.authored_q.get(j['name'], 0.0)))
            pose = pose @ f0 @ motion @ f1inv
        return pose @ self.cameras['Head_Camera']['local']

    def camera_pose(self, poses, camera='Head_Camera'):
        cam = self.cameras[camera]
        return poses[cam['link']] @ cam['local']


def camera_summary(pose):
    """USD cameras look along -Z with +Y up. Returns position, forward, heading and pitch (deg)."""
    forward = -pose[:3, 2]
    return dict(position_m=[round(float(v), 4) for v in pose[:3, 3]],
                forward=[round(float(v), 4) for v in forward],
                heading_deg=round(math.degrees(math.atan2(forward[1], forward[0])), 2),
                pitch_down_deg=round(math.degrees(-math.asin(max(-1.0, min(1.0, float(forward[2]))))), 2))
