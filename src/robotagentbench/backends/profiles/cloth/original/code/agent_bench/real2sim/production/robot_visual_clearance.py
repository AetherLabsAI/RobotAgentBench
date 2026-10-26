"""Whole-robot geometric exclusion from measured joints and current USD.

This checks visual motion, including a conservative bound between recorded
joint samples. It does not certify a new contact-capable physics rollout.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

CLEARANCE_M = 0.025


def matrix(position, quaternion_wxyz):
    q = np.asarray(quaternion_wxyz, dtype=float)
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    result[:3, 3] = position
    return result


def sphere_box_margin(bounds, centers, radii):
    box = np.asarray(bounds, dtype=float).reshape(2, 3)
    centers, radii = np.asarray(centers), np.asarray(radii)
    if (not np.isfinite(box).all() or np.any(box[1] < box[0])
            or not np.isfinite(centers).all() or not np.isfinite(radii).all()):
        raise ValueError("nonfinite or inverted clearance geometry")
    delta = np.maximum(np.maximum(box[0] - centers, centers - box[1]), 0)
    values = np.linalg.norm(delta, axis=-1) - radii
    index = np.unravel_index(np.argmin(values), values.shape)
    return float(values[index]), tuple(int(i) for i in index)


def forward_spheres(joints, roots, local_root, positions, names, link_radii, link_offsets=None):
    """FK spanning tree with conservative joint-interpolation displacement."""
    link_offsets = link_offsets or {}
    n = len(positions)
    if n < 2 or positions.shape != (n, len(names)) or not np.isfinite(positions).all():
        raise ValueError("missing finite measured joint trajectory")
    if len(set(names)) != len(names):
        raise ValueError("duplicate robot joint names")
    roots = np.asarray(roots)
    if roots.shape != (n, 4, 4) or not np.isfinite(roots).all():
        raise ValueError("invalid measured robot roots")
    root_name, root_offset = local_root
    poses = {root_name: roots @ root_offset}
    chains = {root_name: []}
    values = dict(zip(names, positions.T))
    pending = list(joints)
    ordered = []
    while pending:
        ready = [j for j in pending if j['parent'] in poses]
        if not ready:
            raise ValueError("robot joint graph is disconnected or cyclic")
        for j in ready:
            if j['child'] in poses:
                raise ValueError("multiple FK parents")
            kind, name = j['kind'], j['name']
            motion = np.repeat(np.eye(4)[None], n, axis=0)
            if kind != 'PhysicsFixedJoint':
                if name not in values:
                    raise ValueError("missing measured joint: " + name)
                axis = np.eye(3)['XYZ'.index(j['axis'])]
                if kind == 'PhysicsRevoluteJoint':
                    motion[:, :3, :3] = Rotation.from_rotvec(values[name][:, None]*axis).as_matrix()
                elif kind == 'PhysicsPrismaticJoint':
                    motion[:, :3, 3] = values[name][:, None]*axis
                else:
                    raise ValueError("unsupported robot joint: " + kind)
            poses[j['child']] = poses[j['parent']] @ j['T0'] @ motion @ np.linalg.inv(j['T1'])
            chains[j['child']] = [*chains[j['parent']], j]
            ordered.append(j)
            pending.remove(j)
    used = {j['name'] for j in ordered if j['kind'] != 'PhysicsFixedJoint'}
    if used != set(names):
        raise ValueError("measured DOFs and USD spanning tree disagree")
    root_step = np.linalg.norm(np.diff(roots[:, :3, 3], axis=0), axis=1)
    root_angle = Rotation.from_matrix(
        roots[:-1, :3, :3].transpose(0, 2, 1) @ roots[1:, :3, :3]).magnitude()
    centers, radii, links = [], [], []
    for link, radius in link_radii.items():
        if link not in poses:
            raise ValueError("robot geometry absent from FK: " + link)
        chain = chains[link]
        offset = np.asarray(link_offsets.get(link, np.zeros(3)))
        downstream = float(radius) + np.linalg.norm(offset)
        step = np.zeros(n-1)
        for j in reversed(chain):
            downstream += np.linalg.norm(j['T1'][:3, 3])
            if j['kind'] == 'PhysicsRevoluteJoint':
                step += np.abs(np.diff(values[j['name']])) * downstream
            elif j['kind'] == 'PhysicsPrismaticJoint':
                step += np.abs(np.diff(values[j['name']]))
                downstream += np.max(np.abs(values[j['name']]))
            downstream += np.linalg.norm(j['T0'][:3, 3])
        step += root_step + root_angle * (downstream + np.linalg.norm(root_offset[:3, 3]))
        padding = np.maximum(np.r_[step, 0], np.r_[0, step])
        centers.append(poses[link][:, :3, :3] @ offset + poses[link][:, :3, 3])
        radii.append(radius + CLEARANCE_M + padding)
        links.append(link)
    if not links:
        raise ValueError("robot has no bounded visible links")
    return np.stack(centers, axis=1), np.stack(radii, axis=1), links


def measured_roots(state, frames):
    series = []
    for key, width in [('robot_root_xyz_m', 3), ('robot_root_wxyz', 4)]:
        value = np.asarray(state.get(key + '_per_frame', state.get(key)), dtype=float)
        if value.shape == (width,):
            value = np.repeat(value[None], frames, axis=0)
        if value.shape != (frames, width) or not np.isfinite(value).all():
            raise ValueError('invalid measured root series: ' + key)
        series.append(value)
    return np.stack([matrix(p, q) for p, q in zip(*series)])


class RobotSweep:
    def __init__(self, stage, state):
        from pxr import Usd, UsdGeom, UsdPhysics
        from visual_scene_geometry import visible_mesh_triangles

        root = stage.GetPrimAtPath('/World/Robot')
        if not root or not root.IsActive():
            raise ValueError("missing active robot")
        joints = []
        for prim in Usd.PrimRange(root):
            if not prim.IsA(UsdPhysics.Joint) or prim.IsA(UsdPhysics.SphericalJoint):
                continue  # passive loop closure is already in measured follower DOFs
            j = UsdPhysics.Joint(prim)
            parent, child = j.GetBody0Rel().GetTargets(), j.GetBody1Rel().GetTargets()
            if not parent or not child:
                continue
            row = dict(name=prim.GetName(), parent=str(parent[0]), child=str(child[0]),
                       kind=prim.GetTypeName(), axis=prim.GetAttribute('physics:axis').Get() or 'Z')
            for end in (0, 1):
                pos = prim.GetAttribute(f'physics:localPos{end}').Get()
                q = prim.GetAttribute(f'physics:localRot{end}').Get()
                row[f'T{end}'] = matrix(pos, [q.GetReal(), *q.GetImaginary()])
            joints.append(row)
        links = {j['parent'] for j in joints} | {j['child'] for j in joints}
        bases = links - {j['child'] for j in joints}
        if len(bases) != 1:
            raise ValueError("robot requires one FK root")
        base = bases.pop()
        robot_world = np.asarray(UsdGeom.Xformable(root).ComputeLocalToWorldTransform(0)).T
        base_world = np.asarray(UsdGeom.Xformable(stage.GetPrimAtPath(base)).ComputeLocalToWorldTransform(0)).T
        local_root = np.linalg.inv(robot_world) @ base_world
        link_radii, link_offsets = {}, {}
        for link in sorted(links):
            points = visible_mesh_triangles(stage, link).reshape(-1, 3)
            if not len(points):
                continue
            world = np.asarray(UsdGeom.Xformable(stage.GetPrimAtPath(link)).ComputeLocalToWorldTransform(0)).T
            if not np.allclose(world[:3, :3].T @ world[:3, :3], np.eye(3), atol=1e-5):
                raise ValueError("non-rigid robot link transform")
            local = (np.c_[points, np.ones(len(points))] @ np.linalg.inv(world).T)[:, :3]
            offset = (local.min(axis=0) + local.max(axis=0)) / 2
            link_offsets[link] = offset
            link_radii[link] = float(np.max(np.linalg.norm(local - offset, axis=1)))
        # Every visible mesh must belong to a bounded link; never silently omit a body.
        for prim in Usd.PrimRange(root):
            if prim.IsA(UsdGeom.Mesh) and UsdGeom.Imageable(prim).ComputeVisibility() != 'invisible':
                path = str(prim.GetPath())
                if not any(path.startswith(link+'/') for link in link_radii):
                    raise ValueError("unbounded robot mesh: " + path)
        positions = np.asarray(state['robot_joint_position_rad_or_m'], dtype=float)
        roots = measured_roots(state, len(positions))
        self.centers, self.radii, self.links = forward_spheres(
            joints, roots, (base, local_root), positions,
            np.asarray(state['robot_dof_names']).astype(str).tolist(), link_radii, link_offsets)
        self.link_offsets = link_offsets
        self.summary = dict(method='USD_joint_FK_visible_link_spheres_with_joint_interpolation_bound',
            frames=len(positions), links=len(self.links), clearance_m=CLEARANCE_M,
            scope='base_torso_head_both_arms_and_all_gripper_links',
            physical_collision_rollout=False)

    def check(self, bounds):
        margin, (frame, link) = sphere_box_margin(bounds, self.centers, self.radii)
        return dict(passed=margin > 0, minimum_clearance_margin_m=margin,
                    nearest_frame=frame, nearest_robot_link=self.links[link])
