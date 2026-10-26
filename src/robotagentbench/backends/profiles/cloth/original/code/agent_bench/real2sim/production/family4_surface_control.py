"""Regenerate surface-pick commands for a physically randomized robot base.

An optional command/state seed supplies a demonstrated path, not task identity
or success. USD joints define FK. Both training and evaluation use the existing
task-owned Family-4 draw. Every candidate still needs a fresh physical capture.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from real2sim.production.robot_visual_clearance import matrix
from real2sim.production.task_randomization._family4_initial_state import sample
from real2sim.teleop.datagen.init_pose_noise import apply_noise


def task_binding(task_key):
    """Read semantics, station and object from the sole benchmark catalog."""
    from real2sim.production.benchmark_eval_registry import CATALOG
    from real2sim.teleop.robot_positions import robot_position
    from real2sim.teleop.task_registry import REGISTRY
    if task_key not in {'benchmark_4_3', 'benchmark_4_8', 'benchmark_4_9'}:
        raise ValueError('surface controller is not validated for this task')
    task = CATALOG[task_key.removeprefix('benchmark_').replace('_', '.')]['task']
    runtime = task['runtime']['success']['parameters']['runtime_task_id']
    entry = REGISTRY[runtime]
    target = entry['root'].rstrip('/') + '/' + entry['prim']
    station = robot_position(task['runtime']['environment']['robot_station']['key'])
    return runtime, target, station


def same_base_pose(left, right):
    left, right = np.asarray(left), np.asarray(right)
    return bool(left.shape == (3,) and right.shape == (3,)
                and np.isfinite(left).all() and np.isfinite(right).all()
                and np.allclose(left[:2], right[:2], rtol=0, atol=2e-6)
                and abs((left[2] - right[2] + np.pi) % (2*np.pi) - np.pi) < 2e-6)


def command_names(hand):
    if hand not in ('l', 'r'):
        raise ValueError('hand must be l or r')
    start = 20 if hand == 'l' else 60
    return [f'idx{start + i}_arm_{hand}_joint{i}' for i in range(1, 8)]


class ArmKinematics:
    """Read-only USD joint chain; no simulator or cached grasp result."""

    def __init__(self, robot_asset, names, hand):
        from pxr import Usd, UsdGeom, UsdPhysics
        stage = Usd.Stage.CreateInMemory()
        root = UsdGeom.Xform.Define(stage, '/World/Robot').GetPrim()
        root.GetReferences().AddReference(str(Path(robot_asset).resolve()))
        rows = []
        for prim in Usd.PrimRange(root):
            if not prim.IsA(UsdPhysics.Joint) or prim.IsA(UsdPhysics.SphericalJoint):
                continue
            joint = UsdPhysics.Joint(prim)
            parent, child = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
            if not parent or not child:
                continue
            row = dict(name=prim.GetName(), parent=str(parent[0]), child=str(child[0]),
                       kind=prim.GetTypeName(), axis=prim.GetAttribute('physics:axis').Get() or 'Z')
            for end in (0, 1):
                pos = prim.GetAttribute(f'physics:localPos{end}').Get()
                quat = prim.GetAttribute(f'physics:localRot{end}').Get()
                row[f'T{end}'] = matrix(pos, [quat.GetReal(), *quat.GetImaginary()])
            lower, upper = (prim.GetAttribute('physics:' + key).Get() for key in ('lowerLimit', 'upperLimit'))
            if lower is not None and upper is not None:
                scale = np.pi / 180 if row['kind'] == 'PhysicsRevoluteJoint' else 1
                row['limits'] = (float(lower) * scale, float(upper) * scale)
            rows.append(row)
        links = {r['parent'] for r in rows} | {r['child'] for r in rows}
        bases = links - {r['child'] for r in rows}
        if len(bases) != 1 or len(names) != len(set(names)):
            raise ValueError('ambiguous robot joint graph or joint names')
        base = bases.pop()
        self.root_offset = np.asarray(UsdGeom.Xformable(stage.GetPrimAtPath(base)).ComputeLocalToWorldTransform(0)).T
        node = f'/World/Robot/arm_{hand}_end_link'
        chain, visited = [], set()
        while node != base:
            parents = [r for r in rows if r['child'] == node]
            if len(parents) != 1 or node in visited:
                raise ValueError('missing, cyclic or ambiguous arm chain')
            visited.add(node)
            row = dict(parents[0])
            row['index'] = names.index(row['name']) if row['kind'] != 'PhysicsFixedJoint' else None
            row['invT1'] = np.linalg.inv(row['T1'])
            chain.insert(0, row)
            node = row['parent']
        self.chain = chain
        self.indices = [names.index(n) for n in command_names(hand)]
        self.limits = np.array([next(r['limits'] for r in chain if r['index'] == i) for i in self.indices])

    def forward(self, q, roots):
        pose = roots @ self.root_offset
        for row in self.chain:
            motion = np.broadcast_to(np.eye(4), (len(q), 4, 4)).copy()
            axis = np.eye(3)['XYZ'.index(row['axis'])]
            if row['kind'] == 'PhysicsRevoluteJoint':
                motion[:, :3, :3] = Rotation.from_rotvec(q[:, row['index'], None] * axis).as_matrix()
            elif row['kind'] == 'PhysicsPrismaticJoint':
                motion[:, :3, 3] = q[:, row['index'], None] * axis
            elif row['kind'] != 'PhysicsFixedJoint':
                raise ValueError('unsupported arm joint type')
            pose = pose @ row['T0'] @ motion @ row['invT1']
        return pose


def solve_retarget(model, q, old_roots, new_roots):
    q = np.asarray(q, dtype=float)
    if q.ndim != 2 or not np.isfinite(q).all():
        raise ValueError('finite joint command matrix required')
    for roots in (old_roots, new_roots):
        if np.shape(roots) != (len(q), 4, 4) or not np.isfinite(roots).all():
            raise ValueError('finite per-frame root transforms required')
    goals = model.forward(q, old_roots)
    x = q.copy()
    for iteration in range(30):
        pose = model.forward(x, new_roots)
        dp = goals[:, :3, 3] - pose[:, :3, 3]
        dr = Rotation.from_matrix(goals[:, :3, :3] @ pose[:, :3, :3].transpose(0, 2, 1)).as_rotvec()
        if np.max(np.linalg.norm(dp, axis=1)) < 1e-4 and np.max(np.linalg.norm(dr, axis=1)) < .001:
            break
        columns = []
        for idx in model.indices:
            perturbed = x.copy()
            perturbed[:, idx] += 1e-4
            moved = model.forward(perturbed, new_roots)
            columns.append(np.c_[(moved[:, :3, 3] - pose[:, :3, 3]) / 1e-4,
                Rotation.from_matrix(moved[:, :3, :3] @ pose[:, :3, :3].transpose(0, 2, 1)).as_rotvec() * .15 / 1e-4])
        jac = np.stack(columns, axis=2)
        transpose = jac.transpose(0, 2, 1)
        delta = (transpose @ np.linalg.solve(jac @ transpose + 1e-5 * np.eye(6), np.c_[dp, dr * .15][:, :, None]))[:, :, 0]
        x[:, model.indices] = np.clip(x[:, model.indices] + np.clip(delta, -.08, .08), model.limits[:, 0], model.limits[:, 1])
    pose = model.forward(x, new_roots)
    position = np.linalg.norm(goals[:, :3, 3] - pose[:, :3, 3], axis=1)
    angle = Rotation.from_matrix(goals[:, :3, :3] @ pose[:, :3, :3].transpose(0, 2, 1)).magnitude()
    if position.max() >= 1e-4 or angle.max() >= .001:
        raise ValueError(f'unreachable base draw: position={position.max():.6g} m angle={angle.max():.6g} rad')
    if np.any(x[:, model.indices] < model.limits[:, 0]) or np.any(x[:, model.indices] > model.limits[:, 1]):
        raise ValueError('retargeted seed violates joint limits')
    return x, dict(iterations=iteration + 1, maximum_position_error_m=float(position.max()),
                   maximum_orientation_error_rad=float(angle.max()), physical_success=False)


def prepare(*, action, state, robot_asset, seed, task_key='benchmark_4_3', hand='r'):
    """Preserve a bound object's world grasp pose at its randomized station."""
    runtime, target, station = task_binding(task_key)
    if str(np.asarray(state['target_prim']).item()) != target:
        raise ValueError('seed target does not match the benchmark task')
    if hand not in ('l', 'r'):
        raise ValueError('hand must be l or r')
    names = np.asarray(state['robot_dof_names']).astype(str).tolist()
    n = len(action['arm'])
    q = np.asarray(state['robot_joint_position_rad_or_m'], dtype=float).copy()
    if q.shape != (n, len(names)) or np.shape(action['arm']) != (n, 14):
        raise ValueError('seed action/state shapes disagree')
    indices = [names.index(name) for h in ('l', 'r') for name in command_names(h)]
    q[:, indices] = action['arm']
    for name, channel, column in [('idx01_body_joint1', 'waist', 1), ('idx02_body_joint2', 'waist', 0),
                                   ('idx11_head_joint1', 'head', 0), ('idx12_head_joint2', 'head', 1)]:
        q[:, names.index(name)] = action[channel][:, column]
    canonical = np.asarray(station.xy_m, dtype=float)
    old_root = matrix(state['robot_root_xyz_m'], state['robot_root_wxyz'])
    if not np.allclose(old_root[:2, 3], canonical, atol=2e-6, rtol=0):
        raise ValueError('seed is not at the current task station')
    if not np.allclose(old_root[:3, :3], Rotation.from_euler('z', station.yaw_deg, degrees=True).as_matrix(), atol=2e-6, rtol=0):
        raise ValueError('seed task station yaw is incompatible')
    draw = sample(seed)
    new_root = old_root.copy()
    new_root[:2, 3] = canonical + draw['base_dxy_m']
    new_root[:3, :3] = Rotation.from_euler('z', draw['base_dyaw_deg'], degrees=True).as_matrix() @ old_root[:3, :3]
    model = ArmKinematics(robot_asset, names, hand)
    solved, audit = solve_retarget(model, q, np.repeat(old_root[None], n, axis=0), np.repeat(new_root[None], n, axis=0))
    closed = np.flatnonzero(np.asarray(action[hand + 'grip']) >= .5)
    if not len(closed):
        raise ValueError('seed has no grasp phase')
    noisy = apply_noise(solved[:, indices], draw['arm_noise_seed'], np.deg2rad(draw['arm_noise_deg']), 40, int(closed[0]))
    output = {k: np.asarray(v).copy() for k, v in action.items()}
    output['arm'] = noisy['arm']
    for side, columns in [('l', slice(0, 7)), ('r', slice(7, 14))]:
        arm_model = model if side == hand else ArmKinematics(robot_asset, names, side)
        if (np.any(output['arm'][:, columns] < arm_model.limits[:, 0])
                or np.any(output['arm'][:, columns] > arm_model.limits[:, 1])):
            raise ValueError('initial perturbation exceeds arm joint limits')
    # The commanded base channel must describe the newly authored world frame.
    output['base_xyyaw'] = np.repeat(np.array([[*new_root[:2, 3], np.arctan2(new_root[1, 0], new_root[0, 0])]]), n, axis=0)
    audit.update(schema='real2sim.family4_surface_control.v1', task_key=task_key,
                 runtime_task_id=runtime, target_prim=target, hand=hand, draw=draw,
                 canonical_base_xy_m=canonical.tolist(), base_xyyaw=output['base_xyyaw'][0].tolist(),
                 initial_arm_delta_rad=noisy['delta_rad'].tolist(), settle_frames=noisy['settle_frames'],
                 noise_jerk_ratio=noisy['jerk_ratio'], new_physics_required=True)
    return output, audit


def apply_base(scene: Path, plan: dict):
    """Author only the robot root in an open, task-owned working House scene."""
    from pxr import Gf, Usd, UsdGeom

    scene = Path(scene).resolve()
    if scene.name != 'scene_room.usda' or not scene.with_suffix('.usda.session.json').is_file():
        raise ValueError('an open working House scene is required')
    draw = plan['draw']
    if draw != sample(draw['seed']):
        raise ValueError('plan does not match the task-owned Family-4 draw')
    _, _, station = task_binding(plan.get('task_key'))
    expected = np.array([*(np.asarray(station.xy_m) + draw['base_dxy_m']),
                         np.deg2rad(station.yaw_deg + draw['base_dyaw_deg'])])
    if not same_base_pose(plan['base_xyyaw'], expected):
        raise ValueError('plan base does not match its declared draw')
    stage = Usd.Stage.Open(str(scene))
    robot = stage.GetPrimAtPath('/World/Robot')
    before = np.asarray(UsdGeom.Xformable(robot).ComputeLocalToWorldTransform(0)).T
    if not np.allclose(before[:2, 3], station.xy_m, rtol=0, atol=2e-6):
        raise ValueError('initializer did not use the canonical task station')
    if not np.allclose(before[:3, :3], Rotation.from_euler('z', station.yaw_deg, degrees=True).as_matrix(), rtol=0, atol=2e-6):
        raise ValueError('initializer used the wrong station yaw')
    def nonrobot_transforms():
        return {str(p.GetPath()): np.asarray(UsdGeom.Xformable(p).ComputeLocalToWorldTransform(0))
                for p in stage.Traverse() if p.IsA(UsdGeom.Xformable)
                and not str(p.GetPath()).startswith('/World/Robot')}
    others = nonrobot_transforms()
    ops = UsdGeom.Xformable(robot).GetOrderedXformOps()
    translations = [op for op in ops if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]
    if len(translations) != 1 or not robot.GetAttribute('xformOp:orient'):
        raise ValueError('unsupported robot root transform operations')
    translations[0].Set(Gf.Vec3d(float(expected[0]), float(expected[1]), float(before[2, 3])))
    robot.GetAttribute('xformOp:orient').Set(Gf.Quatf(float(np.cos(expected[2] / 2)), 0, 0, float(np.sin(expected[2] / 2))))
    after = np.asarray(UsdGeom.Xformable(robot).ComputeLocalToWorldTransform(0)).T
    after_others = nonrobot_transforms()
    if others.keys() != after_others.keys() or any(not np.array_equal(v, after_others[k]) for k, v in others.items()):
        raise ValueError('a non-robot transform changed')
    observed = np.array([*after[:2, 3], np.arctan2(after[1, 0], after[0, 0])])
    if not same_base_pose(observed, expected):
        raise ValueError('authored base differs from controller plan')
    stage.GetRootLayer().Save()
    return dict(task_key=plan['task_key'], planned_base_xyyaw=expected.tolist(),
                authored_base_xyyaw=observed.tolist(), base_delta_xy_m=draw['base_dxy_m'],
                base_yaw_delta_deg=draw['base_dyaw_deg'], nonrobot_transforms_unchanged=True,
                physical_readback_required=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', type=Path, required=True)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--robot-asset', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--task', choices=('4.3', '4.8', '4.9'), default='4.3')
    parser.add_argument('--hand', choices=('l', 'r'), default='r')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    task_key = 'benchmark_' + args.task.replace('.', '_')
    if args.output.exists():
        raise ValueError('output exists')
    from real2sim.production.git_provenance import source_revision
    repo = Path(__file__).resolve().parents[2]
    revision = source_revision([
        'real2sim/production/family4_surface_control.py',
        'real2sim/production/task_randomization/_family4_initial_state.py',
        f'real2sim/production/task_randomization/{task_key}/inject_robot_initial_state.py',
        f'real2sim/production/task_randomization/{task_key}/policy.json',
        'real2sim/production/benchmark_eval_registry.py',
        'real2sim/production/house_task_environment.py',
        'real2sim/teleop/datagen/init_pose_noise.py',
        'real2sim/teleop/robot_positions.py',
    ], repo=repo)
    if revision['worktree_dirty']:
        raise ValueError('commit controller source before generating a physics plan')
    with np.load(args.action, allow_pickle=False) as a, np.load(args.state, allow_pickle=False) as s:
        output, audit = prepare(action=dict(a), state=dict(s), robot_asset=args.robot_asset,
                               seed=args.seed, task_key=task_key, hand=args.hand)
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / 'action.npz', **output)
    audit.update(seed_action=str(args.action.resolve()), seed_state=str(args.state.resolve()),
                 controller_source_revision=revision)
    (args.output / 'CONTROL_PLAN.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
