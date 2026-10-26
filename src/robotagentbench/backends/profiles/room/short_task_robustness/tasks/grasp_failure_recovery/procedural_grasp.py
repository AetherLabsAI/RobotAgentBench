"""CPU-only target-relative empty grasp IK. Never reads source motions.

Joint anchors/limits and collision geometry are read from the actual USD asset.
All non-active-arm coordinates are frozen. No simulation or physical setter.
"""
import math
from copy import deepcopy


def sample_request(variant, seed, target_bounds, base_rotation, hand, config):
    """Seeded signed lateral gap; orientation comes from fixed configuration."""
    from .catalog import generator
    if variant not in ('shelf_drink', 'coffee_bag') or hand not in ('l', 'r'):
        raise ValueError('Explicit supported task and active hand required')
    if config['gap_range_m'] != [.02, .05] or config['orientation_jitter'] is not False:
        raise ValueError('Only the user-confirmed empty-grasp configuration is supported')
    rng = generator(variant, seed, 'target-relative-empty-grasp-v1')
    return dict(gap_m=rng.uniform(*config['gap_range_m']),
                miss_sign=rng.choice((-1, 1)), hand=hand,
                miss_axis='robot_local_y', orientation_jitter=False,
                target_bounds=deepcopy(target_bounds), seed=seed,
                configuration_version=config['schema'])


class Kinematics:
    def __init__(self, stage, hand, initial):
        import numpy as np
        from scipy.spatial.transform import Rotation
        from pxr import Usd, UsdGeom, UsdPhysics
        self.np, self.Rotation, self.stage, self.hand = np, Rotation, stage, hand
        self.initial = dict(initial)
        self.names = [f'idx{2 if hand == "l" else 6}{i}_arm_{hand}_joint{i}' for i in range(1, 8)]
        self.tool = f'/World/Robot/gripper_{hand}_center_link'
        self.base = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(
            stage.GetPrimAtPath('/World/Robot/base_link'))).T
        self.edges, self.bounds = [], {}
        def anchor(p, i):
            m = np.eye(4)
            q = p.GetAttribute('physics:localRot'+str(i)).Get()
            m[:3,:3] = Rotation.from_quat([*q.GetImaginary(), q.GetReal()]).as_matrix()
            m[:3,3] = p.GetAttribute('physics:localPos'+str(i)).Get()
            return m
        for p in Usd.PrimRange(stage.GetPrimAtPath('/World/Robot')):
            fixed = p.IsA(UsdPhysics.FixedJoint)
            angular = p.IsA(UsdPhysics.RevoluteJoint)
            linear = p.IsA(UsdPhysics.PrismaticJoint)
            if not (fixed or angular or linear):
                continue
            a, b = [p.GetRelationship('physics:body'+str(i)).GetTargets() for i in (0,1)]
            if len(a) != 1 or len(b) != 1:
                continue
            axis = None if fixed else np.eye(3)['XYZ'.index(p.GetAttribute('physics:axis').Get())]
            self.edges.append((str(a[0]), str(b[0]), p.GetName(), fixed, angular,
                               axis, anchor(p,0), np.linalg.inv(anchor(p,1))))
            if p.GetName() in self.names:
                limits = [float(p.GetAttribute('physics:'+s+'Limit').Get()) for s in ('lower','upper')]
                self.bounds[p.GetName()] = [math.radians(v) for v in limits]
        if set(self.bounds) != set(self.names):
            raise ValueError('Missing native active-arm limits')
        # Topologically order the real USD tree once, not on every IK call.
        ordered, known, pending = [], {'/World/Robot/base_link'}, list(self.edges)
        while pending:
            reachable = [e for e in pending if e[0] in known and e[1] not in known]
            if not reachable:
                break
            for e in reachable:
                ordered.append(e); known.add(e[1]); pending.remove(e)
        if self.tool not in known:
            raise ValueError('TCP is not connected to the unchanged robot joint tree')
        self.edges = ordered

    def frames(self, coordinates):
        np = self.np
        frames = {'/World/Robot/base_link': self.base}
        for a,b,name,fixed,angular,axis,A,B in self.edges:
            motion = np.eye(4)
            if not fixed:
                value = coordinates[name]
                if angular:
                    motion[:3,:3] = self.Rotation.from_rotvec(axis*value).as_matrix()
                else:
                    motion[:3,3] = axis*value
            frames[b] = frames[a] @ A @ motion @ B
        return frames

    def solve(self, goal):
        from scipy.optimize import least_squares
        np = self.np
        lo = np.array([self.bounds[n][0] for n in self.names])
        hi = np.array([self.bounds[n][1] for n in self.names])
        seed = np.array([self.initial[n] for n in self.names])
        if np.any(seed < lo) or np.any(seed > hi):
            raise ValueError('Ready configuration outside actual arm limits; no clipping')
        def unpack(x):
            q = dict(self.initial)
            q.update(zip(self.names, map(float, x)))
            return q
        def residual(x):
            actual = self.frames(unpack(x))[self.tool]
            return np.r_[actual[:3,3]-goal[:3,3],
                         self.Rotation.from_matrix(goal[:3,:3].T @ actual[:3,:3]).as_rotvec()]
        # Same pose request, no object/seed redraw. Numerical limits/tolerances
        # follow the existing public pure-IK contract (200, .001m, .01rad).
        fit = least_squares(residual, seed, bounds=(lo,hi), max_nfev=200)
        q = unpack(fit.x)
        error = residual(fit.x)
        accepted = float(np.linalg.norm(error[:3])) <= .001 and float(np.linalg.norm(error[3:])) <= .01
        receipt = dict(converged=accepted, position_error_m=float(np.linalg.norm(error[:3])),
            angle_error_rad=float(np.linalg.norm(error[3:])), evaluations=fit.nfev,
            solver='scipy_least_squares_actual_USD_joint_tree', physics_backend='not_run',
            active_arm_only=True, non_active_coordinates_unchanged=all(
                q[n] == v for n,v in self.initial.items() if n not in self.names),
            joint_limits_read_from_asset=True, coordinates_clipped=False)
        return q, receipt


def collider_points(stage, path):
    import numpy as np
    from pxr import Usd, UsdGeom, UsdPhysics
    cache = UsdGeom.XformCache()
    rows = []
    for p in Usd.PrimRange(stage.GetPrimAtPath(path), Usd.TraverseInstanceProxies()):
        if not p.HasAPI(UsdPhysics.CollisionAPI) or p.GetAttribute('physics:collisionEnabled').Get() is False:
            continue
        shape = UsdGeom.Boundable(p)
        if not shape:
            continue
        extent = UsdGeom.Boundable.ComputeExtentFromPlugins(shape, Usd.TimeCode.Default())
        if extent is None:
            extent = shape.GetExtentAttr().Get()
        if extent is None or len(extent) != 2:
            raise ValueError('Missing original collider extent: '+str(p.GetPath()))
        box = np.array([[extent[0][i] if bits & (1 << i) == 0 else extent[1][i]
                         for i in range(3)] + [1.] for bits in range(8)])
        body = p
        while body and not body.HasAPI(UsdPhysics.RigidBodyAPI):
            body = body.GetParent()
        if not body:
            raise ValueError('Collider lacks a rigid body')
        world = np.asarray(cache.GetLocalToWorldTransform(p)).T
        parent = np.asarray(cache.GetLocalToWorldTransform(body)).T
        rows.append((str(p.GetPath()), str(body.GetPath()),
                     (np.linalg.inv(parent) @ world @ box.T).T,
                     (world @ box.T).T[:,:3]))
    if not rows:
        raise ValueError('No original colliders for '+path)
    return rows


def solve(stage, record, config):
    import numpy as np
    from .shelf_checkpoint import Clearance
    from .common import audit_reset
    hand, variant = record['active_hand'], record['variant']
    # This function only completes asset-native linkage coordinates; it never
    # invokes the legacy motion scanner or reads any source data.
    initialization, closure = complete_closed_reset(stage, record['robot_initialization'], hand)
    model = Kinematics(stage, hand, initialization['positions'])
    target_rows = collider_points(stage, record['target'])
    target = np.concatenate([r[3] for r in target_rows])
    cfg = config['shelf' if variant == 'shelf_drink' else 'coffee']
    rotation = model.base[:3,:3] @ np.asarray(cfg['fixed_tool_rotation_base'][hand])
    request = sample_request(variant, record['seed'],
        [target.min(0).tolist(), target.max(0).tolist()], model.base[:3,:3].tolist(), hand, config)
    initial_frames = model.frames(initialization['positions'])
    tool_inverse = np.linalg.inv(initial_frames[model.tool])
    fingertips = []
    for branch in ('inner','outer'):
        path = f'/World/Robot/gripper_{hand}_{branch}_link4'
        for _,body,local,_ in collider_points(stage,path):
            fingertips.extend((tool_inverse @ initial_frames[body] @ local.T).T[:,:3])
    fingertips = np.asarray(fingertips)
    axis = model.base[:3,1] * request['miss_sign']
    offset_points = fingertips @ rotation.T
    # Align fingertip-centre height/depth to the target; miss along robot Y.
    centre = (target.min(0)+target.max(0))/2
    translation = centre-offset_points.mean(0)
    translation += axis*(float((target @ axis).max())+request['gap_m']-
                          float(((offset_points+translation) @ axis).min()))
    goal = np.eye(4); goal[:3,:3] = rotation; goal[:3,3] = translation
    q, ik = model.solve(goal)
    frames = model.frames(q)
    actual_tool = frames[model.tool]
    actual_tips = fingertips @ actual_tool[:3,:3].T+actual_tool[:3,3]
    measured_gap = float((actual_tips @ axis).min()-(target @ axis).max())
    obstacles = record['neighbors'] if variant == 'shelf_drink' else record['ik_obstacles']
    clearance = Clearance(stage, record['target'], obstacles, hand).evaluate(q)
    # Gap remains an explicitly conservative collider-envelope diagnostic until
    # native contact and rendered-finger checks. Never call it exact mesh gap.
    receipt = dict(policy='target_relative_empty_closed_grasp.v1', request=request,
        goal_tcp_world=goal.tolist(), actual_tcp_world=actual_tool.tolist(), ik=ik,
        fingertip_envelope_signed_gap_m=measured_gap,
        gap_geometry='original_collider_extents_projected_onto_robot_lateral_axis',
        exact_mesh_surface_distance_verified=False, clearance=clearance,
        source_motion_read=False, object_relocated_for_ik=False, seed_retried=False,
        accepted_cpu=bool(ik['converged'] and .02 <= measured_gap <= .05 and not clearance['overlaps']))
    holds = dict(initialization['hold_targets'])
    for name in model.names:
        holds[name] = q[name]
    final = dict(positions=q, hold_targets=holds,
                 source='fixed ready configuration plus target-relative CPU IK; no motion frames')
    audit_reset(stage, final)
    return final, closure, receipt


def complete_closed_reset(stage, issued, hand):
    """Native asset reset plus fixed ready coordinates, close active eight only."""
    from pxr import UsdPhysics
    from arena_api.robot_profiles import NATIVE_JOINTS, FOLLOWERS, GRIPPER_INTERNAL
    from .common import audit_reset
    positions = {}
    inherited = []
    for name in NATIVE_JOINTS:
        p = stage.GetPrimAtPath('/World/Robot/joints/'+name)
        angular = p.IsA(UsdPhysics.RevoluteJoint)
        if name in issued['positions']:
            positions[name] = float(issued['positions'][name])
        else:
            raw = p.GetAttribute('state:'+('angular' if angular else 'linear')+':physics:position').Get()
            if raw is None:
                raise ValueError('Missing existing native linkage reset: '+name)
            positions[name] = math.radians(raw) if angular else float(raw)
            inherited.append(name)
    for follower, driver in FOLLOWERS.items():
        if follower not in issued['positions']:
            positions[follower] = -positions[driver]
    names = [n for n in GRIPPER_INTERNAL if '_gripper_'+hand+'_' in n]
    if len(names) != 8:
        raise ValueError('Incomplete active-hand linkage')
    before = {n:positions[n] for n in names}
    positions.update({n:0. for n in names})
    holds = {n:positions[n] for n in issued['hold_targets']}
    for name in names:
        if stage.GetPrimAtPath('/World/Robot/joints/'+name).HasAPI(UsdPhysics.DriveAPI,'angular'):
            holds[name] = positions[name]
    initialization = dict(positions=positions,hold_targets=holds)
    audit_reset(stage,initialization)
    return initialization,dict(hand=hand,source_coordinates=before,
        reset_coordinates={n:positions[n] for n in names},
        asset_reset_inherited=inherited,other_gripper_unchanged=True,
        drive_parameters_changed=False,coordinates_measured=False)
