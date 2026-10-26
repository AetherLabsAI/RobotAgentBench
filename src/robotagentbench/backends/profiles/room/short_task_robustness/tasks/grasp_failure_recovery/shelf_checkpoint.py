"""Choose a nearby empty-grasp checkpoint, not the end of the withdrawal.

Fresh joint-frame FK is required after the original robot jitter. Conservative
collider boxes only reject proposals; they never certify native contact safety.
No new distance, joint, contact or scoring threshold is introduced here.
"""
import math


def target_hand(target, robot):
    """Robot-local +Y is left; choose by the actual reset base, not image X."""
    delta=[float(a)-float(b) for a,b in zip(target['position_m'],robot['position_m'])]
    q=list(map(float,robot['quaternion_wxyz']))
    if len(delta)!=3 or len(q)!=4 or not all(map(math.isfinite,delta+q)):
        raise ValueError('Finite target/base poses required')
    norm=math.sqrt(sum(v*v for v in q))
    if norm==0.:raise ValueError('Degenerate base quaternion')
    w,x,y,z=[v/norm for v in q]
    lateral=2*(x*y-w*z)*delta[0]+(1-2*(x*x+z*z))*delta[1]+2*(y*z+w*x)*delta[2]
    if lateral==0.:raise ValueError('Target exactly on robot midline has no left/right side')
    hand='l' if lateral>0. else 'r'
    return hand,dict(policy='target_side_in_actual_robot_base_frame.v1',hand=hand,
                     target_local_y_m=lateral,left_axis='+Y',right_axis='-Y')


def close_hand(joints, hand):
    """User-requested empty closed posture, not a measured bottle-grasp opening.

    Native G1 closes at zero; zero for the complete right linkage satisfies its
    joint-anchor closure. Only reset coordinates change, never joint parameters.
    """
    from arena_api.robot_profiles import GRIPPER_INTERNAL
    if hand not in ('l','r'):raise ValueError('Explicit left/right hand required')
    names = [n for n in GRIPPER_INTERNAL if '_gripper_'+hand+'_' in n]
    if len(names) != 8 or not set(names) <= set(joints):
        raise ValueError('Complete native active-hand linkage required')
    closed = dict(joints)
    before = {n:closed[n] for n in names}
    closed.update({n:0. for n in names})
    return closed, dict(policy='user_requested_empty_active_gripper_native_zero.v2',hand=hand,
                        source_coordinates=before, reset_coordinates={n:closed[n] for n in names},
                        other_gripper_unchanged=True, drive_parameters_changed=False)


def close_right(joints):
    """Compatibility helper for old right-only diagnostic callers."""
    return close_hand(joints,'r')


def choose(rows):
    """Nearest non-overlapping source checkpoint; deterministic frame tie-break."""
    valid = [r for r in rows if r['source_lifted'] and r['closed']
             and not r['overlaps'] and r.get('reset_error') is None]
    if not valid:
        raise ValueError('No clear source-backed empty-grasp checkpoint; no seed retry')
    return min(valid, key=lambda r: (r['hand_target_gap_m'], r['frame']))


def gap(a, b):
    return math.sqrt(sum(max(a[0][i]-b[1][i], b[0][i]-a[1][i], 0.)**2
                         for i in range(3)))


class Clearance:
    """Read-only local collider boxes transformed by fresh full-34 FK."""
    def __init__(self, stage, target, objects, hand='r'):
        import numpy as np
        from pxr import Usd, UsdGeom, UsdPhysics
        from arena_api import scene_obstacles
        if hand not in ('l','r'):raise ValueError('Explicit left/right clearance hand required')
        self.np, self.stage, self.hand = np, stage, hand
        self.base = np.array(UsdGeom.Xformable(stage.GetPrimAtPath(
            '/World/Robot/base_link')).ComputeLocalToWorldTransform(0)).T
        def enabled(p):
            return p.HasAPI(UsdPhysics.CollisionAPI) and p.GetAttribute(
                'physics:collisionEnabled').Get() is not False

        def corners(p):
            # Same extent policy as the source shelf Geometry reader: analytic
            # shapes can have stale authored extents, unlike their actual schema.
            shape = UsdGeom.Boundable(p)
            extent = UsdGeom.Boundable.ComputeExtentFromPlugins(shape, Usd.TimeCode.Default())
            if extent is None:
                extent = shape.GetExtentAttr().Get()
            if extent is None or len(extent) != 2:
                raise ValueError('Missing collider extent: '+str(p.GetPath()))
            lo, hi = extent
            return np.array([[lo[0] if i & 1 == 0 else hi[0],
                              lo[1] if i & 2 == 0 else hi[1],
                              lo[2] if i & 4 == 0 else hi[2], 1.] for i in range(8)])

        def bounds(p):
            transform = np.array(UsdGeom.Xformable(p).ComputeLocalToWorldTransform(0)).T
            points = (transform @ corners(p).T).T[:, :3]
            return points.min(axis=0), points.max(axis=0)

        self.obstacles = []
        for root in sorted(set(objects) | {target}):
            for p in Usd.PrimRange(stage.GetPrimAtPath(root), Usd.TraverseInstanceProxies()):
                if enabled(p):
                    self.obstacles.append((str(p.GetPath()), bounds(p), root == target))
        # Read the existing public policy's inventory without applying a policy
        # or changing collision settings. The full shelf mesh AABB is not a
        # useful empty-space test, so shelf mesh overlaps remain native checks.
        for row in scene_obstacles.inventory(stage):
            if row['enabled'] and row['category'] == 'shelf' and row['type'] == 'Cube':
                self.obstacles.append((row['path'], bounds(stage.GetPrimAtPath(row['path'])), False))
        if not any(is_target for _, _, is_target in self.obstacles):
            raise ValueError('Missing target collider')
        self.robot = []
        for p in Usd.PrimRange(stage.GetPrimAtPath('/World/Robot')):
            if not enabled(p):
                continue
            body = p
            while body and not body.HasAPI(UsdPhysics.RigidBodyAPI):
                body = body.GetParent()
            if not body:
                raise ValueError('Robot collider lacks rigid ancestor')
            local = np.array(UsdGeom.Xformable(p).ComputeLocalToWorldTransform(0)).T
            parent = np.array(UsdGeom.Xformable(body).ComputeLocalToWorldTransform(0)).T
            self.robot.append((str(p.GetPath()), str(body.GetPath()),
                               (np.linalg.inv(parent) @ local @ corners(p).T).T))
        if not self.robot:
            raise ValueError('No robot colliders')

    def evaluate(self, joints):
        from .render_initials import forward_frames
        np = self.np
        frames = forward_frames(self.stage, joints, self.base)
        overlaps, hand_gaps = [], []
        for path, body, corners in self.robot:
            points = (frames[body] @ corners.T).T[:, :3]
            box = points.min(axis=0), points.max(axis=0)
            for obstacle, other, is_target in self.obstacles:
                distance = gap(box, other)
                if distance == 0.:
                    overlaps.append([path, obstacle])
                if is_target and '/gripper_'+self.hand+'_' in body:
                    hand_gaps.append(distance)
        if not hand_gaps:
            raise ValueError('No active-hand target clearance evidence')
        return dict(hand_target_gap_m=min(hand_gaps), overlaps=overlaps)


def scan(stage, snapshot, recording, phases, seed):
    import numpy as np
    from evaluation.tasks.shelf_pick import initial_state, episode_randomization, source_episodes
    from .common import audit_reset
    start = next(p['first_frame'] for p in phases['phases'] if p['name'] == 'lift')
    end = next(p['last_frame'] for p in phases['phases'] if p['name'] == 'withdraw')
    hand=phases['hand']
    primary='idx41_gripper_l_outer_joint1' if hand=='l' else 'idx81_gripper_r_outer_joint1'
    geometry = Clearance(stage, snapshot['target_path'], [i['prim_path'] for i in snapshot['items']],hand)
    rows = []
    with np.load(recording, allow_pickle=False) as data:
        index = data['rigid_body_paths'].tolist().index(str(data['target_prim']))
        for frame in range(start, end):
            joints = dict(zip(data['robot_dof_names'].tolist(),
                              map(float, data['robot_joint_position_rad_or_m'][frame])))
            changed, _ = episode_randomization.perturb(dict(snapshot, joint_positions=joints), seed)
            closed, _ = close_hand(changed['joint_positions'],hand)
            row = dict(frame=frame, source_lifted=bool(data['prop_xyz_m'][frame,index,2] >
                       data['prop_xyz_m'][0,index,2]),
                       closed=closed[primary] == 0.,hand=hand,
                       source_grasp_opening_rad=joints[primary])
            try:
                accepted, _ = initial_state.clamp_recorded_joints(stage, closed)
                audit_reset(stage, dict(positions=accepted, hold_targets={
                    name:accepted[name] for name in source_episodes.CONTROLLED_JOINTS}))
                row.update(geometry.evaluate(accepted))
            except ValueError as error:
                row.update(reset_error=str(error), overlaps=[], hand_target_gap_m=None)
            rows.append(row)
    try:
        selected, error = choose(rows), None
    except ValueError as failure:
        selected, error = None, str(failure)
    return dict(policy='nearest_clear_source_lift_or_withdraw_checkpoint.v2',hand=hand,
                selected=selected, selection_error=error, candidates=rows, seed_retried=False,
                source_frames=[start, end], fresh_fk=True,
                conservative_boxes_only=True, shelf_mesh_contact_checked=False,
                native_acceptance_required=True)
