"""Task-owned continuous reset selection in an already composed random room.

Read-only FK and conservative collider checks, NOT native initial acceptance.
Only apply_target authors a reset transform, on the caller's episode layer.
"""
import hashlib
import itertools
import math
from pathlib import Path

import numpy as np

from . import initial_state, range_initial_state

ROBOT = '/World/Robot'
DRAWER = '/World/Assets/CoffeeTable/black_box_1'


class NoFeasibleInitial(ValueError):
    """A geometry rejection eligible for random-layout retry, not a code error."""


def target_row(manifest):
    rows = [r for r in manifest['objects'] if r.get('source') == 'coffee_table.drawer'
            and r.get('asset_id', r.get('id')) in ('wallet', 'glasses_case')]
    if len(rows) != 1 or not manifest.get('openend_drawer'):
        raise ValueError('Exactly one scene-owned drawer target and slide required')
    return rows[0]


def corners(box):
    return np.array(list(itertools.product(*zip(*box))))


def landing_within_bottom(landing, bottom):
    """Conservative XY proposal gate; no epsilon or new physical margin."""
    return bool(np.all(landing[0,:2] >= bottom[0,:2])
                and np.all(landing[1,:2] <= bottom[1,:2]))


def fk(stage, targets, base):
    """Column-vector joint-frame FK; do not author or alter the articulation."""
    from pxr import Usd, UsdPhysics
    from scipy.spatial.transform import Rotation
    joints = []
    seen = set()
    def frame(p, i):
        m = np.eye(4)
        q = p.GetAttribute('physics:localRot'+str(i)).Get()
        m[:3, :3] = Rotation.from_quat([*q.GetImaginary(), q.GetReal()]).as_matrix()
        m[:3, 3] = p.GetAttribute('physics:localPos'+str(i)).Get()
        return m
    for p in Usd.PrimRange(stage.GetPrimAtPath(ROBOT)):
        if not any(p.IsA(t) for t in (UsdPhysics.RevoluteJoint, UsdPhysics.PrismaticJoint, UsdPhysics.FixedJoint)):
            continue
        a, b = [p.GetRelationship('physics:body'+str(i)).GetTargets() for i in (0, 1)]
        if len(a) != 1 or len(b) != 1:
            continue
        motion = np.eye(4)
        if not p.IsA(UsdPhysics.FixedJoint):
            name = p.GetName()
            if name not in targets:
                raise ValueError('Missing reset coordinate: '+name)
            seen.add(name)
            v = targets[name]
            unit = 1 if p.IsA(UsdPhysics.PrismaticJoint) else math.pi/180
            lo, hi = [p.GetAttribute('physics:'+k+'Limit').Get() for k in ('lower', 'upper')]
            if lo is None or hi is None or not lo*unit <= v <= hi*unit:
                raise ValueError('Reset outside unchanged joint limits: '+name)
            axis = np.eye(3)['XYZ'.index(p.GetAttribute('physics:axis').Get())]
            if p.IsA(UsdPhysics.PrismaticJoint):
                motion[:3, 3] = axis*v
            else:
                motion[:3, :3] = Rotation.from_rotvec(axis*v).as_matrix()
        joints.append((str(a[0]), str(b[0]), frame(p, 0) @ motion @ np.linalg.inv(frame(p, 1))))
    if seen != set(targets):
        raise ValueError('Reset joint set differs from room articulation')
    links = {ROBOT+'/base_link': base}
    for _ in joints:
        before = len(links)
        for a, b, delta in joints:
            if a in links and b not in links:
                links[b] = links[a] @ delta
        if len(links) == before:
            break
    rigid = {str(p.GetPath()) for p in Usd.PrimRange(stage.GetPrimAtPath(ROBOT))
             if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    if rigid-set(links):
        raise ValueError('Unresolved robot FK links: '+str(rigid-set(links)))
    return links


def world_matrix(proposal, scale):
    from pxr import Gf
    v = proposal['target_xyz_xyzw']
    m = Gf.Matrix4d().SetRotate(Gf.Quatd(v[6], Gf.Vec3d(*v[3:6])))
    m.SetTranslateOnly(Gf.Vec3d(*v[:3]))
    return Gf.Matrix4d().SetScale(Gf.Vec3d(*scale))*m


def choose(stage, manifest, *, seed, count=100):
    from pxr import Usd, UsdGeom, UsdPhysics
    from scipy.spatial.transform import Rotation
    from arena_api.robot_profiles import FOLLOWERS
    from evaluation.tasks.find_object.isaac_evidence import body_colliders, collider_points
    from openend.tasks.cloth_clear.scene import bounds, points
    from openend.tasks.scene_requirements.source_geometry import floor_paths
    from openend.tasks.scene_requirements.station_scene_checks import intersects, triangles_hit_box
    from openend.tasks.layout_repair_v1.spawn_v2 import inside
    row = target_row(manifest)
    asset = row.get('asset_id', row['id'])
    body_path = row.get('rigid_body_path', row['path']+'/Asset')
    cache = UsdGeom.XformCache()
    frame = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(DRAWER)).RemoveScaleShear()
    q = frame.ExtractRotationQuat()
    pose = [*frame.ExtractTranslation(), *q.GetImaginary(), q.GetReal()]
    container = bounds(stage, DRAWER, frame)
    current = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(body_path))
    scale = [current.GetRow3(i).GetLength() for i in range(3)]
    local_points = points(stage, row['path'], current)
    # Visible bounds alone admit an upright, tilted wallet too close to a wall:
    # its footprint expands while settling. Check its conservative collision
    # box against the actual bottom footprint, not the drawer's outer walls.
    # This rejects proposals; it never clamps poses or changes sampled ranges.
    collision_points = np.vstack([
        collider_points(p, cache.GetLocalToWorldTransform(p)*current.GetInverse())
        for p in Usd.PrimRange(stage.GetPrimAtPath(body_path))
        if p.IsActive() and p.HasAPI(UsdPhysics.CollisionAPI)
        and p.GetAttribute('physics:collisionEnabled').Get() is not False])
    collision_box = corners(np.array([collision_points.min(0), collision_points.max(0)]))
    bottoms = [v for name, v in body_colliders(stage, DRAWER).items() if 'Bottom' in name]
    if not bottoms:
        raise ValueError('Drawer bottom collision geometry required for landing fit')
    bottom_points = np.vstack(bottoms)
    bottom_bounds = np.array([bottom_points.min(0), bottom_points.max(0)])
    floors, polygons = floor_paths(stage), manifest.get('overview', {}).get('floors', [])
    if not floors or not polygons:
        raise ValueError('Physical floor and measured room polygons required')
    bbox = UsdGeom.BBoxCache(0, ['default', 'render', 'proxy', 'guide'], False, True)
    robot_parts, obstacles = [], []
    for p in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if not p.IsActive() or not p.HasAPI(UsdPhysics.CollisionAPI) or p.GetAttribute('physics:collisionEnabled').Get() is False:
            continue
        path = str(p.GetPath())
        r = bbox.ComputeWorldBound(p).ComputeAlignedRange()
        box = np.array([r.GetMin(), r.GetMax()])
        if not np.isfinite(box).all() or np.any(box[0] > box[1]):
            raise ValueError('Invalid collider bounds: '+path)
        if path.startswith(ROBOT+'/'):
            body = p
            while body and not body.IsPseudoRoot() and not body.HasAPI(UsdPhysics.RigidBodyAPI):
                body = body.GetParent()
            if not body or body.IsPseudoRoot():
                raise ValueError('Robot collider without rigid body')
            local = np.c_[corners(box), np.ones(8)] @ np.linalg.inv(np.array(cache.GetLocalToWorldTransform(body)))
            robot_parts.append((path, str(body.GetPath()), local))
            continue
        if any(path == f or path.startswith(f+'/') for f in floors+[row['path']]):
            continue
        tri = None
        if p.IsA(UsdGeom.Mesh):
            vs = p.GetAttribute('points').Get()
            ids = list(p.GetAttribute('faceVertexIndices').Get() or [])
            faces, offset = [], 0
            for n in p.GetAttribute('faceVertexCounts').Get() or []:
                face = ids[offset:offset+n]; offset += n
                faces.extend((face[0], face[i], face[i+1]) for i in range(1, n-1))
            if vs is not None and len(vs) and faces:
                vs = (np.c_[np.array(vs), np.ones(len(vs))] @ np.array(cache.GetLocalToWorldTransform(p)))[:, :3]
                tri = vs[np.array(faces)]
        obstacles.append((path, box, tri))
    if not robot_parts:
        raise ValueError('No robot collider geometry')
    rejections = []
    for proposal in range_initial_state.sample_candidates(asset, seed, pose, count=count):
        transform = world_matrix(proposal, scale)
        actual = (np.c_[local_points, np.ones(len(local_points))] @ np.array(transform*frame.GetInverse()))[:, :3]
        b = np.array([actual.min(0), actual.max(0)])
        reason = None
        if np.any(b[0] < container[0]) or np.any(b[1, :2] > container[1, :2]):
            reason = 'target_outside_drawer_bounds'
        collision_envelope = (np.c_[collision_box, np.ones(len(collision_box))]
                              @ np.array(transform*frame.GetInverse()))[:, :3]
        landing = np.array([collision_envelope.min(0), collision_envelope.max(0)])
        if reason is None and not landing_within_bottom(landing, bottom_bounds):
            reason = 'target_collision_envelope_outside_drawer_bottom'
        targets = dict(proposal['targets_rad'])
        # Match the existing native owner's paired-gripper reset exactly.
        for inner, outer in FOLLOWERS.items():
            targets[inner] = -targets[outer]
        x, y, yaw = proposal['initial_base_xyyaw']
        base = np.eye(4)
        base[:3, :3] = Rotation.from_euler('z', yaw).as_matrix()
        base[:3, 3] = [x, y, 0.]
        links = fk(stage, targets, base)
        boxes = []
        if reason is None:
            for path, body, local in robot_parts:
                v = (local @ links[body].T)[:, :3]
                box = np.array([v.min(0), v.max(0)]); boxes.append(box)
                perimeter = [(a, b) for a in np.linspace(box[0,0],box[1,0],11) for b in (box[0,1],box[1,1])]
                perimeter += [(a,b) for b in np.linspace(box[0,1],box[1,1],11) for a in (box[0,0],box[1,0])]
                if not all(any(inside(v, p) for p in polygons) for v in perimeter):
                    reason = 'robot_outside_floor:'+path; break
                hit = next((p for p,b,t in obstacles if intersects(box,b) and (t is None or triangles_hit_box(t,box))), None)
                if hit:
                    reason = 'robot_room_overlap:'+path+':'+hit; break
            # The target has moved, so check it separately from old obstacles.
            world_points = (np.c_[local_points, np.ones(len(local_points))] @ np.array(transform))[:, :3]
            target_box = np.array([world_points.min(0), world_points.max(0)])
            if reason is None and any(intersects(box, target_box) for box in boxes):
                reason = 'robot_target_overlap'
        if reason:
            rejections.append(dict(candidate_index=proposal['candidate_index'], reason=reason))
            continue
        references = [Path(__file__), Path(initial_state.__file__), initial_state.SOURCE, Path(range_initial_state.__file__)]
        return dict(position=[x,y,0.], yaw_rad=yaw, target=dict(paths=[DRAWER], source='coffee_table.drawer', reference_files=list(map(str,references))),
            method='continuous_drawer_ranges_actual_room_fk', candidate=proposal,
            target_path=row['path'], target_body_path=body_path, target_world_matrix=list(map(list, transform)),
            target_bounds_in_drawer=b.tolist(), targets_rad=targets,
            target_collision_envelope_in_drawer=landing.tolist(),
            drawer_bottom_bounds=bottom_bounds.tolist(),
            landing_fit_policy='collision_box_projection_inside_actual_drawer_bottom',
            candidates_checked=proposal['candidate_index']+1, rejections=rejections,
            target_fit_checked=True, robot_room_geometry_checked=True,
            robot_self_collision_validated=False, target_wall_contact_validated=False,
            native_physics_validated=False, physics_certified=False,
            shared_preparer_connected=True,
            reference_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in references})
    raise NoFeasibleInitial('Drawer range candidates exhausted without widening: '+str(rejections))


def apply_target(stage, receipt):
    from pxr import Gf, UsdGeom
    prim = stage.GetPrimAtPath(receipt['target_body_path'])
    if not prim:
        raise ValueError('Selected target disappeared')
    parent = UsdGeom.XformCache().GetLocalToWorldTransform(prim.GetParent())
    world = Gf.Matrix4d(*np.array(receipt['target_world_matrix']).ravel().tolist())
    UsdGeom.Xformable(prim).MakeMatrixXform().Set(world*parent.GetInverse())


def refresh_manifest(stage, manifest, receipt):
    """Preserve the composer's provenance separately from this episode reset."""
    from openend.tasks.cloth_clear.scene import bounds
    row = target_row(manifest)
    old = {k: row.pop(k) for k in list(row)
           if k.startswith('source_') or k == 'rejected_source_poses'}
    row['composition_initial_pose_history'] = old
    box = bounds(stage, row['path'])
    row['visible_bounds_world_m'] = box.tolist()
    row['polygon'] = [[box[0, 0], box[0, 1]], [box[1, 0], box[0, 1]],
                      [box[1, 0], box[1, 1]], [box[0, 0], box[1, 1]]]
    row['source_pose_policy'] = 'continuous_measured_ranges_v1'
    row['drawer_body_frame_bounds_m'] = receipt['target_bounds_in_drawer']
    row['native_physics_validated'] = False
    manifest['drawer_range_initialization'] = receipt
