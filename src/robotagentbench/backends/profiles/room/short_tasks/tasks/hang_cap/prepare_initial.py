"""Task-owned continuous reset selection in an already composed random room.

Read-only FK and conservative collider checks, NOT native initial acceptance.
Only apply_target authors a reset transform, on the caller's episode layer.
"""
import hashlib
import itertools
import math
from pathlib import Path

import numpy as np

import json
import base64
import zlib
from . import range_initial_state
from . import sofa_domain
from short_tasks.tasks.retrieve_from_drawer.prepare_initial import fk, corners, world_matrix, apply_target

ROBOT = '/World/Robot'
SOFA = '/World/Assets/Sofa'


class NoFeasibleInitial(ValueError):
    """Geometry-only rejection eligible for seeded random-layout retry."""


def target_row(manifest):
    rows = [r for r in manifest['objects'] if r.get('source') == 'sofa'
            and r.get('asset_id', r.get('id')) == 'baseball_cap']
    if len(rows) != 1:
        raise ValueError('Exactly one scene-owned sofa cap required')
    return rows[0]


def sofa_context(stage, manifest):
    """Rigid-fit actual mesh vertices; detect child transforms and scale drift."""
    from pxr import Usd, UsdGeom, Gf
    root = Path(__file__).resolve().parents[3]
    baseline_path = Path(manifest['source_layout']).parent/'ID_01_LIVING_STRAIGHT.usdc'
    baseline = Usd.Stage.Open(str(baseline_path))
    def vertices(scene):
        cache = UsdGeom.XformCache()
        out = {}
        for p in Usd.PrimRange(scene.GetPrimAtPath(SOFA)):
            if p.IsA(UsdGeom.Mesh):
                v = np.asarray(p.GetAttribute('points').Get())
                if len(v):
                    out[str(p.GetPath())] = (np.c_[v,np.ones(len(v))] @ np.asarray(cache.GetLocalToWorldTransform(p)))[:,:3]
        return out
    aa,bb = vertices(baseline),vertices(stage)
    if not aa or set(aa) != set(bb) or any(aa[k].shape != bb[k].shape for k in aa):
        raise ValueError('Sofa differs from canonical placement geometry')
    a,b = (np.concatenate([d[k] for k in sorted(d)]) for d in (aa,bb))
    ac,bc = a.mean(0),b.mean(0)
    u,_,vt = np.linalg.svd((a-ac).T@(b-bc)); rotation = u@vt
    delta = np.eye(4);delta[:3,:3] = rotation;delta[3,:3] = bc-ac@rotation
    if np.linalg.det(rotation)<0 or not np.allclose(rotation[2],[0,0,1],atol=1e-7) or np.max(np.abs(a@rotation+delta[3,:3]-b))>1e-5:
        raise ValueError('Sofa geometry is not a rigid layout transfer')
    source_frame = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(baseline.GetPrimAtPath(SOFA)))
    folder = root/'openend/tasks/scene_requirements/placement_sources'
    policy_path = folder/'sofa-placement-policy-v1.json'
    structure_path = folder/'sofa-structural-obstacles-v2.json'
    policy = json.loads(policy_path.read_text()); structure = json.loads(structure_path.read_text())
    if hashlib.sha256(baseline_path.read_bytes()).hexdigest() != structure['source_sha256']:
        raise ValueError('Canonical sofa obstacle evidence changed')
    return source_frame@delta, np.linalg.inv(delta), policy, structure, [baseline_path,policy_path,structure_path]


def fit_target(world_points, inverse_delta, policy, structure):
    # Same canonical coordinates, padding and exclusions as the approved JS.
    v = (np.c_[world_points,np.ones(len(world_points))]@inverse_delta)[:,:3]
    uv = np.c_[-v[:,1]-.8710, v[:,0]+.3503563938]
    lo,hi = uv.min(0),uv.max(0)
    if not sofa_domain.footprint_inside(uv, policy, structure):
        return 'target_outside_middle_two_cushions_or_backrest_clearance'
    margin,half = policy['edge_margin_m'],policy['reference_approximate_width_m']/2
    if np.any(lo < [-half+margin,margin]) or np.any(hi > [half-margin,policy['reference_depth_m']-margin]):
        return 'target_outside_approved_seats'
    pad = policy['ordinary_object_padding_xy_m']
    for obstacle in structure['obstacles']:
        b = obstacle['bounds']
        if not (hi[0]+pad<=b[0] or lo[0]-pad>=b[2] or hi[1]+pad<=b[1] or lo[1]-pad>=b[3]):
            return 'target_sofa_structural_keepout:'+obstacle['id']
    return None


def geometry_frames(stage, body_path, current, scale):
    """Align the unchanged library body to the source's metric body frame."""
    from pxr import UsdGeom
    path = Path(__file__).with_name('source_geometry_frames.json')
    data = json.loads(path.read_text())
    if data['schema'] != 'short_tasks.hang_cap.geometry_frames.v1':
        raise ValueError('Wrong cap geometry bank')
    reference = np.frombuffer(zlib.decompress(base64.b64decode(data['reference_vertices_zlib_base64'])),
                              dtype='<f8').reshape(data['reference_shape'])
    prim = stage.GetPrimAtPath(body_path+data['mesh_suffix'])
    if not prim or not prim.IsA(UsdGeom.Mesh):
        raise ValueError('Source/library cap topology differs')
    vertices = np.asarray(prim.GetAttribute('points').Get())
    local = np.array(UsdGeom.XformCache().GetLocalToWorldTransform(prim)*current.GetInverse())
    a = (np.c_[vertices,np.ones(len(vertices))]@local)[:,:3]*scale
    if a.shape != reference.shape:
        raise ValueError('Source/library cap vertex count differs')
    ac,bc = a.mean(0),reference.mean(0);u,_,vt = np.linalg.svd((a-ac).T@(reference-bc))
    rotation = u@vt; translation = bc-ac@rotation
    error = float(np.max(np.abs(a@rotation+translation-reference)))
    if np.linalg.det(rotation)<0 or error>1e-5:
        raise ValueError('Cap geometry cannot be aligned by a rigid transform')
    alignment = np.eye(4);alignment[:3,:3]=rotation;alignment[3,:3]=translation
    rows = {r['source_id']:r for r in data['rows']}
    if set(rows) != {r['source_id'] for r in range_initial_state.read_samples()}:
        raise ValueError('Incomplete cap source geometry coverage')
    return alignment,rows,path,error


def choose(stage, manifest, *, seed, count=100):
    from pxr import Usd, UsdGeom, UsdPhysics
    from scipy.spatial.transform import Rotation
    from arena_api.robot_profiles import FOLLOWERS
    from openend.tasks.cloth_clear.scene import bounds, points
    from openend.tasks.scene_requirements.source_geometry import floor_paths
    from openend.tasks.scene_requirements.station_scene_checks import intersects, triangles_hit_box
    from openend.tasks.layout_repair_v1.spawn_v2 import inside
    row = target_row(manifest)
    bodies = [str(p.GetPath()) for p in Usd.PrimRange(stage.GetPrimAtPath(row['path']))
              if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    if len(bodies) != 1:
        raise ValueError('Exactly one cap rigid body required')
    body_path = bodies[0]
    cache = UsdGeom.XformCache()
    sofa, inverse_delta, policy, structure, geometry_files = sofa_context(stage, manifest)
    current = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(body_path))
    scale = [current.GetRow3(i).GetLength() for i in range(3)]
    alignment, geometry_rows, geometry_bank, mesh_error = geometry_frames(stage,body_path,current,scale)
    local_points = points(stage, row['path'], current)
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
    domain=sofa_domain.domain_local(sofa,inverse_delta,policy,structure)
    station,station_offsets=sofa_domain.robot_station_domain(sofa,inverse_delta)
    for proposal in range_initial_state.sample_candidates(seed, sofa, count=count, target_local_bounds=domain,
                                                          fixed_station=station,station_offset_bounds=station_offsets):
        from pxr import Gf
        correction = alignment@np.asarray(geometry_rows[proposal['geometry_reference_source_id']]['geometry_delta'])
        scaled = np.diag([*scale,1.])@correction
        transform = Gf.Matrix4d(*scaled.ravel().tolist())*world_matrix(proposal,[1.,1.,1.])
        actual = (np.c_[local_points, np.ones(len(local_points))] @ np.array(transform))[:, :3]
        b = np.array([actual.min(0), actual.max(0)])
        reason = fit_target(actual, inverse_delta, policy, structure)
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
        from short_tasks.tasks.retrieve_from_drawer import prepare_initial as geometry
        from short_tasks.tasks.retrieve_from_drawer import initial_state as pose_helpers, range_initial_state as rotation_helpers
        references = [Path(__file__), Path(sofa_domain.__file__), sofa_domain.CURRENT_STATION_SOURCE, range_initial_state.SOURCE, Path(range_initial_state.__file__),
                      Path(geometry.__file__),Path(pose_helpers.__file__),Path(rotation_helpers.__file__),geometry_bank,*geometry_files]
        return dict(position=[x,y,0.], yaw_rad=yaw, target=dict(paths=[row['path']], source='sofa', reference_files=list(map(str,references))),
            method='middle_two_cushions_centered_station_range_actual_room_fk', candidate=proposal,
            robot_station_policy='sofa_center_front78to90_lateral17p5_yaw5_v5',
            placement_policy=sofa_domain.POLICY,
            approved_footprint_uv_bounds_m=sofa_domain.usable_bounds(policy,structure).tolist(),
            target_path=row['path'], target_body_path=body_path, target_world_matrix=list(map(list, transform)),
            target_bounds_world=b.tolist(), targets_rad=targets,
            source_library_mesh_fit_max_error_m=mesh_error,
            target_pose_frame='source metric body; mapped to unchanged library body using geometry receipt',
            candidates_checked=proposal['candidate_index']+1, rejections=rejections,
            target_fit_checked=True, robot_room_geometry_checked=True,
            robot_self_collision_validated=False, target_support_contact_validated=False,
            native_physics_validated=False, physics_certified=False,
            shared_preparer_connected=True,
            reference_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in references})
    raise NoFeasibleInitial('Cap range candidates exhausted without widening: '+str(rejections))


def refresh_manifest(stage, manifest, receipt):
    """Keep placement provenance honest after replacing the composer's pose."""
    from openend.tasks.cloth_clear.scene import bounds
    row = target_row(manifest)
    old = {k: row.pop(k) for k in list(row) if k.startswith('source_') or k == 'rejected_source_poses'}
    row['composition_initial_pose_history'] = old
    box = bounds(stage, row['path'])
    row['visible_bounds_world_m'] = box.tolist()
    row['polygon'] = [[box[0,0],box[0,1]], [box[1,0],box[0,1]],
                      [box[1,0],box[1,1]], [box[0,0],box[1,1]]]
    row['source_pose_policy'] = receipt['candidate']['sampling']
    row['source_bounds_world_m'] = receipt['target_bounds_world']
    row['native_physics_validated'] = False
    manifest['cap_range_initialization'] = receipt
