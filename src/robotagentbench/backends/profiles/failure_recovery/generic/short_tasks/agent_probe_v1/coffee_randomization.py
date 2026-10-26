"""Approved reset-only coffee variation, after the Openend workspace placement.

No runtime pose edits, scoring changes, rescaling or physics overrides. A seed
has one proposal; a rejected proposal is retained, never silently resampled.
"""
from copy import deepcopy
import hashlib
import itertools
import json
import math
from pathlib import Path
import random

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STYLES = ('blue_compact', 'cream_classic', 'green_tall')
POLICY = 'coffee_local_initial_random_v1'


def sample(seed):
    if type(seed) is not int or seed < 0:
        raise ValueError('A non-negative integer seed is required')
    rng = random.Random(f'{POLICY}:{seed}')
    return dict(policy=POLICY, seed=seed, cup_style=rng.choice(STYLES),
        cup_offset_workspace_m=[rng.uniform(-.03, .03) for _ in range(2)],
        cup_handle_from_robot_deg=rng.uniform(-90., 90.),
        bag_offset_workspace_m=[rng.uniform(-.03, .03) for _ in range(2)],
        bag_yaw_delta_deg=rng.uniform(-15., 15.), resampled=False)


def workspace_offset(front, offset):
    fx, fy = front
    norm = math.hypot(fx, fy)
    if not math.isfinite(norm) or norm < 1e-8:
        raise ValueError('Workspace front is invalid')
    fx, fy = fx/norm, fy/norm
    forward, left = offset
    return (forward*fx-left*fy, forward*fy+left*fx, 0.)


def handle_yaw_delta(handle_direction, robot_direction, offset_deg):
    if any(not math.isfinite(v) for v in (*handle_direction, *robot_direction, offset_deg)):
        raise ValueError('Non-finite handle direction')
    if min(math.hypot(*handle_direction), math.hypot(*robot_direction)) < 1e-8:
        raise ValueError('Undefined handle or robot direction')
    if not -90 <= offset_deg <= 90:
        raise ValueError('Handle outside the approved front semicircle')
    current = math.atan2(handle_direction[1], handle_direction[0])
    target = math.atan2(robot_direction[1], robot_direction[0]) + math.radians(offset_deg)
    return math.degrees(math.atan2(math.sin(target-current), math.cos(target-current)))


def measured_handle(stage, cup_root):
    """Measure in the rigid cup frame before rotating; a world AABB is not an axis."""
    from pxr import Usd, UsdGeom
    root = stage.GetPrimAtPath(cup_root)
    handle = stage.GetPrimAtPath(cup_root+'/Visual/Props/CupHandle')
    if not handle:
        raise ValueError('Cup lacks its measured handle geometry')
    box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render'])
    center = box.ComputeRelativeBound(handle, root).ComputeAlignedRange().GetMidpoint()
    matrix = UsdGeom.XformCache().GetLocalToWorldTransform(root)
    origin = matrix.ExtractTranslation()
    direction = matrix.Transform(center)-origin
    return origin, (float(direction[0]), float(direction[1]))


def collision_vertices(stage, root):
    import numpy as np
    from pxr import Usd, UsdGeom, UsdPhysics
    from evaluation.tasks.coffee_machine.isaac_evidence import collider_points
    cache = UsdGeom.XformCache()
    rows = []
    for p in Usd.PrimRange(stage.GetPrimAtPath(root)):
        if p.HasAPI(UsdPhysics.CollisionAPI) and p.GetAttribute('physics:collisionEnabled').Get() is not False:
            points, sphere = collider_points(p, cache.GetLocalToWorldTransform(p))
            if sphere is not None:
                center, radius = sphere
                points = center + radius*np.array(list(itertools.product((-1., 1.), repeat=3)))
            rows.extend(points)
    if not rows:
        raise ValueError('Missing enabled collision geometry: '+root)
    return np.asarray(rows)


def geometry_gate(stage, manifest):
    import numpy as np
    from scipy.spatial import ConvexHull
    from pxr import Usd, UsdGeom
    from openend.tasks.scene_requirements.station_scene_checks import check, EPS
    from evaluation.tasks.coffee_machine.openended_composite_v1.scene_binding import ReadOnlySceneBinding
    binding = ReadOnlySceneBinding(stage, manifest)
    roles = binding.resolved['wrappers']
    from openend.tasks.scene_requirements.surface_registry import manifest_surfaces
    support = manifest['supports']['coffee']['path']
    surfaces = manifest_surfaces(stage, manifest)
    vertices = {role: collision_vertices(stage, path) for role, path in roles.items()}
    failures = []
    # Conservative convex footprints, including handle collision spheres.
    for a, b in itertools.combinations(vertices, 2):
        va, vb = vertices[a], vertices[b]
        if min(va[:, 2].max(), vb[:, 2].max()) <= max(va[:, 2].min(), vb[:, 2].min()) + EPS:
            continue
        pa, pb = va[:, :2], vb[:, :2]
        axes = np.vstack([ConvexHull(p).equations[:, :2] for p in (pa, pb)])
        separated = any(min((pa@axis).max(), (pb@axis).max()) <=
                        max((pa@axis).min(), (pb@axis).min()) + EPS for axis in axes)
        if not separated:
            failures.append(a+'/'+b+' conservative collision footprints overlap')
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render'])
    def bounds(path):
        box = bbox.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange()
        return np.array([box.GetMin(), box.GetMax()])
    layout = dict(surfaces=surfaces, floors=manifest['overview']['floors'],
                  blockers=manifest['overview'].get('blockers', []))
    room = check(stage, [dict(id='coffee', members=list(roles.values())+
        [binding.roles['beans']], support=support, support_members=list(roles.values()))], bounds, layout=layout)
    snapshot = binding.geometry_snapshot()
    if snapshot['beans_bound'] != 100 or snapshot['beans_geometrically_in_bag'] != 100:
        failures.append('100 beans must remain in the bag')
    if not snapshot['placement']['outside_bay']:
        failures.append('Cup must initially be outside the receiving bay')
    return dict(passed=not failures and room['passed'], failures=failures,
                room_clearance=room, snapshot=snapshot,
                scope='conservative static geometry only; native settling still required')


def apply(manifest, initial_base_xyyaw, output, seed):
    """Return a new manifest and receipt; never modify the input scene layer."""
    import numpy as np
    from pxr import Gf, Sdf, Usd, UsdGeom
    from evaluation.tasks.coffee_machine.openended_composite_v1.scene_binding import ReadOnlySceneBinding
    from evaluation.tasks.air_fryer.continuous_openended_v1.native import physical_signature
    from short_tasks.tools.prepare_scene import checked_dependencies, sha
    from short_tasks.scene import verify_manifest
    output = Path(output).resolve()
    from short_tasks.tasks.robot_pose_alignment_v1.temporary_outputs import permits
    if (not output.is_relative_to(HERE) and not permits(output)) or output.exists():
        raise ValueError('Fresh randomization output inside this task required')
    if manifest.get('initial_randomization'):
        raise ValueError('Cannot apply initial randomization twice')
    verify_manifest(manifest)
    proposal = sample(seed)
    output.mkdir(parents=True)
    report = dict(proposal=proposal, status='authoring', native_accepted=False,
                  physics_parameters_changed=False, scoring_changed=False)
    result = deepcopy(manifest)
    try:
        source = Usd.Stage.Open(manifest['scene'])
        binding = ReadOnlySceneBinding(source, manifest)
        wrappers = binding.resolved['wrappers']
        cache = UsdGeom.XformCache()
        before = {role: cache.GetLocalToWorldTransform(source.GetPrimAtPath(path))
                  for role, path in wrappers.items()}
        old_min = collision_vertices(source, wrappers['cup'])[:, 2].min()
        scene = output/'scene.usda'
        stage = Usd.Stage.CreateNew(str(scene))
        stage.GetRootLayer().subLayerPaths = [manifest['scene']]
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.)
        stage.SetDefaultPrim(stage.GetPrimAtPath('/World'))
        style = proposal['cup_style']
        entry = ROOT/'all_assets'/('coffee_cup_'+style)/'asset.usda'
        cup_root = binding.roles['cup']
        stage.OverridePrim(cup_root).GetReferences().SetReferences([Sdf.Reference(str(entry))])
        for row in result['objects']:
            if row['path'] == wrappers['cup']:
                row.update(id='coffee_cup_'+style, entry=str(entry), cup_style=style)
            if row['path'] in (wrappers['cup'], wrappers['bag'], binding.roles['beans']):
                row['initial_transform_preserved'] = False
        front = manifest['supports']['coffee']['front']
        cup_offset = list(workspace_offset(front, proposal['cup_offset_workspace_m']))
        # Preserve the approved bottom-to-surface relation across actual cup sizes.
        cup_offset[2] = float(old_min-collision_vertices(stage, wrappers['cup'])[:, 2].min())
        cup_origin, h = measured_handle(stage, cup_root)
        toward = np.asarray(initial_base_xyyaw[:2])-np.asarray(cup_origin)[:2]-np.asarray(cup_offset)[:2]
        yaw = handle_yaw_delta(h, toward, proposal['cup_handle_from_robot_deg'])
        def delta(origin, offset, degrees):
            o = Gf.Vec3d(*map(float, origin))
            return (Gf.Matrix4d(1).SetTranslate(-o)*
                    Gf.Matrix4d(1).SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), degrees))*
                    Gf.Matrix4d(1).SetTranslate(o+Gf.Vec3d(*map(float, offset))))
        def move(path, transform):
            prim = stage.GetPrimAtPath(path)
            xc = UsdGeom.XformCache()
            world, parent = xc.GetLocalToWorldTransform(prim), xc.GetParentToWorldTransform(prim)
            UsdGeom.Xformable(stage.OverridePrim(path)).MakeMatrixXform().Set(world*transform*parent.GetInverse())
        move(wrappers['cup'], delta(cup_origin, cup_offset, yaw))
        bag_origin = before['bag'].ExtractTranslation()
        bag_offset = workspace_offset(front, proposal['bag_offset_workspace_m'])
        bag_delta = delta(bag_origin, bag_offset, proposal['bag_yaw_delta_deg'])
        move(wrappers['bag'], bag_delta)
        move(binding.roles['beans'], bag_delta)
        actual_origin, actual_handle = measured_handle(stage, cup_root)
        actual_toward = np.asarray(initial_base_xyyaw[:2])-np.asarray(actual_origin)[:2]
        angle = math.degrees(math.atan2(
            actual_toward[0]*actual_handle[1]-actual_toward[1]*actual_handle[0],
            float(np.dot(actual_toward, actual_handle))))
        if abs(angle-proposal['cup_handle_from_robot_deg']) > 1e-6:
            raise ValueError('Measured handle angle differs from the approved sample')
        signatures = {}
        for row in result['objects']:
            if row['id'] == 'coffee_beans':
                continue
            asset = Usd.Stage.Open(row['entry'])
            same = physical_signature(stage, row['path']+'/Asset') == physical_signature(asset, str(asset.GetDefaultPrim().GetPath()))
            if not same:
                raise ValueError('Physical signature changed: '+row['id'])
            signatures[row['id']] = 'matches_current_all_assets'
        stage.GetRootLayer().Save()
        gate = geometry_gate(stage, result)
        report.update(geometry=gate, asset_signatures=signatures,
            cup_handle_measured_before_xy=list(map(float, h)), cup_yaw_delta_deg=yaw,
            cup_handle_measured_from_robot_deg=angle,
            cup_offset_world_m=cup_offset, bag_offset_world_m=list(bag_offset),
            bottom_height_preserved=True, beans_move_with_bag=True,
            machine_transform_unchanged=(UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(wrappers['machine'])) == before['machine']))
        if not gate['passed']:
            raise ValueError('Random initial geometry rejected; see randomization.json')
        layers, assets, special = checked_dependencies(scene)
        hashes = dict(manifest['source_hashes'])
        hashes.update({l.realPath: sha(l.realPath) for l in layers if l.realPath})
        hashes.update({str(p): sha(p) for p in assets if Path(p).is_file()})
        hashes[str(scene)] = sha(scene)
        hashes[str(Path(__file__).resolve())] = sha(__file__)
        verify_manifest(manifest)
        result.update(scene=str(scene), scene_sha256=sha(scene), source_hashes=hashes,
            source_initial_states_preserved=False, initial_randomization=proposal,
            preparation_mode='openend_workspace_then_local_initial_random',
            status='randomized_static_not_native_accepted')
        report.update(status='static_passed_not_native_accepted', source_files_unchanged=True,
                      dependency_exceptions=special)
        (output/'manifest.json').write_text(json.dumps(result, indent=2)+'\n')
        return result, report
    except BaseException as exc:
        report.update(status='rejected', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        (output/'randomization.json').write_text(json.dumps(report, indent=2)+'\n')
