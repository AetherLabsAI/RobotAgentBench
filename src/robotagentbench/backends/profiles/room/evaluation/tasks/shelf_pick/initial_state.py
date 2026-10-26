"""Task-owned fixed reset layer. Author activation and transforms only.

Joint coordinates are returned separately for the public RobotInitialization
hook. Historical initializers and their physics setters are never imported.
One implementation serves every shelf-pick API; the TargetSpec selects the
target instance(s), original support slab and stock layer.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from .targets import DEFAULT, SHELF_ITEMS, TIER2_ITEMS, spec as target_spec

TASK = Path(__file__).resolve().parent
REPO = TASK.parents[2]
# 4-1 identities kept as module constants for existing callers and tests.
TARGET = DEFAULT.target
SUPPORT = DEFAULT.support
ROBOT = '/World/Robot'
STOCK = DEFAULT.stock_root
SNAPSHOT = TASK/'resources'/DEFAULT.snapshot
# Per-API v2 snapshots extracted from ziche's accepted 2026-09-16 source episodes
# (user decision 2026-09-23 to rebind every shelf-pick API to that data).
NEWDATA = TASK/'resources/newdata_20260924'
# Receipt-validated House baseline (the only base tools/build_l1_scene.py accepts);
# since revision shelf_w120_aligned_20260914 it carries the 1.20 m rack.
BASELINE = REPO/'assets/house_scene_l1/demo_sim/scene_room_baseline.usda'
# 2026-09-24（用户定方案 B）：14 件货架商品改用数据方 20260916_oreo_v10 发布资产 +
# 数据方场景层的物理写法，由这一层提供。它取代了我方原先自写的五个覆盖层
# （pringles_visual_collision / snack_box_visuals / zbar_mass_properties /
#  bottle_visual_collision / shelf_pick_starbucks_physics）——那些文件保留作历史，
# 但**不再参与任何层栈**，否则会把商品悄悄换回我方旧资产。
ALIGNMENT_LAYER = REPO/'assets/house_scene_l1/demo_sim/source_release_alignment.usda'
TASK_LAYERS = (ALIGNMENT_LAYER,)
# Task scene: TASK_LAYERS composed over BASELINE. Fixed layers, replays and the
# support-scale guard all use this file（其 subLayers 与 TASK_LAYERS 必须一致，
# 由 tests/test_source_asset_alignment.py 守住）。
BASE = REPO/'assets/house_scene_l1/demo_sim/scene_room_shelf_pick.usda'
# v1: recorded L1 station plus a reconstructed single-tier neighbour grid.
# v2: one accepted episode of the 2026-09-16 ziche PICK dataset, with an explicit
# world pose for every placed item on BOTH tiers (see source_episodes.py).
SCHEMA_V1 = 'arena.shelf_pick.initial_state.v1'
SCHEMA_V2 = 'arena.shelf_pick.initial_state.v2'
STOCK_ROOTS = (SHELF_ITEMS, TIER2_ITEMS)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compose_task_layers(environment, output):
    """Write a task-owned scene that composes TASK_LAYERS above a prepared
    runs/ environment (tools/build_l1_scene.py only accepts BASELINE) and
    re-verify the shelf supports on the composed stage."""
    from pxr import Usd, UsdGeom
    from arena_randomization.l1_shelf_support_geometry import validate_shelf_supports
    environment, output = Path(environment).resolve(strict=True), Path(output).resolve()
    from arena_api.l1_runtime import is_scene_output_directory
    if output.exists() or not is_scene_output_directory(output.parent):
        raise ValueError('Use a new scene in an approved output tree for the composed task layers')
    source = Usd.Stage.Open(str(environment))
    stage = Usd.Stage.CreateNew(str(output))
    for key in ('upAxis', 'metersPerUnit', 'timeCodesPerSecond', 'framesPerSecond'):
        if source.HasAuthoredMetadata(key):stage.SetMetadata(key, source.GetMetadata(key))
    stage.GetRootLayer().subLayerPaths = [os.path.relpath(p, output.parent) for p in (*TASK_LAYERS, environment)]
    stage.SetDefaultPrim(stage.GetPrimAtPath('/World'))
    stage.GetRootLayer().Save()
    stage = Usd.Stage.Open(str(output))
    report = validate_shelf_supports(stage)
    return dict(scene=str(output), environment=str(environment),
                task_layers={str(p): digest(p) for p in TASK_LAYERS}, shelf_support_geometry=report)


def snapshot_path(spec=DEFAULT, snapshot=None):
    """Default per-API v1 snapshot, or an explicit override (v1 or v2)."""
    if snapshot is None:
        return TASK/'resources'/spec.snapshot
    path = Path(snapshot)
    return path if path.is_absolute() else TASK/'resources'/path


def load_snapshot(spec=DEFAULT, snapshot=None):
    data = json.loads(snapshot_path(spec, snapshot).read_text())
    if data.get('schema') == SCHEMA_V2:
        from .source_episodes import validate
        if spec.external_target:
            raise ValueError('Source episodes do not carry an external target')
        if data['api_id'] != spec.api_id:
            raise ValueError('Wrong initial-state snapshot')
        validate(data)
        return data
    if spec.external_target:
        # Station reuse is explicit; this is not a recorded remote shelf episode.
        if data['api_id'] != DEFAULT.api_id:
            raise ValueError('Unexpected source station for external target')
        data['api_id'] = spec.api_id
        data['target_path'] = spec.target
        data['source'] = {**data['source'], 'scope': 'robot station only; target placed by current shelf geometry'}
    if data['schema'] != SCHEMA_V1 or data['api_id'] != spec.api_id:
        raise ValueError('Wrong initial-state snapshot')
    if data.get('target_path', spec.target) != spec.target:
        raise ValueError('Snapshot target differs from the API binding')
    for name in ('target', 'robot'):
        p, q = data[name]['position_m'], data[name]['quaternion_wxyz']
        if len(p) != 3 or len(q) != 4 or not all(math.isfinite(x) for x in p+q):
            raise ValueError('Invalid pose '+name)
        if abs(sum(x*x for x in q)-1) > 1e-5:
            raise ValueError('Non-unit quaternion '+name)
    return data


def audit_layer(layer, allowed_paths):
    """Reject any unexpected spec, relationship, metadata, or physics opinion."""
    from pxr import Sdf
    violations = []
    def inspect(path):
        spec = layer.GetObjectAtPath(path)
        if isinstance(spec, Sdf.PrimSpec):
            for key in spec.ListInfoKeys():
                if key not in (('subLayers', 'subLayerOffsets', 'upAxis', 'metersPerUnit', 'defaultPrim', 'timeCodesPerSecond', 'framesPerSecond') if path == Sdf.Path.absoluteRootPath else ('specifier', 'active')):
                    violations.append(f'{path}: metadata {key}')
            if spec.HasInfo('active') and str(path) not in allowed_paths:
                violations.append(f'{path}: activation outside reset instances')
        elif isinstance(spec, Sdf.AttributeSpec):
            if str(path.GetPrimPath()) not in allowed_paths or spec.name not in ('xformOp:transform:shelf_reset', 'xformOpOrder'):
                violations.append(f'{path}: forbidden attribute')
        elif isinstance(spec, Sdf.RelationshipSpec):
            violations.append(f'{path}: forbidden relationship')
    layer.Traverse('/', inspect)
    if violations:
        raise ValueError('State-layer audit failed: '+ '; '.join(violations))


# Recorded L1 joint readings can overshoot a native USD limit by a few tenths of a
# milliradian (sensor noise on a passive gripper link); the public reset adapter
# rejects anything beyond 0.001 deg. Only such rejected readings are clamped to
# the limit (smaller overshoots pass unchanged, so existing reset contracts are
# untouched); the delta is recorded. Larger excursions are real pose problems
# and are still rejected.
CLAMP_LIMIT_RAD = 1e-3
PUBLIC_RESET_TOLERANCE_RAD = math.radians(.001)


def clamp_recorded_joints(stage, joint_positions):
    from pxr import UsdPhysics
    clamped, result = {}, dict(joint_positions)
    for name, value in joint_positions.items():
        prim = stage.GetPrimAtPath(ROBOT+'/joints/'+name)
        if not prim or not prim.IsA(UsdPhysics.RevoluteJoint): continue
        lo, hi = (math.radians(prim.GetAttribute('physics:'+s+'Limit').Get()) for s in ('lower', 'upper'))
        target = min(max(value, lo), hi)
        if abs(target-value) > PUBLIC_RESET_TOLERANCE_RAD:
            if abs(target-value) > CLAMP_LIMIT_RAD:
                raise ValueError(f'Recorded joint {name} is {abs(target-value):.4f} rad outside its native limit')
            result[name] = target; clamped[name] = dict(recorded=value, limit=target, delta_rad=target-value)
    return result, clamped


def prepare(output, *, base=BASE, stock_plan=None, spec=DEFAULT, snapshot=None):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    target, support, stock = spec.target, spec.support, spec.stock_root
    out = Path(output).resolve()
    if not out.is_relative_to(TASK/'artifacts') or out.exists():
        raise ValueError('Use a new directory under shelf_pick/artifacts/')
    source = Path(base).resolve(strict=True)
    source_hash = digest(source)
    data = load_snapshot(spec, snapshot)
    source_episode = data['schema'] == SCHEMA_V2
    if source_episode:
        # A recorded episode places items on both tiers and carries an explicit world
        # pose for each of them; nothing is re-derived from a support top.
        if stock_plan is not None:
            raise ValueError('Source-episode snapshots carry their own layout; no stock plan')
        target = data['target_path']
        stock_roots = STOCK_ROOTS
        item_paths = [item['prim_path'] for item in data['items']]
        if target in item_paths:
            raise ValueError('Target listed as its own neighbor')
    else:
        stock_roots = (stock,)
    if not source_episode and stock_plan is not None:
        data['neighbors'] = {p['prim_path'].rsplit('/',1)[-1]: dict(local_xy_m=[p['x_m'],p['y_m']], support_clearance_m=.004, yaw_deg=p['yaw_deg']) for p in stock_plan['placements']}
        if any(p['prim_path'] != stock+'/'+p['id'] or p['prim_path']==target for p in stock_plan['placements']):
            raise ValueError('Invalid stock plan identities')
    if not source_episode and any(stock+'/'+n == target for n in data['neighbors']):
        raise ValueError('Target listed as its own neighbor')
    # Acceptable targets for this episode. v1 keeps the static binding (4-8 has two
    # Sprite cans in the old scene). v2 follows the user's 2026-09-23 rule: any
    # instance of the target product counts, so only instances actually placed by
    # this source episode are acceptable.
    if source_episode:
        family = target.rsplit('/', 1)[-1].split('_copy_')[0]
        acceptable_targets = [target] + [i['prim_path'] for i in data['items']
                                         if i['prim_path'].rsplit('/', 1)[-1].split('_copy_')[0] == family]
    else:
        acceptable_targets = list(spec.targets)
    out.mkdir(parents=True)
    stage = Usd.Stage.CreateNew(str(out/'initial.usda'))
    layer = stage.GetRootLayer()
    layer.subLayerPaths = [os.path.relpath(source, out)]
    baseline = Usd.Stage.Open(str(source))
    # Stage metadata does not inherit from sublayers. Copy the existing units
    # and axis explicitly, otherwise USD silently defaults to centimetres/Y-up.
    UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.GetStageMetersPerUnit(baseline))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.GetStageUpAxis(baseline))
    stage.SetDefaultPrim(stage.GetPrimAtPath('/World'))
    for key in ('timeCodesPerSecond', 'framesPerSecond'):
        if baseline.HasAuthoredMetadata(key):stage.SetMetadata(key, baseline.GetMetadata(key))
    paths = [target, ROBOT]+(list(item_paths) if source_episode else [stock+'/'+n for n in data['neighbors']])
    def require(path):
        prim = stage.GetPrimAtPath(path)
        if not prim:
            raise ValueError('Missing existing instance '+path)
        return prim
    support_prim = require(support)
    if not support_prim.IsActive():
        raise ValueError('Support inactive in baseline')
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default','render','proxy'])
    bounds = cache.ComputeWorldBound(support_prim).ComputeAlignedRange()
    if bounds.IsEmpty():
        raise ValueError('Support has no bounds')
    reference = Usd.Stage.Open(str(BASE))
    reference_support = UsdGeom.Xformable(reference.GetPrimAtPath(support)).ComputeLocalToWorldTransform(0)
    current_support = UsdGeom.Xformable(support_prim).ComputeLocalToWorldTransform(0)
    if (Gf.Transform(reference_support).GetScale()-Gf.Transform(current_support).GetScale()).GetLength()>1e-6:
        raise ValueError('Shelf scale changed; state adapter requires a rigid layout transform')
    frame_delta = reference_support.RemoveScaleShear().GetInverse()*current_support.RemoveScaleShear()
    stock_children=[p for root in stock_roots for p in require(root).GetAllChildren()]
    stock_paths={str(p.GetPath()) for p in stock_children}
    if target not in stock_paths and not spec.external_target:
        raise ValueError('Target is not a child of the stock layer(s) '+', '.join(stock_roots))
    if source_episode and any(p not in stock_paths for p in item_paths):
        raise ValueError('Snapshot item is not stocked by the scene')
    for p in stock_children:
        p.SetActive(str(p.GetPath()) in paths)
    for path in paths:
        require(path).SetActive(True)
    # Capture physics on composed, activated instances before authoring poses.
    def physics_facts():
        return {str(p.GetPath()): dict(schemas=list(p.GetAppliedSchemas()),
            attributes={a.GetName(): str(a.Get()) for a in p.GetAttributes()
                        if a.GetName().startswith(('physics:', 'physx'))},
            relationships={r.GetName(): [str(t) for t in r.GetTargets()] for r in p.GetRelationships()
                           if r.GetName().startswith(('physics:', 'physx', 'material:binding'))})
            for path in paths for p in Usd.PrimRange(require(path))}
    before = physics_facts()
    hashes = {l.realPath: digest(l.realPath) for l in stage.GetUsedLayers()
              if l.realPath and Path(l.realPath) != out/'initial.usda'}
    def author_world(path, position, quaternion):
        prim = require(path)
        old = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
        # Preserve inherited geometry scale; root poses here are unit-scale.
        scale = Gf.Transform(old).GetScale()
        if any(abs(x-1) > 1e-5 for x in scale) and not (spec.external_target and path == target):
            raise ValueError('Unexpected instance scale '+path)
        q = Gf.Quatd(quaternion[0], Gf.Vec3d(*quaternion[1:])).GetNormalized()
        world = Gf.Matrix4d(1).SetRotate(q)
        world.SetTranslateOnly(Gf.Vec3d(*position))
        if spec.external_target and path == target:
            world = Gf.Matrix4d(1).SetScale(scale)*world
        world = world*frame_delta
        parent = UsdGeom.Xformable(prim.GetParent()).ComputeLocalToWorldTransform(0)
        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        xf.AddTransformOp(opSuffix='shelf_reset').Set(world*parent.GetInverse())
    # The v1 offsets repaired a specific recorded L1 pose; a source episode carries
    # its own accepted layout and is authored exactly as recorded.
    offset = (0., 0., 0.) if source_episode else tuple(float(x) for x in spec.target_offset_m)
    for key, path in [('target', target), ('robot', ROBOT)]:
        position = data[key]['position_m'] if key != 'target' else [a+b for a, b in zip(data[key]['position_m'], offset)]
        author_world(path, position, data[key]['quaternion_wxyz'])
    if spec.external_target:
        from pxr import Sdf
        from evaluation.tasks.tabletop_pick.inspect_initial import source_pose
        reference_path = TASK.parent/'small_table_pick/references/remote/initial_scene.usda'
        original = source_pose(Sdf.Layer.FindOrOpen(str(reference_path)), target)
        q = original.RemoveScaleShear().ExtractRotationQuat().GetNormalized()
        author_world(target, data['target']['position_m'], [q.GetReal(), *q.GetImaginary()])
        # Preserve native scale; align the actual remote footprint, not its pivot.
        def target_bounds():
            # Combining rotated nested AABBs inflates the root bound. Measure
            # the actual shape extents in world space instead of that root box.
            result = Gf.Range3d()
            for p in Usd.PrimRange(require(target)):
                shape = UsdGeom.Boundable(p)
                if not shape:
                    continue
                collision = p.HasAPI(UsdPhysics.CollisionAPI) and p.GetAttribute('physics:collisionEnabled').Get() is not False
                if not collision and UsdGeom.Imageable(p).ComputeVisibility() == 'invisible':
                    continue
                extent = shape.GetExtentAttr().Get()
                if extent is None:
                    extent = UsdGeom.Boundable.ComputeExtentFromPlugins(shape, Usd.TimeCode.Default())
                if extent is None or len(extent) != 2:
                    raise ValueError('Missing remote shape extent '+str(p.GetPath()))
                local = Gf.Range3d(Gf.Vec3d(*extent[0]), Gf.Vec3d(*extent[1]))
                transform = UsdGeom.Xformable(p).ComputeLocalToWorldTransform(0)
                for i in range(8):
                    result.UnionWith(transform.Transform(local.GetCorner(i)))
            if result.IsEmpty():
                raise ValueError('Remote has no measurable geometry')
            return result
        remote_bounds = target_bounds()
        desired = frame_delta.Transform(Gf.Vec3d(*data['target']['position_m']))
        center = remote_bounds.GetMidpoint()
        half = remote_bounds.GetSize()/2
        # Keep the full footprint inside the shelf, as the L1 4.16 contract requires.
        for i in (0, 1):
            if 2*half[i] > bounds.GetSize()[i]:
                raise ValueError('Remote cannot fit the existing shelf')
            desired[i] = min(max(desired[i], bounds.GetMin()[i]+half[i]), bounds.GetMax()[i]-half[i])
        shift = Gf.Vec3d(desired[0]-center[0], desired[1]-center[1], bounds.GetMax()[2]-remote_bounds.GetMin()[2])
        xf = UsdGeom.Xformable(require(target))
        world = xf.ComputeLocalToWorldTransform(0)
        world.SetTranslateOnly(world.ExtractTranslation()+shift)
        xf.GetOrderedXformOps()[0].Set(world*UsdGeom.Xformable(require(target).GetParent()).ComputeLocalToWorldTransform(0).GetInverse())
        placed = target_bounds()
        if any(placed.GetMin()[i] < bounds.GetMin()[i]-1e-9 or placed.GetMax()[i] > bounds.GetMax()[i]+1e-9 for i in (0,1)):
            raise ValueError('Remote footprint exceeds shelf; no silent relocation')
        hashes[str(reference_path)] = digest(reference_path)
    if source_episode:
        for item in data['items']:
            author_world(item['prim_path'], item['position_m'], item['quaternion_wxyz'])
    stock_world = UsdGeom.Xformable(reference.GetPrimAtPath(stock)).ComputeLocalToWorldTransform(0)
    ref_top=UsdGeom.BBoxCache(0,['default','render','proxy']).ComputeWorldBound(reference.GetPrimAtPath(support)).ComputeAlignedRange().GetMax()[2]
    for name, pose in ({} if source_episode else data['neighbors']).items():
        world = stock_world.Transform(Gf.Vec3d(*pose['local_xy_m'],0))
        yaw=math.radians(pose.get('yaw_deg',0.))
        q=pose.get('quaternion_wxyz')
        if q is None:
            q=[math.cos(yaw/2),0,0,math.sin(yaw/2)]
        elif abs(yaw)>1e-12:
            raise ValueError('Neighbor pose must use either yaw_deg or quaternion_wxyz: '+name)
        author_world(stock+'/'+name, [world[0],world[1],ref_top+pose['support_clearance_m']], q)
    for path in [target]+(list(item_paths) if source_episode else [stock+'/'+n for n in data['neighbors']]):
        body = UsdPhysics.RigidBodyAPI(require(path))
        if not body or body.GetKinematicEnabledAttr().Get() or not body.GetRigidBodyEnabledAttr().Get():
            raise ValueError('Existing stock must already be dynamic: '+path)
    errors = [str(e) for p in stage.Traverse() for e in p.GetPrimIndex().localErrors]
    if errors:
        raise ValueError('Composed scene has unresolved references: '+str(errors[:5]))
    audit_layer(layer, set(paths)|stock_paths)
    if before != physics_facts():
        raise ValueError('Physics changed while authoring state')
    if any(digest(p) != h for p,h in hashes.items()) or digest(source) != source_hash:
        raise ValueError('Source files changed during preparation')
    layer.Save()
    joints, clamped = clamp_recorded_joints(stage, data['joint_positions'])
    neighbors=list(item_paths) if source_episode else [stock+'/'+n for n in data['neighbors']]
    record = dict(api_id=spec.api_id, scene=str(out/'initial.usda'), source_scene=str(source),
                  source_sha256=source_hash, dependency_sha256=hashes,
                  snapshot_sha256=digest(snapshot_path(spec, snapshot)), target=target, targets=acceptable_targets, support=support,
                  stock_root=stock, stock_roots=list(stock_roots), neighbors=neighbors,
                  wrong_drinks=[n for n in neighbors if n not in acceptable_targets],
                  robot_initial_positions=joints, recorded_joints_clamped_to_native_limits=clamped, target_offset_m=list(offset),
                  support_bounds_m=[list(bounds.GetMin()),list(bounds.GetMax())],
                  snapshot_schema=data['schema'],
                  source_episode=(dict(data['source'], target_path=target, target_tier=data['target_tier'],
                                       item_supports={i['prim_path']: i['support'] for i in data['items']},
                                       verification=data['verification']) if source_episode else None),
                  physics_unchanged=True, authored_state_only=True,
                  scope='source target/robot adapted to actual shelf; CPU only',
                  external_target=spec.external_target,
                  external_target_placement=({'world_shift_m': list(shift), 'bounds_m': [list(placed.GetMin()), list(placed.GetMax())], 'orientation_source': str(reference_path)} if spec.external_target else None),
                  stock_plan=stock_plan, state_paths=sorted(set(paths)|stock_paths),
                  source_to_layout_matrix=[list(row) for row in frame_delta],
                  physical_acceptance=False)
    (out/'initial-state.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    return record


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    p.add_argument('--base',default=str(BASE))
    p.add_argument('--api',default=DEFAULT.api_id)
    p.add_argument('--snapshot',default=None,help='resources/<file> override (v1 or v2 schema)')
    a=p.parse_args()
    result=prepare(a.output,base=a.base,spec=target_spec(a.api),snapshot=a.snapshot)
    print(json.dumps({k:result[k] for k in ('api_id','scene','physics_unchanged','physical_acceptance')}))

if __name__ == '__main__':
    main()
