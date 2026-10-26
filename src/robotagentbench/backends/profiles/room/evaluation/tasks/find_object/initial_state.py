"""Per-episode scene layer of the find_object family: public house baseline + the producer's furniture and objects from all_assets + the
layout of one packaged source episode (tools/package_source_episode.py).

The station layer /World/FindObject references all_assets entries (sofa_4seat, coffee_table_with_drawers, small_coffee_table, tv_table,
glasses_case / wallet / remote_tv, and every distractor with an all_assets counterpart) and places each at the producer's composed world
transform; the public baseline prims they replace are deactivated; public prims the producer only moved (Tv, trash bins, basket, ...) get
the producer's transform; the public w120 snack shelf stays (user 2026-10-05). Robot, gripper, physics scene, solver and material parameters
stay the public unified ones; nothing of the producer's robot or physics scene is loaded.

Placement arithmetic (row-vector Gf convention, world = local * parent): the station prim's transform op replaces the all_assets root ops, so it
is authored as S * W: S = the asset root's scale factors (same model file and scale as the producer prim), W = the producer's settled frame-0
PhysX body pose (rotation + translation of the prim's world transform). Furniture followers get the producer's full composed transform.
"""
import hashlib
import json
import math
import os
from pathlib import Path

from .specs import (TASK, REPO, ALL_ASSETS, EPISODES, ROBOT, STATION, TARGET, DISTRACTORS, DRAWER, DRAWER_JOINT, DRAWER_SLIDE_LAYER, COFFEE_TABLE, TV, FLOOR,
                    DEACTIVATED_HOUSE, FURNITURE_ASSETS, LAYOUT_FOLLOWERS, TARGET_PHYSICS, SCENE_PHYSICS, spec as key_spec, episodes, episode_manifest)

SCHEMA = 'arena.find_object.initial_state.v1'
PREPARED_MARKER = 'arena:l1ScenePrepared'
SETTLE_SECONDS = 1.0           # DEFINITION_TABLE 初态与扰动: 初态静置 1 s，检查最后 0.5 s
FLOOR_CLUTTER_Z_M = 0.05       # lowest point of the object's geometry at the producer frame-0 pose below this height = scene clutter already lying on the
                               # floor (furniture tops ≥ 0.44 m). Geometry, not body origin: the baseball cap's mesh sits 1.87 m from its body origin
                               # (4.7 ep5: origin z 1.71 m, mesh z 0.00–0.20 m on the floor).
FOLLOWERS = tuple(p for p in LAYOUT_FOLLOWERS if p != '/World/Assets/ShelvingUnit')


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def package(key, label):
    """Verified packaged layout of one source episode."""
    m = episode_manifest(key, label)
    sp = key_spec(key)
    if m['key'] != key or m['label'] != label or m['task_kind'] != sp.kind or m['target']['asset'] != sp.target_asset:
        raise ValueError('Source episode identity mismatch')
    for name, (asset, _) in FURNITURE_ASSETS.items():
        f = m['furniture'].get('/World/Assets/' + name)
        if not f or f['asset'] != asset or len(f['world_matrix']) != 16 or not all(math.isfinite(v) for v in f['world_matrix']):
            raise ValueError('Invalid furniture layout: ' + name)
    for field in ('positions', 'hold_targets'):
        v = m['robot_initialization'][field]
        if len(v) != 20 or not all(type(x) in (int, float) and math.isfinite(x) for x in v.values()):
            raise ValueError('Invalid recorded reset coordinates')
    if sp.kind == 'drawer' and not m.get('drawer'):
        raise ValueError('Drawer episode without a drawer record')
    for asset in [sp.target_asset] + [d['asset'] for d in m['distractors'].values()] + [a for a, _ in FURNITURE_ASSETS.values()]:
        if not (ALL_ASSETS / asset / 'asset.usda').is_file():
            raise ValueError('all_assets entry missing: ' + asset)
    return m


def asset_root_transform(asset):
    """Composed local transform L of /Asset of one all_assets entry."""
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.Open(str(ALL_ASSETS / asset / 'asset.usda'))
    prim = stage.GetPrimAtPath('/Asset')
    if not prim:
        raise ValueError('all_assets entry has no /Asset: ' + asset)
    return UsdGeom.Xformable(prim).GetLocalTransformation()


def pose_matrix(position, quaternion):
    from pxr import Gf
    w, x, y, z = quaternion
    m = Gf.Matrix4d().SetRotate(Gf.Quatd(float(w), Gf.Vec3d(float(x), float(y), float(z))))
    m.SetTranslateOnly(Gf.Vec3d(*[float(v) for v in position]))
    return m



def apply_target_physics(stage, asset_name, settings):
    """Author object-side overrides on the TARGET instance in the station layer: collision approximation of every collider mesh of the
    all_assets entry, optionally a task-owned physics material (friction, combine mode max like the producer scenes) and zero sleep /
    stabilization thresholds. Returns the record that goes into initial.json."""
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf
    unknown = set(settings) - {'approximation', 'static_friction', 'dynamic_friction', 'sleep0', 'source', 'contact_offset', 'rest_offset', 'decomposition', 'solver_iterations'}
    if unknown:
        raise ValueError('Unknown target physics settings: ' + ', '.join(sorted(unknown)))
    asset = Usd.Stage.Open(str(ALL_ASSETS / asset_name / 'asset.usda')); root = asset.GetDefaultPrim().GetPath()
    colliders = [p.GetPath().MakeRelativePath(root) for p in asset.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI) and p.IsA(UsdGeom.Mesh)]
    if not colliders:
        raise ValueError('No collider mesh in ' + asset_name)
    record = dict(colliders=[str(c) for c in colliders])
    material = None
    if 'static_friction' in settings or 'dynamic_friction' in settings:
        material = UsdShade.Material.Define(stage, STATION + '/TargetPhysicsMaterial')
        api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        api.CreateStaticFrictionAttr(float(settings['static_friction'])); api.CreateDynamicFrictionAttr(float(settings['dynamic_friction']))
        api.CreateRestitutionAttr(0.0)
        material.GetPrim().CreateAttribute('physxMaterial:frictionCombineMode', Sdf.ValueTypeNames.Token).Set('max')
        record.update(static_friction=float(settings['static_friction']), dynamic_friction=float(settings['dynamic_friction']), friction_combine='max')
    for rel_path in colliders:
        over = stage.OverridePrim(Sdf.Path(TARGET).AppendPath(rel_path))
        if 'approximation' in settings:
            over.CreateAttribute('physics:approximation', Sdf.ValueTypeNames.Token, variability=Sdf.VariabilityUniform).Set(settings['approximation'])
        if material is not None:
            UsdShade.MaterialBindingAPI.Apply(over).Bind(material, UsdShade.Tokens.strongerThanDescendants, 'physics')
        if 'contact_offset' in settings:
            over.CreateAttribute('physxCollision:contactOffset', Sdf.ValueTypeNames.Float).Set(float(settings['contact_offset']))
        if 'rest_offset' in settings:
            over.CreateAttribute('physxCollision:restOffset', Sdf.ValueTypeNames.Float).Set(float(settings['rest_offset']))
        for name, value in (settings.get('decomposition') or {}).items():   # e.g. hullVertexLimit / maxConvexHulls / shrinkWrap / voxelResolution
            kind = Sdf.ValueTypeNames.Bool if isinstance(value, bool) else Sdf.ValueTypeNames.Int
            over.CreateAttribute('physxConvexDecompositionCollision:' + name, kind).Set(value)
    if 'approximation' in settings:
        record['approximation'] = settings['approximation']
    for k in ('contact_offset', 'rest_offset', 'decomposition'):
        if k in settings:
            record[k] = settings[k]
    if 'solver_iterations' in settings:      # [position, velocity] on the target body (raises the iterations of every contact island it joins)
        body = stage.GetPrimAtPath(TARGET)
        body.CreateAttribute('physxRigidBody:solverPositionIterationCount', Sdf.ValueTypeNames.Int).Set(int(settings['solver_iterations'][0]))
        body.CreateAttribute('physxRigidBody:solverVelocityIterationCount', Sdf.ValueTypeNames.Int).Set(int(settings['solver_iterations'][1]))
        record['solver_iterations'] = list(settings['solver_iterations'])
    if settings.get('sleep0'):
        body = stage.GetPrimAtPath(TARGET)
        body.CreateAttribute('physxRigidBody:sleepThreshold', Sdf.ValueTypeNames.Float).Set(0.0)
        body.CreateAttribute('physxRigidBody:stabilizationThreshold', Sdf.ValueTypeNames.Float).Set(0.0)
        record['sleep_and_stabilization_thresholds'] = 0.0
    return record

def write_station(path, manifest, sp):
    """One layer with the whole station of this episode, references relative to the layer file."""
    from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf
    path = Path(path)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, 'Z')
    stage.OverridePrim('/World')
    placed = {}

    def rel(asset):
        return os.path.relpath((ALL_ASSETS / asset / 'asset.usda').resolve(), path.parent)

    def place(prim, matrix):
        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        xf.AddTransformOp().Set(Gf.Matrix4d(matrix))

    def instance(prim_path, asset, world, note, obj_local=None):
        # The reference brings the asset's own root ops (archive rotate / scale); `place` replaces them, so the station op must be the whole
        # local transform of the geometry: the asset's scale (same model file, same scale as the producer prim) followed by the producer's
        # settled body pose (rotation + translation, which already is the prim's world rotation; the archive's own rotate is not part of it).
        prim = UsdGeom.Xform.Define(stage, prim_path).GetPrim()
        prim.GetReferences().AddReference(rel(asset))
        L = asset_root_transform(asset)
        scale = Gf.Vec3d(*[L.GetRow3(i).GetLength() for i in range(3)])
        T = Gf.Matrix4d().SetScale(scale) * world
        if obj_local is not None:
            # producer mesh = Obj_local (e.g. family4CoordinateCompensation) x scale x body; reproduce it on the instance root
            T = Gf.Matrix4d(*obj_local) * T
        place(prim, T)
        prim.SetCustomDataByKey('arena:allAssetsEntry', asset)
        placed[prim_path] = dict(asset=asset, world=[world[i][j] for i in range(4) for j in range(4)], asset_scale=list(scale), note=note, producer_obj_local=obj_local)
        return prim

    stage.DefinePrim(STATION)
    stage.DefinePrim(DISTRACTORS)
    # furniture = public prims moved to the producer layout (public obstacle / coffee-box policies need the public Sofa and CoffeeTable prims)
    for name, (asset, prim_path) in FURNITURE_ASSETS.items():
        f = manifest['furniture']['/World/Assets/' + name]
        place(stage.OverridePrim(prim_path), Gf.Matrix4d(*f['world_matrix']))
        placed[prim_path] = dict(asset=None, public_prim=True, all_assets_reference=asset, world=f['world_matrix'], note='public baseline furniture moved to the producer composed transform')
    drawer_record = None
    if sp.kind == 'drawer':
        # public drawer slide layer (composed by prepare); the drawer starts at the producer's frame-0 pose (closed in every packaged 4.4 / 4.10 episode)
        d = manifest['drawer']
        table_world = Gf.Matrix4d(*manifest['furniture']['/World/Assets/CoffeeTable']['world_matrix'])
        drawer_record = dict(opening_m=d['opening_m'], public_slide_layer=str(DRAWER_SLIDE_LAYER.relative_to(REPO)), producer_rails=d['rails'],
                             note='joint and wall materials come from the public slide layer (user decision pending on the producer rail parameters); only the frame-0 drawer pose is authored')
        pose = manifest['drawers'].get('black_box_1')
        if pose:
            local = pose_matrix(pose['position_m'], pose['quaternion_wxyz']) * table_world.GetInverse()
            over = stage.OverridePrim(DRAWER)
            over.CreateAttribute('xformOp:translate', Sdf.ValueTypeNames.Double3).Set(Gf.Vec3d(local.ExtractTranslation()))
            drawer_record['frame0_local_translate'] = list(local.ExtractTranslation())
    t = manifest['target']
    instance(TARGET, sp.target_asset, pose_matrix(t['position_m'], t['quaternion_wxyz']), 'producer settled frame-0 body pose', t.get('obj_local_matrix'))
    scene_official = SCENE_PHYSICS.get(sp.key)
    if scene_official:
        if 'drawer_wall_friction' in scene_official:
            mat = stage.OverridePrim(DRAWER + '/DrawerWallMat')
            mat.CreateAttribute('physics:staticFriction', Sdf.ValueTypeNames.Float).Set(float(scene_official['drawer_wall_friction'][0]))
            mat.CreateAttribute('physics:dynamicFriction', Sdf.ValueTypeNames.Float).Set(float(scene_official['drawer_wall_friction'][1]))
        placed['__scene_physics__'] = dict(scene_official, kind='task setting')
    scene_diag = os.environ.get('ARENA_FIND_OBJECT_DIAG_SCENE')
    if scene_diag:
        # DIAGNOSTIC ONLY (never set by batch_queue): scene-side overrides to attribute a failure, e.g.
        # {"drawer_wall_friction": [1.2, 1.0]} -> DrawerWallMat of black_box_1 as in the producer 4.4 / 4.10 scenes
        cfg = json.loads(scene_diag); rec = {}
        if 'drawer_wall_friction' in cfg:
            mat = stage.OverridePrim(DRAWER + '/DrawerWallMat')
            mat.CreateAttribute('physics:staticFriction', Sdf.ValueTypeNames.Float).Set(float(cfg['drawer_wall_friction'][0]))
            mat.CreateAttribute('physics:dynamicFriction', Sdf.ValueTypeNames.Float).Set(float(cfg['drawer_wall_friction'][1]))
            rec['drawer_wall_friction'] = list(cfg['drawer_wall_friction'])
        unknown = set(cfg) - set(rec)
        if unknown:
            raise ValueError('Unknown diagnostic scene settings: ' + ', '.join(sorted(unknown)))
        placed['__diagnostic_scene__'] = dict(rec, note='diagnostic override, not a test configuration')
    official = TARGET_PHYSICS.get(sp.key)
    if official:
        placed[TARGET]['target_physics'] = dict(apply_target_physics(stage, sp.target_asset, official), source=official.get('source'), kind='task setting')
    diag = os.environ.get('ARENA_FIND_OBJECT_DIAG_TARGET_PHYSICS')
    if diag:
        # DIAGNOSTIC ONLY (never set by batch_queue): object-side settings to attribute a failure. Legacy words 'hull' / 'hull_nosleep',
        # or JSON {"approximation": ..., "static_friction": .., "dynamic_friction": .., "sleep0": true}.
        legacy = {'hull': dict(approximation='convexHull'), 'hull_nosleep': dict(approximation='convexHull', sleep0=True)}
        settings = legacy[diag] if diag in legacy else json.loads(diag)
        placed[TARGET]['diagnostic_target_physics'] = dict(apply_target_physics(stage, sp.target_asset, settings), mode=diag,
                                                            note='diagnostic override, not a test configuration')
    static = manifest.get('producer_static_unsupported') or {}
    for name, d in manifest['distractors'].items():
        prim = instance(DISTRACTORS + '/' + name, d['asset'], pose_matrix(d['position_m'], d['quaternion_wxyz']), 'producer settled frame-0 body pose', d.get('obj_local_matrix'))
        if name in static:
            # held where the producer data shows it for the whole episode (nothing in the scene supports it; tools/mark_producer_static.py)
            prim.CreateAttribute('physics:kinematicEnabled', Sdf.ValueTypeNames.Bool).Set(True)
            placed[DISTRACTORS + '/' + name]['held_kinematic'] = static[name]
    for p in FOLLOWERS:
        f = manifest['layout_followers'].get(p)
        if f:
            place(stage.OverridePrim(p), Gf.Matrix4d(*f['world_matrix']))
    place(stage.OverridePrim(ROBOT), Gf.Matrix4d(*manifest['robot']['world_matrix']))
    for house in DEACTIVATED_HOUSE:
        stage.OverridePrim(house).SetActive(False)
    stage.GetRootLayer().Save()
    return path, placed, drawer_record


def inspect_stage(stage, key, manifest):
    """Structural check only; physical acceptance happens after the settle."""
    from pxr import UsdPhysics, UsdGeom
    sp = key_spec(key)
    required = [ROBOT, TARGET, FLOOR] + [p for _, p in FURNITURE_ASSETS.values()] + [DISTRACTORS + '/' + n for n in manifest['distractors']]
    if sp.kind == 'drawer' and not stage.GetPrimAtPath(DRAWER_JOINT):
        raise ValueError('Public drawer slide layer not composed')
    if sp.kind == 'drawer':
        required += [DRAWER, DRAWER_JOINT]
    missing = [p for p in required if not stage.GetPrimAtPath(p) or not stage.GetPrimAtPath(p).IsActive()]
    if missing:
        raise ValueError('Missing initial-state instances: ' + ', '.join(missing))
    if not stage.GetPrimAtPath(TARGET).HasAPI(UsdPhysics.RigidBodyAPI):
        raise ValueError('Target is not a rigid body')
    for house in DEACTIVATED_HOUSE:
        prim = stage.GetPrimAtPath(house)
        if prim and prim.IsActive():
            raise ValueError('House object must be inactive at the station: ' + house)
    if sp.kind == 'drawer':
        if stage.GetPrimAtPath(DRAWER).GetAttribute('physics:kinematicEnabled').Get() is not False:
            raise ValueError('Drawer must be a dynamic rigid body on its rail')
        if stage.GetPrimAtPath(DRAWER_JOINT).GetAttribute('physics:jointEnabled').Get() is False:
            raise ValueError('Drawer joint must be enabled')
    if UsdGeom.GetStageUpAxis(stage) != 'Z' or not math.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1.):
        raise ValueError('Expected Z-up metre scene')
    cache = UsdGeom.XformCache()
    t = manifest['target']
    world = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(TARGET))
    expected = pose_matrix(t['position_m'], t['quaternion_wxyz'])
    # the geometry must sit where the producer's did: compare the station transform * asset root against the producer body pose
    err = max(abs(world.ExtractTranslation()[i] - expected.ExtractTranslation()[i]) for i in range(3))
    return dict(required_instances=required, target_world_translation_error_m=float(err), structural_check='passed', physical_settle_acceptance=None, task_success=None)


def prepare(source, output, *, key, episode=None, seed=None, role='source'):
    from pxr import Sdf, Usd, UsdGeom
    from arena_api import l1_scene
    sp = key_spec(key)
    output = Path(output).resolve()
    if not (output.is_relative_to(TASK.resolve()) or output.is_relative_to((TASK / 'artifacts').resolve())) or output == TASK.resolve():
        raise ValueError('Initial-state output must stay inside the find_object task directory')
    source = Path(source).resolve(strict=True)
    if source != l1_scene.BASE.resolve():
        raise ValueError('The base scene must be the public L1 baseline')
    if role != 'source':
        raise ValueError('Only the source layout is wired (randomization: see randomization.py NOT_RUN)')
    if episode is None:
        episode = episodes(key)[0]
    manifest = package(key, episode)
    output.mkdir(parents=True, exist_ok=False)
    station, placed, drawer_record = write_station(output / 'station.usda', manifest, sp)
    scene = output / 'initial.usda'
    stage = Usd.Stage.CreateNew(str(scene))
    layers = [station] + ([DRAWER_SLIDE_LAYER] if sp.kind == 'drawer' else []) + [*l1_scene.LAYERS, source]
    stage.GetRootLayer().subLayerPaths = [os.path.relpath(p, output) for p in layers]
    stage.SetDefaultPrim(stage.GetPrimAtPath('/World'))
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, 'Z')
    stage.GetPrimAtPath('/World').CreateAttribute(l1_scene.MARKER, Sdf.ValueTypeNames.String).Set(l1_scene.POLICY)
    l1_scene.verify(stage)
    report = inspect_stage(stage, key, manifest)
    robot = UsdGeom.Xformable(stage.GetPrimAtPath(ROBOT)).ComputeLocalToWorldTransform(0)
    from pxr import Gf
    expected = Gf.Matrix4d(*manifest['robot']['world_matrix'])
    if max(abs(robot[i][j] - expected[i][j]) for i in range(4) for j in range(4)) > 1e-6:
        raise ValueError('Robot station differs from the source episode')
    stage.GetRootLayer().Save()
    bbox = UsdGeom.BBoxCache(0, ['default', 'render', 'proxy'])
    lowest = {n: float(bbox.ComputeWorldBound(stage.GetPrimAtPath(DISTRACTORS + '/' + n)).ComputeAlignedRange().GetMin()[2]) for n in manifest['distractors']}
    plan = dict(
        schema=SCHEMA, key=key, runtime_id='find_object', role=role, source_episode=episode, seed=seed, task_kind=sp.kind,
        producer_passed=manifest['producer'].get('passed'), robot_initialization=dict(positions=dict(manifest['robot_initialization']['positions']),
                                                                                       hold_targets=dict(manifest['robot_initialization']['hold_targets'])),
        randomization=None, base_scene='baseline', station_sha256=digest(station), episode_manifest_sha256=digest(EPISODES / key / episode / 'episode.json'),
        settle_seconds=SETTLE_SECONDS, inspection=report, placed=placed, drawer=drawer_record,
        target_physics=placed.get(TARGET, {}).get('target_physics'), scene_physics=placed.get('__scene_physics__'), diagnostic_target_physics=placed.get(TARGET, {}).get('diagnostic_target_physics'),
        distractors={n: d['asset'] for n, d in manifest['distractors'].items()}, omitted_objects=[o['name'] for o in manifest['omitted_objects']],
        floor_clutter=sorted(n for n, z in lowest.items() if z < FLOOR_CLUTTER_Z_M),
        distractor_lowest_point_m={n: round(z, 4) for n, z in lowest.items()},
        floor_clutter_rule=f'distractors whose geometry at the producer frame-0 pose reaches below {FLOOR_CLUTTER_Z_M} m lie on the floor in the source scene; they are '
                           'scene clutter, not knocked-down distractors: excluded from the nothing_on_the_ground acceptance and from the distractor_grounded deduction',
        asset_hashes={a: digest(ALL_ASSETS / a / 'asset.usda') for a in sorted({sp.target_asset, *[d['asset'] for d in manifest['distractors'].values()]})},
        furniture='public baseline prims (Sofa / CoffeeTable / SmallCoffeeTable / TvTable / ShelvingUnit w120) at the producer layout; all_assets furniture entries are the verified reference copies',
        task_objects='all_assets object entries (producer unified archive) at the producer layout of the source episode; public furniture moved to that layout',
        physical_parameters_overridden=False, robot_or_physics_scene_from_producer=False, public_shelf='w120 (user 2026-10-05)')
    (output / 'initial.json').write_text(json.dumps(plan, indent=2, allow_nan=False) + '\n')
    return scene


if __name__ == '__main__':
    import argparse
    from arena_api import l1_scene
    parser = argparse.ArgumentParser()
    parser.add_argument('--key', required=True)
    parser.add_argument('--episode')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(prepare(l1_scene.BASE, args.output, key=args.key, episode=args.episode))
