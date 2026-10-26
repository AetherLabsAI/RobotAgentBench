"""Opt-in unified-scene generation. Author only the caller's new episode layer."""
import hashlib
import json
from pathlib import Path
import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics
from evaluation.tasks.tabletop_pick.book_cloth.placement import sample_centres
from .audit_scene import audit
from .inventory import select_assets
from .registration import declaration, resolve_request

HERE = Path(__file__).resolve().parent
LIB = HERE.parents[2] / 'api-protocol/all_assets'
DYNAMIC = HERE / 'dynamic.f_hyv0g1/report.json'


def points(stage, path, frame=None):
    """Actual visible vertices, not decorative labels or expanded UI protection."""
    xf = UsdGeom.XformCache(); values = []
    prim = stage.GetPrimAtPath(path)
    if not prim:
        raise ValueError('Missing scene prim ' + path)
    inv = Gf.Matrix4d(1) if frame is None else frame.GetInverse()
    for p in Usd.PrimRange(prim):
        im = UsdGeom.Imageable(p)
        if not im or im.ComputeVisibility() == 'invisible' or im.ComputePurpose() not in ('default', 'render'):
            continue
        if p.IsA(UsdGeom.Mesh):
            vs = UsdGeom.Mesh(p).GetPointsAttr().Get()
        elif p.IsA(UsdGeom.Cube):
            h = UsdGeom.Cube(p).GetSizeAttr().Get() / 2
            vs = [(x, y, z) for x in (-h, h) for y in (-h, h) for z in (-h, h)]
        else:
            continue
        if vs is not None and len(vs):
            values.extend((np.c_[np.asarray(vs), np.ones(len(vs))] @ np.asarray(xf.GetLocalToWorldTransform(p) * inv))[:, :3])
    if not values:
        raise ValueError('No visible geometry ' + path)
    return np.asarray(values)


def bounds(stage, path, frame=None):
    p = points(stage, path, frame)
    return np.array([p.min(0), p.max(0)])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def initialize(stage, manifest, request, *, seed):
    """No registration => no objects; explicit existing inventory => no respawn."""
    from .placement_pose import initial_pose, check_occupancy, initial_pose_policy
    task = declaration(request)
    if task is None:
        return None
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Unsigned 32-bit scene seed required')
    if manifest.get('cloth_clear'):
        records = resolve_request(request, manifest)
        for p in records[0]['target_paths']:
            body = UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(p + '/Asset'))
            if not body or not body.GetRigidBodyEnabledAttr().Get() or body.GetKinematicEnabledAttr().Get():
                raise ValueError('Existing cloth target is missing or not dynamic: ' + p)
        manifest['cloth_clear_registration'] = records
        return dict(reused=True, target_paths=records[0]['target_paths'])
    if any(r['id'].startswith('cloth_clear_') for r in manifest['objects']):
        raise ValueError('Unassigned cloth prototypes already present; refusing duplicate generation')
    registry = LIB / 'registry.json'
    pool = [r['id'] for r in json.loads(registry.read_text())['objects'] if r['id'].startswith('cloth_clear_')]
    selected = select_assets(seed, pool)
    variants = {r['id']: r for r in json.loads(DYNAMIC.read_text())['assets']}
    cloth_rows = [r for r in manifest['objects'] if r['id'] == 'tablecloth_dirty']
    tray_rows = [r for r in manifest['objects'] if r['id'] == 'japanese_tray']
    if len(cloth_rows) != 1 or len(tray_rows) != 1:
        raise ValueError('One actual cloth and wooden tray required')
    cloth, tray = cloth_rows[0]['path'], tray_rows[0]['path']
    evidence = audit(stage, manifest)
    # Existing furniture surface itself is expected; any other visible object
    # above/in the cloth is an explicit blocker, never silently deleted/adopted.
    blockers = [h for h in evidence['overlapping_visible_geometry'] if h['path'] != '/World/Assets/CoffeeTable']
    if blockers:
        raise ValueError('Cloth occupied: preserve existing objects and resolve explicit roles first: ' +
                         ', '.join(h['path'] for h in blockers))
    frame = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(cloth))
    linear = np.asarray(frame)[:3, :3]
    if not np.allclose(linear @ linear.T, np.eye(3), atol=1e-7) or not np.allclose(linear[2], [0, 0, 1], atol=1e-7):
        raise ValueError('Cloth frame must be rigid and horizontal; no implicit scaling')
    cb = bounds(stage, cloth, frame)
    sources = []
    for ident in selected:
        item = variants[ident]
        if not item['dynamic'] or not item['collision_material_geometry_preserved']:
            raise ValueError('Missing approved variant verification')
        for path, digest in item['source_layers'].items():
            if sha(path) != digest:
                raise ValueError('Asset changed since dynamic-variant verification: ' + path)
        source = Usd.Stage.Open(item['entry'])
        rotation, b = initial_pose(source, ident)
        sources.append((ident, item, rotation, b))
    centres = sample_centres([b[1] - b[0] for _, _, _, b in sources], cb, np.random.default_rng(seed))
    paths = [f'/World/RoomEpisode/ClothClear_{i:02d}_{ident}' for i, (ident, _, _, _) in enumerate(sources)]
    if any(stage.GetPrimAtPath(p) for p in paths):
        raise ValueError('Target path already exists; refusing overwrite')
    source_hashes = {l.realPath: sha(l.realPath) for l in stage.GetUsedLayers()
                     if l.realPath and l != stage.GetRootLayer()}
    rows = []; placed_boxes = []
    for path, centre, (ident, item, rotation, b) in zip(paths, centres, sources):
        node = UsdGeom.Xform.Define(stage, path)
        child = stage.DefinePrim(path + '/Asset')
        child.GetReferences().AddReference(item['entry'])
        delta = Gf.Matrix4d().SetTranslate(Gf.Vec3d(*map(float, centre - b.mean(0))))
        parent = UsdGeom.XformCache().GetLocalToWorldTransform(node.GetPrim().GetParent())
        node.MakeMatrixXform().Set(rotation * delta * frame * parent.GetInverse())
        child.SetCustomDataByKey('roomAssetId', ident)
        actual = bounds(stage, path, frame)
        if not np.allclose(actual, b + (centre - b.mean(0)), atol=1e-7):
            raise ValueError('Cloth posed footprint mismatch')
        placed_boxes.append(actual)
        if not np.all(actual[0, :2] >= cb[0, :2] + .008 - 1e-7) or not np.all(actual[1, :2] <= cb[1, :2] - .008 + 1e-7):
            raise ValueError('Instantiated object escapes approved cloth sampler boundary')
        row = dict(id=ident, path=path, entry=str(LIB / ident / 'asset.usda'),
                   initial_pose_policy=initial_pose_policy(ident),
                   dynamic_entry=item['entry'], dynamic_entry_sha256=sha(item['entry']),
                   rigid_body_path=path + '/Asset', task_role='cloth_clear_target',
                   mass_kg=item['mass_kg'], native_verified=False,
                   visible_bounds_world_m=bounds(stage, path).tolist(),
                   initial_bounds_cloth_local_m=actual.tolist())
        rows.append(row)
    check_occupancy(placed_boxes, cb)
    manifest['objects'].extend(rows)
    manifest['cloth_clear'] = dict(version=1, seed=seed, target_paths=paths, asset_pool_ids=sorted(pool),
                                  cloth_path=cloth, table_path='/World/Assets/CoffeeTable', exclusion_paths=[tray],
                                  sampler='existing_book_cloth.sample_centres', edge_gap_m=.008,
                                  object_gap_m=.015, initial_bottom_gap_m=.003,
                                  existing_occupancy_audit=evidence, native_verified=False)
    manifest['cloth_clear_registration'] = resolve_request(request, manifest)
    if not all(sha(p) == h for p, h in source_hashes.items()):
        raise ValueError('Referenced source was modified during generation')
    return dict(reused=False, target_paths=paths, source_unchanged=True, native_verified=False)
