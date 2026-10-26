"""L1 cabinet dressing recipe, task-owned and non-scoring; before PhysX only."""
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESOURCES = HERE / 'interior_resources_v1'
CATALOG = ('gatorade_t/asset.usda', 'gatorade_m/asset.usda', 'good2grow/asset.usda',
           'shrimpbox_v1/shrimpbox.usda', 'foodbox_burger_v1/foodbox.usda',
           'saladbox_v1/saladbox_shortrim.usda')
POLICY = 'l1_fridge_interior_background_target_excluded_v2'
# Semantic aliases, not just asset filenames: burger-food boxes are food boxes
# and must not distract from a requested leftover/food box.
TARGET_EXCLUSIONS = {'beef_box': (), 'leftover_box': (4,)}


def selected_variant(plan, manifest):
    variant = plan['runtime_components'][0]['parameters']['payload_variant']
    if variant not in TARGET_EXCLUSIONS:
        raise ValueError('Unsupported heated-box target: ' + str(variant))
    targets = [o for o in manifest['objects'] if o.get('scoring_target') is True]
    if len(targets) != 1 or targets[0].get('id') != variant:
        raise ValueError('Plan and unique fridge scoring target disagree')
    return variant


def sample(seed, sizes, *, target_variant):
    """L1 geometry/counts, with this episode's target category excluded."""
    import numpy as np
    if target_variant not in TARGET_EXCLUSIONS:
        raise ValueError('Unsupported heated-box target: ' + str(target_variant))
    if type(seed) is not int or seed < 0:
        raise ValueError('Nonnegative integer seed required')
    sizes = np.asarray(sizes, dtype=float)
    if sizes.shape != (6, 3) or not np.isfinite(sizes).all() or not (sizes > 0).all() or sizes.max() >= .5:
        raise ValueError('Six original asset dimensions required')
    rng = np.random.default_rng(seed)
    shelves = np.array([.65993, .82682, 1.00803, 1.19702, 1.38261])
    ceilings = np.r_[shelves[1:] - .018, 1.69]
    choices = list(rng.integers(0, 3, size=int(rng.integers(2, 4))))
    foods = [i for i in range(3, 6) if i not in TARGET_EXCLUSIONS[target_variant]]
    choices += [foods[int(i)] for i in rng.integers(0, len(foods), size=int(rng.integers(3, 5)))]
    occupied, rows = [], []
    for index in choices:
        size = sizes[index]; height = size[2]
        for attempt in range(2000):
            shelf = int(rng.integers(5))
            if height + .003 > ceilings[shelf] - shelves[shelf]:
                continue
            yaw = float(rng.uniform(-180, 180)) if index < 3 else float(rng.uniform(-25, 25))
            c, s = abs(np.cos(np.deg2rad(yaw))), abs(np.sin(np.deg2rad(yaw)))
            hx, hz = (c*size[0]+s*size[1])/2, (s*size[0]+c*size[1])/2
            if 2*hx+.03 > .461 or 2*hz+.03 > .383:
                continue
            x = float(rng.uniform(-.044+hx+.012, .417-hx-.012))
            z = float(rng.uniform(-.283+hz+.012, .10-hz-.012))
            if any(o[0] == shelf and abs(x-o[1]) < hx+o[3]+.018 and abs(z-o[2]) < hz+o[4]+.018 for o in occupied):
                continue
            break
        else:
            raise ValueError('Cannot place original fridge contents without overlap')
        occupied.append((shelf, x, z, hx, hz))
        rows.append(dict(catalog_index=int(index), asset_relative=CATALOG[index], shelf_index=shelf,
            fridge_local_base_m=[x, float(shelves[shelf]+.0004), z],
            shelf_top_m=float(shelves[shelf]), yaw_deg=yaw,
            footprint_half_extents_m=[float(hx), float(hz)], height_m=float(height),
            original_dimensions_m=size.tolist(), scale=1.))
    return rows


@lru_cache(maxsize=1)
def asset_info():
    from pxr import Usd, UsdGeom, UsdUtils, Sdf
    receipt = json.loads((RESOURCES / 'manifest.json').read_text())
    for relative, expected in receipt['files_sha256'].items():
        path = RESOURCES / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Fridge background resource changed: ' + relative)
    infos = []
    for relative in CATALOG:
        path = RESOURCES / relative
        _, _, missing = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(str(path)))
        if missing:
            raise ValueError('Missing fridge background dependencies: ' + str(missing))
        src = Usd.Stage.Open(str(path))
        if UsdGeom.GetStageUpAxis(src) != 'Z' or UsdGeom.GetStageMetersPerUnit(src) != 1:
            raise ValueError('Expected original metre Z-up asset')
        bounds = UsdGeom.BBoxCache(0, ['default', 'render']).ComputeWorldBound(src.GetDefaultPrim()).ComputeAlignedRange()
        infos.append(dict(path=str(path), lo=list(bounds.GetMin()), hi=list(bounds.GetMax()), size=list(bounds.GetSize())))
    return infos, receipt


def add(plan, stage, manifest):
    from pxr import Usd, UsdGeom, UsdPhysics, Gf
    layer = stage.GetEditTarget().GetLayer()
    if not layer.anonymous or layer == stage.GetRootLayer():
        raise ValueError('Fridge dressing requires anonymous episode layer before physics')
    fridges = [o for o in manifest['objects'] if o.get('id') == 'refrigerator']
    if len(fridges) != 1:
        raise ValueError('Unique actual refrigerator required')
    body = fridges[0]['path'] + '/body'
    prim = stage.GetPrimAtPath(body)
    if not prim:
        raise ValueError('Missing original refrigerator body')
    root = body + '/InteriorContents'
    if stage.GetPrimAtPath(root):
        raise ValueError('Fridge background already exists; refusing duplicate population')
    grip = stage.GetPrimAtPath('/World/PhysicsMaterials/GripMat')
    if not grip or not grip.HasAPI(UsdPhysics.MaterialAPI):
        raise ValueError('Missing existing GripMat; no replacement material created')
    infos, receipt = asset_info()
    variant = selected_variant(plan, manifest)
    rows = sample(plan['scene_request']['seed'], [i['size'] for i in infos], target_variant=variant)
    # Source shelf heights refer to glass, not the slightly higher front trim.
    cache = UsdGeom.XformCache()
    body_world = cache.GetLocalToWorldTransform(prim)
    if abs(body_world.GetDeterminant()-1) > 1e-6:
        raise ValueError('Original rigid cabinet frame required; no scale fallback')
    shelf_points = []
    for child in Usd.PrimRange(prim):
        if child.IsA(UsdGeom.Mesh) and 'platform_right' in str(child.GetPath()):
            matrix = cache.GetLocalToWorldTransform(child)*body_world.GetInverse()
            shelf_points.extend(matrix.Transform(Gf.Vec3d(*v)) for v in UsdGeom.Mesh(child).GetPointsAttr().Get())
    for row in rows:
        glass = [p for p in shelf_points if abs(p[1]-row['shelf_top_m']) < 1e-5]
        x, _, z = row['fridge_local_base_m']; hx, hz = row['footprint_half_extents_m']
        if (len(glass) < 4 or min(p[0] for p in glass) > x-hx or max(p[0] for p in glass) < x+hx
                or min(p[2] for p in glass) > z-hz or max(p[2] for p in glass) < z+hz):
            raise ValueError('Current cabinet glass does not support the original background recipe')
    # All authoring is in the caller-owned anonymous layer. Avoid serializing
    # the entire 63 MB binary room twice on every reset just to audit this hook;
    # the integration check separately compares full root-layer contents.
    root_layer = stage.GetRootLayer()
    before_root_dirty = root_layer.dirty
    box_paths = [o['path'] for o in manifest['objects'] if o.get('id') in ('beef_box', 'leftover_box', 'salad_box')]
    before_boxes = {p: UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(p)) for p in box_paths}
    stage.DefinePrim(root, 'Xform')
    def transform(p, matrix):
        UsdGeom.Xformable(p).MakeMatrixXform().Set(matrix)
    for k, row in enumerate(rows):
        info = infos[row['catalog_index']]
        path = root + '/item_%02d' % k
        p = stage.DefinePrim(path, 'Xform')
        # Gf row matrices: source Z-up -> cabinet Y-up, then cabinet-local yaw.
        rot = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1,0,0), -90)) * Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0,1,0), row['yaw_deg']))
        rot.SetTranslateOnly(Gf.Vec3d(*row['fridge_local_base_m']))
        transform(p, rot)
        lo, hi = info['lo'], info['hi']
        origin = stage.DefinePrim(path+'/Origin', 'Xform')
        transform(origin, Gf.Matrix4d().SetTranslate(Gf.Vec3d(-(lo[0]+hi[0])/2, -(lo[1]+hi[1])/2, -lo[2])))
        asset = stage.DefinePrim(path+'/Origin/Asset', 'Xform')
        asset.GetReferences().AddReference(info['path'])
        # L1 background semantics: static collision geometry, NOT movable task items.
        for child in Usd.PrimRange(asset):
            if child.HasAPI(UsdPhysics.RigidBodyAPI):
                child.RemoveAPI(UsdPhysics.RigidBodyAPI)
            if child.IsA(UsdGeom.Mesh):
                UsdPhysics.CollisionAPI.Apply(child).CreateCollisionEnabledAttr(True)
                child.CreateRelationship('material:binding:physics').SetTargets(['/World/PhysicsMaterials/GripMat'])
        p.SetCustomDataByKey('taskRole', 'background')
        row.update(id='fridge_background_%02d' % k, path=path, entry=info['path'],
            asset_category=Path(row['asset_relative']).parts[0], source='refrigerator_interior',
            task_role='background', task_id=None, scoring_target=False,
            physics_mode='static cabinet dressing with mesh collision; original textures',
            initialization_owner=POLICY)
    if stage.GetEditTarget().GetLayer() != layer or root_layer.dirty != before_root_dirty:
        raise RuntimeError('Scene authoring escaped the anonymous episode layer')
    for path, pose in before_boxes.items():
        if pose != UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(path)):
            raise RuntimeError('Door box moved while adding cabinet backgrounds')
    result = deepcopy(manifest)
    result['objects'].extend(rows)
    for relative, digest in receipt['files_sha256'].items():
        result['source_hashes'][str(RESOURCES/relative)] = digest
    for path in (Path(__file__), RESOURCES/'manifest.json'):
        result['source_hashes'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    result['fridge_interior_initialization'] = dict(policy=POLICY, seed=plan['scene_request']['seed'],
        count=len(rows), scoring_target_count=0, minimum_gap_m=.018,
        selected_target_variant=variant,
        excluded_background_assets=[CATALOG[i] for i in TARGET_EXCLUSIONS[variant]],
        target_category_excluded=True,
        source_recipe='L1 realistic_fridge_contents.populate_realistic',
        original_boxes_unchanged=True, original_scene_unchanged=True,
        native_validation='not_run', objects=rows)
    return result
