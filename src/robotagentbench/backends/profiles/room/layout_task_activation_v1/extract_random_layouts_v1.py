"""Read-only release geometry for four-zone design previews (no USD writes)."""
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from pxr import Usd, UsdGeom
from openend.tasks.scene_requirements.surface_registry import update_layout

ROOT = Path('/mnt/agent-bench-data/lingjun/releases/living50-candidate-20260919')
OUT = Path(__file__).parent / 'web/random-layouts-v1.json'
PREFIX = '/World/Background/Shell/ConstructorKitchen/MetricScale/'
SMALL_TABLE = '/World/Assets/KitchenCoffeeTable03'
DINING_TABLE = '/World/LargeDiningTable'
PARTS = ['RightWallBase150/Body/Worktop', 'RightWallBase400/Body/Worktop',
         'RightEnd300/Module/Body/Worktop', 'BP_Kitchen36/Body/Worktop',
         'RightRange600/Module/Body/Worktop', 'BP_Kitchen9/Body/Worktop',
         'BP_Kitchen10/Body/Worktop', 'BP_Kitchen8/Body/Worktop',
         'CornerGapBase300_A/Body/Worktop', 'CornerGapBase300_B/Body/Worktop']
MOVED = ['/World/Assets/AirFryer', '/World/Assets/DiningTableMicrowave',
         '/World/Assets/CoffeeGrinder', '/World/Assets/CoffeeMachine',
         '/World/FoodTray', '/World/Cutlery', '/World/MealDiverse']


def main():
    layouts = []
    for plan in json.loads((ROOT / 'plans_release.json').read_text()):
        num = plan['id']
        scene, = (ROOT / 'layouts').glob(f'ID_{num:02d}_*.usdc')
        audit_file = ROOT / f'audits_release/{num:02d}.json'
        audit = json.loads(audit_file.read_text())
        stage = Usd.Stage.Open(str(scene))
        assert UsdGeom.GetStageMetersPerUnit(stage) == 1
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render', 'proxy'])
        xc = UsdGeom.XformCache()

        def measure(prim):
            if not prim:
                return None
            if prim.IsA(UsdGeom.Imageable) and UsdGeom.Imageable(prim).ComputeVisibility() == 'invisible':
                return None
            b = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            if b.IsEmpty():
                return None
            lo, hi = np.array(b.GetMin()), np.array(b.GetMax())
            if not np.isfinite([lo, hi]).all():
                raise ValueError(str(prim.GetPath()))
            return dict(path=str(prim.GetPath()), min=lo.tolist(), max=hi.tolist())

        surfaces, errors, absent = [], [], []
        paths = [('/World/Background/Shell/CounterTop_17', 'island')] + [(PREFIX + x, 'stove') for x in PARTS] + [(SMALL_TABLE, 'small_table'), (DINING_TABLE, 'dining_table')]
        for path, group in paths:
            prim = stage.GetPrimAtPath(path)
            b = measure(prim)
            if b is None:
                absent.append(path)
                continue
            vertices = []
            for child in Usd.PrimRange(prim):
                if not (child.IsA(UsdGeom.Mesh) or child.IsA(UsdGeom.Cube)) or UsdGeom.Imageable(child).ComputeVisibility() == 'invisible':
                    continue
                if child.IsA(UsdGeom.Cube):
                    half = float(UsdGeom.Cube(child).GetSizeAttr().Get()) / 2
                    pts = np.asarray(list(itertools.product([-half, half], repeat=3)))
                else:
                    pts = np.asarray(UsdGeom.Mesh(child).GetPointsAttr().Get(), dtype=float)
                mat = np.asarray(xc.GetLocalToWorldTransform(child))
                vertices.extend((np.column_stack((pts, np.ones(len(pts)))) @ mat)[:, :3])
            if not vertices:
                errors.append('No mesh vertices for support: ' + path)
                continue
            pts = np.asarray(vertices)
            # Worktops have a raised narrow rear lip (~.949 m); the broad
            # horizontal working plane is .920 m. Do not use visual max-Z.
            planes = []
            for height in np.unique(pts[:, 2].round(5)):
                xy = np.unique(pts[np.abs(pts[:, 2] - height) < 1e-5, :2], axis=0)
                try:
                    hull = ConvexHull(xy)
                    if hull.volume > .005:
                        planes.append((height, hull.volume, xy[hull.vertices].tolist()))
                except Exception:
                    pass
            if not planes:
                errors.append('No flat upper support polygon: ' + path)
                continue
            largest = max(p[1] for p in planes)
            top, _, polygon = max((p for p in planes if p[1] >= largest * .8), key=lambda p: p[0])
            # Only the user-approved 92 cm dining tables are eligible. Do not
            # move a table or substitute its whole visual AABB for the top.
            if group == 'dining_table' and abs(float(top) - .92) > 1e-5:
                absent.append(path + ': support height is not 92 cm')
                continue
            b['polygon'] = polygon
            b.update(group=group, name={'island': '岛台', 'stove': '灶台 / 备餐台', 'small_table': '厨房小桌', 'dining_table': '独立餐桌'}[group], top=float(top),
                     support_method='highest broad coplanar vertex hull; raised rear lip excluded')
            surfaces.append(b)
        blockers = []
        tables = []
        for table_path in ['/World/LargeDiningTable', '/World/Assets/KitchenCoffeeTable03']:
            table = measure(stage.GetPrimAtPath(table_path))
            if table:
                table['name'] = '独立餐桌' if table_path.endswith('LargeDiningTable') else '厨房小桌'
                support = next((s for s in surfaces if s['path'] == table_path), None)
                table['included_as_support'] = support is not None
                if support:
                    table['support_top_m'] = support['top']
                    table['support_polygon'] = support['polygon']
                    table['height_delta_from_counter_m'] = support['top'] - .92
                tables.append(table)
        # Some layouts have freestanding furniture directly below /World,
        # not /World/Assets. They must not disappear from obstacle checks.
        for prim in stage.GetPrimAtPath('/World').GetChildren():
            path = str(prim.GetPath())
            if path in MOVED or any(s['path'] == path for s in surfaces) or path in ['/World/Assets', '/World/Background', '/World/Robot']:
                continue
            b = measure(prim)
            if b:
                blockers.append(b)
        for parent in ['/World/Assets', '/World/Background/Shell']:
            for prim in stage.GetPrimAtPath(parent).GetChildren():
                path = str(prim.GetPath())
                if path in MOVED or any(s['path'] == path for s in surfaces) or path.endswith(('/ConstructorKitchen', '/CounterBase_16', '/CounterTop_17', '/Floor')):
                    continue
                if path == '/World/Assets/LivingRoomPlantStands':
                    # This is an organizational group of distant stands, not
                    # one physical object. Preserve each child's own bounds.
                    children = [measure(child) for child in prim.GetChildren()]
                    children = [b for b in children if b is not None]
                    if not children:
                        errors.append('No individual plant stand bounds: ' + path)
                    blockers.extend({**b, 'source_group': path} for b in children)
                    continue
                b = measure(prim)
                if b is not None:
                    blockers.append(b)
        # Keep above-counter kitchen geometry (upper cabinets etc.). Supporting
        # cabinetry and permitted unheated stove are not unrelated obstacles.
        for prim in Usd.PrimRange(stage.GetPrimAtPath(PREFIX.rstrip('/'))):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            path = str(prim.GetPath())
            if '/Worktop' in path or '/Stove' in path or '/Sink' in path:
                continue
            b = measure(prim)
            if b and b['max'][2] > 1.01:
                blockers.append(b)
        sink = measure(stage.GetPrimAtPath(PREFIX + 'BP_Kitchen10/Body/Sink'))
        if sink:
            sink['always_exclude_xy'] = True
            sink['activity_projection_allowed'] = True
            blockers.append(sink)
        else:
            errors.append('Missing sink exclusion geometry')
        # A hand workspace may project over the basin, but the protruding tap
        # is not part of that exemption (nor the overhead-height waiver).
        tap = measure(stage.GetPrimAtPath(PREFIX + 'BP_Kitchen10/Body/Sink/Tap'))
        if tap:
            tap['always_exclude_xy'] = True
            tap['obstacle_kind'] = 'faucet'
            blockers.append(tap)
        else:
            errors.append('Missing faucet exclusion geometry')
        layouts.append(update_layout(stage, dict(id=num, scene=str(scene), scene_sha256=hashlib.sha256(scene.read_bytes()).hexdigest(),
                            audit_sha256=hashlib.sha256(audit_file.read_bytes()).hexdigest(),
                            floors=audit['floors'], bounds=audit['bounds'], surfaces=surfaces,
                            blockers=blockers, tables=tables, errors=errors, absent_or_inactive_surfaces=absent)))
        print(f'{num:02d}: {len(surfaces)} surfaces, {len(blockers)} blockers, {len(errors)} errors', flush=True)
    result = dict(schema='arena.four_zone.layouts.v1', layouts=sorted(layouts, key=lambda x: x['id']),
                  removed_only_in_preview=MOVED, physics_validated=False,
                  limitations=['Support inventory uses actual horizontal faces and enabled collider identities; obstacle bounds remain conservative. Source floors are release audits.',
                               'Planar candidate coverage does not certify robot body clearance, navigation, settling or execution.',
                               'Task fixtures and tray contents relocate only in preview. Original scenes are unchanged.'])
    OUT.write_text(json.dumps(result, ensure_ascii=False, separators=(',', ':')))


if __name__ == '__main__':
    main()
