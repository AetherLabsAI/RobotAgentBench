"""Three fixed table regions; selection is independent of object scoring."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TABLE = '/World/Assets/CoffeeTable'
IDS = ('table_wooden_tray', 'table_cloth', 'table_objects')


def regions_from_local(profile, transform_point):
    rows = profile['regions']
    if tuple(r['id'] for r in rows) != IDS:
        raise ValueError('Exactly the three approved table regions required')
    result = []
    for row in rows:
        if len(row['polygon_local_m']) != 4:
            raise ValueError('Measured four-corner region required')
        result.append(dict(id=row['id'], path=TABLE, hz=60,
            polygon=tuple(tuple(float(v) for v in list(transform_point(p))[:2])
                          for p in row['polygon_local_m'])))
    return result


def bind(stage, manifest):
    """Map saved table-local regions through THIS room's actual transform.

    Generated fixed cloth/tray anchors must match the approved geometry. An
    anchor absent from the scene inventory does not create a workspace. A
    declared but missing, inactive, ambiguous or displaced anchor remains an
    error. Never substitute a whole-table box or a layout01 world polygon.
    """
    from pxr import Usd, UsdGeom, Gf
    from openend.tasks.all_objects_integration.workspace_physics import Workspace
    table = stage.GetPrimAtPath(TABLE)
    if not table or not table.IsActive():
        return []
    profile = manifest.get('coffee_table_workspaces')
    if profile is None:
        profile = json.loads((HERE / 'local_regions.json').read_text())
    if profile['support_path'] != TABLE:
        raise ValueError('Unexpected table support')
    cache = UsdGeom.XformCache()
    transform = cache.GetLocalToWorldTransform(table)
    inverse = transform.GetInverse()
    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render'], useExtentsHint=False)
    # Validate the complete approved profile BEFORE excluding absent anchors.
    rows = regions_from_local(profile, lambda p: transform.Transform(Gf.Vec3d(*p)))
    absent = set()
    # Check the saved geometry against fixed room anchors before using a legacy
    # manifest fallback. A changed arrangement must emit a matching new profile.
    for row in profile['regions'][:2]:
        matches = [o for o in manifest['objects'] if o['id'] == row['anchor_asset_id']]
        if not matches:
            absent.add(row['id'])
            continue
        if len(matches) != 1:
            raise ValueError('Ambiguous table region anchor: ' + row['id'])
        anchor = stage.GetPrimAtPath(matches[0]['path'])
        if not anchor or not anchor.IsActive() or UsdGeom.Imageable(anchor).ComputeVisibility() == 'invisible':
            raise ValueError('Inactive table region anchor: ' + row['id'])
        box = bounds.ComputeWorldBound(anchor).ComputeAlignedRange()
        actual = inverse.Transform((box.GetMin() + box.GetMax()) * .5)
        expected = row['anchor_center_local_m']
        if any(abs(actual[i] - expected[i]) > 1e-4 for i in (0, 1)):
            raise ValueError('Fixed table layout differs from region profile: ' + row['id'])
    return [Workspace(**r) for r in rows if r['id'] not in absent]
