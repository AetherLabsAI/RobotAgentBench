"""Read-only visible geometry audit of the cloth footprint, including inherited USD."""
import argparse
import json
from pathlib import Path
from pxr import Usd, UsdGeom


def audit(stage, manifest):
    cloth = [r for r in manifest['objects'] if r['id'] == 'tablecloth_dirty']
    if len(cloth) != 1:
        raise ValueError('Exactly one manifest cloth required')
    cloth = cloth[0]['path']
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render'], useExtentsHint=False)
    box = cache.ComputeWorldBound(stage.GetPrimAtPath(cloth)).ComputeAlignedRange()
    low, high = list(box.GetMin()), list(box.GetMax())
    hits = {}
    rows = sorted(manifest['objects'], key=lambda r: -len(r['path']))
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh) and not prim.IsA(UsdGeom.Cube):
            continue
        image = UsdGeom.Imageable(prim)
        if image.ComputeVisibility() == 'invisible' or image.ComputePurpose() not in ('default', 'render'):
            continue
        path = str(prim.GetPath())
        if path.startswith(cloth + '/'):
            continue
        b = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        a, z = list(b.GetMin()), list(b.GetMax())
        if any(z[i] <= low[i] or a[i] >= high[i] for i in (0, 1)):
            continue
        # Ignore below-table fixtures but include any tabletop/cloth intersections.
        if z[2] < low[2] - .001 or a[2] > high[2] + .5:
            continue
        row = next((r for r in rows if path == r['path'] or path.startswith(r['path'] + '/')), None)
        owner = row['path'] if row else '/'.join(path.split('/')[:4])
        item = hits.setdefault(owner, dict(id=row['id'] if row else None, path=owner,
                                          manifest_bound=row is not None, meshes=[]))
        item['meshes'].append(dict(path=path, bounds=[a, z]))
    return dict(cloth_path=cloth, cloth_bounds=[low, high], overlapping_visible_geometry=list(hits.values()),
                evidence='USD visible bounds broadphase; not a physical contact test')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('manifest'); args = ap.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    print(json.dumps(audit(Usd.Stage.Open(manifest['scene']), manifest), indent=2))
