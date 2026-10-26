"""Scene-owned support inventory. Read geometry only; never authors USD.

Planning and static verification use geometry_query.cjs (the same JS kernel).
This module extracts facts and resolves the same support-set collider identities
for native contact classification. No scoring or physical parameters live here.
"""
from copy import deepcopy
import itertools
import json
import hashlib
from pathlib import Path
import shutil
import subprocess

SCHEMA = 'arena.support_surfaces.v2'


def geometry_signature(stage, path):
    """Verify cached surface geometry without loading offline polygon tooling."""
    from pxr import Usd, UsdGeom, UsdPhysics
    root = stage.GetPrimAtPath(path)
    if not root or not root.IsActive():
        raise ValueError('Missing registered support: '+path)
    cache = UsdGeom.XformCache(); digest = hashlib.sha256()
    for prim in Usd.PrimRange(root):
        image = UsdGeom.Imageable(prim)
        if image and image.ComputeVisibility() == 'invisible':
            continue
        digest.update(str(prim.GetPath()).encode())
        digest.update(str((prim.HasAPI(UsdPhysics.CollisionAPI),
                           prim.GetAttribute('physics:collisionEnabled').Get())).encode())
        triangles = _mesh_triangles(prim, cache)
        if triangles is not None:
            digest.update(triangles.astype('<f8').tobytes())
    return digest.hexdigest()


def verify_surfaces(stage, surfaces):
    for row in surfaces:
        if row.get('geometry_sha256') != geometry_signature(stage, row['path']):
            raise ValueError('Registered support geometry changed: '+row['path'])
    return surfaces


def collision_vertices(stage, root):
    import numpy as np
    from pxr import Usd, UsdGeom, UsdPhysics
    from evaluation.tasks.coffee_machine.isaac_evidence import collider_points
    cache = UsdGeom.XformCache(); rows = []
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
        if prim.HasAPI(UsdPhysics.CollisionAPI) and prim.GetAttribute('physics:collisionEnabled').Get() is not False:
            points, sphere = collider_points(prim, cache.GetLocalToWorldTransform(prim))
            if sphere is not None:
                center, radius = sphere
                points = center + radius*np.array(list(itertools.product((-1.,1.),repeat=3)))
            rows.extend(points)
    if not rows:
        raise ValueError('Missing enabled collision geometry: '+root)
    return np.asarray(rows)


def _mesh_triangles(prim, cache):
    import numpy as np
    from pxr import UsdGeom
    if prim.IsA(UsdGeom.Mesh):
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
        ids = list(mesh.GetFaceVertexIndicesAttr().Get())
        faces, start = [], 0
        for size in mesh.GetFaceVertexCountsAttr().Get():
            face = ids[start:start+size]; start += size
            faces.extend((face[0], face[i], face[i+1]) for i in range(1, size-1))
    elif prim.IsA(UsdGeom.Cube):
        h = UsdGeom.Cube(prim).GetSizeAttr().Get()/2
        points = np.asarray(list(itertools.product((-h, h), repeat=3)))
        faces = [(0,1,3),(0,3,2),(4,6,7),(4,7,5),(0,4,5),(0,5,1),
                 (2,3,7),(2,7,6),(0,2,6),(0,6,4),(1,5,7),(1,7,3)]
    else:
        return None
    if not len(points) or not faces:
        return None
    points = (np.c_[points, np.ones(len(points))] @
              np.asarray(cache.GetLocalToWorldTransform(prim)))[:, :3]
    return points[np.asarray(faces, dtype=int)]


def _convex_patches(shape):
    from shapely.ops import triangulate
    from shapely.geometry.polygon import orient
    parts = list(shape.geoms) if hasattr(shape, 'geoms') else [shape]
    result = []
    for part in parts:
        if part.is_empty or part.area <= 1e-12:
            continue
        if not part.interiors and part.convex_hull.area-part.area < 1e-12:
            result.append(list(map(list, orient(part, 1).exterior.coords))[:-1])
        else:
            # Clipping preserves holes/concavities. Never fill with the hull.
            for triangle in triangulate(part):
                clipped = triangle.intersection(part)
                if not clipped.is_empty and clipped.area > 1e-12:
                    result.extend(_convex_patches(clipped))
    return result


def extract_surface(stage, path, group, name=None):
    import numpy as np
    from pxr import Usd, UsdGeom, UsdPhysics
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    from shapely.geometry.polygon import orient
    root = stage.GetPrimAtPath(path)
    if not root or not root.IsActive():
        return None
    xc = UsdGeom.XformCache(); planes = {}; heights = {}; colliders = []
    for prim in Usd.PrimRange(root):
        image = UsdGeom.Imageable(prim)
        if image and image.ComputeVisibility() == 'invisible':
            continue
        if prim.HasAPI(UsdPhysics.CollisionAPI) and prim.GetAttribute('physics:collisionEnabled').Get() is not False:
            colliders.append(str(prim.GetPath()))
        triangles = _mesh_triangles(prim, xc)
        if triangles is None:
            continue
        for t in triangles:
            normal = np.cross(t[1]-t[0], t[2]-t[0])
            # Existing geometry-verification tolerances, not physical margins.
            if abs(normal[2]) <= 1e-12 or np.ptp(t[:,2]) > 1e-5:
                continue
            height = round(float(t[:,2].mean()), 6)
            poly = Polygon(t[:,:2])
            if poly.area > 1e-12:
                planes.setdefault(height, []).append(poly)
                heights[height] = max(heights.get(height, -float('inf')), float(t[:,2].max()))
    if not colliders or not planes:
        return None
    merged = {z: unary_union(parts) for z, parts in planes.items()}
    broad = {z: shape for z, shape in merged.items() if shape.area > .005}
    if not broad:
        return None
    largest = max(shape.area for shape in broad.values())
    # Same broad-plane 80% rule, now based on actual faces, not scattered vertices.
    top = max(z for z, shape in broad.items() if shape.area >= largest*.8)
    shape = broad[top]; top = heights[top]
    bbox = UsdGeom.BBoxCache(0, ['default','render','proxy']).ComputeWorldBound(root).ComputeAlignedRange()
    return dict(path=path, group=group, name=name or group, top=top,
                min=list(bbox.GetMin()), max=list(bbox.GetMax()),
                polygon=list(map(list, orient(shape.convex_hull, 1).exterior.coords))[:-1],
                support_patches=_convex_patches(shape), collider_paths=colliders,
                geometry_sha256=geometry_signature(stage, path),
                support_method='actual broad horizontal face union; holes retained',
                support_schema=SCHEMA)


def register_surfaces(stage, previous):
    """Register physical worktops and the explicitly approved stove surface."""
    from pxr import UsdGeom
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    requested = {s.get('source_wrapper', s['path']): (s['group'], s.get('name')) for s in previous}
    for prim in stage.Traverse():
        if prim.GetName() in ('Worktop','Stove'):
            image = UsdGeom.Imageable(prim)
            if image and image.ComputeVisibility() != 'invisible':
                requested.setdefault(str(prim.GetPath()), ('stove', 'Kitchen worktop'))
    rows = []
    for path, (group, name) in requested.items():
        row = extract_surface(stage, path, group, name)
        if row is not None:
            rows.append(row)
    shapes = [unary_union([Polygon(p) for p in row['support_patches']]) for row in rows]
    # A connected component names contacts; coverage still uses the exact union,
    # so the connectivity tolerance cannot fill a real gap or hole.
    parent = list(range(len(rows)))
    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i
    for i, a in enumerate(rows):
        for j in range(i):
            b = rows[j]
            if (a['group'] == b['group'] and abs(a['top']-b['top']) < .002
                    and shapes[i].distance(shapes[j]) <= 1e-5):
                parent[root(i)] = root(j)
    for i, row in enumerate(rows):
        row['support_set_id'] = rows[root(i)]['path']
    return rows


def update_layout(stage, layout):
    value = deepcopy(layout)
    value['surfaces'] = register_surfaces(stage, layout['surfaces'])
    value['support_schema'] = SCHEMA
    # Stove geometry remains an obstacle at its real height; being registered
    # does not exempt a candidate below it from penetration checks.
    paths = {b['path'] for b in value['blockers']}
    for row in value['surfaces']:
        if row['path'].endswith('/Stove') and row['path'] not in paths:
            value['blockers'].append(dict(path=row['path'], min=row['min'], max=row['max'],
                support_surface=True, support_top_m=row['top']))
    for table in value.get('tables', []):
        row = next((s for s in value['surfaces'] if s['path'] == table['path']), None)
        if row:
            table.update(support_top_m=row['top'],support_polygon=row['polygon'],height_delta_from_counter_m=row['top']-.92)
    return value


def surface_set(surfaces, anchor):
    selected = next((s for s in surfaces if s['path'] == anchor or anchor in s.get('collider_paths', [])), None)
    if selected is None:
        raise ValueError('Support anchor absent from registered inventory: '+anchor)
    key = selected.get('support_set_id', selected['path'])
    return [s for s in surfaces if s.get('support_set_id',s['path']) == key]


def manifest_surfaces(stage, manifest):
    old = manifest.get('overview', {}).get('surfaces', [])
    if old and all(s.get('support_schema') == SCHEMA for s in old):
        return verify_surfaces(stage, old)
    profiles = Path(__file__).with_name('placement_sources')/'random-layouts-v1.json'
    if profiles.is_file():
        rows = json.loads(profiles.read_text())['layouts']
        source_hash = manifest.get('source_sha256')
        if isinstance(source_hash, dict):
            # Older short-task manifests use this field for a dependency map,
            # not a scalar release hash. Verify the stated release file and
            # then every composed support's actual geometry signature.
            source = Path(manifest.get('source_layout', ''))
            source_hash = hashlib.sha256(source.read_bytes()).hexdigest() if source.is_file() else None
        match = next((r for r in rows if r['id'] == manifest.get('layout_id')
                      and r['scene_sha256'] == source_hash
                      and r.get('support_schema') == SCHEMA), None)
        if match is not None:
            return verify_surfaces(stage, match['surfaces'])
    # Offline compatibility path for non-release fixtures; native release
    # rooms resolve the verified precomputed registry above without Shapely.
    return register_surfaces(stage, old)


def contact_paths(stage, manifest, anchor):
    if not manifest.get('overview', {}).get('surfaces'):
        # Legacy fixtures expose only their explicit counter, never infer a
        # whole room from a missing inventory. The explicit support may be a
        # wrapper whose enabled collision mesh is a descendant (small table).
        from pxr import Usd, UsdPhysics
        root = stage.GetPrimAtPath(anchor)
        paths = tuple(str(p.GetPath()) for p in Usd.PrimRange(root)
                      if p.HasAPI(UsdPhysics.CollisionAPI)
                      and p.GetAttribute('physics:collisionEnabled').Get() is not False) if root else ()
        if not paths:
            raise ValueError('Missing enabled support collider: '+anchor)
        return paths
    return tuple(dict.fromkeys(p for s in surface_set(manifest_surfaces(stage, manifest), anchor)
                               for p in s['collider_paths']))


def check_footprints(layout, placement):
    """Call the identical kernel used by the JS planner; reset-time only."""
    executable = shutil.which('node')
    if executable is None:
        existing = Path.home()/'.local/bin/node'
        if existing.is_file(): executable = str(existing)
    if executable is None:
        raise RuntimeError('Existing Node scene-planner runtime required')
    query = Path(__file__).with_name('geometry_query.cjs')
    result = subprocess.run([executable, str(query)], input=json.dumps(dict(layout=layout,placement=placement)),
                            text=True,capture_output=True,timeout=30,check=True)
    return json.loads(result.stdout)


def check_station_footprints(stage, layout, group, vertices):
    import numpy as np
    from scipy.spatial import ConvexHull
    surfaces = layout['surfaces']
    selected = surface_set(surfaces, group['support'])
    anchor = next(s for s in selected if s['path']==group['support'] or group['support'] in s['collider_paths'])
    polygons = [v[ConvexHull(v[:,:2]).vertices,:2].tolist() for v in vertices]
    original = group.get('placement', {})
    placement = dict(support=group['support'], group=anchor['group'], top=anchor['top'],
                     polygon=original.get('polygon',polygons[0]),objects=polygons,
                     object_zranges=[[float(v[:,2].min()),float(v[:,2].max())] for v in vertices])
    return check_footprints(layout, placement)
