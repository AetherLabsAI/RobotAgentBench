"""Read-only geometry predicates. No centre-only placement or fixed cloth box."""
import numpy as np
from ..placement import region_hull


def vertices(value):
    p = np.asarray(value, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or not len(p) or not np.isfinite(p).all():
        raise ValueError('Require complete finite world geometry')
    return p


def projected_overlap(polygon, triangle):
    """Convex polygon/triangle SAT; touching edges count as overlap."""
    a, b = np.asarray(polygon), np.asarray(triangle)
    for poly in (a, b):
        for edge in np.roll(poly, -1, axis=0)-poly:
            axis = np.array([-edge[1], edge[0]])
            if np.dot(axis, axis) == 0:
                continue
            pa, pb = a@axis, b@axis
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False
    return True


def fully_outside_cloth(book_points, surface):
    hull = region_hull(vertices(book_points)[:, :2])
    points = vertices(surface.points)
    return not any(projected_overlap(hull, points[list(tri), :2]) for tri in surface.triangles)


def fully_over_table(book_points, table):
    xy = vertices(book_points)[:, :2]
    return bool(np.all(xy >= table['xy_min']) and np.all(xy <= table['xy_max']))


def on_table_face(point, normal_on_book, table, tolerance):
    return bool(normal_on_book[2] > .5 and abs(point[2]-table['z_m']) <= tolerance
                and np.all(np.asarray(point[:2]) >= table['xy_min'])
                and np.all(np.asarray(point[:2]) <= table['xy_max']))


def on_cloth_face(point, normal_on_book, surface, tolerance):
    """Upward contact near a current cloth triangle, not merely its AABB."""
    if normal_on_book[2] <= .5:
        return False
    p = np.asarray(point, dtype=float)
    points = vertices(surface.points)
    for tri in surface.triangles:
        a, b, c = points[list(tri)]
        u, v, w = b-a, c-a, p-a
        n = np.cross(u, v)
        length = np.linalg.norm(n)
        if length == 0:
            raise ValueError('Degenerate cloth triangle')
        if abs(np.dot(w, n)/length) > tolerance:
            continue
        uu, uv, vv = np.dot(u, u), np.dot(u, v), np.dot(v, v)
        den = uu*vv-uv*uv
        if den <= 0:
            raise ValueError('Invalid cloth triangle')
        s = (vv*np.dot(w, u)-uv*np.dot(w, v))/den
        t = (uu*np.dot(w, v)-uv*np.dot(w, u))/den
        if s >= 0 and t >= 0 and s+t <= 1:
            return True
    return False
