"""Convex tabletop-local XY containment. No guessed tabletop dimensions.

Caller supplies actual target footprint and actual available-top outline in the
same measured surface frame. Exclusions include television/decor footprints.
This computes geometry only, never support or release evidence.
"""
import math


def cross(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])


def polygon(points):
    p = tuple(tuple(v) for v in points)
    if len(p) < 3 or any(len(v) != 2 or any(type(x) not in (int, float) or not math.isfinite(x) for x in v) for v in p):
        raise ValueError('Finite convex polygon required')
    if len(set(p)) != len(p):
        raise ValueError('Duplicate polygon vertex')
    area = sum(p[i][0]*p[(i+1)%len(p)][1]-p[(i+1)%len(p)][0]*p[i][1] for i in range(len(p)))
    if area == 0:
        raise ValueError('Degenerate polygon')
    if area < 0:
        p = tuple(reversed(p))
    # Every other vertex must lie on the interior side of every oriented edge.
    if any(cross(a, p[(i+1)%len(p)], v) < 0 for i,a in enumerate(p) for v in p):
        raise ValueError('Polygon must be convex and ordered')
    return p


def inside(point, outer):
    return all(cross(a, outer[(i+1)%len(outer)], point) >= 0 for i,a in enumerate(outer))


def overlap(a, b):
    # Separating axis test; touching an excluded footprint is conservatively blocked.
    for p in (a,b):
        for i,x in enumerate(p):
            y=p[(i+1)%len(p)];axis=(-(y[1]-x[1]), y[0]-x[0])
            pa=[sum(v[k]*axis[k] for k in (0,1)) for v in a]
            pb=[sum(v[k]*axis[k] for k in (0,1)) for v in b]
            if max(pa)<min(pb) or max(pb)<min(pa):return False
    return True


def fully_inside(footprint, tabletop, excluded):
    target, surface = polygon(footprint), polygon(tabletop)
    exclusions = tuple(polygon(p) for p in excluded)
    return all(inside(v,surface) for v in target) and not any(overlap(target,p) for p in exclusions)
