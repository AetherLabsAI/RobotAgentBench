"""Independent whole CAD triangle/convex collision audit using actual body states."""
import sys, json
from pathlib import Path
import numpy as np
import fcl

def matrix(x):
    x, y, z, w = np.asarray(x, float) / np.linalg.norm(x)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
GEOMETRY_BOUNDS = {}
OBJECT_BOUNDS = {}
WORLD_BOUNDS = {}
KEEP = []

def remember(geom, points):
    points = np.asarray(points)
    GEOMETRY_BOUNDS[id(geom)] = np.array([points.min(0), points.max(0)])
    KEEP.append(geom)
    return geom

def collision_object(geom):
    obj = fcl.CollisionObject(geom)
    OBJECT_BOUNDS[id(obj)] = GEOMETRY_BOUNDS[id(geom)]
    WORLD_BOUNDS[id(obj)] = OBJECT_BOUNDS[id(obj)].copy()
    return obj

def place(obj, R, p):
    obj.setTransform(fcl.Transform(R, p))
    b = OBJECT_BOUNDS[id(obj)]
    c = R @ b.mean(0) + p
    h = np.abs(R) @ (np.ptp(b, axis=0) / 2)
    WORLD_BOUNDS[id(obj)] = np.array([c - h, c + h])

def overlap(a, b):
    x, y = (WORLD_BOUNDS[id(a)], WORLD_BOUNDS[id(b)])
    return bool(np.all(x[0] <= y[1]) and np.all(y[0] <= x[1]))

def convex(points, faces):
    return remember(fcl.Convex(np.asarray(points, np.float64), len(faces), np.column_stack((np.full(len(faces), 3), faces)).astype(np.int32).ravel()), points)

def surface(points, faces):
    m = fcl.BVHModel()
    m.beginModel(len(points), len(faces))
    m.addSubModel(np.asarray(points, np.float64), np.asarray(faces, np.int32))
    m.endModel()
    return remember(m, points)
from scipy.spatial import ConvexHull

def oriented_hull_faces(points):
    hull = ConvexHull(points)
    faces = hull.simplices.copy()
    triangles = np.asarray(points)[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    reverse = np.einsum('ij,ij->i', normals, hull.equations[:, :3]) < 0
    faces[reverse] = faces[reverse, ::-1]
    return faces

class CollisionGuard:

    def __init__(self, names, geometry, metadata, calibration):
        self.names = list(names)
        self.indices = {n: i for i, n in enumerate(names)}
        self.objects = []
        self.hand_equations = {}
        hulls = calibration
        for link in self.names:
            if not str(link).startswith('gripper_'):
                continue
            points = hulls['points_' + link]
            obj = collision_object(convex(points, oriented_hull_faces(points)))
            self.objects.append((str(link), obj, np.eye(3), np.zeros(3)))
            self.hand_equations[str(link)] = (ConvexHull(points).equations, points.min(0), points.max(0))
        for shape in metadata['nonhand_collision_shapes']:
            link = shape['link']
            p = shape.get('primitive')
            rot = np.eye(3)
            pos = np.zeros(3)
            if p:
                rot = matrix(p['quaternion_xyzw'])
                pos = np.array(p['position'])
                typ = p['type']
                if typ == 'box':
                    geom = fcl.Box(*np.array(p['half_extents']) * 2)
                elif typ == 'sphere':
                    geom = fcl.Sphere(p['radius'])
                elif typ == 'cylinder':
                    geom = fcl.Cylinder(p['radius'], p['half_height'] * 2)
                else:
                    raise ValueError(typ)
                half = np.array(p['half_extents']) if typ == 'box' else np.full(3, p['radius'])
                if typ == 'cylinder':
                    half[2] = p['half_height']
                remember(geom, np.array([-half, half]))
            else:
                geom = convex(shape['vertices'], shape['triangles'])
            self.objects.append((link, collision_object(geom), rot, pos))
        self.scene = [(e['prim_path'], collision_object(surface(geometry['scene_points_' + e['key']], geometry['scene_triangles_' + e['key']]))) for e in json.loads(str(geometry['scene_collision_json']))['meshes']]
        self.request = fcl.CollisionRequest(num_max_contacts=8, enable_contact=True)

    def examine(self, body, cloth):
        for link, obj, rot, pos in self.objects:
            xf = body[self.indices[link]]
            R = matrix(xf[3:])
            place(obj, R @ rot, R @ pos + xf[:3])
            for path, scene_obj in self.scene:
                if not overlap(obj, scene_obj):
                    continue
                result = fcl.CollisionResult()
                n = fcl.collide(obj, scene_obj, self.request, result)
                if n:
                    return dict(kind='robot_furniture_contact', link=link, scene=path, depth_m=max((float(c.penetration_depth) for c in result.contacts), default=0.0))
            if link in self.hand_equations:
                eq, lo, hi = self.hand_equations[link]
                local = (cloth - xf[:3]) @ R
                ids = np.flatnonzero((local >= lo).all(1) & (local <= hi).all(1))
                if len(ids):
                    d = (local[ids] @ eq[:, :3].T + eq[:, 3]).max(1)
                    hit = d < -0.001
                    if hit.any():
                        return dict(kind='cloth_hand_penetration', link=link, particles=ids[hit].tolist(), depth_m=float(-d.min()))
        left = [(n, o) for n, o, _, _ in self.objects if n.startswith(('arm_l', 'gripper_l'))]
        right = [(n, o) for n, o, _, _ in self.objects if n.startswith(('arm_r', 'gripper_r'))]
        for ln, lo in left:
            for rn, ro in right:
                if not overlap(lo, ro):
                    continue
                result = fcl.CollisionResult()
                n = fcl.collide(lo, ro, self.request, result)
                if n:
                    return dict(kind='cross_arm_collision', left=ln, right=rn)
        return None
