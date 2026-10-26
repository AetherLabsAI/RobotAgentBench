"""Owner planning diagnostic using authored collision shapes, including Shell.

This is a geometric planning filter, not a replacement for physical replay.
Mesh SDFs use their surface mesh; cached distal-finger convexes use the actual
PhysX cooking result. Other decomposed meshes use a conservative convex hull.
"""
from pathlib import Path
import itertools
import json
import sys

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from pxr import Usd, UsdGeom, UsdPhysics

import fcl


def pose(position, quaternion):
    out = np.eye(4)
    out[:3, :3] = Rotation.from_quat(np.asarray(quaternion)[[1, 2, 3, 0]]).as_matrix()
    out[:3, 3] = position
    return out


def world(prim):
    return np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)).T


def bvh(vertices, triangles):
    shape = fcl.BVHModel()
    shape.beginModel(len(vertices), len(triangles))
    shape.addSubModel(np.ascontiguousarray(vertices, dtype=np.float64),
                      np.ascontiguousarray(triangles, dtype=np.int32))
    shape.endModel()
    return shape


def faces(mesh):
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
    out = []
    cursor = 0
    for count in mesh.GetFaceVertexCountsAttr().Get():
        for k in range(1, count - 1):
            out.append([indices[cursor], indices[cursor+k], indices[cursor+k+1]])
        cursor += count
    return np.array(out, dtype=np.int32)


class CollisionModel:
    def __init__(self, stage, names, robot_root, cooked_path=None):
        self.stage = stage
        self.names = list(names)
        self.root = np.asarray(robot_root)
        self.joints = []
        self.all_edges = []
        for prim in Usd.PrimRange(stage.GetPrimAtPath('/World/Robot')):
            if not prim.IsA(UsdPhysics.Joint):
                continue
            joint = UsdPhysics.Joint(prim)
            a, b = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
            if not a or not b:
                continue
            self.all_edges.append((str(a[0]), str(b[0]), prim.GetTypeName()))
            if prim.IsA(UsdPhysics.SphericalJoint):
                continue
            row = dict(parent=str(a[0]), child=str(b[0]), name=prim.GetName(),
                       kind=prim.GetTypeName(), axis=prim.GetAttribute('physics:axis').Get() or 'Z')
            for side in (0, 1):
                rot = prim.GetAttribute(f'physics:localRot{side}').Get()
                row[f'T{side}'] = pose(prim.GetAttribute(f'physics:localPos{side}').Get(),
                                       [rot.GetReal(), *rot.GetImaginary()])
            self.joints.append(row)
        links = {j['parent'] for j in self.joints} | {j['child'] for j in self.joints}
        self.links = links
        self.base = (links - {j['child'] for j in self.joints}).pop()
        self.base_pose = self.root @ np.linalg.inv(world(stage.GetPrimAtPath('/World/Robot'))) @ world(stage.GetPrimAtPath(self.base))
        ordered, ready = [], {self.base}
        pending = self.joints.copy()
        while pending:
            batch = [j for j in pending if j['parent'] in ready]
            if not batch:
                raise ValueError('Disconnected robot joint graph')
            for row in batch:
                ordered.append(row)
                ready.add(row['child'])
                pending.remove(row)
        self.joints = ordered

        parent = {link: link for link in links}
        def representative(link):
            while parent[link] != link:
                parent[link] = parent[parent[link]]
                link = parent[link]
            return link
        for a, b, kind in self.all_edges:
            if a in links and b in links and kind == 'PhysicsFixedJoint':
                parent[representative(a)] = representative(b)
        groups = {link: representative(link) for link in links}
        adjacent = {frozenset((groups[a], groups[b])) for a, b, _ in self.all_edges if a in links and b in links}
        filtered = set()
        for prim in stage.Traverse():
            rel = prim.GetRelationship('physics:filteredPairs')
            if rel:
                filtered.update(frozenset((str(prim.GetPath()), str(other))) for other in rel.GetTargets())
        self.self_pairs = [(a, b) for a, b in itertools.combinations(sorted(links), 2)
                           if groups[a] != groups[b]
                           and frozenset((groups[a], groups[b])) not in adjacent
                           and frozenset((a, b)) not in filtered]

        self.cooked = {}
        if cooked_path:
            self.cooked = {r['path']: r for r in json.loads(Path(cooked_path).read_text())['meshes']}
        self.local_bounds = {}
        self.robot_shapes = {}
        self.static_shapes = []
        self.representations = []
        for prim in stage.Traverse():
            if not prim.HasAPI(UsdPhysics.CollisionAPI) or prim.GetAttribute('physics:collisionEnabled').Get() is False:
                continue
            path = str(prim.GetPath())
            if path.startswith('/World/Robot/'):
                node = prim
                while node and str(node.GetPath()) not in links:
                    node = node.GetParent()
                if not node:
                    raise ValueError('Collision shape has no FK body: ' + path)
                link = str(node.GetPath())
                transform = np.linalg.inv(world(node)) @ world(prim)
                shapes = self.make_shapes(prim, transform)
                self.robot_shapes.setdefault(link, []).extend(shapes)
            else:
                node = prim
                dynamic = False
                while node and not node.IsPseudoRoot():
                    if node.HasAPI(UsdPhysics.RigidBodyAPI):
                        dynamic = not bool(UsdPhysics.RigidBodyAPI(node).GetKinematicEnabledAttr().Get())
                        break
                    node = node.GetParent()
                if not dynamic:
                    self.static_shapes.extend(self.make_shapes(prim, world(prim)))

    def make_shapes(self, prim, transform):
        path, typ = str(prim.GetPath()), prim.GetTypeName()
        scale = np.linalg.norm(transform[:3, :3], axis=0)
        rigid = transform.copy()
        rigid[:3, :3] /= scale
        geometry = None
        if typ == 'Cube':
            geometry = fcl.Box(*(float(prim.GetAttribute('size').Get()) * scale))
        elif typ == 'Sphere':
            geometry = fcl.Sphere(float(prim.GetAttribute('radius').Get()) * max(scale))
        elif typ in ('Cylinder', 'Capsule'):
            axis = prim.GetAttribute('axis').Get() or 'Z'
            index = 'XYZ'.index(axis)
            radial = max(scale[i] for i in range(3) if i != index)
            radius = float(prim.GetAttribute('radius').Get()) * radial
            height = float(prim.GetAttribute('height').Get()) * scale[index]
            geometry = (fcl.Cylinder if typ == 'Cylinder' else fcl.Capsule)(radius, height)
            rotation = {'X': [0, np.pi/2, 0], 'Y': [-np.pi/2, 0, 0], 'Z': [0, 0, 0]}[axis]
            rigid[:3, :3] = rigid[:3, :3] @ Rotation.from_rotvec(rotation).as_matrix()
        elif typ == 'Mesh':
            mesh = UsdGeom.Mesh(prim)
            vertices = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
            approximation = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
            cooked = self.cooked.get(path)
            if cooked and approximation == 'convexDecomposition':
                if 'source_points' not in cooked or not np.array_equal(vertices, np.asarray(cooked['source_points'])):
                    raise ValueError('Cached cooking does not match authored mesh bounds: ' + path)
                results = []
                for index, convex in enumerate(cooked['convexes']):
                    v = np.asarray(convex['vertices']) @ transform[:3, :3].T + transform[:3, 3]
                    triangles = []
                    for polygon in convex['polygons']:
                        ids = convex['indices'][polygon['index_base']:polygon['index_base']+polygon['num_vertices']]
                        triangles.extend([ids[0], ids[k], ids[k+1]] for k in range(1, len(ids)-1))
                    results.append((path+f'/cooked_{index}', fcl.CollisionObject(bvh(v, triangles)), np.eye(4)))
                self.representations.append(dict(path=path, representation='actual_PhysX_cached_convexes'))
                for _, obj, _ in results:
                    # Every cooked convex was authored from the transformed
                    # mesh; its whole-mesh bounds are conservative.
                    all_v = vertices @ transform[:3, :3].T + transform[:3, 3]
                    self.local_bounds[id(obj)] = ((all_v.min(0)+all_v.max(0))/2, (all_v.max(0)-all_v.min(0))/2)
                return results
            v = vertices @ transform[:3, :3].T + transform[:3, 3]
            if approximation in ('convexHull', 'convexDecomposition', 'boundingCube') and np.linalg.matrix_rank(v-v.mean(0),tol=1e-9)<3:
                # Conservative planning-only thickness for coplanar imported box faces.
                # The live USD/PhysX collider is never changed by this diagnostic.
                corners=np.array(list(itertools.product([-.002,.002],repeat=3)))
                v=(v[:,None,:]+corners[None,:,:]).reshape(-1,3)
                self.representations.append(dict(path=path,coplanar_mesh_planning_inflation_m=.002,physics_geometry_changed=False))
            triangles = ConvexHull(v).simplices if approximation in ('convexHull', 'convexDecomposition', 'boundingCube') else faces(mesh)
            geometry = bvh(v, triangles)
            rigid = np.eye(4)
            self.representations.append(dict(path=path, representation=approximation, conservative_whole_hull=approximation == 'convexDecomposition'))
        else:
            raise ValueError('Unsupported authored collision geometry: ' + path + ' ' + typ)
        obj = fcl.CollisionObject(geometry)
        if typ == 'Mesh':
            center, half = (v.min(0)+v.max(0))/2, (v.max(0)-v.min(0))/2
        elif typ == 'Cube':
            center, half = np.zeros(3), float(prim.GetAttribute('size').Get())*scale/2
        elif typ == 'Sphere':
            center, half = np.zeros(3), np.repeat(float(prim.GetAttribute('radius').Get())*max(scale), 3)
        else:
            center = np.zeros(3)
            half = np.array([radius, radius, height/2 + (radius if typ == 'Capsule' else 0)])
        self.local_bounds[id(obj)] = (center, half)
        return [(path, obj, rigid)]

    def transforms(self, configuration):
        out = {self.base: self.base_pose}
        for joint in self.joints:
            motion = np.eye(4)
            if joint['kind'] != 'PhysicsFixedJoint':
                value = configuration[self.names.index(joint['name'])]
                axis = np.eye(3)['XYZ'.index(joint['axis'])]
                if joint['kind'] == 'PhysicsRevoluteJoint':
                    motion[:3, :3] = Rotation.from_rotvec(value * axis).as_matrix()
                else:
                    motion[:3, 3] = value * axis
            out[joint['child']] = out[joint['parent']] @ joint['T0'] @ motion @ np.linalg.inv(joint['T1'])
        return out

    def check(self, configuration, max_hits=10, clearance_m=0.0, allow_base_floor=False, pad_touch_tolerance_m=0.0):
        transforms = self.transforms(configuration)
        bounds = {}
        def bound(obj, m):
            center, half = self.local_bounds[id(obj)]
            center = m[:3,:3] @ center + m[:3,3]
            half = np.abs(m[:3,:3]) @ half
            bounds[id(obj)] = (center-half, center+half)
        for link, shapes in self.robot_shapes.items():
            for _, obj, offset in shapes:
                m = transforms[link] @ offset
                obj.setTransform(fcl.Transform(m[:3, :3], m[:3, 3]))
                bound(obj, m)
        for _, obj, m in self.static_shapes:
            obj.setTransform(fcl.Transform(m[:3, :3], m[:3, 3]))
            bound(obj, m)
        hits = []
        def intersect(a, b, kind):
            if (allow_base_floor and kind == 'world'
                    and a[0].startswith('/World/Robot/base_link/')
                    and b[0] == '/World/Background/Shell/Floor'):
                # The physical floor remains enabled. This permits only the
                # intended base support contact in the planning predicate.
                return
            alo, ahi = bounds[id(a[1])]
            blo, bhi = bounds[id(b[1])]
            if np.any(alo > bhi + clearance_m) or np.any(blo > ahi + clearance_m):
                return
            result = fcl.CollisionResult()
            if fcl.collide(a[1], b[1], fcl.CollisionRequest(num_max_contacts=3, enable_contact=True), result):
                depth=float(max((c.penetration_depth for c in result.contacts), default=0))
                same_hand_pads=any(all('/gripper_'+hand+'_' in path and '_link4/' in path for path in (a[0],b[0])) for hand in ('l','r'))
                # Closed native finger pads intentionally touch; only the planning
                # predicate permits numerical contact. Live colliders stay enabled.
                if kind=='self' and same_hand_pads and depth<=pad_touch_tolerance_m:
                    return
                hits.append(dict(kind=kind, a=a[0], b=b[0], depth_m=depth))
            elif clearance_m > 0 and not (kind == 'self' and '/gripper_' in a[0] and '/gripper_' in b[0]):
                distance = fcl.distance(a[1], b[1], fcl.DistanceRequest(), fcl.DistanceResult())
                if distance < clearance_m:
                    hits.append(dict(kind=kind, a=a[0], b=b[0], clearance_m=float(distance)))
        for a, b in self.self_pairs:
            for sa in self.robot_shapes.get(a, []):
                for sb in self.robot_shapes.get(b, []):
                    intersect(sa, sb, 'self')
                    if len(hits) >= max_hits:
                        return hits
        static_lo = np.array([bounds[id(b[1])][0] for b in self.static_shapes])
        static_hi = np.array([bounds[id(b[1])][1] for b in self.static_shapes])
        for link, shapes in self.robot_shapes.items():
            for a in shapes:
                alo, ahi = bounds[id(a[1])]
                possible = np.flatnonzero(np.all(alo <= static_hi+clearance_m,axis=1) & np.all(static_lo <= ahi+clearance_m,axis=1))
                for index in possible:
                    b = self.static_shapes[index]
                    intersect(a, b, 'world')
                    if len(hits) >= max_hits:
                        return hits
        return hits


def capture_current_gripper_convexes(stage, output):
    """Read live PhysX cooking for this stage, preserving actual concave pad gaps."""
    import omni.usd
    import omni.physx
    from pxr import PhysicsSchemaTools
    api=omni.physx.get_physx_cooking_interface()
    stage_id=omni.usd.get_context().get_stage_id()
    rows=[]
    for prim in Usd.PrimRange(stage.GetPrimAtPath('/World/Robot')):
        if '/gripper_' not in str(prim.GetPath()) or not prim.IsA(UsdGeom.Mesh) or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if prim.GetAttribute('physics:collisionEnabled').Get() is False:
            continue
        if UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()!='convexDecomposition':
            continue
        points=np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get())
        row={'path':str(prim.GetPath()),'source_points':points.tolist()}
        def done(result,convexes):
            row['result']=int(result)
            row['convexes']=[{'vertices':[[float(v.x),float(v.y),float(v.z)] for v in c.vertices], 'indices':list(c.indices), 'polygons':[{'index_base':int(p.index_base),'num_vertices':int(p.num_vertices)} for p in c.polygons]} for c in convexes]
        api.request_convex_collision_representation(stage_id,PhysicsSchemaTools.sdfPathToInt(row['path']),False,done)
        if row.get('result')!=0 or not row.get('convexes'):
            raise RuntimeError('Cannot inspect current native gripper cooking: '+row['path'])
        rows.append(row)
    if not rows:
        raise RuntimeError('No native gripper convex decomposition was found')
    Path(output).write_text(json.dumps({'collision_shapes_changed':False,'meshes':rows}))
    return output
