"""Read-only recheck of source active20 poses in the actual composed room.

Reuses existing FK, floor, collision and coupled-finger contracts unchanged.
Static validation is not native self-collision or manipulation certification.
"""
import copy
import hashlib
import math
from pathlib import Path
import numpy as np
from .postures import candidates, context, source_hashes

ROBOT = '/World/Robot'


def update_candidate(receipt, targets):
    # The existing generic station entry explicitly uses candidate=None.
    # Normalize only the receipt; no scene or physical state is authored.
    if not isinstance(receipt.get('candidate'),dict):
        receipt['candidate'] = {}
    receipt['candidate']['targets_rad'] = targets
    receipt['candidate']['sampling'] = 'whole_recorded_source_posture_no_joint_jitter'


class NoFeasibleSourcePosture(ValueError):
    pass


def checker(stage, manifest, base_xyyaw):
    from pxr import Usd, UsdGeom, UsdPhysics
    from scipy.spatial.transform import Rotation
    from arena_api.robot_profiles import FOLLOWERS
    from short_tasks.tasks.retrieve_from_drawer.prepare_initial import corners, fk
    from openend.tasks.scene_requirements.source_geometry import floor_paths
    from openend.tasks.scene_requirements.station_scene_checks import intersects, triangles_hit_box
    from openend.tasks.layout_repair_v1.spawn_v2 import inside
    if len(base_xyyaw) != 3 or not all(math.isfinite(v) for v in base_xyyaw):
        raise ValueError('Existing finite station XY/yaw required')
    floors, polygons = floor_paths(stage), manifest.get('overview',{}).get('floors',[])
    if not floors or not polygons:
        raise ValueError('Measured floor geometry required')
    cache = UsdGeom.XformCache()
    bbox = UsdGeom.BBoxCache(0,['default','render','proxy','guide'],False,True)
    robot_parts, obstacles, authored = [], [], {}
    for p in Usd.PrimRange(stage.GetPseudoRoot(),Usd.TraverseInstanceProxies()):
        if not p.IsActive():
            continue
        path = str(p.GetPath())
        if path.startswith(ROBOT+'/') and (p.IsA(UsdPhysics.RevoluteJoint) or p.IsA(UsdPhysics.PrismaticJoint)):
            prismatic = p.IsA(UsdPhysics.PrismaticJoint)
            attr = 'state:linear:physics:position' if prismatic else 'state:angular:physics:position'
            value = p.GetAttribute(attr).Get()
            if value is not None:
                authored[p.GetName()] = float(value)*(1 if prismatic else math.pi/180)
        if not p.HasAPI(UsdPhysics.CollisionAPI) or p.GetAttribute('physics:collisionEnabled').Get() is False:
            continue
        r = bbox.ComputeWorldBound(p).ComputeAlignedRange()
        box = np.array([r.GetMin(),r.GetMax()])
        if not np.isfinite(box).all() or np.any(box[0]>box[1]):
            raise ValueError('Invalid original collider bounds: '+path)
        if path.startswith(ROBOT+'/'):
            body = p
            while body and not body.IsPseudoRoot() and not body.HasAPI(UsdPhysics.RigidBodyAPI):
                body = body.GetParent()
            if not body or body.IsPseudoRoot():
                raise ValueError('Robot collider lacks rigid body')
            # Use the same body-local bound as the current shelf/fridge gate.
            # Inverting a WORLD AABB inflates rotated links and is not equivalent.
            relative = bbox.ComputeRelativeBound(p,body).ComputeAlignedRange()
            local_box = np.array([relative.GetMin(),relative.GetMax()])
            if not np.isfinite(local_box).all() or np.any(local_box[0]>local_box[1]):
                raise ValueError('Invalid original body-local collider bounds: '+path)
            local = np.c_[corners(local_box),np.ones(8)]
            robot_parts.append((path,str(body.GetPath()),local))
            continue
        if any(path==f or path.startswith(f+'/') for f in floors):
            continue
        triangles = None
        if p.IsA(UsdGeom.Mesh):
            vertices = p.GetAttribute('points').Get()
            ids = list(p.GetAttribute('faceVertexIndices').Get() or [])
            faces, offset = [], 0
            for n in p.GetAttribute('faceVertexCounts').Get() or []:
                face = ids[offset:offset+n]; offset += n
                faces.extend((face[0],face[i],face[i+1]) for i in range(1,n-1))
            if vertices is not None and len(vertices) and faces:
                vertices = (np.c_[np.array(vertices),np.ones(len(vertices))] @ np.array(cache.GetLocalToWorldTransform(p)))[:,:3]
                triangles = vertices[np.array(faces)]
        obstacles.append((path,box,triangles))
    if not robot_parts:
        raise ValueError('Actual robot collider geometry missing')
    x,y,yaw = base_xyyaw
    base = np.eye(4)
    base[:3,:3] = Rotation.from_euler('z',yaw).as_matrix()
    base[:3,3] = [x,y,0.]

    def check(pose):
        q = dict(authored)
        q.update(pose['targets_rad'])
        for inner,outer in FOLLOWERS.items():
            if outer not in q:
                raise ValueError('Missing source gripper coordinate')
            q[inner] = -q[outer]
        try:
            links = fk(stage,q,base)
        except ValueError as exc:
            if not str(exc).startswith('Reset outside unchanged joint limits:'):
                raise
            return str(exc)
        for path,body,local in robot_parts:
            v = (local @ links[body].T)[:,:3]
            box = np.array([v.min(0),v.max(0)])
            perimeter = [(a,b) for a in np.linspace(box[0,0],box[1,0],11) for b in (box[0,1],box[1,1])]
            perimeter += [(a,b) for b in np.linspace(box[0,1],box[1,1],11) for a in (box[0,0],box[1,0])]
            if not all(any(inside(v,p) for p in polygons) for v in perimeter):
                return 'robot_outside_floor:'+path
            hit = next((p for p,b,t in obstacles if intersects(box,b) and (t is None or triangles_hit_box(t,box))),None)
            if hit:
                return 'robot_room_overlap:'+path+':'+hit
        return None
    return check


def align_recorded_receipt(stage, manifest, plan, receipt, *, seed):
    """Replace ONLY joint posture; keep the preselected base and target reset."""
    out = copy.deepcopy(receipt)
    base = [*out['position'][:2],out['yaw_rad']]
    check = checker(stage,manifest,base)
    selection = context(plan,manifest)
    rejected = []
    for pose in candidates(**selection,seed=seed,previous_targets=out.get('targets_rad')):
        failure = check(pose)
        if failure:
            rejected.append({'source_id':pose['posture_source'],'reason':failure})
            continue
        out['previous_posture_validation_scope'] = out.get('method')
        out['targets_rad'] = pose['targets_rad']
        out['robot_posture_alignment'] = {
            **pose,'rejections':rejected,'base_xyyaw_preserved':base,
            'station_ranges_changed':False,'object_transforms_changed':False,
            'geometry_rechecked_with_source_joints':True,
            'active20_source_exact':True,'passive_internal_fingers': 'original_room_authored_coordinates_and_existing_coupling',
            'robot_self_collision_validated':False,
        }
        out['posture_source'] = pose['posture_source']
        out['robot_room_geometry_checked'] = True
        out['robot_self_collision_validated'] = False
        out['native_physics_validated'] = False
        update_candidate(out,pose['targets_rad'])
        refs = source_hashes()
        refs[str(Path(__file__))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        out.setdefault('reference_hashes',{}).update(refs)
        return out
    raise NoFeasibleSourcePosture('Source poses exhausted at unchanged station: '+str(rejected))


def align_receipt(stage, manifest, plan, receipt, *, seed):
    """Current default: one complete recorded medoid per short task."""
    from .geometry_fixed_v3 import align_receipt as align_fixed
    return align_fixed(stage, manifest, plan, receipt, seed=seed)
