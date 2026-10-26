"""Camera-to-machine clear segment; derived geometry, no fixed meter offset.

The user approved a near-camera unobstructed position BETWEEN camera/machine.
Use the midpoint of the segment from the existing camera near clip to the first
visible mesh AABB entry (or the machine center if nearer). No camera properties
are written. Conservative boxes may reject a possible path, never pass through
a known scene box. Housing containing the camera itself is excluded explicitly.
"""
import math

MACHINE = '/World/Task75/Machine'


def ray_entry(origin, direction, low, high):
    near, far = 0., math.inf
    for o, d, lo, hi in zip(origin, direction, low, high):
        if abs(d) < 1e-12:
            if o < lo or o > hi:
                return None
            continue
        a, b = sorted(((lo-o)/d, (hi-o)/d))
        near, far = max(near,a), min(far,b)
        if near > far:
            return None
    return near if far > 0 else None


def choose(origin, target, boxes, near_clip):
    if not all(math.isfinite(v) for v in (*origin,*target,near_clip)) or near_clip <= 0:
        raise ValueError('Finite native geometry and positive original near clip required')
    distance=math.dist(origin,target)
    if distance <= near_clip:
        raise ValueError('No clear segment before the machine')
    direction=[(t-o)/distance for o,t in zip(origin,target)]
    first, blocker=distance, MACHINE
    for path,low,high in boxes:
        hit=ray_entry(origin,direction,low,high)
        if hit is not None and hit < first:
            first,blocker=hit,path
    if first <= near_clip:
        raise ValueError('No unobstructed near-camera flash position: '+blocker)
    along=(near_clip+first)/2
    position=[o+along*d for o,d in zip(origin,direction)]
    return dict(position_m=position, direction_to_machine=direction,
                distance_from_camera_m=along, first_obstruction_m=first,
                first_obstruction_path=blocker, original_camera_near_clip_m=near_clip,
                placement_rule='midpoint_of_clear_camera_to_machine_segment',
                center_ray_unobstructed=True, whole_emitter_volume_collision_tested=False)


def from_stage(stage, camera_path):
    from pxr import Usd, UsdGeom
    camera=stage.GetPrimAtPath(camera_path)
    origin=list(UsdGeom.XformCache().GetLocalToWorldTransform(camera).ExtractTranslation())
    clipping=UsdGeom.Camera(camera).GetClippingRangeAttr().Get()
    bbox=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render'])
    machine=stage.GetPrimAtPath(MACHINE)
    if not machine:
        raise ValueError('Original coffee machine required for flash placement')
    target=list(bbox.ComputeWorldBound(machine).ComputeAlignedRange().GetMidpoint())
    boxes=[]
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Boundable) or prim.IsA(UsdGeom.Camera):
            continue
        if UsdGeom.Imageable(prim).ComputeVisibility()=='invisible':
            continue
        box=bbox.ComputeWorldBound(prim).ComputeAlignedRange()
        if box.IsEmpty():
            continue
        low,high=list(box.GetMin()),list(box.GetMax())
        path=str(prim.GetPath())
        # A camera is mounted inside the head. Do not mistake its own housing
        # for the scene beyond its lens. Other robot meshes still participate.
        if path.startswith('/World/Robot/') and all(lo<=o<=hi for lo,o,hi in zip(low,origin,high)):
            continue
        boxes.append((path,low,high))
    return choose(origin,target,boxes,float(clipping[0]))
