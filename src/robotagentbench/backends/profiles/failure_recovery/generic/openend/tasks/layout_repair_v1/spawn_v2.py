"""Deterministic initial-pose selection against current room collider geometry.

Read-only geometry; caller authors one initial root translation, before PhysX.
Native collision checks remain authoritative. No trajectory pose writes.
"""
import itertools
import numpy as np

def inside(point,polygon):
    x,y=point;value=False
    for a,b in zip(polygon,polygon[1:]+polygon[:1]):
        if (a[1]>y)!=(b[1]>y) and x<(b[0]-a[0])*(y-a[1])/(b[1]-a[1])+a[0]:value=not value
    return value

def choose(stage,manifest):
    from pxr import Usd,UsdGeom,UsdPhysics
    from openend.tasks.scene_requirements.source_geometry import floor_paths
    from openend.tasks.scene_requirements.station_scene_checks import intersects,triangles_hit_box
    from openend.tasks.layout_repair_v1.ready_envelope import load,bounds
    robot='/World/Robot';root=stage.GetPrimAtPath(robot)
    cache=UsdGeom.XformCache();bbox=UsdGeom.BBoxCache(0,[UsdGeom.Tokens.default_,UsdGeom.Tokens.render,UsdGeom.Tokens.proxy,UsdGeom.Tokens.guide],False,True)
    origin=np.asarray(cache.GetLocalToWorldTransform(root).ExtractTranslation())
    floors=floor_paths(stage)
    if not floors:raise ValueError('No physical floor for spawn')
    parts=[];robot_boxes=[];ready=load(stage)
    for p in Usd.PrimRange(stage.GetPseudoRoot(),Usd.TraverseInstanceProxies()):
        if not p.IsActive() or not p.HasAPI(UsdPhysics.CollisionAPI) or p.GetAttribute('physics:collisionEnabled').Get() is False:continue
        path=str(p.GetPath());r=bbox.ComputeWorldBound(p).ComputeAlignedRange()
        box=np.array([r.GetMin(),r.GetMax()],dtype=float)
        if not np.all(np.isfinite(box)) or np.any(box[0]>box[1]):raise ValueError('Invalid collider bounds '+path)
        if path.startswith(robot+'/'):
            robot_boxes.extend((box,bounds(stage,p,box,ready,cache)));continue
        if any(path==f or path.startswith(f+'/') for f in floors):continue
        tri=None
        if p.IsA(UsdGeom.Mesh):
            vs=p.GetAttribute('points').Get();indices=list(p.GetAttribute('faceVertexIndices').Get() or [])
            faces=[];offset=0
            for n in p.GetAttribute('faceVertexCounts').Get() or []:
                ids=indices[offset:offset+n];offset+=n
                faces.extend((ids[0],ids[i],ids[i+1]) for i in range(1,n-1))
            if vs is not None and len(vs) and faces:
                world=(np.c_[np.asarray(vs),np.ones(len(vs))]@np.asarray(cache.GetLocalToWorldTransform(p)))[:,:3]
                tri=world[np.asarray(faces,dtype=int)]
        parts.append((path,box,tri))
    if not robot_boxes:raise ValueError('No robot collider geometry')
    # A single enclosing body volume is conservative for all initial links.
    envelope=np.array([np.min([b[0] for b in robot_boxes],0),np.max([b[1] for b in robot_boxes],0)])
    polygons=manifest.get('overview',{}).get('floors',[])
    if not polygons:raise ValueError('No measured floor polygon')
    allpoints=np.array([v for polygon in polygons for v in polygon]);lo=allpoints.min(0);hi=allpoints.max(0)
    candidates=[origin[:2]]
    # Search resolution changes only initial placement, not simulation settings.
    candidates+=sorted((np.array(p) for p in itertools.product(np.arange(lo[0],hi[0],.1),np.arange(lo[1],hi[1],.1))),key=lambda p:float(np.linalg.norm(p-origin[:2])))
    rejected=[]
    for xy in candidates:
        delta=np.array([*(xy-origin[:2]),0.]);box=envelope+delta
        # Boundary samples include the whole perimeter; actual wall colliders
        # below also reject concave-floor bridges.
        perimeter=[(x,y) for x in np.linspace(box[0,0],box[1,0],11) for y in (box[0,1],box[1,1])]
        perimeter += [(x,y) for y in np.linspace(box[0,1],box[1,1],11) for x in (box[0,0],box[1,0])]
        if not all(any(inside(p,polygon) for polygon in polygons) for p in perimeter):continue
        hit=next((path for path,b,tri in parts if intersects(box,b) and (tri is None or triangles_hit_box(tri,box))),None)
        if hit:
            if len(rejected)<10:rejected.append(hit)
            continue
        return dict(position=[float(xy[0]),float(xy[1]),float(origin[2])],original_position=origin.tolist(),
            delta=delta.tolist(),robot_envelope=box.tolist(),obstacle_colliders=len(parts),
            original_rejections=rejected,method='current_scene_collider_envelope_and_floor_search',
            ready_posture_envelope='measured_40_frame_native_ready_pose_union',physics_certified=False)
    raise ValueError('No collision-free initial robot envelope in this layout: '+str(dict(envelope=envelope.tolist(),origin=origin.tolist(),rejected=rejected,floors=floors)))
