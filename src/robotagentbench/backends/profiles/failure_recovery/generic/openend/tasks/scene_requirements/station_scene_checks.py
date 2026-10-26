"""Conservative post-composition visible-solid bounds gate, not physics certification."""
import itertools
import numpy as np
from pxr import UsdGeom

EPS=1e-7  # Existing planar numerical epsilon, not a physical clearance.

def intersects(a,b):
    return bool(np.all(np.minimum(a[1],b[1])-np.maximum(a[0],b[0])>EPS))

def triangles_hit_box(triangles,box):
    # Triangle/AABB separating axes. Unlike a whole Worktop AABB, this respects
    # the raised rear lip without treating its height as a solid sheet of wood.
    center=np.mean(box,axis=0);half=(box[1]-box[0])/2-EPS
    v=triangles-center
    keep=np.all(v.min(1)<=half,axis=1)&np.all(v.max(1)>=-half,axis=1)
    v=v[keep]
    if not len(v):return False
    edges=np.roll(v,-1,axis=1)-v
    axes=[np.cross(edges[:,0],edges[:,1])]
    axes.extend(np.cross(edges[:,i],axis) for i in range(3) for axis in np.eye(3))
    valid=np.ones(len(v),dtype=bool)
    for a in axes:
        length=np.linalg.norm(a,axis=1);nonzero=length>1e-14
        a=a/np.maximum(length[:,None],1e-30)
        projections=np.einsum('nvi,ni->nv',v,a);radius=abs(a)@half
        valid &= ~nonzero|((projections.min(1)<=radius)&(projections.max(1)>=-radius))
    return bool(valid.any())

def check(stage,groups,bounds_of,*,layout=None):
    cache=UsdGeom.XformCache();parts=[]
    for p in stage.Traverse():
        im=UsdGeom.Imageable(p)
        if not im or im.ComputeVisibility()=='invisible' or im.ComputePurpose() not in ('default','render'):continue
        if p.IsA(UsdGeom.Mesh):
            vs=p.GetAttribute('points').Get();faces=[];offset=0
            indices=list(p.GetAttribute('faceVertexIndices').Get() or [])
            for count in p.GetAttribute('faceVertexCounts').Get() or []:
                ids=indices[offset:offset+count];offset+=count
                faces.extend((ids[0],ids[i],ids[i+1]) for i in range(1,count-1))
        elif p.IsA(UsdGeom.Cube):
            h=UsdGeom.Cube(p).GetSizeAttr().Get()/2;vs=list(itertools.product((-h,h),repeat=3))
            faces=[(0,1,3),(0,3,2),(4,6,7),(4,7,5),(0,4,5),(0,5,1),(2,3,7),(2,7,6),(0,2,6),(0,6,4),(1,5,7),(1,7,3)]
        else:continue
        if vs is None or not len(vs):continue
        a=(np.c_[np.asarray(vs),np.ones(len(vs))]@np.asarray(cache.GetLocalToWorldTransform(p)))[:,:3]
        triangles=a[np.asarray(faces,dtype=int)] if faces else None
        parts.append((str(p.GetPath()),np.array([a.min(0),a.max(0)]),triangles,p.IsA(UsdGeom.Cube)))
    failures=[];tested=0;kernel_checks=[]
    for group in groups:
        registered = layout is not None and any(s['path']==group['support'] or group['support'] in s.get('collider_paths',[]) for s in layout['surfaces'])
        allowed=group['members']+group.get('internal_supports',[])+([] if registered else [group['support']])
        if registered:
            from openend.tasks.scene_requirements.surface_registry import check_station_footprints,collision_vertices
            measured={path:collision_vertices(stage,path) for path in group.get('support_members',group['members'])}
            vertices=list(measured.values())
            kernel=check_station_footprints(stage,layout,group,vertices)
            kernel_checks.append(dict(station=group['id'],**kernel))
            if not kernel['passed']:
                failures.append(dict(station=group['id'],target=group['members'][0],
                    obstacle='support_registry:'+str(kernel['reason']),reason=kernel['reason'],
                    target_bounds=bounds_of(group['members'][0]).tolist(),obstacle_bounds=bounds_of(group['members'][0]).tolist()))
        for target in group['members']:
            box=np.array([measured[target].min(0),measured[target].max(0)]) if registered and target in measured else bounds_of(target)
            for path,bounds,triangles,is_cube in parts:
                if any(path==p or path.startswith(p+'/') for p in allowed):continue
                tested+=1
                hit=intersects(box,bounds) and (is_cube or triangles is None or triangles_hit_box(triangles,box))
                if hit:failures.append(dict(station=group['id'],target=target,obstacle=path,
                    target_bounds=box.tolist(),obstacle_bounds=bounds.tolist()))
    return dict(passed=not failures,method='actual_composed_station_bounds_vs_descendant_mesh_triangles_and_cube_bounds',
        tested_pairs=tested,station_count=len(groups),failures=failures,
        geometry_kernel_checks=kernel_checks,
        allowed_contacts='same approved assembly; registered support contacts without penetration',
        limitation='Conservative target AABBs against room surfaces; not exact mesh-mesh penetration, closed-mesh containment, full-room settling or physical execution')
