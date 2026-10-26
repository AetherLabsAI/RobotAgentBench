"""Task-local short-wrist actor review; no shared assets or physics changes.

Clip the source rest mesh, preserving skin weights and per-corner UVs. The
same clipped topology defines both the visual skin and collision membership.
This emits a STATIC review, not a tested replacement for the native actor.
"""
import copy
import json
import numpy as np
from scipy.optimize import minimize
from fit_static_grip import HERE, Rig, cylinder_sdf


def crop_wrist(source, plane=-.025):
    data=copy.deepcopy(source)
    points=np.asarray(source['frames'][0]['points'])
    vertices=[]; weights=[]; faces=[]; uvs=[]; keys={}
    boundary=set()
    def vertex(key, point, skin):
        if key not in keys:
            keys[key]=len(vertices);vertices.append(point.tolist());weights.append(skin)
            if abs(point[1]-plane)<1e-8:boundary.add(keys[key])
        return keys[key]
    offset=0
    for face in source['faces']:
        tex=np.asarray(source['uv'][offset:offset+len(face)]);offset+=len(face)
        polygon=[]
        for j,a in enumerate(face):
            b=face[(j+1)%len(face)];ta=tex[j];tb=tex[(j+1)%len(face)]
            pa,pb=points[a],points[b]
            if pa[1]>=plane:
                polygon.append((vertex(('v',a),pa,source['skin_weights'][a]),ta))
            if (pa[1]>=plane)!=(pb[1]>=plane):
                t=(plane-pa[1])/(pb[1]-pa[1]);p=pa+t*(pb-pa)
                wa,wb=source['skin_weights'][a],source['skin_weights'][b]
                w={k:(1-t)*wa.get(k,0)+t*wb.get(k,0) for k in wa.keys()|wb.keys()}
                polygon.append((vertex(('e',min(a,b),max(a,b)),p,w),ta+t*(tb-ta)))
        if len(polygon)>=3:
            faces.append([i for i,_ in polygon]);uvs.extend([uv.tolist() for _,uv in polygon])
    # The wrist is a single cross-section. Close its cut with a triangle fan.
    edge_count={}
    for face in faces:
        for a,b in zip(face,face[1:]+face[:1]):
            k=tuple(sorted((a,b)));edge_count[k]=edge_count.get(k,0)+1
    cut_edges=[(a,b) for (a,b),n in edge_count.items() if n==1 and a in boundary and b in boundary]
    neighbors={}
    for a,b in cut_edges:neighbors.setdefault(a,[]).append(b);neighbors.setdefault(b,[]).append(a)
    if not neighbors or any(len(v)!=2 for v in neighbors.values()):raise ValueError('Wrist cut is not a single closed ring')
    ring=[min(neighbors)]
    while True:
        next_id=next(v for v in neighbors[ring[-1]] if len(ring)==1 or v!=ring[-2])
        if next_id==ring[0]:break
        if next_id in ring:raise ValueError('Invalid cut loop')
        ring.append(next_id)
    if len(ring)!=len(neighbors):raise ValueError('Multiple cut loops')
    center=np.mean(np.asarray(vertices)[ring],axis=0)
    skin={k:sum(weights[i].get(k,0) for i in ring)/len(ring) for k in source['rest_bones']}
    ci=vertex(('cap',),center,skin)
    # Reuse a real wrist UV rather than introducing an unrelated cap texture.
    cap_uv=None
    cursor=0
    for face in faces:
        for j,i in enumerate(face):
            if i in boundary:cap_uv=uvs[cursor+j];break
        if cap_uv is not None:break
        cursor+=len(face)
    for a,b in zip(ring,ring[1:]+ring[:1]):
        f=[ci,a,b];p=np.asarray(vertices)[f]
        if np.cross(p[1]-p[0],p[2]-p[0])[1]>0:f.reverse()
        faces.append(f);uvs.extend([cap_uv]*3)
    groups={}
    for i,w in enumerate(weights):groups.setdefault(max(w,key=w.get),[]).append(i)
    data.update(faces=faces,uv=uvs,skin_weights=weights,collider_vertices=groups,
        frames=[dict(closure=0.,points=vertices,bones=source['rest_bones'])],
        actor_extent='hand_plus_25mm_wrist',physical_validation=False)
    return data


def main():
    out=HERE/'static_short_hand_011'
    if out.exists():raise FileExistsError(out)
    source=json.loads((HERE/'assets/derived/left_arm_forearm_rig.json').read_text())
    data=crop_wrist(source);rig=Rig(data)
    report=json.loads((HERE/'static_grip_010/fit.json').read_text())
    x=np.array(report['parameters']);spread=report['finger_spread_degrees']
    original=x[12:17].copy()
    thumb=np.unique(sum((v for k,v in rig.groups.items() if k.startswith('thumb.')),[]))
    tip=rig.groups['thumb.03.R']
    triangles=np.array([[f[0],f[j],f[j+1]] for f in data['faces'] for j in range(1,len(f)-1)])
    triangles=triangles[np.isin(triangles,thumb).any(axis=1)]
    # Lower the thumb from above the lid to the upper sidewall, keeping the
    # palm and the previously reviewed four-finger pose fixed.
    target=np.array([-.033,.071,.071])
    def objective(a):
        q=x.copy();q[12:17]=a;p,_=rig.pose(q,spread)
        sdf=cylinder_sdf(np.concatenate([p[thumb],p[triangles].mean(1)]),q[-3:])
        return (10000*np.square(np.minimum(sdf,0)).sum()
            +200*np.square(p[tip].mean(0)-target).sum()
            +1e-7*np.square(a-original).sum())
    best=min((minimize(objective,seed,method='L-BFGS-B',
        bounds=[(-65,65),(-65,65),(-65,65),(0,65),(0,65)],
        options={'maxiter':220,'ftol':1e-12})
        for seed in [original,np.array([-20,20,20,35,25])]),key=lambda r:r.fun)
    x[12:17]=best.x;p,bones=rig.pose(x,spread)
    data.update(frames=[dict(closure=0.,points=p.tolist(),bones={k:v.tolist() for k,v in bones.items()}),
                        dict(closure=1.,points=p.tolist(),bones={k:v.tolist() for k,v in bones.items()})],
                texture='../assets/model/new_diff.png',static_only=True)
    report=dict(center_local=report['center_local'],radius_m=report['radius_m'],
        height_m=report['height_m'],finger_spread_degrees=spread,
        source_pose='static_grip_010',optimizer_success=bool(best.success),
        parameters=x.tolist(),thumb_mean_local=p[tip].mean(0).tolist(),
        thumb_change_degrees=(best.x-original).tolist(),physical_validation=False,
        actor_extent=data['actor_extent'],original_scale_preserved=True,
        rest_wrist_cut_y_m=-.025,visual_vertices=len(p),
        collision_vertices=len(set(sum(data['collider_vertices'].values(),[]))),
        min_thumb_vertex_clearance_m=float(cylinder_sdf(p[thumb],x[-3:]).min()))
    out.mkdir()
    (out/'hand.json').write_text(json.dumps(data,separators=(',',':'))+'\n')
    (out/'fit.json').write_text(json.dumps(report,indent=2)+'\n')
    (out/'cropped_rig.json').write_text(json.dumps(crop_wrist(source),separators=(',',':'))+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
