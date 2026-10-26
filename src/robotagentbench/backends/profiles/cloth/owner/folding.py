"""Approved material-pair fold detector, not a compact-AABB/crumpling detector."""
import math
import numpy as np


def fold_measure(mesh,points,*,overlap_fraction_min,layer_distance_max_m,normal_deviation_max_deg,creases=None):
    if not 0<overlap_fraction_min<=1 or layer_distance_max_m<=0 or not 0<normal_deviation_max_deg<90:
        raise ValueError('Explicit approved geometric policy required')
    q=mesh.positions(points);uv=mesh.uv;n=mesh.vertex_normals(q);w=mesh.node_weights
    levels=[np.unique(uv[:,a])for a in (0,1)]
    if len(levels[0])*len(levels[1])!=len(uv):raise ValueError('Structured material-grid pairing required')
    indices=np.column_stack([np.searchsorted(levels[a],uv[:,a])for a in (0,1)])
    grid=np.full((len(levels[0]),len(levels[1])),-1,dtype=int);grid[indices[:,0],indices[:,1]]=np.arange(len(uv))
    if np.any(grid<0):raise ValueError('Incomplete or duplicated UV grid')
    candidates=creases if creases is not None else [(a,float((levels[a][0]+levels[a][-1])/2))for a in (0,1)]
    cosine=math.cos(math.radians(normal_deviation_max_deg));rows=[]
    for axis,middle in candidates:
        if axis not in (0,1)or not levels[axis][0]<middle<levels[axis][-1]:raise ValueError('Invalid material crease')
        reflected=2*middle-uv[:,axis];a=levels[axis]
        hi=np.searchsorted(a,reflected).clip(0,len(a)-1);lo=(hi-1).clip(0,len(a)-1)
        nearest=np.where(abs(a[lo]-reflected)<abs(a[hi]-reflected),lo,hi)
        partner=indices.copy();partner[:,axis]=nearest;paired=grid[partner[:,0],partner[:,1]]
        # Exclude the shared crease band: pinching adjacent points is not folding.
        valid=(abs(a[nearest]-reflected)<=64*np.finfo(np.float32).eps*max(np.ptp(a),1e-30)) & (abs(uv[:,axis]-middle)>layer_distance_max_m)
        distance=np.linalg.norm(q-q[paired],axis=1)
        opposing=np.einsum('ij,ij->i',n,n[paired])<=-cosine
        close=valid&(distance<=layer_distance_max_m)&opposing
        sides=[uv[:,axis]<middle,uv[:,axis]>middle]
        ratios=[float(w[side&close].sum()/w[side].sum())for side in sides]
        # One coherent fold must have approximately opposed panel normals.
        dominant=[];coherence=[]
        for side in sides:
            normal=(n[side]*w[side,None]).sum(0);norm=np.linalg.norm(normal)
            normal=normal/norm if norm else np.zeros(3)
            dominant.append(normal)
            coherence.append(float(w[side&(n@normal>=cosine)].sum()/w[side].sum()))
        panels_opposed=bool(np.dot(dominant[0],dominant[1])<=-cosine)
        accepted=bool(min(ratios+coherence)>=overlap_fraction_min and panels_opposed)
        rows.append(dict(axis=int(axis),crease_material_m=middle,side_matched_area_fractions=ratios,
            side_normal_coherence_fractions=coherence,panels_opposed=panels_opposed,folded=accepted))
    return dict(folded=any(r['folded']for r in rows),candidates=rows,method='UV_material_pairs_and_opposed_coherent_panels',
                layer_distance_max_m=layer_distance_max_m,overlap_fraction_min=overlap_fraction_min,
                normal_deviation_max_deg=normal_deviation_max_deg)
