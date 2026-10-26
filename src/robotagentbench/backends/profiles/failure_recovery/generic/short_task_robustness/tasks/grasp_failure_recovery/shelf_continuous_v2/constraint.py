"""Analytic XY sampling in the intersection of source bounds and 2–5 cm.

No robot coordinate changes, optimizer, IK, target-location table or seed retry.
Caller supplies actual fixed-finger collider points and original support Z.
"""
import math
import numpy as np

from .catalog import POLICY

def interval_on_line(axis,forward,s,lo,hi):
    """t such that lo <= s*axis + t*forward <= hi, without new tolerances."""
    lower,upper=-math.inf,math.inf
    for a,b,x,y in zip(axis,forward,lo,hi):
        if b==0.:
            if not x<=s*a<=y:return None
        else:
            left,right=(x-s*a)/b,(y-s*a)/b
            lower=max(lower,min(left,right));upper=min(upper,max(left,right))
    return (lower,upper) if lower<upper else None

def propose(tips,collision_offsets,footprint_offsets,base_rotation,z,source_xy,support_xy,rng):
    """Uniform gap and a uniform feasible depth, within the unchanged definition."""
    tips=np.asarray(tips,dtype=float);points=np.asarray(collision_offsets,dtype=float)
    footprint=np.asarray(footprint_offsets,dtype=float);basis=np.asarray(base_rotation,dtype=float)
    if not np.isfinite(tips).all() or not np.isfinite(points).all():raise ValueError('Finite geometry required')
    sign=int(rng.choice((-1,1)));gap=float(rng.uniform(.02,.05))
    axis=basis[:2,1]*sign;forward=basis[:2,0]
    if not np.allclose(basis[2,:],[0,0,1],atol=1e-7):raise ValueError('Upright robot base required')
    s=float((tips[:,:2]@axis).min()-gap-(points[:,:2]@axis).max())
    lo=np.maximum(np.asarray(source_xy[0]),np.asarray(support_xy[0])-footprint.min(0))
    hi=np.minimum(np.asarray(source_xy[1]),np.asarray(support_xy[1])-footprint.max(0))
    line=interval_on_line(axis,forward,s,lo,hi)
    result=dict(miss_sign=sign,requested_gap_m=gap,ik_calls=0)
    if line is None:return None,dict(result,reason='no_source_support_intersection')
    zlow,zhigh=points[:,2].min()+z,points[:,2].max()+z
    vertical=max(0.,float(tips[:,2].min()-zhigh),float(zlow-tips[:,2].max()))
    budget=.05**2-gap**2-vertical**2
    if budget<0:return None,dict(result,reason='height_outside_3d_gap_budget')
    depth_slack=math.sqrt(budget)
    depth=(float((tips[:,:2]@forward).min()-(points[:,:2]@forward).max()-depth_slack),
           float((tips[:,:2]@forward).max()-(points[:,:2]@forward).min()+depth_slack))
    lower,upper=max(line[0],depth[0]),min(line[1],depth[1])
    if not lower<upper:return None,dict(result,reason='no_finger_source_intersection')
    t=float(rng.uniform(lower,upper));xy=s*axis+t*forward
    moved=points+np.r_[xy,z]
    a,b=tips@basis,moved@basis
    delta=np.maximum(np.maximum(a.min(0)-b.max(0),b.min(0)-a.max(0)),0.)
    distance=float(np.linalg.norm(delta));signed=float((tips[:,:2]@axis).min()-(moved[:,:2]@axis).max())
    if not(.02<=distance<=.05 and .02<=signed<=.05):return None,dict(result,reason='measured_gap_rejected')
    return xy,dict(result,accepted_analytic=True,measured_lateral_gap_m=signed,
        conservative_3d_envelope_distance_m=distance,forward_interval_m=[lower,upper],
        source_bounds_xy_m=np.asarray(source_xy).tolist(),target_origin_xy_m=xy.tolist(),
        exact_mesh_surface_distance_verified=False)
