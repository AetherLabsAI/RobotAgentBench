"""CPU side-wrist fitting against the bag's measured cross-section.

Fit a pose, not physics parameters or asset geometry. Keep the open anatomical
colliders outside the bag before closure; the unchanged native contact-stop
controller still supplies actual opposed contacts, not this geometry fit.
"""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution,minimize


def _configuration_polygon(hand_points,target_polygon):
    hand=hand_points[ConvexHull(hand_points).vertices]
    points=(hand[:,None,:]-target_polygon[None,:,:]).reshape(-1,2)
    hull=ConvexHull(points)
    return hull.equations


def fit_side_anchor(asset,target_section):
    target=np.asarray(target_section,dtype=float)
    target=target[ConvexHull(target).vertices]
    groups=asset.collision_groups()
    prepared=[]
    for closure in (0.,1.):
        points,_=asset.sample(closure)
        prepared.append({name:_configuration_polygon(points[ids,1:],target) for name,ids in groups.items()})
    def gaps(anchor,polygons):
        return {name:float(np.max(e[:,:2]@anchor+e[:,2])) for name,e in polygons.items()}
    def metrics(anchor):
        opening=gaps(anchor,prepared[0]);closed=gaps(anchor,prepared[1])
        return opening,closed
    def loss(anchor):
        opening,closed=metrics(anchor)
        palm=opening['hand.R']
        thumb=min(abs(d) for name,d in closed.items() if name.startswith('thumb.'))
        finger=min(abs(d) for name,d in closed.items() if name.startswith('f_'))
        penetration=np.minimum(list(opening.values()),0.)
        return float(1000*np.sum(penetration**2)+palm*palm+thumb*thumb+finger*finger)
    p,_=asset.sample(0.)
    # Search extents are geometric sums, not product-specific adjustments.
    lo=p[:,1:].min(0)+target.min(0);hi=p[:,1:].max(0)+target.max(0)
    bounds=list(zip(lo,hi))
    result=differential_evolution(loss,bounds,seed=0,popsize=8,maxiter=65,tol=.0001)
    result=minimize(loss,result.x,method='SLSQP',bounds=bounds,
        constraints=[dict(type='ineq',fun=lambda yz:np.array(list(metrics(yz)[0].values())))],
        options={'maxiter':100,'ftol':1e-12})
    opening,closed=metrics(result.x)
    # This is floating-point polygon computation tolerance, not a collision
    # offset, acquisition threshold or permission to force native attachment.
    if min(opening.values()) < -1e-8:raise ValueError('No nonpenetrating open side grip found')
    return np.array([asset.data['grip_center_local'][0],*result.x]),dict(
        method='measured_bag_minkowski_section_vs_anatomical_side_hand',
        open_gap_m=opening,closed_gap_m=closed,fit_loss=float(result.fun),
        optimizer_success=bool(result.success),native_opposed_contact_verified=False,
        physical_parameters_modified=False)
