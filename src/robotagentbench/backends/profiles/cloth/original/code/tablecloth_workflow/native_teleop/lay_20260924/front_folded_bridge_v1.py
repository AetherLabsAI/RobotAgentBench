"""Isometric nonintersecting initial folded sheet bridging basket rim.
All coordinates are authored before episode start; gravity creates the interior drape.
No geometric projection, stored physical state, pinning or runtime object writes.
"""
import numpy as np

def front_staggered_corner_drape(uv):
 uv=np.asarray(uv,float);u=uv[:,0];v=uv[:,1];levels,first,inv=np.unique(v,return_index=True,return_inverse=True);mid=levels[np.argmin(abs(levels-(levels.max()/2+.04)))];radius=.012;distance=abs(levels-mid);theta=np.minimum(distance/radius,np.pi/2)
 lateral=radius*np.sin(theta)+np.maximum(distance-np.pi*radius/2,0);height=np.sign(levels-mid)*radius*(1-np.cos(theta));curve=np.c_[lateral,height];segments=np.diff(curve,axis=0);segments*=(np.diff(levels)/np.linalg.norm(segments,axis=1))[:,None];curve_exact=np.vstack((np.zeros(2),np.cumsum(segments,axis=0)));k=np.argmin(abs(levels-mid));curve_exact+=curve[k]-curve_exact[k]
 q=np.c_[-2.145+(mid-levels.max()/2)-curve_exact[inv,0],-1.034-(u-.292),.462+curve_exact[inv,1]]
 return q,dict(model='analytic_isometric_folded_bridge_before_gravity',fold_radius_m=radius,fold_material_v_m=float(mid),outside_material_strip_m=.05,layer_edge_stagger_m=float(2*mid-levels.max()),initialization_only=True,stored_physical_state_imported=False,metric_projection=False,interior_drape='gravity and actual basket contacts during fresh settling')
