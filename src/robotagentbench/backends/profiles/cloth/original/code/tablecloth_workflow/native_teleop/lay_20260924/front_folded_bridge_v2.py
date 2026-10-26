"""Rigid tilt of the isometric folded bridge; fresh pre-episode coordinates only."""
import numpy as np
from front_folded_bridge_v1 import front_staggered_corner_drape as folded

def front_staggered_corner_drape(uv):
 q,record=folded(uv);a=np.deg2rad(-35.);rot=np.array([[1.,0.,0.],[0.,np.cos(a),-np.sin(a)],[0.,np.sin(a),np.cos(a)]]);origin=np.array([0.,-1.034,.462]);q=(q-origin)@rot.T+np.array([0.,-1.034,.470]);record.update(model='analytic_isometric_tilted_folded_sheet',rigid_initial_tilt_deg=-35.,front_rim_center_z_m=.470);return q,record
