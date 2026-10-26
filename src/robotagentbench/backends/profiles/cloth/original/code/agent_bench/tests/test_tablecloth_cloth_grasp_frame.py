import numpy as np
import pytest
from experiments.physics_fluid.tablecloth_cloth_grasp_frame import material_edge_grasp_rotation

@pytest.mark.parametrize('mirror',[1.,-1.])
def test_physical_surface_frame_is_proper_for_both_hand_bases(mirror):
 uv=np.array([[u,v] for v in [0.,.02,.04] for u in [0.,.01,.02]])
 q=np.column_stack((uv,np.zeros(len(uv))));basis=np.diag([mirror,1.,1.])
 rot,ns=material_edge_grasp_rotation(q,uv,np.zeros(9),2,[.02,.04],[.01,.02],basis,1.)
 assert ns==[1,5];assert np.allclose(rot.T@rot,np.eye(3));assert np.isclose(np.linalg.det(rot),1.)
 assert np.allclose(rot@basis[:,2],[1,0,0]);assert np.allclose(rot@basis[:,0],[0,0,1])

def test_duplicate_material_coordinates_stay_with_target_cloth():
 uv=np.array([[u,v] for v in [0.,.02,.04] for u in [0.,.01,.02]])
 q=np.column_stack((uv,np.zeros(len(uv))));both=np.r_[q,q+10]
 rot,ns=material_edge_grasp_rotation(both,np.tile(uv,(2,1)),np.repeat([0,1],9),11,[.02,.04],[.01,.02],np.eye(3),1.)
 assert ns==[10,14];assert np.isclose(np.linalg.det(rot),1.)

def test_collapsed_patch_cannot_silently_create_a_grasp_frame():
 uv=np.array([[u,v] for v in [0.,.02,.04] for u in [0.,.01,.02]])
 with pytest.raises(ValueError,match='Collapsed'):
  material_edge_grasp_rotation(np.zeros((9,3)),uv,np.zeros(9),2,[.02,.04],[.01,.02],np.eye(3),1.)
