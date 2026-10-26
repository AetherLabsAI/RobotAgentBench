import numpy as np
import pytest

from experiments.physics_fluid.tablecloth_native_self_collision import measure_self_crossings


def test_crossing_disconnected_triangles_is_detected():
    points = np.array([[-1,-1,0],[1,-1,0],[0,1,0],
                       [0,0,-.1],[0,0,.1],[.5,0,.1]], float)
    result = measure_self_crossings(points, [[0,1,2],[3,4,5]])
    assert result['pairs_over_tolerance'] == 1
    assert result['failing_face_pairs'] == [[0,1]]
    assert result['maximum_edge_piercing_depth_m'] == pytest.approx(.1)
    assert not result['full_self_contact_certified']


def test_folded_adjacent_triangles_and_separated_sheets_are_clear():
    points = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],
                       [0,0,2],[1,0,2],[0,1,2]], float)
    result = measure_self_crossings(points, [[0,1,2],[0,1,3],[4,5,6]])
    assert result['pairs_over_tolerance'] == 0
    assert not result['shared_vertex_pairs_tested']
    assert not result['coplanar_overlap_tested']


def test_invalid_capture_is_rejected():
    with pytest.raises(ValueError, match='Invalid captured'):
        measure_self_crossings([[0,0,float('nan')]]*3, [[0,1,2]])
