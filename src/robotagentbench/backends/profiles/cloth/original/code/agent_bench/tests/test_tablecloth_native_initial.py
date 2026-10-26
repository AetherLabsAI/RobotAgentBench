import numpy as np
import pytest
from experiments.physics_fluid.tablecloth_native_initial import rounded_basket_fold,initial_edge_ratios,refine_initial_mesh


def test_refinement_preserves_material_area_surface_and_separate_cloths():
    uv = np.array([[0.,0.],[1.,0.],[0.,1.]]*2)
    q = np.c_[uv, [0.,0.,0.,2.,2.,2.]]
    faces = np.array([[0,1,2],[3,4,5]])
    ids = np.array([0,0,0,1,1,1])
    refined, material, triangles, identities = refine_initial_mesh(q,uv,faces,ids)
    assert len(refined) == 12 and len(triangles) == 8
    assert np.all(np.diff(identities) >= 0)
    assert np.all(identities[triangles] == identities[triangles[:, :1]])
    assert np.array_equal(refined[:,2], identities*2.)
    assert np.allclose(initial_edge_ratios(refined,material,triangles),1.)
    for cid in (0,1):
        tri = material[triangles[np.all(identities[triangles]==cid,axis=1)]]
        a, b = tri[:,1]-tri[:,0], tri[:,2]-tri[:,0]
        area = (a[:,0]*b[:,1]-a[:,1]*b[:,0]).sum()/2
        assert area == pytest.approx(.5)
    with pytest.raises(ValueError,match='crosses cloth'):
        refine_initial_mesh(q,uv,np.array([[0,1,3]]),ids)


def test_rounded_folds_do_not_collapse_half_cell_at_a_crease():
    u,v=np.meshgrid(np.linspace(0,.342,37),np.linspace(0,.51,55));uv=np.c_[u.ravel(),v.ravel()]
    faces=[]
    for y in range(54):
        for x in range(36):
            a=y*37+x;faces.extend([[a,a+1,a+37],[a+1,a+38,a+37]])
    bounds=[[-2.439,-1.0340841,.26],[-2.041,-.7160841,.45]]
    q=rounded_basket_fold(uv,bounds)
    ratio=initial_edge_ratios(q,uv,np.asarray(faces))
    assert ratio.min()>.97
    assert ratio.max()<1.000001
    assert q[:,2].min()==pytest.approx(.289)
    assert q[:,2].max()>.45
    assert np.ptp(q[:,1])<.15
    assert np.array_equal(q,rounded_basket_fold(uv,bounds))
    with pytest.raises(ValueError):rounded_basket_fold(uv,bounds,radius=.001)
    exact=rounded_basket_fold(uv,bounds,isometric=True)
    assert np.allclose(initial_edge_ratios(exact,uv,np.asarray(faces)),1.,atol=1e-12,rtol=0)
    # Vertex order must not change the setup or its material metric.
    permutation=np.random.default_rng(8).permutation(len(uv))
    assert np.allclose(rounded_basket_fold(uv[permutation],bounds,isometric=True),exact[permutation])
