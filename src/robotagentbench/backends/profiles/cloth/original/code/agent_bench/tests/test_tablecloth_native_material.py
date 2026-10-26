import numpy as np
import pytest
from experiments.physics_fluid.tablecloth_native_material import woven_springs


def grid():
    x,y=np.meshgrid(np.arange(5)*.01,np.arange(6)*.01)
    return np.c_[x.ravel(),y.ravel()]


def energy(uv, q):
    p,k,_,rest,_=woven_springs(uv)
    return .5*np.sum(k*(np.linalg.norm(q[p[:,1]]-q[p[:,0]],axis=1)-rest)**2)


def test_symmetric_shear_energy_does_not_depend_on_diagonal_sign():
    uv=grid()
    plus=np.c_[uv[:,0]+.1*uv[:,1],uv[:,1],np.zeros(len(uv))]
    minus=np.c_[uv[:,0]-.1*uv[:,1],uv[:,1],np.zeros(len(uv))]
    assert energy(uv,plus)==pytest.approx(energy(uv,minus),rel=1e-12)
    assert energy(uv,np.c_[uv,np.zeros(len(uv))])==pytest.approx(0,abs=1e-20)


def test_bending_exists_in_both_axes_without_duplicates_or_fixed_vertices():
    uv=grid();p,k,d,rest,meta=woven_springs(uv)
    assert meta['counts']['bend_warp']==18
    assert meta['counts']['bend_weft']==20
    assert meta['counts']['shear_positive']==meta['counts']['shear_negative']==20
    assert len(np.unique(p,axis=0))==len(p)
    assert np.all(rest>0) and np.all(k>0)
    # An isometric cylinder has finite elastic restoring energy in either axis.
    for axis in (0,1):
        a=uv[:,axis];b=uv[:,1-axis];r=.02
        q=np.c_[r*np.sin(a/r),b,r*(1-np.cos(a/r))]
        assert energy(uv,q)>0
    # Relabelling vertices preserves the same physical network and energy.
    perm=np.random.default_rng(4).permutation(len(uv))
    assert energy(uv[perm],q[perm])==pytest.approx(energy(uv,q))


def test_rejects_invalid_materials_and_unstructured_grid():
    with pytest.raises(ValueError):woven_springs(grid()[:-1])
    with pytest.raises(ValueError):woven_springs(grid(),bend=float('nan'))
