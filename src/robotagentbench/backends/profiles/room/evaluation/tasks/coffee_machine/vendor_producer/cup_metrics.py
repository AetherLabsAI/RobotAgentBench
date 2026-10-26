import numpy as np

def measure_cup(fluid, cup_world_matrix):
    p=fluid.positions()
    inv=np.asarray(cup_world_matrix.GetInverse())
    local=p@inv[:3,:3]+inv[3,:3]
    h=local[:,1]
    radius=np.linalg.norm(local[:,[0,2]],axis=1)
    inner=np.interp(h,1.22*np.array([.007,.012,.072,.083,.087]),1.22*np.array([.022,.026,.033,.036,.0365]))
    inside=(radius<inner+.0003)&(h>.008)&(h<.107)
    n=len(p)
    return dict(emitted=fluid.count,simulated=n,in_cup=int(inside.sum()),
        outside_cup=int((~inside).sum()),retained_fraction=float(inside.mean()) if n else 1.,
        nominal_ml=n*fluid.particle_volume_ml,
        cup_nominal_ml=int(inside.sum())*fluid.particle_volume_ml,
        centroid_world=p.mean(axis=0).tolist() if n else None,
        centroid_cup=local.mean(axis=0).tolist() if n else None,
        cup_origin_world=list(cup_world_matrix.ExtractTranslation()))
