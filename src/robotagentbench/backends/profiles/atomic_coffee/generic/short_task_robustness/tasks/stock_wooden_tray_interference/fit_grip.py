"""Offline CPU geometric fit of a cylinder to the authored finger curl.

No simulation/score parameters. Collision geometry is never reduced.
The candidate must still pass native contact and support validation.
"""
from pathlib import Path
import json
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize
try:
    from .human_asset import HumanAsset
except ImportError:
    from human_asset import HumanAsset


def polygon_distance(point, polygon):
    a=polygon; b=np.roll(a,-1,axis=0); edge=b-a
    t=np.clip(np.sum((point-a)*edge,axis=1)/np.sum(edge*edge,axis=1),0,1)
    distance=np.linalg.norm(point-a-t[:,None]*edge,axis=1).min()
    cross=edge[:,0]*(point-a)[:,1]-edge[:,1]*(point-a)[:,0]
    return -distance if np.all(cross>=0) or np.all(cross<=0) else distance


def fit(asset, radius):
    groups={k:v for k,v in asset.collision_groups().items()
            if k=='hand.R' or k.startswith(('f_','thumb.'))}
    candidates=[]
    # Fit at actual baked poses; test all preceding poses too, so the closing
    # path cannot go through the drink merely to reach a valid endpoint.
    histories=[]
    for frame in asset.frames:
        p=np.asarray(frame['points'])
        histories.append({k:p[v,1:][ConvexHull(p[v,1:]).vertices] for k,v in groups.items()})
    for index in range(6,len(histories)):
        current=histories[index]
        def loss(yz):
            ds=np.array([polygon_distance(yz,p) for h in histories[:index+1:2] for p in h.values()])
            final={k:polygon_distance(yz,p) for k,p in current.items()}
            gaps=[abs(final['hand.R']-radius), min(abs(d-radius) for k,d in final.items() if k.startswith('thumb')),
                  min(abs(d-radius) for k,d in final.items() if k.startswith('f_'))]
            return 1000*np.square(np.minimum(ds-radius,0)).sum()+sum(g*g for g in gaps)
        result=differential_evolution(loss,[(.065,.16),(.015,.095)],seed=0,popsize=8,maxiter=65,tol=.0001)
        def constraints(yz):
            return np.array([polygon_distance(yz,p)-radius for h in histories[:index+1] for p in h.values()])
        result=minimize(loss,result.x,method='SLSQP',bounds=[(.065,.16),(.015,.095)],
                        constraints=[dict(type='ineq',fun=constraints)],options={'maxiter':100,'ftol':1e-12})
        yz=result.x
        clear=min(polygon_distance(yz,p)-radius for h in histories[:index+1] for p in h.values())
        candidates.append(dict(closure=index/(len(histories)-1),center_local=[0.,*yz.tolist()],
            minimum_clearance_m=float(clear),fit_loss=float(result.fun)))
    # Keep only nonpenetrating candidates; prefer a deeper curl among equally
    # close fits rather than the old near-zero-closure hand.
    valid=[c for c in candidates if c['minimum_clearance_m']>=-1e-5]
    if not valid:
        return dict(radius_m=radius,selected=None,candidates=candidates,physical_validation=False)
    best=min(valid,key=lambda c:c['fit_loss'])
    return dict(radius_m=radius,selected=best,candidates=candidates,physical_validation=False)


if __name__=='__main__':
    here=Path(__file__).resolve().parent
    result=fit(HumanAsset(here/'assets/derived/left_arm_grip.json'),.0659136354457665/2)
    (here/'assets/derived/grip_fit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['selected']))
