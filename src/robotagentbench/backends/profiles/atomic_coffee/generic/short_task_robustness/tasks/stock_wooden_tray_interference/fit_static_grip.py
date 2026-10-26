"""Static, CPU, source-weighted finger posing. Not a physics validation."""
from pathlib import Path
import argparse
import json
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

HERE=Path(__file__).resolve().parent
RADIUS=.066/2  # Original can's authored 66 mm body; no asset scaling.
HEIGHT=.1220000013709069
FINGERS=['index','middle','ring','pinky']


class Rig:
    def __init__(self,data):
        self.data=data
        self.names=list(data['rest_bones'])
        self.rest={k:np.array(v) for k,v in data['rest_bones'].items()}
        self.inv={k:np.linalg.inv(v) for k,v in self.rest.items()}
        self.points=np.array(data['frames'][0]['points'])
        self.weights={k:np.array([w.get(k,0) for w in data['skin_weights']]) for k in self.names}
        self.support={k:np.flatnonzero(w) for k,w in self.weights.items()}
        self.groups=data['collider_vertices']

    def pose(self,x,spread=None):
        spread=np.zeros(4) if spread is None else np.asarray(spread)
        rotations={}
        for i,f in enumerate(FINGERS):
            for j in range(3):rotations[f'f_{f}.{j+1:02d}.R']=[x[i*3+j],0,0]
        rotations['thumb.01.R']=x[12:15]
        rotations['thumb.02.R']=[x[15],0,0]
        rotations['thumb.03.R']=[x[16],0,0]
        bones={}
        def visit(k):
            if k in bones:return bones[k]
            parent=self.data['parents'][k]
            m=self.rest[k].copy() if parent is None else visit(parent)@self.inv[parent]@self.rest[k]
            if k in rotations:
                delta=np.eye(4);delta[:3,:3]=Rotation.from_euler('XYZ',rotations[k],degrees=True).as_matrix()
                if k.startswith('f_') and '.01.' in k:
                    index=FINGERS.index(k.split('.')[0][2:])
                    delta[:3,:3]=Rotation.from_euler('Z',spread[index],degrees=True).as_matrix()@delta[:3,:3]
                m=m@delta
            bones[k]=m
            return m
        out=np.zeros_like(self.points)
        for k in self.names:
            m=visit(k)@self.inv[k];ix=self.support[k]
            out[ix]+=(self.points[ix]@m[:3,:3].T+m[:3,3])*self.weights[k][ix,None]
        return out,bones


def cylinder_sdf(p,center):
    q=p-center
    d=np.stack([np.linalg.norm(q[:,1:],axis=1)-RADIUS,abs(q[:,0])-HEIGHT/2],axis=1)
    return np.linalg.norm(np.maximum(d,0),axis=1)+np.minimum(d.max(axis=1),0)


def fit():
    parser=argparse.ArgumentParser()
    parser.add_argument('--radius',type=float,default=RADIUS)
    parser.add_argument('--height',type=float,default=HEIGHT)
    parser.add_argument('--output',default='static_grip_007')
    args=parser.parse_args()
    # These describe the unchanged target asset, not a target scale override.
    globals()['RADIUS']=args.radius;globals()['HEIGHT']=args.height
    data=json.loads((HERE/'assets/derived/left_arm_forearm_rig.json').read_text());rig=Rig(data)
    hand=np.unique(np.concatenate([v for k,v in rig.groups.items() if k!='forearm.R']))
    distal={f:sum((v for k,v in rig.groups.items() if k.startswith('f_'+f) and ('.02.' in k or '.03.' in k)),[]) for f in FINGERS}
    thumb=rig.groups['thumb.03.R']
    palm_ids=np.unique(sum((v for k,v in rig.groups.items() if k=='hand.R' or k.startswith('palm_')),[]))
    palm_ids=palm_ids[(rig.points[palm_ids,1]>.075)&(rig.points[palm_ids,1]<.111)&(rig.points[palm_ids,2]>0)&(abs(rig.points[palm_ids,0])<.026)]
    if len(palm_ids)<3:raise ValueError('Source palm surface could not be identified')
    triangles=np.array([[f[0],f[j],f[j+1]] for f in data['faces'] for j in range(1,len(f)-1)])
    edges=np.array([(f[j],f[(j+1)%len(f)]) for f in data['faces'] for j in range(len(f))])
    # Seat the cylinder against the palm BEFORE closing fingers. Moving the
    # target toward the wrist to satisfy curled fingertips caused hook poses.
    anchor_y=.095
    dy=rig.points[palm_ids,1]-anchor_y
    eligible=palm_ids[abs(dy)<RADIUS]
    anchor_z=float(np.max(rig.points[eligible,2]+np.sqrt(RADIUS**2-(rig.points[eligible,1]-anchor_y)**2)))
    anchor=np.array([.003,anchor_y,anchor_z])
    def evaluate(x,details=False):
        p,bones=rig.pose(x);center=x[-3:];sdf=cylinder_sdf(p,center)
        loss=1000*np.square(np.minimum(sdf[hand],0)).sum()
        surface=np.concatenate([p[triangles].mean(1),p[edges].mean(1)])
        surface_sdf=cylinder_sdf(surface,center)
        loss+=1000*np.square(np.minimum(surface_sdf,0)).sum()
        contacts={}
        # A small central palm patch, not just one edge vertex, must touch the
        # can. This prevents a tip-only pinch with an empty palm cavity.
        palm_cost=sdf[palm_ids]**2+np.maximum(abs(p[palm_ids,0]-center[0])-HEIGHT*.35,0)**2
        palm_near=palm_ids[np.argsort(palm_cost)[:3]]
        loss+=200*np.mean(palm_cost[np.argsort(palm_cost)[:3]])
        contacts['palm']=dict(points=p[palm_near].tolist(),gap_m=float(np.max(sdf[palm_near])))
        for f,ids in {**distal,'thumb':thumb}.items():
            # Contacts on opposite halves are essential: a closed fist beside
            # the can cannot minimize this objective merely by being nearby.
            signed=(p[ids,1]-center[1])*(1 if f!='thumb' else -1)
            cost=sdf[ids]**2
            if f!='pinky':cost+=np.minimum(signed-RADIUS*.25,0)**2
            # Stay on the can sidewall, not a thumb resting on the upper rim.
            cost+=np.maximum(abs(p[ids,0]-center[0])-HEIGHT*.40,0)**2
            ii=ids[int(cost.argmin())]
            loss+=20*cost.min()
            contacts[f]=dict(point=p[ii].tolist(),gap_m=float(sdf[ii]),opposition_m=float((p[ii,1]-center[1])*(1 if f!='thumb' else -1)))
        # Fit a coherent half-closed hand, not twelve independently optimized
        # contact hooks. Adjacent fingers should have similar curvature.
        finger_angles=x[:12].reshape(4,3)
        natural=np.array([[25,50,25],[30,55,25],[35,60,30],[40,65,35]])
        loss+=2e-7*np.square(finger_angles-natural).sum()
        loss+=4e-7*np.square(np.diff(finger_angles,axis=0)).sum()
        if details:return p,bones,dict(loss=float(loss),center_local=center.tolist(),contacts=contacts,
            min_vertex_clearance_m=float(sdf[hand].min()),
            min_sampled_surface_clearance_m=float(surface_sdf.min()),physical_validation=False)
        return loss
    bounds=[(0,60),(0,90),(0,60)]*4+[(-60,60),(-65,65),(-65,65),(0,70),(0,70)]+[(anchor[0],anchor[0]),(anchor[1],anchor[1]),(anchor[2],anchor[2]+.004)]
    best=None
    for offset in (0.,10.):
        x=np.r_[np.array([[25,50,25],[30,55,25],[35,60,30],[40,65,35]]).ravel(),[-15,-15+offset,20,30,20],anchor]
        result=minimize(evaluate,x,method='L-BFGS-B',bounds=bounds,options={'maxiter':300,'ftol':1e-12,'maxfun':15000})
        if best is None or result.fun<best.fun:best=result
        print('FIT',float(result.fun),flush=True)
    p,bones,report=evaluate(best.x,True)
    report.update(radius_m=RADIUS,height_m=HEIGHT,parameters=best.x.tolist(),optimizer_success=bool(best.success))
    report['placement_method']='distal_palm_anchor_then_finger_closure_with_normal_clearance'
    out=HERE/args.output;out.mkdir(exist_ok=True)
    (out/'fit.json').write_text(json.dumps(report,indent=2)+'\n')
    data['frames']=[dict(closure=0.,points=p.tolist(),bones={k:v.tolist() for k,v in bones.items()}),
                    dict(closure=1.,points=p.tolist(),bones={k:v.tolist() for k,v in bones.items()})]
    data['static_only']=True;data['fit_report']='fit.json'
    data['texture']='../assets/model/new_diff.png'
    (out/'hand.json').write_text(json.dumps(data,separators=(',',':'))+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':fit()
