"""Plan CPU finger closure against the actual frozen anatomical rigid hulls.

Task-owned actor motion only: no target writes or collision/solver changes.
The old linear angle interpolation sweeps the little finger through the can.
This planner changes the joint path, rather than disabling those collisions.
"""
import json
import argparse
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from fit_static_grip import HERE, Rig
from human_asset import HumanAsset, transform_points


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--product',choices=['sprite_c1','gatorade_t'],default='sprite_c1')
    args=parser.parse_args()
    bottle=args.product=='gatorade_t'
    name='bottle_018' if bottle else '012'
    output=HERE/('assets/derived/left_hand_side_entry_'+name+'.json')
    if output.exists():raise FileExistsError(output)
    data=json.loads((HERE/'static_short_hand_011/cropped_rig.json').read_text())
    report=json.loads((HERE/('static_grip_010_bottle/fit.json' if bottle else 'static_short_hand_011/fit.json')).read_text())
    def cylinder_sdf(p,center):
        q=p-center
        d=np.stack([np.linalg.norm(q[:,1:],axis=1)-report['radius_m'],abs(q[:,0])-report['height_m']/2],axis=1)
        return np.linalg.norm(np.maximum(d,0),axis=1)+np.minimum(d.max(axis=1),0)
    old=HumanAsset(HERE/'assets/derived/left_hand_short_wrist_011.json')
    rig=Rig(data);spread=report['finger_spread_degrees']
    closed=np.array(report['parameters']);opened=closed.copy()
    opened[:12]=np.tile([10,10,5],4);opened[12:17]=old.data['entry_thumb_pose_degrees']
    # Pull the new hand back by the measured palm-hull intrusion, not by
    # changing collision offsets. The can itself never changes pose or size.
    center=np.array(report['center_local'])
    p0,b0=rig.pose(opened,spread)
    groups=old.collision_groups();local={};samples={}
    for k,ids in groups.items():
        p=transform_points(np.linalg.inv(b0[k]),p0[ids]);local[k]=p
        f=ConvexHull(p).simplices
        samples[k]=np.concatenate([p,p[f].mean(1),p[f[:,:2]].mean(1),p[f[:,1:]].mean(1)])
    palm=transform_points(b0['hand.R'],samples['hand.R'])
    while cylinder_sdf(palm,center).min()<0:center[2]+=.0001
    closed[-3:]=opened[-3:]=center
    def clearances(x,names):
        _,b=rig.pose(x,spread)
        return np.concatenate([cylinder_sdf(transform_points(b[k],samples[k]),center) for k in names])
    thumb_names=[k for k in samples if k.startswith('thumb.')]
    thumb_ids=groups['thumb.03.R']
    # Less extreme opposition than 011: pad against upper sidewall, thumb
    # joints gently flexed. Both skin and rigid hull are checked in fitting.
    desired=np.array([-.046,.065,.070])
    def thumb_cost(a):
        x=closed.copy();x[12:17]=a;p,_=rig.pose(x,spread)
        sdf=np.r_[clearances(x,thumb_names),cylinder_sdf(p[sum((groups[k] for k in thumb_names),[])],center)]
        return 1e5*np.square(np.minimum(sdf,0)).sum()+100*np.square(p[thumb_ids].mean(0)-desired).sum()+1e-7*np.square(a-np.array([-20,5,40,25,20])).sum()
    fit=minimize(thumb_cost,[-20,5,40,25,20],method='L-BFGS-B',
        bounds=[(-45,35),(-40,45),(0,55),(5,50),(5,50)],options={'maxiter':220,'ftol':1e-12})
    closed[12:17]=fit.x
    rng=np.random.default_rng(12)
    paths={}; diagnostics={}
    for digit,indices in [('index',slice(0,3)),('middle',slice(3,6)),('ring',slice(6,9)),('pinky',slice(9,12)),('thumb',slice(12,17))]:
        names=[k for k in samples if k.startswith('thumb.' if digit=='thumb' else 'f_'+digit+'.')]
        a=opened[indices].copy();z=closed[indices].copy()
        def clearance(v):
            x=opened.copy();x[indices]=v
            return float(clearances(x,names).min())
        # Native mesh approximation can differ sub-millimetrically from the
        # envelope. Never admit more intrusion than the actual endpoints.
        floor=min(0.,clearance(a),clearance(z))-1e-7
        def edge_ok(v,w):return all(clearance((1-t)*v+t*w)>=floor for t in np.linspace(0,1,13))
        if edge_ok(a,z):path=[a,z]
        else:
            bounds=np.array([[-35,65],[0,105],[0,85]] if digit!='thumb' else
                [[-60,50],[-50,65],[0,85],[0,65],[0,65]])
            nodes=[a];parents=[-1];goal=None
            for iteration in range(2400):
                target=z if iteration%5==0 else rng.uniform(bounds[:,0],bounds[:,1])
                ni=int(np.argmin([np.linalg.norm(n-target) for n in nodes]))
                delta=target-nodes[ni];length=np.linalg.norm(delta)
                candidate=nodes[ni]+delta*min(1.,12/max(length,1e-9))
                if not edge_ok(nodes[ni],candidate):continue
                nodes.append(candidate);parents.append(ni)
                if edge_ok(candidate,z):
                    goal=len(nodes)-1;break
            if goal is None:raise RuntimeError('No collision-admissible '+digit+' path')
            path=[z]
            while goal>=0:path.append(nodes[goal]);goal=parents[goal]
            path=list(reversed(path))
            # Deterministic shortcutting removes random detours when clear.
            for _ in range(80):
                if len(path)<3:break
                i,j=sorted(rng.choice(len(path),2,replace=False))
                if j>i+1 and edge_ok(path[i],path[j]):path=path[:i+1]+path[j:]
        lengths=np.r_[0.,np.cumsum([np.linalg.norm(v-w) for v,w in zip(path,path[1:])])]
        lengths=lengths/lengths[-1]
        paths[digit]=(indices,path,lengths)
        diagnostics[digit]=dict(joint_waypoints=[v.tolist() for v in path],minimum_endpoint_clearance_m=floor+1e-7)
        print('PLANNED',digit,len(path),flush=True)
    frames=[]; minimum=1.
    for t in np.linspace(0,1,81):
        x=opened.copy()
        for digit,(indices,path,lengths) in paths.items():
            for j in range(len(path)-1):
                if t<=lengths[j+1]+1e-9:
                    s=(t-lengths[j])/(lengths[j+1]-lengths[j]);x[indices]=(1-s)*path[j]+s*path[j+1];break
        p,b=rig.pose(x,spread);minimum=min(minimum,float(clearances(x,list(samples)).min()))
        frames.append(dict(closure=float(t),points=p.tolist(),bones={k:v.tolist() for k,v in b.items()}))
    data.update(frames=frames,texture='../model/new_diff.png',static_only=False,physical_validation=False,
        reviewed_pose='side_entry_'+name+'_unreviewed',grip_center_local=center.tolist(),target_product=args.product,
        target_radius_m=report['radius_m'],target_height_m=report['height_m'],
        closure_plan=diagnostics,minimum_sampled_hull_clearance_m=minimum)
    output.write_text(json.dumps(data,separators=(',',':'))+'\n')
    out=HERE/('static_side_entry_'+name);out.mkdir()
    static=dict(data,frames=[frames[-1],frames[-1]],static_only=True,texture='../assets/model/new_diff.png')
    (out/'hand.json').write_text(json.dumps(static,separators=(',',':'))+'\n')
    (out/'fit.json').write_text(json.dumps(dict(center_local=center.tolist(),height_m=report['height_m'],
        radius_m=report['radius_m'],parameters=closed.tolist(),physical_validation=False,
        minimum_sampled_hull_clearance_m=minimum,closure_plan=diagnostics),indent=2)+'\n')
    print('MINIMUM',minimum,'CENTER',center.tolist(),flush=True)


if __name__=='__main__':main()
