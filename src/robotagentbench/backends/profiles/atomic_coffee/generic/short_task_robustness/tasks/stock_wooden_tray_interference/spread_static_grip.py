"""Static MCP abduction candidates; palm and target transforms stay fixed."""
import argparse
import json
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import fit_static_grip as fit


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',default='static_grip_008_bottle')
    parser.add_argument('--output',default='static_grip_010_bottle')
    args=parser.parse_args()
    report=json.loads((fit.HERE/args.source/'fit.json').read_text())
    data=json.loads((fit.HERE/'assets/derived/left_arm_forearm_rig.json').read_text())
    rig=fit.Rig(data);base=np.array(report['parameters'])
    fit.RADIUS=report['radius_m'];fit.HEIGHT=report['height_m']
    groups={k:np.array(v) for k,v in rig.groups.items() if k.startswith('f_') and len(v)>3}
    finger_ids={f:np.unique(np.concatenate([v for k,v in groups.items() if k.startswith('f_'+f)])) for f in fit.FINGERS}
    triangles=np.array([[f[0],f[j],f[j+1]] for f in data['faces'] for j in range(1,len(f)-1)])
    edges=np.array([(f[j],f[(j+1)%len(f)]) for f in data['faces'] for j in range(len(f))])
    def metrics(values,details=False):
        spread=values[:4]
        angles=base.copy();angles[:12]+=values[4:]
        p,_=rig.pose(angles,spread)
        samples=np.concatenate([p,p[triangles].mean(1),p[edges].mean(1)])
        target=fit.cylinder_sdf(samples,base[-3:])
        # Per-phalange hull checks, not one hull spanning a curled finger's
        # empty interior. This is a geometric proxy, not a physical collider.
        pairs={};total=0.
        for k,ids in groups.items():
            f=k.split('.')[0][2:]
            hull=ConvexHull(p[ids]);eq=hull.equations
            for other,oid in finger_ids.items():
                if other==f:continue
                depth=np.maximum(-(p[oid]@eq[:,:3].T+eq[:,3]).max(1),0)
                total+=float(np.square(depth).sum())
                key=' / '.join(sorted([f,other]))
                pairs[key]=max(pairs.get(key,0.),float(depth.max()))
        if details:return dict(finger_hull_penetration_m=pairs,
            min_target_proxy_clearance_m=float(target.min()))
        # Gentle abduction, with an explicit penalty for finger intersections.
        desired=np.array([10.,2.,-3.,-6.])
        return 1e-6*np.square(spread-desired).sum()+10000*total+10000*np.square(np.minimum(target,0)).sum()
    before=metrics(np.zeros(16),True)
    starting_folder=args.source.replace('008','009')
    previous=json.loads((fit.HERE/starting_folder/'fit.json').read_text())
    initial=np.r_[previous['finger_spread_degrees'],np.zeros(12)]
    def constraints(values):
        m=metrics(values,True)
        return np.array([5e-6-max(m['finger_hull_penetration_m'].values()),
            m['min_target_proxy_clearance_m']-before['min_target_proxy_clearance_m']+2e-5])
    result=minimize(lambda v:float(np.square(v[4:]/8).sum()+.1*np.square((v[:4]-initial[:4])/15).sum()),
        initial,method='SLSQP',bounds=[(-5,32),(-28,15),(-22,16),(-25,15)]+[(-10,8)]*12,
        constraints=[{'type':'ineq','fun':constraints}],
        options={'maxiter':220,'ftol':1e-7})
    final=base.copy();final[:12]+=result.x[4:]
    p,bones=rig.pose(final,result.x[:4])
    report.pop('min_all_sampled_proxy_clearance_m',None)
    report.update(source_pose=args.source,refinement='small_mcp_spread_with_self_collision_proxy',
        parameters=final.tolist(),finger_spread_degrees=result.x[:4].tolist(),
        flexion_correction_degrees=result.x[4:].tolist(),geometry_before=before,
        geometry_after=metrics(result.x,True),optimizer_success=bool(result.success),physical_validation=False)
    out=fit.HERE/args.output;out.mkdir(exist_ok=False)
    (out/'fit.json').write_text(json.dumps(report,indent=2)+'\n')
    frame=dict(points=p.tolist(),bones={k:v.tolist() for k,v in bones.items()})
    data['frames']=[dict(closure=0.,**frame),dict(closure=1.,**frame)]
    data.update(static_only=True,fit_report='fit.json',texture='../assets/model/new_diff.png')
    (out/'hand.json').write_text(json.dumps(data,separators=(',',':'))+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
