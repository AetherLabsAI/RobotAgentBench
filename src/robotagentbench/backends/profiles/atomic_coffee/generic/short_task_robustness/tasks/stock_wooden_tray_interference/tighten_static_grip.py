"""Small closure refinement of the reviewed pose; no native physics changes."""
import argparse
import json
import numpy as np
from scipy.optimize import minimize
import fit_static_grip as fit


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',default='static_grip_007')
    parser.add_argument('--output',default='static_grip_008')
    args=parser.parse_args()
    report=json.loads((fit.HERE/args.source/'fit.json').read_text())
    data=json.loads((fit.HERE/'assets/derived/left_arm_forearm_rig.json').read_text())
    rig=fit.Rig(data)
    fit.RADIUS=report['radius_m'];fit.HEIGHT=report['height_m']
    baseline=np.array(report['parameters'])
    triangles=np.array([[f[0],f[j],f[j+1]] for f in data['faces'] for j in range(1,len(f)-1)])
    edges=np.array([(f[j],f[(j+1)%len(f)]) for f in data['faces'] for j in range(len(f))])
    def clearance(angles):
        p,_=rig.pose(np.r_[angles,baseline[-3:]])
        samples=np.concatenate([p,p[triangles].mean(1),p[edges].mean(1)])
        return float(fit.cylinder_sdf(samples,baseline[-3:]).min())
    original_min=clearance(baseline[:17])
    desired=np.tile([0.,4.,8.],4).tolist()+[0.,0.,0.,5.,8.]
    desired=np.array(desired)
    # Preserve the accepted palm/target transform and thumb base orientation.
    bounds=[]
    for j in range(17):
        if j in (12,13,14):bounds.append((baseline[j],baseline[j]))
        elif j<12 and j%3==0:bounds.append((baseline[j]-3,baseline[j]+2))
        else:bounds.append((baseline[j]-2,baseline[j]+desired[j]))
    target=baseline[:17]+desired
    result=minimize(lambda a:float(np.square((a-target)/8).sum()),baseline[:17],
        method='SLSQP',bounds=bounds,
        constraints=[{'type':'ineq','fun':lambda a:clearance(a)-original_min+2e-5}],
        options={'maxiter':180,'ftol':1e-7})
    angles=result.x
    # Never publish a candidate with substantially worse proxy penetration.
    if clearance(angles)<original_min-2.1e-5:
        raise RuntimeError('Closure violates the static clearance guard')
    x=np.r_[angles,baseline[-3:]];p,bones=rig.pose(x)
    report.update(parameters=x.tolist(),source_pose=args.source,
        refinement='small_extra_closure_fixed_palm_and_target',
        joint_delta_degrees=(angles-baseline[:17]).tolist(),
        previous_min_proxy_clearance_m=original_min,
        min_all_sampled_proxy_clearance_m=clearance(angles),
        optimizer_success=bool(result.success),physical_validation=False)
    # Previous contact coordinates and fit loss no longer describe this pose.
    for key in ('contacts','loss','min_vertex_clearance_m','min_sampled_surface_clearance_m'):
        report.pop(key,None)
    out=fit.HERE/args.output;out.mkdir(exist_ok=False)
    (out/'fit.json').write_text(json.dumps(report,indent=2)+'\n')
    frame=dict(points=p.tolist(),bones={k:v.tolist() for k,v in bones.items()})
    data['frames']=[dict(closure=0.,**frame),dict(closure=1.,**frame)]
    data.update(static_only=True,fit_report='fit.json',texture='../assets/model/new_diff.png')
    (out/'hand.json').write_text(json.dumps(data,separators=(',',':'))+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
