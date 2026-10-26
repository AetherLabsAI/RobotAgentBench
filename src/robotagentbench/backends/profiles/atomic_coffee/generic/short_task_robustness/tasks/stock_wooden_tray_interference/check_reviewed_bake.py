"""Read-only CPU geometry report for the new native closure bake."""
import json
import numpy as np
from scipy.spatial import ConvexHull
from human_asset import HERE,HumanAsset,transform_points
import fit_static_grip as fit

asset=HumanAsset(HERE/'assets/derived/left_arm_reviewed_grip_010b.json')
p0,rest=asset.sample(0)
center=np.array(asset.data['grip_center_local'])
fit.RADIUS=asset.data['target_radius_m'];fit.HEIGHT=asset.data['target_height_m']
trace=[]
for t in np.linspace(0,1,21):
    points,bones=asset.sample(float(t));bodies={}
    for name,indices in asset.collision_groups().items():
        if name=='forearm.R':continue
        p=transform_points(bones[name]@np.linalg.inv(rest[name]),p0[indices])
        hull=ConvexHull(p)
        samples=np.concatenate([p,p[hull.simplices].mean(1),
            p[hull.simplices[:,:2]].mean(1),p[hull.simplices[:,1:]].mean(1)])
        bodies[name]=float(fit.cylinder_sdf(samples,center).min())
    trace.append(dict(closure=float(t),native_rigid_hull_proxy_clearance_m=bodies))
report=dict(physical_validation=False,kind='native_rigid_hull_vs_cylinder_static_sweep',samples=trace)
(HERE/'assets/derived/reviewed_grip_010b_hull_check.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'open':trace[0],'closed':trace[-1]}),flush=True)
