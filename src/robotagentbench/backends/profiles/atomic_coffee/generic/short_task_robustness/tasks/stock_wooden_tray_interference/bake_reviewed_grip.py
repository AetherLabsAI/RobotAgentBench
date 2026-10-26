"""CPU open-to-reviewed-grip bake for the isolated shelf diagnostic."""
import json
import argparse
import numpy as np
from scipy.optimize import minimize
from fit_static_grip import HERE,Rig


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--short-wrist',action='store_true',help='Bake the reviewed 011 hand and 25 mm wrist')
    args=parser.parse_args()
    review='static_short_hand_011' if args.short_wrist else 'static_grip_010'
    report=json.loads((HERE/review/'fit.json').read_text())
    data=json.loads((HERE/review/'cropped_rig.json' if args.short_wrist else
                     HERE/'assets/derived/left_arm_forearm_rig.json').read_text())
    rig=Rig(data);closed=np.array(report['parameters'])
    opened=closed.copy();opened[:12]=np.tile([10.,10.,5.],4)
    spread=np.array(report['finger_spread_degrees'])
    # During front-normal entry, keep the whole thumb outside the target's
    # lateral silhouette. Final thumb pose remains exactly the reviewed one.
    thumb_ids=np.unique(sum((v for k,v in rig.groups.items() if k.startswith('thumb.')),[]))
    limit=closed[-2]-report['radius_m']-.005
    def thumb_y(thumb):
        pose=opened.copy();pose[12:17]=thumb
        points,_=rig.pose(pose,spread)
        return float(points[thumb_ids,1].max())
    result=minimize(lambda a:float(np.square((a-closed[12:17])/45).sum()),
        closed[12:17],method='SLSQP',bounds=[(-85,85),(-85,85),(-85,85),(0,90),(0,90)],
        constraints=[{'type':'ineq','fun':lambda a:limit-thumb_y(a)}],
        options={'maxiter':180,'ftol':1e-8})
    if not result.success or thumb_y(result.x)>limit+1e-6:
        raise RuntimeError('No thumb-clear entry pose found')
    opened[12:17]=result.x
    frames=[]
    for closure in np.linspace(0,1,21):
        points,bones=rig.pose(opened+(closed-opened)*closure,spread)
        frames.append(dict(closure=float(closure),points=points.tolist(),
            bones={k:v.tolist() for k,v in bones.items()}))
    data.update(frames=frames,texture='../model/new_diff.png',static_only=False,
        reviewed_pose=review,physical_validation=False,
        grip_center_local=report['center_local'],target_radius_m=report['radius_m'],
        target_height_m=report['height_m'])
    data['entry_thumb_pose_degrees']=result.x.tolist()
    destination=HERE/'assets/derived'/('left_hand_short_wrist_011.json' if args.short_wrist else
                                     'left_arm_reviewed_grip_010c.json')
    if destination.exists():raise FileExistsError(destination)
    destination.write_text(json.dumps(data,separators=(',',':'))+'\n')
    print(destination,flush=True)


if __name__=='__main__':main()
