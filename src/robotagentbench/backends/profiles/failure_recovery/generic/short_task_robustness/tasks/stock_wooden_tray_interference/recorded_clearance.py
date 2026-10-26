"""Read-only gripper-path reference for a continuous-action diagnostic.

Recorded poses only propose a reserved route before stock is packed. They are
never written to the robot or used as native contact/success evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

HERE=Path(__file__).resolve().parent


def box_distances(points,box):
    points=np.asarray(points,dtype=float);box=np.asarray(box,dtype=float)
    return np.linalg.norm(np.maximum(np.maximum(box[0]-points,points-box[1]),0.),axis=-1)


def away_destinations(box,candidates,points):
    box=np.asarray(box,dtype=float);points=np.asarray(points,dtype=float).reshape(-1,3)
    before=box_distances(points,box)
    # Use the approved proximity envelope, not an invented approach threshold.
    nearby=points[before<=.15]
    if not len(nearby):return []
    before=box_distances(nearby,box)
    accepted=[]
    for xy in candidates:
        delta=np.r_[np.asarray(xy)-box.mean(0)[:2],0.]
        after=box_distances(nearby,box+delta)
        if np.all(after>before):accepted.append(xy)
    return accepted


def entry_ranking_gap(asset,orientation,entry,grip,box,approach_points):
    """Rank existing rear candidates against the moving, not stopped, jaws.

    A center-to-surface distance is only a ranking heuristic. It does not
    certify collision clearance or replace native robot/human contact checks.
    """
    from .motion import smooth,interpolate
    points=np.asarray(approach_points,dtype=float)
    if points.ndim!=3 or points.shape[1:]!=(2,3):
        raise ValueError('Two time-ordered gripper centers required')
    distances=box_distances(points,box).min(axis=1)
    near=np.flatnonzero(distances<=.15)
    if not len(near):raise ValueError('Recorded approach does not reach the gate')
    first=int(near[0]);gaps=[]
    for time in np.linspace(0,.9,55):
        index=first+int(round(time*20))
        if index>=len(points):raise ValueError('Recorded geometry ends during entry')
        closure=0. if time<.6 else smooth((time-.6)/.3)
        wrist=interpolate(entry,grip,time/.6) if time<.6 else grip
        vertices=asset.sample(closure)[0]@orientation.T+wrist
        gaps.append(float(np.linalg.norm(vertices[:,None,:]-points[index][None,:,:],axis=-1).min()))
    return min(gaps)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--launch',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    if not all(p.resolve().is_relative_to(HERE) for p in (a.source,a.launch,a.output)):
        raise ValueError('Only task-owned inputs and output')
    if a.output.exists():raise ValueError('Fresh output required')
    launch=json.loads(a.launch.read_text());m=json.loads(Path(launch['manifest']).read_text())
    r=m['environment_reservations'][0];frame=np.asarray(r['shelf_local_to_world_row_matrix'])
    plan=json.loads((a.source/'PLAN.json').read_text());end=plan['phases'][1]['last_frame']
    episode=json.loads((a.source/'episode.json').read_text())
    recording=a.source/'recording_20hz.npz'
    with np.load(recording,allow_pickle=False) as data:
        paths=list(map(str,data['robot_link_paths']))
        centers=[]
        for hand in ('l','r'):
            ids=[i for i,p in enumerate(paths) if p.endswith('/gripper_'+hand+'_inner_link4') or p.endswith('/gripper_'+hand+'_outer_link4')]
            if len(ids)!=2:raise ValueError('Exactly two recorded jaw pads required: '+str(paths))
            centers.append(data['robot_link_xyz_m'][:end+1,ids].mean(axis=1))
    points=np.stack(centers,axis=1)
    wxyz=episode['robot']['base_wxyz'];oldrot=Rotation.from_quat([*wxyz[1:],wxyz[0]]).as_matrix()
    local=(points-np.asarray(episode['robot']['base_xyz_m']))@oldrot
    x,y,yaw=launch['initial_base_xyyaw'];newrot=Rotation.from_euler('z',yaw).as_matrix()
    world=local@newrot.T+[x,y,0.]
    inv=np.linalg.inv(frame);shelf=world@inv[:3,:3]+inv[3,:3]
    result=dict(schema='recorded_gripper_route_reference.v1',seed=m['seed'],
        source=str(a.source.resolve()),source_sha256=hashlib.sha256(recording.read_bytes()).hexdigest(),
        launch=str(a.launch.resolve()),base_xyyaw=launch['initial_base_xyyaw'],shelf_frame=frame.tolist(),
        approach_points=shelf.tolist(),native_evidence=False,robot_pose_writes=False)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(samples=len(points),output=str(a.output))))


if __name__=='__main__':main()
