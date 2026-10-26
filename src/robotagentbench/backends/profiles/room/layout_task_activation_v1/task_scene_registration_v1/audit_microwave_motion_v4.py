"""Read-only recorded-joint/rigid-pose envelope audit, not a simulation or certificate."""
import argparse
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.spatial.transform import Rotation
from pxr import Usd, UsdGeom, UsdPhysics
from audit_microwave_placement_v1 import HERE, LAYOUT, MICRO, DOOR, corners, bounds, sha
from audit_task2_workspace import joint_frame

ROOT = '/World/Robot'
TAKE_INDEX = Path('/mnt/agent-bench-data/lingjun/arena-data-review-20260920/datasets/microwave_complete50_20261001/manifest.json')
PUT_REPORT = HERE/'microwave_selected50_placement_v2.json'
LEFTOVER_INDEX = TAKE_INDEX.parents[1]/'task9_selected50_20261001/task_9.4_selected50.json'

def union(boxes):
    return bounds([p for b in boxes for p in (b['min'], b['max'])])

def pose_matrices(p, wxyz=False):
    p=np.asarray(p,float); assert p.ndim==2 and p.shape[1]==7 and np.isfinite(p).all()
    q=p[:,3:]; assert np.max(abs(np.linalg.norm(q,axis=1)-1))<1e-3
    m=np.broadcast_to(np.eye(4),(len(p),4,4)).copy()
    m[:,:3,:3]=Rotation.from_quat(q[:,[1,2,3,0]] if wxyz else q).as_matrix()
    m[:,:3,3]=p[:,:3]
    return m

def audit(job):
    kind,item=job
    is_put=kind.startswith('put')
    source=Path(item['source']); trace=source/('control_state_60hz.npz' if is_put else 'initial_probe_trace.npz')
    if is_put:
        scene=source/'initial_scene.usda'
        if kind=='put':assert sha(scene)==item['scene_sha256'] and sha(trace)==item['control_sha256']
    else:
        args=json.loads((source/'launch_arguments.json').read_text())
        assert args['appearance_scene'].startswith('/trial/')
        scene=source.parent/args['appearance_scene'].removeprefix('/trial/')
    source_hashes={str(p):sha(p) for p in (scene,trace)}
    stage=Usd.Stage.Open(str(scene)); assert stage and UsdGeom.GetStageMetersPerUnit(stage)==1
    xf=UsdGeom.XformCache(); cache=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render','proxy'],useExtentsHint=False)
    def world(path):
        pr=stage.GetPrimAtPath(path); assert pr,path
        return np.asarray(xf.GetLocalToWorldTransform(pr)).T
    def rigid(path):
        m=world(path).copy(); u,_,v=np.linalg.svd(m[:3,:3]); m[:3,:3]=u@v
        assert np.linalg.det(m[:3,:3])>.999
        return m
    # Geometry is expressed in metres about a rigid body frame: retain mesh scales,
    # which recorded rigid poses do not carry (e.g. plate Asset has scale 0.001).
    def local_points(path):
        points=[]; inv=np.linalg.inv(rigid(path))
        for pr in Usd.PrimRange(stage.GetPrimAtPath(path),Usd.TraverseInstanceProxies()):
            if not pr.IsA(UsdGeom.Boundable): continue
            im=UsdGeom.Imageable(pr)
            if im.ComputeVisibility()=='invisible' or im.ComputePurpose() not in ('default','render','proxy'):continue
            if pr.IsA(UsdGeom.Mesh):
                xyz=np.asarray(UsdGeom.Mesh(pr).GetPointsAttr().Get(),float)
                if not xyz.size:continue
                pts=np.c_[xyz,np.ones(len(xyz))]
            else:
                b=cache.ComputeUntransformedBound(pr).ComputeAlignedRange()
                if b.IsEmpty():continue
                pts=corners(b.GetMin(),b.GetMax())
            points.append((inv@world(str(pr.GetPath()))@pts.T).T[:,:3])
        assert points,('missing geometry',path)
        b=bounds(np.concatenate(points));return corners(b['min'],b['max'])
    with np.load(trace,allow_pickle=False) as data:
        names=list(data['dof_names']); qs=data['q' if is_put else 'joint_position'].astype(float)
        poses={str(path):data['body_pose'][:,i,:] for i,path in enumerate(data['rigid_body_paths'])} if is_put else {k:data[k] for k in ('plate','beef','door','gripper_l','gripper_r')}
        phases=sorted(set(data['phase'].tolist()))
        if is_put:assert np.allclose(np.diff(data['control_time_s']),1/60,atol=1e-5)
    n=len(qs); assert qs.shape==(n,len(names)) and np.isfinite(qs).all()
    anchor=rigid(MICRO);inv=np.linalg.inv(anchor)
    parents={}; initial={}
    for pr in Usd.PrimRange(stage.GetPrimAtPath(ROOT)):
        if not any(pr.IsA(t) for t in (UsdPhysics.FixedJoint,UsdPhysics.RevoluteJoint,UsdPhysics.PrismaticJoint)):continue
        rel=[pr.GetRelationship('physics:body'+str(i)).GetTargets() for i in (0,1)]
        if len(rel[0])!=1 or len(rel[1])!=1:continue
        a,b=str(rel[0][0]),str(rel[1][0]);assert b not in parents
        f0,f1=joint_frame(pr,0),joint_frame(pr,1)
        moving=not pr.IsA(UsdPhysics.FixedJoint);axis='XYZ'.index(pr.GetAttribute('physics:axis').Get()) if moving else None
        angular=pr.IsA(UsdPhysics.RevoluteJoint)
        relative=np.linalg.inv(rigid(a)@f0)@rigid(b)@f1
        q0=(Rotation.from_matrix(relative[:3,:3]).as_rotvec()[axis] if angular else relative[axis,3]) if moving else 0.
        parents[b]=(a,pr.GetName(),f0,np.linalg.inv(f1),axis,angular)
        initial[pr.GetName()]=q0
    frames={ROOT+'/base_link':np.broadcast_to(rigid(ROOT+'/base_link'),(n,4,4))}
    init_frames={ROOT+'/base_link':rigid(ROOT+'/base_link')}
    def fk(body,check=False):
        store=init_frames if check else frames
        if body in store:return store[body]
        a,name,f0,f1inv,axis,angular=parents[body]
        motion=np.eye(4) if check else np.broadcast_to(np.eye(4),(n,4,4)).copy()
        if axis is not None:
            assert name in names,('unrecorded joint',name)
            v=initial[name] if check else qs[:,names.index(name)]
            if angular:motion[...,:3,:3]=Rotation.from_rotvec(np.asarray(v)[...,None]*np.eye(3)[axis]).as_matrix()
            else:motion[...,axis,3]=v
        store[body]=fk(a,check)@f0@motion@f1inv
        return store[body]
    boxes={};checks=[]; counts={}
    for side in ('l','r'):
        bodies=[str(pr.GetPath()) for pr in Usd.PrimRange(stage.GetPrimAtPath(ROOT)) if pr.HasAPI(UsdPhysics.RigidBodyAPI) and pr.GetName().startswith(('arm_'+side+'_','gripper_'+side+'_'))]
        assert len(bodies)>=8
        all_boxes=[]
        for body in bodies:
            # Center marker links can have no visible geometry; recorded chain still used.
            if body.endswith('_center_link'):continue
            pts=local_points(body)
            checks.append(float(np.max(abs(fk(body,True)-rigid(body)))))
            all_boxes.append(bounds(np.einsum('nij,kj->nki',inv@fk(body),pts)[...,:3]))
        boxes['arm_'+side]=union(all_boxes);counts[side]=len(all_boxes)
    initial_fk_max=max(checks)
    hand_fallback={}
    if initial_fk_max>=1e-4:
        # A closed-loop finger assembly need not equal a tree FK at the authored
        # snapshot. Do not discard the episode or pass its exact finger FK.
        for side in ('l','r'):
            base=ROOT+'/gripper_'+side+'_base_link'
            arms=[str(pr.GetPath()) for pr in Usd.PrimRange(stage.GetPrimAtPath(ROOT)) if pr.HasAPI(UsdPhysics.RigidBodyAPI) and (pr.GetName().startswith('arm_'+side+'_') or str(pr.GetPath())==base)]
            assert max(float(np.max(abs(fk(b,True)-rigid(b)))) for b in arms)<1e-4
            arm_boxes=[bounds(np.einsum('nij,kj->nki',inv@fk(b),local_points(b))[...,:3]) for b in arms]
            radius=0.;authored_radius=0.
            for pr in Usd.PrimRange(stage.GetPrimAtPath(ROOT)):
                b=str(pr.GetPath())
                if not pr.HasAPI(UsdPhysics.RigidBodyAPI) or not pr.GetName().startswith('gripper_'+side+'_') or b.endswith('_center_link'):continue
                pts=local_points(b)
                length=float(np.linalg.norm(pts[:,:3],axis=1).max());cursor=b
                while cursor!=base:
                    parent,name,f0,f1inv,axis,angular=parents[cursor]
                    assert axis is None or angular
                    length+=np.linalg.norm(f0[:3,3])+np.linalg.norm(f1inv[:3,3]);cursor=parent
                radius=max(radius,length)
                initial_pts=(np.linalg.inv(rigid(base))@rigid(b)@pts.T).T[:,:3]
                authored_radius=max(authored_radius,float(np.linalg.norm(initial_pts,axis=1).max()))
            radius=max(radius,authored_radius)
            centers=(inv@fk(base))[:,:3,3]
            hand_box=bounds(np.concatenate([centers-radius,centers+radius]))
            boxes['arm_'+side]=union(arm_boxes+[hand_box])
            hand_fallback[side]=dict(radius_m=radius,authored_radius_m=authored_radius,
                method='All-configuration joint-chain sphere plus authored-geometry enclosure; not exact measured finger poses',bounds=hand_box)
    residuals={}
    if kind=='take':
        for side in ('l','r'):
            actual=pose_matrices(poses['gripper_'+side],True);pred=fk(ROOT+'/gripper_'+side+'_base_link')
            err=np.linalg.norm(actual[:,:3,3]-pred[:,:3,3],axis=1)
            rot=Rotation.from_matrix(np.swapaxes(actual[:,:3,:3],1,2)@pred[:,:3,:3]).magnitude()
            residuals[side]={'max_position_m':float(err.max()),'median_position_m':float(np.median(err)),'max_angle_deg':float(np.rad2deg(rot.max()))}
        # Reject wrong scene/root/kinematics rather than publishing an incorrect envelope.
        assert max(r['max_position_m'] for r in residuals.values())<.002,residuals
    skipped_hidden=[]
    if is_put:
        roles={'door':DOOR,'plate':'/World/Assets/MicrowaveTakePlate/Asset'}
        active='/World/Assets/BeefTray/Asset' if kind=='put' else '/World/Assets/MwFoodbox/Asset'
        inactive='/World/Assets/MwFoodbox/Asset' if kind=='put' else '/World/Assets/BeefTray/Asset'
        # The recording retains an empty/invisible alternative payload root.
        # Exclude it only after proving it has no visible Boundable geometry.
        assert not any(pr.IsA(UsdGeom.Boundable) and UsdGeom.Imageable(pr).ComputeVisibility()!='invisible' and UsdGeom.Imageable(pr).ComputePurpose() in ('default','render','proxy') for pr in Usd.PrimRange(stage.GetPrimAtPath(inactive),Usd.TraverseInstanceProxies())),inactive
        skipped_hidden.append(inactive)
        roles['beeftray' if kind=='put' else 'foodbox']=active
        for role,path in roles.items():
            assert path in poses
            if UsdGeom.Imageable(stage.GetPrimAtPath(path)).ComputeVisibility()=='invisible':
                skipped_hidden.append(path);continue
            boxes[role]=bounds(np.einsum('nij,kj->nki',inv@pose_matrices(poses[path]),local_points(path))[...,:3])
        assert 'door' in boxes and ('foodbox' in boxes or 'beeftray' in boxes)
    else:
        geom=source/'load_clearance_geometry.npz';source_hashes[str(geom)]=sha(geom)
        with np.load(geom,allow_pickle=False) as g:
            np.testing.assert_allclose(anchor[:3,3],g['microwave_xyz'],atol=1e-8)
            np.testing.assert_allclose(anchor[:3,:3],g['microwave_rotation'],atol=1e-8)
            for role,key in [('plate','plate'),('foodbox','beef')]:
                pts=g[key+'_points'];assert np.isfinite(pts).all()
                b=bounds(pts)
                boxes[role]=bounds(np.einsum('nij,kj->nki',inv@pose_matrices(poses[key],True),corners(b['min'],b['max']))[...,:3])
        boxes['door']=bounds(np.einsum('nij,kj->nki',inv@pose_matrices(poses['door'],True),local_points(DOOR))[...,:3])
    boxes['microwave_initial']=bounds(local_points(MICRO)[...,:3])
    station=inv@rigid(ROOT+'/base_link')
    for p,h in source_hashes.items():assert sha(p)==h,p
    return dict(kind=kind,source=str(source),scene=str(scene),hashes=source_hashes,frames=n,sample_hz=60 if is_put else 20,phases=phases,skipped_initially_hidden_objects=skipped_hidden,
        bodies_per_arm=counts,initial_fk_max_matrix_error=initial_fk_max,conservative_hand_fallback=hand_fallback,measured_gripper_fk_residuals=residuals,
        robot_station_in_microwave=station.tolist(),components=boxes,combined=union(list(boxes.values())))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pilot',action='store_true');args=ap.parse_args()
    puts=json.loads(PUT_REPORT.read_text())['records'];takes=json.loads(TAKE_INDEX.read_text());leftovers=[]
    for item in json.loads(LEFTOVER_INDEX.read_text())['selected']:
        copies={'lerobot_foodbox':'microwave_food','lerobot_foodbox_grasp20':'microwave_food_grasp'}
        source=LEFTOVER_INDEX.parents[1]/copies[Path(item['dataset']).name]/'sim_info'/Path(item['sim_info_dir']).name
        a=json.loads((source/'randomization_audit.json').read_text());f=item['features']
        for j,v in enumerate(a['base_requested']['xy_delta_m']):assert abs(v-f[f'base_requested.xy_delta_m.{j}'])<1e-12
        assert abs(a['base_requested']['yaw_delta_deg']-f['base_requested.yaw_delta_deg'])<1e-12
        for j,v in enumerate(a['arm_initial_delta_rad']):assert abs(v-f[f'arm_initial_delta_rad.{j}'])<1e-12
        leftovers.append(dict(source=str(source)))
    jobs=[(kind,r) for kind,rows in [('put',puts),('put_leftover',leftovers),('take',takes)] for r in (rows[:1] if args.pilot else rows)]
    prior=json.loads((HERE/'microwave_motion_envelope_v3.json').read_text())
    assert not args.pilot and prior['expected']==150
    failed={(r['kind'],r['source']) for r in prior['errors']}
    jobs=[j for j in jobs if (j[0],j[1]['source']) in failed]
    results=prior['records'].copy();errors=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(audit,j) for j in jobs]
        for j,f in zip(jobs,futures):
            try:
                r=f.result();results.append(r);print('OK',len(results),r['kind'],r['source'],r['measured_gripper_fk_residuals'],flush=True)
            except Exception as e:
                import traceback
                traceback.print_exc();errors.append(dict(kind=j[0],source=j[1]['source'],error=str(e)))
    grouped={kind:dict(count=sum(r['kind']==kind for r in results),frames=sum(r['frames'] for r in results if r['kind']==kind),
        combined=union([r['combined'] for r in results if r['kind']==kind])) for kind in ('put','put_leftover','take') if any(r['kind']==kind for r in results)}
    report=dict(schema='arena.microwave.recorded_envelope.v4',pilot=args.pilot,expected=prior['expected'],measured=len(results),errors=errors,
        input_hashes={str(p):sha(p) for p in (PUT_REPORT,LEFTOVER_INDEX,TAKE_INDEX,HERE/'microwave_motion_envelope_v3.json')},groups=grouped,records=results,
        combined=union([r['combined'] for r in results]) if results else None,
        limitations=['Measured joints reconstructed with source USD kinematics; not new simulation.',
        '60Hz put / 20Hz take recorded frames, no continuous-time or 240Hz substep coverage.',
        'Visual bounding boxes; take plate/box additionally use saved visual/collision geometry bounds.',
        'Fixed source robot base retained; no navigation trace or target-layout reachability validation.',
        'Union of different source episodes, NOT a single continuous put/take trajectory.',
        'Selected 50 put 9.2 beef + 50 put 9.4 leftover + 50 delivered take recordings, not all possible randomizations.',
        'Four closed-loop finger snapshots fail exact tree reconstruction; those episodes use conservative all-configuration hand spheres, with authored snapshot geometry also enclosed. Not exact physical finger poses.',
        'No additional safety margin or approved placement policy; prior failed audit is preserved.'])
    out=HERE/('microwave_motion_pilot_v3.json' if args.pilot else 'microwave_motion_envelope_v4.json')
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print('SUMMARY',json.dumps({k:v for k,v in report.items() if k not in ('records','limitations')}),flush=True)

if __name__=='__main__':main()

