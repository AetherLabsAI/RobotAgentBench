"""Read-only task 10.5 measured arm audit. Run on L1; output to explicit file.

No scene edits, physics runs, target changes or object envelope contributions.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from pxr import Usd, UsdGeom, UsdPhysics
from audit_task2_workspace import sha, corners, joint_frame, box, union


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    src=args.source;ep=src/'episode';accept=json.loads((ep/'acceptance.json').read_text());assert accept['pass'] is True
    state=json.loads((ep/'states.json').read_text());action=np.load(ep/'action.npz');names=action['actual_joint_names'].tolist()
    q=np.asarray([s['q'] for s in state]);assert q.shape==(len(state),len(names));assert np.isfinite(q).all()
    # The native result records the export start after settle (including a
    # repeated boundary record). Verify offset, ticks AND actual joint values.
    native_result=json.loads((ep/'result.json').read_text())
    idx=action['frame_state_indices']+native_result['data_start_record_index']
    assert np.array_equal(np.array([s['tick'] for s in state])[idx],action['frame_ticks'])
    assert np.allclose(q[idx],action['actual_joint_positions'],atol=1e-7)
    times=np.array([s['time'] for s in state]);dt=np.diff(times);assert np.all(np.isclose(dt,.05,atol=1e-6)|np.isclose(dt,0,atol=1e-9))
    for i in np.where(np.isclose(dt,0,atol=1e-9))[0]:assert np.allclose(q[i],q[i+1],atol=1e-7)
    stage=Usd.Stage.Open(str(src/'scene.usdc'));xf=UsdGeom.XformCache();bc=UsdGeom.BBoxCache(0,['default','render','proxy']);root='/World/Robot'
    def rigid(path):
        p=stage.GetPrimAtPath(path);assert p,path
        m=np.asarray(xf.GetLocalToWorldTransform(p)).T.copy();u,_,v=np.linalg.svd(m[:3,:3]);m[:3,:3]=u@v;return m
    def local_points(path):
        inv=np.linalg.inv(rigid(path));all_points=[]
        for p in Usd.PrimRange(stage.GetPrimAtPath(path),Usd.TraverseInstanceProxies()):
            if not p.IsA(UsdGeom.Boundable):continue
            im=UsdGeom.Imageable(p)
            if im.ComputeVisibility()=='invisible' or im.ComputePurpose() not in ('default','render','proxy'):continue
            if p.IsA(UsdGeom.Mesh):
                pts=np.asarray(UsdGeom.Mesh(p).GetPointsAttr().Get(),float)
                if not pts.size:continue
                pts=np.c_[pts,np.ones(len(pts))]
            else:
                b=bc.ComputeUntransformedBound(p).ComputeAlignedRange()
                if b.IsEmpty():continue
                pts=corners(b.GetMin(),b.GetMax())
            all_points.append((inv@np.asarray(xf.GetLocalToWorldTransform(p)).T@pts.T).T)
        if not all_points:return None
        b=box(np.concatenate(all_points));return corners(b['min'],b['max'])
    parents={};initial={}
    for p in Usd.PrimRange(stage.GetPrimAtPath(root)):
        if not any(p.IsA(t) for t in (UsdPhysics.FixedJoint,UsdPhysics.RevoluteJoint,UsdPhysics.PrismaticJoint)):continue
        rel=[p.GetRelationship('physics:body'+str(i)).GetTargets() for i in (0,1)]
        if any(len(x)!=1 for x in rel):continue
        a,b=[str(x[0]) for x in rel];assert b not in parents
        f0,f1=joint_frame(p,0),joint_frame(p,1);moving=not p.IsA(UsdPhysics.FixedJoint)
        axis='XYZ'.index(p.GetAttribute('physics:axis').Get()) if moving else None;angular=p.IsA(UsdPhysics.RevoluteJoint)
        relative=np.linalg.inv(rigid(a)@f0)@rigid(b)@f1
        initial[p.GetName()]=(Rotation.from_matrix(relative[:3,:3]).as_rotvec()[axis] if angular else relative[axis,3]) if moving else 0
        parents[b]=(a,p.GetName(),f0,np.linalg.inv(f1),axis,angular)
    base=root+'/base_link';base_m=[]
    for s in state:
        o=s['objects'][base];m=np.eye(4);m[:3,:3]=o['rotation'];m[:3,3]=o['xyz'];base_m.append(m)
    frames={base:np.asarray(base_m)};authored={base:rigid(base)}
    def fk(b,check=False):
        cache=authored if check else frames
        if b in cache:return cache[b]
        a,name,f0,f1,axis,angular=parents[b];m=np.eye(4) if check else np.broadcast_to(np.eye(4),(len(state),4,4)).copy()
        if axis is not None:
            value=initial[name] if check else q[:,names.index(name)]
            if angular:m[..., :3,:3]=Rotation.from_rotvec(np.asarray(value)[...,None]*np.eye(3)[axis]).as_matrix()
            else:m[...,axis,3]=value
        cache[b]=fk(a,check)@f0@m@f1;return cache[b]
    tray=rigid('/World/FoodTray');inv=np.linalg.inv(tray);components={};radii={};errors=[];links=[]
    for side in ('l','r'):
        hand=root+'/gripper_'+side+'_base_link';parts=[];radius=0
        for p in Usd.PrimRange(stage.GetPrimAtPath(root)):
            if not p.HasAPI(UsdPhysics.RigidBodyAPI):continue
            path=str(p.GetPath());name=p.GetName()
            if not (name.startswith('arm_'+side+'_') or name.startswith('gripper_'+side+'_')):continue
            pts=local_points(path)
            if pts is None:continue
            if name.startswith('arm_'+side+'_') or path==hand:
                errors.append(float(np.max(abs(fk(path,True)-rigid(path)))));parts.append(box(np.einsum('nij,kj->nki',inv@fk(path),pts)));links.append(path)
            elif not name.endswith('_center_link'):
                length=np.linalg.norm(pts[:,:3],axis=1).max();cursor=path
                while cursor!=hand:
                    a,_,f0,f1,axis,angular=parents[cursor];assert axis is None or angular
                    length+=np.linalg.norm(f0[:3,3])+np.linalg.norm(f1[:3,3]);cursor=a
                radius=max(radius,float(length))
        centers=(inv@fk(hand))[:,:3,3];parts.append(box(np.concatenate([centers-radius,centers+radius])));radii[side]=radius;components['arm_'+side]=union(parts)
    assert max(errors)<1e-4,errors
    cams=np.load(ep/'camera_extrinsics_20hz.npz');cam_errors=[]
    for side,cam_name,leaf in [('l','hand_left','Left_Camera'),('r','hand_right','Right_Camera')]:
        b=root+'/gripper_'+side+'_base_link';offset=np.linalg.inv(rigid(b))@rigid(b+'/'+leaf);pred=fk(b)[idx]@offset
        actual=cams['world_from_usd_camera'][:,cams['camera_names'].tolist().index(cam_name)]
        error=float(np.max(np.abs(pred-actual)));assert error<1e-3,(cam_name,error);cam_errors.append(error)
    raw=union(list(components.values()));lo=np.floor(np.array(raw['min'])/.05)*.05;hi=np.ceil(np.array(raw['max'])/.05)*.05
    rounded=box(np.array([lo,hi]));refs={}
    for name,path in [('tray','/World/FoodTray/Render'),('holder','/World/Cutlery/Holder')]:
        pts=local_points(path);refs[name]=box((inv@rigid(path)@pts.T).T)
    provenance={str(p):sha(p) for p in [src/'scene.usdc',src/'setup.json',ep/'states.json',ep/'action.npz',ep/'result.json',ep/'acceptance.json',ep/'camera_extrinsics_20hz.npz']}
    report=dict(schema='arena.task10_5.standard_arms.v1',task='10.5',task_name='从餐具盒取第一把叉子放入托盘并收臂',source_host='196',source=str(src),dataset_episode=3,source_acceptance_pass=True,selection='First original counter-station episode in curated final50 order; one complete representative, not a union of random episodes.',frames=len(state),video_frames=len(idx),duration_s=float(times[-1]-times[0]),sample_hz=20,physics_hz=240,coordinate_frame='source tray rigid local X/Y horizontal, Z up',source_tray_matrix=tray.tolist(),components=components,raw=raw,standard=rounded,grid_m=.05,reference_objects=refs,source_robot_station=(inv@base_m[0]).tolist(),gripper_radius_m=radii,arm_links=links,authored_fk_max_matrix_error=max(errors),camera_fk_all_video_frames_max_matrix_error=max(cam_errors),hashes=provenance,layer_hashes={l.realPath:sha(l.realPath) for l in stage.GetUsedLayers() if l.realPath},limitations=['Only arm and gripper envelopes contribute. Tray, holder, fork, food and drink are reference objects only.','Arm visible geometry uses per-link local boxes and measured joint FK. Grippers use conservative joint-chain spheres, not exact finger sweeps.','All native 20 Hz states including settling/tail; not 240 Hz continuous sweeps.','Projection to target tray frame is not a physical replay or proof of target support, reachability or holder alignment.','Standard covers selected first-fork action only, not every meal task or all fifty random episodes.'])
    args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:report[k] for k in ['frames','video_frames','raw','standard','camera_fk_all_video_frames_max_matrix_error']},ensure_ascii=False))


if __name__=='__main__':main()
