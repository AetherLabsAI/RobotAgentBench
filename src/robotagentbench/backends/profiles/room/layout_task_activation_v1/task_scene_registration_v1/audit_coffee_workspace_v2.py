"""Measured public replay motion projected onto current public asset geometry.

Read only. No simulator, source mutation, or success/continuous-sweep claim.
"""
import json
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from pxr import Usd, UsdGeom, UsdPhysics
from audit_task2_workspace import sha, matrix, corners, joint_frame, box, union

HERE=Path(__file__).resolve().parent
REPO=Path('/home/lingjun/robot_agent_arena/api-protocol')
TASK=REPO/'evaluation/tasks/coffee_machine'
ROOT='/World/Robot'
sys.path.insert(0,str(REPO))
from arena_api import l1_scene

def geometry():
    stage=Usd.Stage.CreateInMemory()
    stage.GetRootLayer().subLayerPaths=[str(TASK/'resources/reference/station.usda'),*[str(p) for p in l1_scene.LAYERS],str(l1_scene.BASE)]
    xf=UsdGeom.XformCache(); cache=UsdGeom.BBoxCache(0,['default','render','proxy'])
    def rigid(path):
        p=stage.GetPrimAtPath(path); assert p,path
        m=np.asarray(xf.GetLocalToWorldTransform(p)).T.copy()
        u,_,v=np.linalg.svd(m[:3,:3]);m[:3,:3]=u@v
        return m
    def local(path):
        inv=np.linalg.inv(rigid(path)); pts=[]
        for p in Usd.PrimRange(stage.GetPrimAtPath(path),Usd.TraverseInstanceProxies()):
            if not p.IsA(UsdGeom.Boundable):continue
            im=UsdGeom.Imageable(p)
            if im.ComputeVisibility()=='invisible' or im.ComputePurpose() not in ('default','render','proxy'):continue
            if p.IsA(UsdGeom.Mesh):
                xyz=np.asarray(UsdGeom.Mesh(p).GetPointsAttr().Get(),float)
                if not xyz.size:continue
                points=np.c_[xyz,np.ones(len(xyz))]
            else:
                b=cache.ComputeUntransformedBound(p).ComputeAlignedRange()
                if b.IsEmpty():continue
                points=corners(b.GetMin(),b.GetMax())
            pts.append((inv@np.asarray(xf.GetLocalToWorldTransform(p)).T@points.T).T)
        assert pts,path
        b=box(np.concatenate(pts));return corners(b['min'],b['max'])
    parents={};initial={}
    for p in Usd.PrimRange(stage.GetPrimAtPath(ROOT)):
        if not any(p.IsA(t) for t in (UsdPhysics.FixedJoint,UsdPhysics.RevoluteJoint,UsdPhysics.PrismaticJoint)):continue
        rel=[p.GetRelationship('physics:body'+str(i)).GetTargets() for i in (0,1)]
        if any(len(x)!=1 for x in rel):continue
        a,b=[str(x[0]) for x in rel]; assert b not in parents
        f0,f1=joint_frame(p,0),joint_frame(p,1)
        moving=not p.IsA(UsdPhysics.FixedJoint);axis='XYZ'.index(p.GetAttribute('physics:axis').Get()) if moving else None
        angular=p.IsA(UsdPhysics.RevoluteJoint)
        relative=np.linalg.inv(rigid(a)@f0)@rigid(b)@f1
        initial[p.GetName()]=(Rotation.from_matrix(relative[:3,:3]).as_rotvec()[axis] if angular else relative[axis,3]) if moving else 0
        parents[b]=(a,p.GetName(),f0,np.linalg.inv(f1),axis,angular)
    points={}; radii={};check=[]
    init_frames={ROOT+'/base_link':rigid(ROOT+'/base_link')}
    def init_fk(b):
        if b in init_frames:return init_frames[b]
        a,name,f0,f1,axis,angular=parents[b];m=np.eye(4)
        if axis is not None:
            if angular:m[:3,:3]=Rotation.from_rotvec(initial[name]*np.eye(3)[axis]).as_matrix()
            else:m[axis,3]=initial[name]
        init_frames[b]=init_fk(a)@f0@m@f1;return init_frames[b]
    for side in ('l','r'):
        base=ROOT+'/gripper_'+side+'_base_link';radius=0
        for p in Usd.PrimRange(stage.GetPrimAtPath(ROOT)):
            if not p.HasAPI(UsdPhysics.RigidBodyAPI):continue
            name=p.GetName();path=str(p.GetPath())
            if name.startswith('arm_'+side+'_') or path==base:
                points[path]=local(path);check.append(float(np.max(abs(init_fk(path)-rigid(path)))))
            elif name.startswith('gripper_'+side+'_') and not name.endswith('_center_link'):
                length=np.linalg.norm(local(path)[:,:3],axis=1).max();cursor=path
                while cursor!=base:
                    a,_,f0,f1,axis,angular=parents[cursor];assert axis is None or angular
                    length+=np.linalg.norm(f0[:3,3])+np.linalg.norm(f1[:3,3]);cursor=a
                radius=max(radius,float(length))
        radii[side]=radius
    assert max(check)<1e-4,check
    layers={l.realPath:sha(l.realPath) for l in stage.GetUsedLayers() if l.realPath}
    return stage,parents,points,radii,rigid(ROOT+'/base_link'),max(check),layers

def asset_points(path):
    s=Usd.Stage.Open(str(path));b=UsdGeom.BBoxCache(0,['default','render','proxy']).ComputeWorldBound(s.GetDefaultPrim()).ComputeAlignedRange()
    assert not b.IsEmpty();return corners(b.GetMin(),b.GetMax())

def main():
    stage,parents,points,radii,base,err,layers=geometry()
    records=[];errors=[]
    current=REPO/'all_assets'
    machine=asset_points(current/'coffee_machine/asset.usda')
    for api in ('1-1','1-2','1-3','1-4'):
        batch='batch-20260929-r2' if api=='1-2' else 'batch-20260930-r4'
        jobs=[TASK/'artifacts'/batch/api/f'source-run-{i}' for i in range(1,6)]
        jobs+=sorted((TASK/'artifacts/batch-20261001-left'/api).glob('source-run-*')) if api in ('1-1','1-4') else []
        for i,run in enumerate(jobs,1):
            try:
                tp=run/'run/episode/trace.json';t=json.loads(tp.read_text());rp=run/'run/report.json';report=json.loads(rp.read_text())
                episode=t['episode_config']['task_parameters']['source_episode']
                bp=TASK/'resources/source_episodes'/api/episode/'public_binding.json';binding=json.loads(bp.read_text())
                anchor=matrix(binding['world']['machine']);inv=np.linalg.inv(anchor)
                obs=[t['initial_observation']]+[s['observation'] for s in t['steps']]
                obs=list({o['step_index']:o for o in obs}.values());obs.sort(key=lambda o:o['step_index'])
                assert all(o['source']=='isaac_physics' for o in obs)
                assert np.allclose(np.diff([o['time_s'] for o in obs]),.05,atol=1e-5)
                qs=[{**o['joint_positions_rad'],**o['joint_positions_m']} for o in obs];n=len(qs)
                frames={ROOT+'/base_link':np.broadcast_to(base,(n,4,4))}
                def fk(b):
                    if b in frames:return frames[b]
                    a,name,f0,f1,axis,angular=parents[b];m=np.broadcast_to(np.eye(4),(n,4,4)).copy()
                    if axis is not None:
                        values=np.array([q[name] for q in qs]);assert np.isfinite(values).all()
                        if angular:m[:,:3,:3]=Rotation.from_rotvec(values[:,None]*np.eye(3)[axis]).as_matrix()
                        else:m[:,axis,3]=values
                    frames[b]=fk(a)@f0@m@f1;return frames[b]
                components={'machine':box(machine)}
                for side in ('l','r'):
                    bs=[box(np.einsum('nij,kj->nki',inv@fk(b),p)) for b,p in points.items() if ('arm_'+side+'_') in b or ('gripper_'+side+'_') in b]
                    centers=(inv@fk(ROOT+'/gripper_'+side+'_base_link'))[:,:3,3];radius=radii[side]
                    bs.append(box(np.concatenate([centers-radius,centers+radius])))
                    components['arm_'+side]=union(bs)
                facts_path=run/'episodes'/t['episode_id']/'coffee_detection/control_facts.jsonl'
                facts=[json.loads(l) for l in facts_path.open()];assert len(facts)==n,(len(facts),n)
                assert [f['step'] for f in facts]==[o['step_index'] for o in obs]
                # Recorded centers + exact asset-origin bounding radius encloses any orientation.
                objects={}
                for role,asset in [('cup','coffee_cup_'+binding['cup_style']),('bag','coffee_bag')]:
                    ap=current/asset/'asset.usda';pts=asset_points(ap);radius=float(np.linalg.norm(pts[:,:3],axis=1).max())
                    centers=np.array([f[role]['position']+[1] for f in facts]);centers=(inv@centers.T).T[:,:3]
                    components[role]=box(np.concatenate([centers-radius,centers+radius]));objects[role]={'asset':str(ap),'sha256':sha(ap),'all_orientation_radius_m':radius}
                result=dict(api=api,episode=episode,run=str(run),verdict=report['verdict'],frames=n,
                    hashes={str(p):sha(p) for p in (tp,rp,bp,facts_path)},components=components,combined=union(list(components.values())),
                    robot_station=(inv@base).tolist(),hand_radius_m=radii,objects=objects,
                    simulated_hz=240,recorded_hz=20)
                records.append(result);print('OK',api,i,report['verdict'],n,flush=True)
            except Exception as e:
                import traceback;traceback.print_exc();errors.append({'api':api,'run':str(run),'error':str(e)})
    good=[r for r in records if r['verdict']=='success']
    result=dict(schema='arena.coffee.current_geometry_measured_motion.v2',expected=60,measured=len(records),errors=errors,
        coordinate_frame='Coffee machine asset: +Y up, +Z front, +X lateral. Gf row transforms transposed for projection.',
        records=records,successful_records=len(good),successful_unique_segments=len({(r['api'],r['episode']) for r in good}),frames=sum(r['frames'] for r in good),
        groups={api:dict(count=sum(r['api']==api for r in good),combined=union([r['combined'] for r in good if r['api']==api])) for api in ('1-1','1-2','1-3','1-4') if any(r['api']==api for r in good)},
        combined=union([r['combined'] for r in good]) if good else None,
        geometry_layers=layers,initial_fk_max_matrix_error=err,geometry_mode='current public geometry with historical measured joints; not recovered historical full scene',
        limitations=['20 selected public API replays plus 40 left-hand replays; successful completed segments form the default envelope; failed attempts retained separately. Repeat episodes are not counted as new independent segments.',
        'All 20 Hz recorded control frames, not 240 Hz substeps or continuous collision sweeps.',
        'Historical frozen scene paths are no longer available. Current public robot geometry is used with measured joints and source station. This is a design envelope, not exact historic mesh reconstruction.',
        'Hand articulation is enclosed by joint-chain spheres. Cup and bag orientations are not recorded; measured centers use all-orientation asset spheres.',
        'Bean and liquid particle positions are not in these logs and are excluded; spill clearance is not certified.',
        'Fixed source base, no recorded navigation; four independently initialized tasks, not a continuous workflow.',
        'No target-layout collision, support, reachability or physical replay validation; no added safety margin.'])
    output=HERE/'coffee_motion_envelope_v2.json';assert not output.exists();output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('SUMMARY',len(records),len(good),result['frames'],result['combined'],errors,flush=True)

if __name__=='__main__':main()
