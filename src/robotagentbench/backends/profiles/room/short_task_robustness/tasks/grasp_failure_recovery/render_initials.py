"""Task-owned, paused-timeline FK previews; never native/scoring evidence.

Uses the existing public head camera and its current native-owner resolution.
No light/camera/drive/collision settings are authored. Unspecified passive joint
coordinates retain the asset reset position, except the public paired follower.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import time
import traceback

HERE=Path(__file__).resolve().parent
CURRENT=Path('/home/lingjun/robot_agent_arena/short_tasks/agent_probe_v1/deployment_generic_20261010/source')


def owned(path):
    p=Path(path).resolve()
    if p==HERE or not p.is_relative_to(HERE):raise ValueError('Task-owned path required')
    return p


VARIANTS=('shelf_drink','coffee_bag','coffee_cup','coffee_button','cutlery','basket')


def inputs(variants=VARIANTS):
    rows=[]
    for name in variants:
        for seed in (101,102,103):
            folder=(HERE/'random'/f'basket_{seed}_rest_envelope_v2' if name=='basket'
                    else HERE/'batches/random8_v1'/f'{name}_{seed}')
            path=owned(folder/'initial.json')
            value=json.loads(path.read_text())
            if value['variant']!=name or value['seed']!=seed:raise ValueError('Wrong preview binding')
            rows.append(dict(variant=name,seed=seed,initial=str(path)))
    return rows


def forward_frames(stage, coordinates, base):
    """Joint-frame FK after the unchanged trusted native reset validation.

    The source proposal FK uses strict bounds, unlike the actual native reset's
    documented sensor-noise tolerance. Preserve accepted readings verbatim;
    do not clip coordinates, change limits or modify that source checker.
    """
    import numpy as np
    from scipy.spatial.transform import Rotation
    from pxr import Usd,UsdPhysics
    joints=[]
    def anchor(p,index):
        m=np.eye(4);q=p.GetAttribute('physics:localRot'+str(index)).Get()
        m[:3,:3]=Rotation.from_quat([*q.GetImaginary(),q.GetReal()]).as_matrix()
        m[:3,3]=p.GetAttribute('physics:localPos'+str(index)).Get()
        return m
    for p in Usd.PrimRange(stage.GetPrimAtPath('/World/Robot')):
        if not any(p.IsA(t) for t in (UsdPhysics.FixedJoint,UsdPhysics.RevoluteJoint,UsdPhysics.PrismaticJoint)):continue
        a,b=[p.GetRelationship('physics:body'+str(i)).GetTargets() for i in (0,1)]
        if len(a)!=1 or len(b)!=1:continue
        motion=np.eye(4)
        if not p.IsA(UsdPhysics.FixedJoint):
            value=coordinates[p.GetName()]
            axis=np.eye(3)['XYZ'.index(p.GetAttribute('physics:axis').Get())]
            if p.IsA(UsdPhysics.PrismaticJoint):motion[:3,3]=axis*value
            else:motion[:3,:3]=Rotation.from_rotvec(axis*value).as_matrix()
        joints.append((str(a[0]),str(b[0]),anchor(p,0)@motion@np.linalg.inv(anchor(p,1))))
    frames={'/World/Robot/base_link':base}
    for _ in joints:
        count=len(frames)
        for a,b,delta in joints:
            if a in frames and b not in frames:frames[b]=frames[a]@delta
        if len(frames)==count:break
    bodies={str(p.GetPath()) for p in Usd.PrimRange(stage.GetPrimAtPath('/World/Robot')) if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    if bodies-set(frames):raise ValueError('Unresolved static FK bodies: '+str(bodies-set(frames)))
    return frames


def static_reset(stage, record):
    import numpy as np
    from pxr import Usd, UsdGeom, UsdPhysics, Gf
    from arena_api.robot_profiles import FOLLOWERS
    from .common import physics,audit_reset
    before=physics(stage)
    coordinates={}
    inherited=[]
    requested=record['robot_initialization']['positions']
    for p in Usd.PrimRange(stage.GetPrimAtPath('/World/Robot')):
        angular=p.IsA(UsdPhysics.RevoluteJoint)
        if not angular and not p.IsA(UsdPhysics.PrismaticJoint):continue
        name=p.GetName()
        if name in requested:
            coordinates[name]=float(requested[name]);continue
        kind='angular' if angular else 'linear'
        value=p.GetAttribute('state:'+kind+':physics:position').Get()
        value=0. if value is None else float(value)
        coordinates[name]=math.radians(value) if angular else value
        inherited.append(name)
    for follower,driver in FOLLOWERS.items():
        if follower not in requested:coordinates[follower]=-coordinates[driver]
    reset_check=audit_reset(stage,dict(positions=coordinates,
        hold_targets=record['robot_initialization']['hold_targets']))
    cache=UsdGeom.XformCache()
    base=np.array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath('/World/Robot/base_link'))).T
    frames=forward_frames(stage,coordinates,base)
    for path in sorted(frames,key=lambda p:p.count('/')):
        prim=stage.GetPrimAtPath(path)
        scale=Gf.Transform(cache.GetLocalToWorldTransform(prim)).GetScale()
        target=Gf.Matrix4d().SetScale(scale)*Gf.Matrix4d(frames[path].T.tolist())
        parent=UsdGeom.XformCache().GetLocalToWorldTransform(prim.GetParent())
        xf=UsdGeom.Xformable(prim);xf.ClearXformOpOrder()
        xf.AddTransformOp(opSuffix='staticFailurePreview').Set(target*parent.GetInverse())
    if physics(stage)!=before:raise ValueError('Unexpected physical attribute change')
    return dict(method='pre_settle_fk_of_declared_reset_not_native_state',link_count=len(frames),
        passive_coordinates_from_asset_reset=inherited,physics_advanced=False,
        native_reset_validation=reset_check,coordinates_clipped=False,
        physical_settle_verified=False,drive_or_collision_settings_changed=False)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    p.add_argument('--allow-shared-gpu0',action='store_true',required=True)
    p.add_argument('--validate-only',action='store_true')
    p.add_argument('--variants',choices=VARIANTS,nargs='+',default=list(VARIANTS))
    args=p.parse_args()
    rows=inputs(args.variants)
    if args.validate_only:
        print(json.dumps(dict(samples=len(rows),variants=len(args.variants),seeds=[101,102,103])));return
    out=owned(args.output)
    if out.exists():raise ValueError('Fresh preview output required')
    from .common import digest, verify_sources
    for row in rows:
        record=json.loads(Path(row['initial']).read_text())
        if digest(record['scene'])!=record['scene_sha256']:raise ValueError('Prepared scene changed')
        verify_sources(record['source_layers']);verify_sources(record['code_sources'])
    query=lambda *a:subprocess.check_output(['nvidia-smi',*a],text=True)
    occupancy=query('-i','0','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits')
    compute=query('-i','0','--query-compute-apps=pid,process_name','--format=csv,noheader')
    graphics=query('pmon','-i','0','-c','1')
    values=[v.strip() for v in occupancy.strip().split(',')]
    if values[0]!='0' or int(values[2])>66 or int(values[3])!=0 or compute.strip():
        raise RuntimeError('GPU0 no longer meets this preview shared-GPU approval; no process stopped')
    active=[line.split() for line in graphics.splitlines() if line.strip() and not line.lstrip().startswith('#')]
    if any(len(row)<3 or (row[1]!='-' and row[2]!='G') for row in active):
        raise RuntimeError('Unexpected GPU0 compute process; no process stopped')
    out.mkdir(parents=True)
    lock=(HERE/'gpu0-static-preview.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cache=out/'cache'
    for folder in ('cuda','data','tmp','documents'):(cache/folder).mkdir(parents=True)
    os.environ.update(CUDA_VISIBLE_DEVICES=values[1],CUDA_DEVICE_ORDER='PCI_BUS_ID',
        PYTHONDONTWRITEBYTECODE='1',XDG_CACHE_HOME=str(cache),CUDA_CACHE_PATH=str(cache/'cuda'),TMPDIR=str(cache/'tmp'))
    started=time.monotonic()
    report=dict(schema='grasp_failure_recovery.static_preview.v1',status='initializing',
        gpu_admission=dict(index=0,uuid=values[1],inventory=occupancy,compute=compute,graphics=graphics,
            approved_shared=True,processes_stopped=[]),rows=[],physics_advanced=False,
        physical_settle_verified=False,scoring_executed=False)
    def save():
        report['wall_seconds']=time.monotonic()-started
        (out/'render_report.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    from isaacsim import SimulationApp
    app=SimulationApp(dict(headless=True,renderer='RaytracedLighting',multi_gpu=False,active_gpu=0,
        physics_gpu=0,width=1280,height=800,sync_loads=True,limit_cpu_threads=8,fast_shutdown=False,
        extra_args=['--/log/file='+str(out/'kit.log'),'--/app/userConfigPath='+str(out/'kit_user.json'),
            '--/app/tokens/cache='+str(cache),'--/app/tokens/data='+str(cache/'data'),
            '--/app/tokens/documents='+str(cache/'documents')]))
    try:
        import numpy as np
        import omni.usd,omni.timeline
        import omni.replicator.core as rep
        from PIL import Image
        from short_tasks.agent_probe_v1.native_owner import CAMERAS,camera_resolutions,valid_camera_frame
        for row in rows:
            record=json.loads(Path(row['initial']).read_text())
            product=ann=None
            receipt=dict(row,status='rendering',static_preview_only=True,physical_settle_verified=False)
            try:
                context=omni.usd.get_context();context.open_stage(record['scene'])
                omni.timeline.get_timeline_interface().stop()
                stage=context.get_stage();stage.SetEditTarget(stage.GetSessionLayer())
                receipt['reset']=static_reset(stage,record)
                resolution=camera_resolutions(row['variant'])
                product=rep.create.render_product(CAMERAS['head'],resolution['head'])
                ann=rep.AnnotatorRegistry.get_annotator('rgb');ann.attach([product])
                for _ in range(20):app.update()
                for _ in range(4):rep.orchestrator.step(rt_subframes=1,delta_time=0,pause_timeline=True)
                pixels=np.asarray(ann.get_data())
                if not valid_camera_frame('head',pixels,resolution) or pixels[...,:3].std()<2:
                    raise ValueError('Invalid or blank source head frame')
                image=out/(row['variant']+'_'+str(row['seed'])+'_head.png')
                Image.fromarray(pixels[...,:3]).save(image)
                receipt.update(status='rendered',image=str(image),camera=CAMERAS['head'],resolution=resolution['head'])
                print('RENDERED',row['variant'],row['seed'],flush=True)
            except Exception as error:
                receipt.update(status='failed',error=str(error),traceback=traceback.format_exc())
                print('RENDER_FAILED',row['variant'],row['seed'],str(error),flush=True)
            finally:
                if ann is not None:ann.detach()
                if product is not None:product.destroy()
                report['rows'].append(receipt);save()
        report.update(status='completed',rendered=sum(r['status']=='rendered' for r in report['rows']))
        save()
    finally:
        save();app.close();lock.close()


if __name__=='__main__':main()
