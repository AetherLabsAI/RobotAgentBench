"""Actual Isaac static rendering; existing render settings, approved auto framing."""
import argparse,json,hashlib,math,traceback
from pathlib import Path
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('episode',type=Path);ap.add_argument('--output',default='renders_v2',choices=['renders_v2','renders_v3','renders_v4']);ap.add_argument('--reuse-output',choices=['renders_v2','renders_v3']);ap.add_argument('--views',nargs='+');args=ap.parse_args()
if bool(args.reuse_output)!=bool(args.views):ap.error('--reuse-output and --views must be supplied together')
folder=args.episode.resolve();record=json.loads((folder/'composition.json').read_text())
output=folder/args.output;output.mkdir(exist_ok=False)
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'renderer':'RaytracedLighting','multi_gpu':False,'active_gpu':0,'physics_gpu':0,'width':1280,'height':800,'sync_loads':True,'limit_cpu_threads':8})
import omni.usd,omni.timeline,carb
import omni.replicator.core as rep
from pxr import UsdGeom,UsdLux,Gf
from PIL import Image
try:
    carb.settings.get_settings().set('/rtx/ecoMode/enabled',False)
    ctx=omni.usd.get_context();ctx.open_stage(record['scene']);stage=ctx.get_stage();stage.SetEditTarget(stage.GetSessionLayer())
    omni.timeline.get_timeline_interface().stop()
    hidden=[];cache=UsdGeom.BBoxCache(0,['default','render'])
    for p in stage.Traverse():
        path=str(p.GetPath());name=p.GetName().lower()
        if path.startswith('/World/Background/') and '/Lights/' not in path and ('ceiling' in name or 'roof' in name):
            if p.IsA(UsdGeom.Imageable):UsdGeom.Imageable(p).MakeInvisible();hidden.append(path)
        if path.startswith('/World/Background/') and '/Lights/' not in path and p.IsA(UsdGeom.Boundable) and ('wall' in name or '/Shell/' in path):
            r=cache.ComputeWorldBound(p).ComputeAlignedRange()
            if not r.IsEmpty() and r.GetMax()[1]<-2 and r.GetMax()[2]>1.5 and r.GetSize()[0]>.5:
                UsdGeom.Imageable(p).MakeInvisible();hidden.append(path)
    def look(prim,eye,target):
        UsdGeom.Xformable(prim).MakeMatrixXform().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*map(float,eye)),Gf.Vec3d(*map(float,target)),Gf.Vec3d(0,0,1)).GetInverse())
    cam=UsdGeom.Camera.Define(stage,'/World/LayoutPreviewCamera');cam.CreateFocalLengthAttr(28);cam.CreateHorizontalApertureAttr(36);cam.CreateClippingRangeAttr(Gf.Vec2f(.05,100))
    dome=UsdLux.DomeLight.Define(stage,'/World/LayoutPreviewDome');dome.CreateIntensityAttr(350)
    key=UsdLux.RectLight.Define(stage,'/World/LayoutPreviewKey');key.CreateIntensityAttr(900);key.CreateWidthAttr(8);key.CreateHeightAttr(5);look(key.GetPrim(),(0,0,7),(0,0,0))
    def fit(bounds):
        lo,hi=np.asarray(bounds);target=(lo+hi)/2;radius=np.linalg.norm(hi-lo)/2
        vertical_half=math.atan((36*800/1280)/2/28)
        distance=radius/math.sin(vertical_half)
        return target+np.array([0,-.6,.8])*distance,target
    b=record['overview']['bounds'];views=[('overview',*fit([[b[0],b[1],0],[b[2],b[3],2.4]]))]
    furniture=record['overview']['furniture']
    for name,keys in [('living',('Sofa','CoffeeTable','SmallCoffeeTable')),('kitchen',())]:
        boxes=([f['bounds'] for f in furniture if any(f['path'].endswith('/'+k) for k in keys)] if keys else
               [o['visible_bounds_world_m'] for o in record['objects'] if o.get('task_id') in
                {t['id'] for t in record['tasks'] if t['type'] in ('coffee','fryer','fridge_to_microwave')}])
        if boxes:
            lo=np.min(np.array(boxes)[:,0],axis=0);hi=np.max(np.array(boxes)[:,1],axis=0)
            views.append((name,*fit([lo,hi])))
    views=[dict(name=n,label={'overview':'整屋概览','living':'客厅近景','kitchen':'设备区近景'}[n],eye=list(e),target=list(t)) for n,e,t in views]
    from workspace_views import closeups
    known={o['path']:o['visible_bounds_world_m'] for o in record['objects']}
    known.update({f['path']:f['bounds'] for f in furniture})
    def bounds_of(path):
        if path in known:return known[path]
        p=stage.GetPrimAtPath(path)
        if not p:return None
        r=cache.ComputeWorldBound(p).ComputeAlignedRange()
        return None if r.IsEmpty() else [list(r.GetMin()),list(r.GetMax())]
    def front_of(path,axis):
        m=UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(path))
        return list(m.TransformDir(Gf.Vec3d(*axis)))[:2]
    views.extend(closeups(record,bounds_of,front_of))
    product=rep.create.render_product(str(cam.GetPath()),(1280,800));ann=rep.AnnotatorRegistry.get_annotator('rgb');ann.attach([product])
    from render_quality import record_view, completion_fields
    images=[];warnings=[]
    all_views=views;reuse=None
    if args.reuse_output:
        from render_reuse import reuse_images
        images,reuse=reuse_images(folder,output,args.reuse_output,args.views,all_views)
        views=[v for v in all_views if v['name'] in args.views]
    print('ROOM_PROGRESS '+json.dumps(dict(stage='render',message='GPU1：逐张渲染',completed=len(images),total=len(all_views)),ensure_ascii=False),flush=True)
    for attempted,view in enumerate(views,len(images)+1):
        name,eye,target=view['name'],view['eye'],view['target']
        look(cam.GetPrim(),eye,target)
        for _ in range(20):app.update()
        for _ in range(3):rep.orchestrator.step(rt_subframes=4,delta_time=0,pause_timeline=True)
        item,warning=record_view(ann.get_data(),view,output)
        if warning:
            warnings.append(warning)
            (output/'warnings.json').write_text(json.dumps(warnings,ensure_ascii=False,indent=2))
        else:
            images.append(item)
            print('RENDERED',name,flush=True)
        print('ROOM_PROGRESS '+json.dumps(dict(stage='render',message=(view['label']+'：可能存在问题，继续其他视角' if warning else view['label']),completed=attempted,total=len(all_views),valid_images=len(images),warnings=warnings),ensure_ascii=False),flush=True)
    completion=completion_fields(images,warnings,len(all_views),len(all_views))
    (output/'manifest.json').write_text(json.dumps(dict(renderer='Isaac RTX RaytracedLighting',resolution=[1280,800],
        scene_sha256=hashlib.sha256(Path(record['scene']).read_bytes()).hexdigest(),images=images,reuse=reuse,**completion,
        static_scene_only=True,native_physics_run=False,hidden_for_inspection_only=hidden,
        camera_policy='user approved boundary-based framing; robot cameras unchanged'),indent=2))
except BaseException as exc:
    (output/'error.json').write_text(json.dumps(dict(error=str(exc),traceback=traceback.format_exc()),ensure_ascii=False,indent=2))
    print('ROOM_PROGRESS '+json.dumps(dict(stage='render',message=str(exc),failed=True),ensure_ascii=False),flush=True)
    raise
finally:app.close()
