"""Render a measured native-physics recording, never generate a motion path."""
import argparse,json,os
from pathlib import Path
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--preview',action='store_true');args=ap.parse_args()
report=json.loads((args.run/'navigation_result.json').read_text())
assert report['navigation_posture']['episode']=='042_right_239.7deg'
assert report['rows'], 'Need actual recorded pose'
args.output.mkdir(parents=True,exist_ok=False)
poses=np.load(args.run/'native_poses.npz')
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'renderer':'RaytracedLighting','multi_gpu':False,'active_gpu':0,'physics_gpu':0,'width':1280,'height':800,'sync_loads':True,'limit_cpu_threads':8})
import omni.usd,omni.timeline,carb
import omni.replicator.core as rep
from pxr import Usd,UsdGeom,UsdLux,Gf
from PIL import Image,ImageDraw
try:
    carb.settings.get_settings().set('/rtx/ecoMode/enabled',False)
    ctx=omni.usd.get_context();ctx.open_stage(str(args.run/'native_initial.usdc'));stage=ctx.get_stage();stage.SetEditTarget(stage.GetSessionLayer())
    timeline=omni.timeline.get_timeline_interface();timeline.stop()
    # Cutaway visualization only: exterior shell keeps its physics in the native run.
    hidden=[]
    for p in stage.Traverse():
        path=str(p.GetPath());name=p.GetName().lower()
        if '/Background/Shell/' in path and ('ceiling' in name or 'roof' in name):
            if p.IsA(UsdGeom.Imageable):UsdGeom.Imageable(p).MakeInvisible();hidden.append(path)
        # Hide the south exterior wall in this inspection view, not in physics.
        if '/Background/Shell/' in path and p.IsA(UsdGeom.Boundable):
            r=UsdGeom.BBoxCache(0,['default','render']).ComputeWorldBound(p).ComputeAlignedRange()
            if not r.IsEmpty() and r.GetMax()[1]<-2.0 and r.GetMax()[2]>1.5 and r.GetSize()[0]>.5:
                UsdGeom.Imageable(p).MakeInvisible();hidden.append(path)
    def look(prim,eye,target):
        UsdGeom.Xformable(prim).MakeMatrixXform().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye),Gf.Vec3d(*target),Gf.Vec3d(0,0,1)).GetInverse())
    cam=UsdGeom.Camera.Define(stage,'/World/DemoCamera');cam.CreateFocalLengthAttr(28);cam.CreateHorizontalApertureAttr(36)
    cam.CreateClippingRangeAttr(Gf.Vec2f(.05,100))
    dome=UsdLux.DomeLight.Define(stage,'/World/DemoDome');dome.CreateIntensityAttr(350)
    key=UsdLux.RectLight.Define(stage,'/World/DemoKey');key.CreateIntensityAttr(900);key.CreateWidthAttr(8);key.CreateHeightAttr(5);look(key.GetPrim(),(0,0,7),(0,0,0))
    product=rep.create.render_product(str(cam.GetPath()),(1280,800));ann=rep.AnnotatorRegistry.get_annotator('rgb');ann.attach([product])
    body_paths=report['body_paths'];link_paths=['/World/Robot/'+n for n in report['robot_link_names']]
    all_paths=body_paths+link_paths
    scales={}
    for path in all_paths:
        p=stage.GetPrimAtPath(path);assert p,path
        scales[path]=Gf.Transform(UsdGeom.XformCache().GetLocalToWorldTransform(p)).GetScale()
    order=sorted(range(len(all_paths)),key=lambda i:all_paths[i].count('/'))
    def apply_frame(i):
        transforms=np.concatenate([poses['bodies'][i],poses['links'][i,0]],axis=0)
        for j in order:
            path=all_paths[j];p=stage.GetPrimAtPath(path);v=transforms[j]
            world=Gf.Matrix4d().SetScale(scales[path])*Gf.Matrix4d().SetRotate(Gf.Quatd(float(v[6]),Gf.Vec3d(*map(float,v[3:6]))))
            world.SetTranslateOnly(Gf.Vec3d(*map(float,v[:3])))
            parent=UsdGeom.XformCache().GetLocalToWorldTransform(p.GetParent())
            UsdGeom.Xformable(p).MakeMatrixXform().Set(world*parent.GetInverse())
    i=min(39,len(report['rows'])-1)
    apply_frame(i)
    x,y=report['rows'][i]['state'][-3:-1]
    # Pose inspection only: hide the room, preserve the recorded robot geometry.
    for path in ('/World/Background','/World/Assets','/World/RoomEpisode'):
        p=stage.GetPrimAtPath(path)
        if p and p.IsA(UsdGeom.Imageable):UsdGeom.Imageable(p).MakeInvisible();hidden.append(path)
    views=[('front',(x,y+3.0,1.5),(x,y,1.05)),('threequarter',(x+2.2,y+2.2,1.65),(x,y,1.05))]
    for name,eye,target in views:
        look(cam.GetPrim(),eye,target)
        for _ in range(20):app.update()
        for _ in range(3):rep.orchestrator.step(rt_subframes=4,delta_time=0,pause_timeline=True)
        Image.fromarray(np.asarray(ann.get_data())[:,:,:3]).save(args.output/(name+'.png'))
        print('PREVIEW',name,flush=True)
    (args.output/'render_manifest.json').write_text(json.dumps(dict(
        source_recording=str(args.run/'native_poses.npz'),source_posture=report['navigation_posture'],
        measured_frame=i,native_test_status=report['status'],hidden_for_inspection=hidden,
        method='Offline render of measured native link transforms; source dataset initial pose, no IK',
        proves_full_navigation=False),indent=2))
finally:app.close()

