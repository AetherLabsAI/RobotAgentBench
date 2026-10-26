"""Actual USD static previews; deliberately no native-physics success claim."""
import argparse,json
from pathlib import Path
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--layout',type=int,required=True);a=ap.parse_args()
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'renderer':'RaytracedLighting','multi_gpu':False,'active_gpu':0,'physics_gpu':0,'width':1280,'height':800,'sync_loads':True,'limit_cpu_threads':8})
import omni.usd,omni.timeline,carb
import omni.replicator.core as rep
from pxr import UsdGeom,UsdLux,Gf
from PIL import Image
try:
    carb.settings.get_settings().set('/rtx/ecoMode/enabled',False)
    for record in json.loads((a.run/'summary.json').read_text()):
        if a.layout and record['layout_id']!=a.layout:continue
        folder=Path(record['scene']).parent;output=folder/'detail_fix'
        if output.exists():
            for name in ['overview','kitchen','coffee','shelf']:
                with Image.open(output/(name+'.png')) as im:im.load();assert im.size==(1280,800)
            (output/'manifest.json').write_text(json.dumps({'layout_id':record['layout_id'],'static_scene_only':True,'native_physics_run':False,'recovered_after_render_product_cleanup_failure':True,'images_verified':4},indent=2))
            continue
        output.mkdir(exist_ok=False)
        ctx=omni.usd.get_context();ctx.open_stage(record['scene']);stage=ctx.get_stage();stage.SetEditTarget(stage.GetSessionLayer())
        omni.timeline.get_timeline_interface().stop()
        hidden=[]
        for p in stage.Traverse():
            path=str(p.GetPath());name=p.GetName().lower()
            if '/Background/Shell/' in path and ('ceiling' in name or 'roof' in name):
                if p.IsA(UsdGeom.Imageable):UsdGeom.Imageable(p).MakeInvisible();hidden.append(path)
            if '/Background/Shell/' in path and p.IsA(UsdGeom.Boundable):
                r=UsdGeom.BBoxCache(0,['default','render']).ComputeWorldBound(p).ComputeAlignedRange()
                if not r.IsEmpty() and r.GetMax()[1]<-2 and r.GetMax()[2]>1.5 and r.GetSize()[0]>.5:
                    UsdGeom.Imageable(p).MakeInvisible();hidden.append(path)
        def look(prim,eye,target):
            UsdGeom.Xformable(prim).MakeMatrixXform().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*map(float,eye)),Gf.Vec3d(*map(float,target)),Gf.Vec3d(0,0,1)).GetInverse())
        cam=UsdGeom.Camera.Define(stage,'/World/LayoutPreviewCamera');cam.CreateFocalLengthAttr(28);cam.CreateHorizontalApertureAttr(36);cam.CreateClippingRangeAttr(Gf.Vec2f(.05,100))
        dome=UsdLux.DomeLight.Define(stage,'/World/LayoutPreviewDome');dome.CreateIntensityAttr(350)
        key=UsdLux.RectLight.Define(stage,'/World/LayoutPreviewKey');key.CreateIntensityAttr(900);key.CreateWidthAttr(8);key.CreateHeightAttr(5);look(key.GetPrim(),(0,0,7),(0,0,0))
        p=record['supports']['coffee'];target=np.array([*p['origin'],p['actual_top_m']+.18]);front=np.array([*p['front'],0])
        views=[('overview',(0,-6.8,8.8),(.4,0,.3)),('sofa',(-2.6,-1,2.5),(0,-.87,.45)),
               ('table',(-3.6,-1.9,2.4),(-1.92,-.1,.45)),('small_table',(-2.1,-3.6,1.6),(-2.1,-2,.45)),
               ('kitchen',(1,-3.8,4.8),(3.5,0,.9)),('coffee',target+front*1.7+np.array([0,0,.8]),target),('shelf',(-2.35,0,2.1),(-2.35,2.25,1.02))]
        views=[('small_table',(-2.1,-.9,1.5),(-2.1,-2,.45))]
        product=rep.create.render_product(str(cam.GetPath()),(1280,800));ann=rep.AnnotatorRegistry.get_annotator('rgb');ann.attach([product])
        for name,eye,target in views:
            look(cam.GetPrim(),eye,target)
            for _ in range(20):app.update()
            for _ in range(3):rep.orchestrator.step(rt_subframes=4,delta_time=0,pause_timeline=True)
            Image.fromarray(np.asarray(ann.get_data())[:,:,:3]).save(output/(name+'.png'))
            print('RENDERED',record['layout_id'],name,flush=True)
        (output/'manifest.json').write_text(json.dumps({'layout_id':record['layout_id'],'static_scene_only':True,'native_physics_run':False,'hidden_for_inspection_only':hidden},indent=2))
finally:app.close()
