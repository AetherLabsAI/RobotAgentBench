"""Render a saved Newton rule trial on the current House and current G1.

Robot links and cloth points come from the same saved state. This is a visual
replay, never an additional physics validation or a source-scene mutation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback

import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode",required=True)
    parser.add_argument("--geometry",required=True)
    parser.add_argument("--robot",required=True)
    parser.add_argument("--scene",default="/geniesim/main/demo_sim/scene_room.usda")
    parser.add_argument("--stride",type=int,default=2)
    parser.add_argument("--frames",help="Optional comma-separated diagnostic frame indexes")
    parser.add_argument("--views",nargs="+",choices=("overview","basket","head","left","right"),
                        help="Render only selected review views; defaults to all four")
    parser.add_argument("--end-seconds",type=float)
    args=parser.parse_args()
    from isaacsim import SimulationApp
    app=SimulationApp({"headless":True,"renderer":"RayTracedLighting","active_gpu":0,"physics_gpu":0,"multi_gpu":False})
    try:
        import omni.usd
        import omni.replicator.core as rep
        from pxr import Usd,UsdGeom,UsdShade,UsdPhysics,Gf,Sdf,Vt
        from PIL import Image,ImageDraw
        from scipy.spatial.transform import Rotation
        from tablecloth_robot_kinematics import Robot
        from tablecloth_scene_binding import verify_scene_collisions,station_xyyaw,source_record

        episode=Path(args.episode)
        output=episode/"render_4k"
        output.mkdir(exist_ok=True)
        state=dict(np.load(episode/"full_state.npz",allow_pickle=False))
        metadata=json.loads((episode/"trial.json").read_text())
        omni.usd.get_context().open_stage(args.scene)
        stage=omni.usd.get_context().get_stage()
        with np.load(args.geometry,allow_pickle=False) as geo:
            verify_scene_collisions(stage,geo)
        task=str(state["benchmark_task_id"])
        station = station_xyyaw(task)
        if "base_xyyaw_m_deg" in state and "robot_root_pose_xyzw" not in state:
            measured = np.asarray(state["base_xyyaw_m_deg"], dtype=float)
            if measured.shape != (len(state["time_s"]), 3) or not np.isfinite(measured).all():
                raise ValueError("Invalid measured base poses")
            if not np.allclose(measured, measured[0], atol=1e-8, rtol=0.):
                raise ValueError("Tablecloth renderer requires a fixed recorded initial base")
            station = measured[0]
        robot=Robot(json.loads(Path(args.robot).read_text()),station)
        root=stage.GetPrimAtPath("/World/Robot")
        UsdGeom.Xformable(root).ClearXformOpOrder()
        UsdGeom.Xformable(root).AddTransformOp().Set(Gf.Matrix4d(1))
        link_ops={}
        link_names=set(robot.fk({}))
        # Inner visual meshes can share a link name. Only direct rigid-link
        # children receive FK; retain every authored descendant transform.
        for prim in root.GetChildren():
            if prim.GetName() in link_names:
                xf=UsdGeom.Xformable(prim)
                xf.ClearXformOpOrder()
                link_ops[prim.GetName()]=xf.AddTransformOp()
        disabled=[]
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Set(False)
            if prim.IsA(UsdPhysics.Joint):
                UsdPhysics.Joint(prim).GetJointEnabledAttr().Set(False)
        # The tablecloth owner scenario starts with a clear manipulation area.
        # Keep table/basket transforms and assets unchanged; record this setup.
        hidden = metadata.get('scenario_hidden_prims',
            [f'/World/Assets/{name}' for name in ("Book","Note","Remote","CoffeeBeans","BaseballCap","Tray","PaperBall","GlassesCase","CushionOnBook")])
        for path in hidden:
            prim=stage.GetPrimAtPath(path)
            if prim and prim.IsActive():
                prim.SetActive(False);disabled.append(str(prim.GetPath()))
        mesh=UsdGeom.Mesh.Define(stage,"/World/Assets/TableclothRuleState")
        faces=state["faces"]
        mesh.CreateFaceVertexCountsAttr([3]*len(faces))
        mesh.CreateFaceVertexIndicesAttr(faces.reshape(-1).tolist())
        mesh.CreateDoubleSidedAttr(True)
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        uv=state["uv"]/state["uv"].max(0)
        UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st",Sdf.ValueTypeNames.TexCoord2fArray,UsdGeom.Tokens.vertex).Set(Vt.Vec2fArray.FromNumpy(np.ascontiguousarray(uv,dtype=np.float32)))
        # A simple woven-look border, authored procedurally rather than copied
        # from a photograph; the texture has no role in task success metrics.
        from tablecloth_texture import cloth_texture
        def cloth_material(label, dirty):
            texture=cloth_texture(dirty=dirty)
            texture_path=output/(label+"_cloth_border.png")
            texture.save(texture_path)
            material_path="/World/Looks/TableclothRuleMaterial_"+label
            mat=UsdShade.Material.Define(stage,material_path)
            shader=UsdShade.Shader.Define(stage,material_path+"/Surface")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("roughness",Sdf.ValueTypeNames.Float).Set(.9)
            reader=UsdShade.Shader.Define(stage,material_path+"/UV")
            reader.CreateIdAttr("UsdPrimvarReader_float2")
            reader.CreateInput("varname",Sdf.ValueTypeNames.Token).Set("st")
            tex=UsdShade.Shader.Define(stage,material_path+"/Texture")
            tex.CreateIdAttr("UsdUVTexture")
            tex.CreateInput("file",Sdf.ValueTypeNames.Asset).Set(str(texture_path))
            tex.CreateInput("sourceColorSpace",Sdf.ValueTypeNames.Token).Set("sRGB")
            tex.CreateInput("st",Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(),"result")
            shader.CreateInput("diffuseColor",Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(),"rgb")
            mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),"surface")
            return mat
        mat = cloth_material('primary', task in {'8.7', '8.8'})
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
        if 'cloth_id' in state and np.any(state['cloth_id'] == 1):
            face_cloth = state['cloth_id'][faces[:, 0]]
            if not np.all(state['cloth_id'][faces] == face_cloth[:, None]):
                raise ValueError('Cloth topology joins two distinct physical cloths')
            subset = UsdGeom.Subset.Define(stage, str(mesh.GetPath())+'/clean')
            subset.CreateElementTypeAttr(UsdGeom.Tokens.face)
            subset.CreateFamilyNameAttr('materialBind')
            subset.CreateIndicesAttr(np.flatnonzero(face_cloth == 1).tolist())
            UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(cloth_material('clean', False))
        support=metadata.get("linen_support")
        if support:
            centre=np.asarray(support["centre"]);half=np.asarray(support["half_extents"])
            for layer in range(5):
                item=UsdGeom.Cube.Define(stage,f"/World/Assets/TableclothNeighborLinens/layer_{layer}")
                item.CreateSizeAttr(1.)
                xform=UsdGeom.Xformable(item)
                position=centre.copy();position[2]=centre[2]-half[2]+(layer+.5)*2*half[2]/5
                xform.AddTranslateOp().Set(Gf.Vec3d(*position))
                xform.AddScaleOp().Set(Gf.Vec3f(float(half[0]*2),float(half[1]*2),float(half[2]*2/5-.001)))
                item.CreateDisplayColorAttr([(0.85,0.83,0.78)])
        overview=UsdGeom.Camera.Define(stage,"/World/TableclothOverview")
        overview.CreateFocalLengthAttr(28.)
        overview.CreateHorizontalApertureAttr(36.)
        overview.CreateClippingRangeAttr(Gf.Vec2f(.01,100.))
        view=Gf.Matrix4d().SetLookAt(Gf.Vec3d(-1.25,-1.55,1.50),Gf.Vec3d(-2.03,-.65,.75),Gf.Vec3d(0,0,1))
        UsdGeom.Xformable(overview).AddTransformOp().Set(view.GetInverse())
        cameras={"overview":overview.GetPrim()}
        basket_camera=UsdGeom.Camera.Define(stage,"/World/TableclothBasketReview")
        basket_camera.CreateFocalLengthAttr(35.)
        basket_camera.CreateHorizontalApertureAttr(36.)
        basket_camera.CreateClippingRangeAttr(Gf.Vec2f(.01,100.))
        basket_view=Gf.Matrix4d().SetLookAt(Gf.Vec3d(-2.24,-1.82,.92),
            Gf.Vec3d(-2.24,-.92,.38),Gf.Vec3d(0,0,1))
        UsdGeom.Xformable(basket_camera).AddTransformOp().Set(basket_view.GetInverse())
        cameras["basket"]=basket_camera.GetPrim()
        for label,name in (("head","Head_Camera"),("left","Left_Camera"),("right","Right_Camera")):
            found=[p for p in Usd.PrimRange(root) if p.GetName()==name and p.IsA(UsdGeom.Camera)]
            if len(found)!=1:raise ValueError(f"Expected one {name}")
            cameras[label]=found[0]
        if args.views:
            cameras={label:camera for label,camera in cameras.items() if label in args.views}
        annotators={}
        resolutions={}
        camera_readback={}
        for label,camera in cameras.items():
            native_size=camera.GetAttribute('omni:lensdistortion:opencvPinhole:imageSize').Get()
            resolution=tuple(map(int,native_size)) if native_size is not None else (960,640)
            original_size=resolution
            resolution={"head":(3840,2400),"overview":(3840,2560)}.get(label,resolution)
            if abs(resolution[0]/resolution[1]-original_size[0]/original_size[1]) > 1e-8:
                raise ValueError("HD render must preserve the recorded camera aspect ratio")
            resolutions[label]=resolution
            camera_readback[label]=dict(path=str(camera.GetPath()),parent=str(camera.GetParent().GetPath()),resolution=list(resolution),local_transform_row_major=np.asarray(UsdGeom.Xformable(camera).GetLocalTransformation()).tolist(),attributes={str(a.GetName()):str(a.Get()) for a in camera.GetAttributes() if str(a.GetName()).startswith(('omni:lensdistortion:','focalLength','horizontalAperture','verticalAperture','clippingRange','xformOp:'))},mount_and_intrinsics_modified=False)
            product=rep.create.render_product(str(camera.GetPath()),resolution)
            annotator=rep.AnnotatorRegistry.get_annotator("rgb")
            annotator.attach([product]);annotators[label]=annotator
            (output/label).mkdir(exist_ok=True)
        frames=[int(x) for x in args.frames.split(",")] if args.frames else list(range(0,len(state["cloth_q"]),args.stride))
        if not args.frames and frames[-1]!=len(state["cloth_q"])-1:frames.append(len(state["cloth_q"])-1)
        if args.end_seconds is not None:
            cutoff=int(round(args.end_seconds*metadata["fps"]))
            frames=[f for f in frames if f < cutoff]
        for index,frame in enumerate(frames):
            values=dict(zip(state["joint_names"].tolist(),state["joint_position"][frame]))
            poses=robot.fk(values)
            if 'robot_root_pose_xyzw' in state:
                # Native world link transforms include actual base motion and
                # joint response. Never replace these with prescribed FK.
                poses = {}
                for name, xf in zip(state['body_names'].tolist(), state['body_q'][frame]):
                    matrix = np.eye(4)
                    matrix[:3,:3] = Rotation.from_quat(xf[3:]).as_matrix()
                    matrix[:3,3] = xf[:3]
                    poses[name] = matrix
                missing = set(link_ops)-set(poses)
                if missing:
                    raise ValueError(f'Native state is missing robot link transforms: {sorted(missing)}')
            for link,op in link_ops.items():op.Set(Gf.Matrix4d(poses[link].T.tolist()))
            q=state["cloth_q"][frame]
            mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(q,dtype=np.float32)))
            normal=np.cross(q[faces[:,1]]-q[faces[:,0]],q[faces[:,2]]-q[faces[:,0]])
            normals=np.zeros_like(q)
            for j in range(3):np.add.at(normals,faces[:,j],normal)
            normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)
            mesh.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(normals,dtype=np.float32)))
            for _ in range(100 if index==0 else 2):app.update()
            rep.orchestrator.step(rt_subframes=4 if index==0 else 1,pause_timeline=True)
            # A cold shader/asset cache can delay annotator registration. Keep
            # the same state and allow bounded warm-up, never skip a data frame.
            for retry in range(40):
                if all(np.asarray(a.get_data()).shape[:2]==(resolutions[label][1],resolutions[label][0]) for label,a in annotators.items()):
                    break
                app.update()
                rep.orchestrator.step(rt_subframes=1,pause_timeline=True)
            for label,annotator in annotators.items():
                rgb=np.asarray(annotator.get_data())
                if rgb.shape[:2]!=(resolutions[label][1],resolutions[label][0]):raise ValueError(f"Missing {label} RGB: {rgb.shape}")
                Image.fromarray(rgb[:,:,:3]).save(output/label/f"{frame:06d}.jpg",quality=98,subsampling=0)
            if index%20==0:print(f"[render] {index+1}/{len(frames)} frame={frame}",flush=True)
        (output/'native_camera_readback.json').write_text(json.dumps(camera_readback,indent=2)+'\n')
        (output/"render.json").write_text(json.dumps(dict(schema="real2sim.tablecloth_rule_render.v1",benchmark_task_id=task,
            frames=frames,source_fps=metadata["fps"],stride=args.stride,views=list(cameras),
            source=source_record(),physics_source=metadata["source"],scene=args.scene,
            cloth_appearance=dict(dirty=task in {'8.7', '8.8'}, generator='procedural_uv_stains_v1', seed=8),
            visual_replay_only=True,scenario_hidden_prims=disabled),indent=2)+"\n")
    except Exception:
        traceback.print_exc()
        raise
    finally:
        app.close()


if __name__=="__main__":
    main()
