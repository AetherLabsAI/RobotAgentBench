"""Read-only original camera projection of a solved reset; not occlusion proof."""


def project_target(stage, frames, target):
    import numpy as np
    from pxr import UsdGeom,Gf
    from ..poses import world
    from short_tasks.agent_probe_v1.native_owner import CAMERAS
    bounds=UsdGeom.BBoxCache(0,['default','render','proxy'],False,True).ComputeWorldBound(stage.GetPrimAtPath(target))
    rr=bounds.ComputeAlignedRange();centre=(rr.GetMin()+rr.GetMax())/2
    result={}
    for name,path in CAMERAS.items():
        ancestors=[body for body in frames if path.startswith(body+'/')]
        if not ancestors:raise ValueError('Original camera has no solved FK body')
        body=max(ancestors,key=len)
        local=np.linalg.inv(world(stage,body))@world(stage,path)
        camera=UsdGeom.Camera(stage.GetPrimAtPath(path)).GetCamera(0)
        camera.transform=Gf.Matrix4d((frames[body]@local).T.tolist())
        frustum=camera.frustum
        p=Gf.Vec4d(*centre,1.)*(frustum.ComputeViewMatrix()*frustum.ComputeProjectionMatrix())
        ndc=[float(v/p[3]) for v in p[:3]] if p[3]!=0 else None
        inside=bool(p[3]>0 and ndc is not None and all(-1<=v<=1 for v in ndc))
        corners=[Gf.Vec3d(x,y,z) for x in (rr.GetMin()[0],rr.GetMax()[0])
            for y in (rr.GetMin()[1],rr.GetMax()[1]) for z in (rr.GetMin()[2],rr.GetMax()[2])]
        projected=[Gf.Vec4d(*v,1.)*(frustum.ComputeViewMatrix()*frustum.ComputeProjectionMatrix()) for v in corners]
        complete=all(v[3]>0 and all(-1<=u/v[3]<=1 for u in v[:3]) for v in projected)
        result[name]=dict(whole_target_envelope_inside_original_frustum=bool(complete),
            centre_inside_original_frustum=inside,centre_ndc=ndc,
            camera_path=path,parameters_changed=False,occlusion_verified=False)
    return result
