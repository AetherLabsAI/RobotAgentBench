"""Read-only camera-frustum diagnostic for a statically restored pose.

Projection is not an occlusion/visibility certificate. Camera parameters stay
unchanged; no physics, score, VLA or source scene edits.
"""
import argparse
import json
from pathlib import Path
from ..common import owned,write_json


def main():
    import numpy as np
    from pxr import Usd,UsdGeom,Gf
    from ..render_ready_fleet import static_reset
    from short_tasks.agent_probe_v1.native_owner import CAMERAS,camera_resolutions
    p=argparse.ArgumentParser();p.add_argument('--initial',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();record=json.loads(owned(a.initial).read_text())
    stage=Usd.Stage.Open(record['scene']);stage.SetEditTarget(stage.GetSessionLayer())
    static_reset(stage,record)
    target=stage.GetPrimAtPath(record['target'])
    bounds=UsdGeom.BBoxCache(0,['default','render','proxy'],False,True).ComputeWorldBound(target)
    r=bounds.ComputeAlignedRange();centre=(r.GetMin()+r.GetMax())/2
    result=dict(target=record['target'],bounds=[list(r.GetMin()),list(r.GetMax())],cameras={},
        physics_advanced=False,camera_parameters_changed=False,occlusion_verified=False)
    for name,path in CAMERAS.items():
        camera=UsdGeom.Camera(stage.GetPrimAtPath(path)).GetCamera(0)
        frustum=camera.frustum
        p4=Gf.Vec4d(*centre,1.)*(frustum.ComputeViewMatrix()*frustum.ComputeProjectionMatrix())
        ndc=[float(x/p4[3]) for x in p4[:3]] if p4[3]!=0 else None
        width,height=camera_resolutions(record['variant'])[name]
        result['cameras'][name]=dict(frustum_intersects=bool(frustum.Intersects(bounds)),
            centre_homogeneous_w=float(p4[3]),centre_ndc=ndc,
            centre_pixel=([float((ndc[0]+1)*width/2),float((1-ndc[1])*height/2)] if ndc else None),
            resolution=[width,height],camera_world=np.asarray(camera.transform).tolist())
    write_json(owned(a.output,fresh=True),result);print(json.dumps(result))


if __name__=='__main__':main()
