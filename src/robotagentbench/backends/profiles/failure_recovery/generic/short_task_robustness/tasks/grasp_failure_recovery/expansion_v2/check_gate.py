"""Read-only comparison with the original full source geometric gate."""
import argparse
import json
from pathlib import Path
from ..common import owned,write_json
from .tipped_initial import load_base


def main():
    import numpy as np
    from pxr import Usd,UsdGeom
    from ..procedural_grasp import Kinematics
    from short_tasks.tasks.robot_pose_alignment_v1.geometry import checker
    from .operating_station import RoomGate
    p=argparse.ArgumentParser()
    p.add_argument('--initial',required=True)
    p.add_argument('--base-key',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    initial=json.loads(owned(a.initial).read_text())
    _,_,manifest,_,_=load_base(a.base_key,initial['original_bindings']['task_name'])
    stage=Usd.Stage.Open(initial['scene'])
    q=initial['robot_initialization']['positions']
    model=Kinematics(stage,initial['active_hand'],q)
    cache=RoomGate(stage,manifest)
    cached=cache.check(model,q)
    original=checker(stage,manifest,initial['initial_base_xyyaw'])({'targets_rad':q})
    report=dict(cached=cached,original=original,identical=cached==original,
        original_source_gate_called=True,source_parameter_changes=False,
        physics_advanced=False,native_contact_verified=False)
    if cached and cached.startswith('robot_room_overlap:'):
        collider=cached.split(':')[1]
        prim=stage.GetPrimAtPath(collider)
        body=next((b for path,b,_ in cache.robot if path==collider),None)
        if prim.IsA(UsdGeom.Cylinder) and body:
            frames=model.frames(q)
            xf=UsdGeom.XformCache()
            authored=np.asarray(xf.GetLocalToWorldTransform(stage.GetPrimAtPath(body))).T
            local=np.linalg.inv(authored)@np.asarray(xf.GetLocalToWorldTransform(prim)).T
            actual=frames[body]@local
            shape=UsdGeom.Cylinder(prim)
            axis='XYZ'.index(shape.GetAxisAttr().Get())
            radial=[i for i in range(3) if i!=axis]
            radial_extent=float(shape.GetRadiusAttr().Get())*float(np.linalg.norm(actual[2,radial]))
            axial_extent=float(shape.GetHeightAttr().Get())/2*abs(float(actual[2,axis]))
            report['analytic_cylinder']=dict(path=collider,
                radius_m=float(shape.GetRadiusAttr().Get()),height_m=float(shape.GetHeightAttr().Get()),
                actual_world_z_bounds_m=[float(actual[2,3]-radial_extent-axial_extent),
                                         float(actual[2,3]+radial_extent+axial_extent)],
                target_support_plane_z_m=initial['procedural_grasp']['height']['support_plane_z_m'],
                interpretation='plane_bounds_diagnostic_not_a_mesh_contact_certificate')
    write_json(owned(a.output,fresh=True),report)
    print(json.dumps(report))
    if cached!=original:raise ValueError('Cached gate differs from original; do not accept a candidate')


if __name__=='__main__':main()
