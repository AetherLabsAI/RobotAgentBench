"""Apply the confirmed source domain only before native physics attachment."""
from copy import deepcopy
import hashlib
from pathlib import Path
import sys
import numpy as np
from . import initial_station, position_domain


def prepare(plan, stage, manifest):
    from pxr import Gf, UsdGeom
    from short_tasks.scene import ROOT
    from short_tasks.tasks.sofa_to_drawer.native_geometry import shape_points
    if not stage.GetEditTarget().GetLayer().anonymous:
        raise ValueError('Task initialization requires an anonymous scene layer')
    delta, context, _ = initial_station.sofa_context(stage,manifest)
    path=context['target_path']; inverse=np.linalg.inv(delta)
    world=UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(path))
    origin=(np.array(world)[3]@inverse)[:3]
    points=np.vstack([v for _,v,c,visible in shape_points(stage,path) if c or visible])
    canonical=(np.c_[points,np.ones(len(points))]@inverse)[:,:3]
    offsets=canonical-origin
    footprint=np.c_[-offsets[:,1],offsets[:,0]].tolist()
    candidates=position_domain.sample_positions(plan['asset'],manifest['seed'],footprint,count=100)
    reg=ROOT/'layout_task_activation_v1/task_scene_registration_v1'
    previous=list(sys.path)
    try:
        sys.path.insert(0,str(reg))
        from randomize_other_living_v1 import seat_height_function, setworld, world as getworld
        from build_room_demo_v1 import exact, T
        height=seat_height_function(stage)
        other=[np.asarray(exact(stage,r['path'])) for r in manifest['objects']
            if r.get('source')=='sofa' and r['path']!=path]
        chosen=None
        for row in candidates:
            u,v=row['origin_uv_m']
            target=np.array([-.3503563938+v,-.8710-u,origin[2],1.])@delta
            translation=target[:3]-np.asarray(world.ExtractTranslation())
            setworld(stage,path,world*T(translation))
            box=exact(stage,path)
            setworld(stage,path,getworld(stage,path)*T([0,0,height(box)-box[0,2]]))
            box=np.asarray(exact(stage,path))
            if any(np.all(box[0]<b[1]) and np.all(b[0]<box[1]) for b in other):
                continue
            chosen=dict(row,visible_bounds_world_m=box.tolist())
            break
        if chosen is None:raise ValueError('No collision-free position in approved sofa domain')
    finally:
        sys.path[:]=previous
    result=deepcopy(manifest)
    for row in result['objects']:
        if row['path']==path:row['visible_bounds_world_m']=chosen['visible_bounds_world_m']
    result['sofa_table_source_initialization']=dict(chosen,
        orientation='unchanged_scene_owned_asset_orientation',
        support_height='existing_actual_seat_geometry_sampler',
        posture='approved_generic_folded_ready_not_dataset_fit',
        native_physics_validated=False)
    refs=[Path(__file__),Path(position_domain.__file__),position_domain.POLICY,
        Path(initial_station.__file__),initial_station.POLICY,
        reg/'randomize_other_living_v1.py',reg/'build_room_demo_v1.py',
        *map(Path,context['reference_files'])]
    result.setdefault('source_hashes',{}).update(
        {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in refs})
    result['short_task_scene_preparation']='sofa_to_coffee_table.approved_source_domain.v1'
    return result
