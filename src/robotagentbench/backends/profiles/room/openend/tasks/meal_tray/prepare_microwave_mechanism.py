"""Install the approved EXISTING rotary mechanism in a new task-owned room layer.

Copies only the extracted visual shell, knob/joint and actual cavity light.
Never imports a source robot, source room layout, door setup or payload poses.
"""
import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf, Gf

HERE=Path(__file__).resolve().parent
ROOT='/World/Assets/DiningTableMicrowave/Asset'
SOURCE=Path('/mnt/agent-bench-data/lingjun/workspaces/l2-task106-pilot-20260925/task106_pilot/experiments/object_positions_20260925/microwave_yaw10_v3')


def physical_records(stage):
    result={}
    for p in stage.Traverse():
        schemas=tuple(s for s in p.GetAppliedSchemas() if 'Physics' in str(s) or 'Physx' in str(s))
        attrs={str(a.GetName()):str(a.Get()) for a in p.GetAttributes()
               if str(a.GetName()).startswith(('physics:','physx','drive:'))}
        rels={str(r.GetName()):list(map(str,r.GetTargets())) for r in p.GetRelationships()
              if str(r.GetName()).startswith(('physics:','physx'))}
        if schemas or attrs or rels:
            result[str(p.GetPath())]=dict(schemas=schemas,attrs=attrs,rels=rels)
    return result


def prepare(room, output):
    room=Path(room).resolve();output=Path(output).resolve()
    if not output.is_relative_to(HERE) or output.exists():raise ValueError('Fresh task-owned output required')
    current=Usd.Stage.Open(str(room))
    source=Usd.Stage.Open(str(SOURCE/'rotary_start_asset_v3_normals_fixed/rotary_start_overlay.usda'))
    rendered=Usd.Stage.Open(str(SOURCE/'rotary_materialfix_v4/R004/render_appearance_v2/scene_room.usda'))
    if not current or not source or not rendered:raise ValueError('Missing verified source or current room')
    if current.GetPrimAtPath(ROOT+'/StartKnob'):raise ValueError('Do not overwrite a room with an existing mechanism')
    if not current.GetPrimAtPath(ROOT+'/body'):raise ValueError('Missing current microwave')
    before=physical_records(current)
    # The measured pivot/collision geometry is valid only for the same asset
    # internal transform. A layout rigid transform is allowed; rescaling is not.
    caches=[UsdGeom.XformCache(),UsdGeom.XformCache()]
    root_linear=[np.asarray(c.GetLocalToWorldTransform(s.GetPrimAtPath(ROOT)))[:3,:3]
                 for c,s in zip(caches,(current,source))]
    if not np.allclose(root_linear[0]@root_linear[0].T,root_linear[1]@root_linear[1].T,atol=1e-7,rtol=0):
        raise ValueError('Mechanism mounting cannot silently rescale the microwave')
    # Door opening pose is scene state, not a knob mounting calibration. It is
    # deliberately preserved from the target room and never copied from source.
    for suffix in ('','/body','/body/col_panel'):
        prims=[s.GetPrimAtPath(ROOT+suffix) for s in (current,source)]
        if not all(prims):
            if suffix=='/body/col_panel':continue
            raise ValueError('Missing mechanism anchor '+suffix)
        if not all(UsdGeom.Xformable(p) for p in prims):continue
        relative=[c.GetLocalToWorldTransform(p)*c.GetLocalToWorldTransform(s.GetPrimAtPath(ROOT)).GetInverse()
                  for c,p,s in zip(caches,prims,(current,source))]
        if not np.allclose(relative[0],relative[1],atol=1e-7,rtol=0):
            raise ValueError('Current microwave internal geometry differs: '+suffix)
    output.mkdir()
    stage=Usd.Stage.CreateNew(str(output/'room.usda'))
    stage.GetRootLayer().subLayerPaths=[str(room)]
    UsdGeom.SetStageMetersPerUnit(stage,1);UsdGeom.SetStageUpAxis(stage,'Z')
    flat=source.Flatten()
    for suffix in ('/body/visual_shell','/StartKnob','/StartKnobJoint'):
        path=ROOT+suffix
        Sdf.CreatePrimInLayer(stage.GetRootLayer(),Sdf.Path(path).GetParentPath())
        if not Sdf.CopySpec(flat,path,stage.GetRootLayer(),path):raise ValueError('Copy failed '+path)
    # Reuse the CURRENT room shell material (source-layer look bindings are not
    # permission to recolour the room or introduce dangling source materials).
    material,_=UsdShade.MaterialBindingAPI(current.GetPrimAtPath(ROOT+'/body/visual_shell')).ComputeBoundMaterial()
    if not material:raise ValueError('Missing current shell material')
    for suffix in ('/body/visual_shell','/StartKnob/VisualAndCollision'):
        UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(ROOT+suffix)).Bind(
            UsdShade.Material(stage.GetPrimAtPath(material.GetPath())))
    lamp_path=ROOT+'/OpenendCavityLamp'
    actual_light='/World/MicrowaveCavityLamp'
    if not Sdf.CopySpec(rendered.Flatten(),actual_light,stage.GetRootLayer(),lamp_path):
        raise ValueError('Missing actual rendered source lamp')
    # Rigid relocation follows the microwave. Keep radius, colour and optics.
    cache=UsdGeom.XformCache()
    rel=cache.GetLocalToWorldTransform(rendered.GetPrimAtPath(actual_light))*cache.GetLocalToWorldTransform(rendered.GetPrimAtPath(ROOT)).GetInverse()
    UsdGeom.Xformable(stage.GetPrimAtPath(lamp_path)).MakeMatrixXform().Set(rel)
    stage.GetPrimAtPath(lamp_path).GetAttribute('inputs:intensity').Set(0.)
    if stage.GetPrimAtPath(actual_light):stage.OverridePrim(actual_light).SetActive(False)
    after=physical_records(stage)
    changed=[p for p in set(before)|set(after) if before.get(p)!=after.get(p)]
    allowed=[ROOT+'/StartKnob',ROOT+'/StartKnob/VisualAndCollision',ROOT+'/StartKnobJoint']
    if set(changed)!=set(allowed):raise ValueError('Unapproved physical changes '+str(changed))
    stage.GetRootLayer().Save()
    report=dict(schema='openend.microwave.mechanism_layer.v1',room=str(output/'room.usda'),
        source_room=str(room),source_room_sha256=hashlib.sha256(room.read_bytes()).hexdigest(),
        knob_path=ROOT+'/StartKnob',joint_path=ROOT+'/StartKnobJoint',lamp_path=lamp_path,
        on_intensity=600.,physical_changes=sorted(changed),existing_physics_preserved=True,
        source_assets_modified=False,native_physics_verified=False,
        runtime_parameters='existing RotaryReturnRuntime defaults, unchanged')
    (output/'mechanism.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--room',type=Path,required=True);a=p.parse_args()
    parent=Path(tempfile.mkdtemp(prefix='mechanism_',dir=HERE))
    print(json.dumps(prepare(a.room,parent/'layer'),indent=2))
