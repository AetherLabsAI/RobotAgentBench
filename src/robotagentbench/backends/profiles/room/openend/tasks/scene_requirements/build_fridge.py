"""Compose the existing fridge-to-microwave assembly, selected payload only.

Scene initialization only; shared Openend runtime integration is not implied.
"""
from pxr import UsdPhysics
from task_choices import BY_TYPE


def compose(stage, data, source_hashes, sha):
    requested=data['requirements'].get('fixed_task_requirements', [])
    if not requested:return [],[],{},[]
    task,=requested
    from replace_room_refrigerator_v1 import apply as replace_fridge
    from initialize_room_fridge_boxes_v1 import apply as initialize_boxes
    from fridge_to_microwave_v1.support import apply as initialize_support
    fridge=replace_fridge(stage)
    boxes=initialize_boxes(stage,seed=data['request']['seed'])
    selected=[]
    for row in boxes:
        if row['id']==task['payload_variant']:selected.append(row)
        else:
            # Only remove new, unrequested episode-local targets, never assets.
            if not stage.RemovePrim(row['path']):raise ValueError('Cannot omit unrequested fridge target')
    if len(selected)!=1:raise ValueError('Expected exactly one selected fridge box')
    supports=initialize_support(stage)
    rows=[fridge]+selected+supports
    for row in rows:
        row.update(task_id=task['id'],source='refrigerator')
        source_hashes[row['entry']]=sha(row['entry'])
    target=selected[0]
    p=stage.GetPrimAtPath(target['path'])
    if not p.HasAPI(UsdPhysics.RigidBodyAPI) or p.GetAttribute('physics:kinematicEnabled').Get() is True:
        raise ValueError('Fridge payload must preserve its dynamic rigid body')
    from fridge_to_microwave_v1.scene import Scene
    binding=Scene(stage,{'objects':rows},task['payload_variant'])
    result=dict(id=task['id'],type=task['type'],name=BY_TYPE[task['type']]['name'],
        description=BY_TYPE[task['type']]['description'],parameters={'payload_variant':task['payload_variant']},
        source='refrigerator',target_path=target['path'],target_count=1,
        object_paths=[r['path'] for r in rows],status='scene_bound',score=None,runtime_started=False)
    receipt=dict(fridge_box_initialization=dict(owner=target['initialization_owner'],seed=data['request']['seed'],
        box_count=1,selected_variant=task['payload_variant'],native_settling_validated=False),
        fridge_static_binding=dict(target=binding.target,source_paths=binding.source_paths,
                                  variant=binding.variant,native_physics_validated=False))
    pending=[]  # Identity/runtime connected; native validation remains separate.
    return rows,[result],receipt,pending
