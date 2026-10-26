"""Approved per-object pickup plus three reversible legal-table-region points."""
from short_tasks.tasks.pack_object.adapter import request_parts
from short_tasks.tasks.sofa_to_drawer.adapter import bind, normalize

RULE_VERSION='openend.short_task.sofa_to_coffee_table.v1'
VARIANTS={'book_red':('book_transfer',5),'remote_tv':('remote_transfer',3),
    'wallet':('wallet_transfer',3),'glasses_case':('glasses_transfer',3)}


def resolve(request):
    options,_,_=request_parts(request,'sofa_to_coffee_table',RULE_VERSION,('asset',))
    asset=options.get('asset','remote_tv')
    if asset not in VARIANTS:raise ValueError('Select book, remote, wallet or glasses case')
    component,points=VARIANTS[asset]
    return dict(task_name='sofa_to_coffee_table',rule_version=RULE_VERSION,available=True,
        asset=asset,source='sofa',destination='coffee_table.objects',
        scene_request=dict(schema='openend.scene_requirements.request.v2',seed=request['seed'],
            layout_id=request.get('layout_id'),tasks=[dict(id='sofa_to_coffee_table',type=component,source='sofa')],extras=[]),
        runtime_components=[dict(instance_id='sofa_to_coffee_table',type=component,parameters={})],
        instruction='Pick up the '+{'book_red':'book','remote_tv':'TV remote control','wallet':'wallet','glasses_case':'sunglasses case'}[asset]+' from the sofa, place it in the permitted object area on the coffee table, and release it.',
        score_policy=dict(maximum=points+3,process_maximum=points,final_maximum=3),
        initial_requirements=['unique_scene_owned_sofa_target','approved_table_objects_region'],
        scene_preparation='short_tasks.tasks.sofa_to_coffee_table.scene.prepare',
        scene_runtime_available=True,native_physics_validated=False)


def factory_override(plan, declaration):
    if declaration not in plan['runtime_components']:raise ValueError('Unbound declaration')
    from short_tasks.tasks.sofa_to_drawer.native_port import CombinationPort
    return type('SofaTable_'+plan['asset'],(CombinationPort,),dict(asset=plan['asset'],
        mode='surface',key=plan['asset']+'_to_coffee_table'))
