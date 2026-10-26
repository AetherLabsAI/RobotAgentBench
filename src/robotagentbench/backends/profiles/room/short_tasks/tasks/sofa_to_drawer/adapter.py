"""Scene-owned wallet/notebook -> existing drawer; no object reset in scoring."""
from copy import deepcopy
import math
from short_tasks.tasks.pack_object.adapter import request_parts

RULE_VERSION='openend.short_task.sofa_to_drawer.v1'
VARIANTS={'wallet':('wallet_transfer',3,4), 'notebook_green':('notebook_transfer',5,6)}


def resolve(request):
    options,_,_=request_parts(request,'sofa_to_drawer',RULE_VERSION,('asset',))
    asset=options.get('asset','wallet')
    if asset not in VARIANTS: raise ValueError('Select wallet or notebook_green')
    component,pick,place=VARIANTS[asset]
    return dict(task_name='sofa_to_drawer',rule_version=RULE_VERSION,available=True,
        asset=asset,source='sofa',destination='coffee_table.drawer',
        scene_request=dict(schema='openend.scene_requirements.request.v2',seed=request['seed'],
            layout_id=request.get('layout_id'),tasks=[dict(id='sofa_to_drawer',type=component,source='sofa')],extras=[]),
        runtime_components=[dict(instance_id='sofa_to_drawer',type=component,parameters={})],
        instruction='Pick up the '+('wallet' if asset=='wallet' else 'green notebook')+' from the sofa, open the coffee table drawer, place it inside, and close the drawer.',
        score_policy=dict(maximum=pick+place,process_maximum=pick+(2 if asset=='wallet' else 3),
            final_maximum=2 if asset=='wallet' else 3),
        scene_preparation='short_tasks.tasks.sofa_to_drawer.scene.prepare',
        initial_requirements=['one_scene_owned_sofa_target','existing_closed_dynamic_drawer'],
        native_physics_validated=False)


def bind(plan, manifest):
    rows=[r for r in manifest['objects'] if r.get('id')==plan['asset']]
    if len(rows)!=1:raise ValueError('Exactly one selected target required')
    row=rows[0]
    if (row.get('source')!='sofa' or row.get('task_id')!=plan['task_name']
            or row.get('role') in ('background','extra','infrastructure')):
        raise ValueError('Target must be scene-owned by the selected sofa task')
    destination='/World/Assets/CoffeeTable/black_box_1' if plan['destination']=='coffee_table.drawer' else '/World/Assets/CoffeeTable'
    plan['target_bindings']=[dict(asset=plan['asset'],target_path=row['path'],destination_path=destination)]
    return deepcopy(plan['runtime_components'])


def normalize(plan, summaries):
    name=plan['task_name']
    if set(summaries)!={name}:raise ValueError('One current component summary required')
    row=summaries[name];scores=row['scores']
    if row['instance_id']!=name or row['asset']!=plan['asset']:
        raise ValueError('Wrong component identity')
    if set(scores)!={'process','current_final','deductions','total','maximum'}:
        raise ValueError('Explicit process/final score channels required')
    if any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in scores.values()):
        raise ValueError('Invalid score channels')
    policy=plan['score_policy']
    if (scores['maximum']!=policy['maximum'] or scores['process']>policy['process_maximum']
            or scores['current_final'] not in (0,policy['final_maximum'])
            or scores['total']!=max(0,scores['process']+scores['current_final']-scores['deductions'])):
        raise ValueError('Score arithmetic or approved maximum mismatch')
    if type(row['success']) is not bool or type(row['operation_complete']) is not bool:
        raise ValueError('Measured current success required')
    if row['success'] and (not row['operation_complete'] or not scores['current_final']):
        raise ValueError('Success requires completed process and current destination')
    for bound in plan.get('target_bindings',[]):
        if any(row[k]!=bound[k] for k in ('target_path','destination_path')):
            raise ValueError('Current evidence does not match bound target/destination')
    return dict(scores=deepcopy(scores),success=row['success'],
        stages=[deepcopy(row['pickup']),deepcopy(row['placement'])],
        targets=[dict(asset=plan['asset'],target_path=row['target_path'],destination_path=row['destination_path'])])


def factory_override(plan, declaration):
    if declaration not in plan['runtime_components']:raise ValueError('Unbound declaration')
    from .native_port import WalletDrawerPort,NotebookDrawerPort
    return {'wallet':WalletDrawerPort,'notebook_green':NotebookDrawerPort}[plan['asset']]
