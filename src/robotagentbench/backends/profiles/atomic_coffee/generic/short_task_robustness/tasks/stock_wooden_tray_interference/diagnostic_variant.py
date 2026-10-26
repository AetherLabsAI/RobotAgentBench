"""Select an existing shelf instance for isolated diagnostics, never move it."""
from copy import deepcopy
from short_tasks.tasks.stock_wooden_tray_interference.unique_target import require_unique_target


def bind_existing_drink(manifest, product):
    if product not in ('sprite_c1','gatorade_t'):
        raise ValueError('No fitted diagnostic grip for '+product)
    m=deepcopy(manifest)
    require_unique_target(m,product)
    if product=='sprite_c1':
        matches=[o for o in m['objects'] if o.get('task_id')=='stock_item_2' and o['id']==product]
    else:
        matches=[o for o in m['objects'] if o.get('id')==product and o.get('source')=='snack_shelf']
    if len(matches)!=1:raise ValueError('Diagnostic requires one exact existing drink')
    obj=matches[0]
    task_id=obj.get('task_id')
    if not task_id:
        task_id='human_diagnostic_drink'
        obj.update(task_id=task_id,task_role='requested_full_product',task_group='snack_drink_to_tray')
        m['tasks'].append(dict(id=task_id,type='snack_drink_to_tray',source='snack_shelf',
            parameters={'product':product},target_path=obj['path'],object_paths=[obj['path']],
            status='diagnostic_binding_only',runtime_started=False))
        declarations=m.setdefault('registration',{}).setdefault('request',{}).setdefault('tasks',[])
        declarations.append(dict(instance_id=task_id,type='snack_drink_to_tray',parameters={'product':product}))
    return m,task_id,obj
