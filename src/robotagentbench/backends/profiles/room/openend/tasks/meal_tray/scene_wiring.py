"""Bind approved episode-owned assemblies; no transforms or physics writes."""
from .native_port import one

NAMES={'coffee_to_meal_tray':'咖啡摆盘','bread_to_meal_tray':'面包摆盘','cutlery_to_meal_tray':'餐具摆盘'}
NAMES['microwave_box_to_meal_tray']='微波炉旋钮亮灯、取盒摆盘'
NAMES.update(salad_to_meal_tray='沙拉盒摆盘',drink_to_meal_tray='饮料摆盘')

def select_cutlery_members(rows,requests):
    """Pick authored instances without cloning, moving or changing assets."""
    from copy import deepcopy
    manifest={'objects':rows}
    members=[deepcopy(one(manifest,asset)) for asset in ('meal_tray_large','cutlery_holder')]
    used={r['path'] for r in members}
    for request in requests:
        choices=sorted((r for r in rows if r['id']==request['asset'] and r['path'] not in used),
                       key=lambda r:r['path'])
        if not choices:raise ValueError('Not enough distinct authored cutlery instances')
        row=deepcopy(choices[0]);row['meal_task_id']=request['id']
        used.add(row['path']);members.append(row)
    return members

def attach_tasks(requests,rows,tasks):
    if not requests:return
    manifest={'objects':rows}
    tray=one(manifest,'meal_tray_large')
    for task in requests:
        kind=task['type'];parameters={}
        if kind=='coffee_to_meal_tray':
            target=one(manifest,'coffee_cup_blue_compact')
        elif kind=='bread_to_meal_tray':
            upstream,=[t for t in tasks if t['type']=='fryer']
            variant=upstream.get('parameters',{'bread_variant':'no_sugar'})['bread_variant']
            target=one(manifest,'cake_plain_yellow' if variant=='no_sugar' else 'cake_frosted_white')
        elif kind=='cutlery_to_meal_tray':
            owned=[r for r in rows if r['id']==task['asset'] and r.get('meal_task_id')==task['id']]
            if len(owned)==1:target=owned[0]
            elif not owned:
                target=one(manifest,task['asset'])
                if target.get('meal_task_id') not in (None,task['id']):
                    raise ValueError('Cutlery already belongs to a different task')
            else:raise ValueError('Ambiguous task-owned cutlery')
            parameters={'asset':target['id'],'target_path':target['path']}
        elif kind=='microwave_box_to_meal_tray':
            target=one(manifest,task['payload_variant'])
            parameters={'payload_variant':task['payload_variant']}
            upstream,=[t for t in tasks if t['type']=='fridge_to_microwave']
            if upstream['target_path']!=target['path'] or upstream['parameters']!=parameters:
                raise ValueError('Microwave must use the fridge task same-box handoff')
        elif kind=='salad_to_meal_tray':
            target=one(manifest,'salad_box')
        elif kind=='drink_to_meal_tray':
            from .product_claims import exact_object
            parameters={'product':task['product']}
            target=exact_object(parameters,manifest,task['id'])
        else:raise ValueError('Unimplemented meal scene recipe '+kind)
        target['meal_task_id']=task['id']
        if kind=='microwave_box_to_meal_tray':
            target.pop('score_owner_task_id',None)
            target['score_owner_task_ids']=[upstream['id'],task['id']]
            target['sequential_score_owners']={'fridge_to_microwave':upstream['id'],
                'illumination_extract_meal':task['id']}
        else:target['score_owner_task_id']=task['id']
        tasks.append(dict(id=task['id'],type=kind,name=NAMES[kind],parameters=parameters,
            target_path=target['path'],destination_path=tray['path'],
            object_paths=[target['path'],tray['path']],target_count=1,
            status='scene_bound',score=None,runtime_started=False))
        if kind=='microwave_box_to_meal_tray':
            tasks[-1].update(prerequisite_instance_id=upstream['id'],
                handoff_status='waiting_for_fridge_to_microwave',same_target_at_handoff=True,
                source='refrigerator',scene_reset_at_handoff=False)
