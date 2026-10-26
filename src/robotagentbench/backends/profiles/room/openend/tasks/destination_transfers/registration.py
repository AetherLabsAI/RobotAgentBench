"""Explicit destination selection; omitted destination retains the old bag."""
BASKET='/World/Assets/TableclothBasket'
TABLE='/World/Assets/CoffeeTable'

def destination(parameters):
    if not isinstance(parameters,dict) or set(parameters)-{'destination'}:
        raise ValueError('Only destination may be selected')
    value=parameters.get('destination','fixed_cpu_bag')
    if value not in ('fixed_cpu_bag','storage_basket'):
        raise ValueError('Select fixed_cpu_bag or storage_basket')
    return value

def tray_parameters(parameters):
    if parameters != {}:raise ValueError('Tray identity is scene-owned; no overrides')

def attach(stage,requirements,rows,tasks):
    """Bind existing infrastructure and one shared tray; never spawn duplicates."""
    for task in tasks:
        if task['type'] not in ('glasses_transfer','wallet_transfer'):continue
        item=next(t for t in requirements['portable_inventory']['tasks'] if t['id']==task['id'])
        value=destination(item.get('parameters',{}))
        task['parameters']={'destination':value}
        task['destination_role']=value
        label='垃圾桶旁收纳篮' if value=='storage_basket' else '固定开口包'
        task['name']=('墨镜盒' if task['type']=='glasses_transfer' else '钱包')+'归位'
        task['description']=('墨镜盒' if task['type']=='glasses_transfer' else '钱包')+' → '+label+'；沿用7分'
        if value=='storage_basket':
            p=stage.GetPrimAtPath(BASKET)
            if not p or not p.IsActive():raise ValueError('Layout lacks selected storage basket')
            task['destination_path']=BASKET
    for t in requirements.get('destination_task_requirements',[]):
        if t['type']!='meal_tray_transfer':raise ValueError('Unknown destination task')
        matches=[r for r in rows if r['id']=='meal_tray_large']
        if len(matches)!=1:raise ValueError('Exactly one shared meal tray required')
        p=stage.GetPrimAtPath(TABLE)
        if not p or not p.IsActive():raise ValueError('Layout lacks coffee table')
        tasks.append(dict(id=t['id'],type=t['type'],name='搬运大餐盘到茶几',
            description='空盘可搬；持有并抬离原承托2分，茶几指定餐盘区终态3分',
            parameters={},target_path=matches[0]['path'],object_paths=[matches[0]['path']],
            destination_path=TABLE,destination_workspace='table_objects',
            status='scene_bound',score=None,runtime_started=False,target_count=1))
