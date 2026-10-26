"""Add device-station constraints to the isolated inventory proposal.

Kitchen assemblies remain recipe requirements, not fabricated placed instances.
placement_constraints.cjs resolves them using original room planar geometry.
"""
from copy import deepcopy
from planner import compile_plan, fields, identifier, RequirementError
from task_choices import SHELF_PRODUCTS
import importlib.util
from pathlib import Path
_spec=importlib.util.spec_from_file_location('meal_product_claims',Path(__file__).resolve().parents[1]/'meal_tray/product_claims.py')
_claims=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_claims)


def compile_request(request):
    from short_tasks.tasks.microwave_heat_take.generation import selected, compile_request as compile_short
    if selected(request):
        return compile_short(request)
    fields(request, ('schema', 'layout_id', 'seed', 'tasks', 'extras', 'scenario', 'background'),
           ('schema', 'layout_id', 'seed', 'tasks'))
    background = None
    if 'scenario' in request or 'background' in request:
        from short_tasks.agent_probe_v1.cap_v1.native_tools.openend_control_adapter_v1.outing_scene_v3.policy import validate_background
        background = validate_background(request)
    if request['schema'] != 'openend.scene_requirements.request.v2':
        raise RequirementError('Expected v2 proposal schema')
    if type(request['layout_id']) is not int or request['layout_id'] < 1:
        raise RequirementError('Explicit positive layout ID required')
    if type(request['seed']) is not int or not 1 <= request['seed'] <= 4294967295:
        raise RequirementError('Station generator uses seed 1..4294967295')
    if not isinstance(request['tasks'], list):
        raise RequirementError('Task list required')
    stations, portable, fixed, shelf, meals, ids = [], [], [], [], [], set()
    destinations=[]
    wiping=[]
    # Validate original fields before expanding quantities; count cannot hide extras.
    for task in request['tasks']:
        if task.get('type')=='snack_drink_to_tray':
            fields(task, ('id','type','product','count'), ('id','type','product'))
    try:expanded=_claims.expand_scene_tasks(request['tasks'])
    except ValueError as exc:raise RequirementError(str(exc)) from exc
    for task in expanded:
        fields(task, ('id', 'type', 'source', 'count', 'items', 'categories', 'placement', 'payload_variant', 'product', 'asset', 'destination'), ('id', 'type'))
        ident = identifier(task['id'])
        if ident in ids:
            raise RequirementError('Duplicate task ID')
        ids.add(ident)
        if task['type']=='wipe_table':
            fields(task,('id','type'),('id','type'))
            if wiping:raise RequirementError('One wiping task per room')
            wiping.append(deepcopy(task))
        elif task['type']=='meal_tray_transfer':
            fields(task,('id','type'),('id','type'))
            if destinations:raise RequirementError('One shared meal tray transfer per room')
            destinations.append(deepcopy(task))
        elif task['type'] in ('coffee_to_meal_tray','bread_to_meal_tray','cutlery_to_meal_tray','microwave_box_to_meal_tray','salad_to_meal_tray','drink_to_meal_tray'):
            keys=('id','type','asset') if task['type']=='cutlery_to_meal_tray' else ('id','type','product') if task['type']=='drink_to_meal_tray' else ('id','type','payload_variant') if task['type']=='microwave_box_to_meal_tray' else ('id','type')
            fields(task,keys,keys)
            if task['type']=='microwave_box_to_meal_tray' and task['payload_variant'] not in ('beef_box','leftover_box'):
                raise RequirementError('微波炉任务必须选择牛肉盒或剩饭盒')
            if task['type']=='cutlery_to_meal_tray' and task['asset'] not in ('meal_fork','meal_knife'):
                raise RequirementError('餐具任务必须明确选择叉或刀')
            if task['type']=='drink_to_meal_tray' and task['product'] not in dict(SHELF_PRODUCTS[:8]):
                raise RequirementError('摆盘饮料必须选择已支持的饮料品类')
            if task['type']=='cutlery_to_meal_tray':
                cutlery=[m for m in meals if m['type']==task['type']]
                limit={'meal_knife':1,'meal_fork':2}[task['asset']]
                if sum(m['asset']==task['asset'] for m in cutlery)>=limit:
                    raise RequirementError('餐具架只有一刀两叉，不能重复使用同一实例')
            elif any(m['type']==task['type'] for m in meals):
                raise RequirementError('每类摆盘任务选择一次')
            meals.append(deepcopy(task))
        elif task['type']=='snack_drink_to_tray':
            fields(task, ('id','type','product'), ('id','type','product'))
            if task['product'] not in dict(SHELF_PRODUCTS):raise RequirementError('请选择已支持的货架品类')
            if len(shelf)>=4:raise RequirementError('饮料和零食合计最多选择 4 件')
            shelf.append(deepcopy(task))
        elif task['type']=='fridge_to_microwave':
            fields(task, ('id','type','payload_variant'), ('id','type','payload_variant'))
            if task['payload_variant'] not in ('beef_box','leftover_box'):
                raise RequirementError('冰箱任务必须选择牛肉盒或剩饭盒')
            if fixed:raise RequirementError('每间房只注册一个冰箱取盒任务')
            fixed.append(deepcopy(task))
        elif task['type'] in ('coffee', 'fryer'):
            fields(task, ('id', 'type', 'placement'), ('id', 'type'))
            if not isinstance(task.get('placement', {}), dict):
                raise RequirementError('Placement constraint must be an object')
            stations.append(deepcopy(task))
        else:
            if 'placement' in task or 'payload_variant' in task:
                raise RequirementError('Device station constraints cannot move portable objects')
            portable.append(deepcopy(task))
    if wiping and any(t['type']=='cloth_clear' for t in portable):
        raise RequirementError('Wiping replaces the exposed cloth-clear task; cannot combine both')
    for meal in meals:
        if meal['type']!='microwave_box_to_meal_tray':continue
        if len(fixed)!=1:
            raise RequirementError('微波炉取盒必须同时注册前置任务：冰箱取盒放入微波炉')
        if fixed[0]['payload_variant']!=meal['payload_variant']:
            raise RequirementError('冰箱交接与微波炉取盒必须选择同一种盒子')
    if len(shelf)+sum(m['type']=='drink_to_meal_tray' for m in meals)>4:
        raise RequirementError('货架目标合计最多4件，包含饮料摆盘和木托盘任务')
    for meal in meals:
        prerequisite={'coffee_to_meal_tray':'coffee','bread_to_meal_tray':'fryer'}.get(meal['type'])
        if prerequisite and sum(s['type']==prerequisite for s in stations)!=1:
            raise RequirementError('摆盘必须同时注册前置制作任务：'+prerequisite)
    if meals or destinations:
        tray_id='meal_tray_station'
        if tray_id in ids:raise RequirementError('Task ID reserved for the shared meal-tray station')
        stations.append(dict(id=tray_id,type='tray',placement={}))
    drawer=[t for t in portable if t.get('source')=='coffee_table.drawer']
    if any(t['type'] not in ('glasses_transfer','wallet_transfer') for t in drawer):
        raise RequirementError('抽屉来源目前仅支持墨镜盒或钱包')
    if sum(t.get('count',1) for t in drawer)>1:
        raise RequirementError('抽屉初态最多一个物体；请给墨镜盒或钱包之一选择其他来源')
    if background is not None:
        stations.extend(dict(id='background_'+kind,type=kind,placement={}) for kind in background['devices'])
    if len({s['type'] for s in stations}) != len(stations):
        raise RequirementError('One station per device task family')
    for extra in request.get('extras', []):
        if isinstance(extra, dict) and extra.get('id') in ids:
            raise RequirementError('Extra ID conflicts with task')
    base = compile_plan(dict(schema='openend.scene_requirements.request.v1', seed=request['seed'],
                             tasks=portable, extras=deepcopy(request.get('extras', []))))
    return dict(schema='openend.scene_requirements.plan.v2', executable=False,
                fixed_task_requirements=fixed, shelf_task_requirements=shelf, meal_task_requirements=meals,
                destination_task_requirements=destinations,
                **({'background_requirements':background} if background is not None else {}),
                **({'wipe_task_requirements':wiping} if wiping else {}),
                portable_inventory=base,
                station_request=dict(layout_id=request['layout_id'], seed=request['seed'], stations=stations),
                assembly_requirements=[dict(task_id=s['id'], type=s['type'],
                    initial_profile=('existing_corrected_coffee_source_relative_assembly' if s['type']=='coffee'
                                     else 'existing_task2_1_fryer_plate_two_cakes_assembly' if s['type']=='fryer'
                                     else 'existing_task10_5_meal_tray_and_cutlery_holder_assembly'),
                    transform_policy='one rigid station transform; preserve component relative transforms',
                    asset_instance_expansion_complete=False) for s in stations],
                pending=['station constraint validation/planar selection', 'full assembly geometry and asset expansion',
                         'unified scene generation and native validation'])
