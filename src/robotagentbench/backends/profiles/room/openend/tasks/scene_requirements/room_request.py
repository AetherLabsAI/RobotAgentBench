"""Resolve explicit layout, or a seeded non-repeating fallback order."""
import json
import random
from pathlib import Path
from planner_v2 import compile_request
from task_choices import BY_TYPE

HERE = Path(__file__).resolve().parent

def resolve(request):
    from short_tasks.tasks.microwave_heat_take.generation import selected, resolve_request
    if selected(request):
        return resolve_request(request)
    request = json.loads(json.dumps(request))
    layouts = json.loads((HERE/'placement_sources/random-layouts-v1.json').read_text())['layouts']
    ids = sorted(l['id'] for l in layouts)
    seed = request.get('seed')
    if type(seed) is not int or not 1 <= seed <= 4294967295:
        raise ValueError('seed 必须是 1–4294967295 的整数')
    requested = request.get('layout_id')
    if requested is not None and (type(requested) is not int or requested not in ids):
        raise ValueError('未知布局编号')
    if requested is None:
        for t in request.get('tasks', []):
            c = t.get('placement', {})
            if any(k in c for k in ('surface_path','edge_index','world_xy_m','local_xy_m','local_bounds_m')):
                raise ValueError('固定坐标或特定台面约束必须显式指定布局')
    order=ids.copy()
    random.Random(seed).shuffle(order)
    if requested is not None:order=[requested]
    request['layout_id'] = order[0]
    if not request.get('tasks') or any(t.get('type') not in BY_TYPE for t in request['tasks']):
        raise ValueError('至少选择一项已接入的任务')
    # Cutlery multiplicity is bounded and validated against authored slots
    # by compile_request, just as shelf items have per-instance ownership.
    non_shelf=[t for t in request['tasks'] if t['type'] not in ('snack_drink_to_tray','cutlery_to_meal_tray')]
    if len({t['type'] for t in non_shelf}) != len(non_shelf):
        raise ValueError('每类任务选择一次；多件垃圾和书本请使用数量')
    for t in request['tasks']:
        allowed=BY_TYPE[t['type']].get('sources',[])
        if t.get('source') is not None and t['source'] not in allowed:
            raise ValueError('此预览尚无该来源区域的摆放适配：'+t['type'])
    if request.get('extras'):
        raise ValueError('此生成入口暂不接入额外物体')
    requirements = compile_request(request)
    groups={s['group'] for l in layouts for s in l['surfaces']}
    for t in request['tasks']:
        group=t.get('placement',{}).get('surface_group')
        if group is not None and group not in groups:raise ValueError('未知台面类型 '+str(group))
    layout = next(l for l in layouts if l['id'] == request['layout_id'])
    return dict(request=request, requirements=requirements, layout=layout,
                layout_candidates=order,
                layout_selection=dict(mode='specified' if requested is not None else 'seeded_random',
                                      requested=requested, selected=layout['id'], seed=seed, rerolls=0))
