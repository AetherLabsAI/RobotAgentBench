"""Shared placement family; USD/Isaac imports stay lazy. No physics pass implied."""
from arena_api.bimanual import CONTROL_DT
from arena_api.task_runtime import TaskRegistration

TASK_ID = '5-5'
TIMEOUT_S = 60   # 5-x: user 2026-09-23. 30-x stay on the older budget below.
MEAL_TIMEOUT_S = 180
MAX_STEPS = round(TIMEOUT_S / CONTROL_DT)
MEAL_MAX_STEPS = round(MEAL_TIMEOUT_S / CONTROL_DT)
INSTRUCTION = '将手中已持有的儿童果汁瓶放到茶几上的托盘内部，然后松开夹爪。'
OBJECTS = ('酒精饮料', '橙色运动饮料', '蓝色运动饮料', '含咖啡因饮料', '儿童果汁瓶',
           '气泡水', '可乐', '雪碧', '牛肉棒', '薯片', '奥利奥饼干', '花生酱饼干', '蛋白棒', '糖果')
# Every 5-x places into the tray (user 2026-09-23); the old four-task split is gone.
TRAY_IDS = frozenset(f'5-{i}' for i in range(1, 15))
MEALS = {'30-1': '盒装牛排', '30-2': '盒装剩饭', '30-3': '沙拉饭盒'}
TASK_IDS = tuple(f'5-{i}' for i in range(1, 15))+tuple(MEALS)
PARAMETER_SCHEMA = dict(type='object', properties={
    'local_randomization': dict(type='boolean', default=False,
        description='Robot and held group: world X/Y each +/-2cm, yaw +/-8deg; destinations unchanged')},
    additionalProperties=False)


def normalize_parameters(parameters):
    if not isinstance(parameters, dict) or set(parameters) - {'local_randomization'}:
        raise ValueError('Only local_randomization is supported')
    enabled = parameters.get('local_randomization', False)
    if type(enabled) is not bool:
        raise ValueError('local_randomization must be boolean')
    return dict(local_randomization=enabled)


def runtime(context):
    from .api_runtime import CoffeeTablePlaceRuntime
    return CoffeeTablePlaceRuntime(context)


def registration(task_id=TASK_ID):
    if task_id not in TASK_IDS:
        raise ValueError('Unknown coffee-table placement task '+str(task_id))
    if task_id in MEALS:
        instruction = f'将手中托有{MEALS[task_id]}的盘子放到茶几桌面上，然后松开夹爪，保持饭盒留在盘子上。'
    else:
        instruction = f'将手中已持有的{OBJECTS[int(task_id.split("-")[1])-1]}放到茶几上的托盘内部，然后松开夹爪。'
    meal = task_id in MEALS
    steps = MEAL_MAX_STEPS if meal else MAX_STEPS
    description = instruction + (' 已校准初态，可选任务局部扰动；启动时必须通过物理验收；仅计分，完整回放及成功终止标准待验收。'
        if meal else ' 放入托盘内底、松开夹爪并连续静止 1 秒即成功结束；60 秒仿真时限，启动时必须通过物理验收。')
    return TaskRegistration(
        task_id=task_id,
        description=description,
        instruction=instruction,
        max_steps=steps, default_steps=steps, max_inferences=steps,
        evaluator_id='coffee_table_place.score_only.v1' if meal else 'coffee_table_place.settle.v3',
        runtime_factory=runtime,
        randomization_flags=(),
        parameter_schema=PARAMETER_SCHEMA, normalize_parameters=normalize_parameters,
    )


def registrations():
    return tuple(registration(task_id) for task_id in TASK_IDS)
