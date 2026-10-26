"""Read a packaged reset candidate; live physical acceptance is still mandatory."""
import json
from pathlib import Path
from arena_api.task_runtime import RobotInitialization
from .source_initial import TASK
from .initial_state import REPO, digest
from .bindings import binding_for

MEASURED_RESETS = frozenset(('30-1', '30-3', '5-10'))
SELECTED_RESETS = {'5-14': 'resources/candy_standup_v6/reset.json'}
# 用户 2026-09-23 的 5-1…5-14 定义表：每个 API 都放进茶几上的托盘，来源改为
# 2026-09-18 ziche 数据。5-x 默认加载 newdata_reset.py 生成的包；30-1…30-3 与
# 旧资源路径原样保留。把 newdata=False 传给 load_accepted_reset（或翻转
# USE_NEWDATA）即可回到旧资源做对比。
NEWDATA_RESETS = frozenset(f'5-{i}' for i in range(1, 15))
NEWDATA_DIRECTORY = 'resources/newdata_reset_20260924'
USE_NEWDATA = True


def legacy_manifest(task_id):
    """The pre-2026-09-18 resource for this API, kept for explicit comparison."""
    return TASK/(SELECTED_RESETS[task_id] if task_id in SELECTED_RESETS else
                 'resources/fixed_5_5_v1/reset.json' if task_id == '5-5' else
                 f'resources/measured_initial_v1/{task_id}/reset.json' if task_id in MEASURED_RESETS else
                 f'resources/fixed_variants_v1/{task_id}/reset.json')


def accepted_manifest(task_id, *, newdata=None):
    binding_for(task_id)  # Reject unknown IDs before resolving resources.
    if (USE_NEWDATA if newdata is None else newdata) and task_id in NEWDATA_RESETS:
        return TASK/NEWDATA_DIRECTORY/task_id/'reset.json'
    return legacy_manifest(task_id)


def load_accepted_reset(task_id, *, newdata=None):
    return load_reset_manifest(task_id, accepted_manifest(task_id, newdata=newdata))


def load_reset_manifest(task_id, manifest):
    """Validate a task-owned resource, including diagnostic pose candidates."""
    binding = binding_for(task_id)
    manifest = Path(manifest).resolve(strict=True)
    if not manifest.is_relative_to((TASK/'resources').resolve()):
        raise ValueError('Reset manifest must stay in task resources')
    data = json.loads(manifest.read_text())
    if (data['api_id'], data['held_body'], data['destination'], data['destination_kind']) != (
            task_id, binding.held_body, binding.destination, binding.destination_kind):
        raise ValueError('Reset/object/destination identity mismatch')
    if data.get('payload') != binding.payload:
        raise ValueError('Reset payload identity mismatch')
    scene = (manifest.parent/data['scene']).resolve(strict=True)
    if not scene.is_relative_to(manifest.parent.resolve()):
        raise ValueError('Reset scene escaped resource directory')
    deps = data['dependencies_repo_relative']
    if str(scene.relative_to(REPO.resolve())) not in deps:
        raise ValueError('Scene hash missing')
    for relative, expected in deps.items():
        path = (REPO/relative).resolve(strict=True)
        if not path.is_relative_to(REPO.resolve()) or digest(path) != expected:
            raise ValueError('Reset dependency changed: '+relative)
    values = data['robot_initialization']
    initialization = RobotInitialization(dict(values['positions']), dict(values['hold_targets']))
    # No action target conversion: the native hold targets are preserved.
    # Public reset validation still checks native limits/drives at execution.
    return scene, initialization, data
