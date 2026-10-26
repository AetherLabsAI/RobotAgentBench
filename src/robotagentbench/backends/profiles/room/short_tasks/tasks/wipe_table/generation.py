"""Narrow task-owned output admission for the unchanged CPU room composer."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
POLICY = dict(task_name="wipe_table", version="short_tasks.wipe_table.scene.v1")


def validate_output(source, root, data):
    source, root = Path(source).resolve(), Path(root).resolve()
    task_root = (root/"short_tasks/tasks/wipe_table").resolve()
    if task_root != HERE or task_root.is_symlink():
        raise ValueError("Wiping generator must belong to the registered task")
    if (not source.is_relative_to(task_root/"execution")
            or source.name != "placement.json" or source.is_symlink()):
        raise ValueError("Fresh wiping placement inside task execution required")
    if (data.get("task_scene_policy") != POLICY
            or data.get("openend_episode_policy") is not None
            or data.get("request", {}).get("tasks") != [dict(id="wipe", type="wipe_table")]):
        raise ValueError("Exactly the registered wiping recipe is required")
    if (source.parent/"room.usda").exists():
        raise ValueError("Existing episode scene cannot be replaced")
