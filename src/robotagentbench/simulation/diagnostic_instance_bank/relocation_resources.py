"""Original resource aliases for reserve-first shelf instance preparation.

The logical paths are compatibility names inside the isolated namespace, not
host installation paths. All bytes already belong to assets-0.1.0; no assets
are embedded, copied or changed by this module.
"""
import hashlib
import json
from pathlib import Path

LOGICAL_ROOT = '/home/lingjun/robot_agent_arena/'
RUNTIME_PREFIX = 'libraries/original_absolute/'
ALIASES = (
    ('layout_task_activation_v1/task_scene_registration_v1/coffee_initialized_rooms_v1/layout04/room.usda',
     '17ae94d735d85e2f6a080b8f5174fd91507bb86ace85812d6f6b86e19abb872d'),
    ('layout_task_activation_v1/task_scene_registration_v1/coffee_initialized_rooms_v1/layout04/room_fridge_boxes_v1.usda',
     '3e569cccc84a2d261bf8d8e413e9a87566c239c861d6196b0b630c0599a5748b'),
    ('layout_task_activation_v1/task_scene_registration_v1/coffee_initialized_rooms_v1/layout04/room_refrigerator_v1.usda',
     '1bac26b15dcca5ac1a7bd0364628c648b62b6063d9d83f855084ca2bc71d8ffd'),
    ('layout_task_activation_v1/task_scene_registration_v1/coffee_initialized_rooms_v1/layout04/room_task9_support_v1.usda',
     'f6b03eaffea899e7460eaadf0b9a7be51645caa48f1212c1e3d58e67cf19690f'),
    ('layout_task_activation_v1/task_scene_registration_v1/coffee_initialized_rooms_v1/layout04/room_tote_fixed_cpu_v1.usda',
     '7434ff6833158748cc3911d107971e2240648c3009f5705bb6d8e741aa87e796'),
    ('layout_task_activation_v1/task_scene_registration_v1/coffee_initialized_rooms_v1/layout04/room_tote_v1.usda',
     '4b140ad1177dcbd3811f9af4f69f80e558ddbd7d426be9f54347657d239db50d'),
    ('layout_task_activation_v1/task_scene_registration_v1/other_layouts_v3/layout04/room.usda',
     'be59086fe02c2998aac21e317550465076976a60c7c8e27630fe02ec0b8664c3'),
    ('layout_task_activation_v1/task_scene_registration_v1/other_layouts_v4/layout04/room.usda',
     'e0b05413e17bb8a1a16a8923b10d55d5ed72c45d8970df2637f69c8d60aadf00'),
    ('layout_task_activation_v1/task_scene_registration_v1/recorded_ready_pose_042_v1.json',
     '923cfd3ed5b4aae8b1d43687d83e2375364b1a89916ff0c4b186608f3ee73f41'),
    ('openend/tasks/all_objects_integration/room_all_dynamic_v2/room.usda',
     '6a8c78041b34969c7e9f52a8043f21f8c925506ad4130b96d4eed51e822ebadb'),
    ('openend/tasks/all_objects_integration/room_with_existing_drawer_v1/room.usda',
     'f4a1509c0c3e44eb69ab1f8f6459f2b9a3bc8517108d04c79cc216cb080e4ba4'),
)


def original_resource_aliases(runtime):
    """Return hash-verified read-only binds for a configured local Runtime.

    The caller mounts these specific files after their source-tree parents so
    that a source placeholder cannot shadow the separately installed asset.
    No directories, mountpoints, namespace or runtime metadata are modified.
    """
    root = Path(runtime).resolve(strict=True)
    metadata = json.loads((root/'runtime.json').read_text())
    if metadata.get('schema') != 'robotagentbench.local_installation.v1':
        raise ValueError('A configured local Runtime is required')
    original = Path(metadata['runtime'])
    bindings = []
    for tail, expected in ALIASES:
        target = LOGICAL_ROOT + tail
        relative = RUNTIME_PREFIX + target.lstrip('/')
        recorded = Path(metadata['resources'][relative])
        if recorded.relative_to(original).as_posix() != relative:
            raise ValueError('Original Runtime resource mapping changed: ' + relative)
        source = root/relative
        if source.resolve(strict=True) != source or not source.is_file():
            raise ValueError('Canonical installed resource file required: ' + relative)
        with source.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != expected:
            raise ValueError('Original resource bytes differ: ' + relative)
        bindings.append(dict(source=str(source), target=target, read_only=True))
    return bindings
