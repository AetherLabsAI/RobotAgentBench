"""Tiny fake original-runtime modules for routing tests, not task scorers."""
import hashlib
from pathlib import Path

from robotagentbench.simulation import original_backend as routes
from robotagentbench.simulation.source_tree import NativeSourceTree
from robotagentbench.tasks.catalog import load_catalog


class SourceFixture:
    def __init__(self, directory):
        self.root = Path(directory).resolve()
        self.pins = {}
        self.write('short_tasks/__init__.py', '')
        self.write('short_tasks/registry.py',
            'TASK_NAMES = ' + repr(tuple(load_catalog('standalone').identifiers)) + '\n'
            + 'FAILURE_RECOVERY_TASKS = frozenset(name for name in TASK_NAMES if name.startswith("failure_recovery_"))\n'
            + 'ATOMIC_COFFEE_TASKS = frozenset(("coffee_bag_interference", "coffee_machine_push_interference", '
              '"coffee_bag_then_machine_interference", "coffee_button_lighting_interference"))\n')
        self.write('short_tasks/scene.py', 'def verify_scene_sources():\n    return {"original_scene_check": True}\n')
        for module, marker in ((routes.SHORT_SESSION, 'standalone'), (routes.TRASH_SELECTION, 'trash'),
                               (routes.OTHER_SELECTION, 'selection')):
            self.write(module.replace('.', '/') + '.py',
                'def build_session(request, *args, **kwargs):\n'
                f'    return {marker!r}, request, args, kwargs\n')
        self.write(routes.ROOM_OWNER.replace('.', '/') + '.py',
            'def admit_launch(*args, **kwargs):\n    return "room_admit", args, kwargs\n'
            'def build_room(*args, **kwargs):\n    return "room_build", args, kwargs\n'
            'def accepted_session(*args, **kwargs):\n    return "room_accept", args, kwargs\n')
        self.write(routes.MOBILE.replace('.', '/') + '.py', 'class Controller:\n    pass\n')
        self.write(routes.GUARD.replace('.', '/') + '.py', 'class NativeCollisionGuard:\n    pass\n')
        self.write(routes.ALIGNMENT.replace('.', '/') + '.py',
            'def apply(*args): pass\ndef verify(*args): pass\ndef verify_native_forces(*args): pass\n')
        self.write(routes.CODING.replace('.', '/') + '.py',
            'class InputRejected(ValueError):\n    pass\n'
            'def CodingHost(*args, **kwargs):\n    return "code_host", args, kwargs\n')
        self.write('vendor/fcl/__init__.py', 'FAKE_TEST_LIBRARY = True\n')

    def write(self, relative, text, *, pinned=True):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        if pinned:
            self.pins[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    def tree(self, *, native=False, coding=False):
        groups = {}
        if native:
            groups['native'] = ['vendor/fcl/__init__.py']
        if coding:
            groups['coding'] = [routes.CODING.replace('.', '/') + '.py']
        return NativeSourceTree(self.root, self.pins, groups=groups,
                                fcl_namespace='vendor/fcl' if native else None)
