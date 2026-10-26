"""Owned temporary prepared-scene fixtures; no simulator or task scoring."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from robotagentbench.simulation.task_backend import TaskBackend
from robotagentbench.simulation.workspace import NativeWorkspace


class AdmissionFixture:
    def __init__(self, directory):
        self.root = Path(directory).resolve()
        self.inputs = self.root / 'inputs'
        self.outputs = self.root / 'outputs'
        self.locks = self.root / 'locks'
        for path in (self.inputs, self.outputs, self.locks):
            path.mkdir()
        self.workspace = NativeWorkspace((self.inputs,), self.outputs, self.locks)
        self.scene = self.inputs / 'scene.usda'
        self.scene.write_text('#usda 1.0\n')
        self.manifest = self.inputs / 'manifest.json'
        self.request = self.inputs / 'request.json'
        self.ready = self.inputs / 'ready.json'
        self.launch = self.inputs / 'launch.json'
        self.output = self.outputs / 'episode'
        self.events = []
        self.plan = {'task_name': 'test_task', 'available': True, 'instruction': 'Test request',
                     'rule_version': 'test.rule.v1', 'score_policy': {'maximum': 17}}
        fixture = self

        class Registry:
            FAILURE_RECOVERY_TASKS = frozenset({'recovery_case'})
            ATOMIC_COFFEE_TASKS = frozenset({'atomic_case'})

            def resolve(self, request):
                fixture.events.append('resolve')
                return dict(fixture.plan, task_name=request['task_name'])

            def is_gpu_cloth_preview(self, plan):
                return plan.get('preview', False)

            def bind(self, plan, manifest):
                fixture.events.append('bind')
                return plan

            def verify_sources(self, plan):
                fixture.events.append('verify_task_sources')

            def failure_recovery_adapter(self):
                return self

            def atomic_coffee_adapter(self):
                return self

            def admit_launch(self, launch, output):
                fixture.events.append('special_admit')
                return (json.loads(fixture.request.read_text()), json.loads(fixture.manifest.read_text()),
                        dict(fixture.plan, task_name=launch['task_name']))

        self.registry = Registry()
        self.backend = TaskBackend(self.registry,
            verify_scene_sources=lambda: self.events.append('verify_scene_sources'),
            build_short_session=lambda *args, **kwargs: None,
            load_native_bindings=lambda: None)
        self.configure()

    def configure(self, task='test_task', schema='openend.short_task.launch.v1'):
        self.request.write_text(json.dumps({'task_name': task}))
        self.manifest.write_text(json.dumps({'scene': str(self.scene),
            'source_hashes': {str(self.scene): hashlib.sha256(self.scene.read_bytes()).hexdigest()}}))
        self.ready.write_text(json.dumps({'targets_rad': {}}))
        self.launch.write_text(json.dumps({'schema': schema, 'task_name': task,
            'request': str(self.request), 'manifest': str(self.manifest), 'ready_pose': str(self.ready),
            'initial_base_xyyaw': [0., 0., 0.], 'sensor': {}}))

    def with_room_backend(self):
        def admit(launch, output):
            self.events.append('room_admit')
            # Long-task admission can legitimately return a compiled runtime request.
            return ({'compiled_components': []}, json.loads(self.manifest.read_text()), self.plan)
        self.backend = replace(self.backend, admit_room=admit,
                               build_room=lambda *args, **kwargs: None,
                               accept_room=lambda *args, **kwargs: None)
