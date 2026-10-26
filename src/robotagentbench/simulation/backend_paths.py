"""Per-task source, asset and native bindings for the original owner factories."""
from dataclasses import dataclass
from types import MappingProxyType

from .source_tree import SourceDependencyError
from robotagentbench.tasks.catalog import load_catalog

ROOM_TASKS = frozenset(('openend', 'task_discovery', *(f'long_task_{i}' for i in range(15, 23))))
TASKS = frozenset(load_catalog('standalone').identifiers) | ROOM_TASKS


@dataclass(frozen=True)
class ResourceGroups:
    source: tuple[str, ...] = ()
    assets: tuple[str, ...] = ()
    native: tuple[str, ...] = ()
    coding: tuple[str, ...] = ()

    def __post_init__(self):
        for name in ('source', 'assets', 'native', 'coding'):
            values = getattr(self, name)
            if (not isinstance(values, tuple) or len(set(values)) != len(values)
                    or any(not isinstance(value, str) or not value for value in values)):
                raise ValueError('Distinct explicit resource identifiers required')


class BackendPaths:
    """A complete task routing table, not a new task or scoring registry.

    The original owners still resolve requests, bind instances, check their
    source records and score physical evidence. Missing asset/native profiles
    may not be treated as successful native admission.
    """
    def __init__(self, dependencies, *, common, tasks):
        if not isinstance(common, ResourceGroups) or not common.source:
            raise ValueError('Reviewed common source resources required')
        if set(tasks) != TASKS or any(not isinstance(value, ResourceGroups) for value in tasks.values()):
            raise ValueError('Explicit bindings for every standalone and room task required')
        for group in (common, *tasks.values()):
            for field, kinds in (('source', {'source'}), ('assets', {'asset'}),
                                 ('native', {'native'}), ('coding', {'source', 'native'})):
                for key in getattr(group, field):
                    if key not in dependencies.resources or dependencies.resources[key].kind not in kinds:
                        raise ValueError('Task resource has no matching reviewed dependency kind')
        self.dependencies, self.common = dependencies, common
        self.tasks = MappingProxyType(dict(tasks))

    @classmethod
    def from_config(cls, dependencies, config):
        """Load operator-owned file groups, never derive them from task requests."""
        if (not isinstance(config, dict) or set(config) != {'schema', 'common', 'tasks'}
                or config['schema'] != 'robotagentbench.backend_paths.v1'
                or not isinstance(config['tasks'], dict)):
            raise ValueError('Explicit backend-path configuration required')

        def group(value):
            fields = {'source', 'assets', 'native', 'coding'}
            if (not isinstance(value, dict) or set(value) != fields
                    or any(not isinstance(value[field], list) for field in fields)):
                raise ValueError('Explicit source, asset, native and coding lists required')
            return ResourceGroups(**{field: tuple(value[field]) for field in fields})

        return cls(dependencies, common=group(config['common']),
                   tasks={name: group(value) for name, value in config['tasks'].items()})

    def validate_source_tree(self, sources):
        files = {(self.dependencies.roots[value.root] / value.path, value.sha256)
                 for value in self.dependencies.resources.values()}
        if any((sources.root / path, digest) not in files for path, digest in sources.pins.items()):
            raise SourceDependencyError('Original import tree differs from the configured dependency files')
        self.verify_common()

    def verify_common(self):
        for key in self.common.source:
            self.dependencies.file(key)

    def verify(self, task_name, phase='source'):
        if task_name not in self.tasks or phase not in ('source', 'assets', 'native', 'coding'):
            raise SourceDependencyError('Task or dependency phase has no explicit path binding')
        group = self.tasks[task_name]
        fields = ['source']
        if phase in ('assets', 'native', 'coding'):
            if not self.common.assets and not group.assets:
                raise SourceDependencyError('Original task asset profile is not bound')
            fields.append('assets')
        if phase in ('native', 'coding'):
            if not self.common.native and not group.native:
                raise SourceDependencyError('Original native environment profile is not bound')
            fields.append('native')
        if phase == 'coding':
            if not self.common.coding and not group.coding:
                raise SourceDependencyError('Original code-tool resource profile is not bound')
            fields.append('coding')
        keys = dict.fromkeys(key for field in fields
                             for values in (getattr(self.common, field), getattr(group, field)) for key in values)
        for key in keys:
            self.dependencies.file(key)
        return dict(task_name=task_name, phase=phase, verified=True, file_count=len(keys))
