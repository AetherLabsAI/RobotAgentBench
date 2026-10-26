"""Read-only native-interpreter and installed-package preflight; no Isaac import."""
from dataclasses import dataclass
import json
from pathlib import Path
import re
import subprocess
from types import MappingProxyType

from .source_tree import SourceDependencyError, _relative

PROBE = '''import importlib.metadata, json, platform, sys
wanted, paths = json.loads(sys.argv[1]), json.loads(sys.argv[2])
versions = {}
for distribution in importlib.metadata.distributions(path=paths):
    name = distribution.metadata.get("Name", "").lower().replace("_", "-").replace(".", "-")
    if name not in wanted:
        continue
    if name in versions:
        raise RuntimeError("Ambiguous native package metadata")
    versions[name] = distribution.version
forbidden = {"isaacsim", "omni", "pxr", "torch", "arena_api", "short_tasks", "openend"}
loaded = sorted(name for name in sys.modules if name.split(".")[0] in forbidden)
print(json.dumps(dict(python=list(sys.version_info[:2]), implementation=sys.implementation.name,
    cache_tag=sys.implementation.cache_tag, platform=sys.platform, machine=platform.machine(),
    packages=versions, native_modules_loaded=loaded)))
'''


@dataclass(frozen=True)
class Directory:
    root: str
    path: str = ''

    def __post_init__(self):
        if not isinstance(self.path, str):
            raise ValueError('Explicit relative environment directory required')
        if self.path:
            _relative(self.path)


class NativeEnvironment:
    """Explicit launcher, ABI and directory bindings from trusted operator code.

    A venv launcher may be a leaf symlink only when its exact canonical binary
    is separately hash-pinned. No native package or simulator is imported by the
    isolated metadata probe. Installed packages and assets are never downloaded.
    """
    def __init__(self, dependencies, *, launcher, binary, python_version, platform, machine,
                 packages, metadata_paths, python_paths=(), library_paths=(), plugin_paths=(),
                 asset_paths=(), material_paths=(), required_resources=()):
        self.dependencies = dependencies
        self.launcher, self.binary = Path(launcher), binary
        if (not isinstance(python_version, tuple) or len(python_version) != 2
                or any(type(value) is not int or value < 0 for value in python_version)
                or not isinstance(platform, str) or not platform or not isinstance(machine, str) or not machine):
            raise ValueError('Explicit native Python ABI, platform and architecture required')
        if (not self.launcher.is_absolute() or self.launcher.parent.resolve() != self.launcher.parent
                or not any(self.launcher.is_relative_to(root) for root in dependencies.roots.values())):
            raise ValueError('Interpreter launcher must be inside a configured canonical root')
        if binary not in dependencies.resources or dependencies.resources[binary].kind != 'native':
            raise ValueError('Native interpreter binary needs a reviewed hash binding')
        self.python_version, self.platform, self.machine = python_version, platform, machine
        self.packages = {}
        for name, version in packages.items():
            if not isinstance(name, str) or not re.fullmatch('[a-z0-9]+(?:-[a-z0-9]+)*', name):
                raise ValueError('Canonical native distribution names required')
            if not isinstance(version, str) or not version:
                raise ValueError('Exact reviewed native package versions required')
            self.packages[name] = version
        self.packages = MappingProxyType(self.packages)
        for field, values in (('metadata_paths', metadata_paths), ('python_paths', python_paths),
                              ('library_paths', library_paths), ('plugin_paths', plugin_paths),
                              ('asset_paths', asset_paths), ('material_paths', material_paths)):
            if not isinstance(values, (tuple, list)) or any(not isinstance(item, Directory) for item in values):
                raise ValueError('Explicit environment directory bindings required')
            setattr(self, field, tuple(values))
        self.required_resources = tuple(required_resources)
        if any(key not in dependencies.resources for key in self.required_resources):
            raise ValueError('Native prerequisite has no reviewed resource binding')

    @classmethod
    def from_config(cls, dependencies, config):
        """Load an explicit operator profile without importing native packages."""
        directories = ('metadata_paths', 'python_paths', 'library_paths', 'plugin_paths',
                       'asset_paths', 'material_paths')
        fields = {'schema', 'launcher', 'binary', 'python_version', 'platform', 'machine',
                  'packages', 'required_resources', *directories}
        if (not isinstance(config, dict) or set(config) != fields
                or config['schema'] != 'robotagentbench.native_environment.v1'
                or not isinstance(config['python_version'], list)
                or not isinstance(config['packages'], dict)
                or not isinstance(config['required_resources'], list)
                or any(not isinstance(key, str) or not key for key in config['required_resources'])
                or len(set(config['required_resources'])) != len(config['required_resources'])):
            raise ValueError('Explicit native-environment configuration required')
        values = {key: value for key, value in config.items() if key != 'schema'}
        values['python_version'] = tuple(config['python_version'])
        for field in directories:
            entries = config[field]
            if (not isinstance(entries, list) or any(not isinstance(item, dict)
                    or set(item) != {'root', 'path'} for item in entries)):
                raise ValueError('Explicit root/path environment directory lists required')
            values[field] = tuple(Directory(**item) for item in entries)
        return cls(dependencies, **values)

    def _paths(self, entries):
        paths = [str(self.dependencies.directory(item.root, item.path)) for item in entries]
        if any(':' in path or '\n' in path for path in paths) or len(paths) != len(set(paths)):
            raise ValueError('Distinct canonical environment directories required')
        return paths

    def verify(self):
        binary = self.dependencies.file(self.binary)
        if not self.launcher.is_file() or self.launcher.resolve() != binary:
            raise SourceDependencyError('Native launcher no longer resolves to its pinned interpreter')
        for key in self.required_resources:
            self.dependencies.file(key)
        for values in (self.metadata_paths, self.python_paths, self.library_paths, self.plugin_paths,
                       self.asset_paths, self.material_paths):
            self._paths(values)

    def environment(self, base):
        self.verify()
        env = dict(base)
        for key in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONSTARTUP', 'PYTHONUSERBASE', 'PYTHONPLATLIBDIR',
                    'LD_PRELOAD', 'LD_AUDIT', 'DYLD_INSERT_LIBRARIES', 'DYLD_LIBRARY_PATH',
                    'DYLD_FALLBACK_LIBRARY_PATH', 'DYLD_FRAMEWORK_PATH', 'DYLD_FALLBACK_FRAMEWORK_PATH',
                    'LD_LIBRARY_PATH', 'PXR_PLUGINPATH_NAME', 'PXR_AR_DEFAULT_SEARCH_PATH', 'MDL_SYSTEM_PATH'):
            env.pop(key, None)
        env.update(PYTHONPATH=':'.join(self._paths(self.python_paths)),
                   PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1')
        for key, values in (('LD_LIBRARY_PATH', self.library_paths), ('PXR_PLUGINPATH_NAME', self.plugin_paths),
                            ('PXR_AR_DEFAULT_SEARCH_PATH', self.asset_paths), ('MDL_SYSTEM_PATH', self.material_paths)):
            if values:
                env[key] = ':'.join(self._paths(values))
        return env

    def probe(self, *, timeout=30):
        if type(timeout) is not int or not 1 <= timeout <= 60:
            raise ValueError('A bounded native metadata-probe timeout is required')
        # -S avoids sitecustomize and project imports; -I ignores ambient Python
        # paths and the current directory. Distribution metadata is read directly.
        try:
            result = subprocess.run([str(self.launcher), '-I', '-S', '-c', PROBE,
                json.dumps(list(self.packages)), json.dumps(self._paths(self.metadata_paths))],
                env=self.environment({}), stdin=subprocess.DEVNULL, text=True,
                capture_output=True, timeout=timeout, check=False)
        except (OSError, subprocess.TimeoutExpired):
            raise SourceDependencyError('Native metadata probe could not finish within its execution boundary') from None
        if result.returncode:
            raise SourceDependencyError('Isolated native metadata probe failed (exit code %d)' % result.returncode)
        try:
            report = json.loads(result.stdout)
        except (ValueError, TypeError):
            raise SourceDependencyError('Native interpreter returned invalid metadata') from None
        if (report.get('python') != list(self.python_version) or report.get('implementation') != 'cpython'
                or report.get('cache_tag') != 'cpython-' + ''.join(map(str, self.python_version))
                or report.get('platform') != self.platform or report.get('machine') != self.machine
                or report.get('packages') != dict(self.packages) or report.get('native_modules_loaded') != []):
            raise SourceDependencyError('Native interpreter ABI or package versions do not match the reviewed profile')
        self.verify()
        return report
