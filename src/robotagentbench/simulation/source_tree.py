"""Hash-pinned original task imports in an isolated native worker process."""
from contextlib import contextmanager
import hashlib
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
from pathlib import Path, PurePosixPath
import re
import sys
from types import MappingProxyType

NAMESPACES = frozenset({'short_tasks', 'openend', 'arena_api', 'arena_randomization',
    'evaluation', 'layout_task_activation_v1', 'short_task_ContextReasoning', 'short_task_robustness'})


class SourceDependencyError(RuntimeError):
    """Original runtime source or an explicitly required dependency is missing."""


def _relative(value):
    if (not isinstance(value, str) or not value or '\\' in value or '\x00' in value
            or PurePosixPath(value).is_absolute() or PurePosixPath(value).as_posix() != value
            or any(part in ('.', '..') for part in value.split('/'))):
        raise ValueError('Canonical relative dependency paths are required')
    return value


class _PythonLoader(importlib.abc.Loader):
    def __init__(self, tree, relative):
        self.tree, self.relative = tree, relative

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        raw = self.tree.read_verified(self.relative)
        # Never execute an unverified .pyc or write caches into original sources.
        exec(compile(raw, str(self.tree.root / self.relative), 'exec'), module.__dict__)


class _ExtensionLoader(importlib.machinery.ExtensionFileLoader):
    def __init__(self, tree, fullname, relative):
        self.tree, self.relative = tree, relative
        super().__init__(fullname, str(tree.root / relative))

    def create_module(self, spec):
        self.tree.read_verified(self.relative)
        return super().create_module(spec)

    def exec_module(self, module):
        self.tree.read_verified(self.relative)
        return super().exec_module(module)


class _SourceFinder(importlib.abc.MetaPathFinder):
    def __init__(self, tree):
        self.tree = tree

    def find_spec(self, fullname, path=None, target=None):
        prefix = fullname.split('.')[0]
        if prefix not in self.tree.namespace_paths:
            return None
        relative, package = self.tree.module_file(fullname)
        if relative is None:
            folder = self.tree.module_path(fullname)
            if any(item.startswith(folder + '/') for item in self.tree.pins):
                # A real __init__ must not be silently replaced by a namespace.
                if (self.tree.root / folder / '__init__.py').exists():
                    raise ModuleNotFoundError('Unpinned task package initializer', name=fullname)
                spec = importlib.machinery.ModuleSpec(fullname, None, is_package=True)
                spec.submodule_search_locations = [str(self.tree.root / folder)]
                return spec
            raise ModuleNotFoundError('Task dependency is absent from the pinned source selection', name=fullname)
        self.tree.read_verified(relative)
        if relative.endswith('.py'):
            loader = _PythonLoader(self.tree, relative)
        else:
            if not self.tree.native_extensions_enabled:
                raise SourceDependencyError('Native extensions require explicit native dependency admission')
            loader = _ExtensionLoader(self.tree, fullname, relative)
        return importlib.util.spec_from_file_location(fullname, self.tree.root / relative,
            loader=loader, submodule_search_locations=[str((self.tree.root / relative).parent)] if package else None)


class NativeSourceTree:
    """Trusted code-side source selection, not an agent-selectable plugin loader.

    Hash pins must come from a separately reviewed receipt. This class never
    generates or refreshes pins. Original task-specific source/asset checks still
    execute. Native dependencies and code-tool resources remain separate groups.
    """
    def __init__(self, root, pins, *, groups=None, fcl_namespace=None, namespace_paths=None):
        self.root = Path(root)
        if (not self.root.is_absolute() or self.root.resolve() != self.root or not self.root.is_dir()
                or self.root == Path(self.root.anchor) or self.root == Path.home()):
            raise ValueError('An existing canonical dedicated runtime source root is required')
        if not isinstance(pins, dict) or not pins:
            raise ValueError('Nonempty reviewed dependency pins are required')
        self.pins = {}
        for relative, digest in pins.items():
            _relative(relative)
            if not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest):
                raise ValueError('Every dependency needs its original SHA-256 pin')
            self.pins[relative] = digest
        self.groups = {}
        for name, values in (groups or {}).items():
            if name not in ('native', 'coding') or not isinstance(values, (list, tuple)) or not values:
                raise ValueError('Explicit nonempty native/coding dependency groups are required')
            if len(set(values)) != len(values) or any(value not in self.pins for value in values):
                raise ValueError('Dependency group files must have reviewed pins')
            self.groups[name] = tuple(values)
        self.namespace_paths = {name: name for name in NAMESPACES}
        for name, relative in (namespace_paths or {}).items():
            if (not isinstance(name, str) or name.split('.')[0] not in NAMESPACES
                    or any(not part.isidentifier() for part in name.split('.'))):
                raise ValueError('Only original task-runtime namespace prefixes may be relocated')
            self.namespace_paths[name] = _relative(relative)
        if fcl_namespace is not None:
            self.namespace_paths['fcl'] = _relative(fcl_namespace)
        self.pins = MappingProxyType(self.pins)
        self.groups = MappingProxyType(self.groups)
        self.namespace_paths = MappingProxyType(self.namespace_paths)
        self.native_extensions_enabled = False
        self.active = False
        self.finder = _SourceFinder(self)

    def read_verified(self, relative):
        if relative not in self.pins:
            raise SourceDependencyError('Required dependency has no reviewed pin')
        path = self.root / relative
        if path.resolve() != path or not path.is_file():
            raise SourceDependencyError('Pinned dependency is missing or uses a symbolic link')
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != self.pins[relative]:
            raise SourceDependencyError('Pinned dependency changed; no source refresh is allowed')
        return raw

    def verify(self):
        for relative in self.pins:
            self.read_verified(relative)

    def require_group(self, name):
        if name not in self.groups:
            raise SourceDependencyError('Original ' + name + ' dependency group is not bound')
        for relative in self.groups[name]:
            self.read_verified(relative)

    def module_path(self, fullname):
        parts = fullname.split('.')
        if any(not part.isidentifier() for part in parts) or parts[0] not in self.namespace_paths:
            raise SourceDependencyError('Module is outside the original runtime namespaces')
        prefix = max((name for name in self.namespace_paths
                      if fullname == name or fullname.startswith(name + '.')), key=len)
        return '/'.join([self.namespace_paths[prefix], *parts[len(prefix.split('.')):]])

    def module_file(self, fullname):
        base = self.module_path(fullname)
        candidates = [(base + '/__init__.py', True), (base + '.py', False)]
        candidates.extend((base + suffix, False) for suffix in importlib.machinery.EXTENSION_SUFFIXES)
        found = [(relative, package) for relative, package in candidates if relative in self.pins]
        if len(found) > 1:
            raise SourceDependencyError('Ambiguous pinned module implementation')
        return found[0] if found else (None, False)

    def import_module(self, fullname):
        if not self.active:
            raise SourceDependencyError('Original source binding must remain active for the worker lifetime')
        self.module_path(fullname)
        module = importlib.import_module(fullname)
        relative, _ = self.module_file(fullname)
        if relative is not None:
            if (Path(getattr(module, '__file__', '')).resolve() != self.root / relative
                    or not isinstance(getattr(module, '__loader__', None), (_PythonLoader, _ExtensionLoader))
                    or module.__loader__.tree is not self):
                raise SourceDependencyError('Original task module was replaced by an unverified import')
            self.read_verified(relative)
        return module

    @contextmanager
    def activated(self):
        if self.active:
            raise SourceDependencyError('A runtime source binding cannot be entered twice')
        if any(isinstance(finder, _SourceFinder) for finder in sys.meta_path):
            raise SourceDependencyError('Another original source binding is active; use a fresh worker')
        if any(name.split('.')[0] in self.namespace_paths for name in sys.modules):
            raise SourceDependencyError('Original runtime namespace is already loaded; use a fresh worker')
        self.verify()
        before_path = list(sys.path)
        sys.meta_path.insert(0, self.finder)
        self.active = True
        try:
            yield self
        finally:
            self.active = False
            self.native_extensions_enabled = False
            if self.finder in sys.meta_path:
                sys.meta_path.remove(self.finder)
            # Supported only in a dedicated worker. Do not reuse task objects
            # after leaving this source binding or share it with another thread.
            for name in list(sys.modules):
                if name.split('.')[0] in self.namespace_paths:
                    del sys.modules[name]
            sys.path[:] = before_path
