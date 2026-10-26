"""Explicit, read-only source/native/asset locations with original file hashes."""
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
from pathlib import PurePosixPath
import re
from types import MappingProxyType

from .source_tree import SourceDependencyError, _relative
from .workspace import _root


@dataclass(frozen=True)
class Resource:
    root: str
    path: str
    sha256: str
    kind: str

    def __post_init__(self):
        _relative(self.path)
        if self.kind not in ('source', 'native', 'asset'):
            raise ValueError('Explicit source, native or asset resource kind required')
        if not isinstance(self.sha256, str) or not re.fullmatch('[0-9a-f]{64}', self.sha256):
            raise ValueError('Original SHA-256 required for every resource')


class RuntimeDependencies:
    """Operator-owned installation layout, never an agent or task-score option.

    Original absolute source names may be mapped individually to reviewed files.
    No wildcard search, automatic fallback, download, symlink creation, manifest
    rewrite or hash refresh is performed. File checks do not prove USD closure.
    """
    def __init__(self, roots, resources, *, legacy_sources=None):
        if not isinstance(roots, Mapping) or not roots or not isinstance(resources, Mapping) or not resources:
            raise ValueError('Nonempty explicit roots and reviewed resources required')
        resolved = {}
        for name, value in roots.items():
            if not isinstance(name, str) or not re.fullmatch('[a-z][a-z0-9_]*', name):
                raise ValueError('Stable installation root names required')
            resolved[name] = _root(value)
        values = {}
        for key, resource in resources.items():
            if not isinstance(key, str) or not re.fullmatch('[a-zA-Z0-9_.-]+', key):
                raise ValueError('Stable resource identifiers required')
            if not isinstance(resource, Resource) or resource.root not in resolved:
                raise ValueError('Every resource must use an explicitly configured root')
            values[key] = resource
        aliases = dict(legacy_sources or {})
        for original, key in aliases.items():
            if (not isinstance(original, str) or not PurePosixPath(original).is_absolute()
                    or PurePosixPath(original).as_posix() != original or '\\' in original or '\x00' in original
                    or '..' in original.split('/') or key not in values or values[key].kind != 'source'):
                raise ValueError('Exact original source names must bind reviewed source files')
        self.roots = MappingProxyType(resolved)
        self.resources = MappingProxyType(values)
        self.legacy_sources = MappingProxyType(aliases)

    @classmethod
    def from_config(cls, config):
        if (not isinstance(config, dict) or set(config) != {'schema', 'roots', 'resources', 'legacy_sources'}
                or config['schema'] != 'robotagentbench.runtime_dependencies.v1'):
            raise ValueError('Explicit runtime dependency configuration required')
        resources = {key: Resource(**value) for key, value in config['resources'].items()}
        return cls(config['roots'], resources, legacy_sources=config['legacy_sources'])

    def file(self, key):
        try:
            resource = self.resources[key]
        except KeyError:
            raise SourceDependencyError('Resource has no reviewed binding') from None
        path = self.roots[resource.root] / resource.path
        if path.resolve() != path or not path.is_file():
            raise SourceDependencyError('Required resource is missing or uses a symbolic link: ' + key)
        with path.open('rb') as stream:
            header = stream.read(1024)
            stream.seek(0)
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != resource.sha256:
            raise SourceDependencyError('Required resource differs from its original hash: ' + key)
        if header.startswith((b'version https://git-lfs.github.com/spec/v1\n',
                              b'version https://git-lfs.github.com/spec/v1\r\n')):
            raise SourceDependencyError('Required resource is an unresolved Git LFS pointer: ' + key)
        return path

    def directory(self, root, relative=''):
        if relative:
            _relative(relative)
        if root not in self.roots:
            raise SourceDependencyError('Directory has no configured installation root')
        path = self.roots[root] / relative
        if path.resolve() != path or not path.is_dir():
            raise SourceDependencyError('Required installation directory is missing or uses a symbolic link')
        return path

    def verify_original_sources(self, original_hashes):
        if not isinstance(original_hashes, Mapping) or not original_hashes:
            raise ValueError('Nonempty original source manifest required')
        for original, expected in original_hashes.items():
            key = self.legacy_sources.get(original)
            if key is None or self.resources[key].sha256 != expected:
                raise SourceDependencyError('Original source identity has no exact reviewed binding')
            self.file(key)
        return dict(verified=True, file_count=len(original_hashes))

    def verify(self):
        for key in self.resources:
            self.file(key)
        return dict(verified=True, file_count=len(self.resources),
                    kinds=dict(sorted(Counter(item.kind for item in self.resources.values()).items())))
