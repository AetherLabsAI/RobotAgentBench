"""Operator-owned native input, output, and cooperative GPU-lock boundaries."""
from dataclasses import dataclass
from pathlib import Path


def _root(value):
    path = Path(value)
    if (not path.is_absolute() or path.resolve() != path or not path.is_dir()
            or path == Path(path.anchor) or path == Path.home()):
        raise ValueError('An existing canonical dedicated directory is required')
    return path


@dataclass(frozen=True)
class NativeWorkspace:
    input_roots: tuple[Path, ...]
    output_root: Path
    lock_root: Path

    def __post_init__(self):
        if not self.input_roots:
            raise ValueError('Explicit trusted native input roots are required')
        object.__setattr__(self, 'input_roots', tuple(_root(path) for path in self.input_roots))
        object.__setattr__(self, 'output_root', _root(self.output_root))
        object.__setattr__(self, 'lock_root', _root(self.lock_root))

    def input(self, value):
        path = Path(value)
        if (not path.is_absolute() or path.resolve() != path or not path.is_file()
                or not any(path != root and path.is_relative_to(root) for root in self.input_roots)):
            raise ValueError('Native input must be a real file below a configured input root')
        return path

    def output(self, value):
        path = Path(value)
        if (not path.is_absolute() or path.resolve() != path
                or path == self.output_root or not path.is_relative_to(self.output_root)
                or path.exists() or path.is_symlink()):
            raise ValueError('Native output must be fresh and below the configured output root')
        return path

    def create_output(self, value):
        path = self.output(value)
        path.mkdir(parents=True, exist_ok=False)
        return path
