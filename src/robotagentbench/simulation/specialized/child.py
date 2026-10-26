"""Fixed fresh child: verify namespace, then execute the unchanged source owner."""
import importlib
import os
from pathlib import Path
import sys

from . import config


def propagate_import_roots(verified):
    """Preserve trusted roots for original execv/-m grandchildren as well.

    Changing this process's sys.path alone is insufficient: Recovery's original
    execv inherits os.environ, not sys.path. Atomic's own execve/environment
    retains its original fixed Frozen-first recipe; this does not replace it.
    """
    environment = verified['environment']
    roots = verified['profile']['bindings']['import_roots']
    paths = list(dict.fromkeys(roots + environment._paths(environment.python_paths)))
    if any(':' in path or '\n' in path for path in paths):
        raise ValueError('Explicit canonical import paths required')
    os.environ['PYTHONPATH'] = ':'.join(paths)
    sys.path[:0] = paths
    return paths


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 6:
        raise ValueError('The trusted specialized handoff has exactly six arguments')
    operator_path, operator_sha, family, task, launch, output = args
    if config.sha(operator_path) != operator_sha:
        raise ValueError('Operator changed across the fresh-interpreter boundary')
    operator = config.read_operator(operator_path)
    verified = config.verify(operator, family, task, mounted=True)
    if verified['launch_path'] != Path(launch):
        raise ValueError('Task launch changed at the source boundary')
    workspace = verified['workspace']
    out = workspace.output(output)
    if out.parent != workspace.output_root:
        raise ValueError('Only the exact original-owner output scope is allowed')
    from robotagentbench.simulation.source_tree import NAMESPACES
    if any(name.split('.')[0] in NAMESPACES for name in sys.modules):
        raise ValueError('The original task source must start in a fresh interpreter')
    # -I/-S intentionally excluded ambient import paths. Reintroduce only the
    # operator-reviewed native directories and these fixed original namespaces.
    propagate_import_roots(verified)
    entry = verified['profile']['bindings']['entry_file']
    expected = verified['profile']['entry_sha256']
    if config.sha(entry) != expected:
        raise ValueError('Fixed original entry identity changed')
    adapter = importlib.import_module(config.MODULES[family])
    if Path(adapter.__file__) != Path(entry) or config.sha(adapter.__file__) != expected:
        raise ValueError('Fixed original adapter was shadowed by another package')
    sys.argv = [config.MODULES[family], '--launch', launch, '--output', output,
                '--gpu', str(operator['gpu'])]
    # AtomicCoffee and Recovery have different original inner dispatchers.
    # Do not replace them with a generic owner or copy their parameter logic.
    adapter.exec_native()
    raise RuntimeError('Original task-specific native interpreter returned')


if __name__ == '__main__':
    main()
