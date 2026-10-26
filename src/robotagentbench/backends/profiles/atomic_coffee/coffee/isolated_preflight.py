"""Run the existing CPU USD admission without importing USD into Isaac's parent."""
import json
import os
from pathlib import Path
import subprocess
import sys


def environment(parent, usd, physx):
    result = dict(parent)
    additions = {
        'PYTHONPATH': [str(usd), str(physx)],
        'LD_LIBRARY_PATH': [str(usd / 'bin'), str(physx / 'bin')],
        'PXR_PLUGINPATH_NAME': [str(physx / 'plugins' / name / 'resources') for name in
            ('PhysxSchema', 'PhysxSchemaAddition', 'OmniUsdPhysicsDeformableSchema')],
    }
    for key, paths in additions.items():
        result[key] = os.pathsep.join([*paths, *([parent[key]] if parent.get(key) else [])])
    return result


def installation():
    base = Path(sys.executable).parent.parent / 'lib' / f'python{sys.version_info.major}.{sys.version_info.minor}' / 'site-packages' / 'isaacsim' / 'extscache'
    choices = [list(base.glob(pattern)) for pattern in ('omni.usd.libs-*', 'omni.usd.schema.physx-*')]
    if any(len(paths) != 1 for paths in choices):
        raise RuntimeError('Need one existing bundled USD and PhysX schema installation; no automatic install or version selection')
    return tuple(paths[0] for paths in choices)


def preflight(scene):
    usd, physx = installation()
    result = subprocess.run([sys.executable, '-B', '-m', __package__ + '.preflight_atomic', str(scene)],
        env=environment(os.environ, usd, physx), text=True, capture_output=True, check=True)
    return json.loads(result.stdout)
