"""Open the data producer's USD layer stacks on 202 without touching them (CPU USD only, read-only on the producer side).

The producer authored its layers inside a container, so they reference each other by container paths
(`/geniesim/main/...`). On 202 the same files live under /home/lingjun/coffee-workspace (and worker copies). This module
writes path-rewritten COPIES of the text layers that need it into a task-owned mirror directory and leaves every
producer file untouched. Binary assets (usdz, textures) are referenced in place by absolute 202 path.

    mirror = Mirror(workbench=<202 copy of coffee_task_workbench_20260922 that produced the episode>, out=<mirror dir>)
    root = mirror.layer('/geniesim/main/demo_sim/coffee_task_workbench_20260922/runs/<run>/initial/initial.usda')
    stage = Usd.Stage.Open(root)
"""
import hashlib
import json
import re
from pathlib import Path

WORKSPACE = Path('/home/lingjun/coffee-workspace')
LAYER_SUFFIXES = ('.usda', '.usd', '.usdc')
TOKEN = re.compile(r'@([^@\n]+)@')


class Mirror:
    def __init__(self, workbench, out):
        self.workbench = Path(workbench).resolve(strict=True)
        self.out = Path(out).resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        # most specific container prefix first
        self.prefixes = (
            ('/geniesim/main/demo_sim/coffee_task_workbench_20260922', self.workbench),
            ('/geniesim/main/demo_sim/coffee_scene_preview_20260922', WORKSPACE / 'coffee_scene_preview_20260922'),
            ('/data/admin/work/coffee-machine-asset-20260922', WORKSPACE / 'coffee-machine-asset-20260922'),
            ('/geniesim/main', WORKSPACE / 'dependencies/geniesim-main'),
        )
        self.done = {}      # host path -> path to use (mirror copy or the host path itself)
        self.record = {}    # host path -> dict(sha256, mirrored, rewritten tokens)

    def host(self, reference, base):
        """202 path of one asset reference written in the layer that lives at `base`."""
        for prefix, target in self.prefixes:
            if reference == prefix or reference.startswith(prefix + '/'):
                return (target / reference[len(prefix):].lstrip('/')).resolve()
        path = Path(reference)
        return (path if path.is_absolute() else base.parent / path).resolve()

    def layer(self, reference, base=None):
        host = self.host(str(reference), base or Path('/'))
        if host in self.done:
            return self.done[host]
        if not host.is_file():
            raise FileNotFoundError(f'{reference} -> {host}')
        self.done[host] = str(host)          # breaks reference cycles
        if host.suffix not in LAYER_SUFFIXES or host.read_bytes()[:8] == b'PXR-USDC':
            self.record[str(host)] = dict(mirrored=False, kind='binary_or_asset')
            return str(host)
        text = host.read_text()
        changed = {}

        def rewrite(match):
            ref = match.group(1)
            if ref.startswith(('http:', 'https:', 'omniverse:')) or '<' in ref:
                return match.group(0)
            target = self.host(ref, host)
            if not target.exists():
                # udim patterns, render-only textures that are absent on 202: keep the absolute 202 path, record it
                new = str(target)
                missing.append(ref)
            elif target.suffix in LAYER_SUFFIXES:
                new = self.layer(ref, host)
            else:
                new = str(target)
            if new != ref:
                changed[ref] = new
            return '@' + new + '@'

        missing = []
        new_text = TOKEN.sub(rewrite, text)
        digest = hashlib.sha256(host.read_bytes()).hexdigest()
        if not changed:
            self.record[str(host)] = dict(mirrored=False, sha256=digest, missing_assets=missing)
            return str(host)
        copy = self.out / (hashlib.sha256(str(host).encode()).hexdigest()[:12] + '__' + host.name)
        copy.write_text(new_text)
        self.done[host] = str(copy)
        self.record[str(host)] = dict(mirrored=True, sha256=digest, copy=str(copy), rewritten=len(changed), missing_assets=missing)
        return str(copy)

    def save_record(self, name='mirror-record.json'):
        (self.out / name).write_text(json.dumps(dict(
            note='Path-rewritten copies of producer text layers; producer files untouched. Keys are 202 paths of the originals.',
            workbench=str(self.workbench), prefixes=[[p, str(t)] for p, t in self.prefixes], layers=self.record), indent=1) + '\n')
