"""Freeze the source tree the Task 1 (coffee machine) batch runs against, and record per-file SHA-256.

Lean snapshot: the host has little free disk, so only what the coffee machine job path loads is copied:
the public service and worker (arena_api, arena_randomization, configs, tools, vendor, resources), the public house
scene (assets/house_scene_l1), the evaluation package with this family complete and every other family reduced to
its Python / JSON files (the public registry imports every family's registration), minus job output, plus the
pre-built appearance environments runs/coffee_machine_environment/ (built in the live Git tree, see tools/build_environments.py).
Every copied file is hashed. Jobs run from the snapshot, never from the live working tree, so an edit on 202 during
the batch cannot change a running job. `artifacts/` of this family inside the snapshot is a link to the live
artifacts directory, so results land where the review page reads them.
"""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
TASK = Path(__file__).resolve().parents[1]
FAMILY = 'evaluation/tasks/coffee_machine'
RULES = [
    '- .git/', '- __pycache__/', '- artifacts/',
    '+ /runs/', '+ /runs/coffee_machine_environment/', '+ /runs/coffee_machine_environment/**', '- /runs/**', '- runs/', '- container/', '- experiments/', '- agent_probe_v1/', '- *.mp4',
    '- /docs/', '- /assets/house_scene_v1/', '- /assets/source_release_*/', '- /assets/source_appearance_*/',
    f'- /{FAMILY}/preview/', f'- /{FAMILY}/references/source_sim_videos/', f'- /{FAMILY}/references/l2_real_batch/',
    f'+ /{FAMILY}/***',
    '- /evaluation/tasks/*/preview/', '- /evaluation/tasks/*/references/',
    '+ /evaluation/tasks/**/', '+ /evaluation/tasks/**.py', '+ /evaluation/tasks/**.json', '- /evaluation/tasks/**',
]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--destination', type=Path, required=True)
    p.add_argument('--manifest', type=Path, default=TASK / 'SNAPSHOT-HASHES.json')
    a = p.parse_args()
    dest = a.destination.resolve()
    if dest == ROOT or dest.is_relative_to(ROOT) or ROOT.is_relative_to(dest):
        raise SystemExit('Snapshot must live outside the working tree')
    if dest.exists() and any(dest.iterdir()):
        raise SystemExit('Snapshot destination must be new or empty')
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(['rsync', '-a', '--prune-empty-dirs', *[f'--filter={r}' for r in RULES], str(ROOT) + '/', str(dest) + '/'], check=True)
    # this family needs selection.json (episode order) although the media references are not copied
    selection = Path(FAMILY) / 'references/source_sim_videos/selection.json'
    (dest / selection).parent.mkdir(parents=True, exist_ok=True)
    (dest / selection).write_bytes((ROOT / selection).read_bytes())
    link = dest / FAMILY / 'artifacts'
    link.symlink_to(ROOT / FAMILY / 'artifacts', target_is_directory=True)
    head = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
    hashes, size = {}, 0
    for f in sorted(dest.rglob('*')):
        if f.is_symlink() or not f.is_file():
            continue
        rel = f.relative_to(dest)
        h = hashlib.sha256()
        with f.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b''):
                h.update(chunk)
        hashes[str(rel)] = h.hexdigest()
        size += f.stat().st_size
    manifest = dict(schema='arena.coffee_machine.frozen_source.v1', created=time.strftime('%Y-%m-%d %H:%M:%S'), snapshot=str(dest),
                    source_tree=str(ROOT), head=head, head_note='the live tree has uncommitted work of several families; the per-file hashes are the fingerprint',
                    filter_rules=RULES, artifacts_link=str(ROOT / FAMILY / 'artifacts'), file_count=len(hashes), bytes=size, files=hashes)
    for out in (a.manifest, dest / FAMILY / 'SNAPSHOT-HASHES.json'):
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(manifest, indent=1) + '\n')
    print(json.dumps(dict(head=head, files=len(hashes), megabytes=round(size / 1e6, 1), destination=str(dest))))


if __name__ == '__main__':
    main()
