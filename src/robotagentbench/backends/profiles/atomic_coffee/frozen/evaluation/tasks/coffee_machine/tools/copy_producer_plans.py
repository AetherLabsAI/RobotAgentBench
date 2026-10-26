"""Copy the data producer's executed joint-target plan of every packaged source episode, byte-identical, next to
episode.json as producer_plan.json, with producer_plan.source.json recording where it came from (path, sha256, rows).
This is the only input of tools/convert_trajectory.py; nothing in it is changed (user 2026-09-30: action values are
never modified). CPU only; skips episodes that already have the copy (and verifies it still matches the source)."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

TASK = Path(__file__).resolve().parents[1]
EPISODES = TASK / 'resources/source_episodes'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def copy(folder):
    manifest = json.loads((folder / 'episode.json').read_text())
    source = Path(manifest['trajectory']['file'])
    target = folder / 'producer_plan.json'
    record = folder / 'producer_plan.source.json'
    if target.exists() and record.exists():
        if json.loads(record.read_text())['sha256'] != sha(target):
            raise ValueError('Existing copy does not match its record: ' + str(target))
        return 'kept'
    shutil.copyfile(source, target)
    digest = sha(target)
    if digest != sha(source):
        raise RuntimeError('Copy differs from the source: ' + str(source))
    rows = len(json.loads(target.read_text())['steps'])
    record.write_text(json.dumps(dict(
        source=str(source), sha256=digest, bytes=target.stat().st_size, rows=rows,
        recorded_in_episode_manifest=[p for p in manifest['inputs_sha256'] if p.endswith(('plan.json',))],
        note="Byte-identical copy of the data producer's executed joint-target plan; the only input of tools/convert_trajectory.py"),
        indent=1) + '\n')
    return 'copied'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--api', nargs='+', default=['1-1', '1-4'])
    a = p.parse_args()
    for api in a.api:
        for folder in sorted((EPISODES / api).iterdir()):
            if (folder / 'episode.json').is_file():
                print(api, folder.name, copy(folder))


if __name__ == '__main__':
    main()
