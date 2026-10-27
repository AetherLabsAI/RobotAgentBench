"""Read saved instances without invoking any randomizer or preparation code."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(path):
    path = Path(path).resolve(strict=True)
    value = json.loads(path.read_text())
    if value.get('schema') != 'robotagentbench.composite_instance.v31':
        raise ValueError('Unknown saved instance schema')
    for name, row in value['files'].items():
        source = Path(row['path'])
        if source.resolve() != source or not source.is_file() or sha(source) != row['sha256']:
            raise ValueError('Saved instance changed: ' + name)
    result = dict(value)
    result['reloaded_read_only'] = True
    result['resampled'] = False
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('instance', type=Path)
    args = parser.parse_args(argv)
    value = load(args.instance)
    print(json.dumps({k: value[k] for k in ('task_name','seed','status','launch','reloaded_read_only','resampled')}, indent=2))


if __name__ == '__main__':
    main()
