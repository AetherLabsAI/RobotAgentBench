"""Task-local reviewed source pins, in addition to shared scene verification."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAMES = {'build_shelf.py', 'build_selected_room.py'}


def verify_reviewed_generators(scene_root, record=None):
    if record is None:
        record = json.loads((HERE / 'generator_sources.json').read_text())
    if (record.get('schema') != 'shelf_interference.reviewed_generators.v1'
            or set(record.get('sources', {})) != NAMES):
        raise ValueError('Invalid task-owned generator review record')
    checked = {}
    for name, expected in record['sources'].items():
        path = Path(scene_root) / name
        if not path.is_file():
            raise ValueError('Reviewed room generator missing: ' + name)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError('Room generator changed since task review: ' + name)
        checked[name] = actual
    return dict(verified=True, sources=checked, review_date=record.get('review_date'))
