"""Explicit historical recordings authorized for reuse on 2026-09-17."""
import json
from pathlib import Path

W = Path(__file__).resolve().parent

def imported(spec):
    provenance = spec.get('legacy_import')
    if not provenance:
        return False
    manifest = json.loads((W / 'LEGACY_IMPORTS.json').read_text())
    entry = next((r for r in manifest['episodes'] if r['episode_id'] == spec['episode_id']), None)
    return bool(entry and entry['source_trial'] == provenance['source_trial']
                and entry['slot'] == spec['bulk_slot']
                and entry['frozen_scene_sha256'] == spec['frozen_scene_sha256'])

def counts_allowed(spec):
    if imported(spec):
        return 3 <= spec['target_layer_objects'] <= 12 and 2 <= spec['other_layer_objects'] <= 7
    return 4 <= spec['target_layer_objects'] <= 10 and 4 <= spec['other_layer_objects'] <= 10
