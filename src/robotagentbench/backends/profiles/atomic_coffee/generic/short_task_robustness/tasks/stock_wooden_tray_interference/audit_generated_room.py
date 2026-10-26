"""Read-only CPU audit of full-room generation and independent task binding."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from short_tasks.scene import verify_manifest
from short_tasks.tasks.stock_wooden_tray_interference.adapter import bind
from short_tasks.tasks.stock_wooden_tray_interference.endpoint_lanes import validate_endpoint_lanes
from short_tasks.tasks.stock_wooden_tray_interference.distance_bands import require_manifest_band


def audit(folder):
    folder = Path(folder)
    manifest = json.loads((folder / 'composition.json').read_text())
    plan = json.loads((folder / 'task_plan.json').read_text())
    receipt = json.loads((folder / 'generation_receipt.json').read_text())
    verify_manifest(manifest)
    assert hashlib.sha256(Path(manifest['scene']).read_bytes()).hexdigest() == manifest['scene_sha256']
    assert manifest['scene_generated'] and receipt['reservation_frozen']
    require_manifest_band(plan, manifest)
    reservation, = manifest['environment_reservations']
    validate_endpoint_lanes(reservation)
    product, = plan['target_categories']
    targets = [row for row in manifest['objects'] if row['id'] == product]
    assert len(targets) == 1
    assert not manifest['interference_background_exclusion']['excluded_paths']
    assert manifest['shelf_initialization']['inventory_check']['inventory_matches_plan']
    binding = bind(plan, manifest)
    center = lambda box: [(box[0][i]+box[1][i])/2 for i in range(2)]
    distance = 100 * math.dist(center(reservation['initial_target_bounds']), center(reservation['reserved_bounds']))
    return dict(passed=True, layout_id=manifest['layout_id'], seed=manifest['seed'],
                distance_cm=distance, distance_selection=reservation['relocation_distance'],
                target_count=len(targets), grasp_trapezoids=len(reservation['endpoint_grasp_lanes']['lanes']),
                inventory_preserved=True, task_adapter_bound=binding is not None,
                source_hashes_verified=len(manifest['source_hashes']),
                scene_hash_verified=True, shared_sources=receipt['shared_integrity'],
                native_physics_validated=False, folder=str(folder))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path, nargs='+')
    print(json.dumps([audit(p) for p in parser.parse_args().folder], indent=2))
