"""CPU-only composition regression; never render or publish current_room."""
import json
import subprocess
import uuid
from pathlib import Path
from generate_room import HERE, choose_plan, cpu_command
from room_request import resolve

def main():
    folder = HERE / 'generated_rooms' / ('cloth_books_check_' + uuid.uuid4().hex[:12])
    folder.mkdir(parents=True, exist_ok=False)
    request = dict(schema='openend.scene_requirements.request.v2', seed=42, layout_id=48,
        tasks=[dict(id='cloth_clear', type='cloth_clear', items=[dict(kind='book_red', count=2)]),
               dict(id='book_transfer', type='book_transfer', source='sofa', count=1)], extras=[])
    (folder / 'request.json').write_text(json.dumps(request))
    planned = choose_plan(resolve(request), folder)
    (folder / 'input.json').write_text(json.dumps(planned))
    result = subprocess.run(cpu_command(str(folder)), capture_output=True, text=True, timeout=300)
    (folder / 'build.log').write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(str(folder) + '\n' + (result.stderr or result.stdout)[-3000:])
    manifest = json.loads((folder / 'composition.json').read_text())
    from openend.room import prepare_registration
    runtime = json.loads((folder / 'runtime_request.json').read_text())
    registration = prepare_registration(runtime, manifest)
    group, = registration['cloth_clear_groups']
    assert group['target_count'] == 2 and group['maximum_score'] == 6
    ordinary = registration['resolved_objects']['book_transfer']
    assert ordinary not in group['target_paths']
    assert manifest['source_unchanged']
    receipt = dict(folder=str(folder), target_count=2, maximum_score=6,
        ordinary_book=ordinary, cloth_books=group['target_paths'],
        source_unchanged=True, native_physics_validated=False, published=False)
    (folder / 'verification.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt))

if __name__ == '__main__':
    main()
