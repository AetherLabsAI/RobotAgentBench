"""Outing-only dependency closure, verified against original training audit."""
from pathlib import Path
import json,sys

def activate(source_hashes,sha):
    here=Path(__file__).resolve().parent
    manifest=here/'geometry_dependencies.json'
    record=json.loads(manifest.read_text())
    for item in record['files']:
        path=here/'original_geometry_helpers'/item['file']
        if sha(path)!=item['sha256']:raise ValueError('Changed original geometry helper '+str(path))
        source_hashes[str(path)]=item['sha256']
    source_hashes[str(manifest)]=sha(manifest)
    source_hashes[str(Path(__file__))]=sha(__file__)
    sys.path.insert(0,str(here/'original_geometry_helpers'))
    # The full-source deployment already contains its own shelf sampler copy.
    source=here.parents[5]/'openend/tasks/scene_requirements/shelf_sources'
    for path in source.glob('*.py'):source_hashes[str(path)]=sha(path)
    sys.path.append(str(source))
