"""Shared coffee geometry and physics assets; no task motion or dispensing logic."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DIRECTORY=ROOT/'resources/shared_assets/coffee_v1'
SCENE_LAYER=DIRECTORY/'scene_bindings.usda'
def verify_files():
    data=json.loads((DIRECTORY/'manifest.json').read_text())
    for name,digest in data['files'].items():
        if hashlib.sha256((DIRECTORY/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Shared coffee asset changed: '+name)
    return data

def apply(stage):
    data=verify_files()
    session=stage.GetSessionLayer()
    path=str(SCENE_LAYER)
    session.subLayerPaths=[path]+[p for p in session.subLayerPaths if p!=path]
    return dict(revision=data['revision'],activation='inherited from Task75',bean_count=100,task_controller_installed=False)
