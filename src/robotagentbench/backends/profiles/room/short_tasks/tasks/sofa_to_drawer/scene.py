"""Only the existing drawer mechanism, installed in the anonymous scene layer."""
from copy import deepcopy
import hashlib
from pathlib import Path


def prepare(plan, stage, manifest):
    if not stage.GetEditTarget().GetLayer().anonymous:
        raise ValueError('Native preparation requires anonymous scene layer')
    result=deepcopy(manifest)
    if not result.get('openend_drawer'):
        from .prepare_scene import add_existing_drawer
        result['openend_drawer']=add_existing_drawer(stage)
    source=Path(result['openend_drawer']['slide_source'])
    result['source_hashes'][str(source)]=hashlib.sha256(source.read_bytes()).hexdigest()
    result['short_task_scene_preparation']=dict(owner=plan['task_name'],existing_slide=True,physics_parameters_changed=False)
    return result
