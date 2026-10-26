"""Approved size definitions; no USD writes, sampling or physical overrides."""
from copy import deepcopy
import json
from pathlib import Path


def definition(variant):
    data = json.loads(Path(__file__).with_name('paper_variants.json').read_text())
    if not isinstance(variant, str) or variant not in data['variants']:
        raise ValueError('Paper variant must be large or small')
    return deepcopy(dict(asset_id=data['asset_id'], asset_entry=data['asset_entry'],
                         trash_category=data['trash_category'], paper_variant=variant,
                         **data['variants'][variant]))


def annotate_object(obj, variant):
    """Scene producer calls AFTER creating/measuring the chosen variant.

    This declaration alone does not resize an object or prove its dimensions.
    Stable task identity remains the exact object path, not the variant name.
    """
    if obj.get('id') != 'paper_ball':
        raise ValueError('Paper variant requires the paper_ball asset')
    spec = definition(variant)
    if obj.get('trash_category', 'paper') != 'paper':
        raise ValueError('Conflicting garbage category')
    if obj.get('paper_variant', variant) != variant:
        raise ValueError('Conflicting paper variant')
    result = deepcopy(obj)
    result.update(trash_category='paper', paper_variant=variant,
                  paper_effective_mesh_axis_scale=spec['effective_mesh_axis_scale'])
    return result
