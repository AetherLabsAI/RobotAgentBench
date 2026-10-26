"""Bind identities from unified scene; never initialize or randomize objects."""
from dataclasses import dataclass
import math
from .definitions import definition
from openend.tasks.remote_transfer.selection import ContactSelection

def path(p):
    if not isinstance(p,str) or not p.startswith('/') or p=='/' or p.endswith('/'):
        raise ValueError('Explicit non-root prim path required')
    return p

@dataclass(frozen=True)
class Binding:
    key: str
    episode_id: str
    instance_id: str
    target: str
    source_furniture: str
    destination: str
    destination_furniture: str
    initial_supports: frozenset
    initial_height_m: float
    initial_time_s: float
    scenery: frozenset

def bind(manifest, *, key, episode_id, instance_id, target_path, source_furniture,
         destinations, initial_supports, initial_height_m, height_reference,
         initial_time_s, scenery=frozenset()):
    d=definition(key)
    rows=[o for o in manifest['objects'] if o['id']==d.asset]
    matches=[o for o in rows if o['path']==target_path]
    if len(matches)!=1 or (d.asset!='paper_cup' and len(rows)!=1):
        raise ValueError('Exact scene-owned object required; only cups may repeat')
    if any(not isinstance(v,str) or not v.strip() for v in (episode_id,instance_id)):
        raise ValueError('Episode and task instance required')
    dest=destinations[d.destination_role]
    for p in (target_path,source_furniture,dest['path'],dest['furniture']): path(p)
    if target_path in (source_furniture,dest['path'],dest['furniture']):
        raise ValueError('Object and furniture must be distinct')
    if height_reference!=d.height_reference:
        raise ValueError('Incorrect initial height reference; cap needs AABB center')
    for v in (initial_height_m,initial_time_s):
        if type(v) not in (int,float) or not math.isfinite(v): raise ValueError('Finite measurements required')
    if initial_time_s<0: raise ValueError('Invalid simulation time')
    if not isinstance(initial_supports,frozenset) or not initial_supports:
        raise ValueError('Measured initial supports required')
    if not isinstance(scenery,frozenset): raise ValueError('Explicit scenery identities required')
    for p in initial_supports|scenery: path(p)
    if target_path in initial_supports|scenery: raise ValueError('Self support/scenery')
    return Binding(key,episode_id,instance_id,target_path,source_furniture,
        dest['path'],dest['furniture'],initial_supports,initial_height_m,initial_time_s,scenery)

def validate_exclusive(bindings):
    # Multiple same-class cups are allowed, one task per exact object instance.
    if len({b.instance_id for b in bindings})!=len(bindings): raise ValueError('Duplicate task instance')
    if len({b.target for b in bindings})!=len(bindings): raise ValueError('Two tasks claim the same object')
    if len({b.episode_id for b in bindings})>1: raise ValueError('Mixed episodes')
