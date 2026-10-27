"""Task-owned, pre-physics cloth reset variation; never a runtime action.

Human-approved range: independent XY +/- 0.03 m and yaw +/- 10 degrees.
Only rigid material transforms are allowed. UVs, faces, masses, rest lengths,
fold recipes, heights, native drives, contacts, clocks and scoring are unchanged.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random

SCHEMA = 'robotagentbench.cloth_initial_variation.v1'
POLICY = dict(version='cloth_light_reset_20261026.v1', xy_half_range_m=.03,
              yaw_half_range_deg=10., preserve_height=True, preserve_fold_shape=True,
              preserve_task_workstation=True, illumination_unchanged=True)
TARGETS = {'collect_tablecloth': (0,), 'spread_tablecloth': (1,), 'long_task_18': (0,1)}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _seed(seed):
    if type(seed) is not int or not 1 <= seed <= 4294967295:
        raise ValueError('An explicit positive uint32 seed is required')


def draw(task, seed, attempt=0):
    _seed(seed)
    if task not in TARGETS or type(attempt) is not int or not 0 <= attempt < 256:
        raise ValueError('Unknown cloth task or invalid bounded candidate index')
    result = {}
    for cid in TARGETS[task]:
        rng = random.Random(f'{SCHEMA}:{task}:{seed}:{cid}:{attempt}')
        result[str(cid)] = dict(translation_xy_m=[rng.uniform(-.03,.03), rng.uniform(-.03,.03)],
                               yaw_deg=rng.uniform(-10.,10.))
    return result


def validate_poses(task, poses):
    if task not in TARGETS or not isinstance(poses,dict) or set(poses) != set(map(str,TARGETS[task])):
        raise ValueError('Exact task-owned cloth identities are required')
    for value in poses.values():
        if not isinstance(value,dict) or set(value) != {'translation_xy_m','yaw_deg'}:
            raise ValueError('Only planar position and yaw may vary')
        xy = value['translation_xy_m']
        if not isinstance(xy,list) or len(xy) != 2:
            raise ValueError('Require explicit XY translation')
        if any(type(v) not in (int,float) or not math.isfinite(v) or abs(v) > .03 for v in xy):
            raise ValueError('XY variation exceeds the approved 3 cm range')
        angle = value['yaw_deg']
        if type(angle) not in (int,float) or not math.isfinite(angle) or abs(angle) > 10.:
            raise ValueError('Yaw variation exceeds the approved 10 degree range')
    return deepcopy(poses)


def transform_initial(initial, cloth_ids, task, poses):
    import numpy as np
    poses = validate_poses(task, poses)
    q = np.array(initial, dtype=np.float64, copy=True)
    ids = np.asarray(cloth_ids)
    if q.ndim != 2 or q.shape[1] != 3 or ids.shape != (len(q),) or not np.isfinite(q).all():
        raise ValueError('Finite original material-node positions and IDs required')
    for key, pose in poses.items():
        mask = ids == int(key)
        if not mask.any():
            raise ValueError('Requested cloth is absent from the original initial state')
        center = q[mask,:2].mean(0)
        angle = math.radians(pose['yaw_deg'])
        rotation = np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
        q[mask,:2] = (q[mask,:2]-center)@rotation.T + center + pose['translation_xy_m']
    if not np.array_equal(q[:,2], np.asarray(initial)[:,2]):
        raise AssertionError('Cloth reset changed height')
    return q


def choose_initial(original, task, seed, *, geometry_gate):
    """Bounded deterministic CPU candidates, with every rejection recorded.

    A geometry gate is mandatory and supplied only by trusted preparation code.
    This does not retry seeds after native failures or bless CPU evidence as a
    physically settled episode. The original native admission still applies.
    """
    if not callable(geometry_gate):
        raise ValueError('An original-geometry initial-state gate is required')
    import numpy as np
    required = {'initial_cloth_q','cloth_id','uv','faces','joint_names','initial_joint_targets'}
    if not required <= set(original):
        raise ValueError('Incomplete original prepared state')
    rejected = []
    for attempt in range(256):
        poses = draw(task, seed, attempt)
        q = transform_initial(original['initial_cloth_q'], original['cloth_id'], task, poses)
        verdict = geometry_gate(q)
        if not isinstance(verdict,dict) or type(verdict.get('accepted')) is not bool:
            raise ValueError('Geometry gate must return explicit measured acceptance')
        if not verdict['accepted']:
            rejected.append(dict(attempt=attempt, reasons=deepcopy(verdict.get('reasons',[]))))
            continue
        value = {k: np.array(v,copy=True) for k,v in original.items()}
        value['initial_cloth_q'] = q
        return value, dict(schema=SCHEMA, task_name=task, seed=seed, policy=deepcopy(POLICY),
            attempt=attempt, poses=poses, geometry=deepcopy(verdict), rejected_candidates=rejected,
            status='cpu_geometry_prepared', native_initial_state_accepted=False)
    raise ValueError('No valid cloth initial state within the approved range; seed is not replaced')


def verify_record(record, *, task=None):
    required = {'schema','task_name','seed','policy','attempt','poses','geometry','rejected_candidates',
                'status','native_initial_state_accepted','baseline_sha256','initial_sha256',
                'layout','appearance','source_hashes'}
    if not isinstance(record,dict) or set(record) != required or record['schema'] != SCHEMA:
        raise ValueError('Exact prepared cloth variation record required')
    name = record['task_name']
    if task is not None and name != task:
        raise ValueError('Cloth variation belongs to another task')
    _seed(record['seed'])
    if record['policy'] != POLICY or record['status'] != 'cpu_geometry_prepared' or record['native_initial_state_accepted'] is not False:
        raise ValueError('Unknown reset policy or inherited native acceptance')
    if record['poses'] != draw(name, record['seed'], record['attempt']):
        raise ValueError('Prepared cloth pose differs from its deterministic seed')
    validate_poses(name,record['poses'])
    if record['geometry'].get('accepted') is not True:
        raise ValueError('Rejected initial geometry cannot become a launch')
    layout = record['layout']
    if not isinstance(layout,dict) or set(layout) != {'id','catalog','catalog_sha256','workstation_unchanged'} or layout['workstation_unchanged'] is not True:
        raise ValueError('Only existing fixed-workstation room layouts are allowed')
    look = record['appearance']
    if not isinstance(look,dict) or set(look) != {'style','source','source_sha256','wall_rgb'}:
        raise ValueError('Only original room-colour suites are supported')
    for key in ('baseline_sha256','initial_sha256'):
        if not isinstance(record[key],str) or len(record[key]) != 64 or any(c not in '0123456789abcdef' for c in record[key]):
            raise ValueError('Explicit prepared-state SHA256 required')
    if not isinstance(record['source_hashes'],dict) or not record['source_hashes']:
        raise ValueError('Pinned reset and original appearance/layout sources required')
    return record


def load_variation(value, workspace):
    """Read-only admission of an operator-created per-instance sidecar."""
    if not isinstance(value,dict) or set(value) != {'record','record_sha256','initial','initial_sha256'}:
        raise ValueError('Explicit immutable variation sidecars required')
    record_path, initial_path = workspace.input(value['record']), workspace.input(value['initial'])
    if digest(record_path) != value['record_sha256'] or digest(initial_path) != value['initial_sha256']:
        raise ValueError('Prepared cloth variation changed')
    record = verify_record(json.loads(record_path.read_text()))
    if record['initial_sha256'] != value['initial_sha256']:
        raise ValueError('Cloth record and initial-state identities differ')
    for path, expected in record['source_hashes'].items():
        if digest(workspace.input(path)) != expected:
            raise ValueError('Original reset/layout/appearance dependency changed')
    return record, initial_path


def verify_initial(original, prepared, task, record):
    """Require exactly the seeded planar reset, not an arbitrary NPZ override."""
    import numpy as np
    verify_record(record, task=task)
    if set(original) != set(prepared):
        raise ValueError('Prepared variation changed the original array inventory')
    for name in original:
        expected = (transform_initial(original[name], original['cloth_id'], task, record['poses'])
                    if name == 'initial_cloth_q' else original[name])
        if prepared[name].dtype != expected.dtype or not np.array_equal(prepared[name], expected):
            raise ValueError('Variation changed more than the approved rigid cloth reset: '+name)


def colour_suites(source):
    """Read existing literal colour definitions without running decor/light code."""
    import ast
    tree = ast.parse(Path(source).read_text())
    values = {}
    for node in tree.body:
        target = (node.target if isinstance(node,ast.AnnAssign) else
                  node.targets[0] if isinstance(node,ast.Assign) and len(node.targets)==1 else None)
        if isinstance(target,ast.Name) and target.id == 'STYLE_SUITES':
            values = ast.literal_eval(node.value)
    if not values:
        raise ValueError('Original room colour definitions missing')
    result = {name:list(value['wall_rgb']) for name,value in values.items()}
    if any(len(rgb)!=3 or any(not 0<=c<=1 for c in rgb) for rgb in result.values()):
        raise ValueError('Invalid original wall palette')
    return result


def scene_choices(catalog, appearance_source, task, seed):
    """Choose existing recipes only; never relocate the cloth workstation."""
    _seed(seed)
    value = json.loads(Path(catalog).read_text())
    if value.get('schema') != 'real2sim.house_layouts.v1' or value.get('status') != '2d_candidate':
        raise ValueError('Original candidate layout catalog required')
    rows = [r for r in value['layouts']
            if r['cluster_transforms'].get('coffee_table_cluster') ==
            dict(translate_xy_m=[0.,0.],yaw_deg=0.)]
    if not rows or set(value['clusters']['coffee_table_cluster']['prim_roots']) != {
            '/World/Assets/CoffeeTable','/World/Assets/TableclothBasket',
            '/World/Assets/TrashBinTeal','/World/Assets/TrashBinRed'}:
        raise ValueError('Original table/basket cluster must remain fixed')
    rng = random.Random(f'{SCHEMA}:room:{task}:{seed}')
    rows = sorted(rows,key=lambda r:r['ordinal'])
    # Consecutive preparation seeds visit different catalog layouts before reuse.
    layout = rows[(seed-1)%len(rows)]
    suites = colour_suites(appearance_source)
    style = rng.choice(sorted(suites))
    return (dict(id=layout['id'],catalog=str(catalog),catalog_sha256=digest(catalog),
                 workstation_unchanged=True),
            dict(style=style,source=str(appearance_source),source_sha256=digest(appearance_source),
                 wall_rgb=suites[style]))


def verify_scene_choices(record):
    layout, look = scene_choices(record['layout']['catalog'],record['appearance']['source'],
                                 record['task_name'],record['seed'])
    if record['layout'] != layout or record['appearance'] != look:
        raise ValueError('Prepared layout or colour is not the recorded seeded original recipe')
