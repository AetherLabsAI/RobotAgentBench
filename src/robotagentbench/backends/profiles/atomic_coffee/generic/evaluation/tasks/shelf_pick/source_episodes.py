"""Convert one ziche shelf-pick episode into an `arena.shelf_pick.initial_state.v2` snapshot.

The 2026-09-16 PICK dataset (`shelf_pick_visible360_lerobot_20260916`, 210 accepted
episodes) replaces the old L1 snapshots as the source of the fixed initial state for
API 4-1 … 4-14. This module is pure CPU: it reads the read-only dataset plus the 202
baseline scene text, verifies the episode against the 202 stock inventory and against
the recorded trajectory, and writes the snapshot consumed by `initial_state.prepare()`.

Nothing is silently repaired. Every disagreement raises.
"""
import hashlib
import json
import math
import re
from pathlib import Path

from .initial_state import BASELINE, ROBOT
from .targets import SHELF_ITEMS, TIER2_ITEMS, TIER3, TIER2, spec as target_spec

SCHEMA = 'arena.shelf_pick.initial_state.v2'
DATASET = Path('/mnt/agent-bench-data/lingjun/ziche/data/shelf_pick_visible360_lerobot_20260916')
# Source tier id -> (202 stock layer, 202 support slab).
TIERS = {'tier3': (SHELF_ITEMS, TIER3), 'tier2': (TIER2_ITEMS, TIER2)}
# The 20 joints the source controller commands; the remaining 14 gripper linkage
# DOFs are passive followers and are not cross-checked against the recording.
CONTROLLED_JOINTS = ('idx01_body_joint1', 'idx02_body_joint2', 'idx11_head_joint1', 'idx12_head_joint2',
                     *(f'idx{prefix}{k}_arm_{side}_joint{k}' for k in range(1, 8) for prefix, side in ((2, 'l'), (6, 'r'))),
                     'idx41_gripper_l_outer_joint1', 'idx81_gripper_r_outer_joint1')
JOINT_TOLERANCE_RAD = 1e-6
POSE_TOLERANCE_M = 1e-6
# `initial_state.npz` is authored, not executed; the actual start is recording frame 0.
AUTHORED_STATE_KIND = 'AUTHORED_INITIAL_STATE_ONLY_NOT_ACTUAL'
CLEAR_PREFIX = 'CLEAR'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


_PRIM = re.compile(r'^(?:def|over|class)\s+(?:\w+\s+)?"([^"]+)"')


ALIGNMENT_LAYER = BASELINE.parent/'source_release_alignment.usda'


def baseline_stock_children(baseline=BASELINE, *, alignment=None):
    """Child prim paths of the 202 stock layers, parsed from the scene text.

    Text parsing keeps extraction CPU-only and free of a USD runtime; the composed
    stage is re-checked anyway when `initial_state.prepare()` activates the prims.

    Since 2026-09-24 the stock is the union of two layers: the baseline, and
    `source_release_alignment.usda`, which carries the data owner's own
    `<product>_copy_NN` duplicates. Reading only the baseline would report those
    duplicates as "not stocked" and throw away every source episode that uses one
    (148 of the 210 shelf-pick episodes do).
    """
    alignment = ALIGNMENT_LAYER if alignment is None else alignment
    lines = Path(baseline).read_text().splitlines()
    extra = Path(alignment).read_text().splitlines() if Path(alignment).is_file() else []
    roots = {}
    for root in (SHELF_ITEMS, TIER2_ITEMS):
        name = root.rsplit('/', 1)[-1]
        start = next((i for i, line in enumerate(lines)
                      if _PRIM.match(line.strip()) and _PRIM.match(line.strip()).group(1) == name), None)
        if start is None:
            raise ValueError('Baseline has no stock layer ' + root)
        indent = len(lines[start]) - len(lines[start].lstrip())
        i = start
        while not lines[i].strip().startswith('{'):
            i += 1
        children = set()
        for line in lines[i + 1:]:
            stripped = line.strip()
            depth = len(line) - len(line.lstrip())
            if stripped == '}' and depth == indent:
                break
            match = _PRIM.match(stripped)
            if match and depth == indent + 4:
                children.add(root + '/' + match.group(1))
        if not children:
            raise ValueError('Baseline stock layer is empty ' + root)
        children |= _layer_children(extra, root)
        roots[root] = children
    return roots


def _layer_children(lines, root):
    """同样的文本解析，用在对齐层上；该层没有这一节时返回空集。"""
    name = root.rsplit('/', 1)[-1]
    start = next((i for i, line in enumerate(lines)
                  if _PRIM.match(line.strip()) and _PRIM.match(line.strip()).group(1) == name),
                 None)
    if start is None:
        return set()
    indent = len(lines[start]) - len(lines[start].lstrip())
    i = start
    while i < len(lines) and not lines[i].strip().startswith('{'):
        i += 1
    children = set()
    for line in lines[i+1:]:
        stripped = line.strip()
        depth = len(line) - len(line.lstrip())
        if stripped == '}' and depth == indent:
            break
        match = _PRIM.match(stripped)
        if match and depth == indent + 4:
            children.add(root + '/' + match.group(1))
    return children


def _finite(values):
    return all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)


def _pose(xyz, wxyz, label):
    position = [float(v) for v in xyz]
    quaternion = [float(v) for v in wxyz]
    if len(position) != 3 or len(quaternion) != 4 or not _finite(position) or not _finite(quaternion):
        raise ValueError('Invalid pose ' + label)
    norm = math.sqrt(sum(v * v for v in quaternion))
    if not math.isfinite(norm) or abs(norm - 1) > 1e-6:
        raise ValueError(f'Non-unit quaternion {label}: |q| = {norm!r}')
    return dict(position_m=[v for v in position], quaternion_wxyz=[v / norm for v in quaternion])


def episode_directory(episode_index, dataset=DATASET):
    return Path(dataset) / 'sim_info' / f'episode_{int(episode_index):06d}'


def extract(episode_dir, api_id, *, baseline=BASELINE, stock_children=None, dataset=None):
    """Build a v2 snapshot for `api_id` from one dataset episode directory.

    Raises on a non-CLEAR world audit, a prim the 202 baseline does not stock, a joint
    that disagrees with recorded frame 0, or a pose that disagrees with the authored
    `initial_state.npz`.

    `<product>_copy_NN` duplicates used to be rejected outright, because the 202
    baseline stocked exactly one instance of each product. Since 2026-09-24 the
    evaluation scene carries the data owner's own duplicates (see
    `assets/house_scene_l1/demo_sim/source_release_alignment.usda`), so a duplicate is
    accepted exactly like any other prim: it has to be stocked by the baseline, and if
    it is not, the same "not stocked" error fires with its path.
    """
    import numpy as np
    spec = target_spec(api_id)
    episode_dir = Path(episode_dir).resolve(strict=True)
    stock_children = stock_children or baseline_stock_children(baseline)
    randomization_path = episode_dir / 'randomization.json'
    authored_path = episode_dir / 'initial_state.npz'
    recording_path = episode_dir / 'recording_20hz.npz'
    randomization = json.loads(randomization_path.read_text())
    sim_info = json.loads((episode_dir / 'sim_info.json').read_text())
    audit = json.loads((episode_dir / 'WORLD_AUDIT.json').read_text())
    # Absent for the three legacy imports; the policy is then the sampling policy.
    selection_path = episode_dir / 'TARGET_SELECTION.json'
    selection = json.loads(selection_path.read_text()) if selection_path.is_file() else {}
    source_metadata = json.loads((episode_dir / 'source_metadata.json').read_text())

    status = audit.get('world_geometry_status')
    if not isinstance(status, str) or not status.startswith(CLEAR_PREFIX):
        raise ValueError(f'Episode world geometry is not CLEAR: {status!r}')

    # `objects` already carries the target entry (role 'target'); the top-level
    # `target` block is the same pose, kept for provenance.
    entries = list(randomization['objects'])

    target_entry = randomization['target']
    target_path = target_entry['prim_path']
    roles = [e for e in entries if e['role'] == 'target']
    if len(roles) != 1 or roles[0]['prim_path'] != target_path or roles[0]['xyz_m'] != target_entry['xyz_m'] \
            or roles[0]['wxyz'] != target_entry['wxyz']:
        raise ValueError('Episode target block disagrees with the sampled layout')
    if not spec.is_target(target_path):
        raise ValueError(f'Episode target {target_path} is not an accepted target of API {api_id}')
    tier = target_entry.get('support_id')
    if tier not in TIERS or not target_path.startswith(TIERS[tier][0] + '/'):
        raise ValueError(f'Target {target_path} does not sit on its declared tier {tier!r}')

    seen = set()
    placed = []
    for entry in entries:
        path = entry['prim_path']
        entry_tier = entry.get('support_id')
        if entry_tier not in TIERS:
            raise ValueError(f'Unknown support tier {entry_tier!r} for {path}')
        root, support = TIERS[entry_tier]
        if not path.startswith(root + '/') or path.count('/') != root.count('/') + 1:
            raise ValueError(f'{path} is not a direct child of its tier layer {root}')
        if path not in stock_children[root]:
            raise ValueError(f'{path} is not stocked by the 202 baseline layer {root}')
        if path in seen:
            raise ValueError('Duplicate prim in episode layout: ' + path)
        seen.add(path)
        placed.append((entry, support))

    authored = np.load(authored_path, allow_pickle=True)
    recorded = np.load(recording_path, allow_pickle=True)
    if str(authored['state_kind']) != AUTHORED_STATE_KIND:
        raise ValueError('Unexpected authored state kind ' + str(authored['state_kind']))

    dof_names = [str(n) for n in authored['robot_dof_names']]
    authored_q = [float(v) for v in authored['robot_initial_joint_position_rad_or_m']]
    if len(dof_names) != len(authored_q) or len(set(dof_names)) != len(dof_names):
        raise ValueError('Malformed robot dof list')
    if not _finite(authored_q):
        raise ValueError('Non-finite authored joint position')
    joint_positions = dict(zip(dof_names, authored_q))
    recorded_names = [str(n) for n in recorded['robot_dof_names']]
    frame0 = [float(v) for v in recorded['robot_joint_position_rad_or_m'][0]]
    missing = [n for n in CONTROLLED_JOINTS if n not in joint_positions or n not in recorded_names]
    if missing:
        raise ValueError('Recording does not carry the controlled joints: ' + ', '.join(missing))
    joint_deviations = {n: abs(joint_positions[n] - frame0[recorded_names.index(n)]) for n in CONTROLLED_JOINTS}
    joint_deviation = max(joint_deviations.values())
    if joint_deviation > JOINT_TOLERANCE_RAD:
        worst = max(joint_deviations, key=joint_deviations.get)
        raise ValueError(f'Authored joint {worst} deviates {joint_deviation:.3e} from recorded frame 0')

    body_paths = [str(p) for p in authored['rigid_body_paths']]
    authored_xyz = authored['prop_xyz_m'][0]
    authored_wxyz = authored['prop_wxyz'][0]
    if len(body_paths) != len(authored_xyz) or len(body_paths) != len(authored_wxyz):
        raise ValueError('Malformed authored prop arrays')
    if set(body_paths) != seen:
        raise ValueError('Authored rigid bodies differ from the sampled layout: '
                         + ', '.join(sorted(set(body_paths) ^ seen)))
    pose_deviation = 0.
    for entry, _ in placed:
        index = body_paths.index(entry['prim_path'])
        pose_deviation = max(pose_deviation,
                             max(abs(float(a) - float(b)) for a, b in zip(entry['xyz_m'], authored_xyz[index])),
                             max(abs(float(a) - float(b)) for a, b in zip(entry['wxyz'], authored_wxyz[index])))
    if pose_deviation > POSE_TOLERANCE_M:
        raise ValueError(f'Sampled layout deviates {pose_deviation:.3e} from the authored initial state')

    robot = _pose(authored['robot_root_xyz_m'], authored['robot_root_wxyz'], 'robot')
    items = [dict(prim_path=entry['prim_path'], support=support,
                  **_pose(entry['xyz_m'], entry['wxyz'], entry['prim_path']))
             for entry, support in placed if entry['prim_path'] != target_path]

    dataset_root = Path(dataset) if dataset else episode_dir.parents[1]
    source = dict(episode_id=str(authored['episode_uid']),
                  lerobot_episode_index=int(sim_info['episode_index']),
                  sim_info_dir=str(episode_dir),
                  dataset=str(dataset_root),
                  placement_policy=randomization['placement_policy'],
                  hand=source_metadata['hand'],
                  world_geometry_status=status,
                  seed=int(randomization['seed']),
                  sha256=dict(randomization_json=digest(randomization_path),
                              initial_state_npz=digest(authored_path),
                              recording_20hz_npz=digest(recording_path)),
                  target_selection_policy=selection.get('policy', randomization.get('scene_sampling_policy')),
                  asset_release=sim_info['asset_release'],
                  physics_hz=sim_info['physics_hz'], control_hz=sim_info['control_hz'],
                  robot_frame=ROBOT,
                  scope='object layout, target instance and robot station of one accepted source episode; '
                        'source physics (60 Hz) and controller are NOT reproduced')
    snapshot = dict(schema=SCHEMA, api_id=spec.api_id, source=source,
                    target_path=target_path, target_tier=tier,
                    target=_pose(target_entry['xyz_m'], target_entry['wxyz'], target_path),
                    robot=robot, joint_positions=joint_positions, items=items,
                    verification=dict(controlled_joints=list(CONTROLLED_JOINTS),
                                      max_joint_deviation_rad=joint_deviation,
                                      joint_tolerance_rad=JOINT_TOLERANCE_RAD,
                                      max_pose_deviation_m=pose_deviation,
                                      pose_tolerance_m=POSE_TOLERANCE_M,
                                      joint_reference='recording_20hz.npz frame 0',
                                      pose_reference='initial_state.npz prop arrays',
                                      baseline_sha256=digest(baseline),
                                      baseline=str(baseline)),
                    physical_acceptance=False)
    validate(snapshot, stock_children=stock_children)
    return snapshot


def validate(snapshot, *, stock_children=None, baseline=BASELINE):
    """Re-check a snapshot document without touching the dataset."""
    stock_children = stock_children or baseline_stock_children(baseline)
    if snapshot.get('schema') != SCHEMA:
        raise ValueError('Not a v2 shelf-pick snapshot')
    spec = target_spec(snapshot['api_id'])
    if not spec.is_target(snapshot['target_path']):
        raise ValueError('Snapshot target is not an accepted target of ' + spec.api_id)
    for label, pose in (('target', snapshot['target']), ('robot', snapshot['robot'])):
        _pose(pose['position_m'], pose['quaternion_wxyz'], label)
    if not snapshot['joint_positions'] or not _finite(list(snapshot['joint_positions'].values())):
        raise ValueError('Invalid joint positions')
    for name in CONTROLLED_JOINTS:
        if name not in snapshot['joint_positions']:
            raise ValueError('Snapshot is missing controlled joint ' + name)
    seen = {snapshot['target_path']}
    for item in snapshot['items']:
        path = item['prim_path']
        root = path.rsplit('/', 1)[0]
        if root not in stock_children or path not in stock_children[root]:
            raise ValueError('Snapshot item is not stocked by the 202 baseline: ' + path)
        if item['support'] != TIERS['tier3' if root == SHELF_ITEMS else 'tier2'][1]:
            raise ValueError('Snapshot item support does not match its tier: ' + path)
        if path in seen:
            raise ValueError('Duplicate snapshot item: ' + path)
        seen.add(path)
        _pose(item['position_m'], item['quaternion_wxyz'], path)
    verification = snapshot['verification']
    if verification['max_joint_deviation_rad'] > JOINT_TOLERANCE_RAD or verification['max_pose_deviation_m'] > POSE_TOLERANCE_M:
        raise ValueError('Snapshot records a verification deviation beyond tolerance')
    return snapshot
