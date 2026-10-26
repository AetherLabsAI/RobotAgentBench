"""Task boundary and provenance helpers. No Isaac imports or physical setters."""
import hashlib
import json
from pathlib import Path
from .catalog import CASES

HERE = Path(__file__).resolve().parent
CURRENT = Path('/home/lingjun/robot_agent_arena/short_tasks/agent_probe_v1/deployment_generic_20261010/source')
COFFEE_SOURCE = Path('/home/lingjun/robot_agent_arena/api-protocol-st18-20261013-frozen4')
SHELF_DATA = Path('/mnt/agent-bench-data/lingjun/ziche/data/shelf_pick_visible360_lerobot_20260916/sim_info/episode_000128')
COFFEE_EPISODE = 'bag_large_extra_gpu4_seed_140045_episode_0000'
VARIANTS = CASES


def digest(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def owned(path, *, fresh=False):
    path = Path(path).resolve()
    if path == HERE or not path.is_relative_to(HERE) or (fresh and path.exists()):
        raise ValueError('Fresh output below this exact recovery task directory required')
    return path


def write_json(path, value):
    path = owned(path)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def verify_sources(record):
    for path, expected in record.items():
        if digest(path) != expected:
            raise ValueError('Read-only source changed: ' + path)


def require_source(module, root):
    actual = Path(module.__file__).resolve(strict=True)
    if not actual.is_relative_to(root.resolve(strict=True)):
        raise ValueError('Wrong source import environment: ' + str(actual))
    return {str(actual): digest(actual)}


def audit_reset(stage, initialization):
    """Read existing asset limits and invoke the unchanged reset validator."""
    import math
    from pxr import UsdPhysics
    from arena_api.robot_initialization import validate
    from arena_api.task_runtime import RobotInitialization
    names = set(initialization['positions']) | set(initialization['hold_targets'])
    limits, driven, tolerances = {}, [], {}
    for name in names:
        prim = stage.GetPrimAtPath('/World/Robot/joints/' + name)
        angular = prim.IsA(UsdPhysics.RevoluteJoint)
        if not angular and not prim.IsA(UsdPhysics.PrismaticJoint):
            raise ValueError('Unknown existing reset joint: ' + name)
        bounds = [prim.GetAttribute('physics:' + side + 'Limit').Get() for side in ('lower', 'upper')]
        if any(v is None or not math.isfinite(v) for v in bounds):
            raise ValueError('Missing existing native joint limits')
        limits[name] = [math.radians(v) for v in bounds] if angular else bounds
        # Identical native-unit tolerance to the existing reset adapter apply().
        tolerances[name] = math.radians(.001) if angular else .001
        if prim.HasAPI(UsdPhysics.DriveAPI, 'angular' if angular else 'linear'):
            driven.append(name)
    validate(RobotInitialization(positions=initialization['positions'],
        hold_targets=initialization['hold_targets']),
        names, limits, driven, tolerances)
    return dict(existing_native_limit_check='passed', joint_count=len(initialization['positions']),
                drive_parameters_written=False, native_reset_executed=False)


def physics(stage):
    """Include collision, articulation, drives, material bindings and masses."""
    return {str(prop.GetPath()): repr(prop.Get() if hasattr(prop, 'Get') else prop.GetTargets())
            for prim in stage.Traverse() for prop in prim.GetProperties()
            if prop.GetName().startswith(('physics:', 'physx', 'drive:', 'material:binding'))}


def record_layers(stage, scene):
    return {layer.realPath: digest(layer.realPath) for layer in stage.GetUsedLayers()
            if layer.realPath and Path(layer.realPath).resolve() != scene.resolve()}


def finish(stage, out, record, before):
    if physics(stage) != before:
        raise ValueError('Unexpected physical parameter changes during reset composition')
    errors = [str(e) for prim in stage.Traverse() for e in prim.GetPrimIndex().localErrors]
    if errors:
        raise ValueError('Unresolved USD references: ' + repr(errors[:5]))
    record['robot_reset_structure'] = audit_reset(stage, record['robot_initialization'])
    sources = record_layers(stage, out/'initial.usda')
    stage.GetRootLayer().Save()
    verify_sources(sources)
    record.update(schema='short_task_robustness.grasp_failure.initial.v1',
        scene=str(out/'initial.usda'), scene_sha256=digest(out/'initial.usda'), source_layers=sources,
        failure_kind=record.get('failure_kind', 'counterfactual_empty_grasp'), source_is_recorded_failure=False,
        source_progress_inherited=False, initial_score_expected=0,
        initial_score=None, initial_score_measured=False, online_interference=False,
        physics_unchanged=True, physical_settle_verified=False, native_grasp_clearance_verified=False,
        runtime_registered=False, full_recovery_actions_verified=False, layout_scope='atomic_source')
    write_json(out/'initial.json', record)
    return record


def validate_native_start(checks):
    """Only the trusted native reader may supply these startup facts.

    These are prerequisites, not new scoring thresholds. Use the existing source
    readers' support/upright/grasp definitions. A closed command alone proves none.
    """
    required = ('target_supported', 'target_upright', 'empty_gripper',
                'no_robot_target_contact', 'robot_reset_accepted', 'original_initial_accepted')
    if set(checks) != set(required) or any(type(checks[k]) is not bool for k in required):
        raise ValueError('Complete native startup evidence required')
    if not all(checks.values()):
        raise ValueError('Not an accepted empty-grasp recovery initial state: ' + repr(checks))
    return dict(accepted=True, initial_score=0, source_progress_inherited=False)
