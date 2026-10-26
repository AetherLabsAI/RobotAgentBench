"""Human-selected five starts, not five replacement scoring policies.

The historical eight-case catalog is intentionally left byte-for-byte intact:
published v1 source records pin it. This is the current development scope.
"""
from copy import deepcopy
from ..catalog import definition as previous_definition

CASES = {
    'shelf_drink': dict(previous_definition('shelf_drink'), number=1),
    'coffee_bag': dict(previous_definition('coffee_bag'), number=2),
    'sofa': dict(previous_definition('sofa'), number=3,
        original_task='sofa_to_coffee_table',
        completion='same_target_released_in_permitted_coffee_table_object_region',
        active_hand_policy='target_side_in_actual_robot_base_frame.v1',
        gap_range_m=(.02, .05), orientation_jitter=False,
        target_policy='one_remote_tv',
        robot_station_policy='post_failure_operating_station.v1',
        initial_history_inherited=False),
    'table_waste_tipped_cup': dict(number=4,
        title='Clear table waste (Tipped cup)', original_task='clear_table_trash',
        failure_kind='early_release_tipped_cup', source_component='trash_transfer',
        category='cup', asset='paper_cup', destination_role='ordinary_trash_bin',
        source_disposal_api='18-4', component_maximum_score=8,
        maximum_score='resolved_full_tree_plan_including_existing_common_stages',
        instruction='Put the used cup in the correct bin.',
        endpoint='same_cup_released_inside_correct_bin',
        completion='same_cup_released_inside_correct_bin',
        preserved=('same_target_identity', 'cup_on_its_side', 'supported_on_coffee_table',
                   'target_visible', 'released_target', 'target_inside_workspace'),
        generation='physical_grasp_lift_early_release_then_freeze_measured_rest_state',
        physical_history_required=True, initial_history_inherited=False,
        no_completion_for=('righting_cup', 'grasping_cup', 'lifting_cup', 'wrong_bin',
                           'bin_rim', 'held_inside_bin', 'out_of_workspace'),
        randomization='existing_table_task_initialization_and_accepted_failed_release'),
    'table_waste_empty_grasp': dict(number=5,
        title='Clear table waste (Missed grasp)', original_task='clear_table_trash',
        failure_kind='empty_grasp', source_component='trash_transfer',
        endpoint='same_target_released_inside_correct_bin',
        completion='same_target_released_inside_correct_bin',
        preserved=('same_target_identity', 'supported_on_coffee_table', 'target_visible',
                   'empty_attempted_grasp', 'closed_active_gripper', 'no_robot_target_contact'),
        randomization='existing_table_task_initialization',
        target_policy='one_upright_empty_paper_cup',
        initialization_order='solve_empty_closed_arm_then_place_same_upright_cup',
        placement_policy='original_full_living_plan_table_region_xy_only',
        active_hand_policy='target_side_in_actual_robot_base_frame.v1',
        gap_range_m=(.02, .05), orientation_jitter=False,
        robot_station_policy='post_failure_operating_station.v1',
        initial_history_inherited=False),
}


def definition(name):
    try:
        return deepcopy(CASES[name])
    except KeyError as exc:
        raise ValueError('Not one of the five active recovery variants: '+str(name)) from exc


def base_request(name, seed, *, sofa_asset=None, waste_categories=None):
    """Explicit original task request; no invented distribution or public ID."""
    d = definition(name)
    if type(seed) is not int or not 1 <= seed < 2**32:
        raise ValueError('Positive uint32 seed required')
    task = d.get('original_task')
    if task not in ('sofa_to_coffee_table', 'clear_table_trash'):
        raise ValueError('Existing shelf/coffee registrations remain on v1')
    q = dict(schema='openend.short_task.request.v1', task_name=task,
             rule_version='openend.short_task.'+task+'.v1', seed=seed, layout_id=None)
    if task == 'sofa_to_coffee_table':
        if sofa_asset not in ('remote_tv', 'book_red', 'wallet', 'glasses_case') or waste_categories is not None:
            raise ValueError('Explicit original sofa asset required')
        q['options'] = dict(asset=sofa_asset)
    else:
        if sofa_asset is not None or not isinstance(waste_categories, list) or not waste_categories:
            raise ValueError('Explicit original waste categories required')
        if any(c not in ('paper', 'snack', 'can', 'cup') for c in waste_categories):
            raise ValueError('Unknown original trash category')
        if name == 'table_waste_tipped_cup' and waste_categories != ['cup']:
            raise ValueError('The tipped-cup task has exactly one used cup')
        q['targets'] = [dict(category=c, quantity=1) for c in waste_categories]
    return q


def validate_tipped_history(history, *, target, bin_path):
    """An authored sideways pose is not evidence of a physical failed attempt."""
    if history.get('target') != target or history.get('destination') != bin_path:
        raise ValueError('Failure-generation witness belongs to a different cup/bin')
    if history.get('synthetic_pose_only') is not False:
        raise ValueError('A sideways reset alone does not certify the requested failed attempt')
    rows = history.get('events', [])
    expected = ('physical_grasp', 'lift_without_support', 'release_before_bin', 'rest_on_side_on_table')
    if tuple(r.get('kind') for r in rows) != expected:
        raise ValueError('Ordered grasp/lift/early-release/rest witnesses required')
    times = [r.get('time_s') for r in rows]
    import math
    if any(type(t) not in (int, float) or not math.isfinite(t) or t < 0 for t in times):
        raise ValueError('Finite simulation times required')
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError('Failure generation phases must be physically ordered')
    checks = ('measured_contacts', 'measured_supported_rest', 'measured_released',
              'visible', 'inside_workspace', 'same_dynamic_asset', 'physics_unchanged')
    if any(history.get(k) is not True for k in checks):
        raise ValueError('Complete native failed-state evidence required')
    return dict(accepted=True, initial_points=0, source_progress_inherited=False,
                physical_failure_generation_verified=True)
