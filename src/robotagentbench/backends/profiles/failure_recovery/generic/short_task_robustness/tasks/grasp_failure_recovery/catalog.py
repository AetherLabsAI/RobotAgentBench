"""The eight user-selected failure starts, not eight new scoring definitions.

Keys below identify task-local variants, not registered public service IDs.
The abandoned food-box placement candidate is deliberately absent.
"""
from copy import deepcopy
import hashlib
import random

CASES = {
    'shelf_drink': dict(number=1, title='Shelf drink missed grasp', base_task='shelf_pick',
        source_api='4-3', failure_kind='empty_grasp', maximum_score=4,
        endpoint='original_pick_success_only',
        instruction='Pick up the Gatorade T bottle from the shelf and hold it steadily.',
        preserved=('upright_supported_target', 'empty_attempted_grasp', 'closed_active_gripper',
                   'hand_matches_target_side', 'unique_target'),
        randomization='accepted_source_shelf_layout_and_existing_robot_perturbation'),
    'coffee_bag': dict(number=2, title='Coffee bag missed grasp', base_task='atomic_coffee',
        source_api='1-2', failure_kind='empty_grasp', maximum_score=6,
        endpoint='pour_beans_and_return_bag',
        instruction='Pick up the coffee bean bag, pour the beans into the machine hopper, then return the bag upright to the counter.',
        checkpoint_phases=('lift_bag', 'carry_bag'),
        preserved=('upright_supported_target', 'all_100_beans_in_bag', 'empty_hopper',
                   'empty_attempted_grasp', 'closed_active_gripper'),
        randomization='original_atomic_1_2_sampler'),
    'coffee_cup': dict(number=3, title='Coffee cup misplaced', base_task='atomic_coffee',
        source_api='1-1', failure_kind='misplaced_upright_cup', maximum_score=5,
        endpoint='original_cup_placement',
        instruction='Place the coffee cup under the coffee machine outlet, then release it and leave it stable.',
        checkpoint_phases=('withdraw',),
        preserved=('upright_supported_target', 'released_target', 'placement_goal_not_met',
                   'whole_cup_outside_receiving_bay'),
        randomization='original_atomic_1_1_sampler'),
    'coffee_button': dict(number=4, title='Coffee button missed press', base_task='atomic_coffee',
        source_api='1-3', failure_kind='missed_button_press', maximum_score=5,
        endpoint='original_start_and_stop_dispensing',
        instruction='Press the coffee machine button to start dispensing coffee, then press it again to stop after collecting 25–60 ml.',
        checkpoint_phases=('horizontal_return',),
        preserved=('machine_idle', 'button_released', 'hand_clear_of_button', 'cup_correctly_on_tray'),
        randomization='original_atomic_1_3_sampler'),
    'cutlery': dict(number=5, title='Cutlery missed grasp', base_task='prepare_cutlery',
        failure_kind='empty_grasp', endpoint='original_three_utensils_to_meal_tray',
        instruction='Pick all three utensils from their holder slots and place them neatly in parallel on the meal tray.',
        preserved=('three_source_slot_targets', 'empty_attempted_grasp', 'none_already_on_tray'),
        randomization='existing_equal_three_families_geometry_fitted_and_source_environment_ranges'),
    'drawer': dict(number=6, title='Drawer handle lost', base_task='retrieve_from_drawer',
        failure_kind='partial_drawer_handle_lost', endpoint='original_drawer_target_and_configured_destination',
        preserved=('drawer_strictly_partly_open', 'handle_not_held', 'target_supported_inside_drawer'),
        randomization='existing_drawer_source_ranges_and_measured_partial_opening'),
    'basket': dict(number=7, title='Air fryer basket handle lost', base_task='air_fryer_pull',
        source_subtask='2.3', failure_kind='partial_basket_handle_lost', endpoint='original_basket_pull_and_stable_placement',
        preserved=('basket_strictly_partly_open', 'rail_still_engaged', 'handle_not_held', 'cake_supported_in_basket'),
        randomization='original_2_3_sampler_and_measured_partial_stroke'),
    'sofa': dict(number=8, title='Sofa item missed grasp', base_task='sofa_to_coffee_table',
        failure_kind='empty_grasp', endpoint='original_selected_object_to_coffee_table',
        preserved=('target_supported_on_sofa', 'backrest_and_armrest_clearance', 'empty_attempted_grasp'),
        randomization='existing_approved_sofa_target_domain_and_station_ranges'),
}


def definition(name):
    if name not in CASES:
        raise ValueError('Unknown failure recovery variant: ' + str(name))
    return deepcopy(CASES[name])


def generator(name, seed, purpose):
    definition(name)
    if type(seed) is not int or not 1 <= seed <= 4294967295:
        raise ValueError('Positive uint32 seed required')
    key = f'grasp_failure_recovery.v1:{name}:{seed}:{purpose}'
    return random.Random(int.from_bytes(hashlib.sha256(key.encode()).digest(), 'little'))


def select(name, seed, purpose, candidates):
    if not candidates:
        raise ValueError('No source-backed candidates: ' + name + ':' + purpose)
    index = generator(name, seed, purpose).randrange(len(candidates))
    return deepcopy(candidates[index]), dict(seed=seed, candidate_index=index,
        candidate_count=len(candidates), selection='seeded_source_candidate', resampled=False)


def validate_recovery_start(name, facts):
    """Native reader facts only. No thresholds invented by this coordinator."""
    required = definition(name)['preserved']
    if set(facts) != set(required) or any(type(v) is not bool for v in facts.values()):
        raise ValueError('Complete native evidence for the selected failure start is required')
    if not all(facts.values()):
        raise ValueError('Invalid failure start: ' + repr(facts))
    return dict(accepted=True, initial_awarded_points=0, inherited_progress=False)
