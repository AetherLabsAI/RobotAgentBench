"""Category-only atomic robot initialization; no scene or physics writes."""
import json
from pathlib import Path
from .bank import BANK_DIR

HERE = Path(__file__).resolve().parent


def initial_for_kind(kind):
    data = json.loads((HERE / 'atomic_initial_profiles.json').read_text())
    if kind not in data['profiles']:
        raise ValueError('Expected drink or snack variant')
    profile = data['profiles'][kind]
    return dict(
        policy=data['schema'], variant=kind,
        source_apis=[s['api_id'] for s in profile['sources']],
        joint_positions=profile['joint_positions'],
        recorded_source_robot=profile['source_robot'],
        station_mapper='short_tasks.tasks.stock_wooden_tray.initial_station.world_pose',
        station_rule='Map the source station relative to the actual shelf front; never copy source world XY into another layout.',
        target_dependent=False, native_validated_here=False)


def initial_for_question(question_id):
    questions = json.loads((BANK_DIR / 'public_questions.json').read_text())['questions']
    matches = [q for q in questions if q['id'] == question_id]
    if len(matches) != 1:
        raise ValueError('Unique public question ID required')
    # No private answer or target identity is used to select a robot pose.
    return initial_for_kind(matches[0]['kind'])


def map_station(kind, front_midpoint_xy, inward_heading_rad, perturbation):
    """Use the existing approved local-frame station ranges unchanged.

    Returns XY/yaw plus atomic joints. The scene owner must retain its validated
    floor/root Z and run collision/settling checks before applying this plan.
    """
    from short_tasks.tasks.stock_wooden_tray.initial_station import world_pose
    result = initial_for_kind(kind)
    result['mapped_station'] = world_pose(front_midpoint_xy, inward_heading_rad, perturbation)
    return result
