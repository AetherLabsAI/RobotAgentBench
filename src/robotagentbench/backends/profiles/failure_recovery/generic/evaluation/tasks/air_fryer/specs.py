"""Static identity of the Task 2 (air fryer) family. Rules and constants live in DEFINITION_TABLE.md and process_rules.py.

Data subtasks (keys of DEFINITION_TABLE.md, the data producer's numbering) and the server runtime ids that evaluate them:

    key   data subtask                 runtime id        bread_variant      target cake colour
    2.1   黄色蛋糕放入炸篮               air_fryer_load    no_sugar           yellow (plain_cake_a)
    2.2   白色蛋糕放入炸篮               air_fryer_load    sugar              white  (frosted_cake)
    2.3   拉出炸篮                      air_fryer_pull    sugar / no_sugar   white / yellow cake in the basket
    2.5   把面包从炸篮里拿出来           air_fryer_take    sugar / no_sugar   white / yellow cake in the basket

The colour <-> sugar mapping is the data producer's (frosted = white = sugar, plain = yellow = no sugar; README). In every
source scene the task cake is the prim `Cakes/PlateCakeA` and the other cake is `Cakes/PlatePlainCakeA`; the colour is
the referenced asset, never the prim name.
"""
from dataclasses import dataclass
import json
from pathlib import Path

TASK = Path(__file__).resolve().parent
REPO = TASK.parents[2]
EPISODES = TASK / 'resources/source_episodes'
PRODUCER_ASSETS = TASK / 'resources/producer_assets_20260922'
APPROVED_FRYER = REPO / 'resources/shared_assets/air_fryer_task63_visualfinish_v5'
EVALUATOR_ID = 'air_fryer.process.v2'
RESET_SCHEMA = 'arena.air_fryer.reset.v2'

ROBOT = '/World/Robot'
STATION = '/World/Task66_67'
FRYER_PLACEMENT = STATION + '/AirFryerPlacement'
FRYER = FRYER_PLACEMENT + '/AirFryer'
CENTERED = FRYER + '/Centered'
BODY, TRAY = CENTERED + '/Body', CENTERED + '/Tray'
JOINT = CENTERED + '/Joints/DrawerSlideOldParams'
BASKET_FLOOR = TRAY + '/Collision/BasketFloor'
TARGET = STATION + '/Cakes/PlateCakeA'           # the cake the task is about (target of 2.1/2.2, basket cake of 2.3/2.5)
OTHER = STATION + '/Cakes/PlatePlainCakeA'       # the other cake (distractor of 2.1/2.2, plate cake of 2.3/2.5)
INACTIVE_CAKES = (STATION + '/Cakes/PlateCakeB', STATION + '/Cakes/PlatePlainCakeB')
PLATE_PLACEMENT = STATION + '/DisplayPlatePlacement'
PLATE = PLATE_PLACEMENT + '/Plate'
PLATE_SUPPORT = STATION + '/SourcePlateSupport'
COUNTER = '/World/Background/Shell/CounterTop_17'
FLOOR = '/World/Background/Shell/Floor'
DEACTIVATED_HOUSE = ('/World/Assets/DiningTableMicrowave', '/World/Assets/MicrowaveBeefbox', '/World/Assets/MicrowavePlateStand')
COLOURS = {'frosted_cake': 'white', 'plain_cake_a': 'yellow'}
VARIANT_OF_COLOUR = {'white': 'sugar', 'yellow': 'no_sugar'}
COLOUR_OF_VARIANT = {v: k for k, v in VARIANT_OF_COLOUR.items()}
FRYER_VARIANTS = {   # producer fryer asset of each group -> how it differs from the approved v5
    'approved_v5': dict(file=APPROVED_FRYER / 'air_fryer/usd/air_fryer_chefman_visualfinish_v5.usda', rail='fixed 0-0.145 m'),
    'release145_v2': dict(file=PRODUCER_ASSETS / 'release145_v2/air_fryer_release145_v2.usda', rail='limit 0.146 m; joint released at 0.145 m stroke'),
    'guided_unlimited_v1': dict(file=PRODUCER_ASSETS / 'guided_unlimited_v1/air_fryer_guided_unlimited_v1.usda', rail='limit inf; joint disabled (free basket)'),
}


@dataclass(frozen=True)
class Spec:
    key: str                  # data subtask
    runtime_id: str
    title: str
    operation: str            # insert | pull | take
    fryer_variant: str
    rail_release: bool        # 2.3: the joint is released when the stroke reaches 0.145 m (producer mechanism)
    basket_free: bool         # 2.5: joint disabled from the start, the basket is a free rigid body on the counter
    target_in_basket: bool    # initial state: the task cake is inside the basket (2.3 / 2.5) or on the plate (2.1 / 2.2)
    fixed_variant: str | None # 2.1 / 2.2 have one colour; 2.3 / 2.5 take it from the episode


SPECS = {
    '2.1': Spec('2.1', 'air_fryer_load', '黄色蛋糕放入炸篮', 'insert', 'approved_v5', False, False, False, 'no_sugar'),
    '2.2': Spec('2.2', 'air_fryer_load', '白色蛋糕放入炸篮', 'insert', 'approved_v5', False, False, False, 'sugar'),
    '2.3': Spec('2.3', 'air_fryer_pull', '拉出炸篮', 'pull', 'release145_v2', True, False, True, None),
    '2.5': Spec('2.5', 'air_fryer_take', '把面包从炸篮里拿出来', 'take', 'guided_unlimited_v1', False, True, True, None),
}
KEYS = tuple(SPECS)
RUNTIME_IDS = ('air_fryer_load', 'air_fryer_pull', 'air_fryer_take')


def spec(key):
    if key not in SPECS:
        raise ValueError('Not an air fryer data subtask: ' + str(key))
    return SPECS[key]


def key_for(runtime_id, variant):
    """Runtime id + bread_variant -> data subtask key."""
    if variant not in VARIANT_OF_COLOUR.values():
        raise ValueError('bread_variant must be sugar or no_sugar')
    for s in SPECS.values():
        if s.runtime_id == runtime_id and (s.fixed_variant is None or s.fixed_variant == variant):
            return s.key
    raise ValueError('Unknown runtime id: ' + str(runtime_id))


def episodes(key):
    """Packaged source episodes of one data subtask: the five of the review selection first (the first one is the fixed
    initial state), then every other packaged episode in name order."""
    order = json.loads((TASK / 'references/source_sim_videos/selection.json').read_text())[spec(key).key]['episodes']
    selected = [e['label'] for e in order if (EPISODES / key / e['label'] / 'episode.json').is_file()]
    others = sorted(p.parent.name for p in (EPISODES / key).glob('*/episode.json')) if (EPISODES / key).is_dir() else []
    return tuple(selected + [e for e in others if e not in selected])


def episode_manifest(key, label):
    return json.loads((EPISODES / spec(key).key / label / 'episode.json').read_text())


def variant_of(key, label):
    s = spec(key)
    return s.fixed_variant or VARIANT_OF_COLOUR[episode_manifest(key, label)['target_colour']]
