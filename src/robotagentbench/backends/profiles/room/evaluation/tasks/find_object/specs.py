"""Static identity of the Task 4 family (find_object, 寻找物品). Rules and constants live in DEFINITION_TABLE.md and process_rules.py.

Data subtasks (keys of DEFINITION_TABLE.md = data_requirement.csv numbering) evaluated by ONE runtime id `find_object` with
task_parameters.subtask (the CSV API ids 6-1 / 7-1 / 8-1 / … and their existing entries are left untouched; the mapping is the
user's decision, README):

    key    kind     target        support (where the target starts)                       destination
    4.1    pick     GlassesCase   sofa seat (all_assets/sofa_4seat)
    4.2    pick     GlassesCase   coffee table top (all_assets/coffee_table_with_drawers)
    4.3    pick     GlassesCase   small coffee table top (producer's "边柜")
    4.4    drawer   GlassesCase   coffee table drawer black_box_1 floor (drawer open at start)
    4.7    pick     Wallet        sofa seat
    4.8    pick     Wallet        coffee table top
    4.9    pick     Wallet        small coffee table top
    4.10   drawer   Wallet        coffee table drawer black_box_1 floor
    4.13   pick     Remote        sofa seat
    4.14   pick     Remote        small coffee table top
    4.16   pick     Remote        snack shelf tier (public w120 shelf, user 2026-10-05)
    4.17   place    Remote        in hand at the start                                     tv table top (all_assets/tv_table)
    4.6 / 4.12: no data, no runtime.
"""
from dataclasses import dataclass
import json
from pathlib import Path

TASK = Path(__file__).resolve().parent
REPO = TASK.parents[2]
ALL_ASSETS = REPO / 'all_assets'
EPISODES = TASK / 'resources/source_episodes'
RUNTIME_ID = 'find_object'
EVALUATOR_ID = 'find_object.process.v1'
RESET_SCHEMA = 'arena.find_object.reset.v1'

ROBOT = '/World/Robot'
STATION = '/World/FindObject'                 # task layer: producer furniture + objects referenced from all_assets
TARGET = STATION + '/Target'
DISTRACTORS = STATION + '/Distractors'        # /Distractors/<name>
# Furniture: the PUBLIC baseline prims, moved to the producer layout (the public obstacle / coffee-box policies require the public Sofa and
# CoffeeTable prims; the producer's furniture files are byte-identical to the public ones, all_assets/{sofa_4seat,coffee_table_with_drawers,
# small_coffee_table,tv_table} hold the verified reference copies). Drawer subtasks compose the public drawer slide layer (livingroom_drawer_slide.usda).
SOFA, COFFEE_TABLE, SMALL_TABLE, TV_TABLE = ('/World/Assets/Sofa', '/World/Assets/CoffeeTable', '/World/Assets/SmallCoffeeTable', '/World/Assets/TvTable')
DRAWER = COFFEE_TABLE + '/black_box_1'
DRAWER_JOINT = COFFEE_TABLE + '/black_box_1_slide'
DRAWER_SLIDE_LAYER = REPO / 'assets/house_scene_l1/demo_sim/livingroom_drawer_slide.usda'
SHELF = '/World/Assets/ShelvingUnit'          # public w120 snack shelf (baseline prim; user 2026-10-05: 零食架按加宽版来)
SHELF_SUPPORTS = ('/World/Assets/ShelvingUnit', '/World/Assets/Tier2Overlay', '/World/Assets/Tier3Overlay', '/World/ShelfSupports', '/World/ShelfSteelColliders')
TV = '/World/Assets/Tv'                       # public baseline prim, moved to the episode layout; 4.17 碰电视 −1
FLOOR = '/World/Background/Shell/Floor'
# public baseline prims replaced by the station (deactivated by the task layer); the public shelf stays
DEACTIVATED_HOUSE = ('/World/Assets/SeatOverlay', '/World/Assets/SeatRiser', '/World/Assets/GlassesCase', '/World/Assets/Wallet', '/World/Assets/Remote', '/World/Assets/CushionOnBook',
                     '/World/Assets/Book', '/World/Assets/Note', '/World/Assets/BaseballCap', '/World/Assets/PaperCup', '/World/Assets/PaperBall', '/World/Assets/EmptyCan',
                     '/World/Assets/PowerBank', '/World/Assets/Umbrella', '/World/Assets/CoffeeBeans', '/World/Assets/Tray')
# producer prim name -> all_assets entry (objects); furniture handled by FURNITURE_ASSETS
OBJECT_ASSETS = {'GlassesCase': 'glasses_case', 'Wallet': 'wallet', 'Remote': 'remote_tv', 'PaperCup': 'paper_cup', 'PaperBall': 'paper_ball', 'Book': 'book_red',
                 'Note': 'notebook_green', 'BaseballCap': 'baseball_cap', 'EmptyCan': 'empty_can', 'PowerBank': 'power_bank', 'Bottle': 'water_bottle'}
SHELF_ITEM_ASSETS = {n: n for n in ('cayman_jacked', 'gatorade_m', 'gatorade_t', 'starbucks', 'good2grow', 'grapefruit_can', 'coke_c3', 'sprite_c1',
                                     'chomps_r', 'pring_bbq', 'oreo_gf', 'honeymaid', 'zbar_t', 'skit_littles')}
FURNITURE_ASSETS = {'Sofa': ('sofa_4seat', SOFA), 'CoffeeTable': ('coffee_table_with_drawers', COFFEE_TABLE), 'SmallCoffeeTable': ('small_coffee_table', SMALL_TABLE),
                    'TvTable': ('tv_table', TV_TABLE)}      # all_assets reference copies of the producer furniture (not instanced: public prims are moved instead)
# public baseline prims whose world transform follows the episode layout (not replaced): kept at their public physics
LAYOUT_FOLLOWERS = ('/World/Assets/Sofa', '/World/Assets/CoffeeTable', '/World/Assets/SmallCoffeeTable', '/World/Assets/TvTable', '/World/Assets/ShelvingUnit', '/World/Assets/Tv', '/World/Assets/TrashBinRed', '/World/Assets/TrashBinTeal', '/World/Assets/TableclothBasket',
                    '/World/Assets/NightStand', '/World/Assets/Fireplace', '/World/Assets/Clock', '/World/Assets/Door')


@dataclass(frozen=True)
class Spec:
    key: str
    title: str
    kind: str                 # pick | drawer | place
    target_name: str          # producer prim name under /World/Assets
    target_asset: str         # all_assets entry
    support_name: str         # producer furniture prim name that supports the target at the start ('' for place)
    support_root: str         # prim path of that support in the evaluation scene
    destination_root: str     # place: where the target must end
    distractor_deduction: bool
    time_limit_s: float


SPECS = {
    '4.1': Spec('4.1', '在沙发座面找到并抓取墨镜盒（含靠枕下）', 'pick', 'GlassesCase', 'glasses_case', 'Sofa', SOFA, '', True, 121.),
    '4.2': Spec('4.2', '在茶几表面找到并抓取墨镜盒', 'pick', 'GlassesCase', 'glasses_case', 'CoffeeTable', COFFEE_TABLE, '', True, 116.),
    '4.3': Spec('4.3', '在边柜找到并抓取墨镜盒', 'pick', 'GlassesCase', 'glasses_case', 'SmallCoffeeTable', SMALL_TABLE, '', True, 111.),
    '4.4': Spec('4.4', '在茶几抽屉里找到并抓取墨镜盒', 'drawer', 'GlassesCase', 'glasses_case', 'CoffeeTable', DRAWER, '', True, 116.),
    '4.7': Spec('4.7', '在沙发座面找到并抓取钱包（含靠枕下）', 'pick', 'Wallet', 'wallet', 'Sofa', SOFA, '', True, 221.),
    '4.8': Spec('4.8', '在茶几表面找到并抓取钱包', 'pick', 'Wallet', 'wallet', 'CoffeeTable', COFFEE_TABLE, '', True, 76.),
    '4.9': Spec('4.9', '在小茶几找到并抓取钱包', 'pick', 'Wallet', 'wallet', 'SmallCoffeeTable', SMALL_TABLE, '', False, 81.),
    '4.10': Spec('4.10', '在茶几抽屉找到并抓取钱包', 'drawer', 'Wallet', 'wallet', 'CoffeeTable', DRAWER, '', True, 184.),
    '4.13': Spec('4.13', '在沙发座面找到并抓取电视遥控器（含靠枕下）', 'pick', 'Remote', 'remote_tv', 'Sofa', SOFA, '', True, 166.),
    '4.14': Spec('4.14', '在小茶几表面找到并抓取电视遥控器', 'pick', 'Remote', 'remote_tv', 'SmallCoffeeTable', SMALL_TABLE, '', False, 101.),
    '4.16': Spec('4.16', '在零食架上找到并抓取电视遥控器', 'pick', 'Remote', 'remote_tv', 'ShelvingUnit', SHELF, '', True, 81.),
    '4.17': Spec('4.17', '携带电视遥控器并放到电视柜上', 'place', 'Remote', 'remote_tv', '', '', TV_TABLE, False, 124.),
}
KEYS = tuple(SPECS)

# Task-owned object-side settings of the TARGET per data subtask (object collision representation / contact material only; the robot,
# finger pads, physics rate and solver stay the unified public ones). Each entry names its evidence. Applied by
# initial_state.write_station as overrides on the all_assets instance and recorded in initial.json and the reset contract.
TARGET_PHYSICS = {
    # user 2026-10-06: "api测之后4.3都用convexHull来跑". Evidence: debug-20261005/diag-4_3-* (convexDecomposition 0/3 -> convexHull 3/3, two episodes).
    '4.3': dict(approximation='convexHull', source='producer 4.3 scene (glasses case convexHull); user decision 2026-10-06'),
    # 2026-10-06: producer 4.10 wallet = WalletGripMat 1.4/1.2 + solver 64/16 (all five scenes). With SCENE_PHYSICS below, debug-20261005/
    # diag-4_10-ep3-wall-wallet and -ep4-wall-wallet 3/3 (wall friction alone: ep3 3/3, ep4 0/3; wallet alone: ep4 lifted then slipped).
    '4.10': dict(static_friction=1.4, dynamic_friction=1.2, solver_iterations=[64, 16], source='producer 4.10 scenes (WalletGripMat 1.4/1.2, solver 64/16); object side'),
}

# Scene-side (furniture) physics taken from the producer scenes of the same subtask; robot and physics-scene parameters stay public.
SCENE_PHYSICS = {
    # producer 4.4 and 4.10 scenes: DrawerWallMat static 1.2 / dynamic 1.0 (public slide layer 0.7 / 0.6)
    '4.4': dict(drawer_wall_friction=[1.2, 1.0], source='producer 4.4 scenes DrawerWallMat'),
    '4.10': dict(drawer_wall_friction=[1.2, 1.0], source='producer 4.10 scenes DrawerWallMat'),
}


def spec(key):
    if key not in SPECS:
        raise ValueError('Not a find_object data subtask with data: ' + str(key))
    return SPECS[key]


def episodes(key):
    """Packaged source episodes of one subtask: the five of the review selection first (the first one is the fixed initial state)."""
    order = json.loads((TASK / 'references/source_sim_videos/selection.json').read_text())[spec(key).key]['episodes']
    selected = [e['label'] for e in order if (EPISODES / key / e['label'] / 'episode.json').is_file()]
    others = sorted(p.parent.name for p in (EPISODES / key).glob('*/episode.json')) if (EPISODES / key).is_dir() else []
    return tuple(selected + [e for e in others if e not in selected])


def episode_manifest(key, label):
    return json.loads((EPISODES / spec(key).key / label / 'episode.json').read_text())
