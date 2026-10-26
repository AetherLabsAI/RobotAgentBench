"""Static identity of the four APIs. Rules and constants live in apis/<api>/README.md and process_rules.py."""
from dataclasses import dataclass
import json
from pathlib import Path

TASK = Path(__file__).resolve().parent
REPO = TASK.parents[2]
EPISODES = TASK / 'resources/source_episodes'
EVALUATOR_ID = 'coffee_machine.process.v2'
RESET_SCHEMA = 'arena.coffee_machine.reset.v1'

ROBOT = '/World/Robot'
# public shared coffee assets (resources/shared_assets/coffee_v1), composed by arena_api.l1_scene under Task75
STATION = '/World/Task75'
MACHINE, CUP, BAG = STATION + '/Machine', STATION + '/Cup', STATION + '/Bag'
MACHINE_BODY, BUTTON = MACHINE + '/Body', MACHINE + '/Button'
CUP_SUPPORT = MACHINE_BODY + '/Collision/CupSupport'
TRAY_COLLIDER = MACHINE_BODY + '/Collision/Tray'
BEANS, LIQUID = STATION + '/CoffeeBeans', '/World/CoffeeLiquid'
COUNTER = '/World/Background/Shell/CounterTop_17'
FLOOR = '/World/Background/Shell/Floor'
# prim paths of the same objects inside a packaged source episode (coffee_objects.usdc)
PACKAGED = dict(machine='/World/CoffeeTaskPreview/Machine', cup='/World/CoffeeTaskPreview/Cup',
                bag='/World/CoffeeTaskPreview/BeanBag', beans='/World/PourBeans', liquid='/World/CoffeeLiquid')
CUP_STYLES = ('cream_classic', 'blue_compact', 'green_tall')
SHARED_ASSETS = REPO / 'resources/shared_assets/coffee_v1'
# data producer's robot station for the whole family (identical in all 200 source episodes)
ROBOT_BASE_XYZ = (1.762096825, 0.091750471, 0.0)


@dataclass(frozen=True)
class Spec:
    api: str
    task_no: str
    name: str
    time_limit_s: float
    maximum_score: int
    target: str               # the object the robot manipulates
    has_beans: bool
    has_liquid: bool
    has_device: bool          # button + dispensing logic is live


SPECS = {
    '1-1': Spec('1-1', '1.1', '放杯子', 72., 5, CUP, False, False, False),
    '1-2': Spec('1-2', '1.2', '倾倒咖啡豆', 93., 6, BAG, True, False, False),
    '1-3': Spec('1-3', '1.3', '按按钮', 67., 5, BUTTON, True, True, True),
    '1-4': Spec('1-4', '1.4', '取杯子', 83., 5, CUP, False, True, False),
}
APIS = tuple(SPECS)


def spec(api):
    if api not in SPECS:
        raise ValueError('Not a coffee machine API: ' + str(api))
    return SPECS[api]


def episodes(api):
    """Packaged source episodes of one API: first the five of the review selection (the first one is the fixed initial
    state), then every other packaged episode (e.g. the left-hand batch of 2026-10-01) in name order."""
    order = json.loads((TASK / 'references/source_sim_videos/selection.json').read_text())[spec(api).api]['episodes']
    selected = [e['label'] for e in order if (EPISODES / api / e['label'] / 'episode.json').is_file()]
    others = sorted(p.parent.name for p in (EPISODES / api).glob('*/episode.json')) if (EPISODES / api).is_dir() else []
    return tuple(selected + [e for e in others if e not in selected])


def episode_manifest(api, label):
    path = EPISODES / spec(api).api / label / 'episode.json'
    return json.loads(path.read_text())
