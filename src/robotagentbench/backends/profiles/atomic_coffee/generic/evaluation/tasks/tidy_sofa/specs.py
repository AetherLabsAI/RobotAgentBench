"""Per-API bindings for the tidy_sofa family (VLA_task.csv 任务编号 6.1 … 6.17).

Identity, target/scenery prims, destination geometry, time limits and point values as
approved (or drafted) in apis/<api>/README.md. No physics parameters, no Isaac imports.
"""
from dataclasses import dataclass, field

ROBOT = '/World/Robot'
ASSETS = '/World/Assets'
SEAT_OVERLAY = ASSETS + '/SeatOverlay'          # invisible sofa seat collision slab, top z = 0.452
SOFA = ASSETS + '/Sofa'
COFFEE_TABLE = ASSETS + '/CoffeeTable'
DRAWER = COFFEE_TABLE + '/black_box_1'
DRAWER_JOINT = COFFEE_TABLE + '/black_box_1_slide'
SMALL_TABLE = ASSETS + '/SmallCoffeeTable'
TV_TABLE = ASSETS + '/TvTable'
COAT_RACK = ASSETS + '/CoatRack'
BASKET = ASSETS + '/TableclothBasket'
CUSHION = ASSETS + '/CushionOnBook'

# Drawer slide contract (public scene layer livingroom_drawer_slide.usda, ported from the
# L1 2026-08-07 drawer_articulation.py that produced the 6.5/6.15 source episodes).
DRAWER_INITIAL_LIP_M = 0.020        # drawer starts 20 mm proud of the flush/closed pose
DRAWER_MAX_TRAVEL_M = 0.26          # from the flush pose
DRAWER_OPEN_M = 0.10                # stage ① threshold (from flush)  – source pulls reach 0.209
DRAWER_CLOSED_M = 0.03              # stage ③ threshold (from flush)  – source returns to 0.006–0.026 (1/50: 0.047)


@dataclass(frozen=True)
class TaskSpec:
    api_id: str
    benchmark: str
    kind: str                       # pick | place | drawer | hang | basket
    title: str
    instruction: str
    target: str                     # target prim path
    small_task_ids: tuple           # accepted source small task ids / runtime labels
    source_benchmark_dir: str       # references/l1_task6_20260914/l1/<dir>
    time_limit_s: float
    points: dict                    # stage name -> points
    maximum_score: int
    scenery: tuple = ()             # extra prims activated at their source frame-0 pose
    support: str = ''               # pick: support the target must rest on initially
    destination: str = ''           # place/basket/hang/drawer: destination prim (surface or container)
    furniture: tuple = ()           # prims whose robot contact is a deduction (support/destination contact exempt)
    zone_xy: tuple = ()             # place: (cx, cy, hx, hy) destination surface zone (world, canonical layout)
    zone_top_z: float = 0.0
    in_hand: bool = False           # episode starts with the target held
    hold_s: float = 1.0
    lift_m: float = 0.05
    edge_margin_m: float = 0.0      # 19-1 易取位置: distance from table edges
    notes: str = ''


def _pick(api, bm, title, target, small, srcdir, scenery=(), limit=30.0, notes=''):
    return TaskSpec(api_id=api, benchmark=bm, kind='pick', title=title, instruction=title + '。',
                    target=ASSETS + '/' + target, small_task_ids=tuple(small), source_benchmark_dir=srcdir,
                    time_limit_s=limit, points=dict(grasp=3, lift=2), maximum_score=5,
                    scenery=tuple(ASSETS + '/' + s for s in scenery), support=SEAT_OVERLAY, furniture=(SOFA,), notes=notes)


SPECS = {
    '15-1': _pick('15-1', '6.1', '抓取沙发上的书籍', 'Book', ('13', '12'), 'benchmark_6_1', scenery=('CushionOnBook',),
                  notes='书半压在靠枕下、伸出沙发前沿约 4 cm；靠枕为可动刚体'),
    '13-6': _pick('13-6', '6.4', '抓取账单', 'Note', ('11',), 'benchmark_6_4', scenery=('CushionOnBook',),
                  notes='账单半压在靠枕下、伸出沙发前沿约 6 cm'),
    '15-2': _pick('15-2', '6.6', '抓取沙发上的帽子', 'BaseballCap', ('32',), 'benchmark_6_6', notes='帽子原点远离网格，判定用 AABB 中心'),
    '6-3': _pick('6-3', '6.8', '抓取沙发上的电视遥控器（含靠枕下）', 'Remote', ('31',), 'benchmark_4_13'),
    '13-7': _pick('13-7', '6.11', '抓取空一次性杯子', 'PaperCup', ('33',), 'benchmark_6_11'),
    '6-2': _pick('6-2', '6.14', '抓取沙发上的钱包（含靠枕下）', 'Wallet', ('53',), 'benchmark_4_7'),
    '6-1': _pick('6-1', '6.17', '抓取沙发上的墨镜盒（含靠枕下）', 'GlassesCase', ('91',), 'benchmark_4_1'),
    '10-4': TaskSpec('10-4', '6.2', 'place', '将书籍放到茶几', '将书籍放到茶几。', ASSETS + '/Book', ('20', '27'), 'benchmark_6_2', 20.0,
                     dict(place=2, settle=2), 4, destination=COFFEE_TABLE, furniture=(COFFEE_TABLE,),
                     zone_xy=(-1.950, -0.121, 0.40, 0.60), zone_top_z=0.45, in_hand=True),
    '10-5': TaskSpec('10-5', '6.3', 'place', '将书籍放到小茶几', '将书籍放到小茶几。', ASSETS + '/Book', ('benchmark_tidy_sofa_6_3',), 'benchmark_6_3', 20.0,
                     dict(place=2, settle=2), 4, destination=SMALL_TABLE, furniture=(SMALL_TABLE,),
                     zone_xy=(-2.115, -1.987, 0.484, 0.635), zone_top_z=0.457, in_hand=True),
    '19-1': TaskSpec('19-1', '6.9', 'place', '将电视遥控器放到茶几易取位置', '将电视遥控器放到茶几易取位置。', ASSETS + '/Remote', ('benchmark_tidy_sofa_6_9',), 'benchmark_6_9', 20.0,
                     dict(place=2, settle=2), 4, destination=COFFEE_TABLE, furniture=(COFFEE_TABLE,),
                     zone_xy=(-1.950, -0.121, 0.40, 0.60), zone_top_z=0.45, in_hand=True, edge_margin_m=0.05),
    '10-3': TaskSpec('10-3', '6.10', 'place', '将电视遥控器放到电视柜', '将电视遥控器放到电视柜。', ASSETS + '/Remote', ('benchmark_tidy_sofa_6_10',), 'benchmark_6_10', 30.0,
                     dict(place=2, settle=2), 4, destination=TV_TABLE, furniture=(TV_TABLE, ASSETS + '/Tv'),
                     zone_xy=(-3.710, -0.299, 0.15, 0.84), zone_top_z=0.45, in_hand=True),
    '16-1': TaskSpec('16-1', '6.5', 'drawer', '将账单放入茶几抽屉', '将账单放入茶几抽屉。', ASSETS + '/Note', ('28', 'benchmark_tidy_sofa_6_5'), 'benchmark_6_5', 40.0,
                     dict(open=2, place=3, close=1), 6, destination=DRAWER, furniture=(COFFEE_TABLE,), in_hand=True),
    '16-2': TaskSpec('16-2', '6.15', 'drawer', '将钱包放到茶几抽屉', '将钱包放到茶几抽屉。', ASSETS + '/Wallet', ('92', 'benchmark_tidy_sofa_6_15'), 'benchmark_6_15', 40.0,
                     dict(open=1, place=2, close=1), 4, destination=DRAWER, furniture=(COFFEE_TABLE,), in_hand=True),
    '17-1': TaskSpec('17-1', '6.7', 'hang', '将帽子挂在衣架', '将帽子挂在衣架。', ASSETS + '/BaseballCap', ('70', 'benchmark_tidy_sofa_6_7'), 'benchmark_6_7', 20.0,
                     dict(hang=2, settle=2), 4, destination=COAT_RACK, furniture=(COAT_RACK,), in_hand=True),
    '14-4': TaskSpec('14-4', '6.16', 'basket', '将钱包放到指定收纳区', '将钱包放到指定收纳区。', ASSETS + '/Wallet', ('benchmark_tidy_sofa_6_16',), 'benchmark_6_16', 20.0,
                     dict(place=2, settle=2), 4, destination=BASKET, furniture=(BASKET, COFFEE_TABLE),
                     zone_xy=(-2.24, -0.875, 0.16, 0.12), zone_top_z=0.26, in_hand=True),
}
ORDER = ['15-1', '10-4', '10-5', '13-6', '16-1', '15-2', '17-1', '6-3', '19-1', '10-3', '13-7', '6-2', '16-2', '14-4', '6-1']


def spec(api_id):
    if api_id not in SPECS:
        raise ValueError('Unknown tidy_sofa API: ' + str(api_id))
    return SPECS[api_id]
