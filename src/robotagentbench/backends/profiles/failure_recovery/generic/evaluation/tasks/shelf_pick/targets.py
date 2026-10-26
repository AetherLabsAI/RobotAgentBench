"""Per-API target bindings for the shared shelf-pick family (4-1 … 4-14).

Only identities and approved instruction text live here. Scoring arithmetic,
detection tolerances and physics stay shared and unchanged across APIs.
"""
import re
from dataclasses import dataclass, field

SHELF_ITEMS = '/World/Assets/ShelfItems'
TIER2_ITEMS = '/World/Assets/Tier2Items'
TIER3 = '/World/Assets/Tier3Overlay'
TIER2 = '/World/Assets/Tier2Overlay'
# 数据方按需复制货架商品时用的后缀，例如 `oreo_gf_copy_01`。
COPY_SUFFIX = re.compile(r'_copy_\d+$')


@dataclass(frozen=True)
class TargetSpec:
    api_id: str
    description: str
    instruction: str
    small_task_ids: tuple[str, ...]
    targets: tuple[str, ...]          # any of these counts as the correct object (4-8: two Sprite cans)
    support: str                      # original support slab; only its top face exempts target contact
    stock_root: str                   # stock layer whose children are activated for this task
    snapshot: str                     # resources/<file> with source target/robot pose and fixed neighbors
    policy: str                       # resources/<file> with the randomized front-row packing policy
    benchmark: str = ''
    notes: tuple[str, ...] = field(default_factory=tuple)
    # World-frame offset (m) applied to the recorded source target pose in the fixed
    # initial state; only for recorded poses that are physically invalid on the
    # collision-enabled rack (e.g. a box overhanging the front board edge onto the
    # bracket lip). Recorded in initial-state.json; scoring/physics untouched.
    target_offset_m: tuple[float, float, float] = (0., 0., 0.)
    external_target: bool = False

    @property
    def target(self):
        """Primary target: the instance the source initial state places."""
        return self.targets[0]

    @property
    def alternates(self):
        return self.targets[1:]

    def is_target(self, path):
        """接受本体，也接受数据方的复制件 `<product>_copy_NN`。

        用户 2026-09-24 定：复制件算同一个任务的目标。它与本体引用同一份数据方发布资产、
        质量/碰撞/摩擦逐项相同，数据方也把这些回合归在同一个 benchmark 下；先例是 4-8
        早就把两罐雪碧都算作可接受目标。这样 4-x 的可复现来源回合从 161 增到 177 / 210。
        """
        return path in self.targets or self._copy_of_a_target(path)

    def _copy_of_a_target(self, path):
        base = COPY_SUFFIX.sub('', path)
        return base != path and base in self.targets


def _drink(api, description, small, prim, benchmark, snapshot=None, targets=None, notes=()):
    return TargetSpec(api, description, f'{description}，并抬离层板稳定持有。', tuple(small),
                      targets or (SHELF_ITEMS + '/' + prim,), TIER3, SHELF_ITEMS,
                      snapshot or f'source_initial_{api.replace("-", "_")}.json', 'shelf_policy.json', benchmark, tuple(notes))


def _snack(api, description, small, prim, benchmark, notes=(), target_offset_m=(0., 0., 0.)):
    return TargetSpec(api, description, f'{description}，并抬离层板稳定持有。', tuple(small),
                      (TIER2_ITEMS + '/' + prim,), TIER2, TIER2_ITEMS,
                      f'source_initial_{api.replace("-", "_")}.json', 'tier2_policy.json', benchmark, tuple(notes), target_offset_m)


SPECS = {
    '4-1': _drink('4-1', '从零食架抓取酒精饮料', ('85',), 'cayman_jacked', '3.1', snapshot='source_initial_v1.json'),
    '4-2': _drink('4-2', '从零食架抓取橙色运动饮料', ('6',), 'gatorade_m', '3.3'),
    '4-3': _drink('4-3', '从零食架抓取蓝色运动饮料', ('9', '22'), 'gatorade_t', '3.5'),
    '4-4': _drink('4-4', '从零食架抓取含咖啡因饮料', ('48',), 'starbucks', '3.7',
                  notes=('baseline starbucks prim has no rigid body or collision; option A (user 2026-09-14) audited block composed by assets/house_scene_l1/demo_sim/shelf_pick_starbucks_physics.usda',)),
    '4-5': _drink('4-5', '从零食架抓取儿童果汁', ('47',), 'good2grow', '3.9'),
    '4-6': _drink('4-6', '从零食架抓取气泡水', ('50',), 'grapefruit_can', '3.11'),
    '4-7': _drink('4-7', '从零食架抓取可乐', ('8',), 'coke_c3', '3.13',
                  notes=('diet_coke stays a neighbor; grasping it is a wrong-target failure (user decision 2026-09-14)',)),
    '4-8': _drink('4-8', '从零食架抓取雪碧', ('10', '21'), 'sprite_c1', '3.15',
                  targets=(SHELF_ITEMS + '/sprite_c1', SHELF_ITEMS + '/sprite_c2'),
                  notes=('either Sprite can counts; the first confirmed grasp commits the target, the other can is then a neighbor (user decision 2026-09-14)',)),
    '4-9': _snack('4-9', '从零食架抓取牛肉棒', ('35',), 'chomps_r', '3.19'),
    '4-10': _snack('4-10', '从零食架抓取薯片', ('36',), 'pring_bbq', '3.21'),
    '4-11': _snack('4-11', '从零食架抓取奥利奥饼干', ('40',), 'oreo_gf', '3.23',
                   notes=('recorded pose overhangs the tier-2 front edge by ~2 mm and rests on the bracket lip (initial shelf_collision, batch-w120 fixed-probe 58376fae); moved 1 cm deeper (+y) 2026-09-14',),
                   target_offset_m=(0., .01, 0.)),
    '4-12': _snack('4-12', '从零食架抓取花生酱饼干', ('37',), 'honeymaid', '3.25',
                   notes=('source registry and real recordings bind benchmark 3.25 to the Honey Maid box (user decision 2026-09-14)',)),
    '4-13': _snack('4-13', '从零食架抓取蛋白棒', ('39',), 'zbar_t', '3.27'),
    '4-14': _snack('4-14', '从零食架抓取糖果', ('38',), 'skit_littles', '3.29'),
    '4-15': TargetSpec('4-15', '从零食架抓取电视遥控器',
        '从零食架抓取电视遥控器，并抬离层板稳定持有。', (),
        ('/World/Assets/Remote',), TIER3, SHELF_ITEMS,
        'source_initial_v1.json', 'shelf_policy.json', '4.16',
        notes=('Reuse existing shelf robot station; remote resting orientation from the small-table reference; physical acceptance pending',),
        external_target=True),
}
DEFAULT = SPECS['4-1']


def spec(api_id):
    if api_id not in SPECS:
        raise ValueError('Unknown shelf-pick API: ' + str(api_id))
    return SPECS[api_id]
