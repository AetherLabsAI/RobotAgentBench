"""One task kind, fourteen product parameters; no new public API ID."""
from dataclasses import dataclass
from types import MappingProxyType
from evaluation.tasks.shelf_pick.targets import SPECS

TASK_KIND = 'snack_drink_to_tray'
PRODUCT_NAMES = (
    ('cayman_jacked', '酒精饮料'), ('gatorade_m', '橙色运动饮料'),
    ('gatorade_t', '蓝色运动饮料'), ('starbucks', '含咖啡因饮料'),
    ('good2grow', '儿童果汁'), ('grapefruit_can', '气泡水'),
    ('coke_c3', '可乐'), ('sprite_c1', '雪碧'), ('chomps_r', '牛肉棒'),
    ('pring_bbq', '薯片'), ('oreo_gf', '奥利奥饼干'),
    ('honeymaid', 'Honey Maid 饼干'), ('zbar_t', '蛋白棒'), ('skit_littles', '糖果'),
)


@dataclass(frozen=True)
class Product:
    key: str
    label: str
    pick_api: str
    place_api: str
    tier: str


PRODUCTS = MappingProxyType({key: Product(key, label, f'4-{i}', f'5-{i}',
                            'tier3' if i <= 8 else 'tier2')
                            for i, (key, label) in enumerate(PRODUCT_NAMES, 1)})
for product in PRODUCTS.values():
    if SPECS[product.pick_api].target.rsplit('/', 1)[-1] != product.key:
        raise RuntimeError('Upstream product mapping changed; review before use')


def parameter_schema():
    return dict(type='object', required=['product'], additionalProperties=False,
                properties=dict(product=dict(type='string', enum=list(PRODUCTS))))


def normalize_parameters(parameters):
    if not isinstance(parameters, dict) or set(parameters) != {'product'}:
        raise ValueError('Exactly one required parameter: product')
    value = parameters['product']
    if not isinstance(value, str) or value not in PRODUCTS:
        raise ValueError('Unknown snack/drink product')
    return dict(product=value)


def instruction(parameters):
    p = PRODUCTS[normalize_parameters(parameters)['product']]
    return f'从货架拿取一件{p.label}，抬离层板连续持有3秒，然后放入茶几上的木托盘，松手并连续放稳1秒。'


def scene_requirements(parameters):
    p = PRODUCTS[normalize_parameters(parameters)['product']]
    return dict(task_kind=TASK_KIND, task_parameters=dict(product=p.key),
                required_categories=[p.key], count=1, source_support=p.tier,
                destination_role='coffee_table_wooden_tray',
                initialization_owner='unified_scene', create_objects=False,
                instruction=instruction(parameters))

