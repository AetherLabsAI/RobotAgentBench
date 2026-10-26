"""Task-local shelf products; original all_assets stays read-only."""
from pathlib import Path
from openend.tasks.scene_requirements.shelf_inventory_policy import POOLS as BASE_POOLS, LABELS as BASE_LABELS

HERE = Path(__file__).resolve().parent
ADDED = {'tier3': ('red_bull_pink',), 'tier2': ('lindt','pringles_green')}
POOLS = {tier: tuple(names)+ADDED[tier] for tier,names in BASE_POOLS.items()}
LABELS = dict(BASE_LABELS, red_bull_pink='红牛莓果味能量饮料', lindt='Lindt 牛奶巧克力球', pringles_green='品客酸奶油洋葱味薯片')
NEW_PRODUCTS = frozenset(name for names in ADDED.values() for name in names)

def asset_path(name):
    if name not in LABELS:
        raise ValueError('Unknown shelf product '+name)
    return (HERE/'expanded_assets'/name/'asset.usda' if name in NEW_PRODUCTS
            else HERE.parents[2]/'all_assets'/name/'asset.usda')
