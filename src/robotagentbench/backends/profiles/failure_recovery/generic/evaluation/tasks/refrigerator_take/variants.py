"""Explicit trusted fridge targets; API IDs never select arbitrary prim paths."""
from dataclasses import dataclass
from .initial_state import TARGET


@dataclass(frozen=True)
class Variant:
    task_id: str
    kind: str
    name: str
    root: str
    body: str
    source_task: str

    @property
    def instruction(self):
        return f'打开冰箱，抓住并完整取出{self.name}，关闭冰箱，最后继续持有{self.name}至少五秒。'


VARIANTS = {
    '26-1': Variant('26-1', 'beef', '牛排盒', TARGET, TARGET, '9.1'),
    '26-2': Variant('26-2', 'food', '剩饭盒', '/World/Assets/BinFoodR',
                    '/World/Assets/BinFoodR/Asset', '9.3'),
    '26-3': Variant('26-3', 'salad', '沙拉盒', '/World/Assets/SaladBoxR',
                    '/World/Assets/SaladBoxR/Asset', '9.5'),
}


def variant_for(task_id):
    try:
        return VARIANTS[task_id]
    except KeyError:
        raise ValueError('Unknown refrigerator task: '+str(task_id)) from None
