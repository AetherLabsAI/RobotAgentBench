"""Task identities only; no scoring, physics or runtime-registration defaults."""
from dataclasses import dataclass

@dataclass(frozen=True)
class Variant:
    name: str
    benchmark_id: str
    api_id: str
    instruction: str

VARIANTS = {
    "beef_box": Variant("beef_box", "9.2", "27-1",
        "打开微波炉，将手中的牛肉盒放入炉内，松开夹爪，再关好炉门。"),
    "leftover_box": Variant("leftover_box", "9.4", "27-2",
        "打开微波炉，将手中的剩饭盒放入炉内，松开夹爪，再关好炉门。"),
}

def variant_for(name):
    if not isinstance(name, str) or name not in VARIANTS:
        raise ValueError("Unknown microwave_put payload variant")
    return VARIANTS[name]
