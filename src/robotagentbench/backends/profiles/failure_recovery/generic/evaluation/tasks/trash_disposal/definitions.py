"""Approved disposal-only semantics; no guessed public IDs or physics knobs."""
from dataclasses import dataclass

MAXIMUM_SCORE = 3.0
HOLD_SECONDS = 1.0
TIMEOUT_SECONDS = 180.0
RULE_VERSION = 'trash_disposal.process.v1'


@dataclass(frozen=True)
class Definition:
    kind: str
    source_id: str
    target: str
    destination: str
    instruction: str


DEFINITIONS = {
    kind: Definition(kind, source, '/World/Assets/'+target,
                     '/World/Assets/'+destination, instruction)
    for kind, source, target, destination, instruction in (
        ('paper', '7.2', 'PaperBall', 'TrashBinRed', '将手中使用过的纸巾投入普通垃圾桶并松手。'),
        ('snack', '7.4', 'Family7SnackBox', 'TrashBinRed', '将手中空食品包装投入普通垃圾桶并松手。'),
        ('can', '7.8', 'EmptyCan', 'TrashBinTeal', '将手中空易拉罐投入可回收垃圾桶并松手。'),
        ('cup', '7.10', 'Task29Cup0', 'TrashBinRed', '将手中空一次性杯子投入普通垃圾桶并松手。'),
    )
}


def definition_for(kind):
    if not isinstance(kind, str) or kind not in DEFINITIONS:
        raise ValueError('Unknown disposal variant')
    return DEFINITIONS[kind]


def step_budget(control_dt_s):
    """Convert the approved duration using the existing caller's control dt."""
    import math
    if type(control_dt_s) not in (float, int) or not math.isfinite(control_dt_s) or control_dt_s <= 0:
        raise ValueError('Invalid control period')
    value = TIMEOUT_SECONDS / control_dt_s
    steps = round(value)
    if steps < 1 or not math.isclose(steps*control_dt_s, TIMEOUT_SECONDS, abs_tol=1e-9):
        raise ValueError('180 seconds must align with the existing control clock')
    return steps
