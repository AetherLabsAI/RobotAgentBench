"""Approved identity mapping, scoring, fixed base and task-local perturbations."""
from pathlib import Path

TASK = Path(__file__).resolve().parent
SUPPORT = '/World/Assets/SmallCoffeeTable'
ROBOT = '/World/Robot'
SPECS = {
    'wallet': dict(api_id='11-1', benchmark='4.9', source_runtime='60',
                   target='/World/Assets/Wallet', instruction='抓取小茶几表面的钱包。'),
    'remote': dict(api_id='11-2', benchmark='4.14', equivalent_benchmark='7.17',
                   source_runtime='109', target='/World/Assets/Remote',
                   instruction='抓取小茶几表面的电视遥控器。'),
}


def spec(kind):
    if kind not in SPECS:
        raise ValueError('Unknown small-table object variant: ' + str(kind))
    return dict(SPECS[kind])


def fresh_output(output):
    out = Path(output).resolve()
    root = (TASK / 'artifacts').resolve()
    if root != TASK / 'artifacts' or not out.is_relative_to(root) or out == root:
        raise ValueError('Output must be inside this task artifacts directory')
    if out.exists():
        raise FileExistsError(out)
    return out
