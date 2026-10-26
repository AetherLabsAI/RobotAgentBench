"""Task-owned object/destination identities; no controller or physics settings."""
from dataclasses import dataclass
from .source_initial import catalog

PLATE = '/World/Assets/MicrowaveTakePlate/Asset'
PAYLOADS = {'30-1': '/World/Assets/MicrowaveBeefbox',
            '30-2': '/World/Assets/MicrowaveFoodbox',
            '30-3': '/World/Assets/MicrowaveSaladbox'}


@dataclass(frozen=True)
class PlacementBinding:
    task_id: str
    held_body: str
    destination: str
    destination_kind: str
    payload: str | None = None
    robot: str = '/World/Robot'

    def __post_init__(self):
        roots = [self.held_body, self.destination, self.robot]
        if self.payload is not None:
            roots.append(self.payload)
        if self.destination_kind not in ('table', 'tray'):
            raise ValueError('Explicit table/tray destination required')
        if any(not isinstance(p, str) or not p.startswith('/World/') or
               p.endswith('/') or '//' in p or any(x in ('.', '..') for x in p.split('/'))
               for p in roots):
            raise ValueError('Absolute canonical prim paths required')
        if any(a == b or a.startswith(b+'/') or b.startswith(a+'/')
               for i, a in enumerate(roots) for b in roots[i+1:]):
            raise ValueError('Placement roles must not overlap')


def binding_for(task_id):
    row = catalog()[task_id]
    if row['held_body'] != 'object':
        return PlacementBinding(task_id, PLATE, '/World/Assets/CoffeeTable',
                                'table', PAYLOADS[task_id])
    # User approved the 5-1 … 5-14 definition tables on 2026-09-23: every one of
    # them places into the tray on the coffee table, which is also what all 210
    # episodes of the new data do. The earlier 2026-09-17 table/tray split (from
    # the old L1 source) no longer applies to 5-x. 30-x keep the plate-on-table
    # binding above.
    return PlacementBinding(task_id, row['source_target'], '/World/Assets/Tray', 'tray')
