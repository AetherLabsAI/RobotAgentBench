"""Existing component IDs, no new public task IDs or inferred destinations."""
from dataclasses import dataclass
from evaluation.tasks.personal_item_pick.specs import task_spec
from evaluation.tasks.tidy_sofa.specs import spec
from evaluation.tasks.tidy_sofa.hang_hat import process as hat
from evaluation.tasks.trash_disposal import definitions as trash
from evaluation.tasks.basket_place import process_rules as basket

@dataclass(frozen=True)
class Definition:
    key: str
    asset: str
    pickup_api: str
    placement_api: str
    mode: str
    destination_role: str
    height_reference: str = 'rigid_origin'

DEFINITIONS = {d.key:d for d in (
    Definition('umbrella_to_bag','umbrella','13-1','14-1','bag','fixed_cpu_bag'),
    Definition('water_bottle_to_bag','water_bottle','13-5','14-5','bag','fixed_cpu_bag'),
    Definition('book_to_coffee_table','book_red','15-1','10-4','surface','coffee_table'),
    Definition('book_to_small_table','book_red','15-1','10-5','surface','small_table'),
    Definition('cap_to_rack','baseball_cap','15-2','17-1','hang','coat_rack','world_aabb_center'),
    Definition('note_to_drawer','notebook_green','13-6','16-1','drawer','coffee_table_drawer'),
    Definition('paper_cup_to_bin','paper_cup','13-7','18-4','trash','ordinary_trash_bin'),
)}

# User-approved defaults for this task family; not the room's default episode.
DEFAULT_BY_ASSET = {
    'umbrella': 'umbrella_to_bag',
    'water_bottle': 'water_bottle_to_bag',
    'book_red': 'book_to_small_table',
    'baseball_cap': 'cap_to_rack',
    'notebook_green': 'note_to_drawer',
    'paper_cup': 'paper_cup_to_bin',
}

def definition(key):
    if key not in DEFINITIONS: raise ValueError('Explicit supported task variant required')
    return DEFINITIONS[key]

def pickup_spec(d):
    return task_spec(d.pickup_api) if d.pickup_api in ('13-1','13-5') else spec(d.pickup_api)

def placement_max(d):
    if d.mode == 'bag': return basket.MAXIMUM_SCORE
    if d.mode == 'hang': return sum(hat.POINTS.values())
    if d.mode == 'trash': return trash.MAXIMUM_SCORE
    return spec(d.placement_api).maximum_score
