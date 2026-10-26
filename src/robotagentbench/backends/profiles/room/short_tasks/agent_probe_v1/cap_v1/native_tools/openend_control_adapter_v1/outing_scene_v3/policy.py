"""Outing-only authoring policy: real objectives separate from background."""
from copy import deepcopy
import random

VERSION = 'openend.outing.background_inventory.v3'
CORE = (
    dict(id='wallet', type='wallet_transfer', source='sofa', destination='fixed_cpu_bag'),
    dict(id='powerbank', type='powerbank_transfer', source='sofa'),
    dict(id='water', type='water_bottle_transfer', source='sofa'),
    dict(id='glasses', type='glasses_transfer', source='small_table', destination='fixed_cpu_bag'),
    dict(id='umbrella', type='umbrella_transfer', source='sofa'),
)
CLEANUP = {
    'book': dict(id='book', type='book_transfer', source='sofa', count=1),
    'remote': dict(id='remote', type='remote_transfer', source='sofa'),
    'cap': dict(id='cap', type='cap_transfer', source='sofa'),
    'notebook': dict(id='notebook', type='notebook_transfer', source='sofa'),
}
DEVICES = frozenset(('coffee', 'fryer', 'microwave'))


def validate_background(request):
    """No background behavior for callers that do not opt into this scenario."""
    if 'scenario' not in request and 'background' not in request:
        return None
    if request.get('scenario') != 'prepare_to_leave':
        raise ValueError('Background extension is enabled only for the outing scenario')
    background = request.get('background')
    if not isinstance(background, dict) or set(background) != {'shelf_stock', 'devices'}:
        raise ValueError('Explicit outing background inventory required')
    if type(background['shelf_stock']) is not bool:
        raise ValueError('Explicit shelf-stock choice required')
    devices = background['devices']
    if (not isinstance(devices, list) or any(d not in DEVICES for d in devices)
            or len(set(devices)) != len(devices)):
        raise ValueError('Only distinct existing appliance assemblies are supported')
    kinds = [t.get('type') for t in request['tasks']]
    for task in CORE:
        if kinds.count(task['type']) != 1:
            raise ValueError('One of each outing packing objective is required')
    allowed = {t['type'] for t in CORE}|{t['type'] for t in CLEANUP.values()}|{'trash_transfer', 'wipe_table'}
    if set(kinds)-allowed:
        raise ValueError('Background appliances and shelf stock must not be registered as outing tasks')
    for task in request['tasks']:
        if task.get('type') in ('wallet_transfer', 'glasses_transfer') and task.get('destination', 'fixed_cpu_bag')!='fixed_cpu_bag':
            raise ValueError('Outing personal items belong in the fixed-open bag')
    if len(request['tasks']) < 6:
        raise ValueError('At least one departure-cleanup objective is required in addition to packing')
    if request.get('extras'):
        raise ValueError('No ambiguous duplicate portable background objects')
    return deepcopy(background)


def make_request(*, seed, housekeeping, background_devices, shelf_stock, layout_id=None):
    """Caller selects household-clutter types; do not invent their probabilities.

    Waste/stain presence retains the approved four equiprobable combinations.
    Physical stock counts, products, spacing and poses remain source-sampler owned.
    """
    if type(seed) is not int or not 1 <= seed <= 4294967295:
        raise ValueError('Existing scene seed range required')
    if (not isinstance(housekeeping, list) or len(set(housekeeping))!=len(housekeeping)
            or any(k not in CLEANUP for k in housekeeping)):
        raise ValueError('Explicit distinct housekeeping choices required')
    rng = random.Random('outing_cleanliness_v3:'+str(seed))
    state = rng.randrange(4)
    tasks = deepcopy(list(CORE))+[deepcopy(CLEANUP[k]) for k in housekeeping]
    if state & 1:
        tasks.append(dict(id='trash', type='trash_transfer', source='coffee_table.objects', count=rng.randint(2,4)))
    if state & 2:
        tasks.append(dict(id='wipe', type='wipe_table'))
    request = dict(schema='openend.scene_requirements.request.v2', seed=seed, tasks=tasks,
        scenario='prepare_to_leave', extras=[],
        background=dict(shelf_stock=shelf_stock, devices=list(background_devices)))
    if layout_id is not None:
        request['layout_id'] = layout_id
    validate_background(request)
    receipt = dict(policy=VERSION, seed=seed, core_packing_tasks=5,
        housekeeping_choices=list(housekeeping), cleanliness_state=state,
        waste_present=bool(state&1), stain_present=bool(state&2),
        cleanliness_probabilities=[.25,.25,.25,.25], people_dependent=False,
        background_task_count=0, registered_declarations=len(tasks),
        physical_randomization_owner='existing_full_source_generators')
    return request, receipt
