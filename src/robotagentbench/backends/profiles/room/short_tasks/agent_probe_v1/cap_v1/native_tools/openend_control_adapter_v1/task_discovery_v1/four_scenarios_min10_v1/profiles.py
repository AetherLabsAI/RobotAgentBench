"""Private authoring requests, never sent to either coding policy."""
from copy import deepcopy

FAMILIES = ('prepare_dinner', 'host_guests', 'prepare_to_leave', 'tidy_room')
PROMPTS = dict(zip(FAMILIES, ('dinner_emma', 'visit_alex', 'outing_emma_work', 'tidy_room')))


def request(family):
    if family not in FAMILIES:
        raise ValueError('One of the four existing scenario families required')
    outing = family == 'prepare_to_leave'
    tasks = [
        dict(id='wallet', type='wallet_transfer', source='small_table',
             destination='fixed_cpu_bag' if outing else 'storage_basket'),
        dict(id='glasses', type='glasses_transfer', source='coffee_table.objects', destination='fixed_cpu_bag'),
        dict(id='powerbank', type='powerbank_transfer', source='sofa'),
        dict(id='water', type='water_bottle_transfer', source='sofa'),
        dict(id='remote', type='remote_transfer', source='sofa'),
        dict(id='book', type='book_transfer', source='sofa', count=1),
        dict(id='cap', type='cap_transfer', source='sofa'),
        dict(id='notebook', type='notebook_transfer', source='sofa'),
        dict(id='trash', type='trash_transfer', source='coffee_table.objects', count=2),
    ]
    if family == 'prepare_dinner':
        tasks += [
            dict(id='fridge', type='fridge_to_microwave', payload_variant='leftover_box'),
            dict(id='warm_meal', type='microwave_box_to_meal_tray', payload_variant='leftover_box'),
            dict(id='fork', type='cutlery_to_meal_tray', asset='meal_fork'),
            dict(id='meal_tray', type='meal_tray_transfer'),
        ]
    elif family == 'host_guests':
        tasks += [dict(id='snack', type='snack_drink_to_tray', product='oreo_gf'),
                  dict(id='drink', type='snack_drink_to_tray', product='sprite_c1')]
    return dict(schema='openend.scene_requirements.request.v2', layout_id=43,
                seed=2026102101 + FAMILIES.index(family), tasks=deepcopy(tasks), extras=[])


def profiles():
    return {family: dict(family=family, prompt_id=PROMPTS[family], request=request(family))
            for family in FAMILIES}
