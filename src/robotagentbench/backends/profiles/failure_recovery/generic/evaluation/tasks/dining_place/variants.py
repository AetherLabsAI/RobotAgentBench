"""Task-owned selection; shared fridge assets remain read-only."""
PAYLOADS = {
    'beef_box': '/World/Assets/MicrowaveBeefbox',
    'leftover_box': '/World/Assets/MicrowaveFoodbox',
    'salad_box': '/World/Assets/MicrowaveSaladbox',
}
PLACEMENT_VARIANTS = ('leftover_box', 'salad_box')


def selected(parameters):
    value=parameters.get('payload_variant','beef_box')
    if value not in PAYLOADS: raise ValueError('Unknown dining payload variant')
    return value


def payload_for(variant):
    return PAYLOADS[selected({'payload_variant':variant})]


def instruction_for(parameters):
    name={'beef_box':'牛排盒','leftover_box':'剩饭盒','salad_box':'沙拉盒'}[selected(parameters)]
    return f'将装有{name}的盘子放到餐桌上。'
