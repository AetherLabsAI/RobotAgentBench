"""Task-owned payload choices; the generic API only transports this object."""

PAYLOAD_LABELS = {'beef_box': '牛肉盒', 'leftover_box': '剩饭盒'}
DEFAULT_PAYLOAD_VARIANT = 'beef_box'
PARAMETER_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'properties': {'payload_variant': {'type': 'string',
        'enum': list(PAYLOAD_LABELS), 'default': DEFAULT_PAYLOAD_VARIANT,
        'description': '盘子上承载的目标盒子：beef_box=牛肉盒，leftover_box=剩饭盒',
        'x-runtime-status': {'beef_box': 'existing_runtime',
                             'leftover_box': 'native_asset_runtime'}}},
}


def normalize_parameters(parameters):
    if not isinstance(parameters, dict) or set(parameters) - {'payload_variant'}:
        raise ValueError('Only payload_variant is supported for microwave take')
    value = parameters.get('payload_variant', DEFAULT_PAYLOAD_VARIANT)
    if not isinstance(value, str) or value not in PAYLOAD_LABELS:
        raise ValueError('payload_variant must be beef_box or leftover_box')
    return {'payload_variant': value}


def instruction_for_parameters(parameters):
    variant = normalize_parameters(parameters)['payload_variant']
    # Preserve the accepted default instruction byte-for-byte for existing runs.
    name = '牛排盒' if variant == 'beef_box' else PAYLOAD_LABELS[variant]
    return f'打开微波炉，抓住盘子并抬起取出，保持{name}在盘上，关门后稳定持有至少五秒。'
