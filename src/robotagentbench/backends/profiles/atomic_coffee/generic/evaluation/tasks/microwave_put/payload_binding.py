"""Task-local selected instance identity, shared by all read-only evidence."""
from .variants import variant_for

PATHS = {
    'beef_box': '/World/Assets/MicrowaveBeefbox',
    'leftover_box': '/World/Assets/MicrowaveFoodbox',
}


def payload_for(variant):
    return PATHS[variant_for(variant).name]
