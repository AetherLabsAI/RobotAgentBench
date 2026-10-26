"""Enable the user-approved stacked packing rule only in the OpenEnd factory."""
RULE_VERSION = 'openend.actual_bag_cavity_any_measured_support.v2'
KINDS = frozenset(('wallet_transfer', 'glasses_transfer', 'powerbank_transfer',
                   'water_bottle_transfer', 'umbrella_transfer'))


def enable_for_ports(kinds, ports):
    from openend.tasks.all_objects_integration.geometry import DESTINATIONS
    if set(kinds) != set(ports):
        raise ValueError('Exact registered port identities required')
    bag = DESTINATIONS['bag'][0]
    enabled = {}
    for key, kind in kinds.items():
        if kind not in KINDS:
            continue
        port = ports[key]
        # Wallet/glasses may instead target the small storage basket; its
        # original support rule must not be relaxed by the bag-only change.
        value = port.geometry.destination == bag
        port.allow_packed_support = value
        enabled[key] = value
    return dict(rule_version=RULE_VERSION, enabled=enabled,
        geometry='existing_actual_bag_cavity_not_AABB',
        support='any_measured_nonrobot_upward_support_while_inside',
        unsupported_flight_accepted=False, rim_only_accepted=False,
        floor_contact_accepted=False, physics_parameters_changed=False,
        process_points_and_release_requirements_changed=False,
        ordinary_short_task_default_changed=False)
