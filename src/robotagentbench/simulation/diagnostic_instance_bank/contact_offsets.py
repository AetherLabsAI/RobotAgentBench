"""Read effective offsets from existing native human views; never set physics.

The USD -inf fallback is an automatic sentinel, not a numerical margin. This
module does not approximate the SDK's shape/gravity/timestep-dependent result.
It must be called by a trusted operator after the unchanged owner has created
its native shapes. It does not start a simulator or create/enable any shape.
"""
import math


class EffectiveOffsetUnavailable(ValueError):
    pass


def _finite(value, name):
    value = float(value)
    if not math.isfinite(value):
        raise EffectiveOffsetUnavailable(name + ' is not a finite native value')
    return value


def capture_existing_human_views(*, views, links, physics_step, time_s,
                                 sdk_identity, shape_identity,
                                 scene_physics_identity):
    """Return per-bone measured values without choosing a new scalar margin.

    ``views`` and ``links`` are the original NativeHuman's existing dictionaries.
    Each original bone owns one rigid body and one convex-hull Collision mesh.
    An absent, padded, multi-shape or unresolved view fails closed. In particular
    no zero/max/default is substituted for an unavailable measurement.
    """
    if set(views) != set(links) or not views:
        raise EffectiveOffsetUnavailable('Complete original human view/link mapping required')
    if type(physics_step) is not int or physics_step < 0:
        raise EffectiveOffsetUnavailable('Measured native physics step required')
    time_s = _finite(time_s, 'time_s')
    if time_s < 0:
        raise EffectiveOffsetUnavailable('Measured non-negative simulation time required')
    for label, identity in (('SDK', sdk_identity), ('shape', shape_identity),
                            ('scene physics', scene_physics_identity)):
        if not isinstance(identity, dict) or not identity:
            raise EffectiveOffsetUnavailable(label + ' provenance is required')
    rows = {}
    for bone in sorted(links):
        body = links[bone]
        view = views[bone]
        if (not isinstance(body, str) or not body.startswith('/World/HumanInterference/Bodies/')
                or view.count != 1 or view.max_shapes != 1
                or list(view.prim_paths) != [body]):
            raise EffectiveOffsetUnavailable('Exact original one-body/one-shape view required: ' + bone)
        contact = view.get_contact_offsets()
        rest = view.get_rest_offsets()
        if len(contact) != 1 or len(contact[0]) != 1 or len(rest) != 1 or len(rest[0]) != 1:
            raise EffectiveOffsetUnavailable('Native shape output dimensions changed: ' + bone)
        offset = _finite(contact[0][0], bone + ' contact offset')
        resting = _finite(rest[0][0], bone + ' rest offset')
        if offset < max(0., resting):
            raise EffectiveOffsetUnavailable('Native offsets violate the installed schema range: ' + bone)
        rows[bone] = dict(body=body, collider=body+'/Collision',
                          contact_offset_stage_units=offset,
                          rest_offset_stage_units=resting,
                          source='original_existing_RigidBodyView_getters')
    return dict(schema='robotagentbench.existing_native_contact_offsets.v1',
                physics_step=physics_step, time_s=time_s, rows=rows,
                sdk_identity=dict(sdk_identity), shape_identity=dict(shape_identity),
                scene_physics_identity=dict(scene_physics_identity),
                physics_attributes_written=False, shapes_created_or_enabled=False,
                scalar_margin_selected=False)
