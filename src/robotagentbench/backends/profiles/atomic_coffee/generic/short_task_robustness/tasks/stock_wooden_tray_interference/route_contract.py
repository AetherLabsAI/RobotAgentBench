"""Task-local rear-entry and robot-obstacle admission contract."""


def require_rear_entry(reservation):
    entry = reservation.get('route_evidence', {}).get('entry', {})
    if (entry.get('method') != 'rear_short_entry'
            or entry.get('angle_deg') not in (180, 165, -165, 150, -150, 135, -135)):
        raise ValueError('Rear entry required; regenerate this legacy lateral reservation')


def robot_obstacle_shape(prim):
    """Physical shapes count even when hidden or marked proxy/guide."""
    from pxr import UsdGeom, UsdPhysics
    if not prim.IsA(UsdGeom.Boundable):
        return False
    if (prim.HasAPI(UsdPhysics.CollisionAPI)
            and prim.GetAttribute('physics:collisionEnabled').Get() is not False):
        return True
    image = UsdGeom.Imageable(prim)
    return (image.ComputeVisibility() != 'invisible'
            and image.ComputePurpose() in ('default', 'render'))
