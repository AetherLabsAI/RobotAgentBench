"""Read-only identification of the existing physical room floor families."""


def physical_floor_paths(stage):
    from pxr import Usd, UsdPhysics

    def colliding(prim):
        return (bool(prim) and prim.IsActive()
            and prim.HasAPI(UsdPhysics.CollisionAPI)
            and prim.GetAttribute('physics:collisionEnabled').Get() is not False)

    legacy = stage.GetPrimAtPath('/World/Background/Shell/Floor')
    generated = stage.GetPrimAtPath('/World/HouseTopologyCandidate/Shell/Floors')
    generated_paths = [str(p.GetPath()) for p in Usd.PrimRange(generated) if colliding(p)] if generated else []
    # Existing released layouts use Background/<layout-root>/Floor.
    # Exact child identity, not a substring scan: refrigerator shelf floors,
    # hopper/basket floors and movable objects are never room floors.
    background = stage.GetPrimAtPath('/World/Background')
    layout_paths = []
    if background:
        for layout in background.GetChildren():
            if layout.GetName() == 'Shell':
                continue
            floor = layout.GetChild('Floor')
            if colliding(floor):
                layout_paths.append(str(floor.GetPath()))
    groups = [paths for paths in (
        [str(legacy.GetPath())] if colliding(legacy) else [],
        generated_paths, layout_paths) if paths]
    if len(groups) > 1:
        raise ValueError('Duplicate physical floor sources')
    if not groups:
        raise ValueError('No physical floor')
    return groups[0]
