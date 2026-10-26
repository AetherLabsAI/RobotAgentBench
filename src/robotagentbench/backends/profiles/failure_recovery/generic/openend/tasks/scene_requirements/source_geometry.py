"""Read-only layout floor discovery for portable source adapters."""
def floor_paths(stage):
    from pxr import Usd,UsdPhysics
    old=[p for p in ('/World/Background/Shell/Floor','/World/HouseTopologyCandidate/Shell/Floors') if stage.GetPrimAtPath(p)]
    if old:return old
    # Alternate release layouts use Background/<layout>/Floor. Only accept
    # actual floor collider prims, never refrigerator shelf-floor names.
    root=stage.GetPrimAtPath('/World/Background')
    return [str(p.GetPath()) for p in Usd.PrimRange(root)
            if p.GetName()=='Floor' and p.HasAPI(UsdPhysics.CollisionAPI)
            and p.GetAttribute('physics:collisionEnabled').Get() is not False] if root else []
