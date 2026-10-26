"""Bind native contacts to the same registered support set as scene planning."""

def install(service, manifest):
    from pxr import UsdGeom, UsdPhysics
    from openend.tasks.scene_requirements.surface_registry import contact_paths,manifest_surfaces,surface_set
    primary=service.roles['counter']
    paths=contact_paths(service.stage,manifest,primary)
    surfaces=manifest_surfaces(service.stage,manifest) if manifest.get('overview',{}).get('surfaces') else []
    declared=surface_set(surfaces,primary) if surfaces else []
    boxes={};tops={}
    cache=UsdGeom.BBoxCache(0,[UsdGeom.Tokens.default_,UsdGeom.Tokens.render])
    for path in paths:
        p=service.stage.GetPrimAtPath(path)
        if not p or not p.IsActive() or not p.HasAPI(UsdPhysics.CollisionAPI) or p.GetAttribute('physics:collisionEnabled').Get() is False:
            raise ValueError('Missing declared worktop collider '+path)
        ancestor=p
        while ancestor and not ancestor.IsPseudoRoot():
            if ancestor.HasAPI(UsdPhysics.RigidBodyAPI):raise ValueError('Worktop must be static: '+path)
            ancestor=ancestor.GetParent()
        box=cache.ComputeWorldBound(p).ComputeAlignedRange()
        boxes[path]=[list(box.GetMin()),list(box.GetMax())]
        row=next((s for s in declared if path in s['collider_paths']),None)
        tops[path]=row['top'] if row else boxes[path][1][2]
        if not boxes[path][0][2]-1e-5<=tops[path]<=boxes[path][1][2]+1e-5:
            raise ValueError('Declared worktop height outside collider '+path)
    # SceneCoffeeService performs the sole classification; no second per-pair
    # contact replay or AABB-only connectivity graph.
    service.counter_paths=paths
    return dict(status='scene_connected_static_counter_binding',paths=paths,bounds=boxes,
                support_heights=tops,raw_contacts_modified=False,physics_modified=False,
                support_schema='arena.support_surfaces.v2')
