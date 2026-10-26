"""Scene-specific segmented worktop binding; raw native contacts stay unchanged."""
from types import MethodType

def install(service, manifest):
    from pxr import UsdGeom, UsdPhysics
    from evaluation.tasks.coffee_machine import isaac_evidence as old
    from evaluation.tasks.coffee_machine.facts import COUNTER, OTHER
    # Only the audited, contiguous A/B worktop pair in layout04. Other layouts
    # retain their original binding until their declared surfaces are audited.
    root='/World/Background/Shell/ConstructorKitchen/MetricScale/'
    paths=tuple(root+'CornerGapBase300_'+s+'/Body/Worktop' for s in ('A','B'))
    if manifest['layout_id']!=4 or service.roles['counter']!=paths[0]:
        return dict(status='original_single_surface_binding')
    boxes=[]
    cache=UsdGeom.BBoxCache(0,[UsdGeom.Tokens.default_,UsdGeom.Tokens.render])
    for path in paths:
        p=service.stage.GetPrimAtPath(path)
        if not p or not p.IsActive() or not p.HasAPI(UsdPhysics.CollisionAPI) or p.GetAttribute('physics:collisionEnabled').Get() is False:
            raise ValueError('Missing declared worktop collider '+path)
        ancestor=p
        while ancestor and not ancestor.IsPseudoRoot():
            if ancestor.HasAPI(UsdPhysics.RigidBodyAPI):raise ValueError('Worktop must be static')
            ancestor=ancestor.GetParent()
        box=cache.ComputeWorldBound(p).ComputeAlignedRange()
        boxes.append([list(box.GetMin()),list(box.GetMax())])
    if abs(boxes[0][1][2]-boxes[1][1][2])>1e-5:
        raise ValueError('Worktops do not share a surface height')
    if any(max(boxes[0][0][k],boxes[1][0][k])-min(boxes[0][1][k],boxes[1][1][k])>1e-5 for k in (0,1)):
        raise ValueError('Declared worktops are not contiguous')
    original=service._contacts
    def contacts(self,pairs,floor):
        result=original(pairs,floor)
        # Ask the unchanged classifier which exact pairs produced OTHER.
        # Reclassify only upward positive-impulse contacts to these worktops.
        for obj,flags in result[0].items():
            has_other=False
            for pair in pairs:
                if OTHER not in original((pair,),floor)[0][obj]['supports']:continue
                names=pair[0]
                other=names[1] if old.under(names[0],obj) else names[0]
                if other in paths:flags['supports'].add(COUNTER)
                else:has_other=True
            if not has_other:flags['supports'].discard(OTHER)
        return result
    service._contacts=MethodType(contacts,service)
    return dict(status='audited_segmented_counter_binding',paths=paths,bounds=boxes,
                raw_contacts_modified=False,physics_modified=False)
