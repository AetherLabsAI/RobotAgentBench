"""Resolve and audit required robot/environment collision pairs before reset."""
from __future__ import annotations


def enable_required_contacts(stage):
    """Remove the historical basket/robot exclusion in the episode's stage.

    Keep native internal robot filtering (including the gripper four-bar loops).
    This must run once at episode initialization, never at task boundaries.
    """
    from pxr import Sdf, Usd, UsdPhysics

    basket = stage.GetPrimAtPath('/World/CanonicalTableclothBasketCollisionGroup')
    target = Sdf.Path('/World/CanonicalRobotCollisionGroup')
    repairs = []
    assets = stage.GetPrimAtPath('/World/Assets')
    if assets:
        for asset in assets.GetChildren():
            colliders = [p for p in Usd.PrimRange(asset) if p.HasAPI(UsdPhysics.CollisionAPI)]
            # Preserve intentional visual-mesh exclusions where enabled proxy
            # colliders already exist (for example the shelving unit).
            if colliders and not any(UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() is not False
                                     for p in colliders):
                for prim in colliders:
                    UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(True)
                    repairs.append(dict(attribute=str(prim.GetPath())+'.physics:collisionEnabled',
                        previous=False, actual=True, reason='Asset had no enabled collision geometry'))
    if basket:
        relation = UsdPhysics.CollisionGroup(basket).GetFilteredGroupsRel()
        if target in relation.GetTargets():
            relation.RemoveTarget(target)
            repairs.append(dict(relationship=str(relation.GetPath()), removed_target=str(target),
                                reason='Physical basket/robot contacts explicitly required'))
    return repairs


def audit_required_contacts(stage):
    """Reject disabled required shapes and cross-group or direct exclusions."""
    from pxr import UsdPhysics

    roots = ('/World/Robot', '/World/Assets/CoffeeTable', '/World/Assets/TableclothBasket')

    def overlaps(path, root):
        return path == root or path.startswith(root+'/') or root.startswith(path+'/')

    inventory, groups, pairs = [], {}, []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            path = str(prim.GetPath())
            enabled = prim.GetAttribute('physics:collisionEnabled').Get() is not False
            inventory.append(dict(path=path, enabled=enabled))
            if any(overlaps(path, root) for root in roots) and not enabled:
                raise ValueError('Required physical collision is disabled: '+path)
        if prim.IsA(UsdPhysics.CollisionGroup):
            group = UsdPhysics.CollisionGroup(prim)
            members = [str(p) for p in group.GetCollidersCollectionAPI().GetIncludesRel().GetTargets()]
            groups[str(prim.GetPath())] = dict(members=members,
                filtered=[str(p) for p in group.GetFilteredGroupsRel().GetTargets()])
        relation = prim.GetRelationship('physics:filteredPairs')
        if relation:
            pairs.extend((str(prim.GetPath()), str(target)) for target in relation.GetTargets())
    for root in roots:
        if not any(row['path'].startswith(root+'/') and row['enabled'] for row in inventory):
            raise ValueError('Missing required collision group: '+root)

    def rejects(left, right):
        # Native filtering within the robot is part of its shared CAD model.
        return any(overlaps(left, roots[i]) and overlaps(right, roots[j])
                   for i in range(3) for j in range(3) if i != j)

    for left, right in pairs:
        if rejects(left, right):
            raise ValueError(f'Required contact pair filtered: {left} -> {right}')
    for path, group in groups.items():
        for filtered in group['filtered']:
            if filtered not in groups:
                raise ValueError('Unresolved collision filter group: '+filtered)
            if any(rejects(a, b) for a in group['members'] for b in groups[filtered]['members']):
                raise ValueError(f'Required collision groups filtered: {path} -> {filtered}')
    return dict(colliders=inventory, groups=groups, direct_filtered_pairs=pairs)
