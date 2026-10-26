"""Non-overlapping background identities from existing collider hierarchies."""
if __package__:
    from .object_bindings import within
else:
    from object_bindings import within


def background_catalogue(colliders, object_paths, targets):
    goals = [root for roots in targets.values() for root in roots]
    selected = sorted({p for p in colliders if not within(p, '/World/Robot')
                       and not any(within(p, goal) for goal in goals)},
                      key=lambda p: (p.count('/'), p))
    roots = []
    for path in selected:
        # CollisionAPI can be authored at both kitchen body and its descendants.
        # These form one background collider hierarchy, not separate objects.
        if any(within(path, ancestor) for ancestor in roots):
            continue
        if any(within(goal, path) for goal in goals):
            raise ValueError('Background ancestor overlaps a target: '+path)
        roots.append(path)
    background = {}
    for path in roots:
        owners = [owner for owner in object_paths if within(path, owner)]
        owner = max(owners, key=len) if owners else path
        background.setdefault('background:'+owner, []).append(path)
    return background
