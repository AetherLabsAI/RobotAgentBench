"""User-approved hand/target-only collision filtering after full opening.

This is an explicit scripted-interference simplification, not a physical
collision-free release claim. Target, robot and background assets stay intact.
"""
import math


def discard_filtered_pairs(active, human_paths, target_root):
    """Invalidate only pairs made impossible by the authored filter change.

    This is explicit cache lifecycle bookkeeping, not a fabricated native
    CONTACT_LOST event. Other awake-pair completeness checks remain intact.
    """
    def under(path,root):return path==root or path.startswith(root.rstrip('/')+'/')
    keys=[]
    for key,item in active.items():
        paths=item[0]
        if len(paths)!=2:raise ValueError('Expected two cached contact paths')
        if any(under(paths[i],target_root) and any(under(paths[1-i],h) for h in human_paths) for i in (0,1)):
            keys.append(key)
    for key in keys:del active[key]
    return len(keys)


def release_ready(*, phase, released, constraint_active, previous_closure, supported):
    values=list(previous_closure.values()) if isinstance(previous_closure,dict) else [previous_closure]
    if not values or any(not math.isfinite(v) or not 0<=v<=1 for v in values):
        raise ValueError('Measured prior hand-command closure required')
    return (phase=='retreat' and released is True and constraint_active is False
            and supported is True and all(v==0 for v in values))


class ReleasedTargetFilter:
    def __init__(self, stage, human_root, human_links, target_path):
        from pxr import UsdPhysics
        target=stage.GetPrimAtPath(target_path)
        if (not target or not target.HasAPI(UsdPhysics.RigidBodyAPI)
                or target_path.startswith('/World/Robot/') or target_path.startswith(human_root+'/')):
            raise ValueError('Exact external target rigid body required')
        if not human_links or len(set(human_links))!=len(human_links):
            raise ValueError('Unique human body paths required')
        self.relations=[]
        # Validate every path first; never partially alter an invalid set.
        for path in human_links:
            prim=stage.GetPrimAtPath(path)
            if not path.startswith(human_root+'/Bodies/') or not prim or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                raise ValueError('Only this human actor may own filter opinions')
            existing=UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel()
            if existing and existing.GetTargets():raise ValueError('Unexpected pre-existing human pair filters')
        for path in human_links:
            # Empty at initialization: all human/target grip contacts remain.
            self.relations.append(UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(path)).CreateFilteredPairsRel())
        self.target_path=target_path
        self.human_paths=tuple(human_links)
        self.enabled=False

    def update(self, **evidence):
        if self.enabled or not release_ready(**evidence):return False
        for relationship in self.relations:
            relationship.SetTargets([self.target_path])
        self.enabled=True
        return True

    def invalidate_reader_cache(self, source):
        if not self.enabled:raise RuntimeError('Cannot expire contacts before physical pair filtering')
        self.evidence()
        count=discard_filtered_pairs(source.contacts.active,self.human_paths,self.target_path)
        placement_count=0
        for reader in source.placements.values():
            if reader.binding.held_body==self.target_path:
                placement_count+=discard_filtered_pairs(reader.stream.active,
                    tuple('/Unbound'+p for p in self.human_paths),reader._held_role)
        return dict(source_pairs=count,placement_pairs=placement_count,
                    reason='explicit_human_target_filter_activation',synthetic_contact_events=False)

    def evidence(self):
        expected=[self.target_path] if self.enabled else []
        if any([str(p) for p in rel.GetTargets()]!=expected for rel in self.relations):
            raise RuntimeError('Release filter scope changed')
        return dict(enabled=self.enabled,target_path=self.target_path,human_body_count=len(self.relations),
                    only_human_target_pair=True,robot_and_environment_pairs_unmodified=True,
                    policy='after_constraint_release_and_fully_open_hand')
