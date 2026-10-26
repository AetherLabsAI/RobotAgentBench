"""Explicit user-approved human/gripper pair filter for coffee only.

Author relationships on this new human, never on robot, target or fixtures.
All robot arm/wrist-camera, bag, bean and scene collisions remain enabled.
The earlier approved release filter is composed by union, never replacement.
"""
from short_task_robustness.tasks.stock_wooden_tray_interference.release_collision import release_ready


def gripper_body_paths(stage):
    from pxr import Usd,UsdPhysics
    root=stage.GetPrimAtPath('/World/Robot')
    if not root:raise ValueError('Robot root required')
    paths=tuple(sorted(str(p.GetPath()) for p in Usd.PrimRange(root)
        if p.HasAPI(UsdPhysics.RigidBodyAPI) and p.GetName().startswith(('gripper_l_','gripper_r_'))))
    for hand in ('l','r'):
        if not any('/gripper_'+hand+'_' in p for p in paths):raise ValueError('Both grippers must resolve')
    return paths


class CoffeeHumanPairs:
    def __init__(self,stage,human_root,human_links,target_path):
        from pxr import UsdPhysics
        if not human_links or len(set(human_links))!=len(human_links):raise ValueError('Unique human bodies required')
        self.grippers=gripper_body_paths(stage)
        target=stage.GetPrimAtPath(target_path)
        if not target or not target.HasAPI(UsdPhysics.RigidBodyAPI) or target_path.startswith('/World/Robot/'):
            raise ValueError('Exact dynamic target binding required')
        self.relations=[];self.target_path=target_path;self.enabled=False
        for path in human_links:
            prim=stage.GetPrimAtPath(path)
            if not path.startswith(human_root+'/Bodies/') or not prim or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                raise ValueError('Only this actor may own filter opinions')
            prior=UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel()
            if prior and prior.GetTargets():raise ValueError('Unexpected actor filters; refuse overwrite')
        for path in human_links:
            rel=UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(path)).CreateFilteredPairsRel()
            rel.SetTargets(list(self.grippers));self.relations.append(rel)

    def update(self,**evidence):
        self.evidence()
        if self.enabled or not release_ready(**evidence):return False
        for rel in self.relations:rel.SetTargets([*self.grippers,self.target_path])
        self.enabled=True
        return True

    def evidence(self):
        expected=[*self.grippers,*([self.target_path] if self.enabled else [])]
        if any([str(p) for p in rel.GetTargets()]!=expected for rel in self.relations):
            raise RuntimeError('Human filter scope changed')
        return dict(gripper_pairs_filtered=True,gripper_paths=list(self.grippers),
            release_target_filter_enabled=self.enabled,target_path=self.target_path,
            authoring='human rigid bodies only',robot_asset_modified=False,
            other_robot_links_filtered=False,bag_collision_preserved_during_grip=True)
