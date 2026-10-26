"""Bind actual room destinations; sample one measured base pose per control."""
from .rules import Progress

def eligible(name,port):
    c=port.composite
    if name=='fridge_to_heated_box':
        return bool(c.phase=='microwave' and port.source.latest['held'])
    if name=='stock_wooden_tray':
        return bool(c.completed_pick and port.source.latest['grasped'][port.source.active])
    held=bool(port.latest['held'])
    if name=='carry_meal_tray':return bool(c.operation_complete and held)
    if hasattr(c,'pickup_complete'):return bool(c.pickup_complete and held)
    if hasattr(c,'permit_place'):return bool(c.permit_place and held)
    # Object-transfer cores create pickup_complete on the first observed frame.
    if hasattr(c,'permit'):return False
    raise ValueError('No explicit pickup/navigation handoff for '+name)

class NativeNavigation:
    def __init__(self,stage,plan,ports,selector):
        from pxr import Usd,UsdGeom
        self.name=plan['task_name'];self.ports=ports;self.view=None;self.rebase=False
        regions={};bound={w.id:w for w in selector.workspaces}
        cache=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render'],useExtentsHint=False)
        for key,port in ports.items():
            workspace={'stock_wooden_tray':'table_wooden_tray','carry_meal_tray':'table_objects',
                       'sofa_to_coffee_table':'table_objects','hang_cap':'coat','pack_object':'bag'}.get(self.name)
            if workspace:
                if workspace not in bound:raise ValueError('Missing navigation workspace '+workspace)
                w=bound[workspace];regions[key]=dict(path=w.path,polygon=w.polygon)
            elif self.name=='fridge_to_heated_box':
                lo,hi=port.scene.activation['microwave']
                regions[key]=dict(path='bound_microwave_activation',polygon=((float(lo[0]),float(lo[1])),
                    (float(hi[0]),float(lo[1])),(float(hi[0]),float(hi[1])),(float(lo[0]),float(hi[1]))))
            else:
                path=port.geometry.destination;prim=stage.GetPrimAtPath(path)
                if not prim or not prim.IsActive():raise ValueError('Missing navigation destination '+path)
                r=cache.ComputeWorldBound(prim).ComputeAlignedRange();lo,hi=r.GetMin(),r.GetMax()
                regions[key]=dict(path=path,polygon=((float(lo[0]),float(lo[1])),(float(hi[0]),float(lo[1])),
                    (float(hi[0]),float(hi[1])),(float(lo[0]),float(hi[1]))))
        # A multi-item shelf task can require a legitimate return trip. One
        # route per delivery count, not one reward per remaining target.
        self.return_keys=[];self.delivered=set()
        if self.name=='stock_wooden_tray' and len(ports)>1:
            if 'shelf' not in bound:raise ValueError('Missing return-to-shelf workspace')
            w=bound['shelf']
            for i in range(1,len(ports)):
                key='return_to_shelf:'+str(i);self.return_keys.append(key)
                regions[key]=dict(path=w.path,polygon=w.polygon)
        self.progress=Progress(regions,selector.radius)
    def initialize(self,pool):
        self.view=pool.create_rigid_body_view(['/World/Robot/base_link'])
        if self.view.count!=1 or tuple(self.view.prim_paths)!=('/World/Robot/base_link',):
            raise ValueError('Unique measured base required')
    def sample(self,time_s,*,baseline=False):
        pose=self.view.get_transforms()[0]
        flags={key:eligible(self.name,p) for key,p in self.ports.items()}
        if self.return_keys:
            self.delivered.update(k for k,p in self.ports.items() if p.composite.destination_state.met)
            remaining=any(p.composite.completed_pick is None for p in self.ports.values())
            outstanding=any(p.composite.completed_pick is not None and not p.composite.destination_state.met
                            for p in self.ports.values())
            selected='return_to_shelf:'+str(len(self.delivered))
            flags.update({k:bool(remaining and not outstanding and k==selected) for k in self.return_keys})
        self.progress.observe(float(time_s),tuple(float(v) for v in pose[:2]),flags,baseline=baseline or self.rebase)
        self.rebase=False
    def augment(self,value):return self.progress.augment(value)
    def pause(self):self.rebase=True
