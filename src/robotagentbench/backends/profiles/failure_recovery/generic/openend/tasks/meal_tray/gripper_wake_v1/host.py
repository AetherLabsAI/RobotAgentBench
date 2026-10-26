"""Explicit unified-room adapter; no replacement of scoring or physics methods."""
from .archive import ContactArchive
from .router import GripperWakeIndex


def bindings_for(kind,port,resolved,device_roots):
    if kind=='cloth_clear':return tuple(port.geometry.targets)
    if kind=='microwave_box_to_meal_tray':
        roots=[port.geometry.target,port.source_path]
        if port.microwave.plate:roots.append(port.microwave.plate.target)
        return tuple(roots)
    if getattr(port,'selection_mode','sector')=='contact':
        return (port.geometry.target,)
    return device_roots(kind,port,resolved)


class UnifiedGripperRoutes:
    def __init__(self,stage,bindings):
        from openend.tasks.meal_tray.task_contact_routing import NativeContactRoutes
        self.native=NativeContactRoutes(stage,bindings)
        self.interface=self.native.interface
        self.index=GripperWakeIndex(bindings,self.native.fingers)
        self.archive=ContactArchive(self.native.contacts.identity.resolve,
            (self.native.contacts.found,self.native.contacts.persist,self.native.contacts.lost))
    def after_physics(self,report,time_s):
        self.archive.observe(report,time_s)
        self.native.contacts.ingest(report)
        self.index.observe(self.native.contacts.pairs(),time_s)
    def arm(self,time_s):self.index.arm_after_initial_acceptance(time_s)
    def requests(self,ports,time_s):return self.index.requests(ports,time_s)


def prime_port(port,archive):
    """Once after dormancy; restore real contact caches, not physical state."""
    if hasattr(port,'contacts'):archive.prime(port.contacts)
    source=getattr(port,'source',None)
    if source is not None:
        if hasattr(source,'placements'):
            from evaluation.tasks.shelf_pick.geometry import rotation
            archive.prime(source.contacts)
            for path,reader in source.placements.items():
                _,q=source.views[path].get_world_pose()
                archive.prime(reader,rotation(q).apply([0,0,1]))
        elif hasattr(source,'scene') and hasattr(source,'contacts'):
            from evaluation.tasks.microwave_put.physical_geometry import PLATE_BODY,rotation
            _,q=source.views[PLATE_BODY].get_world_pose()
            archive.prime(source.contacts,rotation(q).apply([0,0,1]))
        elif hasattr(source,'contacts'):archive.prime(source.contacts)
    owner=getattr(port,'owner',None)
    if owner is not None and hasattr(owner,'source'):archive.prime(owner.source.contacts)


def dormant(port,helper):
    """Clock-only for unstarted portable/shelf/fridge readers.

Device maintenance is explicitly retained (no task selection or scoring).
"""
    from openend.tasks.all_objects_integration.idle_devices import check_tick
    # These own actual physical/animation mechanisms, not just score readers.
    if hasattr(port,'owner') or (hasattr(port,'source') and hasattr(port.source,'rail')):
        port.after_inactive(helper)
        return 'device_maintenance'
    source=getattr(port,'source',None)
    reader=source if source is not None else port
    now,dt=check_tick(port.clock,reader.last_time,helper)
    if getattr(port,'pending',()) or getattr(port,'buffer',()):
        raise RuntimeError('Unconsumed evidence before dormancy')
    # A knob/lamp mechanism must keep its own clock; it cannot be frozen by
    # disabling the box's scoring reader.
    if hasattr(port,'_mechanics'):port._mechanics(helper['_contact_report'])
    if hasattr(port.composite,'suspend_substep'):port.composite.suspend_substep(now,dt)
    elif hasattr(port.composite,'pause'):port.composite.pause()
    detector=getattr(reader,'detector',None)
    if detector is not None:
        for h in detector.history.values():h.clear()
    reader.last_time=now
    port.inactive_substeps=getattr(port,'inactive_substeps',0)+1
    return 'clock_only'
