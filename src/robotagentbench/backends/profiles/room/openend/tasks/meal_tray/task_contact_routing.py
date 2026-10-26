"""Contact-only task wake routing, independent of workspace rate selection.

No scoring, scene pose writes or physics changes. Existing portable task wake
readers remain in use. This supplies the same 10-second wake to device ports.
"""
from openend.tasks.all_objects_integration.touch_wake import TouchWake


def under(path, root):
    return path == root or path.startswith(root.rstrip('/')+'/')

def registered_pair(paths,fingers,targets):
    return any(any(under(paths[i],f) for f in fingers) and
        any(under(paths[1-i],p) for p in targets) for i in (0,1))


def roots_for(kind, port, resolved):
    """Bind real interaction objects, never a workspace or a whole stock shelf."""
    if hasattr(port, 'task_contact_roots'):
        roots = port.task_contact_roots()
    elif kind == 'coffee':
        roles = port.owner.service.roles
        roots = [roles[k] for k in ('machine', 'button', 'cup', 'bag')]
    elif kind == 'fryer':
        roots = [port.source.paths[k] for k in ('body', 'tray', 'target', 'other', 'plate')]
    elif kind == 'fridge_to_microwave':
        roots = [port.scene.target, port.binding.objects['plate']]
        roots += [body for joint in port.scene.joints.values() for body in joint[0]]
    elif kind == 'snack_drink_to_tray':
        # Repeated same-product instances are distinct. Touching a stock peer,
        # the shelf, or the shared destination never wakes this instance.
        roots = [resolved['path']]
    else:
        raise ValueError('Missing explicit contact binding for '+kind)
    roots = tuple(dict.fromkeys(roots))
    if not roots or any(not isinstance(p,str) or not p.startswith('/World/') for p in roots):
        raise ValueError('Actual task prim paths required')
    return roots


class ContactRoutes:
    """Pure matching for already measured contact pairs; never fabricates facts."""
    def __init__(self, bindings, finger_roots):
        self.bindings = {k:tuple(v) for k,v in bindings.items()}
        self.fingers = tuple(finger_roots)
        if not self.fingers or any(not roots for roots in self.bindings.values()):
            raise ValueError('Explicit task objects and finger bodies required')
        self.wakes = {k:TouchWake() for k in bindings}
        self.time = None

    def observe(self, pairs, time_s):
        pairs = tuple(pairs)
        for key, roots in self.bindings.items():
            touched = any(bool(samples) and any(
                any(under(paths[i],f) for f in self.fingers) and
                any(under(paths[1-i],r) for r in roots) for i in (0,1))
                for paths,samples in pairs)
            self.wakes[key].observe(time_s=time_s, finger_contact=touched,
                supports=frozenset(), external_contact=touched, on_floor=False)
        self.time = time_s

    def requests(self, ports, time_s):
        if self.time is None or abs(self.time-time_s)>1e-6:
            raise RuntimeError('Stale task-contact routing evidence')
        return {k:w.request(ended=ports[k].ended,time_s=time_s) for k,w in self.wakes.items()}


class NativeContactRoutes(ContactRoutes):
    def __init__(self, stage, bindings):
        from pxr import Usd, UsdPhysics, PhysicsSchemaTools
        from omni.physx import get_physx_simulation_interface
        from omni.physx.bindings._physx import ContactEventType
        from evaluation.tasks.tidy_sofa.isaac_reader import Contacts
        robot = stage.GetPrimAtPath('/World/Robot')
        fingers = []
        for p in Usd.PrimRange(robot, Usd.TraverseInstanceProxies()):
            name = p.GetName()
            if p.HasAPI(UsdPhysics.RigidBodyAPI) and any(name.startswith('gripper_'+h+'_'+side+'_link')
                    for h in ('l','r') for side in ('inner','outer')):
                fingers.append(str(p.GetPath()))
                # Existing ports already install these observers. Never silently
                # add or alter an incompatible report threshold here.
                a = p.GetAttribute('physxContactReport:threshold')
                if not a or a.Get()!=0.:
                    raise ValueError('Missing existing finger contact reporter '+str(p.GetPath()))
        for roots in bindings.values():
            for path in roots:
                p = stage.GetPrimAtPath(path)
                if not p or not p.IsActive():raise ValueError('Missing contact target '+path)
        super().__init__(bindings, fingers)
        self.interface = get_physx_simulation_interface()
        def sleeping(sid,actor):
            p = stage.GetPrimAtPath(PhysicsSchemaTools.intToSdfPath(actor))
            if not p:raise RuntimeError('Unknown contact actor')
            return bool(not p.HasAPI(UsdPhysics.RigidBodyAPI) or self.interface.is_sleeping(sid,actor))
        self.contacts = Contacts(PhysicsSchemaTools.intToSdfPath, sleeping,
            (ContactEventType.CONTACT_FOUND,ContactEventType.CONTACT_PERSIST,ContactEventType.CONTACT_LOST), fingers)
        from arena_api.contact_identity import ContactIdentityCache
        targets=tuple(dict.fromkeys(p for roots in bindings.values() for p in roots))
        # The wake sensor owns only finger-to-registered-object evidence.
        # Environment collisions remain in the independent native safety audit.
        self.contacts.identity=ContactIdentityCache(PhysicsSchemaTools.intToSdfPath,
            lambda paths: registered_pair(paths,fingers,targets))

    def after_physics(self, report, time_s):
        self.contacts.ingest(report)
        self.observe(self.contacts.pairs(),time_s)


def rate_compatible(port, hz):
    """Never change workspace Hz to satisfy a legacy scorer's sampling contract."""
    if hasattr(port,'supported_physics_hz'):return hz in port.supported_physics_hz
    if getattr(port,'selection_mode','sector')=='contact':return hz in (60,240)
    return hz == getattr(port,'requested_physics_hz',getattr(port,'physics_hz',240))
