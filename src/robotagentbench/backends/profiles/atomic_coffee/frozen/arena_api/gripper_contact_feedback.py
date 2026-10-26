"""Task-independent read-only contact reporting for the four existing pads.

Keep contact-report instrumentation separate from optional force reduction.
Only a controller that consumes pad forces subscribes to the per-step Python
reduction. Other modes retain the same reports for independent task readers.
"""
import math
from .bimanual import GRIPPERS
from .contact_identity import ContactIdentityCache

PAD_BODIES = {f"/World/Robot/gripper_{hand}_{side}_link4": (joint, i)
              for hand,joint in zip(("l","r"), GRIPPERS)
              for i,side in enumerate(("inner","outer"))}


def enable_pad_contact_reports(stage):
    """Preserve existing measurement coverage, without force computation."""
    from pxr import Usd, UsdPhysics, PhysxSchema
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for path in PAD_BODIES:
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                raise RuntimeError("Missing physical gripper pad: " + path)
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)


def create_for_mode(stage, dt, mode):
    """No subscriber or reducer when the selected controller has no consumer.

    Unknown modes fail closed; adding a controller requires an explicit choice.
    Collision response and task-owned contact readers are never disabled here.
    """
    if mode in ('surface_stop.v1', 'contact_close_zero.v1'):
        return GripperContactFeedback(stage, dt)
    if mode not in ('close_zero.v1', 'close_ramp.v1', 'direct_target.v1'):
        raise ValueError('Unknown gripper force-feedback mode: ' + str(mode))
    enable_pad_contact_reports(stage)
    return None


class ContactReduction:
    def __init__(self):
        self.reset()

    def reset(self):
        self.forces = {joint:{} for joint in GRIPPERS}

    def add(self, joint, pad, body, normal, impulse, sign, dt):
        if not all(math.isfinite(x) for x in (*normal,*impulse,dt)) or dt <= 0:
            raise RuntimeError("Invalid contact-force sample")
        n = math.sqrt(sum(x*x for x in normal))
        if n < 1e-9:
            raise RuntimeError("Invalid contact normal")
        direction = [x/n for x in normal]
        load = abs(sum(x*y for x,y in zip(impulse,direction))) / dt
        if load <= 1e-9:
            return
        forces = self.forces[joint].setdefault(body, [[0.,0.,0.], [0.,0.,0.]])
        for k in range(3):
            forces[pad][k] += sign*direction[k]*load


class GripperContactFeedback:
    def __init__(self, stage, dt):
        from pxr import UsdPhysics, PhysicsSchemaTools
        from omni.physx import get_physx_simulation_interface
        self.stage, self.dt, self.UsdPhysics = stage, dt, UsdPhysics
        self.sample = 0
        self.reduction = ContactReduction()
        self.error = None
        self._bodies = {}
        self.identity = ContactIdentityCache(PhysicsSchemaTools.intToSdfPath, self.classify)
        enable_pad_contact_reports(stage)
        self.subscription = get_physx_simulation_interface().subscribe_contact_report_events(self.on_contacts)

    def body(self, path):
        if path not in self._bodies:
            p = self.stage.GetPrimAtPath(path)
            while p and not p.HasAPI(self.UsdPhysics.RigidBodyAPI):
                p = p.GetParent()
            if p:
                api = self.UsdPhysics.RigidBodyAPI(p)
                eligible = (api.GetRigidBodyEnabledAttr().Get() is not False
                            and api.GetKinematicEnabledAttr().Get() is not True)
                self._bodies[path] = str(p.GetPath()) if eligible else None
            else:
                self._bodies[path] = None
        return self._bodies[path]

    def classify(self, paths):
        for index,path in enumerate(paths):
            for pad,(joint,side) in PAD_BODIES.items():
                if path == pad or path.startswith(pad+'/'):
                    other = paths[1-index]
                    if other == '/World/Robot' or other.startswith('/World/Robot/'):
                        return None
                    body = self.body(other)
                    if body:
                        # Normal on the SAME pad actor, regardless of header order.
                        return joint, side, body, 1 if index == 0 else -1
        return None

    def before_step(self):
        self.reduction.reset()

    def on_contacts(self, headers, data):
        try:
            for h in headers:
                _,_,match = self.identity.get(h.stage_id,h.collider0,h.collider1)
                if match is None or not h.num_contact_data:
                    continue
                joint,pad,body,sign = match
                lo,hi = h.contact_data_offset,h.contact_data_offset+h.num_contact_data
                if lo < 0 or hi > len(data):
                    raise RuntimeError('Incomplete gripper contact report')
                for index in range(lo,hi):
                    p = data[index]
                    self.reduction.add(joint,pad,body,tuple(p.normal),tuple(p.impulse),sign,self.dt)
        except Exception as exc:
            # A Python exception in a native callback must not disappear silently.
            self.error = exc

    def after_step(self):
        if self.error is not None:
            raise RuntimeError('Gripper contact feedback failed') from self.error
        self.sample += 1

    @property
    def forces(self):
        return self.reduction.forces

    def close(self):
        self.subscription = None
        self.identity.clear()
        self._bodies.clear()
