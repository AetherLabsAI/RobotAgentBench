"""Opt-in notice-invalidated guard; same validator, no physics authoring.

The default worker remains exhaustive. The recorded-replay profile may reuse
a successful audit until USD changes. Only native dynamic-body pose/velocity
updates and the exact renderer dispatch-clock default value are ignored;
topology, material, collision, static transforms, drives,
bindings, schemas and unknown changes always invalidate the audit.
"""
from time import perf_counter

# Exact renderer-owned dispatch clock value, not the whole /Render subtree.
# A value update schedules image readback; it is not a physics parameter.
RENDER_CLOCK = '/Render/PostProcess/SDGPipeline/DispatchSync.inputs:simTimesToWrite'


class PhysicsGuard:
    def __init__(self, stage, validate):
        from pxr import Tf, Usd, UsdPhysics
        self.stage = stage
        self.validate = validate
        self.dirty = True
        self.checks = self.audits = self.invalidations = 0
        self.audit_seconds = 0.0
        self.invalidation_examples = {}
        # Compute before subscribing. Topology/configuration edits invalidate
        # this set too, and rebuild it only after a successful full audit.
        self.dynamic_paths = set()
        self.joint_paths = set()
        self._rebuild_dynamic_paths()
        self.notice = Tf.Notice.Register(Usd.Notice.ObjectsChanged, self._changed, stage)

    def _rebuild_dynamic_paths(self):
        from pxr import UsdPhysics
        self.dynamic_paths = set()
        self.joint_paths = set()
        for prim in self.stage.Traverse():
            path=str(prim.GetPath())
            if (prim.HasAPI(UsdPhysics.RigidBodyAPI)
                    and prim.GetAttribute('physics:rigidBodyEnabled').Get() is not False
                    and prim.GetAttribute('physics:kinematicEnabled').Get() is not True):
                self.dynamic_paths.add(path)
            if prim.IsA(UsdPhysics.Joint):
                self.joint_paths.add(path)

    def _changed(self, notice, sender):
        resynced = notice.GetResyncedPaths()
        changed = notice.GetChangedInfoOnlyPaths()
        invalidate = bool(resynced)
        reason = str(resynced[0]) if resynced else None
        for path in changed:
            fields = set(notice.GetChangedFields(path))
            if (path.IsPropertyPath() and str(path) == RENDER_CLOCK and fields
                    and fields == {'default'}):
                continue
            name = path.name
            dynamic = str(path.GetPrimPath()) in self.dynamic_paths
            # xformOpOrder and prim metadata are NOT whitelisted.
            ordinary_motion = path.IsPropertyPath() and dynamic and (
                (name.startswith('xformOp:') and name.split(':')[1] in (
                    'translate', 'orient', 'transform', 'rotateXYZ'))
                or name in ('physics:velocity', 'physics:angularVelocity'))
            # Native measured joint state, NOT drive targets/limits/gains.
            ordinary_motion = ordinary_motion or (path.IsPropertyPath()
                and str(path.GetPrimPath()) in self.joint_paths and name in (
                    'state:angular:physics:position', 'state:angular:physics:velocity',
                    'state:linear:physics:position', 'state:linear:physics:velocity'))
            if not ordinary_motion:
                invalidate = True
                reason = str(path)
                break
        if invalidate:
            self.dirty = True
            self.invalidations += 1
            if reason in self.invalidation_examples or len(self.invalidation_examples) < 16:
                self.invalidation_examples[reason] = self.invalidation_examples.get(reason, 0) + 1

    def check(self, force=False):
        self.checks += 1
        if force or self.dirty:
            start = perf_counter()
            self.validate()  # exceptions stay fatal; never repair/mask changes
            self._rebuild_dynamic_paths()
            self.audit_seconds += perf_counter() - start
            self.audits += 1
            self.dirty = False

    def report(self):
        return dict(mode='usd_notice_invalidated', checks=self.checks,
                    full_audits=self.audits, invalidations=self.invalidations,
                    audit_wall_seconds=self.audit_seconds, invalidation_examples=self.invalidation_examples)

    def close(self):
        self.notice.Revoke()
