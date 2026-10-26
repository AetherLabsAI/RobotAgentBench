"""Task-local binding of the existing read-only interaction measurements."""
from pathlib import Path

from short_tasks.tasks.interaction_progress_v1.native import NativeProgress
from short_tasks.tasks.interaction_progress_v1.scoring import Progress
from .process_rules import KNOB


class KnobInteraction(NativeProgress):
    def __init__(self, stage, episode_id, output):
        from pxr import Usd, UsdPhysics, PhysxSchema, PhysicsSchemaTools
        from omni.physx.bindings._physx import ContactEventType
        from short_tasks.tasks.interaction_progress_v1.geometry import collision_surfaces
        from short_tasks.tasks.interaction_progress_v1.contact import ContactEdges
        self.output = Path(output)
        self.bindings = {KNOB: dict(roots=(KNOB,), owner="heat_take", role="target")}
        self.surfaces = {KNOB: collision_surfaces(stage, (KNOB,))}
        self.progress = Progress(episode_id, (KNOB,))
        self.pads = {h: tuple("/World/Robot/gripper_"+h+"_"+side+"_link4"
                             for side in ("inner", "outer")) for h in ("l", "r")}
        groups = {h: [] for h in ("l", "r")}
        for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/Robot"), Usd.TraverseInstanceProxies()):
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            for hand in groups:
                if any(prim.GetName().startswith("gripper_"+hand+"_"+side+"_link")
                       for side in ("inner", "outer")):
                    threshold = prim.GetAttribute("physxContactReport:threshold")
                    if not threshold:
                        if not stage.GetSessionLayer().anonymous:
                            raise ValueError("Anonymous contact instrumentation required")
                        # Same observational reporter used by the original
                        # portable-task readers. No collision/drive edits.
                        with Usd.EditContext(stage, stage.GetSessionLayer()):
                            threshold = PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
                    if threshold.Get() != 0.:
                        raise ValueError("Incompatible existing finger contact reporter")
                    groups[hand].append(str(prim.GetPath()))
        if not all(groups.values()):
            raise ValueError("Both native finger contact groups required")
        self.contacts = ContactEdges(self.bindings, groups, PhysicsSchemaTools.intToSdfPath,
            (ContactEventType.CONTACT_FOUND, ContactEventType.CONTACT_PERSIST,
             ContactEventType.CONTACT_LOST))
        self.events, self.previous, self.views = set(), {}, {}
        self.index, self.rebase_needed, self.last_written = -1, False, -1

    def eligible(self):
        return {KNOB: True}

    def measured_sample(self, time_s, *, baseline=False):
        rebase = self.rebase_needed
        self.sample(time_s, baseline=baseline)
        return self.progress.last[2][KNOB], rebase
