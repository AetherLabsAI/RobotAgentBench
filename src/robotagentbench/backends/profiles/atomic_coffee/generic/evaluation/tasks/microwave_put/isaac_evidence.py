"""Isaac pose/contact bridge. Read-only: refuses incomplete instrumentation."""
from dataclasses import dataclass, asdict
import numpy as np

from arena_api.episode_clock import ControlClock
from arena_api.task_runtime import TaskEvidence
from .physical_geometry import (PutGeometry, report_coverage, PAYLOAD, PLATE_BODY,
                                 MICROWAVE_ROOT, rotation)
from .physical_contacts import PutContactStream
from .process_rules import PutFacts, PutProgressTracker
from .scoring import ScoringFrame
from .standards import door_state
from .payload_binding import payload_for


@dataclass(frozen=True)
class MeasuredPutSample:
    time_s: float
    facts: PutFacts
    wall_impact: bool
    details: dict


def final_facts(facts):
    return bool(facts.door_closed and facts.payload_inside and facts.payload_supported
                and not facts.held_by_robot and facts.released is True)


class EvidenceBridge:
    """Control-boundary snapshots plus substep interruption/impact accumulation."""
    def __init__(self, episode_id, task_id, execution, *, payload_variant="beef_box"):
        self.episode_id, self.task_id = episode_id, task_id
        self.execution = execution
        self.tracker = PutProgressTracker(episode_id=episode_id, task_id=task_id,
            payload_variant=payload_variant, payload_instance=payload_for(payload_variant),
            control_dt_s=execution["control_dt_s"])
        self.current = None
        self.pending = []
        self.last_sample = None

    def begin(self, sample):
        if self.current is not None:
            raise RuntimeError("Episode evidence already started")
        if sample.facts.released is not False or not sample.facts.held_by_right:
            raise ValueError("Initial right-hand physical hold is required")
        if sample.facts.payload_inside:
            raise ValueError("Put initial payload must start outside the cavity")
        raw = TaskEvidence(self.episode_id, self.task_id, 0, sample.time_s, sample.facts)
        progress = self.tracker.update(raw)
        self.current = TaskEvidence(self.episode_id, self.task_id, 0, sample.time_s,
                                   ScoringFrame(progress, False, False))
        self.last_sample = sample
        return self.current

    def append(self, sample):
        if self.current is None:
            raise RuntimeError("Initial acceptance must precede task sampling")
        if len(self.pending) >= self.execution["physics_substeps"]:
            raise RuntimeError("Missed control-boundary evidence collection")
        interval = ControlClock(1, self.execution["physics_dt_s"])
        interval.start(self.last_sample.time_s)
        interval.observe(sample.time_s)
        if type(sample.wall_impact) is not bool or type(sample.facts.released) is not bool:
            raise ValueError("Missing measured impact/release bit")
        self.tracker._validate(TaskEvidence(self.episode_id, self.task_id,
            self.current.step_index+1, sample.time_s, sample.facts))
        self.pending.append(sample)
        self.last_sample = sample

    def collect(self):
        if self.current is None:
            raise RuntimeError("No accepted initial evidence")
        if not self.pending:
            return self.current
        if len(self.pending) != self.execution["physics_substeps"]:
            raise RuntimeError("Incomplete physical interval")
        sample = self.pending[-1]
        raw = TaskEvidence(self.episode_id, self.task_id, self.current.step_index+1,
                           sample.time_s, sample.facts)
        progress = self.tracker.update(raw)
        self.current = TaskEvidence(self.episode_id, self.task_id, raw.step_index, raw.time_s,
            ScoringFrame(progress, any(s.wall_impact for s in self.pending),
                          any(not final_facts(s.facts) for s in self.pending)))
        self.pending.clear()
        return self.current


class IsaacPutEvidence:
    def __init__(self, stage, episode_id, task_id, execution, *, payload_variant="beef_box"):
        self.variant, self.payload = payload_variant, payload_for(payload_variant)
        self.coverage = report_coverage(stage, payload=self.payload)
        missing = [r["path"] for r in self.coverage if not r["complete"]]
        if missing:
            raise RuntimeError("Missing zero-threshold contact reporting; no settings changed: " + ", ".join(missing))
        self.stage, self.execution = stage, execution
        self.geometry = PutGeometry(stage, payload_variant=payload_variant)
        self.bridge = EvidenceBridge(episode_id, task_id, execution, payload_variant=payload_variant)
        self.sample = None
        self.stream = None
        self.started = False
        self.settle_samples = 0

    @staticmethod
    def pose(view):
        pos, quat = view.get_world_pose()
        rotation(quat)
        if not np.isfinite(pos).all():
            raise ValueError("Nonfinite physical position")
        return dict(position_m=np.asarray(pos).tolist(), quaternion_wxyz=np.asarray(quat).tolist())

    def initialize(self, world):
        from isaacsim.core.prims import SingleArticulation
        from evaluation.physics_readers import ReadOnlyRigidBody
        from omni.physx import get_physx_simulation_interface
        from omni.physx.bindings._physx import ContactEventType
        from pxr import PhysicsSchemaTools, UsdPhysics
        self.world = world
        self.interface = get_physx_simulation_interface()
        self.microwave = SingleArticulation(prim_path=MICROWAVE_ROOT+"/Asset", name="put_evidence_microwave")
        self.microwave.initialize()
        if self.microwave.dof_names != ["door_hinge"]:
            raise ValueError("Unknown microwave articulation joint")
        self.views = {}
        for role, path in (("payload",self.payload),("plate",PLATE_BODY)):
            v = ReadOnlyRigidBody(prim_path=path)
            v.initialize()
            self.views[role] = v
        self.com = np.asarray(self.views["payload"].get_com()[0]).reshape(3)
        if not np.isfinite(self.com).all():
            raise ValueError("Invalid physical payload COM")
        def sleeping(stage_id, actor):
            p = self.stage.GetPrimAtPath(PhysicsSchemaTools.intToSdfPath(actor))
            if not p:
                raise RuntimeError("Unresolved physical actor")
            return bool(not p.HasAPI(UsdPhysics.RigidBodyAPI) or self.interface.is_sleeping(stage_id,actor))
        self.stream = PutContactStream(self.geometry, PhysicsSchemaTools.intToSdfPath, sleeping,
            (ContactEventType.CONTACT_FOUND, ContactEventType.CONTACT_PERSIST, ContactEventType.CONTACT_LOST),
            payload_variant=self.variant)
        self.last_time = float(world.current_time)

    def after_step(self):
        now = float(self.world.current_time)
        interval = ControlClock(1, self.execution["physics_dt_s"])
        interval.start(self.last_time)
        interval.observe(now)
        poses = {role:self.pose(view) for role,view in self.views.items()}
        contents = self.geometry.contents(poses)
        plate_up = rotation(poses["plate"]["quaternion_wxyz"]).apply([0,0,1])
        contacts = self.stream.read(self.interface.get_contact_report(), plate_up)
        angle = float(np.degrees(self.microwave.get_joint_positions()[0])) * self.geometry.opening_sign
        opened, closed = door_state(angle)
        on_plate = self.geometry.com_on_plate(poses, self.com)
        facts = PutFacts(self.variant, self.payload, opened, closed, contents["payload"]["inside"],
            bool(contacts.plate_load and on_plate), contacts.held_by_robot, contacts.held_by_right,
            contacts.released)
        self.sample = MeasuredPutSample(now, facts, contacts.wall_impact,
            dict(poses=poses, containment=contents, door_angle_deg=angle, contacts=asdict(contacts),
                 payload_com_over_plate=on_plate))
        self.last_time = now
        if self.started:
            self.bridge.append(self.sample)
        else:
            self.settle_samples += 1

    def begin_episode(self):
        if self.sample is None or not self.settle_samples:
            raise RuntimeError("No measured initial physics samples")
        if self.started:
            raise RuntimeError("Episode already started")
        if self.sample.time_s != float(self.world.current_time):
            raise RuntimeError("Stale initial physics sample")
        if not self.sample.details["containment"]["plate"]["inside"]:
            raise ValueError("Initial support plate is not completely inside the microwave")
        if self.sample.details["contacts"]["plate_supported_by_stand"] is not True:
            raise ValueError("Initial plate lacks actual support from the existing stand")
        result = self.bridge.begin(self.sample)
        self.started = True
        return result

    def collect_evidence(self):
        if self.sample is None or self.sample.time_s != float(self.world.current_time):
            raise RuntimeError("Stale physical evidence")
        return self.bridge.collect()

    def close(self):
        if self.stream is not None:
            self.stream.close()
