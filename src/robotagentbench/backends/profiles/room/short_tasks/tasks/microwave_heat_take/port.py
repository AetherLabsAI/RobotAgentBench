"""One native Short Task, using original physical readers and knob mechanism."""
from collections import deque
from copy import deepcopy
from dataclasses import asdict
import json
import math
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from evaluation.tasks.microwave_take import initial_state as old
from evaluation.tasks.microwave_take.process_rules import ProcessFrame
from evaluation.tasks.microwave_take.isaac_process import (
    IsaacProcessReader, ContactClassifier, carrying_facts, score_contacts,
    transform, under, supported,
)
from openend.tasks.meal_tray.microwave_device import MicrowaveDevice
from .registration import HeatBinding, bind_manifest, KNOB, INSTANCE
from .geometry import Geometry
from .illumination import HeatGate
from .interaction import KnobInteraction
from .process_rules import HeatTakeRules, HeatingEvidence
from .contacts import reader_state


class HeatTakePort:
    selection_mode = "sector"
    initial_workspace = "microwave"
    supported_physics_hz = (60, 240)
    physics_hz = 60

    def __init__(self, stage, manifest, output, episode_id, instance_id, parameters):
        bind_manifest(parameters, manifest)
        if instance_id != INSTANCE:
            raise ValueError("Wrong standalone instance")
        from pxr import Usd, UsdPhysics, PhysxSchema
        from omni.physx import get_physx_simulation_interface
        from evaluation.tasks.shelf_pick.physical_detection import DetectionConfig
        if not stage.GetSessionLayer().anonymous:
            raise ValueError("Anonymous native episode instrumentation required")
        self.stage, self.episode_id, self.instance_id = stage, episode_id, instance_id
        self.parameters = deepcopy(parameters)
        self.identity = HeatBinding(episode_id, instance_id, parameters["payload_variant"])
        self.geometry = Geometry(stage, parameters["payload_variant"])
        self.binding = SimpleNamespace(objects=dict(target=old.PLATE, payload=self.geometry.payload,
            source=old.MICROWAVE, knob=KNOB, stand=old.STAND))
        self.interface = get_physx_simulation_interface()
        self.output = Path(output); self.output.mkdir(exist_ok=False)
        self.trace = (self.output/"facts.jsonl").open("x")
        self.closed = False
        self.device = None
        try:
            self.device = MicrowaveDevice(stage)
            with Usd.EditContext(stage, stage.GetSessionLayer()):
                for root in (old.ROBOT, old.MICROWAVE, old.PLATE, self.geometry.payload):
                    for p in Usd.PrimRange(stage.GetPrimAtPath(root), Usd.TraverseInstanceProxies()):
                        if p.HasAPI(UsdPhysics.RigidBodyAPI):
                            a = p.GetAttribute("physxContactReport:threshold")
                            if a and a.Get() != 0.:
                                raise ValueError("Incompatible original contact reporter")
                            PhysxSchema.PhysxContactReportAPI.Apply(p).CreateThresholdAttr(0.)
            self.interaction = KnobInteraction(stage, episode_id, self.output/"interaction_progress.json")
        except BaseException:
            self.close(); raise
        self.config = DetectionConfig()
        self.history = deque(maxlen=math.ceil(self.config.initial_history_s*240)+2)
        self.contact_state = reader_state(stage, self.interface, self.geometry.payload)
        self.classifier = ContactClassifier([w.collider for w in self.geometry.walls], self.geometry.payload)
        self.pending, self.latest, self.composite, self.score = [], None, None, None
        self.sample_index = -1
        self.initial_bottom = None
        self.previous_held = False
        self.previous_payload = True

    @property
    def ended(self):
        return self.composite is not None and self.composite._finalized

    def task_contact_roots(self):
        return (old.MICROWAVE, old.PLATE, self.geometry.payload)

    def validate_physics_hz(self, hz):
        if hz not in self.supported_physics_hz:
            raise ValueError("Unknown workspace clock")

    def set_physics_hz(self, hz):
        self.validate_physics_hz(hz)
        self.physics_hz = hz
        if self.composite is not None:
            self.composite.configure_clock(1/hz)

    def initialize(self, clock, view):
        self.clock, self.last_time = clock, clock.current_time
        self.views = {}
        for role, path in (("plate", self.geometry.plate_body), ("payload", self.geometry.payload),
                           ("microwave", self.geometry.body)):
            v = view.create_rigid_body_view(path)
            if v.count != 1 or tuple(v.prim_paths) != (path,):
                raise ValueError("Wrong native body: "+path)
            self.views[role] = v
        self.payload_com = np.asarray(self.views["payload"].get_coms()).copy()[0,:3]
        if self.payload_com.shape != (3,) or not np.isfinite(self.payload_com).all():
            raise ValueError("Original native payload COM required")
        self.device.initialize(clock, view)
        self.interaction.initialize(view)

    def after_physics(self, helper):
        from openend.tasks.all_objects_integration.selective_read import contact_report
        if any(helper.get(k) for k in ("events", "docked_object_count", "tray_dock_enabled")):
            raise RuntimeError("Assisted evidence cannot score")
        now, dt = self.clock.current_time, self.clock.get_physics_dt()
        if abs(now-self.last_time-dt) > 1e-6:
            raise ValueError("Missing actual physical substep")
        report = contact_report(self, helper)
        # The original reader sees the same already-borrowed snapshot, never a
        # second World or a model-supplied contact stream.
        interface = self.interface
        self.contact_state.interface = SimpleNamespace(
            get_contact_report=lambda: report, is_sleeping=interface.is_sleeping)
        rows = IsaacProcessReader.contact_rows(self.contact_state)
        self.interaction.after_physics(report, admitted=self.composite is not None)
        poses = {}
        for role, view in self.views.items():
            p = np.asarray(view.get_transforms()).copy()[0]
            if p.shape != (7,) or not np.isfinite(p).all():
                raise ValueError("Invalid current body pose")
            poses[role] = dict(position_m=p[:3].tolist(), quaternion_wxyz=p[[6,3,4,5]].tolist())
        plate_rows, walls = self.classifier.split(rows)
        carry = carrying_facts(plate_rows, poses["plate"], poses["payload"],
            self.payload_com, self.geometry.plate_footprint, self.geometry.payload)
        bottom = float(transform(self.geometry.extrema["plate"], poses["plate"])[:,2].min())
        carry["plate_lifted_from_support"] = bool(self.initial_bottom is not None
            and bottom > self.initial_bottom+.0001 and not carry["plate_supported_by_stand"])
        drop = bool((self.previous_held and not carry["plate_held_by_robot"])
                    or (self.previous_payload and not carry["payload_on_plate"]))
        self.previous_held, self.previous_payload = carry["plate_held_by_robot"], carry["payload_on_plate"]
        self.latest = dict(time_s=now, physics_dt_s=dt, poses=poses, carry=carry,
            contents=self.geometry.contents(poses), door=self.device._measure()["door_angle_deg"],
            dropped=drop, contacts=score_contacts(walls, poses["microwave"]), raw=rows)
        self.history.append(self.latest)
        self.last_time = now
        if self.composite is not None:
            self.device.after_physics(report)
            self.sample_index += 1
            self.pending.append((self._frame(self.latest), self._heating()))

    def _frame(self, row):
        c, g = row["carry"], row["contents"]
        return ProcessFrame(self.episode_id, self.sample_index, row["time_s"], row["door"],
            g["plate"]["front_clearance_m"], g["payload"]["front_clearance_m"],
            c["plate_held_by_robot"], c["payload_on_plate"], row["dropped"],
            row["contacts"], True, "isaac_physics", old.TASK_ID,
            old.bindings_for(self.parameters["payload_variant"]),
            plate_grasped_by_robot=c["plate_grasped_by_robot"],
            plate_lifted_from_support=c["plate_lifted_from_support"])

    def _heating(self):
        g = self.device.gate
        return HeatingEvidence(self.episode_id, self.sample_index, self.clock.current_time,
            self.device.runtime.cycle.state, float(self.device.runtime.lamp.Get()),
            bool(self.device.runtime.cycle.lamp_on), bool(g.seen_running_lit),
            bool(g.completed), bool(g.completed),
            tuple(deepcopy(self.device.runtime.cycle.events)), not g.failed)

    def accept_initial(self):
        from openend.tasks.meal_tray.nonmotion_closeout_v1.initial_windows import initial_window
        if self.composite is not None or self.latest is None:
            raise RuntimeError("Initial state accepted once only")
        rows, complete = initial_window(self.history, self.config.initial_history_s, self.clock.get_physics_dt())
        raw = [v for r in rows for v in r["raw"]]
        checks = dict(history_complete=complete,
            closed=bool(rows) and all(abs(r["door"]) <= 2. for r in rows),
            both_inside=bool(rows) and all(all(v["inside"] for v in r["contents"].values()) for r in rows),
            supported_plate=bool(rows) and all(r["carry"]["plate_supported_by_stand"] for r in rows),
            supported_payload=bool(rows) and all(r["carry"]["payload_on_plate"] for r in rows),
            no_robot_contact=not any(any(under(x, old.ROBOT) for x in r["colliders"]) for r in raw),
            no_initial_interpenetration=bool(raw) and min(r["separation_m"] for r in raw) >= -.0001,
            stable_plate=bool(rows) and np.ptp([r["poses"]["plate"]["position_m"] for r in rows],axis=0).max() < .001,
            stable_payload=bool(rows) and np.ptp([r["poses"]["payload"]["position_m"] for r in rows],axis=0).max() < .001,
            lamp_off=self.device.runtime.lamp.Get() == 0.)
        receipt = dict(accepted=bool(all(checks.values())), checks={k:bool(v) for k,v in checks.items()},
                       scene_pose_writes=0, native_full_trajectory_verified=False)
        (self.output/"initial_acceptance.json").write_text(json.dumps(receipt,indent=2)+"\n")
        if not receipt["accepted"]:
            raise RuntimeError("Standalone microwave initial rejected: "+str(checks))
        self.device.accept_initial(self.identity, gate_factory=HeatGate)
        self.initial_bottom = float(transform(self.geometry.extrema["plate"], self.latest["poses"]["plate"])[:,2].min())
        self.latest["dropped"] = False
        self.sample_index = 0
        self.composite = HeatTakeRules(self.episode_id, self.clock.get_physics_dt(), self.geometry.walls,
            initial_state_accepted=True, payload_variant=self.parameters["payload_variant"])
        sample, _ = self.interaction.measured_sample(self.clock.current_time, baseline=True)
        self.composite.observe(self._frame(self.latest), self._heating(), sample)
        self.score = self._summary()
        return receipt

    def _summary(self):
        return dict(self.composite.summary(), instance_id=self.instance_id,
            public_runtime_id="microwave_heat_take", evidence_status="native_physx",
            native_runtime_connected=True, native_full_trajectory_verified=False,
            physical_validation="native_initial_accepted")

    def consume_control(self, active):
        if type(active) is not bool or len(self.pending) != self.clock.hz//20:
            raise ValueError("One full actual control interval required")
        sample, rebase = self.interaction.measured_sample(self.clock.current_time)
        if rebase:
            raise ValueError("Unobserved resumed interaction cannot create task progress")
        for i, (f,h) in enumerate(self.pending):
            self.composite.observe(f,h,sample if i == len(self.pending)-1 else None)
        self.pending = []
        self.score = self._summary()
        self.trace.write(json.dumps(dict(time_s=self.clock.current_time, score=self.score))+"\n")
        self.trace.flush()
        return self.score

    def pause(self):
        if self.composite is not None:
            self.composite.pause()

    def finalize(self, reason):
        if self.pending:
            raise ValueError("Cannot settle an incomplete physical interval")
        self.composite.finalize()
        self.score = dict(self._summary(), termination_reason=reason)
        return self.score

    def close(self):
        if not getattr(self, "closed", True):
            if self.device is not None:
                self.device.close()
            self.trace.close()
            self.closed = True
