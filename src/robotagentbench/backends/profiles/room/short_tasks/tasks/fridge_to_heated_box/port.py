"""Native one-port task 07, retaining the upstream scene/reader/mechanism.

Only the existing refrigerator helper may maintain its episode mechanism.
This adapter never writes the food box, doors, robot, or scene physics mode.
"""
from dataclasses import asdict
import json

from openend.tasks.scene_requirements.runtime_ports import FridgePort
from .evaluator import FridgeHeatingEvaluator
from .sources import verify_sources
from layout_task_activation_v1.task_scene_registration_v1.room_fridge_helper_v1 import RoomFridgeHelper


class FridgeHeatingPort(FridgePort):
    """Typed take/put task. Legacy class name retained for existing launches."""
    supported_physics_hz = (60, 240)

    def __init__(self, stage, manifest, output, episode_id, instance_id, parameters):
        verify_sources()
        self.fridge_helper = None
        self.heating = None
        self.heating_closed = False
        try:
            super().__init__(stage, manifest, output, episode_id, instance_id, parameters)
            fridge_rows = [row for row in manifest["objects"] if row["id"] == "refrigerator"]
            if len(fridge_rows) != 1:
                raise ValueError("One exact refrigerator instance required")
            self.fridge_root = fridge_rows[0]["path"]
            # Exactly the upstream room-owned original L1 door mechanism,
            # initialized once before PhysX attach. It does not award stages or
            # move the food box on behalf of a scorer.
            from openend.tasks.meal_tray.fridge_helper_dependencies import ensure
            ensure(manifest)
            self.fridge_helper = RoomFridgeHelper.prepare(stage, manifest)
            if self.fridge_helper is None:
                raise ValueError("Task 07 requires its declared refrigerator mechanism")
        except Exception:
            # A failed constructor is never inserted into root's ports map.
            # Close resources already acquired here before propagating failure.
            if self.fridge_helper is not None:
                self.fridge_helper.close()
            if getattr(self, "trace", None) is not None:
                self.trace.close()
            raise

    @property
    def ended(self):
        # The host settles the versioned task after its success observation.
        return self.heating_closed

    def initialize(self, clock, view):
        super().initialize(clock, view)
        from .contact_evidence import ContactEvidence
        self.source.contacts = ContactEvidence(self.source.contacts, self.source.decode,
            clock, self.output, self.fridge_root)

    def task_contact_roots(self):
        roots = [self.scene.target, dict(self.binding.objects)["plate"]]
        roots += [body for joint in self.scene.joints.values() for body in joint[0]]
        return tuple(dict.fromkeys(roots))

    def accept_initial(self):
        row = self.source.latest
        if row is None:
            raise RuntimeError("Task 07 requires actual initial native evidence")
        facts = row["facts"]
        checks = dict(box_in_refrigerator=not row["out"], box_not_held=not row["held"],
            fridge_closed=all(abs(a) <= 5. for a in row["angles"]),
            microwave_closed=facts.door_closed, box_not_initially_in_microwave=not facts.payload_inside)
        if not all(checks.values()):
            (self.output / "short_initial_acceptance.json").write_text(json.dumps(dict(accepted=False,
                checks=checks, evidence_source="native_physx"), indent=2) + "\n")
            raise RuntimeError("Task 07 document initial state rejected: " + str(checks))
        receipt = super().accept_initial()
        self.heating = FridgeHeatingEvaluator(self.composite)
        self.score = self.heating.observe_current(facts, self.clock.current_time)
        receipt = dict(accepted=True, checks=checks, fridge_put=receipt,
            scene_pose_writes=0, evidence_source="native_physx", native_physics_verified=False)
        (self.output / "short_initial_acceptance.json").write_text(json.dumps(receipt, indent=2) + "\n")
        return receipt

    def _mechanics(self, report):
        # Shared-host dormant() discovers this hook and advances it even when
        # the task's high-frequency source reader is not selected.
        if self.fridge_helper is not None:
            self._fridge_mechanics()

    def validate_physics_hz(self, hz):
        self.fridge_helper.validate_hz(hz)

    def set_physics_hz(self, hz):
        self.fridge_helper.set_hz(hz)

    def _fridge_mechanics(self):
        before = self.fridge_helper.controller.ns['FR_LATCH']['zone']
        self.fridge_helper.after_physics(self.clock.hz)
        after = self.fridge_helper.controller.ns['FR_LATCH']['zone']
        self.source.contacts.note_transition(before, after)

    def after_physics(self, helper_report):
        report = helper_report.get("_contact_report")
        if report is None:
            report = self.source.interface.get_contact_report()
        self._mechanics(report)
        super().after_physics(helper_report)

    def after_inactive(self, helper_report):
        report = helper_report.get("_contact_report")
        if report is None:
            report = self.source.interface.get_contact_report()
        self._mechanics(report)
        super().after_inactive(helper_report)

    def observe(self):
        from openend.tasks.meal_tray.physics_process_final_v1.clock_contract import validate_interval
        if not self.enabled:
            raise RuntimeError("Unselected task 07 control evidence")
        validate_interval(self.buffer, self.clock.hz, self.clock.current_time)
        frame = self.source.frame(self.episode, self.index + 1, self.buffer)
        if not self.composite.success:
            self.composite.observe(frame)
        self.index += 1
        self.score = self.heating.observe_current(self.source.latest["facts"], self.clock.current_time)
        self.trace.write(json.dumps(dict(facts=asdict(frame), score=self.score)) + "\n")
        self.trace.flush()
        self.discard_control()
        return self.score

    def pause(self):
        super().pause()
        if self.heating is not None:
            self.heating.pause()
            self.score = self.heating.summary()

    def summary_current(self):
        """Expose the Short Task summary, never the inherited put-only score."""
        if self.heating is None:
            return self.score
        if (not self.heating.ended and (self.heating.last_time != self.clock.current_time
                or self.heating.last_facts is None)):
            self.heating.observe_current(self._current_facts(), self.clock.current_time)
        self.score = self.heating.summary()
        return self.score

    def _current_facts(self):
        """A single settlement snapshot, never an extra physics step."""
        if self.source.latest["time_s"] == self.clock.current_time:
            return self.source.latest["facts"]
        from evaluation.tasks.microwave_put.physical_geometry import PLATE_BODY, rotation
        from evaluation.tasks.microwave_put.process_rules import PutFacts
        from evaluation.tasks.microwave_put.standards import door_state
        from layout_task_activation_v1.task_scene_registration_v1.fridge_to_microwave_v1.native_source import joint_angle
        poses = {}
        for path, view in self.source.views.items():
            p, q = view.get_world_pose()
            poses[path] = dict(position_m=p.tolist(), quaternion_wxyz=q.tolist())
        held, details = self.source.detector.observe(self.clock.current_time, poses)
        payload, plate = poses[self.scene.target], poses[PLATE_BODY]
        contacts = self.source.contacts.read(self.source.interface.get_contact_report(),
            rotation(plate["quaternion_wxyz"]).apply([0, 0, 1]))
        pp = dict(payload=payload, plate=plate)
        contents = self.scene.geometry.contents(pp)
        angle = joint_angle(self.scene.joints["microwave"], poses) * self.scene.geometry.opening_sign
        opened, closed = door_state(angle)
        target_held = bool(held[self.scene.target])
        return PutFacts(self.scene.variant, self.scene.target, opened, closed,
            contents["payload"]["inside"],
            bool(contacts.plate_load and self.scene.geometry.com_on_plate(pp, self.source.com)),
            target_held, bool(details[self.scene.target + ":r"]["confirmed"]),
            bool(contacts.released and not target_held))

    def finalize(self, reason):
        # Task settlement rechecks current measured box state even after the
        # inherited put scorer has ended. No physics step or pose write.
        self.discard_control()
        if self.heating is None:
            raise RuntimeError("No accepted task 07 baseline")
        if self.heating.last_time != self.clock.current_time or self.heating.last_facts is None:
            self.heating.observe_current(self._current_facts(), self.clock.current_time)
        self.score = self.heating.finalize(reason)
        self.heating_closed = True
        return self.score

    def close(self):
        if getattr(self, "fridge_helper", None) is not None:
            self.fridge_helper.close()
        if getattr(self, "source", None) is not None:
            self.source.close()
        if self.composite.last is not None:
            self.composite.finalize()
        self.trace.close()
