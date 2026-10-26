"""Task-local dynamic-machine adaptation of the original atomic 1-2 reader.

Same contact ingestion, grasp detector, bean debounce, clocks and Facts. The
only changed geometry assumption is that the native MACHINE_BODY pose now
defines the machine frame. No source patch, scene pose update or stale-anchor
guard bypass is used. API 1-2 only: no liquid/device variants are accepted.
"""
import json
import numpy as np
from evaluation.tasks.coffee_machine.isaac_evidence import (
    CoffeeReader, PHYSICS_DT, rotation, under, serializable, CAMERA_ROOTS, camera_contacts)
from evaluation.tasks.coffee_machine.specs import MACHINE, MACHINE_BODY, BAG, CUP, BUTTON, ROBOT
from .moving_frame import MovingMachineFrame, pose_matrix, mouth_above_opening
from .trigger import TriggerFacts


class MovingCoffeeReader(CoffeeReader):
    def __init__(self, stage, output, episode_id):
        from pxr import UsdGeom
        if stage.GetPrimAtPath(MACHINE_BODY).GetAttribute('physics:kinematicEnabled').Get() is not False:
            raise ValueError('Approved task-owned dynamic coffee body required')
        super().__init__(stage, output, episode_id, '1-2')
        cache = UsdGeom.XformCache()
        root = np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(MACHINE)), dtype=float).T
        body = np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(MACHINE_BODY)), dtype=float).T
        self.live_machine = MovingMachineFrame(np.linalg.inv(root) @ body)
        from .geometry import bag_mouth_local
        self.bag_mouth = bag_mouth_local(stage)

    def initialize(self, world):
        super().initialize(world)
        from evaluation.physics_readers import ReadOnlyRigidBody
        view = ReadOnlyRigidBody(prim_path=MACHINE_BODY)
        view.initialize()
        self.views[MACHINE_BODY] = view

    def after_step(self):
        # Adaptation of original after_step, API 1-2-specific. Same substep clock,
        # contact validation, detector and bean controller; native moving frame
        # replaces only the original immutable USD-root matrix assertion.
        if self.closed:
            raise RuntimeError('Reader closed')
        now = float(self.world.current_time)
        if abs(now - self.last_time - PHYSICS_DT) > 1e-6:
            raise RuntimeError('Missing physical substep')
        poses = {}
        for path, view in self.views.items():
            position, quaternion = view.get_world_pose()
            if np.asarray(position).shape != (3,) or not np.isfinite(position).all():
                raise ValueError('Invalid physical position')
            rotation(quaternion)
            poses[path] = dict(position_m=[float(v) for v in position],
                               quaternion_wxyz=[float(v) for v in quaternion])
        body = poses[MACHINE_BODY]
        self.live_machine.update(body['position_m'], body['quaternion_wxyz'], now)
        matrix = self.live_machine.current(now)
        self.machine_rotation = matrix[:3, :3].T
        self.machine_origin = matrix[:3, 3].copy()
        # Atomic placement and hopper methods now share this same physical frame.
        self.anchor = self.machine_point(self.contract.cup)
        self.outlet = self.machine_point(self.contract.outlet)
        self.contacts.ingest(self.interface.get_contact_report())
        if self.diag is not None:
            rows = []
            for paths, samples in self.contacts.pairs():
                if any(under(path, ROBOT) for path in paths):
                    rows.append(dict(a=paths[0], b=paths[1], impulse=max(s for _, s in samples), n=len(samples)))
            if rows:
                self.diag.write(json.dumps(dict(step=self.step + 1, time_s=now, pairs=rows)) + '\n')
        contact, bilateral, robot_machine, robot_on_button, button_impulse = self.summarize()
        grasped, grasp_details = self.detector.observe(now, poses, bilateral)
        displacement = float(self.machine_local(poses[BUTTON]['position_m'])[1])
        beans = self.machine_local(self.beans.positions())
        pressed_before = self.controller.pressed
        device = self.controller.update(PHYSICS_DT, {i: tuple(v) for i, v in enumerate(beans)}, displacement)
        if displacement < -0.001 and robot_on_button:
            self.press_by_robot = True
        if device['event'] is not None:
            self.window['events'].append(device['event'])
            self.window['event_by_robot'].append(bool(self.press_by_robot))
        if pressed_before and not self.controller.pressed:
            self.press_by_robot = False
        if displacement >= -0.001 and not robot_on_button:
            self.press_by_robot = False
        device = dict(bean_count=device['bean_count'], raw_bean_count=device['raw_bean_count'],
                      dispensing=device['dispensing'], event=device['event'], pressed=device['button_pressed'],
                      emitted_now=0, emitted_total=0)
        for obj in self.objects:
            self.window['ground'][obj] |= contact[obj]['ground']
            self.window['machine'][obj] |= contact[obj]['machine']
        self.window['robot_machine'] |= robot_machine
        self.window['robot_on_button'] |= robot_on_button
        self.step += 1
        self.latest = dict(time_s=now, reader_step=self.step, poses=poses,
            contacts={obj: dict(robot=contact[obj]['robot'], supports=sorted(contact[obj]['supports']),
                ground=contact[obj]['ground'], machine=contact[obj]['machine'], bilateral=bilateral[obj],
                grasped=bool(grasped[obj])) for obj in self.objects},
            tilt_deg={obj: self.tilt(poses[obj]['quaternion_wxyz']) for obj in self.objects},
            camera_contacts=camera_contacts(self.contacts.pairs()), robot_machine=robot_machine,
            robot_on_button=robot_on_button, button_impulse=button_impulse,
            button_displacement_m=displacement, device=device, grasp_details=grasp_details,
            measured_machine_world_matrix=matrix.tolist(), machine_frame_source='native_dynamic_body')
        self.history.append(self.latest)
        self.trace.write(json.dumps(serializable(self.latest), allow_nan=False) + '\n')
        self.last_time = now

    def trigger_facts(self):
        row = self.latest
        if row is None or row['time_s'] != float(self.world.current_time) or row['time_s'] != self.last_time:
            raise RuntimeError('Fresh fully validated physical frame required')
        bag = row['poses'][BAG]
        bag_matrix = pose_matrix(bag['position_m'], bag['quaternion_wxyz'])
        mouth = bag_matrix[:3, :3] @ self.bag_mouth + bag_matrix[:3, 3]
        return TriggerFacts(time_s=row['time_s'], holding_bag=row['contacts'][BAG]['grasped'],
                            mouth_above_opening=mouth_above_opening(
                                mouth, self.live_machine.current(row['time_s']), self.contract.hopper),
                            bag_tilt_deg=row['tilt_deg'][BAG],
                            contacts_complete=self.contacts.complete_for_roots(BAG))
