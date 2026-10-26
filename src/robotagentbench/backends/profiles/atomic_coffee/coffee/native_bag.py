"""Lifecycle extraction of the existing coffee-bag preview actor.

Same 10cm trigger, contact closure, 1.5s motion, temporary HUMAN constraint,
collision policy and training-domain admission. No robot/target pose commands.
This class supplies no score and never steps the owner's world.
"""
import numpy as np
import warnings


class NativeBagRelocation:
    def __init__(self, stage, initial):
        from evaluation.tasks.coffee_machine.specs import BAG
        from short_task_robustness.tasks.stock_wooden_tray_interference.adaptive_hand import AdaptiveHand
        from short_task_robustness.tasks.stock_wooden_tray_interference.human_asset import NativeHuman, TemporaryHumanGrasp
        from short_task_robustness.tasks.stock_wooden_tray_interference.contact_closure import ContactClosure
        from short_task_robustness.tasks.coffee_bag_interference.human_collision import CoffeeHumanPairs
        from short_task_robustness.tasks.coffee_bag_interference.motion import RelocationMotion
        from short_task_robustness.tasks.coffee_bag_interference.preview_geometry import plan
        if 'relocation_pair' not in initial:
            raise ValueError('Paired bag initial required, never invent a runtime destination')
        self.stage, self.target_path = stage, BAG
        self.asset = AdaptiveHand()
        self.reservation = plan(stage, self.asset, initial['seed'], initial)
        self.route_options = initial.get('relocation_options')
        self.route_geometry = None
        self.route_wait_event = None
        self.motion = RelocationMotion(*self.reservation['waypoints'])
        self.matrix = np.eye(4)
        self.matrix[:3, :3] = self.reservation['orientation']
        self.matrix[:3, 3] = self.motion.points[0]
        self.human = NativeHuman(stage, '/World/CoffeeBagHuman', self.matrix, self.asset)
        self.absence = self.human.wait_for_trigger()
        self.grasp = TemporaryHumanGrasp(stage, self.human.root + '/ExternalGrasp',
                                        self.human.links['hand.R'], BAG)
        self.release_filter = CoffeeHumanPairs(stage, self.human.root, list(self.human.links.values()), BAG)
        self.closure = ContactClosure()
        self.last_closure = {digit: 0. for digit in self.closure.last}
        self.reader = self.target = self.facts = self.trigger = None
        self.start = None
        self.retired = self.released = self.completed = self.armed = False
        self.pairs = {}
        self.digits = []
        self.events = []
        self.completion_evidence = None
        self.contact_gap_during_motion=False
        self.intervention_evidence_unknown=False

    def bind(self, simulation_view, reader):
        from short_task_robustness.tasks.coffee_bag_interference.proximity import NativeCoffeeBagTriggerFacts
        self.reader = reader
        self.human.bind(simulation_view)
        self.target = simulation_view.create_rigid_body_view(self.target_path)
        if self.target.count != 1:
            raise RuntimeError('Exact unique native bag required')
        self.facts = NativeCoffeeBagTriggerFacts(self.stage, simulation_view, reader)
        if self.route_options is not None:
            from short_task_robustness.tasks.coffee_bag_interference.route_options import LiveRouteGeometry
            self.route_geometry = LiveRouteGeometry(self.stage, simulation_view, self.release_filter.grippers)

    def arm(self, *, robot_native_poses):
        from short_task_robustness.tasks.coffee_bag_interference.preview_geometry import robot_visual_boxes
        from short_task_robustness.tasks.stock_wooden_tray_interference.retreat_geometry import check_transport
        from short_task_robustness.tasks.coffee_bag_interference.proximity import CoffeeBagApproachTrigger
        if self.reader is None or self.armed:
            raise RuntimeError('Bind then arm bag actor exactly once after initial acceptance')
        if self.route_options is not None:
            from short_task_robustness.tasks.coffee_bag_interference.route_options import ReservedRouteTrigger
            self.trigger = ReservedRouteTrigger(self.target_path, self.route_options)
            self.trigger.arm()
            self.armed = True
            return
        boxes = robot_visual_boxes(self.stage, self.release_filter.grippers, robot_native_poses)
        clear, reason = check_transport(self.motion, self.asset, self.matrix[:3, :3], boxes,
            self.reservation['fixture_boxes'], self.reservation['target_bounds'])
        if not clear:
            raise RuntimeError('Settled native robot blocks bag route: ' + str(reason))
        self.trigger = CoffeeBagApproachTrigger(self.target_path, reservation=self.reservation)
        self.trigger.arm()
        self.armed = True

    def _current(self):
        row = self.reader.latest
        if row is None or row['time_s'] != float(self.reader.world.current_time) or row['time_s'] != self.reader.last_time:
            raise RuntimeError('Fresh validated native bag frame required')
        return row

    def command_before_step(self, time_s):
        if self.start is None or self.retired:
            return {'phase': 'absent'}
        from short_task_robustness.tasks.coffee_bag_interference.acquisition import acquisition_ready
        from short_task_robustness.tasks.stock_wooden_tray_interference.release_collision import discard_filtered_pairs
        from evaluation.tasks.coffee_machine.facts import COUNTER
        row = self._current()
        if float(time_s) != row['time_s']:
            raise RuntimeError('Actor command must use the current physical clock')
        sample = self.motion.sample(float(time_s) - self.start)
        contact = any(self.target_path in pair and any(link in pair for link in self.human.links.values())
                      for pair in self.pairs.values())
        if acquisition_ready(phase=sample.phase, already_acquired=self.grasp.ever_acquired,
                measured_contact=contact, opposed_contact=self.closure.ready(self.digits),
                closure_values=self.last_closure):
            self.grasp.acquire(self.human.views['hand.R'].get_transforms()[0], self.target.get_transforms()[0],
                hand_target_contact=contact, hand_closed=self.closure.ready(self.digits))
            self.events.append(dict(kind='acquired', time_s=time_s, digits=list(self.digits)))
        if sample.phase == 'carry' and not self.grasp.ever_acquired:
            if not self.contact_gap_during_motion:
                raise RuntimeError('No carrying without measured opposed-contact acquisition')
            self.intervention_evidence_unknown=True
        if sample.phase == 'release' and self.grasp.active:
            self.grasp.release()
            self.released = True
        if self.release_filter.update(phase=sample.phase, released=self.released, constraint_active=self.grasp.active,
                previous_closure=self.last_closure, supported=COUNTER in row['contacts'][self.target_path]['supports']):
            discard_filtered_pairs(self.reader.contacts.active, list(self.human.links.values()), self.target_path)
            self.reader.contacts.identity.clear()
            self.pairs = {key: pair for key, pair in self.pairs.items()
                          if not (self.target_path in pair and any(link in pair for link in self.human.links.values()))}
        self.matrix[:3, 3] = sample.wrist_position
        self.last_closure = self.closure.command(sample.closure, sample.phase)
        self.human.command(self.matrix, self.last_closure)
        return dict(phase=sample.phase, bag_pose_commands=False)

    def _contacts(self):
        from pxr import PhysicsSchemaTools
        from omni.physx.bindings._physx import ContactEventType
        from short_task_robustness.tasks.stock_wooden_tray_interference.contact_closure import digit_for_bone
        for header in self.reader.interface.get_contact_report()[0]:
            key = tuple(sorted((int(header.actor0), int(header.actor1))))
            if 0 in key:
                continue  # Same native preview policy; strict atomic reader validates its own contacts.
            pair = tuple(str(PhysicsSchemaTools.intToSdfPath(value)) for value in key)
            if not any(path.startswith(self.human.root + '/') for path in pair):
                continue
            if header.type == ContactEventType.CONTACT_LOST:
                self.pairs.pop(key, None)
            else:
                self.pairs[key] = pair
        self.digits = sorted({digit_for_bone(bone) for bone, path in self.human.links.items()
            if digit_for_bone(bone) and any(path in pair and self.target_path in pair for pair in self.pairs.values())})
        return [pair for pair in self.pairs.values() if any(path.startswith('/World/Robot/') for path in pair)]

    def _validate_completion(self, row):
        from evaluation.tasks.coffee_machine.facts import COUNTER
        from short_task_robustness.tasks.coffee_bag_interference.training_domain import contains_origin
        from short_task_robustness.tasks.coffee_bag_interference.joint_initialization import direction_check
        if self.grasp.active or not self.grasp.ever_acquired or not self.released:
            raise RuntimeError('Bag relocation did not acquire/release through native contact')
        position = row['poses'][self.target_path]['position_m']
        reference = self.trigger.locked_direction
        if reference is None:
            raise RuntimeError('Missing frozen triggering-jaw direction')
        direction = direction_check(reference['start_origin_m'], position, reference['reference_position_m'])
        supported = COUNTER in row['contacts'][self.target_path]['supports']
        in_domain = contains_origin(position, self.reservation['training_domain'])
        if not (supported and in_domain and direction['allowed']):
            raise RuntimeError('Native bag completion rejected: support/domain/away direction')
        return dict(time_s=row['time_s'], supported=supported, in_training_domain=in_domain,
                    acquired=self.grasp.ever_acquired, released=self.released, constraint_active=False,
                    direction=direction, beans_in_bag=self.reader.beans_in_bag())

    def observe(self, *, robot_native_poses):
        if not self.armed:
            raise RuntimeError('Initial native acceptance must arm bag actor first')
        row = self._current()
        if self.retired:
            return dict(start=False, reason='bag_human_retired', time_s=row['time_s'])
        collisions = self._contacts()
        if not self.reader.contacts.complete_for_roots(self.target_path,self.human.root):
            if not self.contact_gap_during_motion:
                warnings.warn('Coffee human contact evidence missing; continuing without '
                              'claiming acquisition, safe separation, or completion.',RuntimeWarning)
            self.contact_gap_during_motion|=self.start is not None
            self.pairs.clear();self.digits=[]
            return dict(start=False,reason='waiting_for_contact_evidence',time_s=row['time_s'])
        if self.start is None:
            if self.route_options is None:
                event = self.trigger.observe(self.facts.read(), route_clear=True)
            else:
                obstacles = None
                def clear(route):
                    nonlocal obstacles
                    if obstacles is None:
                        obstacles = self.route_geometry.snapshot(robot_native_poses)
                    return self.route_geometry.check(route, obstacles)
                event = self.trigger.observe(self.facts.read(), route_clear=clear)
                if event['reason'] == 'no_currently_clear_away_reserved_route':
                    if self.route_wait_event is None:
                        self.route_wait_event = dict(kind='route_selection_wait', observations=0,
                            first_time_s=row['time_s'])
                        self.events.append(self.route_wait_event)
                    self.route_wait_event.update(last_time_s=row['time_s'], latest_decision=event,
                        observations=self.route_wait_event['observations']+1)
            if event['start']:
                if self.route_options is not None:
                    from short_task_robustness.tasks.coffee_bag_interference.motion import RelocationMotion
                    self.reservation = self.trigger.selected['reservation']
                    self.motion = RelocationMotion(*self.reservation['waypoints'])
                self.human.activate_on_trigger()
                self.start = row['time_s']
                self.events.append(dict(event, kind='started'))
            return event
        phase = self.motion.sample(row['time_s'] - self.start).phase
        self.closure.observe(self.digits, phase, row['time_s'] - self.start)
        if phase == 'done':
            if collisions:
                raise RuntimeError('Unsafe bag-human retirement: native robot contact')
            if self.intervention_evidence_unknown:
                if self.grasp.active:
                    return dict(start=False,reason='waiting_for_constraint_release',time_s=row['time_s'])
                self.human.retire_after_exit();self.retired=True
                return dict(start=False,reason='intervention_contact_evidence_unknown',time_s=row['time_s'])
            self.completion_evidence = self._validate_completion(row)
            disappearance = self.human.retire_after_exit()
            self.completed = self.retired = True
            self.events.append(dict(kind='completed', disappearance=disappearance, **self.completion_evidence))
        return dict(start=False, reason=phase, time_s=row['time_s'])

    def close(self):
        self.grasp.close()
        if self.human.appearance_state == 'active':
            self.human.retire_after_exit()
