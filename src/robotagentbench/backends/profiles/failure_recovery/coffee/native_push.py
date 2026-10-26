"""CPU native open-palm actor session; never commands coffee/robot poses.

Owner creates it before reset, binds the simulation view, arms only after
physical initial acceptance, observes validated reader frames, then commands
the actor before the next substep. Framework registration is not done here.
"""
import numpy as np
from .moving_frame import pose_matrix
from .trigger import PourApproachTrigger
from .push_motion import route_candidates, select_admitted_direction, palm_fit, PalmPushMotion, PUSH_DURATION_S
from .geometry import collision_vertices, collision_shapes, main_case_shapes, static_collision_vertices, box, swept, intersect


class NativePalmPush:
    def __init__(self, stage, initial_plan):
        from pxr import Usd, UsdGeom, UsdPhysics
        from evaluation.tasks.coffee_machine.specs import MACHINE_BODY, COUNTER
        from .open_palm import OpenPalmHand
        from short_task_robustness.tasks.stock_wooden_tray_interference.human_asset import NativeHuman
        self.stage = stage
        self.seed = initial_plan['seed']
        self.asset = OpenPalmHand()
        self.trigger = PourApproachTrigger()
        self.start = None
        self.pending = None
        self.completed = False
        self.retired = False
        self.events = []
        self.motion = None
        self.reader = None
        cache = UsdGeom.XformCache()
        body_world = np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(MACHINE_BODY)), dtype=float).T
        world_vertices = collision_vertices(stage, MACHINE_BODY)
        self.local_machine_vertices = (world_vertices - body_world[:3, 3]) @ body_world[:3, :3]
        self.local_machine_shapes = [(shape - body_world[:3, 3]) @ body_world[:3, :3]
                                     for shape in collision_shapes(stage, MACHINE_BODY)]
        self.local_contact_shapes = [(shape - body_world[:3, 3]) @ body_world[:3, :3]
                                     for shape in main_case_shapes(stage, MACHINE_BODY, include_back=True)]
        self.counter = box(static_collision_vertices(stage, COUNTER))
        self.fixture_boxes = []
        bounds = UsdGeom.BBoxCache(0, ['default', 'render', 'proxy'])
        background = stage.GetPrimAtPath('/World/Background')
        if not background:
            raise ValueError('Existing background fixtures required')
        for prim in Usd.PrimRange(background):
            if str(prim.GetPath()) == COUNTER or str(prim.GetPath()).startswith(COUNTER + '/'):
                continue
            if prim.HasAPI(UsdPhysics.CollisionAPI) and prim.GetAttribute('physics:collisionEnabled').Get() is not False:
                r = bounds.ComputeWorldBound(prim).ComputeAlignedRange()
                if not r.IsEmpty():
                    self.fixture_boxes.append(np.array([r.GetMin(), r.GetMax()], dtype=float))
        routes = initial_plan['push_routes']
        selected = next(row for row in routes['routes'] if row['proposal']['side'] == routes['selected_side'])
        if not selected['static_admitted']:
            raise ValueError('Selected initial palm route rejected; no opposite-side fallback')
        fit = selected['palm_fit']
        matrix = np.eye(4)
        matrix[:3, :3] = fit['orientation']
        matrix[:3, 3] = fit['wrist_position_m']
        self.human = NativeHuman(stage, '/World/CoffeeMachinePushHuman', matrix, self.asset)
        self.absence = self.human.wait_for_trigger()
        self.orientation = matrix[:3, :3].copy()

    def bind(self, simulation_view, reader):
        self.human.bind(simulation_view)
        self.reader = reader

    def arm(self):
        if self.reader is None:
            raise RuntimeError('Bind native actor and dynamic reader before arming')
        self.trigger.arm()

    def _admit(self, robot_native_poses):
        from evaluation.tasks.coffee_machine.specs import BAG
        row = self.reader.latest
        bag_pose = row['poses'][BAG]
        bag_matrix = pose_matrix(bag_pose['position_m'], bag_pose['quaternion_wxyz'])
        mouth = bag_matrix[:3, :3] @ self.reader.bag_mouth + bag_matrix[:3, 3]
        candidates = route_candidates(self.seed, self.reader.live_machine.current(row['time_s']),
                                      mouth, self.reader.contract.hopper)
        return select_admitted_direction(candidates,
            lambda candidate: self._admit_direction(candidate, robot_native_poses))

    def _admit_direction(self, p, robot_native_poses):
        from evaluation.tasks.coffee_machine.specs import MACHINE_BODY, CUP, BAG
        from short_task_robustness.tasks.coffee_bag_interference.preview_geometry import robot_visual_boxes
        row = self.reader.latest
        body = row['poses'][MACHINE_BODY]
        tf = pose_matrix(body['position_m'], body['quaternion_wxyz'])
        vertices = self.local_machine_vertices @ tf[:3, :3].T + tf[:3, 3]
        direction = np.asarray(p['direction_world_xy_unit'])
        # Transform the actual source cup outline and handle spheres, not a root
        # centre. All measured source shapes remain unchanged.
        from evaluation.tasks.coffee_machine.isaac_evidence import body_colliders
        def current_box(path):
            values = []
            for points, sphere in body_colliders(self.stage, path).values():
                if points is not None:
                    values.extend(points)
                if sphere is not None:
                    from itertools import product
                    c, r = sphere
                    values.extend(np.array(list(product((-r, r), repeat=3))) + c)
            pose = row['poses'][path]
            native = pose_matrix(pose['position_m'], pose['quaternion_wxyz'])
            return box(np.asarray(values) @ native[:3, :3].T + native[:3, 3])
        cup_bounds, bag_bounds = current_box(CUP), current_box(BAG)
        shapes = [shape @ tf[:3, :3].T + tf[:3, 3] for shape in self.local_machine_shapes]
        contact_shapes = [shape @ tf[:3, :3].T + tf[:3, 3] for shape in self.local_contact_shapes]
        delta = direction * p['distance_m']
        route = swept(box(vertices), delta)
        robot = [np.asarray(value) for value in robot_visual_boxes(self.stage, (), robot_native_poses)]
        if not (np.all(route[0, :2] >= self.counter[0, :2]) and np.all(route[1, :2] <= self.counter[1, :2])):
            return False, dict(reason='machine_sweep_outside_counter')
        # Cup rests inside the machine; its physical movement is not prescribed.
        # Other fixtures, robot and the held bag must not block the machine.
        for index, obstacle in enumerate([*self.fixture_boxes, *robot, bag_bounds]):
            if intersect(route, obstacle):
                group = ('fixture' if index < len(self.fixture_boxes) else
                         'robot' if index < len(self.fixture_boxes) + len(robot) else 'bag')
                return False, dict(reason='native_machine_sweep_blocked', blocked_group=group,
                                   blocked_box_m=obstacle.tolist(), machine_swept_box_m=route.tolist())
        fit = palm_fit(self.asset, vertices, direction, avoid_boxes=[cup_bounds, self.counter], machine_shapes_world=shapes,
                       contact_shapes_world=contact_shapes, actor_travel_world=delta)
        fit['contact_region'] = 'lower_main_case_wall'
        fit['push_direction'] = p['side']
        points, _ = self.asset.sample(0.)
        orientation = np.asarray(fit['orientation'])
        contact = np.asarray(fit['wrist_position_m'])
        points = points @ orientation.T + contact
        shape_boxes = [box(points[indices]) for indices in self.asset.collision_groups().values()]
        # Visible retreat reverses the same palm path at the same mean speed,
        # over one planned push distance. No new retreat speed/clearance tuning.
        for hand_box in shape_boxes:
            envelope = swept(hand_box, delta)
            for obstacle in [self.counter, cup_bounds, bag_bounds, *self.fixture_boxes, *robot]:
                if intersect(envelope, obstacle):
                    return False, dict(reason='native_palm_route_blocked')
        return True, dict(proposal=p, palm_fit=fit, route_locked=True,
                          contacts_verified=False, machine_pose_commands=False)

    def observe(self, *, robot_native_poses):
        if self.reader is None:
            raise RuntimeError('Native reader required')
        facts = self.reader.trigger_facts()
        if self.pending is not None:
            if facts.time_s <= self.pending['queued_time_s']:
                raise RuntimeError('Advance the absent actor one real physical substep before appearance')
            self.start = facts.time_s
            event = self.pending['event']
            event['appearance_time_s'] = facts.time_s
            event['appearance'] = self.human.activate_on_trigger()
            self.events.append(event)
            self.pending = None
            # Appearance confirmation is not a second trigger event.
            return dict(event, start=False, reason='human_appeared')
        if self.start is not None:
            return dict(start=False, reason='already_triggered', time_s=facts.time_s)
        eligible = facts.holding_bag and facts.mouth_above_opening and facts.bag_tilt_deg >= 45.
        admission = dict(reason='not_yet_eligible')
        clear = False
        if eligible:
            # Four directions in THIS physical frame. No route or invalid
            # evidence is an explicit error, never a delayed larger-angle push.
            clear, admission = self._admit(robot_native_poses)
        event = self.trigger.observe(facts, route_clear=clear)
        event['admission'] = admission
        if event['start']:
            self.motion = PalmPushMotion(admission['palm_fit']['wrist_position_m'], admission['proposal'])
            self.orientation = np.asarray(admission['palm_fit']['orientation'])
            # Reposition while the actor is still visually/collision absent.
            self._position_absent(self.motion.contact)
            # Native kinematic targets take effect at the next physical step.
            # Keep shapes absent during that update instead of activating at
            # their old authored pose and causing an invisible impact.
            self.pending = dict(queued_time_s=facts.time_s, event=event)
            event['appearance_pending_physical_step'] = True
        return event

    def _position_absent(self, position):
        """Kinematic actor ONLY; no source body/robot setter or attachment."""
        from scipy.spatial.transform import Rotation
        points, bones = self.asset.sample(0.)
        matrix = np.eye(4)
        matrix[:3, :3] = self.orientation
        matrix[:3, 3] = position
        if self.human.appearance_state != 'waiting':
            raise RuntimeError('Initial appearance relocation only while absent')
        self.human.visual.GetPointsAttr().Set((points @ self.orientation.T + position).tolist())
        for bone, view in self.human.views.items():
            transform = matrix @ bones[bone]
            pose = np.concatenate((transform[:3, 3], Rotation.from_matrix(transform[:3, :3]).as_quat()))
            view.set_kinematic_targets(pose.astype(np.float32)[None, :], np.array([0], dtype=np.uint32))
        self.human.last_matrix = matrix

    def command_before_step(self, time_s):
        if self.start is None or self.retired:
            return dict(phase='absent', machine_pose_commands=False)
        elapsed = float(time_s) - self.start
        if elapsed < 0:
            raise ValueError('Actor clock cannot go backwards')
        if elapsed < PUSH_DURATION_S:
            sample = self.motion.sample(elapsed)
            position = sample['wrist_position_m']
        elif elapsed < 2 * PUSH_DURATION_S:
            fraction = (elapsed - PUSH_DURATION_S) / PUSH_DURATION_S
            position = self.motion.contact + (1 - fraction) * self.motion.distance * self.motion.direction
            sample = dict(phase='retreat', closure=0., machine_pose_commands=False)
        else:
            self.completed = True
            self.retired = True
            self.events.append(dict(phase='retired', time_s=time_s,
                                    disappearance=self.human.retire_after_exit()))
            return dict(phase='retired', machine_pose_commands=False)
        matrix = np.eye(4)
        matrix[:3, :3] = self.orientation
        matrix[:3, 3] = position
        self.human.command(matrix, 0.)
        return dict(sample, wrist_position_m=list(position))

    def close(self):
        if self.human.appearance_state == 'active':
            self.human.retire_after_exit()
        self.retired = True
