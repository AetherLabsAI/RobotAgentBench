"""Coffee target adaptation of the task-local one-shot 10 cm trigger."""
from .trigger_rules import ProximityTrigger, TRIGGER_DISTANCE_M, distance_to_box
from copy import deepcopy


class NativeCoffeeBagTriggerFacts:
    """Read original atomic 1-2's fresh facts and measured native finger pads.

    No FK estimates, scene edits, physics stepping, sensor setup or scoring.
    A successfully completed strict atomic reader frame is required; its
    last_time is committed only after current contacts were validated.
    """
    def __init__(self, stage, simulation_view, atomic_reader):
        import numpy as np
        from evaluation.tasks.coffee_machine.specs import BAG
        from evaluation.tasks.coffee_machine.isaac_evidence import body_colliders, PADS
        from evaluation.tasks.coffee_machine.openended_composite_v1.native_source import _BoundBodies
        if atomic_reader.spec.api != '1-2':
            raise ValueError('Coffee-bag trigger requires the atomic 1-2 reader')
        self.reader, self.target = atomic_reader, BAG
        points = []
        for vertices, sphere in body_colliders(stage, BAG).values():
            if vertices is not None: points.extend(vertices)
            if sphere is not None:
                center,radius = sphere
                points.extend([center-np.array([radius]*3),center+np.array([radius]*3)])
        vertices = np.asarray(points,dtype=float)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
            raise ValueError('Measured coffee-bag collision geometry required')
        self.bounds = [vertices.min(0).tolist(),vertices.max(0).tolist()]
        self.pads = {hand:[_BoundBodies(simulation_view,[p]) for p in paths]
                     for hand,paths in PADS.items()}

    def read(self):
        import numpy as np
        from evaluation.tasks.shelf_pick.geometry import rotation
        from evaluation.tasks.coffee_machine.facts import COUNTER
        reader = self.reader; row = reader.latest
        if (row is None or row['time_s'] != float(reader.world.current_time)
                or row['time_s'] != reader.last_time or row['reader_step'] != reader.step):
            raise RuntimeError('Fresh fully validated atomic contact frame required')
        pose = row['poses'][self.target]
        inverse = rotation(pose['quaternion_wxyz']).inv()
        center = np.asarray(pose['position_m'],dtype=float)
        if center.shape != (3,) or not np.isfinite(center).all():
            raise RuntimeError('Invalid current coffee-bag position')
        centers = {};world_centers={}
        for hand,pads in self.pads.items():
            values = np.asarray([p.get_world_pose()[0] for p in pads],dtype=float)
            if values.shape != (2,3) or not np.isfinite(values).all():
                raise RuntimeError('Invalid measured native jaw pad positions')
            world_centers[hand]=values.mean(0).tolist()
            centers[hand] = inverse.apply(values.mean(0)-center).tolist()
        c = row['contacts'][self.target]
        if type(c['robot']) is not bool or type(c['grasped']) is not bool:
            raise RuntimeError('Explicit native coffee-bag contact and holding required')
        if not isinstance(c['supports'],(list,tuple,set,frozenset)):
            raise RuntimeError('Current native support evidence required')
        return dict(simulation_time_s=row['time_s'],target_path=self.target,
            target_bounds_local_m=self.bounds,gripper_centers_target_local_m=centers,
            target_origin_world_m=center.tolist(),gripper_centers_world_m=world_centers,
            robot_contacting_target=c['robot'],robot_holding_target=c['grasped'],
            target_supported=COUNTER in c['supports'],
            contacts_complete=reader.contacts.complete_for_roots(self.target),
            evidence_source='fresh_completed_strict_atomic_contact_frame')


class CoffeeBagApproachTrigger:
    """Admission-gated trigger; the future native mover must consume start once."""
    def __init__(self,target_path,*,reservation=None):
        self.trigger = ProximityTrigger(target_path)
        self.armed = False
        self.reservation=reservation
        self.locked_direction=None

    def arm(self):
        if self.armed: raise RuntimeError('Initial admission arms this trigger once')
        self.armed = True

    def observe(self,facts,*,route_clear):
        if not self.armed: raise RuntimeError('No interference before initial admission')
        direction=None
        if self.reservation is not None:
            if self.locked_direction is not None:
                direction=deepcopy(self.locked_direction)
            else:
                from .joint_initialization import direction_check
                centers=facts['gripper_centers_target_local_m']
                hand=min(centers,key=lambda h:distance_to_box(centers[h],facts['target_bounds_local_m']))
                direction=direction_check(facts['target_origin_world_m'],
                    self.reservation['planned_bag_origin'],facts['gripper_centers_world_m'][hand])
                direction['measured_hand']=hand
            route_clear=route_clear and direction['allowed']
        event=self.trigger.observe(facts,route_clear=route_clear)
        if direction is not None:
            event['direction_check']=direction
            if event['start']:self.locked_direction=deepcopy(direction)
            if event['reason']=='reserved_route_blocked' and not direction['allowed']:
                event['reason']=('destination_approaches_measured_jaw' if direction['direction_defined']
                                 else 'undefined_measured_away_direction')
        return event
