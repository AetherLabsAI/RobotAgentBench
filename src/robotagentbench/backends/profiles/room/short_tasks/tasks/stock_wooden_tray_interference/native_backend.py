"""CPU actor for ShortTaskSession; no robot commands, stepping or score writes."""
from copy import copy, deepcopy
from pathlib import Path
import numpy as np
import warnings
from .adaptive_hand import AdaptiveHand
from .contact_closure import ContactClosure, digit_for_bone
from .human_asset import NativeHuman, TemporaryHumanGrasp, pose_matrix
from .frozen_native_route import world_motion, check_live_route
from .release_collision import ReleasedTargetFilter, discard_filtered_pairs
from .proximity import NativeTriggerFacts
from .reporting import report_json

HERE = Path(__file__).resolve().parent


def actor_contact_relevant(paths, *, human_root, target_path, support_path):
    """Only facts consumed by the human actor, not the separate stock scorer.

    Strict completeness still covers human/robot collision episodes, the
    human's target grasp, and target support for release. Robot/background
    collisions stay owned by the unchanged stock scoring contact stream.
    """
    from evaluation.tasks.shelf_pick.contact_stream import under
    if len(paths) != 2:
        raise ValueError('Exactly two resolved collider paths required')
    for first, second in (paths, paths[::-1]):
        if under(first, human_root) and (
                under(second, '/World/Robot') or under(second, target_path)):
            return True
        if under(first, target_path) and under(second, support_path):
            return True
    return False


class NativeBackend:
    physics_backend = 'cpu_physx'
    motion_backend = 'cpu'

    def __init__(self, stage, manifest, output, episode_id):
        from pxr import Usd, UsdPhysics
        from .cpu_contract import validate_cpu_scene, validate_human_rigid_prims
        validate_cpu_scene(stage)
        self.output = Path(output).resolve()
        if not self.output.is_relative_to(HERE) or self.output == HERE:
            raise ValueError('Human evidence must stay in its task directory')
        reservations = manifest.get('environment_reservations', [])
        if len(reservations) != 1:
            raise ValueError('Exactly one frozen initialization reservation required')
        self.initialization_reservation = deepcopy(reservations[0])
        self.frozen_bilateral = (deepcopy(self.initialization_reservation)
            if 'bilateral_routes' in self.initialization_reservation else None)
        self.selected_bilateral = None
        r = self.initialization_reservation
        self.target_path, self.shelf_level_path = r['target_path'], r['shelf_level_path']
        self.stage, self.manifest = stage, manifest
        self.asset = AdaptiveHand()
        self.motion, self.orientation, self.destination = world_motion(r)
        self.inverse_frame = np.linalg.inv(np.asarray(r['shelf_local_to_world_row_matrix']))
        matrix = np.eye(4)
        matrix[:3,:3], matrix[:3,3] = self.orientation, self.motion.points[0]
        if not stage.GetSessionLayer().anonymous:
            raise ValueError('Anonymous episode session required')
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            self.human = NativeHuman(stage, '/World/HumanInterference', matrix, self.asset)
            waiting = self.human.wait_for_trigger()
            validate_human_rigid_prims([stage.GetPrimAtPath(p) for p in self.human.links.values()])
            self.grasp = TemporaryHumanGrasp(stage, self.human.root+'/ExternalGrasp',
                                           self.human.links['hand.R'], self.target_path)
            self.release_filter = ReleasedTargetFilter(stage, self.human.root,
                                                       list(self.human.links.values()), self.target_path)
        self.robot_body_ids = tuple(str(p.GetPath()) for p in Usd.PrimRange(stage.GetPrimAtPath('/World/Robot'))
                                    if p.HasAPI(UsdPhysics.RigidBodyAPI))
        self.human_body_ids = tuple(self.human.links.values())
        self.closure = ContactClosure()
        self.digits, self.pairs = [], []
        self.last_closure = 0.
        self.phase = 'waiting'
        self.started_at = None
        self.closed = self.released = False
        self.contact_stream = None
        self.contact_gap_during_motion = False
        self.intervention_evidence_unknown = False
        self.evidence = dict(episode_id=episode_id, waiting=waiting, events=[])

    def bind_port(self, port):
        if tuple(port.scene.identity.candidates) != (self.target_path,):
            raise ValueError('Actor and scoring must bind the same unique instance')
        self.port = port

    def initialize(self, clock, view):
        self.clock = clock
        from openend.tasks.meal_tray.nonmotion_closeout_v1.step_snapshot import SnapshotViews
        # Unwrap only to actuate our human; object reads retain shared snapshots.
        native = view.view if isinstance(view, SnapshotViews) else view
        native.initialize_kinematic_bodies()
        self.human.bind(native)
        self.robot = view.create_articulation_view('/World/Robot')
        self.target_view = view.create_rigid_body_view(self.target_path)
        self.stock_views = {p: view.create_rigid_body_view(p) for p in self.port.scene.stock}
        self.trigger_reader = NativeTriggerFacts(self.port.source, view)

    def robot_link_poses(self):
        return {'/World/Robot/'+name: pose_matrix(row[:3], row[3:7]) for name, row in
                zip(self.robot.shared_metatype.link_names, self.robot.get_link_transforms()[0])}

    def stock_vertices_local(self, path):
        row = self.stock_views[path].get_transforms()[0]
        tf = pose_matrix(row[:3], row[3:7]).T @ self.inverse_frame
        vertices = np.asarray(self.port.scene.workspace.collision_vertices[path])
        return vertices @ tf[:3,:3] + tf[3,:3]

    def reservation_obstacle_bounds(self):
        return [np.array([v.min(0), v.max(0)]) for p in self.stock_views if p != self.target_path
                for v in [self.stock_vertices_local(p)]]

    def trigger_facts(self):
        return self.trigger_reader.read()

    def route_admission(self, reservation, facts):
        from .near_entry import check_away_from_gripper
        from .reservation import slot_is_clear
        from evaluation.tasks.shelf_pick.geometry import rotation
        if not slot_is_clear(reservation,self.reservation_obstacle_bounds()):
            raise ValueError('Reserved destination now occupied')
        motion,orientation,_ = world_motion(reservation)
        if (not np.allclose(orientation,self.orientation,atol=1e-9,rtol=0)
                or not np.allclose(np.asarray(motion.points)[:3],
                                   np.asarray(self.motion.points)[:3],atol=1e-9,rtol=0)):
            raise ValueError('Reserved option changed the common rear entry/grip')
        inverse = rotation(self.port.source.latest['poses'][self.target_path]['quaternion_wxyz']).inv()
        away = check_away_from_gripper(facts,
            inverse.apply(np.asarray(motion.points[4])-motion.points[1]))
        live = check_live_route(self.stage,self.manifest,reservation,self.asset,
            source_support_verified=True,robot_link_poses=self.robot_link_poses())
        return dict(away_from_gripper=away,route_admission=live)

    def choose_reservation(self, facts):
        """Read-only live selection; no robot pause, endpoint search or writes."""
        from .bilateral_routes import choose
        from .proximity import distance_to_box, TRIGGER_DISTANCE_M
        if self.frozen_bilateral is None or self.started_at is not None:
            raise RuntimeError('Select only a new bilateral initial, before start')
        if (not facts['contacts_complete'] or not facts['target_supported']
                or facts['robot_contacting_target'] or facts['robot_holding_target']
                or min(distance_to_box(p,facts['target_bounds_local_m']) for p in
                       facts['gripper_centers_target_local_m'].values())>TRIGGER_DISTANCE_M):
            return None
        route,evidence = choose(self.frozen_bilateral,
            evaluate=lambda r:self.route_admission(r,facts))
        self.selected_bilateral = deepcopy(route)
        self.evidence['bilateral_selection'] = deepcopy(evidence)
        return route

    def start(self, duration_s, reservation):
        if self.closed or self.started_at is not None or duration_s != self.motion.duration_s:
            raise RuntimeError('One approved three-second intervention only')
        if self.frozen_bilateral is not None:
            if self.selected_bilateral is None or reservation != self.selected_bilateral:
                raise ValueError('Activation must use the natively admitted frozen option')
            # Bind exactly once to one of the already reserved routes. Legacy
            # initials never enter this branch; no mid-motion replanning.
            from .bilateral_routes import reservations
            if reservation not in reservations(self.frozen_bilateral):
                raise ValueError('Cannot invent a runtime endpoint')
            self.initialization_reservation = deepcopy(reservation)
            self.motion,self.orientation,self.destination = world_motion(reservation)
        elif reservation != self.initialization_reservation:
            raise ValueError('Cannot change frozen reservation at activation')
        facts = self.trigger_facts()
        if not facts['contacts_complete'] or not facts['target_supported']:
            raise RuntimeError('Current native shelf support required')
        from .near_entry import check_away_from_gripper
        from evaluation.tasks.shelf_pick.geometry import rotation
        inverse = rotation(self.port.source.latest['poses'][self.target_path]['quaternion_wxyz']).inv()
        self.evidence['away_from_gripper'] = check_away_from_gripper(facts,
            inverse.apply(np.asarray(self.motion.points[4])-self.motion.points[1]))
        self.evidence['route_admission'] = check_live_route(self.stage, self.manifest,
            reservation, self.asset, source_support_verified=True, robot_link_poses=self.robot_link_poses())
        self.evidence['appearance'] = self.human.activate_on_trigger()
        self.started_at = float(self.clock.current_time)
        self.phase = 'approach'

    def before_physics(self, dt):
        if self.closed or self.started_at is None or abs(dt-self.clock.get_physics_dt()) > 1e-12:
            raise RuntimeError('Human requires current owner timestep')
        elapsed = float(self.clock.current_time)-self.started_at
        sample = self.motion.sample(elapsed)
        if sample.phase == 'carry' and not self.grasp.active and not self.grasp.ever_acquired:
            contact = any(self.target_path in pair and any(p in self.human_body_ids for p in pair) for pair in self.pairs)
            closed = self.closure.ready(self.digits)
            if self.contact_gap_during_motion and not (contact and closed):
                if not self.intervention_evidence_unknown:
                    warnings.warn('Grasp evidence missing during interference; '
                                  'continuing without fabricating acquisition or pickup credit.', RuntimeWarning)
                self.intervention_evidence_unknown = True
            else:
                self.grasp.acquire(self.human.views['hand.R'].get_transforms()[0], self.target_view.get_transforms()[0],
                    hand_target_contact=contact, hand_closed=closed)
        if sample.phase == 'release' and self.grasp.active:
            self.grasp.release()
            self.released = True
        if self.release_filter.update(phase=sample.phase, released=self.released, constraint_active=self.grasp.active,
                previous_closure=self.last_closure, supported=bool(self.port.source.latest['contacts']['target_supported'])):
            self.release_filter.invalidate_reader_cache(self.port.source)
            if self.contact_stream is not None:
                discard_filtered_pairs(self.contact_stream.active, self.human_body_ids, self.target_path)
                self.contact_stream.unknown_pairs.intersection_update(self.contact_stream.active)
        matrix = np.eye(4)
        matrix[:3,:3], matrix[:3,3] = self.orientation, sample.wrist_position
        self.last_closure = self.closure.command(sample.closure, sample.phase)
        self.human.command(matrix, self.last_closure)
        self.phase = sample.phase

    def read_native_facts(self):
        """This substep's contacts, independently before the scoring port reads."""
        from arena_api.contact_identity import ContactIdentityCache
        from evaluation.tasks.shelf_pick.contact_stream import under
        from pxr import UsdPhysics
        source = self.port.source
        if self.contact_stream is None:
            self.contact_stream = copy(source.contacts)
            support = source.contacts.object_supports.get(self.target_path, source.contacts.support)
            classifier = lambda paths: actor_contact_relevant(paths, human_root=self.human.root,
                target_path=self.target_path, support_path=support)
            self.contact_stream.active = {k: deepcopy(v) for k,v in source.contacts.active.items()
                                         if classifier(v[0])}
            self.contact_stream.unknown_pairs = set(source.contacts.unknown_pairs).intersection(
                self.contact_stream.active)
            self.contact_stream.identity = ContactIdentityCache(source.contacts.identity.resolve, classifier)
            self.contact_stream.allow_missing_awake = True
        stream = self.contact_stream
        stream.ingest(source.interface.get_contact_report())
        complete = stream.evidence_status()['complete']
        if not complete and self.started_at is not None:
            self.contact_gap_during_motion = True
        pose = self.target_view.get_transforms()[0]
        local = np.asarray(self.port.scene.workspace.collision_vertices[self.target_path])
        tf = pose_matrix(pose[:3], pose[3:7])
        contacts = stream.summarize(self.target_path, tf[:3,:3] @ local.mean(0)+tf[:3,3])
        def body(path):
            prim = self.stage.GetPrimAtPath(path)
            while prim and not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                prim = prim.GetParent()
            return str(prim.GetPath()) if prim else path
        self.pairs = [tuple(body(p) for p in row[0]) for key,row in stream.active.items()
                      if key not in stream.unknown_pairs
                      if any(under(p,self.human.root) for p in row[0])]
        self.digits = sorted({digit_for_bone(bone) for bone,path in self.human.links.items()
            if digit_for_bone(bone) and any(path in pair and self.target_path in pair for pair in self.pairs)})
        if self.started_at is not None:
            self.closure.observe(self.digits, self.phase, float(self.clock.current_time)-self.started_at)
        vertices = self.stock_vertices_local(self.target_path)
        center = (vertices.min(0)+vertices.max(0))/2
        reserved = np.asarray(self.initialization_reservation['reserved_bounds'])
        destination_valid = bool(np.all(center[:2] >= reserved[0,:2]) and np.all(center[:2] <= reserved[1,:2]))
        actor_clear = complete and not any(any(p in self.robot_body_ids for p in pair) for pair in self.pairs)
        if self.phase == 'done' and self.human.appearance_state == 'active' and complete:
            if self.grasp.active or (not self.released and not self.intervention_evidence_unknown) or not actor_clear:
                raise RuntimeError('Cannot disappear before release and clear robot separation')
            self.evidence['retired'] = self.human.retire_after_exit()
        facts = dict(simulation_time_s=float(self.clock.current_time), target_path=self.target_path,
            shelf_level_path=self.shelf_level_path, contact_pairs=self.pairs,
            contacts_complete=complete, phase=self.phase,
            intervention_evidence_unknown=self.intervention_evidence_unknown,
            human_released=self.released, target_supported=bool(contacts['target_supported']),
            destination_valid=destination_valid, actor_clear=actor_clear)
        if not self.evidence['events'] or self.evidence['events'][-1]['phase'] != self.phase:
            self.evidence['events'].append(deepcopy(facts))
        return facts

    def close(self):
        if self.closed:
            return
        self.grasp.close()
        self.closed = True
        self.evidence.update(closed=True, phase=self.phase, release_filter=self.release_filter.evidence())
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output/'human_actor.json').write_text(report_json(self.evidence)+'\n')
