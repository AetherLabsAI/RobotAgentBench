"""Native OpenEnd brush contact/coverage adapter. No scripted manipulation."""
from collections import deque
from pathlib import Path
from types import SimpleNamespace
import json
import math
import numpy as np
from .registration import WipeRegistration, TOOL, STAIN, SUPPORT
from .process_rules import GraspGate, ProcessScorer, StepFacts, WipeFrame
from . import stain as st


class WipePort(WipeRegistration):
    def __init__(self, stage, manifest, output, episode_id, instance_id, parameters):
        self.resolve_object(parameters, manifest)
        from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, PhysicsSchemaTools
        from omni.physx import get_physx_simulation_interface
        from omni.physx.bindings._physx import ContactEventType
        from evaluation.tasks.shelf_pick.contact_stream import ContactStream, under
        from evaluation.tasks.shelf_pick.geometry import HANDS
        from evaluation.tasks.shelf_pick.physical_detection import DetectionConfig, GraspDetector
        from evaluation.tasks.tabletop_pick.placement import tabletop_rectangle
        self.stage, self.episode_id, self.hands = stage, episode_id, HANDS
        self.instance_id = instance_id
        self.geometry = SimpleNamespace(target=TOOL)
        self.binding = SimpleNamespace(objects={'tool':TOOL,'stain':STAIN,'support':SUPPORT})
        self.interface = get_physx_simulation_interface()
        self.config = DetectionConfig()
        self.detector = GraspDetector((TOOL,), self.config)
        self.surface = tabletop_rectangle(stage, SUPPORT)
        self.top = float(self.surface['z_m'])
        table = stage.GetPrimAtPath(SUPPORT)
        body = stage.GetPrimAtPath(TOOL)
        if not table or not table.HasAPI(UsdPhysics.CollisionAPI) or not body.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError('Physical tabletop and brush required')
        if body.GetAttribute('physics:kinematicEnabled').Get() is True:
            raise ValueError('Kinematic brush is not allowed')
        from .floors import physical_floor_paths
        floors = physical_floor_paths(stage)
        self.fixed = {p: UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(p))
                      for p in (SUPPORT, STAIN, *floors)}
        matrix = np.asarray(self.fixed[STAIN])
        rotation = matrix[:3,:3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7) or not np.allclose(rotation[2], [0,0,1], atol=1e-7):
            raise ValueError('Stain must be rigid, upright and unscaled')
        self.layout = st.StainLayout.from_file(manifest['wipe_table']['layout_file'], (0,0), self.top)
        # Original cell coordinates transformed with this room's actual stain.
        self.layout.centers = (np.c_[self.layout.centers,np.zeros(self.layout.count),np.ones(self.layout.count)] @ matrix)[:,:2]
        if abs(matrix[3,2]-self.top) > 1e-4:
            raise ValueError('Stain not on the physical tabletop')
        grid = stage.GetPrimAtPath(STAIN+'/Grid')
        self.opacity = grid.GetAttribute('primvars:displayOpacity')
        self.base_opacity = np.asarray(self.opacity.Get(), dtype=np.float32)
        if len(self.base_opacity) != self.layout.count*self.layout.vertices_per_cell:
            raise ValueError('Stain opacity/layout mismatch')
        self.stain = st.StainState(self.layout)
        class BrushContacts(ContactStream):
            def classify(self, paths):
                return any(under(p, TOOL) for p in paths)
        if not stage.GetSessionLayer().anonymous:
            raise ValueError('Anonymous session instrumentation required')
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            PhysxSchema.PhysxContactReportAPI.Apply(body).CreateThresholdAttr(0.)
        def sleeping(sid, actor):
            p = stage.GetPrimAtPath(PhysicsSchemaTools.intToSdfPath(actor))
            if not p:
                raise RuntimeError('Unresolved native contact actor')
            return bool(not p.HasAPI(UsdPhysics.RigidBodyAPI) or self.interface.is_sleeping(sid, actor))
        self.contacts = BrushContacts(PhysicsSchemaTools.intToSdfPath, sleeping,
            (ContactEventType.CONTACT_FOUND, ContactEventType.CONTACT_PERSIST, ContactEventType.CONTACT_LOST),
            (), self.top, self.config.support_face_tolerance_m, floor_paths=floors, targets=(TOOL,), support=SUPPORT)
        self.output = Path(output)
        self.output.mkdir(exist_ok=False)
        (self.output/'floor_binding.json').write_text(json.dumps(dict(
            floor_paths=floors, scene_mutations=0, physical_parameters_changed=False)))
        self.trace = (self.output/'facts.jsonl').open('x')
        self.history = deque(maxlen=math.ceil(self.config.initial_history_s*240)+2)
        self.pending, self.latest, self.composite, self.score = [], None, None, None
        self.index, self.sample_index, self.closed = -1, 0, False
        self.grasp_gate = GraspGate()

    @property
    def ended(self):
        return self.composite is not None and self.composite.ended

    def initialize(self, clock, view):
        self.clock, self.last_time = clock, clock.current_time
        self.views = {}
        for path in (TOOL, *self.hands.values()):
            v = view.create_rigid_body_view(path)
            if v.count != 1 or list(v.prim_paths) != [path]:
                raise ValueError('Native body identity mismatch '+path)
            self.views[path] = v

    def after_physics(self, helper):
        from pxr import UsdGeom
        from evaluation.tasks.shelf_pick.geometry import rotation
        from openend.tasks.all_objects_integration.selective_read import contact_report
        if any(helper.get(k) for k in ('events','docked_object_count','tray_dock_enabled')):
            raise RuntimeError('Helper-modified evidence')
        now, dt = self.clock.current_time, self.clock.get_physics_dt()
        if self.clock.hz not in self.supported_physics_hz or abs(now-self.last_time-dt) > 1e-6:
            raise RuntimeError('Missing native physical substep')
        for path, transform in self.fixed.items():
            if UsdGeom.XformCache().GetLocalToWorldTransform(self.stage.GetPrimAtPath(path)) != transform:
                raise RuntimeError('Scoring support or stain moved: '+path)
        poses = {}
        for path, v in self.views.items():
            p = np.asarray(v.get_transforms()).copy()[0]
            if p.shape != (7,) or not np.isfinite(p).all():
                raise ValueError('Invalid native pose')
            poses[path] = dict(position_m=p[:3].tolist(),quaternion_wxyz=p[[6,3,4,5]].tolist())
        contacts = self.contacts.read(contact_report(self, helper))
        held, _ = self.detector.observe(now, poses, contacts['bilateral_contacts'])
        activated = self.grasp_gate.observe(bool(held[TOOL]))
        tool = poses[TOOL]
        rot = rotation(tool['quaternion_wxyz']).as_matrix()
        covered, height = st.covered_cells(self.layout, st.face_corners(tool['position_m'],rot), bool(contacts['target_support_contact']))
        row = dict(time=now,dt=dt,time_s=now,physics_dt_s=dt,xyz=tool['position_m'],
                   held=bool(held[TOOL]),robot=TOOL in contacts['robot_contacts'],
                   height=float(height),contact=bool(contacts['target_support_contact']),
                   supported=bool(contacts['target_supported']),floor=bool(contacts['target_on_floor']),
                   xy=tuple(tool['position_m'][:2]),yaw=float(np.arctan2(rot[1,0],rot[0,0])),
                   activated=activated,covered=covered)
        self.latest = row
        self.history.append(row)
        self.last_time = now
        if self.composite is not None:
            self.pending.append(row)

    def _frame(self, row):
        frame = WipeFrame(self.episode_id,self.sample_index,row['time'],row['held'],row['robot'],
            row['height'],row['contact'],row['supported'],row['floor'],row['xy'],row['yaw'])
        self.sample_index += 1
        return frame

    def _summary(self):
        return dict(self.composite.summary(),evidence_status='native_physx',
                    instance_id=getattr(self,'instance_id',None),timeout_owner='room',
                    native_full_trajectory_verified=False)

    def accept_initial(self):
        from openend.tasks.meal_tray.nonmotion_closeout_v1.initial_windows import initial_window
        if self.composite is not None or self.latest is None:
            raise RuntimeError('Initial acceptance exactly once')
        rows, complete = initial_window(self.history,self.config.initial_history_s,self.clock.get_physics_dt())
        checks = dict(history_complete=complete,tool_supported=bool(rows) and all(r['supported'] for r in rows),
            no_robot_contact=bool(rows) and not any(r['robot'] or r['held'] for r in rows),
            no_floor=bool(rows) and not any(r['floor'] for r in rows),
            face_on_table=bool(rows) and all(abs(r['height']) <= self.config.support_face_tolerance_m for r in rows),
            stable_tool=bool(rows) and np.ptp([r['xyz'] for r in rows],axis=0).max() <= self.config.initial_position_span_m,
            stain_untouched=bool(rows) and not any(r['covered'].any() for r in rows))
        receipt = dict(accepted=bool(all(checks.values())),checks={k:bool(v) for k,v in checks.items()},
                       native_full_trajectory_verified=False,scene_pose_writes=0)
        (self.output/'initial_acceptance.json').write_text(json.dumps(receipt,indent=2)+'\n')
        if not receipt['accepted']:
            raise RuntimeError('Wiping initial rejected: '+str(checks))
        # Initial acceptance still checks raw physical coverage/contact, but
        # pre-episode observations cannot activate the actual episode.
        self.grasp_gate = GraspGate()
        row = self.latest
        self.composite = ProcessScorer(self.episode_id,row['dt'],row['xy'],row['yaw'])
        self.composite.observe_step(StepFacts((self._frame(row),),1.,1.,False,0))
        self.score = self._summary()
        self.index = 0
        return receipt

    def consume_control(self, active):
        from pxr import Usd
        if active is not True or not self.pending:
            raise RuntimeError('Missing selected physical evidence')
        rows, self.pending = self.pending, []
        if any(abs(r['dt']-rows[0]['dt']) > 1e-12 for r in rows):
            raise ValueError('Room changed rate inside one held control step')
        eligible = [r for r in rows if r['activated']]
        # A contact wake only supplies brush/gripper evidence for detection.
        # Never count passes, rearm cells or fade from pre-grasp substeps.
        newly = (self.stain.update(np.logical_or.reduce([r['covered'] for r in eligible]))
                 if eligible else np.empty(0, dtype=int))
        if len(newly):
            with Usd.EditContext(self.stage,self.stage.GetSessionLayer()):
                self.opacity.Set(self.stain.vertex_opacity(self.base_opacity).tolist())
        self.composite.configure_clock(rows[0]['dt'])
        face_contact = any(r['contact'] and r['height'] <= st.FACE_HEIGHT_TOLERANCE_M for r in eligible)
        facts = StepFacts(tuple(self._frame(r) for r in rows),self.stain.coverage(),self.stain.initial_coverage,
                          face_contact,int(len(newly)))
        self.composite.observe_step(facts)
        self.index += 1
        self.score = self._summary()
        self.trace.write(json.dumps(dict(time_s=rows[-1]['time'],coverage=self.stain.coverage(),
            passes=self.stain.passes.tolist(),score=self.score,newly_passed_cells=newly.tolist()))+'\n')
        self.trace.flush()
        return self.score

    def pause(self):
        self.composite.pause()

    def finalize(self, reason):
        self.pending = []
        self.composite.finalize(self.composite.reason or ('client_closed' if reason=='client_closed' else 'time_limit'))
        self.score = dict(self._summary(),room_termination_reason=reason)
        return self.score

    def close(self):
        if not self.closed:
            self.trace.close()
            self.closed = True
