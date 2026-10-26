"""Selected-task adapter for the existing unified room, not an HTTP runtime."""
import json
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace
from evaluation.tasks.shelf_pick.openended_composite_v1.catalog import normalize_parameters
from .shelf_composite import SnackTrayComposite,Frame
from .shelf_instances import peer_bodies, scoped_pick


class SnackTrayPort:
    asset='snack_shelf'
    physics_hz=60
    # Workspace selection owns the actual clock. Native source and composite
    # consume real dt at both approved rates; carrying past another workspace
    # must not suspend this instance's process/final-state observations.
    supported_physics_hz=(60,240)
    initial_workspace='shelf'
    required_workspaces=('shelf','table_wooden_tray')
    requested_physics_hz=60
    validate_parameters=staticmethod(normalize_parameters)

    @staticmethod
    def resolve_instance(parameters,manifest,instance_id):
        from openend.tasks.meal_tray.product_claims import exact_object
        return exact_object(normalize_parameters(parameters),manifest,instance_id)

    @staticmethod
    def resolve_object(parameters,manifest):
        product=normalize_parameters(parameters)['product']
        matches=[r for r in manifest['objects'] if r['id']==product and r.get('task_role')=='requested_full_product']
        if len(matches)!=1:raise ValueError('Exactly one explicitly requested shelf target required')
        return matches[0]

    @property
    def workspace_id(self):
        return 'shelf' if self.composite.phase=='pick' else 'table_wooden_tray'

    def __init__(self,stage,manifest,output,episode_id,instance_id,parameters):
        from pxr import Usd, UsdPhysics, PhysxSchema
        from .shelf_binding import SceneBinding
        self.product=normalize_parameters(parameters)['product']
        self.scene=SceneBinding(stage,manifest,self.product,episode_id,instance_id)
        self.peer_targets=peer_bodies(manifest,instance_id,self.product,self.scene.body_by_wrapper)
        self.manifest=manifest
        self.output=Path(output);self.output.mkdir(exist_ok=False)
        self.binding=SimpleNamespace(objects=dict(targets=list(self.scene.identity.candidates),
            source=self.scene.identity.source_support,destination=self.scene.tray))
        # Observation sensors only, with the same existing zero report threshold.
        # No collision/drive/material/pose overrides; never edit asset layers.
        if not stage.GetSessionLayer().anonymous:raise ValueError('Anonymous session required')
        paths=list(self.scene.stock)+[str(p.GetPath()) for p in Usd.PrimRange(stage.GetPrimAtPath('/World/Robot')) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        for p in paths:
            a=stage.GetPrimAtPath(p).GetAttribute('physxContactReport:threshold')
            if a and a.HasAuthoredValueOpinion() and a.Get()!=0:raise ValueError('Conflicting sensor threshold '+p)
        with Usd.EditContext(stage,stage.GetSessionLayer()):
            for p in paths:PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath(p)).CreateThresholdAttr(0.)
        self.composite=SnackTrayComposite(self.scene.identity,room_managed=True)
        self.score=self.composite.summary();self.index=0;self.samples=0
        self.enabled=False;self.buffer=[];self.source=None
        self.trace=(self.output/'selected_scores.jsonl').open('x')

    @property
    def ended(self):return self.composite.ended

    def initialize(self,clock,view):
        from evaluation.tasks.shelf_pick.openended_composite_v1.native_source import NativeSnackTraySource
        self.clock=clock
        # Leave source.process unattached: read at 60/240, score ONLY selected
        # native substeps below. No 20 Hz downsampling of original dense facts.
        self.source=NativeSnackTraySource(self.scene,clock,view)
        self.source.contacts.allow_missing_awake=True
        from openend.tasks.holding_follow_v1.detection import GraspDetector
        self.source.detector=GraspDetector(tuple(self.scene.stock))

    def accept_initial(self):
        if self.source.latest is None:raise RuntimeError('Native observation missing')
        self.source.initial_z={p:float(self.source.latest['poses'][p]['position_m'][2]) for p in self.scene.identity.candidates}
        return dict(initial_state_gate_enabled=False,status='not_checked_user_disabled',
                    native_reader_connected=True,scoring_started=False)

    def begin_selected_control(self,enabled):
        if self.buffer:raise RuntimeError('Unconsumed native evidence')
        self.enabled=bool(enabled)

    def after_physics(self,helper_report):
        self.source.after_physics(helper_report)
        if self.enabled and not self.ended:
            if not self.source.contacts.evidence_status()['complete']:
                # A gap is not "no contact" and cannot prove a lift/release.
                # Keep a marker so a later good frame in this same control
                # batch cannot bridge a continuous-hold requirement.
                self.buffer.append(None)
                return
            dt=self.clock.get_physics_dt()
            n=self.samples+sum(f is not None for f in self.buffer)
            previous=next((f for f in reversed(self.buffer) if f is not None),self.composite.last)
            time=0. if previous is None else previous.pick.time_s+dt
            raw=self.source.frame(n)
            self.last_raw_wrong_grasps=raw.pick.wrong_drinks_grasped
            pick=scoped_pick(raw.pick,self.peer_targets)
            self.buffer.append(Frame(self.scene.identity,replace(pick,time_s=time),replace(raw.place,time_s=time),dt))

    def after_inactive(self,helper_report):
        # Keep contact lifecycle current; do not read all body poses or score.
        now=float(self.clock.current_time);dt=self.clock.get_physics_dt()
        if abs(now-self.source.last_time-dt)>1e-6:raise RuntimeError('Missing inactive substep')
        if helper_report.get('events') or helper_report.get('docked_object_count') or helper_report.get('tray_dock_enabled'):raise RuntimeError('Assisted evidence')
        self.source.last_time=now
        report=self.source.interface.get_contact_report();self.source.contacts.ingest(report)
        if self.source.latest is not None:
            from evaluation.tasks.shelf_pick.geometry import rotation
            for path,reader in self.source.placements.items():
                # Only orientation needed to maintain persistent support contacts.
                _,q=self.source.views[path].get_world_pose()
                reader.read(report,rotation(q).apply([0,0,1]))

    def finalize(self,reason):
        self.discard_control()
        if not self.ended and self.composite.last is not None:self.composite.finalize('client_closed')
        self.score=self.composite.summary()
        self.score['room_termination_reason']=reason
        self._contact_status()
        return self.score

    def _contact_status(self):
        evidence=self.source.contacts.evidence_status()
        self.score['contact_evidence']=evidence
        self.score['evidence_coverage_complete']=evidence['gap_steps']==0
        self.score['scoring_status']='current' if evidence['complete'] else 'waiting_for_contact_evidence'
        if not evidence['complete']:
            # Preserve historical points, but do not certify a current success
            # from a destination state that could no longer be observed.
            self.score['success']=False

    def selection_geometry(self):
        boxes=self.activation_geometry()
        low=[min(b[0][i] for b in boxes) for i in (0,1)]
        high=[max(b[1][i] for b in boxes) for i in (0,1)]
        return (tuple((low[i]+high[i])/2 for i in (0,1)),((low[0],low[1]),(high[0],low[1]),(high[0],high[1]),(low[0],high[1])))

    def observe(self):
        if not self.enabled or not self.buffer:raise RuntimeError('Missing selected substeps')
        for frame in self.buffer:
            if self.ended:break
            if frame is None:
                self.composite.pause()
                continue
            self.score=self.composite.observe(frame);self.samples+=1
        self.score=self.composite.summary()
        self.buffer.clear();self.enabled=False;self.index+=1
        self.score.update(evidence_source='native_physx',room_runtime_registered=True,
                          full_trajectory_verified=False,native_scoring_samples=self.samples,
                          registered_peer_target_paths=sorted(self.peer_targets),
                          raw_other_grasped_paths=sorted(getattr(self,'last_raw_wrong_grasps',())))
        self._contact_status()
        self.trace.write(json.dumps(dict(simulation_time_s=self.clock.current_time,score=self.score))+'\n');self.trace.flush()
        return self.score

    def discard_control(self):
        self.buffer.clear();self.enabled=False

    def pause(self):self.composite.pause();self.discard_control()

    def activation_geometry(self):
        # Use actual requested stock, then the actual destination object, never
        # a fixed shelf region throughout the carry-and-place stage.
        if self.composite.phase=='pick':
            if self.source is not None and getattr(self.source,'latest',None) is not None:
                from evaluation.tasks.shelf_pick.geometry import rotation
                bounds=[]
                for path in self.scene.identity.candidates:
                    pose=self.source.latest['poses'][path]
                    vertices=rotation(pose['quaternion_wxyz']).apply(self.scene.workspace.collision_vertices[path])+pose['position_m']
                    bounds.append([vertices.min(axis=0).tolist(),vertices.max(axis=0).tolist()])
                return bounds
            targets=[o for o in self.manifest['objects'] if o['path']+'/Asset' in self.scene.identity.candidates]
        else:targets=[o for o in self.manifest['objects'] if o['path']==self.scene.tray]
        if not targets:raise ValueError('Missing activation bounds')
        return [o['visible_bounds_world_m'] for o in targets]

    def close(self):
        if self.source:self.source.close()
        self.trace.close()
