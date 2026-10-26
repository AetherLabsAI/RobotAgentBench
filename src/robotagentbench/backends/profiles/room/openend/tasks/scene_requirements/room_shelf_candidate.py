"""Generic selected-task room host, configured before native initialization.

This is the unified-room entry, not the public standalone-task HTTP registry.
Persistent native device/readers are independent of selected scoring consumers.
"""
from collections import deque
from dataclasses import asdict,replace
from pathlib import Path
import hashlib,json,math
from .dispatch import MultiTaskDispatcher,TaskRegistration
from .tasks.scores import CoffeeTask,FryerTask,interrupt_hold
from .tasks.remote_transfer.port import RemotePort
from .tasks.scene_requirements.runtime_ports import GlassesPort,WalletPort,FridgePort
from .tasks.scene_requirements.shelf_port import SnackTrayPort
from .tasks.all_objects_integration.ports import FACTORIES as OBJECT_FACTORIES
from .tasks.trash_transfer.port import TrashPort
from .tasks.trash_transfer.registration import expand_request as expand_trash_request
from .tasks.cloth_clear.registration import ClothClearRegistration, resolve_request as resolve_cloth_request
from .tasks.cloth_clear.port import ClothClearPort
from evaluation.tasks.coffee_machine.openended_composite_v1.composite import Frame as CoffeeFrame
from evaluation.tasks.air_fryer.continuous_openended_v1.composite import Frame as FryerFrame


class CoffeePort:
    asset='coffee_machine'
    physics_hz=240
    def __init__(self,stage,manifest,output,episode_id,instance_id,parameters):
        if parameters:raise ValueError('No coffee parameter override supported')
        from .tasks.coffee_startup import CoffeeSceneStartup
        from evaluation.tasks.coffee_machine import isaac_evidence as old
        self.output=Path(output);self.output.mkdir(exist_ok=False)
        self.owner=CoffeeSceneStartup()
        self.owner.prepare(stage,manifest,episode_id=episode_id,instance_id=instance_id)
        from .tasks.coffee_counter import install
        counter_receipt=install(self.owner.service,manifest)
        (self.output/'counter_binding.json').write_text(json.dumps(counter_receipt,indent=2)+'\n')
        self.binding=self.owner.service.identity
        self.history=deque(maxlen=math.ceil(old.INITIAL_WINDOW_S/old.PHYSICS_DT-1e-9)+2)
        self.bag_geometry=old.BagGeometry(stage,self.owner.service.roles['bag'])
        self.hashes={l.realPath:hashlib.sha256(Path(l.realPath).read_bytes()).hexdigest()
                     for l in stage.GetUsedLayers() if l.realPath}
        self.consecutive_240=0;self.initial=None;self.composite=None;self.score=None;self.index=-1
        self.trace=(self.output/'selected_facts.jsonl').open('x')
    @property
    def ended(self):return self.composite is not None and self.composite.ended
    def initialize(self,clock,view):
        self.clock=clock
        # Existing source's idle-60 support; no dense scorer attached to service.
        self.owner.hotwake=self
        self.owner.on_physics_ready(clock,simulation_view=view)
    def after_physics(self,helper_report):
        self.owner.after_physics(helper_report)
        self._reader_was_idle=False
        if self.clock.hz==240:
            self.consecutive_240+=1;s=self.owner.service
            self.history.append(dict(poses=s.latest['sample'].poses,contacts=s.latest['contacts'],
                camera=list(self.owner.source.last_camera_contacts),
                machine=s.window['robot_machine'],button=s.window['robot_button']))
        else:self.consecutive_240=0;self.history.clear()
    def after_inactive(self,helper_report):
        from .tasks.all_objects_integration.idle_devices import idle_coffee
        idle_coffee(self,helper_report)

    def accept_initial(self):
        import numpy as np
        from evaluation.tasks.coffee_machine import isaac_evidence as old
        from evaluation.tasks.coffee_machine.api_runtime import INITIAL_TILT_DEG, INITIAL_BAG_TILT_DEG, BUTTON_RELEASED_M
        from evaluation.tasks.coffee_machine.facts import COUNTER
        s = self.owner.service
        roles = s.roles
        rows = list(self.history)
        now = s.latest
        facts = s.facts(0, '1-1')
        bag = now['sample'].poses[roles['bag']]
        local = old.rotation(bag['quaternion_wxyz']).inv().apply(
            np.asarray(now['sample'].bean_positions)-bag['position_m'])
        in_bag = int(self.bag_geometry.inside(local).sum())
        span = lambda p: float(np.ptp([r['poses'][p]['position_m'] for r in rows], axis=0).max())
        checks = dict(history_complete=len(rows)==self.history.maxlen and self.consecutive_240>=240,
            sources_unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in self.hashes.items()),
            objects_at_rest=max(span(roles['cup']),span(roles['bag']))<=old.INITIAL_SPAN_M,
            robot_touches_nothing=not any(r['contacts'][p]['robot'] for r in rows for p in s.objects)
                and not any(r['machine'] or r['button'] for r in rows),
            wrist_cameras_clear=not any(r['camera'] for r in rows),
            no_ground=not any(r['contacts'][p]['ground'] for r in rows for p in s.objects),
            cup_upright=facts.cup.tilt_deg<=INITIAL_TILT_DEG,
            bag_upright=facts.bag.tilt_deg<=INITIAL_BAG_TILT_DEG,
            cup_on_counter=COUNTER in facts.cup.supports and not facts.cup.machine_body_contact,
            bag_on_counter=COUNTER in facts.bag.supports,
            beans_in_bag=in_bag==facts.beans_total==100, hopper_empty=facts.beans_in_hopper==0,
            device_idle=not facts.button.dispensing and not facts.liquid.emitted,
            button_up=-now['button_depth']<=BUTTON_RELEASED_M)
        receipt = dict(accepted=all(checks.values()), checks=checks, beans_in_bag=in_bag,
                       measured_facts=facts.as_dict(), roles=roles,
                       support_contact_pairs=list(self.owner.source.contacts.pairs()),
                       scope='task-local native initial acceptance; not full-room collision certification')
        (self.output/'coffee_initial.json').write_text(json.dumps(receipt,indent=2)+'\n')
        if not receipt['accepted']: raise RuntimeError('Coffee initial rejected: '+str(checks))
        self.initial = receipt
        self.composite = CoffeeTask(s.identity)
        self.composite.source='native_physx'
        self.score=self.composite.observe(CoffeeFrame(s.identity,facts))
        self.index=0
        s._clear_window()
        return receipt


    def observe(self):
        if self.clock.hz!=240:raise RuntimeError('Selected coffee must run at approved rate')
        if self.owner.service.latest['sample'].time_s!=self.clock.current_time:raise RuntimeError('Stale selected coffee evidence')
        self.index+=1;s=self.owner.service
        facts=s.facts(self.index,self.composite.phase)
        self.score=self.composite.observe(CoffeeFrame(self.binding,facts))
        self.trace.write(json.dumps(dict(facts=facts.as_dict(),score=self.score,
                         simulation_time_s=self.clock.current_time),default=list)+'\n');self.trace.flush()
        s._clear_window()
        return self.score
    def pause(self):interrupt_hold(self.composite)
    def finalize(self,reason):
        self.score=self.composite.finalize(reason)
        return self.score
    def close(self):
        self.owner.close();self.trace.close()


class FryerPort:
    asset='air_fryer'
    physics_hz=240
    def __init__(self,stage,manifest,output,episode_id,instance_id,parameters):
        if set(parameters)!={'bread_variant'} or parameters['bread_variant'] not in ('sugar','no_sugar'):
            raise ValueError('Explicit approved fryer variant required')
        from .tasks.fryer_reader import SelectedFryerReader
        self.source=SelectedFryerReader(stage,manifest,output,episode_id,instance_id,parameters['bread_variant'])
        self.binding=self.source.binding;self.variant=parameters['bread_variant']
        self.initial=None;self.composite=None;self.score=None;self.index=-1
        self.trace=(Path(output)/'selected_scores.jsonl').open('x')
    @property
    def ended(self):return self.composite is not None and self.composite.termination is not None
    def initialize(self,clock,view):
        self.clock=clock;self.source.initialize(clock,view,allow_idle_60hz=True)
    def after_physics(self,helper_report):
        self.source.after_physics(helper_report)
        # Inherited countertop exemption requires uninterrupted real support,
        # including time while another task is selected.
        if self.composite is not None and self.composite.inherited_counter_support:
            c=self.source.latest['cakes'][self.source.paths['other']]
            if c['robot'] or c['ground'] or c['basket'] or c['supports']!=['counter']:
                self.composite.inherited_counter_support=False
    def after_inactive(self,helper_report):
        from .tasks.all_objects_integration.idle_devices import idle_fryer
        idle_fryer(self,helper_report)

    def accept_initial(self):
        self.initial=self.source.accept_initial()
        self.composite=FryerTask(self.binding,self.variant)
        self.index=0
        self.score=self.composite.observe(FryerFrame(self.binding,self.source.selected_facts(0)))
        return self.initial
    def observe(self):
        if self.clock.hz!=240:raise RuntimeError('Selected fryer must run at approved rate')
        if self.source.latest['time_s']!=self.clock.current_time:raise RuntimeError('Stale selected fryer evidence')
        self.index+=1;facts=self.source.selected_facts(self.index)
        self.score=self.composite.observe(FryerFrame(self.binding,facts))
        self.trace.write(json.dumps(dict(facts=facts.as_dict(),score=self.score,
                         simulation_time_s=self.clock.current_time),default=list)+'\n');self.trace.flush()
        return self.score
    def pause(self):interrupt_hold(self.composite)
    def finalize(self,reason):
        self.score=(self.composite.summary() if self.composite.termination is not None
                    else self.composite.finalize(reason))
        return self.score
    def close(self):self.source.close();self.trace.close()


FACTORIES={'coffee':CoffeePort,'fryer':FryerPort,'remote_transfer':RemotePort,'glasses_transfer':GlassesPort,'wallet_transfer':WalletPort}
FACTORIES.update(OBJECT_FACTORIES)
FACTORIES['trash_transfer']=TrashPort
FACTORIES['cloth_clear']=ClothClearPort
FACTORIES['fridge_to_microwave']=FridgePort
FACTORIES['snack_drink_to_tray']=SnackTrayPort


def validate_request(request):
    if set(request)!={'tasks','no_progress_timeout_s','selection_order'} or not isinstance(request['tasks'],list):
        raise ValueError('Explicit task list and room policy required')
    if request['selection_order'] not in ('front_then_distance','distance_then_front'):raise ValueError('Unknown selection order')
    n=request['no_progress_timeout_s']
    if isinstance(n,bool) or not isinstance(n,(int,float)) or not math.isfinite(n) or n<=0:raise ValueError('Positive inactivity timeout required')
    ids=set();kinds=set()
    for t in request['tasks']:
        if set(t)!={'instance_id','type','parameters'}:raise ValueError('Explicit task declaration required')
        k=t['instance_id'];kind=t['type']
        if not isinstance(k,str) or not k or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in k):raise ValueError('Safe instance ID required')
        if k in ids or (kind in kinds and kind not in ('paper_cup_transfer','trash_transfer')):raise ValueError('Duplicate instance or same actual asset')
        if kind not in FACTORIES or not isinstance(t['parameters'],dict):raise ValueError('Unsupported adapter')
        if kind=='coffee' and t['parameters']:raise ValueError('No coffee overrides')
        if kind=='remote_transfer' and t['parameters']:raise ValueError('No remote overrides')
        if kind=='glasses_transfer' and t['parameters']:raise ValueError('No glasses overrides')
        if kind=='wallet_transfer' and t['parameters']:raise ValueError('No wallet overrides')
        if kind=='fridge_to_microwave':FridgePort.validate_parameters(t['parameters'])
        if kind=='snack_drink_to_tray':SnackTrayPort.validate_parameters(t['parameters'])
        if kind in OBJECT_FACTORIES:OBJECT_FACTORIES[kind].validate_parameters(t['parameters'])
        if kind=='trash_transfer':TrashPort.validate_parameters(t['parameters'])
        if kind=='cloth_clear':ClothClearRegistration.validate_parameters(t['parameters'])
        if kind=='fryer' and (set(t['parameters'])!={'bread_variant'} or t['parameters']['bread_variant'] not in ('sugar','no_sugar')):raise ValueError('Fryer variant required')
        ids.add(k);kinds.add(kind)
    return request['tasks']


def resolve_objects(tasks,manifest):
    """Resolve all exact target identities before constructing any reader."""
    result={};claimed=set()
    for t in tasks:
        cls=FACTORIES[t['type']]
        target=t['parameters'].get('target_path')
        matches=([cls.resolve_object(t['parameters'],manifest)] if hasattr(cls,'resolve_object') else
                 [o for o in manifest['objects'] if o['id']==cls.asset and (target is None or o['path']==target)])
        if len(matches)!=1:raise ValueError('Missing or ambiguous scene object '+cls.asset)
        obj=matches[0]
        if obj['path'] in claimed:raise ValueError('Two tasks claim the same scene object '+obj['path'])
        claimed.add(obj['path']);result[t['instance_id']]=obj
    return result


def prepare_registration(request,manifest):
    """Read-only pre-native binding, also used by unified room generation.

    Catalog/request registration is not proof a native scoring reader exists.
    """
    tasks=validate_request(request)
    cloth=resolve_cloth_request(request,manifest)
    expanded=expand_trash_request(request,manifest) if any(t['type']=='trash_transfer' for t in tasks) else request
    resolved=resolve_objects(validate_request(expanded),manifest)
    targets={p for group in cloth for p in group['target_paths']}
    if any(o['path'] in targets for t,o in resolved.items() if t not in {g['instance_id'] for g in cloth}):
        raise ValueError('Cloth-clear member is also claimed by another task')
    return dict(request=request,resolved_objects={k:v['path'] for k,v in resolved.items()},
                cloth_clear_groups=cloth,native_ready=True,
                pending=['cloth_clear_native_manipulation_validation'] if cloth else [],
                validation_scope='identity_binding_only_not_native_acceptance')


class MultiTaskRoom:
    def __init__(self,stage,manifest,output,episode_id,*,request,sensor,workspace_selector=None):
        original_request=request
        # Only selected garbage families invoke semantic expansion. Old episodes
        # with no garbage declaration retain their original request contract.
        if any(t.get('type')=='trash_transfer' for t in request.get('tasks',[])):
            validate_request(request)
            request=expand_trash_request(request,manifest)
        tasks=validate_request(request)
        if any(t['type']=='cloth_clear' for t in tasks):
            prepare_registration(request,manifest)
        for t in tasks:
            if t['type']=='trash_transfer':TrashPort.preflight(t['parameters'],manifest)
        if workspace_selector is None and stage is not None:
            from .tasks.all_objects_integration.workspace_physics import from_scene
            workspace_selector=from_scene(stage,manifest,sensor)
        if workspace_selector is not None:
            bound={w.id for w in workspace_selector.workspaces}
            missing={w for t in tasks if getattr(FACTORIES[t['type']],'selection_mode','sector')=='sector'
                     for w in getattr(FACTORIES[t['type']],'required_workspaces',(t['type'],))}-bound
            if missing:raise ValueError('Unbound requested device workspaces: '+str(sorted(missing)))
        self.output=Path(output);self.output.mkdir(exist_ok=False)
        self.ports={};self.pending=None;self.started=False;self.closed=False;self.faulted=False
        self.clock=None;self.transitions=[];self.previous=None;self.previous_active=set();self.frame=0
        self.workspace_selector=workspace_selector
        registrations=[]
        resolved=resolve_objects(tasks,manifest)
        # Preflight every requested object before any partial source construction.
        for t in tasks:
            cls=FACTORIES[t['type']]
            lo,hi=resolved[t['instance_id']]['visible_bounds_world_m']
            registrations.append(TaskRegistration(t['instance_id'],getattr(cls,'initial_workspace',t['type']),
                tuple((lo[k]+hi[k])/2 for k in (0,1)),
                ((lo[0],lo[1]),(hi[0],lo[1]),(hi[0],hi[1]),(lo[0],hi[1])),cls.physics_hz,getattr(cls,'selection_mode','sector')))
        self.dispatch=MultiTaskDispatcher(registrations,sector_radius=sensor['range'],sector_half_angle=sensor['halfAngle'],
            no_progress_timeout_s=request['no_progress_timeout_s'],selection_order=request['selection_order'])
        try:
            for t in tasks:
                self.ports[t['instance_id']]=FACTORIES[t['type']](stage,manifest,self.output/t['instance_id'],episode_id,t['instance_id'],t['parameters'])
                self.ports[t['instance_id']].workspace_rates=workspace_selector is not None
        except Exception:
            for p in self.ports.values():p.close()
            raise
        self.reader_counts={k:dict(full=0,idle=0) for k in self.ports}
        if workspace_selector is not None:
            (self.output/'workspace_physics.json').write_text(json.dumps(workspace_selector.receipt(),indent=2)+'\n')
        self.trace=(self.output/'controls.jsonl').open('x')
        (self.output/'registration.json').write_text(json.dumps(dict(request=request,requested_task_families=original_request,objects={k:(dict(p.binding.objects) if p.binding is not None else {'target':p.geometry.target}) for k,p in self.ports.items()},
            geometry=[asdict(r) for r in registrations],sensor=sensor,
            scope='unified_room_in_process_not_standalone_HTTP_registry'),indent=2)+'\n')

    def _switch_clock(self,target):
        old=self.clock.hz
        if old==target:return
        snapshot,t=self.clock.snapshot(),self.clock.current_time
        if not isinstance(snapshot,bytes):raise RuntimeError('Immutable native state required')
        self.faulted=True;self.clock.set_hz(target)
        if self.clock.hz!=target or self.clock.current_time!=t or self.clock.snapshot()!=snapshot:
            raise RuntimeError('Clock switch changed physical state/time')
        self.faulted=False
        self.transitions.append(dict(frame=self.frame,from_hz=old,to_hz=target,time_s=t,physical_state_unchanged=True,
                                     reason='initial_acceptance' if not self.started else ('selected_workspace' if self.workspace_selector is not None else 'selected_task')))

    def initialize(self,clock,view):
        if self.clock is not None:raise RuntimeError('One binding per room')
        self.clock=clock
        self._switch_clock(240)  # common original 1s native initial acceptance
        for p in self.ports.values():p.initialize(clock,view)

    def after_physics(self,helper_report):
        if self.closed or self.faulted or self.clock is None:raise RuntimeError('Room not ready')
        if self.started and self.pending is None:raise RuntimeError('Substep outside control action')
        if not self.started:
            for p in self.ports.values():p.after_physics(helper_report)
            return
        plan=self.pending[1]
        active=set(plan.get('scoring_tasks',plan.get('active_tasks',[] if plan['current_task'] is None else [plan['current_task']])))
        if 'scoring_tasks' not in plan and plan['switch_pending']:active=set()
        helper_report=dict(helper_report)
        # One native report fetch for all portable contact detectors.
        contact_port=next((p for p in self.ports.values() if getattr(p,'selection_mode','sector')=='contact'),None)
        if contact_port is not None:
            helper_report['_contact_report']=contact_port.interface.get_contact_report()
        for name,p in self.ports.items():
            if name in active and not p.ended:
                p.after_physics(helper_report)
                if hasattr(self,'reader_counts'):self.reader_counts[name]['full']+=1
            else:
                p.after_inactive(helper_report)
                if hasattr(self,'reader_counts'):self.reader_counts[name]['idle']+=1

    def accept_initial(self):
        if self.started:raise RuntimeError('Never reset on entry')
        receipts={k:p.accept_initial() for k,p in self.ports.items()}
        if any(p.score['score']!=0 for p in self.ports.values()):raise RuntimeError('Initial state cannot award task progress')
        self.dispatch.start(accepted_scene=True,simulation_time_s=self.clock.current_time)
        self.started=True
        (self.output/'initial_acceptance.json').write_text(json.dumps(receipts,indent=2,default=list)+'\n')
        return receipts

    def begin_control(self,robot,*,command_vw,base_velocity):
        if not self.started or self.closed or self.faulted or self.pending is not None:raise RuntimeError('Room action unavailable')
        if len(command_vw)!=2 or len(base_velocity)!=6 or not all(math.isfinite(v) for v in (*command_vw,*base_velocity)):raise ValueError('Actual command/native velocity required')
        for key,port in self.ports.items():
            if hasattr(port,'workspace_id'):
                # Routing only: preserve identity, score and watchdog across phases.
                self.dispatch.registrations=tuple(replace(r,kind=port.workspace_id,
                    physics_hz=port.requested_physics_hz) if r.instance_id==key else r
                    for r in self.dispatch.registrations)
            if hasattr(port,'selection_geometry') and (self.dispatch.order=='front_then_distance' or hasattr(port,'workspace_id')):
                geometry=port.selection_geometry()
                if geometry is not None:self.dispatch.update_geometry(key,*geometry)
        activity={key:port.selection_request() for key,port in self.ports.items() if getattr(port,'selection_mode','sector')=='contact'}
        workspace=None if self.workspace_selector is None else self.workspace_selector.select(robot)
        selection=self.dispatch.select(robot,simulation_time_s=self.clock.current_time,activity=activity,workspace=workspace)
        if selection['timed_out']:
            self._settle_terminal()
            return dict(selection,substeps=0,physics_hz=self.clock.hz,physics_dt_s=1/self.clock.hz)
        selected=selection['current_task']
        active=set(selection['active_tasks'])
        self.previous=selected
        target=selection['requested_physics_hz']
        stopped=(max(map(abs,command_vw))<.001 and math.sqrt(sum(v*v for v in base_velocity[:3]))<.005 and math.sqrt(sum(v*v for v in base_velocity[3:]))<.01)
        if target!=self.clock.hz and stopped:self._switch_clock(target)
        hz=self.clock.hz
        if workspace is not None:
            # Portable scoring consumes the real 60/240 dt even when the base
            # guard defers a scene-rate switch. Device scoring still requires
            # its original 240 Hz and its own selected workspace.
            scoring={name for name in active if getattr(self.ports[name],'selection_mode','sector')=='contact'
                     or hz==getattr(self.ports[name],'requested_physics_hz',240)}
        else:scoring=active if hz==target else set()
        for name in self.previous_active-scoring:self.ports[name].pause()
        self.previous_active=scoring
        for name,port in self.ports.items():
            if hasattr(port,'begin_selected_control'):port.begin_selected_control(name in scoring)
        plan=dict(selection,physics_hz=hz,physics_dt_s=1/hz,substeps=hz//20,frame=self.frame,
                  switch_pending=hz!=target,scoring_tasks=sorted(scoring))
        self.pending=(self.clock.current_time,plan)
        return plan

    def finish_control(self,*,completed_substeps):
        if self.pending is None or self.faulted:raise RuntimeError('No pending action')
        start,plan=self.pending
        if completed_substeps!=plan['substeps'] or self.clock.hz!=plan['physics_hz'] or abs(self.clock.current_time-start-.05)>1e-6:
            self.faulted=True;raise RuntimeError('Missing substeps or broken control clock')
        try:
            timing=self.dispatch.complete_physics_frame(simulation_time_s=self.clock.current_time)
            key=plan['current_task']
            if not timing['timed_out']:
                scores={}
                for name in plan['scoring_tasks']:
                    port=self.ports[name]
                    if not port.ended:scores[name]=port.consume_control(True) if hasattr(port,'consume_control') else port.observe()
                if scores:self.dispatch.record_scores(scores)
                timing=self.dispatch.finish_frame()
            if self.dispatch.ended:timing=self._settle_terminal()
            result=dict(plan=plan,outcome=timing,selected_control_steps={k:p.index for k,p in self.ports.items()},
                        reader_substeps={k:dict(v) for k,v in self.reader_counts.items()})
            self.trace.write(json.dumps(result,default=list)+'\n');self.trace.flush()
        except Exception:
            self.faulted=True;raise
        self.pending=None;self.frame+=1
        return result

    def _settle_terminal(self):
        if hasattr(self,'terminal_result'):return self.terminal_result
        reason=self.dispatch.termination_reason
        scores={k:p.finalize(reason) for k,p in self.ports.items()}
        self.terminal_result=self.dispatch.settle_terminal_scores(scores)
        (self.output/'final_result.json').write_text(json.dumps(self.terminal_result,indent=2,default=list)+'\n')
        return self.terminal_result

    def close(self):
        if self.closed:return
        for p in self.ports.values():p.close()
        self.trace.close();self.closed=True
