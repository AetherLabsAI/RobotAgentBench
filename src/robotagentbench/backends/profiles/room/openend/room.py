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
from .tasks.umbrella_transfer.port import UmbrellaPort
from .tasks.trash_transfer.port import TrashPort
from .tasks.trash_transfer.registration import expand_request as expand_trash_request
from .tasks.cloth_clear.registration import ClothClearRegistration, resolve_request as resolve_cloth_request
from .tasks.cloth_clear.port import ClothClearPort
from .tasks.wipe_table.port import WipePort
from .tasks.meal_tray.native_port import FACTORIES as MEAL_FACTORIES,link_ports as link_meal_ports,preflight as meal_preflight
from .tasks.meal_tray.registration import dependencies as meal_dependencies
from .tasks.meal_tray.preparation import Preparation
from .tasks.meal_tray.task_contact_routing import NativeContactRoutes,roots_for,rate_compatible
from .tasks.process_final.lifecycle import ContactLifecycle
from .tasks.destination_transfers.tray_port import TrayPort
from evaluation.tasks.coffee_machine.openended_composite_v1.composite import Frame as CoffeeFrame
from evaluation.tasks.air_fryer.continuous_openended_v1.composite import Frame as FryerFrame


class CoffeePort:
    asset='coffee_machine'
    physics_hz=240
    supported_physics_hz=(60,240)
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
        self.consecutive_240=self.consecutive_240+1 if self.clock.hz==240 else 0
        s=self.owner.service
        self.history.append(dict(time_s=self.clock.current_time,physics_dt_s=self.clock.get_physics_dt(),
            poses=s.latest['sample'].poses,contacts=s.latest['contacts'],
            camera=list(self.owner.source.last_camera_contacts),
            machine=s.window['robot_machine'],button=s.window['robot_button']))
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
        from openend.tasks.meal_tray.nonmotion_closeout_v1.initial_windows import initial_window
        rows,complete = initial_window(self.history,old.INITIAL_WINDOW_S,self.clock.get_physics_dt())
        now = s.latest
        facts = s.facts(0, '1-1')
        bag = now['sample'].poses[roles['bag']]
        local = old.rotation(bag['quaternion_wxyz']).inv().apply(
            np.asarray(now['sample'].bean_positions)-bag['position_m'])
        in_bag = int(self.bag_geometry.inside(local).sum())
        span = lambda p: float(np.ptp([r['poses'][p]['position_m'] for r in rows], axis=0).max())
        checks = dict(history_complete=complete,
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
        if getattr(self,'meal_preparation_kind',None)=='coffee':
            self.composite=Preparation(self.composite,kind='coffee')
        self.composite.source='native_physx'
        self.score=self.composite.observe(CoffeeFrame(s.identity,facts))
        self.index=0
        s._clear_window()
        return receipt


    def observe(self):
        if self.clock.hz not in self.supported_physics_hz:raise RuntimeError('Unsupported coffee workspace clock')
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
        self.score=(self.composite.summary() if isinstance(self.composite,Preparation) and self.composite.ended
                    else self.composite.finalize(reason))
        return self.score
    def close(self):
        self.owner.close();self.trace.close()


class FryerPort:
    asset='air_fryer'
    physics_hz=240
    supported_physics_hz=(60,240)
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
        if getattr(self,'meal_preparation_kind',None)=='fryer':
            self.composite=Preparation(self.composite,kind='fryer')
        self.index=0
        self.score=self.composite.observe(FryerFrame(self.binding,self.source.selected_facts(0)))
        return self.initial
    def observe(self):
        if self.clock.hz not in self.supported_physics_hz:raise RuntimeError('Unsupported fryer workspace clock')
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
FACTORIES['umbrella_transfer']=UmbrellaPort
FACTORIES.update(MEAL_FACTORIES)
FACTORIES['trash_transfer']=TrashPort
FACTORIES['cloth_clear']=ClothClearPort  # retained internally for ordinary short tasks
FACTORIES['wipe_table']=WipePort
FACTORIES['fridge_to_microwave']=FridgePort
FACTORIES['snack_drink_to_tray']=SnackTrayPort
FACTORIES['meal_tray_transfer']=TrayPort


def validate_request(request):
    if set(request)!={'tasks','no_progress_timeout_s','selection_order'} or not isinstance(request['tasks'],list):
        raise ValueError('Explicit task list and room policy required')
    if request['selection_order'] not in ('front_then_distance','distance_then_front'):raise ValueError('Unknown selection order')
    from .tasks.meal_tray.product_claims import validate_groups
    validate_groups(request['tasks'])
    n=request['no_progress_timeout_s']
    if isinstance(n,bool) or not isinstance(n,(int,float)) or not math.isfinite(n) or n<=0:raise ValueError('Positive inactivity timeout required')
    ids=set();kinds=set()
    for t in request['tasks']:
        if set(t)!={'instance_id','type','parameters'}:raise ValueError('Explicit task declaration required')
        k=t['instance_id'];kind=t['type']
        if not isinstance(k,str) or not k or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in k):raise ValueError('Safe instance ID required')
        if k in ids or (kind in kinds and kind not in ('paper_cup_transfer','trash_transfer','snack_drink_to_tray','cutlery_to_meal_tray')):raise ValueError('Duplicate instance or same actual asset')
        if kind not in FACTORIES or not isinstance(t['parameters'],dict):raise ValueError('Unsupported adapter')
        if kind=='coffee' and t['parameters']:raise ValueError('No coffee overrides')
        if kind=='remote_transfer' and t['parameters']:raise ValueError('No remote overrides')
        if kind=='glasses_transfer':GlassesPort.validate_parameters(t['parameters'])
        if kind=='wallet_transfer':WalletPort.validate_parameters(t['parameters'])
        if kind=='meal_tray_transfer':TrayPort.validate_parameters(t['parameters'])
        if kind=='fridge_to_microwave':FridgePort.validate_parameters(t['parameters'])
        if kind=='snack_drink_to_tray':SnackTrayPort.validate_parameters(t['parameters'])
        if kind in OBJECT_FACTORIES:OBJECT_FACTORIES[kind].validate_parameters(t['parameters'])
        if kind=='umbrella_transfer':UmbrellaPort.validate_parameters(t['parameters'])
        if kind in MEAL_FACTORIES:MEAL_FACTORIES[kind].validate_parameters(t['parameters'])
        if kind=='trash_transfer':TrashPort.validate_parameters(t['parameters'])
        if kind=='cloth_clear':ClothClearRegistration.validate_parameters(t['parameters'])
        if kind=='wipe_table':WipePort.validate_parameters(t['parameters'])
        if kind=='fryer' and (set(t['parameters'])!={'bread_variant'} or t['parameters']['bread_variant'] not in ('sugar','no_sugar')):raise ValueError('Fryer variant required')
        ids.add(k);kinds.add(kind)
    from .tasks.scene_requirements.shelf_instances import validate_instances
    validate_instances(request['tasks'])
    meal_dependencies(request['tasks'])
    return request['tasks']


def resolve_objects(tasks,manifest):
    """Resolve all exact target identities before constructing any reader."""
    result={};claimed={}
    # The only shared portable target is this explicitly declared sequential
    # handoff. Unrelated/doubled task claims remain invalid.
    sequential={frozenset((e['instance_id'],e['prerequisite_instance_id']))
        for e in meal_dependencies(tasks) if e['type']=='microwave_box_to_meal_tray'}
    for t in tasks:
        cls=FACTORIES[t['type']]
        target=t['parameters'].get('target_path')
        matches=([cls.resolve_instance(t['parameters'],manifest,t['instance_id'])] if hasattr(cls,'resolve_instance') else
                 [cls.resolve_object(t['parameters'],manifest)] if hasattr(cls,'resolve_object') else
                 [o for o in manifest['objects'] if o['id']==cls.asset and (target is None or o['path']==target)])
        if len(matches)!=1:raise ValueError('Missing or ambiguous scene object '+cls.asset)
        obj=matches[0]
        owners=claimed.setdefault(obj['path'],[])
        if owners and (len(owners)!=1 or frozenset((*owners,t['instance_id'])) not in sequential):
            raise ValueError('Two tasks claim the same scene object '+obj['path'])
        owners.append(t['instance_id']);result[t['instance_id']]=obj
    return result


def prepare_registration(request,manifest):
    """Read-only pre-native binding, also used by unified room generation.

    Catalog/request registration is not proof a native scoring reader exists.
    """
    tasks=validate_request(request)
    meal_preflight(tasks,manifest)
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
    def __init__(self,stage,manifest,output,episode_id,*,request,sensor,workspace_selector=None,native_factories=None):
        original_request=request
        # Only selected garbage families invoke semantic expansion. Old episodes
        # with no garbage declaration retain their original request contract.
        if any(t.get('type')=='trash_transfer' for t in request.get('tasks',[])):
            validate_request(request)
            request=expand_trash_request(request,manifest)
        tasks=validate_request(request)
        # Trusted in-process callers may select existing task-owned native ports.
        # Public JSON cannot supply factories. Identity/type validation above and
        # below stays authoritative; this does not mutate the shared registry.
        native_factories={} if native_factories is None else dict(native_factories)
        if set(native_factories)-{t['instance_id'] for t in tasks}:
            raise ValueError('Native factory override for an unregistered instance')
        factories={t['instance_id']:native_factories.get(t['instance_id'],FACTORIES[t['type']]) for t in tasks}
        for t in tasks:
            if t['instance_id'] not in native_factories:continue
            cls=factories[t['instance_id']]; original=FACTORIES[t['type']]
            if not isinstance(cls,type) or cls.asset!=original.asset or cls.physics_hz!=original.physics_hz:
                raise ValueError('Native factory must retain the registered asset and physics rate')
        if any(t['type']=='cloth_clear' or t['type'] in MEAL_FACTORIES for t in tasks):
            prepare_registration(request,manifest)
        for t in tasks:
            if t['type']=='trash_transfer':TrashPort.preflight(t['parameters'],manifest)
        if workspace_selector is None and stage is not None:
            from .tasks.all_objects_integration.workspace_physics import from_scene
            workspace_selector=from_scene(stage,manifest,sensor)
        if workspace_selector is not None:
            bound={w.id for w in workspace_selector.workspaces}
            missing={w for t in tasks if getattr(factories[t['instance_id']],'selection_mode','sector')=='sector'
                     for w in getattr(factories[t['instance_id']],'required_workspaces',(t['type'],))}-bound
            if missing:raise ValueError('Unbound requested device workspaces: '+str(sorted(missing)))
        self.output=Path(output);self.output.mkdir(exist_ok=False)
        self.fridge_helper=None
        self.ports={};self.pending=None;self.started=False;self.closed=False;self.faulted=False
        self.clock=None;self.transitions=[];self.previous=None;self.previous_active=set();self.frame=0
        self.workspace_selector=workspace_selector
        registrations=[]
        resolved=resolve_objects(tasks,manifest)
        # Preflight every requested object before any partial source construction.
        for t in tasks:
            cls=factories[t['instance_id']]
            lo,hi=resolved[t['instance_id']]['visible_bounds_world_m']
            registrations.append(TaskRegistration(t['instance_id'],getattr(cls,'initial_workspace',t['type']),
                tuple((lo[k]+hi[k])/2 for k in (0,1)),
                ((lo[0],lo[1]),(hi[0],lo[1]),(hi[0],hi[1]),(lo[0],hi[1])),cls.physics_hz,
                'contact' if stage is not None else getattr(cls,'selection_mode','sector')))
        self.dispatch=MultiTaskDispatcher(registrations,sector_radius=sensor['range'],sector_half_angle=sensor['halfAngle'],
            no_progress_timeout_s=request['no_progress_timeout_s'],
            selection_order='distance_then_front' if stage is not None else request['selection_order'])
        try:
            if stage is not None:
                from .tasks.meal_tray.fridge_helper_dependencies import ensure
                ensure(manifest)
                from room_fridge_helper_v1 import RoomFridgeHelper
                self.fridge_helper=RoomFridgeHelper.prepare(stage,manifest)
            for t in tasks:
                self.ports[t['instance_id']]=factories[t['instance_id']](stage,manifest,self.output/t['instance_id'],episode_id,t['instance_id'],t['parameters'])
                self.ports[t['instance_id']].workspace_rates=workspace_selector is not None
            link_meal_ports(self.ports,tasks,manifest)
            from .tasks.meal_tray.gripper_wake_v1.host import bindings_for,UnifiedGripperRoutes
            bindings={t['instance_id']:bindings_for(t['type'],self.ports[t['instance_id']],resolved[t['instance_id']],roots_for)
                      for t in tasks if stage is not None}
            self.contact_routes=UnifiedGripperRoutes(stage,bindings) if bindings else None
            self._dormant_readers=set()
        except Exception:
            for p in self.ports.values():p.close()
            if self.fridge_helper is not None:self.fridge_helper.close()
            raise
        self.reader_counts={k:dict(full=0,idle=0) for k in self.ports}
        self.process_lifecycle=ContactLifecycle()
        if workspace_selector is not None:
            (self.output/'workspace_physics.json').write_text(json.dumps(workspace_selector.receipt(),indent=2)+'\n')
        self.trace=(self.output/'controls.jsonl').open('x')
        (self.output/'registration.json').write_text(json.dumps(dict(request=request,requested_task_families=original_request,objects={k:(dict(p.binding.objects) if p.binding is not None else {'target':p.geometry.target}) for k,p in self.ports.items()},
            geometry=[asdict(r) for r in registrations],sensor=sensor,
            task_activation='contact_started_persistent_process_then_20hz_final' if stage is not None else 'legacy_non_native_fixture',device_contact_roots=bindings,
            physics_rate_owner='single_frontal_workspace',
            scope='unified_room_in_process_not_standalone_HTTP_registry'),indent=2)+'\n')

    def _switch_clock(self,target):
        old=self.clock.hz
        if old==target:return
        snapshot,t=self.clock.snapshot(),self.clock.current_time
        if not isinstance(snapshot,bytes):raise RuntimeError('Immutable native state required')
        if self.fridge_helper is not None:self.fridge_helper.validate_hz(target)
        self.faulted=True;self.clock.set_hz(target)
        if self.clock.hz!=target or self.clock.current_time!=t or self.clock.snapshot()!=snapshot:
            raise RuntimeError('Clock switch changed physical state/time')
        if self.fridge_helper is not None:self.fridge_helper.set_hz(target)
        self.faulted=False
        self.transitions.append(dict(frame=self.frame,from_hz=old,to_hz=target,time_s=t,physical_state_unchanged=True,
                                     reason='initial_acceptance' if not self.started else ('selected_workspace' if self.workspace_selector is not None else 'selected_task')))

    def initialize(self,clock,view):
        if self.clock is not None:raise RuntimeError('One binding per room')
        self.clock=clock
        self._switch_clock(60)  # approved four-second 60 Hz initial settling
        from openend.tasks.meal_tray.nonmotion_closeout_v1.step_snapshot import SnapshotViews,bind_contacts
        self.snapshot_views=SnapshotViews(view)
        for p in self.ports.values():p.initialize(clock,self.snapshot_views)
        bind_contacts(self.ports,self.snapshot_views,self.contact_routes)

    def after_physics(self,helper_report):
        pool=getattr(self,'snapshot_views',None)
        if pool is None:return MultiTaskRoom._after_physics(self,helper_report)
        pool.begin()
        try:
            if '_contact_report' in helper_report:
                helper_report=dict(helper_report)
                helper_report['_contact_report']=pool.accept_contact_report(helper_report['_contact_report'])
            return self._after_physics(helper_report)
        finally:pool.end()

    def _after_physics(self,helper_report):
        if self.closed or self.faulted or self.clock is None:raise RuntimeError('Room not ready')
        if self.started and self.pending is None:raise RuntimeError('Substep outside control action')
        if self.fridge_helper is not None:self.fridge_helper.after_physics(self.clock.hz)
        if not self.started:
            routes=getattr(self,'contact_routes',None)
            if routes is not None and hasattr(routes,'archive'):
                routes.after_physics(routes.interface.get_contact_report(),self.clock.current_time)
            for p in self.ports.values():p.after_physics(helper_report)
            return
        plan=self.pending[1]
        active=set(plan.get('scoring_tasks',plan.get('active_tasks',[] if plan['current_task'] is None else [plan['current_task']])))
        if 'scoring_tasks' not in plan and plan['switch_pending']:active=set()
        dependencies={dep for name in active
            for dep in getattr(self.ports[name],'reader_dependencies',())}
        helper_report=dict(helper_report)
        # One report also serves device activation; no full pose/scoring reads.
        routes=getattr(self,'contact_routes',None)
        contact_port=routes or next((p for p in self.ports.values() if getattr(p,'selection_mode','sector')=='contact'),None)
        if contact_port is not None:
            if '_contact_report' not in helper_report:
                helper_report['_contact_report']=contact_port.interface.get_contact_report()
        if routes is not None:routes.after_physics(helper_report['_contact_report'],self.clock.current_time)
        for name,p in self.ports.items():
            if (name in active and not p.ended) or name in dependencies:
                if name in getattr(self,'_dormant_readers',set()):
                    from .tasks.meal_tray.gripper_wake_v1.host import prime_port
                    prime_port(p,routes.archive);self._dormant_readers.remove(name)
                p.after_physics(helper_report)
                if hasattr(self,'reader_counts'):self.reader_counts[name]['full']+=1
            else:
                if routes is not None and hasattr(routes,'archive'):
                    from .tasks.meal_tray.gripper_wake_v1.host import dormant
                    mode=dormant(p,helper_report);self._dormant_readers.add(name)
                    if not hasattr(self,'dormant_counts'):self.dormant_counts={}
                    self.dormant_counts[name]=self.dormant_counts.get(name,0)+1
                else:p.after_inactive(helper_report)
                if hasattr(self,'reader_counts'):self.reader_counts[name]['idle']+=1

    def accept_initial(self):
        if self.started:raise RuntimeError('Never reset on entry')
        receipts={k:p.accept_initial() for k,p in self.ports.items()}
        if any(p.score['score']!=0 for p in self.ports.values()):raise RuntimeError('Initial state cannot award task progress')
        if self.contact_routes is not None:
            if hasattr(self.contact_routes,'arm'):self.contact_routes.arm(self.clock.current_time)
            else:self.contact_routes.observe((),self.clock.current_time)
        self.dispatch.start(accepted_scene=True,simulation_time_s=self.clock.current_time)
        self.started=True
        (self.output/'initial_acceptance.json').write_text(json.dumps(receipts,indent=2,default=list)+'\n')
        return receipts

    def begin_control(self,robot,*,command_vw,base_velocity):
        if not self.started or self.closed or self.faulted or self.pending is not None:raise RuntimeError('Room action unavailable')
        if len(command_vw)!=2 or len(base_velocity)!=6 or not all(math.isfinite(v) for v in (*command_vw,*base_velocity)):raise ValueError('Actual command/native velocity required')
        # Retire only terminal shelf candidates, never their score/state records.
        # Multiple shelf tasks share the tray; a completed one must not win again.
        retired={key for key,port in self.ports.items() if isinstance(port,SnackTrayPort) and port.ended}
        self.dispatch.registrations=tuple(r for r in self.dispatch.registrations if r.instance_id not in retired)
        for key,port in self.ports.items():
            if key in retired:continue
            if hasattr(port,'workspace_id'):
                # Routing only: preserve identity, score and watchdog across phases.
                self.dispatch.registrations=tuple(replace(r,kind=port.workspace_id,
                    physics_hz=port.requested_physics_hz) if r.instance_id==key else r
                    for r in self.dispatch.registrations)
            if hasattr(port,'selection_geometry') and (self.dispatch.order=='front_then_distance' or hasattr(port,'workspace_id')):
                geometry=port.selection_geometry()
                if geometry is not None:self.dispatch.update_geometry(key,*geometry)
        routes=getattr(self,'contact_routes',None)
        centralized=routes is not None and hasattr(routes,'archive')
        activity={} if centralized else {key:port.selection_request() for key,port in self.ports.items()
                  if key not in retired and getattr(port,'selection_mode','sector')=='contact'}
        if routes is not None:
            activity.update({k:v for k,v in routes.requests(self.ports,self.clock.current_time).items() if k not in retired})
        if centralized:
            for key,value in activity.items():
                port=self.ports[key]
                if hasattr(port,'filter_wake_request'):
                    if not port.handoff.ready and port.handoff.parent.success:
                        from .tasks.meal_tray.gripper_wake_v1.host import prime_port
                        prime_port(port,routes.archive)
                    activity[key]=port.filter_wake_request(value)
        if hasattr(self,'process_lifecycle'):
            activity=self.process_lifecycle.activity(activity,self.ports)
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
            # Contact chooses task eligibility, NEVER the physics rate. Keep
            # incompatible old fixed-rate readers paused, visibly, not retimed.
            scoring={name for name in active if rate_compatible(self.ports[name],hz)}
        else:scoring=active if hz==target else set()
        for name in self.previous_active-scoring:self.ports[name].pause()
        self.previous_active=scoring
        for name,port in self.ports.items():
            if hasattr(port,'begin_selected_control'):port.begin_selected_control(name in scoring)
        plan=dict(selection,physics_hz=hz,physics_dt_s=1/hz,substeps=hz//20,frame=self.frame,
                  switch_pending=hz!=target,scoring_tasks=sorted(scoring),
                  task_activation='contact_started_persistent_process_then_20hz_final',
                  deferred_scoring_tasks={name:'reader_rate_incompatible_with_workspace'
                                          for name in sorted(active-scoring)})
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
                        reader_substeps={k:dict(v) for k,v in self.reader_counts.items()},
                        centralized_gripper_wake=(self.contact_routes.index.receipt() if hasattr(getattr(self,'contact_routes',None),'index') else None),
                        dormant_clock_steps=dict(getattr(self,'dormant_counts',{})),
                        environment_helpers={} if self.fridge_helper is None else {'refrigerator':self.fridge_helper.report()})
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
        if self.fridge_helper is not None:self.fridge_helper.close()
        self.trace.close();self.closed=True
