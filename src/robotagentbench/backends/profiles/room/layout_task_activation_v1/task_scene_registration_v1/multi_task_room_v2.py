"""Generic selected-task room host, configured before native initialization.

This is the unified-room entry, not the public standalone-task HTTP registry.
Persistent native device/readers are independent of selected scoring consumers.
"""
from collections import deque
from dataclasses import asdict,replace
from pathlib import Path
import hashlib,json,math
from multi_task_dispatch_v1 import MultiTaskDispatcher,TaskRegistration
from selected_scores_v1 import SelectedCoffee,SelectedFryer,interrupt_hold
from evaluation.tasks.coffee_machine.openended_composite_v1.composite import Frame as CoffeeFrame
from evaluation.tasks.air_fryer.continuous_openended_v1.composite import Frame as FryerFrame
from evaluation.tasks.shelf_pick.openended_composite_v1.room_port import SnackTrayPort
from evaluation.tasks.shelf_pick.openended_composite_v1.catalog import normalize_parameters as snack_parameters
from sector_activation_v1 import sector_polygon,angle_error
from room_fridge_helper_v1 import RoomFridgeHelper
from fridge_to_microwave_v1.room_port import FridgeMicrowavePort, normalize_parameters as fridge_parameters


class CoffeePort:
    asset='coffee_machine'
    physics_hz=240
    def __init__(self,stage,manifest,output,episode_id,instance_id,parameters):
        if parameters:raise ValueError('No coffee parameter override supported')
        from coffee_startup_v1 import CoffeeSceneStartup
        from evaluation.tasks.coffee_machine import isaac_evidence as old
        self.output=Path(output);self.output.mkdir(exist_ok=False)
        self.owner=CoffeeSceneStartup()
        self.owner.prepare(stage,manifest,episode_id=episode_id,instance_id=instance_id)
        from coffee_counter_binding_v1 import install
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
        if self.clock.hz==240:
            self.consecutive_240+=1;s=self.owner.service
            self.history.append(dict(poses=s.latest['sample'].poses,contacts=s.latest['contacts'],
                camera=list(self.owner.source.last_camera_contacts),
                machine=s.window['robot_machine'],button=s.window['robot_button']))
        else:self.consecutive_240=0;self.history.clear()
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
        self.composite = SelectedCoffee(s.identity)
        self.composite.source='native_physx'
        self.score=self.composite.observe(CoffeeFrame(s.identity,facts))
        self.index=0
        s._clear_window()
        return receipt


    def observe(self):
        if self.clock.hz!=240:raise RuntimeError('Selected coffee must run at approved rate')
        self.index+=1;s=self.owner.service
        facts=s.facts(self.index,self.composite.phase)
        self.score=self.composite.observe(CoffeeFrame(self.binding,facts))
        self.trace.write(json.dumps(dict(facts=facts.as_dict(),score=self.score,
                         simulation_time_s=self.clock.current_time),default=list)+'\n');self.trace.flush()
        s._clear_window()
        return self.score
    def pause(self):interrupt_hold(self.composite)
    def close(self):
        self.owner.close();self.trace.close()


class FryerPort:
    asset='air_fryer'
    physics_hz=240
    def __init__(self,stage,manifest,output,episode_id,instance_id,parameters):
        if set(parameters)!={'bread_variant'} or parameters['bread_variant'] not in ('sugar','no_sugar'):
            raise ValueError('Explicit approved fryer variant required')
        from selected_fryer_reader_v1 import SelectedFryerReader
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
    def accept_initial(self):
        self.initial=self.source.accept_initial()
        self.composite=SelectedFryer(self.binding,self.variant)
        self.index=0
        self.score=self.composite.observe(FryerFrame(self.binding,self.source.selected_facts(0)))
        return self.initial
    def observe(self):
        if self.clock.hz!=240:raise RuntimeError('Selected fryer must run at approved rate')
        self.index+=1;facts=self.source.selected_facts(self.index)
        self.score=self.composite.observe(FryerFrame(self.binding,facts))
        self.trace.write(json.dumps(dict(facts=facts.as_dict(),score=self.score,
                         simulation_time_s=self.clock.current_time),default=list)+'\n');self.trace.flush()
        return self.score
    def pause(self):interrupt_hold(self.composite)
    def close(self):self.source.close();self.trace.close()


FACTORIES={'coffee':CoffeePort,'fryer':FryerPort,'snack_drink_to_tray':SnackTrayPort,
           'fridge_to_microwave':FridgeMicrowavePort}


def validate_request(request):
    if set(request)!={'tasks','no_progress_timeout_s','selection_order'} or not isinstance(request['tasks'],list):
        raise ValueError('Explicit task list and room policy required')
    if request['selection_order']!='front_then_distance':raise ValueError('Most frontal only')
    n=request['no_progress_timeout_s']
    if isinstance(n,bool) or not isinstance(n,(int,float)) or not math.isfinite(n) or n<=0:raise ValueError('Positive inactivity timeout required')
    ids=set();kinds=set()
    for t in request['tasks']:
        if set(t)!={'instance_id','type','parameters'}:raise ValueError('Explicit task declaration required')
        k=t['instance_id'];kind=t['type']
        if not isinstance(k,str) or not k or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in k):raise ValueError('Safe instance ID required')
        if k in ids or kind in kinds:raise ValueError('Duplicate instance or same actual asset')
        if kind not in FACTORIES or not isinstance(t['parameters'],dict):raise ValueError('Unsupported adapter')
        if kind=='coffee' and t['parameters']:raise ValueError('No coffee overrides')
        if kind=='fryer' and (set(t['parameters'])!={'bread_variant'} or t['parameters']['bread_variant'] not in ('sugar','no_sugar')):raise ValueError('Fryer variant required')
        if kind=='snack_drink_to_tray':snack_parameters(t['parameters'])
        if kind=='fridge_to_microwave':fridge_parameters(t['parameters'])
        ids.add(k);kinds.add(kind)
    return request['tasks']


class MultiTaskRoom:
    def __init__(self,stage,manifest,output,episode_id,*,request,sensor):
        tasks=validate_request(request)
        self.output=Path(output);self.output.mkdir(exist_ok=False)
        self.ports={};self.pending=None;self.started=False;self.closed=False;self.faulted=False
        self.clock=None;self.transitions=[];self.previous=None;self.frame=0
        self.fridge_helper=None
        registrations=[]
        # Preflight every requested object before any partial source construction.
        for t in tasks:
            cls=FACTORIES[t['type']]
            matches=[o for o in manifest['objects'] if o['id']==cls.asset]
            if len(matches)!=1:raise ValueError('Missing or ambiguous scene object '+cls.asset)
            lo,hi=matches[0]['visible_bounds_world_m']
            registrations.append(TaskRegistration(t['instance_id'],t['type'],
                tuple((lo[k]+hi[k])/2 for k in (0,1)),
                ((lo[0],lo[1]),(hi[0],lo[1]),(hi[0],hi[1]),(lo[0],hi[1])),cls.physics_hz))
        self.dispatch=MultiTaskDispatcher(registrations,sector_radius=sensor['range'],sector_half_angle=sensor['halfAngle'],
            no_progress_timeout_s=request['no_progress_timeout_s'],selection_order=request['selection_order'])
        try:
            self.fridge_helper=RoomFridgeHelper.prepare(stage,manifest)
            for t in tasks:
                self.ports[t['instance_id']]=FACTORIES[t['type']](stage,manifest,self.output/t['instance_id'],episode_id,t['instance_id'],t['parameters'])
        except Exception:
            for p in self.ports.values():p.close()
            if self.fridge_helper is not None:self.fridge_helper.close()
            raise
        self._refresh_targets()
        self.trace=(self.output/'controls.jsonl').open('x')
        (self.output/'registration.json').write_text(json.dumps(dict(request=request,objects={k:dict(p.binding.objects) for k,p in self.ports.items()},
            environment_helpers=self.environment_helpers(),
            geometry=[asdict(r) for r in self.dispatch.registrations],sensor=sensor,
            scope='unified_room_in_process_not_standalone_HTTP_registry'),indent=2)+'\n')

    def _refresh_targets(self,robot=None):
        registrations=[]
        for r in self.dispatch.registrations:
            port=self.ports[r.instance_id]
            if hasattr(port,'requested_physics_hz'):
                r=replace(r,physics_hz=port.requested_physics_hz)
            if hasattr(port,'activation_geometry'):
                candidates=[]
                for lo,hi in port.activation_geometry():
                    xy=tuple((lo[k]+hi[k])/2 for k in (0,1))
                    poly=((lo[0],lo[1]),(hi[0],lo[1]),(hi[0],hi[1]),(lo[0],hi[1]))
                    key=(0,0,0)
                    if robot is not None:
                        dx,dy=(xy[i]-robot['xy'][i] for i in (0,1))
                        key=(not sector_polygon(robot,self.dispatch.radius,self.dispatch.half,poly),
                             angle_error(robot['yaw'],math.degrees(math.atan2(dy,dx))),math.hypot(dx,dy))
                    candidates.append((key,xy,poly))
                _,xy,poly=min(candidates,key=lambda c:c[0])
                r=replace(r,target_xy=xy,target_polygon=poly)
            registrations.append(r)
        self.dispatch.registrations=tuple(registrations)

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
                                     reason='initial_acceptance' if not self.started else 'selected_task'))

    def initialize(self,clock,view):
        if self.clock is not None:raise RuntimeError('One binding per room')
        self.clock=clock
        self._switch_clock(240)  # common original 1s native initial acceptance
        for p in self.ports.values():p.initialize(clock,view)

    def after_physics(self,helper_report):
        if self.closed or self.faulted or self.clock is None:raise RuntimeError('Room not ready')
        if self.started and self.pending is None:raise RuntimeError('Substep outside control action')
        try:
            if self.fridge_helper is not None:self.fridge_helper.after_physics(self.clock.hz)
            # Preserve the existing task-specific report unchanged. Fridge assistance
            # has its own explicit namespace in the room trace, not coffee/fryer events.
            for p in self.ports.values():p.after_physics(helper_report)
        except Exception:
            self.faulted=True
            raise

    def environment_helpers(self):
        return {} if self.fridge_helper is None else {'refrigerator':self.fridge_helper.report()}

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
        self._refresh_targets(robot)
        selection=self.dispatch.select(robot,simulation_time_s=self.clock.current_time)
        if selection['timed_out']:return dict(selection,substeps=0,physics_hz=self.clock.hz,physics_dt_s=1/self.clock.hz)
        selected=selection['current_task']
        if selected!=self.previous and self.previous is not None:self.ports[self.previous].pause()
        self.previous=selected
        selected_hz=selection['requested_physics_hz']
        try:
            reservation=(None if self.fridge_helper is None else self.fridge_helper.rate_request(
                robot,radius=self.dispatch.radius,half_angle=self.dispatch.half))
        except Exception:
            self.faulted=True
            raise
        fridge_required=reservation is not None and reservation['required']
        target=60 if fridge_required else selected_hz
        rate_conflict=bool(fridge_required and selected is not None and selected_hz!=60)
        # A single native scene has one clock. Do not score a 240-Hz task while
        # the approved refrigerator interaction owns the 60-Hz reservation.
        if rate_conflict:self.ports[selected].pause()
        stopped=(max(map(abs,command_vw))<.001 and math.sqrt(sum(v*v for v in base_velocity[:3]))<.005 and math.sqrt(sum(v*v for v in base_velocity[3:]))<.01)
        if target!=self.clock.hz and stopped:self._switch_clock(target)
        hz=self.clock.hz
        plan=dict(selection,physics_hz=hz,physics_dt_s=1/hz,substeps=hz//20,frame=self.frame,
                  switch_pending=hz!=target or rate_conflict,
                  clock_target_hz=target,clock_switch_pending=hz!=target,
                  refrigerator_rate=reservation,
                  refrigerator_rate_ready=bool(fridge_required and hz==60),
                  task_deferred_reason='refrigerator_requires_60hz' if rate_conflict else None)
        for key,port in self.ports.items():
            if hasattr(port,'begin_selected_control'):
                port.begin_selected_control(key==selected and not plan['switch_pending'] and not port.ended)
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
                if key is not None and not plan['switch_pending'] and not self.ports[key].ended:
                    self.dispatch.record_score(key,self.ports[key].observe())
                timing=self.dispatch.finish_frame()
            result=dict(plan=plan,outcome=timing,selected_control_steps={k:p.index for k,p in self.ports.items()})
            result['environment_helpers']=self.environment_helpers()
            self.trace.write(json.dumps(result,default=list)+'\n');self.trace.flush()
            for port in self.ports.values():
                if hasattr(port,'discard_control'):port.discard_control()
        except Exception:
            self.faulted=True;raise
        self.pending=None;self.frame+=1
        return result

    def close(self):
        if self.closed:return
        try:
            for p in self.ports.values():p.close()
        finally:
            try:
                if self.fridge_helper is not None:
                    self.fridge_helper.close()
                    (self.output/'environment_helpers_final.json').write_text(json.dumps(self.environment_helpers(),indent=2)+'\n')
            finally:
                self.trace.close();self.closed=True
