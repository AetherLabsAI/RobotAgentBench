"""SUPERSEDED DRAFT: do not launch this dual-active room loop.

User changed the contract to a start-selected multi-task registry, one current
sector-selected task, and a shared room timeout. multi_task_dispatch_v1.py is
the new policy component; native scoring/clock integration is not yet complete.

Historical draft: scene-owned coffee + fryer registry and the 20 Hz room loop.

No initialization of task props, no HTTP registry changes, no action generation.
The host supplies measured robot localization and advances the returned substeps.
Sector wake enables a registered task after measured initial acceptance, once.
Leaving never pauses an unfinished task's original clock or rebuilds its state.
"""
from collections import deque
from pathlib import Path
import hashlib
import json
import math
from sector_activation_v1 import step as sector_step
from coffee_hotwake_v1 import CoffeeHotwakeCycle

TASKS = {'coffee': ('1-1', '1-2', '1-3', '1-4'), 'fryer': ('2.1|2.2', '2.3', '2.5')}


def activation_configs(manifest, calibration, sensor):
    configs = {}
    for key, asset in (('coffee', 'coffee_machine'), ('fryer', 'air_fryer')):
        rows = [o for o in manifest['objects'] if o['id'] == asset]
        if len(rows) != 1:
            raise ValueError('Exactly one registered object required: '+asset)
        c = dict(next(e['config'] for e in calibration['entries'] if e['id'] == key))
        lo, hi = rows[0]['visible_bounds_world_m']
        center = [(lo[k]+hi[k])/2 for k in (0, 1)]
        c.update(target=center, center=center, targetPolygon=[
            [lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]],
            wake=sensor['range'], wakeHalfAngle=sensor['halfAngle'])
        # Same existing navigation probe's calibrated exit/preload policy.
        c['exit'] = max(c['exit'], sensor['range']+.5)
        c['preload'] = max(c['preload'], c['exit']+.3)
        configs[key] = c
    return configs


class DualTaskCycle:
    """Lightweight orchestration; callbacks own real evidence and score gates."""
    def __init__(self, coffee, fryer, clock, configs):
        if set(configs) != set(TASKS):
            raise ValueError('Both distinct task bindings required')
        self.coffee, self.fryer, self.clock = coffee, fryer, clock
        self.configs = configs
        self.activation = {k: {'active': False} for k in TASKS}
        self.cycle = CoffeeHotwakeCycle(coffee, clock)
        self.closed = False

    def begin_control(self, robot, *, command_vw, base_velocity, flags=None):
        if self.closed or self.cycle.pending is not None or self.cycle.faulted:
            raise RuntimeError('Room loop closed or previous action incomplete')
        if len(robot['xy']) != 2 or not all(math.isfinite(x) for x in (*robot['xy'], robot['yaw'])):
            raise ValueError('Finite measured localization required')
        flags = flags or {}
        if set(flags)-set(TASKS): raise ValueError('Unknown task flags')
        activations = {k: sector_step(self.activation[k], self.configs[k], robot, flags.get(k, {}))
                       for k in TASKS}
        unfinished = self.fryer.consumer is not None and self.fryer.consumer.termination is None
        fryer_hz = 240 if activations['fryer']['active'] or unfinished else 60
        plan = self.cycle.begin_control(activations['coffee'], command_vw=command_vw,
                                       base_velocity=base_velocity, other_required_hz=fryer_hz)
        self.activation = activations
        plan.update(activations=activations, fryer_unfinished_pins_240=unfinished)
        return plan

    def finish_control(self, *, completed_substeps):
        return self.cycle.finish_control(completed_substeps=completed_substeps)


class RoomTaskRuntime:
    """Trusted in-process room registration; not a new public network task ID."""
    def __init__(self, stage, manifest, output, episode_id, *, fryer_variant, configs):
        from coffee_startup_v1 import CoffeeSceneStartup
        from evaluation.tasks.air_fryer.continuous_openended_v1.native import NativeAirFryer
        from evaluation.tasks.coffee_machine import isaac_evidence as old
        self.stage, self.configs = stage, configs
        self.output = Path(output)
        self.output.mkdir(exist_ok=False)
        self.coffee = CoffeeSceneStartup()
        self.coffee.prepare(stage, manifest, episode_id=episode_id, instance_id='coffee')
        self.fryer = NativeAirFryer(stage, manifest, self.output/'fryer', episode_id, 'fryer', fryer_variant)
        self.registry = dict(coffee=self.coffee.service.identity, fryer=self.fryer.binding)
        self.history = deque(maxlen=math.ceil(old.INITIAL_WINDOW_S/old.PHYSICS_DT-1e-9)+2)
        self.bag_geometry = old.BagGeometry(stage, self.coffee.service.roles['bag'])
        self.hashes = {l.realPath: hashlib.sha256(Path(l.realPath).read_bytes()).hexdigest()
                       for l in stage.GetUsedLayers() if l.realPath}
        self.consecutive_240 = 0
        self.initial = {}
        self.scores = {k: None for k in TASKS}
        self.starts = {k: 0 for k in TASKS}
        self.closed = False
        self.trace = (self.output/'controls.jsonl').open('x')
        self.coffee_trace = (self.output/'coffee_facts.jsonl').open('x')
        self.cycle = None

    def initialize(self, clock, simulation_view):
        if self.cycle is not None: raise RuntimeError('One native binding per room episode')
        self.clock = clock
        self.cycle = DualTaskCycle(self.coffee, self.fryer, clock, self.configs)
        self.coffee.hotwake = self.cycle.cycle
        self.coffee.on_physics_ready(clock, simulation_view=simulation_view)
        self.fryer.initialize(clock, simulation_view, allow_idle_60hz=True)

    def after_physics(self, helper_report):
        if self.closed or self.cycle is None or self.cycle.cycle.pending is None:
            raise RuntimeError('No active room control frame')
        self.coffee.after_physics(helper_report)
        self.fryer.after_physics(helper_report)
        if self.clock.hz == 240:
            self.consecutive_240 += 1
            s = self.coffee.service
            self.history.append(dict(poses=s.latest['sample'].poses,
                contacts=s.latest['contacts'], camera=list(self.coffee.source.last_camera_contacts),
                machine=s.window['robot_machine'], button=s.window['robot_button']))
        else:
            self.consecutive_240 = 0
            self.history.clear()

    def begin_control(self, robot, *, command_vw, base_velocity):
        # Bodies stay in the shared scene. Physics is never disabled on exit.
        # Scorers pin 240 while unfinished; held props also protect region wake.
        flags = {k: {} for k in TASKS}
        if self.coffee.service.latest:
            flags['coffee']['held'] = any(self.coffee.service.latest['grasped'].values())
        if self.fryer.latest:
            flags['fryer']['held'] = any(v['grasped'] for v in self.fryer.latest['cakes'].values())
            flags['fryer']['held'] |= any(self.fryer.latest['tray']['bilateral'].values())
        return self.cycle.begin_control(robot, command_vw=command_vw, base_velocity=base_velocity, flags=flags)

    def _accept_coffee(self):
        import numpy as np
        from evaluation.tasks.coffee_machine import isaac_evidence as old
        from evaluation.tasks.coffee_machine.api_runtime import INITIAL_TILT_DEG, INITIAL_BAG_TILT_DEG, BUTTON_RELEASED_M
        from evaluation.tasks.coffee_machine.facts import COUNTER
        s = self.coffee.service
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
                       scope='task-local native initial acceptance; not full-room collision certification')
        (self.output/'coffee_initial.json').write_text(json.dumps(receipt,indent=2)+'\n')
        if not receipt['accepted']: raise RuntimeError('Coffee initial rejected: '+str(checks))
        self.coffee.attach_score(accepted_binding=s.identity)
        self.initial['coffee'] = receipt

    def finish_control(self, *, completed_substeps):
        timing = self.cycle.finish_control(completed_substeps=completed_substeps)
        for key in TASKS:
            if self.scores[key] is None:
                if not self.cycle.activation[key]['active'] or self.clock.hz!=240 or self.consecutive_240<240:
                    continue
                if key == 'coffee':
                    self._accept_coffee()
                else:
                    self.scores[key] = self.fryer.accept_initial()
                    self.initial[key] = json.loads((self.output/'fryer/initial_acceptance.json').read_text())
                self.starts[key] += 1
                if key == 'fryer': continue  # accept_initial already scored frame zero
            if key == 'coffee' and not self.coffee.bridge.process.ended:
                bridge = self.coffee.bridge
                f = self.coffee.service.facts(bridge.index+1, bridge.process.phase)
                self.scores[key] = bridge.collect_and_score()
                self.coffee_trace.write(json.dumps(dict(facts=f.as_dict(), score=self.scores[key]),default=list)+'\n')
            elif key == 'fryer' and self.fryer.consumer.termination is None:
                self.scores[key] = self.fryer.score_control()
        result = dict(**timing, activations=self.cycle.activation, scores=self.scores, start_counts=self.starts.copy())
        self.trace.write(json.dumps(result,default=list)+'\n')
        self.trace.flush(); self.coffee_trace.flush()
        return result

    def close(self):
        if self.closed: return
        if self.cycle and self.cycle.cycle.pending is not None:
            self.cycle.cycle.faulted = True
        self.coffee.close(); self.fryer.close()
        self.trace.close(); self.coffee_trace.close()
        if self.cycle: self.cycle.closed = True
        self.closed = True
