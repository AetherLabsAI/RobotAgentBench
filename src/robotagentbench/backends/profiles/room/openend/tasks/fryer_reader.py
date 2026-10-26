"""Read-only selected-control extraction; native substep stream stays continuous.

NativeAirFryer consumer stays None: the room owns a separate persistent scorer.
No pretending that 60Hz idle ticks are 240Hz scoring ticks. Facts carry selected
control index plus actual native time/step in the trace. Original predicates
and initial checks below are ported without threshold changes.
"""
import json, math
import numpy as np
from pathlib import Path
import hashlib
from evaluation.tasks.air_fryer.continuous_openended_v1.native import NativeAirFryer
from evaluation.tasks.air_fryer.isaac_evidence import PHYSICS_DT,CONTROL_DT,ON_PLATE,serializable,CakeFacts,TrayFacts,Facts

class SelectedFryerReader(NativeAirFryer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        from openend.tasks.holding_follow_v1.detection import GraspDetector
        self.detector=GraspDetector((self.paths['target'],))

    def after_physics(self,helper_report):
        super().after_physics(helper_report)
        if self.start_step is None:
            from collections import deque
            if not hasattr(self,'initial_history'):
                self.initial_history=deque(maxlen=self.history.maxlen)
            self.initial_history.append(dict(self.latest,physics_dt_s=self.world.get_physics_dt()))

    def accept_initial(self):
        if self.consumer is not None or self.start_step is not None:
            raise RuntimeError('Initial acceptance once only')
        from evaluation.tasks.air_fryer.isaac_evidence import INITIAL_WINDOW_S
        from openend.tasks.meal_tray.nonmotion_closeout_v1.initial_windows import initial_window
        rows,complete=initial_window(getattr(self,'initial_history',()),INITIAL_WINDOW_S,self.world.get_physics_dt())
        p=self.paths; now=self.latest
        if not complete:raise RuntimeError('Need complete original initial duration at actual native Hz')
        span=lambda obj: max(math.dist(a['poses'][obj]['position_m'],b['poses'][obj]['position_m']) for a in rows for b in rows)
        checks=dict(history_complete=complete,
                    sources_unchanged=all(hashlib.sha256(Path(f).read_bytes()).hexdigest()==h for f,h in self.hashes.items()),
                    rest=max(span(o) for o in (*self.cakes,p['tray']))<=.005,
                    no_robot_contact=not any(r['cakes'][o]['robot'] for r in rows for o in self.cakes)
                        and not any(r['tray']['robot'] for r in rows),
                    no_ground=not any(r['cakes'][o]['ground'] for r in rows for o in self.cakes)
                        and not any(r['tray']['ground'] for r in rows),
                    closed=abs(now['tray']['opening_m'])<=.005,
                    basket_empty=not any(now['cakes'][o]['inside'] for o in self.cakes),
                    cakes_on_plate=all(ON_PLATE in now['cakes'][o]['supports'] and now['cakes'][o]['tilt_deg']<=5 for o in self.cakes))
        report=dict(checks=checks,accepted=all(checks.values()),latest=now,
                    scope='task-local measured initial state; not whole-room collision certification')
        (self.output/'initial_acceptance.json').write_text(json.dumps(serializable(report),indent=2)+'\n')
        if not report['accepted']: raise RuntimeError('Initial rejection: '+str(checks))
        self.begin_episode()
        return report


    def selected_facts(self, index):
        TARGET, OTHER, TRAY, BODY, JOINT, BASKET_FLOOR, PLATE, PLATE_SUPPORT, COUNTER, FLOOR, ROBOT = (self.paths[k] for k in ('target', 'other', 'tray', 'body', 'joint', 'basket_floor', 'plate', 'plate_support', 'counter', 'floor', 'robot'))
        """Facts of the control frame that ends at the latest substep."""
        if self.start_step is None:
            raise RuntimeError('Initial state not accepted')
        if type(index) is not int or index < 0: raise ValueError('Selected control index required')
        row, window = self.latest, self.window
        cakes = {}
        for obj in self.cakes:
            c, pose = row['cakes'][obj], row['poses'][obj]
            cakes[obj] = CakeFacts(
                position=tuple(pose['position_m']), quaternion=tuple(pose['quaternion_wxyz']), tilt_deg=c['tilt_deg'],
                pads_bilateral=bool(any(c['bilateral'].values()) and c['robot']), grasped=bool(c['grasped']),
                robot_contact=bool(c['robot']), supports=frozenset(c['supports']), ground_contact=bool(window['ground'][obj] or c['ground']),
                basket_contact=bool(window['basket'][obj] or c['basket']), inside_basket=bool(c['inside']), outside_basket=bool(c['outside']),
                displacement_m=float(np.linalg.norm(np.asarray(pose['position_m']) - self.initial_position[obj])))
        t, pose = row['tray'], row['poses'][TRAY]
        tray = TrayFacts(position=tuple(pose['position_m']), quaternion=tuple(pose['quaternion_wxyz']), tilt_deg=t['tilt_deg'], opening_m=float(t['opening_m']),
                         body_contact=bool(t['body']), counter_supported=bool(t['counter']), robot_contact=bool(t['robot']),
                         pads_bilateral=bool(any(t['bilateral'].values()) and t['robot']), ground_contact=bool(window['ground'][TRAY] or t['ground']),
                         joint_released=bool(t['joint_released']),
                         displacement_m=float(np.linalg.norm(np.asarray(pose['position_m']) - self.initial_position[TRAY])))
        facts = Facts(step=index, time_s=index * CONTROL_DT, target=cakes[TARGET], other=cakes[OTHER], tray=tray)
        self.reset_accumulators()
        self.frames.write(json.dumps(serializable(dict(facts.as_dict(), simulation_time_s=row['time_s'], reader_step=self.step)), allow_nan=False) + '\n')
        self.frames.flush()
        self.trace.flush()
        return facts
