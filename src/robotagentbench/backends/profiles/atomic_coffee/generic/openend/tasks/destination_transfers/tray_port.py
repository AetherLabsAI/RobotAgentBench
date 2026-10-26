"""Scene-owned native tray reader; reuses common contact/holding/clock pipeline."""
from dataclasses import asdict
from types import SimpleNamespace
import json
import numpy as np
from openend.tasks.glasses_transfer.port import GlassesPort
from .registration import tray_parameters
from .scoring import TrayScore

class TrayPort(GlassesPort):
    asset='meal_tray_large'
    physics_hz=60
    supported_physics_hz=(60,240)
    validate_parameters=staticmethod(tray_parameters)
    def make_geometry(self,stage,manifest,parameters):
        from .geometry import TrayGeometry
        return TrayGeometry(stage,manifest)
    def reduce_contacts(self):
        result=super().reduce_contacts()
        # Only upward contacts on the measured tabletop mesh count as final
        # support; contact on a drawer, table leg or another object does not.
        roots=self.geometry.contact_roots;target=self.geometry.target
        from openend.tasks.glasses_transfer.contacts import under
        result['table_supported']=any(under(paths[i],target) and any(under(paths[1-i],p) for p in roots)
            and any((1 if i==0 else -1)*n[2]>.5 for n,_,_ in samples)
            for paths,samples in self.contacts.pairs() for i in (0,1))
        return result
    def accept_initial(self):
        from openend.tasks.meal_tray.nonmotion_closeout_v1.initial_windows import initial_window
        rows,complete=initial_window(self.history,self.config.initial_history_s,self.clock.get_physics_dt())
        checks=dict(history_complete=complete,supported=bool(rows) and all(bool(r['supports']) for r in rows),
            unheld=bool(rows) and not any(r['held'] or r['robot_contact'] or r['on_floor'] for r in rows),
            stable=bool(rows) and bool(np.ptp([r['pose'].position_m for r in rows],axis=0).max()<=self.config.initial_position_span_m),
            not_already_at_destination=not(self.latest['table_supported'] and self.geometry.insertion_evidence(self.latest['pose'])[0]))
        receipt=dict(accepted=all(checks.values()),checks=checks,evidence_source='native_physx',scene_pose_writes=0,
            empty_tray_allowed=True,preparation_dependency=False,full_physical_trajectory_verified=False)
        (self.output/'initial_acceptance.json').write_text(json.dumps(receipt,indent=2)+'\n')
        if not receipt['accepted']:raise RuntimeError('Tray initial rejected '+str(checks))
        self.initial_supports=frozenset(self.latest['supports'])
        self.initial_bottom=float(self.geometry.points(self.latest['pose'])[:,2].min())
        self.binding=SimpleNamespace(objects=dict(target=self.geometry.target,destination=self.geometry.destination),
            initial_supports=self.initial_supports)
        self.composite=TrayScore();self.score=self.composite.summary();self.index=0;self.pending=[]
        return receipt
    def consume_control(self,active):
        if not self.pending:raise RuntimeError('Missing tray native samples')
        rows,self.pending=self.pending,[]
        for i,r in enumerate(rows):
            self.index+=1
            f=dict(time_s=r['time_s'],held=r['held'],clear_original_support=not bool(r['supports']),
                raised=bool(self.geometry.points(r['pose'])[:,2].min()>self.initial_bottom+1e-9),
                inside=bool(self.geometry.insertion_evidence(r['pose'])[0]),supported=bool(r['table_supported']),
                on_floor=r['on_floor'],active=bool(active),final_sample=i==len(rows)-1)
            self.score=self.composite.observe(**f)
            self.trace.write(json.dumps(dict(index=self.index,facts=f,pose=asdict(r['pose']),score=self.score['score']))+'\n')
        self.trace.flush();return dict(self.score,evidence_status='native_physx')
    def pause(self):self.composite.pause()
    def finalize(self,reason):
        self.pending=[];self.score=self.composite.finalize(reason);return self.score
