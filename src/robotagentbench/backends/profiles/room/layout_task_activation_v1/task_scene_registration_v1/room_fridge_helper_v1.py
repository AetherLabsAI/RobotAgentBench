"""Room-owned original L1 assistance, separate from task selection and scoring.

No new door parameters, actions, pose writes, or inferred task-phase triggers.
The original initializer is additive, so a stage session must never run it twice.
"""
from copy import deepcopy

FRIDGE='/World/Background/Shell/RefrigeratorRecovered'
MARKER='openendFridgeHelperInitialization'


class RoomFridgeHelper:
    @classmethod
    def prepare(cls,stage,manifest):
        rows=[o for o in manifest.get('objects',[]) if o.get('id')=='refrigerator']
        prim=stage.GetPrimAtPath(FRIDGE)
        present=bool(prim and prim.IsActive())
        if not rows and not present:return None
        if len(rows)!=1 or not present or rows[0].get('path')!=FRIDGE:
            raise ValueError('Room refrigerator manifest and actual instance must match')
        if rows[0].get('helper_runtime')!='arena_api.l1_fridge.Controller':
            raise ValueError('Expected approved L1 refrigerator assistance')
        if prim.GetCustomDataByKey('roomAssetId')!='refrigerator':
            raise ValueError('Expected packaged room refrigerator instance')
        hz=stage.GetPrimAtPath('/World/PhysicsScene').GetAttribute('physxScene:timeStepsPerSecond').Get()
        if hz not in (60,240):raise ValueError('Existing room 60/240 Hz clock required')
        session=stage.GetSessionLayer()
        data=dict(session.customLayerData)
        if MARKER in data:raise RuntimeError('Fridge initializer already used; load a fresh scene session')
        # Mark before construction: partial initialization must not be retried on the same stage.
        data[MARKER]='initializing';session.customLayerData=data
        from pxr import Usd
        from arena_api.l1_fridge import Controller
        try:
            with Usd.EditContext(stage,session):controller=Controller(stage,physics_dt=1/hz)
        except Exception:
            data[MARKER]='failed_requires_fresh_session';session.customLayerData=data
            raise
        data[MARKER]='prepared';session.customLayerData=data
        return cls(controller,hz,stage)

    def __init__(self,controller,hz,stage):
        self.controller=controller;self.hz=hz;self.closed=False;self.substeps=0;self.stage=stage

    def validate_hz(self,hz):
        if self.closed:raise RuntimeError('Closed room refrigerator helper')
        if hz not in (60,240):raise ValueError('Unsupported room clock')
        self.controller.validate_physics_dt(1/hz)

    def set_hz(self,hz):
        self.validate_hz(hz)
        self.controller.set_physics_dt(1/hz);self.hz=hz

    def after_physics(self,hz):
        if self.closed or hz!=self.hz:raise RuntimeError('Room/helper clock mismatch')
        from pxr import Usd
        with Usd.EditContext(self.stage,self.stage.GetSessionLayer()):
            self.controller.after_physics_step()
        self.substeps+=1

    def rate_request(self,robot,*,radius,half_angle):
        """Approved fridge operation at 60 Hz; no scores or synthetic actions.

        Use the fixed cabinet's actual world bounds for the existing sector.
        Keep the reservation through the original helper's whole door cycle,
        even when the robot turns away. No inferred grasp/close-window signal.
        """
        from pxr import Usd,UsdGeom
        from sector_activation_v1 import sector_polygon
        if self.closed:raise RuntimeError('Closed room refrigerator helper')
        zone=self.controller.report()['state']['zone']
        if zone not in ('latch','opening','open','closing'):
            raise RuntimeError('Unknown original refrigerator state')
        cabinet=self.stage.GetPrimAtPath(FRIDGE+'/body')
        if not cabinet:cabinet=self.stage.GetPrimAtPath(FRIDGE)
        bounds=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render','proxy']).ComputeWorldBound(cabinet).ComputeAlignedRange()
        if bounds.IsEmpty():raise RuntimeError('Actual refrigerator activation geometry missing')
        lo,hi=bounds.GetMin(),bounds.GetMax()
        polygon=((lo[0],lo[1]),(hi[0],lo[1]),(hi[0],hi[1]),(lo[0],hi[1]))
        hit=sector_polygon(robot,radius,half_angle,polygon)
        active=hit or zone!='latch'
        return dict(required=active,physics_hz=60,sector_hit=hit,door_zone=zone,
                    reason='door_cycle' if zone!='latch' else ('sector_hit' if hit else 'inactive'),
                    target_polygon=[list(p) for p in polygon])

    def report(self):
        return dict(owner='unified_openend_room',physics_hz=self.hz,substeps=self.substeps,
                    interaction_physics_hz=60,control_hz=20,
                    helper_ticks=self.controller.steps//self.controller.period,
                    closed=self.closed,scoring_connected=False,
                    phase_window_source='none_no_inferred_retract_close_phase',
                    assistance=deepcopy(self.controller.report()))

    def close(self):
        if self.closed:return
        from pxr import Usd
        with Usd.EditContext(self.stage,self.stage.GetSessionLayer()):self.controller.close()
        self.closed=True
