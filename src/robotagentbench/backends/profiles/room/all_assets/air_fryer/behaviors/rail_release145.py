"""User-authorized, explicitly simplified stroke-triggered guide release.

Common mechanism for insertion and removal. Reads LIVE body/tray poses. The
only state write is jointEnabled=False once stroke >=145mm. Does not change
object pose/velocity, collision, mass, friction, robot command or success.
Call restore_before_reset() BEFORE world.reset(), then bind_closed_zero().
No relatching during an episode; reset is the only way to restore guidance.
"""
import numpy as np
from scipy.spatial.transform import Rotation

def matrix(view):
    p,q=view.get_world_poses()
    p=np.asarray(p[0],float);q=np.asarray(q[0],float)
    assert np.isfinite(p).all() and np.isfinite(q).all()
    t=np.eye(4);t[:3,3]=p;t[:3,:3]=Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
    return t

class RailRelease145:
    def __init__(self,stage,fryer_path,tray,body):
        self.stage=stage;self.path=fryer_path;self.tray=tray;self.body=body
        prim=stage.GetPrimAtPath(fryer_path)
        self.threshold=float(prim.GetAttribute('arena:railReleaseStrokeM').Get())
        assert abs(self.threshold-.145)<1e-8
        joint=stage.GetPrimAtPath(fryer_path+'/Centered/Joints/DrawerSlideOldParams')
        assert joint and joint.IsActive()
        self.attr=joint.GetAttribute('physics:jointEnabled')
        assert self.attr and abs(joint.GetAttribute('limit:transX:physics:high').Get()-.146)<1e-7
        self.released=False;self.origin=None;self.events=[]

    def restore_before_reset(self):
        from pxr import Usd
        with Usd.EditContext(self.stage,self.stage.GetSessionLayer()):self.attr.Set(True)
        self.released=False;self.origin=None;self.events=[]

    def bind_closed_zero(self):
        assert self.attr.Get() is True
        self.origin=(np.linalg.inv(matrix(self.body))@matrix(self.tray))[:3,3].copy()

    def step(self,physics_step,sim_time):
        assert self.origin is not None
        before=matrix(self.tray);relative=np.linalg.inv(matrix(self.body))@before
        stroke=float(relative[0,3]-self.origin[0])
        if not self.released and stroke>=self.threshold:
            from pxr import Usd
            velocity=np.asarray(self.tray.get_velocities()[0],float).copy()
            with Usd.EditContext(self.stage,self.stage.GetSessionLayer()):self.attr.Set(False)
            after=matrix(self.tray);after_v=np.asarray(self.tray.get_velocities()[0],float)
            assert np.array_equal(before,after) and np.array_equal(velocity,after_v), 'Unexpected direct state change'
            self.released=True
            self.events.append(dict(physics_step=int(physics_step),sim_time=float(sim_time),
                measured_stroke_m=stroke,threshold_m=self.threshold,
                direct_pose_delta_m=float(np.linalg.norm(after[:3,3]-before[:3,3])),
                direct_velocity_delta=float(np.max(abs(after_v-velocity))),
                pose_before=before.tolist(),velocity_before=velocity.tolist(),
                mechanism='stroke-triggered joint release; not geometry-only detachment'))
        return stroke
