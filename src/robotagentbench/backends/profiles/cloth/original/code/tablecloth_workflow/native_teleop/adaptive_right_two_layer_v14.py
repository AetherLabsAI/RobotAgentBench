"""Live corner readback retargets robot only, with right grasp and table guards."""
from adaptive_brush import AdaptiveBrush
from tablecloth_robot_kinematics import Robot
import json,sys,numpy as np
from pathlib import Path

class AdaptiveRightBasket(AdaptiveBrush):
    def __init__(self,action,*args,**kwargs):
        super().__init__(action,*args,**kwargs)
        from pxr import Gf,UsdGeom
        from scipy.spatial import ConvexHull
        self.cloth_ids=np.asarray(args[1]);uv=np.asarray(args[0]);target_uv=np.asarray(action.get('right_grasp_reference_uv',[0.,.51]));ids=np.flatnonzero(self.cloth_ids==0);self.right_corner=int(ids[np.argmin(np.linalg.norm(uv[ids]-target_uv,axis=1))]);self.right_layer_indices=[int(ids[np.argmin(np.linalg.norm(uv[ids]-x,axis=1))]) for x in action['right_grasp_layer_uv']];self.layer_masks=[uv[ids,1]<.255,uv[ids,1]>.255];self.phases=action['phase'].tolist();self.right_ref=np.asarray(action['right_grasp_reference_corner'],dtype=float)
        self.right_start=self.phases.index('R_edge_joint_pregrasp');self.right_pre_end=max(i for i,p in enumerate(self.phases) if p=='R_edge_joint_pregrasp')
        self.vertical_start=self.phases.index('R_vertical_clear_table');self.vertical_end=max(i for i,p in enumerate(self.phases) if p=='R_vertical_clear_table')
        self.lift_start=self.phases.index('R_lift_folded_cloth');self.lift_end=max(i for i,p in enumerate(self.phases) if p=='R_lift_folded_cloth')
        self.transfer_start=self.phases.index('R_transfer_over_original_basket');self.transfer_end=max(i for i,p in enumerate(self.phases) if p=='R_transfer_over_original_basket')
        self.action_targets=np.asarray(action['joint_target']);self.pre_correction=None;self.right_shift=None;self.right_prior=None;self.right_trace=[];self.right_arm=Robot.arm_names('R')
        self.robot.by_name['idx62_arm_r_joint2']['limits'][0]=-1.7
        self.robot.by_name['idx64_arm_r_joint4']['limits'][:]=[.25,2.5]
        stage=args[2];prim=stage.GetPrimAtPath('/World/Assets/CoffeeTable/Mesh');T=UsdGeom.XformCache().GetLocalToWorldTransform(prim)
        world=np.array([T.Transform(Gf.Vec3d(p)) for p in UsdGeom.Mesh(prim).GetPointsAttr().Get()]);self.table_planes=np.unique(ConvexHull(world).equations,axis=0);self.table_bounds=np.array([world.min(0),world.max(0)])
        self.basket_boxes=[]
        basket=stage.GetPrimAtPath('/World/Assets/TableclothBasket/PhysicalColliderCandidate')
        if not basket:raise RuntimeError('Expected current physical basket boxes for live collision audit')
        import itertools
        for child in basket.GetChildren():
            if not child.IsA(UsdGeom.Cube):continue
            size=float(UsdGeom.Cube(child).GetSizeAttr().Get());transform=UsdGeom.XformCache().GetLocalToWorldTransform(child)
            corners=np.array([transform.Transform(Gf.Vec3d(*(size*.5*np.array(v)))) for v in itertools.product([-1,1],repeat=3)])
            self.basket_boxes.append((str(child.GetPath()),corners.min(0),corners.max(0)))
        if len(self.basket_boxes)!=5:raise RuntimeError('Incomplete physical basket collision audit')
        cal=np.load('/home/aether/workspace/agent_bench/.codex-tmp/tablecloth-action-loop-20260905/robot_native_collision.npz')
        sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'protocol_live')]
        import audit_arm_posture
        from cad_body_guard import CadBodyGuard
        self.self_audit=audit_arm_posture.audit;self.cad_guard=CadBodyGuard(cal)
        self.right_collision_points={k[len('points_'):]:np.unique(cal[k],axis=0) for k in cal.files if k.startswith('points_gripper_r_')}
        for shape,points in zip(audit_arm_posture.shapes,audit_arm_posture.samples):
            if shape['link'].startswith('arm_r'):self.right_collision_points[shape['link']]=points
        from cad_surface_distance import CadSurfaceDistance
        self.pad_trees={side:CadSurfaceDistance(cal['points_gripper_r_'+side+'_link4'],cal['triangles_gripper_r_'+side+'_link4']) for side in ['inner','outer']}
        self.lost_frames=0
    def on_physics(self,source_frame,physical_frame,cloth,bodies,body_names):
        phase=self.phases[source_frame]
        if phase not in ['R_higher_clearance_after_outward_lift','R_vertical_clear_table','R_lift_folded_cloth','R_hold_lifted_cloth','R_transfer_over_original_basket','R_lower_into_basket']:return
        from scipy.spatial.transform import Rotation
        q=cloth[self.cloth_ids==0] if hasattr(self,'cloth_ids') else cloth[:len(cloth)//2]
        distances=[]
        for side in ['inner','outer']:
            b=bodies[body_names.index('gripper_r_'+side+'_link4')];R=Rotation.from_quat(b[3:]).as_matrix()
            distance,_=self.pad_trees[side].query((q-b[:3])@R);distances.append(distance)
        counts=[[int(np.sum((distance<.005)&mask)) for mask in self.layer_masks] for distance in distances]
        # Opposite pads must each have proximity to a distinct material half.
        two_layers=(counts[0][0]>0 and counts[1][1]>0) or (counts[0][1]>0 and counts[1][0]>0)
        self.lost_frames=self.lost_frames+1 if not two_layers else 0
        if physical_frame%20==0:self.right_trace.append(dict(event='right_two_layer_pad_proximity',physical_frame=physical_frame,pad_by_layer_counts=counts,two_layers=two_layers,distance_not_force_certificate=True))
        if self.lost_frames>=10:raise RuntimeError(f'Right grasp lacks opposite-pad proximity to both material layers for10frames at {physical_frame}; stop before transport')

    def on_frame(self,source_frame,physical_frame,cloth):
        following=super().on_frame(source_frame,physical_frame,cloth)
        if source_frame==self.right_start-1:
            self.right_shift=cloth[self.right_layer_indices].mean(0).astype(float)-self.right_ref
            if np.linalg.norm(self.right_shift)>.16:raise RuntimeError('Right corner moved beyond diagnostic retarget range')
            row=dict(event='right_corner_after_left_withdraw',physical_frame=physical_frame,corner_xyz_m=cloth[self.right_layer_indices].mean(0).tolist(),layer_xyz_m=cloth[self.right_layer_indices].tolist(),shift_m=self.right_shift.tolist());self.right_trace.append(row);print(json.dumps(row),flush=True)
        if self.phases[source_frame]=='R_edge_physical_hold' and self.phases[source_frame+1]!='R_edge_physical_hold':
            self.before_lift=cloth[self.right_layer_indices].copy()
        if self.phases[source_frame]=='R_hold_lifted_cloth' and self.phases[source_frame+1]!='R_hold_lifted_cloth':
            lifts=cloth[self.right_layer_indices,2]-self.before_lift[:,2];lift=float(lifts.min());self.right_trace.append(dict(event='right_retention_gate',min_layer_lift_m=lift,layer_lift_m=lifts.tolist(),physical_frame=physical_frame))
            if lift<.06:
                self.failure=f'Right physical grasp did not lift cloth at least6cm (measured {lift:.4f}m); basket transport stopped'
                return None
        return following
    def command(self,source_frame,physical_frame,template):
        result=super().command(source_frame,physical_frame,template)
        if source_frame<self.right_start or self.right_shift is None:return result
        values=dict(zip(self.names,map(float,result)))
        in_pre=source_frame<=self.right_pre_end
        if in_pre:
            if self.pre_correction is None:
                end=dict(zip(self.names,map(float,self.action_targets[self.right_pre_end])))
                H=self.robot.fk(end)['gripper_r_base_link']
                fitted,err=self.robot.solve(end,'R',H[:3,3]+self.right_shift,np.zeros(3),H[:3,:3],max_steps=600)
                from scipy.spatial.transform import Rotation
                from scipy.optimize import least_squares
                bounds=np.array([self.robot.by_name[name]['limits'] for name in self.right_arm]);bounds[self.right_arm.index('idx64_arm_r_joint4'),0]=.25
                def endpoint_residual(x):
                    T=self.robot.fk(end|dict(zip(self.right_arm,x)))['gripper_r_base_link']
                    return np.r_[T[:3,3]-H[:3,3]-self.right_shift,.065*Rotation.from_matrix(H[:3,:3]@T[:3,:3].T).as_rotvec()]
                fit=least_squares(endpoint_residual,np.clip([fitted[name] for name in self.right_arm],bounds[:,0]+1e-8,bounds[:,1]-1e-8),bounds=(bounds[:,0],bounds[:,1]),max_nfev=1000,gtol=1e-11,ftol=1e-11,xtol=1e-11)
                fitted.update(zip(self.right_arm,fit.x.tolist()))
                F=self.robot.fk(fitted)['gripper_r_base_link'];err=float(np.linalg.norm(F[:3,3]-H[:3,3]-self.right_shift));angle=np.degrees(Rotation.from_matrix(H[:3,:3]@F[:3,:3].T).magnitude())
                if err>.002 or angle>2:raise RuntimeError(f'Pregrasp retarget endpoint failed {err=} {angle=}')
                self.pre_correction={name:fitted[name]-end[name] for name in self.right_arm}
                print(json.dumps(dict(event='right_joint_pregrasp_retarget',endpoint_error_m=err,endpoint_rotation_deg=angle,endpoint_arm={name:fitted[name] for name in self.right_arm})),flush=True)
            u=(source_frame-self.right_start+1)/(self.right_pre_end-self.right_start+1);blend=u*u*(3-2*u)
            for name in self.right_arm:values[name]+=blend*self.pre_correction[name]
        pose=self.robot.fk(values)['gripper_r_base_link']
        if self.right_prior is None:self.right_prior={n:values[n] for n in self.right_arm}
        u=np.clip(((source_frame-self.right_start)/(self.right_pre_end-self.right_start)-.5)/.5,0,1);blend=u*u*(3-2*u)
        shift=np.zeros(3) if in_pre else self.right_shift.copy()
        if source_frame>=self.vertical_start:
            u=np.clip((source_frame-self.vertical_start)/(self.vertical_end-self.vertical_start),0,1);shift[2]*=1-u*u*(3-2*u)
        if source_frame>=self.lift_start:
            u=np.clip((source_frame-self.lift_start)/(self.lift_end-self.lift_start),0,1);shift[:2]*=1-u*u*(3-2*u)
        seed=values.copy()
        if not in_pre:seed.update(self.right_prior)
        solved,error=self.robot.solve(seed,'R',pose[:3,3]+shift,np.zeros(3),pose[:3,:3],max_steps=500)
        for name in self.right_arm:solved[name]=self.right_prior[name]+float(np.clip(solved[name]-self.right_prior[name],-1.2/20,1.2/20))
        error=float(np.linalg.norm(self.robot.fk(solved)['gripper_r_base_link'][:3,3]-(pose[:3,3]+shift)))
        actual=self.robot.fk(solved);rotation=np.degrees(np.arccos(np.clip((np.trace(pose[:3,:3]@actual['gripper_r_base_link'][:3,:3].T)-1)/2,-1,1)))
        if error>.0015 or rotation>1.5:
            from scipy.optimize import least_squares
            from scipy.spatial.transform import Rotation
            bounds=np.array([self.robot.by_name[name]['limits'] for name in self.right_arm]);prior=np.array([self.right_prior[name] for name in self.right_arm]);lo=np.maximum(bounds[:,0]+1e-7,prior-1.2/20);hi=np.minimum(bounds[:,1]-1e-7,prior+1.2/20)
            target=pose[:3,3]+shift
            def residual(x):
                H=self.robot.fk(solved|dict(zip(self.right_arm,x)))['gripper_r_base_link']
                return np.r_[H[:3,3]-target,.065*Rotation.from_matrix(pose[:3,:3]@H[:3,:3].T).as_rotvec(),(x-prior)*1e-6]
            fit=least_squares(residual,np.clip([solved[name] for name in self.right_arm],lo+1e-9,hi-1e-9),bounds=(lo,hi),max_nfev=100,gtol=1e-10,xtol=1e-10,ftol=1e-10)
            solved.update(zip(self.right_arm,fit.x.tolist()));actual=self.robot.fk(solved);error=float(np.linalg.norm(actual['gripper_r_base_link'][:3,3]-target));rotation=float(np.degrees(Rotation.from_matrix(pose[:3,:3]@actual['gripper_r_base_link'][:3,:3].T).magnitude()))

        rate=max(abs(solved[n]-self.right_prior[n])*20 for n in self.right_arm)
        if error>.002 or rotation>2 or rate>1.5:raise RuntimeError(f'Right live IK failed {physical_frame=} {error=} {rotation=} {rate=}')
        hits=self.self_audit(actual)+self.cad_guard.check(actual)
        if hits:raise RuntimeError(f'Right live self collision guard at frame {physical_frame}: {hits[:3]}')
        for link,points in self.right_collision_points.items():
            H=actual[link];world=points@H[:3,:3].T+H[:3,3]
            for box,lo,hi in self.basket_boxes:
                if np.any(world.max(0)<=lo) or np.any(world.min(0)>=hi):continue
                if np.any(np.all((world>lo+.001)&(world<hi-.001),axis=1)):
                    raise RuntimeError(f'Right live basket collision guard: {link} vs {box} at frame {physical_frame}')
            if np.any(world.max(0)<=self.table_bounds[0]) or np.any(world.min(0)>=self.table_bounds[1]):continue
            signed=world@self.table_planes[:,:3].T+self.table_planes[:,3]
            if np.any(np.max(signed,axis=1)<-.001):raise RuntimeError(f'Right live table collision guard: {link} at frame {physical_frame}')
        for name in self.right_arm:result[self.names.index(name)]=solved[name]
        self.right_prior={n:solved[n] for n in self.right_arm}
        return result
    def report(self):
        return dict(super().report(),right_grasp_feedback=self.right_trace,right_table_guard='native CAD sample points against actual convex table hull,1mm interior rejection',right_basket_guard='all five existing physical collider boxes',right_feedback_changes_robot_only=True)
