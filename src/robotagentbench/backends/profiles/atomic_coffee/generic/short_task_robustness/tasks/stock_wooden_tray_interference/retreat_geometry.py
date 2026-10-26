"""Conservative CPU retreat selection for a held-ready diagnostic robot.

Uses complete open-hand bounds and observed robot/scene geometry, not a
wrist-point test. No physics/score thresholds or collider changes.
"""
import numpy as np


def choose_overhead_entry(asset,base_orientation,grip_center,target_box,stock_bounds,ceiling_z,
                          hand_contact_offset=0.,stock_contact_offsets=None,ceiling_contact_offset=0.):
    """Descend inside this shelf compartment, never through the upper board.

    This changes the approach direction, not the common wrap pose or any
    object's physics. Candidate height comes from the measured free space.
    """
    box=np.asarray(target_box);center=box.mean(0);p,_=asset.sample(0.)
    if not np.isfinite(ceiling_z) or ceiling_z<=box[1,2]:raise ValueError('No overhead compartment clearance')
    groups=list(asset.collision_groups().values());stock=np.asarray(stock_bounds,dtype=float)
    offsets=np.asarray(stock_contact_offsets if stock_contact_offsets is not None else [0.]*len(stock))
    if offsets.shape!=(len(stock),) or not np.isfinite(offsets).all() or np.any(offsets<0):
        raise ValueError('Resolved stock contact offsets required')
    if not np.isfinite([hand_contact_offset,ceiling_contact_offset]).all() or min(hand_contact_offset,ceiling_contact_offset)<0:
        raise ValueError('Resolved hand/ceiling contact offsets required')
    # Collision contacts start outside the visible mesh. Read those existing
    # envelopes rather than weakening them or accepting a skin-only gap.
    if len(stock):stock=stock+np.array([-1.,1.])[None,:,None]*(offsets+hand_contact_offset)[:,None,None]
    effective_ceiling=ceiling_z-hand_contact_offset-ceiling_contact_offset
    def hit(shapes,obs):
        if not len(obs):return False
        return bool(np.any(np.all(shapes[:,None,0]<obs[None,:,1],axis=-1)&
                           np.all(obs[None,:,0]<shapes[:,None,1],axis=-1)))
    rejected=[]
    for angle in dict.fromkeys((65,90,115,-65,-90,-115,0,180,*range(-180,180,15))):
        a=np.deg2rad(angle)
        rz=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
        orient=rz@base_orientation;grip=center-orient@grip_center
        points=p@orient.T;shape=np.array([[points[g].min(0),points[g].max(0)] for g in groups])
        free=float(effective_ceiling-(points+grip)[:,2].max())
        if free<=0:continue
        for fraction in (.75,.5,.25):
            height=free*fraction;entry=grip+np.array([0.,0.,height])
            reason=None
            if hit(shape+entry,box[None]):reason='initial_target_overlap'
            elif any(hit(shape+entry+t*(grip-entry),stock) for t in np.linspace(0,1,49)):
                reason='stock_blocks_descent'
            if reason:
                rejected.append(dict(angle_deg=angle,height_m=height,reason=reason));continue
            return orient,grip,entry,dict(angle_deg=angle,height_m=height,ceiling_z=float(ceiling_z),
                hand_top_at_entry=float((points+entry)[:,2].max()),rejected=rejected,
                hand_contact_offset_m=float(hand_contact_offset),stock_contact_offsets_m=offsets.tolist(),
                ceiling_contact_offset_m=float(ceiling_contact_offset),
                method='vertical_descent_within_shelf_compartment',native_validation_required=True)
    raise ValueError('No geometry-clear overhead descent: '+str(rejected))


def choose_side_entry(asset,base_orientation,grip_center,target_box,stock_bounds,*,rear_short=False,approach_points=None):
    """Choose a clear lateral entry from scene geometry, not product identity."""
    box=np.asarray(target_box);center=box.mean(0);p,_=asset.sample(0.)
    groups=list(asset.collision_groups().values());stock=np.asarray(stock_bounds)
    def hit(shapes,obs):
        if not len(obs):return False
        return bool(np.any(np.all(shapes[:,None,0]<obs[None,:,1],axis=-1)&
                           np.all(obs[None,:,0]<shapes[:,None,1],axis=-1)))
    rejected=[];candidates=[]
    for angle in ((180,165,-165,150,-150,135,-135) if rear_short else (65,90,115,-65,-90,-115)):
        a=np.deg2rad(angle)
        rz=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
        orient=rz@base_orientation;grip=center-orient@grip_center
        points=p@orient.T;shape=np.array([[points[g].min(0),points[g].max(0)] for g in groups])
        lengths=(2*max(box[1,:2]-box[0,:2]),) if rear_short else (2*(box[1,2]-box[0,2]),2*max(box[1,:2]-box[0,:2]))
        for length in dict.fromkeys(lengths):
            entry=grip+length*(orient@np.array([0.,0.,-1.]))
            reason=None
            if hit(shape+entry,box[None]):reason='initial_target_overlap'
            elif any(hit(shape+entry+t*(grip-entry),stock) for t in np.linspace(0,1,49)):
                reason='stock_blocks_entry'
            if reason:
                rejected.append(dict(angle_deg=angle,length_m=float(length),reason=reason));continue
            info=dict(angle_deg=angle,length_m=float(length),rejected=rejected,
                method='rear_short_entry' if rear_short else 'sampled_open_hand_anatomical_bounds_lateral_entry',native_validation_required=True)
            if rear_short and approach_points is not None:
                from .recorded_clearance import entry_ranking_gap
                gap=entry_ranking_gap(asset,orient,entry,grip,box,approach_points)
                candidates.append((gap,orient,grip,entry,info))
            else:return orient,grip,entry,info
    if candidates:
        best=max(candidates,key=lambda row:row[0])
        best[4]['recorded_gripper_candidate_gaps_m']={str(c[4]['angle_deg']):c[0] for c in candidates}
        best[4]['ranking_only_not_collision_certificate']=True
        return best[1:]
    raise ValueError('No clear lateral hand entry around existing stock: '+str(rejected))


def separation(a,b):
    delta=np.maximum(np.maximum(b[0]-a[1],a[0]-b[1]),0.)
    if np.any(delta):return float(np.linalg.norm(delta))
    return -float(np.min(np.minimum(a[1],b[1])-np.maximum(a[0],b[0])))


def check_transport(motion,asset,orientation,robot_bounds,stock_bounds,target_box):
    """Sample approach/carry/release using the new actor's anatomical bounds.

    Read-only conservative geometry, not a native physics certificate. The
    shelf itself is guarded separately by native contact evidence.
    """
    robot=np.asarray(robot_bounds);stock=np.asarray(stock_bounds)
    if not len(robot):raise ValueError('Robot geometry missing')
    groups=list(asset.collision_groups().values())
    original_grip=np.array(motion.points[1])
    def intersects(shapes,obs):
        if not len(obs):return False
        return bool(np.any(np.all(shapes[:,None,0]<obs[None,:,1],axis=-1)&
                           np.all(obs[None,:,0]<shapes[:,None,1],axis=-1)))
    for t in np.linspace(0,2.4,49):
        s=motion.sample(float(t));p,_=asset.sample(s.closure)
        p=p@orientation.T+np.array(s.wrist_position)
        bounds=np.array([[p[g].min(0),p[g].max(0)] for g in groups])
        if intersects(bounds,robot):return False,dict(phase=s.phase,time_s=float(t),obstacle='robot')
        if intersects(bounds,stock):return False,dict(phase=s.phase,time_s=float(t),obstacle='stock')
        if s.phase in ('carry','release'):
            carried=np.asarray(target_box)+(np.array(s.wrist_position)-original_grip)
            if intersects(carried[None],stock) or intersects(carried[None],robot):
                return False,dict(phase=s.phase,time_s=float(t),obstacle='carried_drink')
    return True,dict(sample_count=49,native_validation_required=True)


def choose_retreat(start,open_points_world_offset,robot_bounds,scene_bounds,length,hand_groups=None,target_bounds=None,*,rear_short=False):
    start=np.asarray(start);points=np.asarray(open_points_world_offset)
    ids=hand_groups or [list(range(len(points)))]
    shapes=np.array([[points[g].min(0),points[g].max(0)] for g in ids])
    if not robot_bounds:raise ValueError('Robot geometry missing; retreat not admitted')
    def gaps(hand,obstacles):
        if not len(obstacles):return float('inf')
        obs=np.asarray(obstacles)
        a,b=hand[:,None,:,:],obs[None,:,:,:]
        delta=np.maximum(np.maximum(b[:,:,0]-a[:,:,1],a[:,:,0]-b[:,:,1]),0)
        distance=np.linalg.norm(delta,axis=-1)
        depth=np.min(np.minimum(a[:,:,1],b[:,:,1])-np.maximum(a[:,:,0],b[:,:,0]),axis=-1)
        # Zero-thickness triangles must still reject a crossing, rather than
        # turning it into signed zero and treating the surface as empty.
        return float(np.where(distance>0,distance,-np.maximum(depth,1e-12)).min())
    def inspect(end):
        robot_gap=float('inf');scene_gap=float('inf')
        scene=np.asarray(scene_bounds)
        if len(scene)>128:
            sweep_lo=shapes[:,0].min(0)+np.minimum(start,end)
            sweep_hi=shapes[:,1].max(0)+np.maximum(start,end)
            scene=scene[np.all(scene[:,1]>=sweep_lo,axis=1)&np.all(scene[:,0]<=sweep_hi,axis=1)]
        for t in np.linspace(0,1,49):
            b=shapes+start+(np.asarray(end)-start)*t
            robot_gap=min(robot_gap,gaps(b,robot_bounds))
            scene_gap=min(scene_gap,gaps(b,scene))
            if robot_gap<=0 or scene_gap<0:break
        final_gap=gaps(shapes+end,robot_bounds)
        return dict(robot_min_bbox_gap_m=robot_gap,scene_min_bbox_gap_m=scene_gap,
                    robot_final_bbox_gap_m=final_gap)
    candidates=[]
    # Keep exiting to the right, but avoid the front-diagonal path toward
    # the right gripper. Angles are actor route candidates, not physics knobs.
    directions=([dx,dy,0.] for dx,dy in
        ([(0.,1.),(-.25,1.),(.25,1.),(-.5,1.),(.5,1.),(-1.,.5),(1.,.5)] if rear_short
         else [(1.,dy) for dy in [0.,.25,.5,-.25,-.5,.75,-.75]]))
    for direction_row in directions:
        for scale in [1.,.75,.5]:
            dz=0.
            direction=np.array(direction_row);direction/=np.linalg.norm(direction)
            end=start+length*scale*direction;info=inspect(end)
            if target_bounds is not None:
                info['target_final_bbox_gap_m']=gaps(shapes+end,[target_bounds])
                if info['target_final_bbox_gap_m']<=0:continue
            if info['robot_min_bbox_gap_m']>0 and info['scene_min_bbox_gap_m']>=0:
                candidates.append((min(info['robot_min_bbox_gap_m'],info['robot_final_bbox_gap_m'],info['scene_min_bbox_gap_m']),
                                   info['robot_final_bbox_gap_m'],-abs(dz),end,info))
    if not candidates:raise ValueError('No geometry-clear side retreat; do not run through robot/stock')
    best=max(candidates,key=lambda c:c[:3])
    return best[3],dict(best[4],candidate_count=len(candidates),
                      method='sampled_anatomical_hand_aabbs_vs_robot_visual_and_stock_aabbs',
                      moving_robot_verified=False)
