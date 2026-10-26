"""Approved real-material patch retention; never reads action commands."""
from collections import deque
import math
import numpy as np
from contact_geometry import quaternion_matrix_xyzw


class PatchRetention:
    def __init__(self,mesh,*,window_s,position_tolerance_m,normal_tolerance_deg):
        if min(window_s,position_tolerance_m,normal_tolerance_deg)<=0:raise ValueError('Explicit source detection settings required')
        self.mesh,self.window,self.position_tol=mesh,window_s,position_tolerance_m
        self.cosine=math.cos(math.radians(normal_tolerance_deg))
        self.patch=self.history=self.last_time=None

    def observe(self,*,time_s,points,hand_transform_xyzw,bilateral_node_mask):
        if self.last_time is not None and time_s<=self.last_time:raise ValueError('Stale native retention sample')
        self.last_time=time_s;q=self.mesh.positions(points);mask=np.asarray(bilateral_node_mask)
        if mask.dtype!=bool or mask.shape!=(len(q),):raise ValueError('Measured bilateral material mask required')
        pose=np.asarray(hand_transform_xyzw,float)
        if pose.shape!=(7,):raise ValueError('Native hand pose required')
        r=quaternion_matrix_xyzw(pose[3:])
        if self.patch is not None and not mask[self.patch].all():self.patch=self.history=None
        if self.patch is None:
            # A triangle is the smallest material patch with measurable normal.
            candidates=np.flatnonzero(mask[self.mesh.faces].all(1))
            if not len(candidates):return dict(held=False,patch_nodes=[],relative_window_s=0.,force_measured=False)
            self.patch=self.mesh.faces[candidates[0]].copy();self.history=deque()
        patch=q[self.patch];cross=np.cross(patch[1]-patch[0],patch[2]-patch[0]);length=np.linalg.norm(cross)
        if length==0:
            self.patch=self.history=None
            return dict(held=False,patch_nodes=[],relative_window_s=0.,force_measured=False)
        local=(patch-pose[:3])@r;normal=(cross/length)@r
        self.history.append((time_s,local.copy(),normal.copy()))
        while len(self.history)>1 and self.history[1][0]<=time_s-self.window+1e-9:self.history.popleft()
        first=self.history[0];span=time_s-first[0]
        follows=all(np.linalg.norm(row[1]-first[1],axis=1).max()<=self.position_tol and row[2]@first[2]>=self.cosine for row in self.history)
        if not follows:
            self.history=deque([(time_s,local.copy(),normal.copy())]);span=0.
        return dict(held=bool(follows and span+1e-9>=self.window),patch_nodes=self.patch.tolist(),
            relative_window_s=span,force_measured=False,method='same_live_material_triangle_in_measured_hand_frame')
