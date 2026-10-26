"""Approved post-failure base search inside the EXISTING calibrated sector.

Reuses target_spawn's original grid/radius/sector and the original floor and
mesh collision predicates. No new stand-off range, yaw range or scene edit.
"""
import math
import numpy as np


def with_fixed_heading(rows, yaw):
    if not math.isfinite(yaw):raise ValueError('Finite original task heading required')
    result=[]
    for index,(distance,xy,_look_at_yaw) in enumerate(rows):
        xy=np.asarray(xy,dtype=float)
        if xy.shape!=(2,) or not np.isfinite(xy).all() or not math.isfinite(distance) or distance<=0:
            raise ValueError('Existing finite operating-sector candidate required')
        result.append(dict(candidate_index=index,initial_base_xyyaw=[*map(float,xy),float(yaw)],
            distance_to_target_m=float(distance),
            selection_domain='existing_calibrated_task_front_sector_and_room_floor',
            heading='unchanged_original_task_heading'))
    return result


def with_source_heading(rows):
    result=[]
    for index,(distance,xy,yaw) in enumerate(rows):
        entry=with_fixed_heading([(distance,xy,yaw)],float(yaw))[0]
        entry.update(candidate_index=index,heading='unchanged_existing_task_front_look_at_rule')
        result.append(entry)
    return result


def proposals(stage, manifest, task, launch):
    from short_tasks.agent_probe_v1 import target_spawn
    target=target_spawn.target_geometry(stage,manifest,task)
    sensor=launch['sensor']
    polygons=manifest.get('overview',{}).get('floors',[])
    if not polygons:raise ValueError('Original measured legal floor required')
    raw=target_spawn.candidates(target['target_xy'],target['front'],target['front_origin'],
        polygons,float(sensor['range']),float(sensor['halfAngle']))
    receipt=dict(policy='post_failure_operating_station.v1',
        human_authorized=True,opening_station_rectangle_replaced=True,
        existing_operating_sector_unchanged=True,floor_domain_unchanged=True,
        target_geometry=target,sensor=dict(sensor),
        hand_orientation_reference_base_yaw_rad=float(launch['initial_base_xyyaw'][2]),
        base_heading_policy='unchanged_existing_task_front_look_at_rule',
        fixed_hand_world_orientation_preserved=True,
        candidate_count=len(raw),physical_validation=False)
    return with_source_heading(raw),receipt


class RoomGate:
    """Cache static geometry once, preserving the full source checker tests.

Only base pose and the active arm differ per proposal. A preflight can omit
that active arm, but final acceptance always checks every robot collider.
The original full checker also independently rechecks any winning candidate.
"""
    def __init__(self, stage, manifest):
        from pxr import Usd,UsdGeom,UsdPhysics
        from short_tasks.tasks.retrieve_from_drawer.prepare_initial import corners
        from openend.tasks.scene_requirements.source_geometry import floor_paths
        from openend.tasks.scene_requirements.station_scene_checks import intersects,triangles_hit_box
        from openend.tasks.layout_repair_v1.spawn_v2 import inside
        self.intersects,self.triangles_hit_box,self.inside=intersects,triangles_hit_box,inside
        floors=floor_paths(stage)
        self.polygons=manifest.get('overview',{}).get('floors',[])
        if not floors or not self.polygons:raise ValueError('Original physical floor and legal polygons required')
        cache=UsdGeom.XformCache();bbox=UsdGeom.BBoxCache(0,['default','render','proxy','guide'],False,True)
        self.robot=[];self.obstacles=[]
        for p in Usd.PrimRange(stage.GetPseudoRoot(),Usd.TraverseInstanceProxies()):
            if not p.IsActive() or not p.HasAPI(UsdPhysics.CollisionAPI) or p.GetAttribute('physics:collisionEnabled').Get() is False:
                continue
            path=str(p.GetPath())
            r=bbox.ComputeWorldBound(p).ComputeAlignedRange();box=np.array([r.GetMin(),r.GetMax()])
            if not np.isfinite(box).all() or np.any(box[0]>box[1]):raise ValueError('Invalid original collider: '+path)
            if path.startswith('/World/Robot/'):
                body=p
                while body and not body.IsPseudoRoot() and not body.HasAPI(UsdPhysics.RigidBodyAPI):body=body.GetParent()
                if not body or body.IsPseudoRoot():raise ValueError('Robot collider has no body')
                r=bbox.ComputeRelativeBound(p,body).ComputeAlignedRange()
                local_box=np.array([r.GetMin(),r.GetMax()])
                if not np.isfinite(local_box).all() or np.any(local_box[0]>local_box[1]):
                    raise ValueError('Invalid body-local original collider: '+path)
                self.robot.append((path,str(body.GetPath()),np.c_[corners(local_box),np.ones(8)]))
                continue
            if any(path==f or path.startswith(f+'/') for f in floors):continue
            triangles=None
            if p.IsA(UsdGeom.Mesh):
                vertices=p.GetAttribute('points').Get();ids=list(p.GetAttribute('faceVertexIndices').Get() or [])
                faces=[];offset=0
                for n in p.GetAttribute('faceVertexCounts').Get() or []:
                    face=ids[offset:offset+n];offset+=n
                    faces.extend((face[0],face[i],face[i+1]) for i in range(1,n-1))
                if vertices is not None and len(vertices) and faces:
                    v=(np.c_[np.array(vertices),np.ones(len(vertices))]@np.array(cache.GetLocalToWorldTransform(p)))[:,:3]
                    triangles=v[np.array(faces)]
            self.obstacles.append((path,box,triangles))
        if not self.robot:raise ValueError('Missing original robot collider inventory')

    def check(self, model, q, *, omit_active_arm=False):
        links=model.frames(q)
        for path,body,local in self.robot:
            if omit_active_arm and (body.startswith('/World/Robot/arm_'+model.hand+'_') or
                                    body.startswith('/World/Robot/gripper_'+model.hand+'_')):
                continue
            v=(local@links[body].T)[:,:3];box=np.array([v.min(0),v.max(0)])
            perimeter=[(a,b) for a in np.linspace(box[0,0],box[1,0],11) for b in (box[0,1],box[1,1])]
            perimeter += [(a,b) for b in np.linspace(box[0,1],box[1,1],11) for a in (box[0,0],box[1,0])]
            if not all(any(self.inside(v,p) for p in self.polygons) for v in perimeter):
                return 'robot_outside_floor:'+path
            hit=next((p for p,b,t in self.obstacles if self.intersects(box,b) and
                (t is None or self.triangles_hit_box(t,box))),None)
            if hit:return 'robot_room_overlap:'+path+':'+hit
        return None
