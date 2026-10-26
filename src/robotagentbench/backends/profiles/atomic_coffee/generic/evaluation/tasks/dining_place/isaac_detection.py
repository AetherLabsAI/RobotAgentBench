"""Task-local live PhysX reader -> measured facts -> 10.2 three-point score.

Create before reset, initialize after reset, read after every physics step.
Only contact-report instrumentation is authored, in the session layer. No
kinematic/pose/drive/material/collision flags or target geometry are modified.
"""
from dataclasses import asdict
from dataclasses import replace
from collections import deque
import hashlib
import json
import os
import warnings
from pathlib import Path
import numpy as np

from arena_api.contact_identity import ContactIdentityCache
from .physical_detection import DetectionConfig, PhysicalDetector, PhysicalFacts
from .initial_state import DESTINATION, ROBOT
from .variants import payload_for
from arena_api.physics_defaults import steps_for_seconds

PLATE = '/World/Assets/MicrowaveTakePlate/Asset'
PAYLOAD = '/World/Assets/MicrowaveBeefbox'
PADS = {h: tuple(ROBOT+f'/gripper_{h}_{s}_link4/' for s in ('inner','outer')) for h in ('l','r')}


def under(path, root):
    return path == root or path.startswith(root.rstrip('/')+'/')


def rotation(q):
    from scipy.spatial.transform import Rotation
    q = np.asarray(q,dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1)>1e-4:
        raise ValueError('Invalid measured orientation')
    return Rotation.from_quat([*q[1:],q[0]])


class ContactStream:
    """Reduce relevant contact points immediately; never build raw point dicts.

    Absent reports may reuse an observed pair only if BOTH actors are sleeping
    or static. An awake missing pair fails closed, never invents no-contact.
    """
    def __init__(self, resolve, sleeping, event_types, table_top_m, face_tolerance_m=.003, *, payload_variant='beef_box'):
        self.payload=payload_for(payload_variant)
        self.resolve,self.sleeping = resolve,sleeping
        self.found,self.persist,self.lost = event_types
        self.top,self.tolerance = table_top_m,face_tolerance_m
        self.identity = ContactIdentityCache(resolve,self.classify)
        self.active = {}
        self.unknown_pairs=set()
        self.unresolved_report_gap=False
        self.gap_steps=self.gap_events=0

    def _unknown(self,key):
        if key not in self.unknown_pairs:
            self.gap_events+=1
            warnings.warn('Placement contact evidence gap; continuing without '
                          'stale grasp/support evidence: '+str(key),RuntimeWarning)
        self.unknown_pairs.add(key)
        if key in self.active:
            paths,actors,stage,*_=self.active[key]
            self.active[key]=(paths,actors,stage,np.zeros(3),(),False,False,False,False)

    def evidence_status(self):
        return dict(complete=not self.unresolved_report_gap and not self.unknown_pairs,
            unknown_pairs=[list(k) for k in sorted(self.unknown_pairs)],
            unresolved_report_gap=self.unresolved_report_gap,gap_steps=self.gap_steps,gap_events=self.gap_events)

    def classify(self,paths):
        return any(under(p,root) for p in paths for root in (PLATE,self.payload))

    def read(self, report, plate_up):
        if not isinstance(report,tuple) or len(report)<2 or report[0] is None or report[1] is None:
            if not self.unresolved_report_gap:
                warnings.warn('Current PhysX placement contact report unavailable; '
                              'continuing with unknown evidence.',RuntimeWarning)
            self.unresolved_report_gap=True
            for key in self.active:self._unknown(key)
            report=((),())
        self.unknown_pairs.intersection_update(self.active)
        headers,data=report[:2]
        seen=set();reads=0;cached=0
        for h in headers:
            paths,key,relevant=self.identity.get(h.stage_id,h.collider0,h.collider1)
            if not relevant:continue
            seen.add(key)
            if h.type==self.lost:
                self.active.pop(key,None);self.unknown_pairs.discard(key);continue
            if h.type not in (self.found,self.persist):raise RuntimeError('Unknown PhysX event')
            n,offset=int(h.num_contact_data),int(h.contact_data_offset)
            if n<=0 or offset<0 or offset+n>len(data):
                self.active[key]=(paths,(int(h.actor0),int(h.actor1)),int(h.stage_id),np.zeros(3),(),False,False,False,False)
                self._unknown(key);continue
            normal_sum=np.zeros(3);up0=up1=top0=top1=False
            # Retain upward normals for payload/plate classification at the
            # current measured plate orientation; stale geometry is not used.
            normals=[]
            for i in range(offset,offset+n):
                v=data[i];imp=v.impulse;normal=v.normal;pos=v.position
                values=(imp.x,imp.y,imp.z,normal.x,normal.y,normal.z,pos.x,pos.y,pos.z)
                if not np.isfinite(values).all():raise RuntimeError('Nonfinite physical contact')
                reads+=1
                if imp.x**2+imp.y**2+imp.z**2<=1e-18:continue
                vec=np.array([normal.x,normal.y,normal.z]);normal_sum+=vec;normals.append(vec)
                up0 |= normal.z>.5;up1 |= normal.z<-.5
                on_face=abs(pos.z-self.top)<=self.tolerance
                top0 |= bool(normal.z>.5 and on_face);top1 |= bool(normal.z<-.5 and on_face)
            self.active[key]=(paths,(int(h.actor0),int(h.actor1)),int(h.stage_id),normal_sum,
                              tuple(normals),up0,up1,top0,top1)
            self.unknown_pairs.discard(key)
        sleeping={}
        for key,item in self.active.items():
            if key in seen:continue
            for actor in item[1]:
                token=(item[2],actor)
                if token not in sleeping:sleeping[token]=self.sleeping(*token)
                if sleeping[token] is not True:
                    self._unknown(key)
            cached+=1
        grasp={role:{h:[np.zeros(3),np.zeros(3)] for h in PADS} for role in ('plate','payload')}
        robot_contact=dict(plate=False,payload=False)
        table=box_plate=box_other=False
        for item in self.active.values():
            paths,_,_,normal_sum,normals,up0,up1,top0,top1=item
            for role,root in (('plate',PLATE),('payload',self.payload)):
                index=0 if under(paths[0],root) else (1 if under(paths[1],root) else None)
                if index is None:continue
                other=paths[1-index];sign=1 if index==0 else -1
                if normals and under(other,ROBOT):robot_contact[role]=True
                for hand,pads in PADS.items():
                    for k,pad in enumerate(pads):
                        if under(other,pad):grasp[role][hand][k]+=sign*normal_sum
                if role=='plate' and under(other,DESTINATION):table |= top0 if index==0 else top1
                if role=='payload':
                    if under(other,PLATE):
                        box_plate |= any(float(np.dot(sign*v,plate_up))>.5 for v in normals)
                    elif not under(other,ROBOT) and not under(other,self.payload):
                        box_other |= up0 if index==0 else up1
        grasped={role:bool(any(np.dot(a,b)<0 for a,b in hands.values())) for role,hands in grasp.items()}
        evidence=self.evidence_status()
        if not evidence['complete']:
            self.gap_steps+=1
            grasped=dict(plate=False,payload=False)
            table=box_plate=box_other=False
        return dict(plate_grasped=grasped['plate'],payload_grasped=grasped['payload'],
                    plate_robot_contact=robot_contact['plate'],payload_robot_contact=robot_contact['payload'],
                    plate_on_table=bool(table),payload_supported_by_plate=bool(box_plate),
                    payload_on_other_support=bool(box_other),reported_point_reads=reads,
                    active_pair_count=len(self.active),sleep_cached_pairs=cached,
                    contact_evidence=self.evidence_status())


class Geometry:
    def __init__(self,stage,*,payload_variant='beef_box'):
        self.payload=payload_for(payload_variant)
        from scipy.spatial import ConvexHull
        from pxr import Gf,Usd,UsdGeom,UsdPhysics
        if UsdGeom.GetStageUpAxis(stage)!='Z' or abs(UsdGeom.GetStageMetersPerUnit(stage)-1)>1e-9:
            raise ValueError('Expected meter-scale Z-up scene')
        table=stage.GetPrimAtPath(DESTINATION)
        if not table or not table.IsA(UsdGeom.Cube) or not table.HasAPI(UsdPhysics.CollisionAPI):
            raise ValueError('Expected the actual colliding dining tabletop')
        if table.GetAttribute('physics:collisionEnabled').Get() is False:raise ValueError('Disabled table collision')
        self.table_tf=UsdGeom.Xformable(table).ComputeLocalToWorldTransform(0)
        up=self.table_tf.TransformDir(Gf.Vec3d(0,0,1)).GetNormalized()
        if (up-Gf.Vec3d(0,0,1)).GetLength()>1e-6:raise ValueError('Tilted table not supported')
        self.half=float(UsdGeom.Cube(table).GetSizeAttr().Get())/2
        self.top=float(self.table_tf.Transform(Gf.Vec3d(0,0,self.half))[2])
        self.table_inverse=np.array(self.table_tf.GetInverse())
        self.vertices={}
        for role,path in (('plate',PLATE),('payload',self.payload)):
            body=stage.GetPrimAtPath(path)
            if not body or not body.HasAPI(UsdPhysics.RigidBodyAPI):raise ValueError('Missing rigid target '+path)
            api=UsdPhysics.RigidBodyAPI(body)
            if api.GetKinematicEnabledAttr().Get() is True or api.GetRigidBodyEnabledAttr().Get() is False:
                raise ValueError('Expected dynamic target '+path)
            rigid_inverse=UsdGeom.Xformable(body).ComputeLocalToWorldTransform(0).RemoveScaleShear().GetInverse()
            vertices=[]
            for prim in Usd.PrimRange(body):
                if not prim.IsA(UsdGeom.Mesh):continue
                visible=UsdGeom.Imageable(prim).ComputeVisibility()!='invisible'
                collider=prim.HasAPI(UsdPhysics.CollisionAPI) and prim.GetAttribute('physics:collisionEnabled').Get() is not False
                if not(visible or collider):continue
                tf=UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)*rigid_inverse
                vertices.extend(list(tf.Transform(Gf.Vec3d(*v))) for v in prim.GetAttribute('points').Get())
            v=np.asarray(vertices,dtype=float)
            if v.ndim!=2 or v.shape[1]!=3 or not np.isfinite(v).all():raise ValueError('Missing mesh geometry')
            # Measurement-only extrema; never alter the rendered/collision mesh.
            self.vertices[role]=v[ConvexHull(v).vertices]
        self.footprint=ConvexHull(self.vertices['plate'][:,:2]).equations

    def facts(self,poses,payload_com):
        plate=poses['plate'];box=poses['payload']
        pr=rotation(plate['quaternion_wxyz']);br=rotation(box['quaternion_wxyz'])
        world=pr.apply(self.vertices['plate'])+plate['position_m']
        local=world@self.table_inverse[:3,:3]+self.table_inverse[3,:3]
        inside=bool(np.all(np.abs(local[:,:2])<=self.half+1e-7))
        com_world=br.apply(payload_com)+box['position_m']
        box_local=pr.inv().apply(com_world-np.asarray(plate['position_m']))
        box_inside=bool(np.all(self.footprint[:,:2]@box_local[:2]+self.footprint[:,2]<=1e-7))
        return dict(plate_in_region=inside,payload_inside_plate=box_inside,
                    payload_above_plate=bool(box_local[2]>0),payload_z_in_plate_m=float(box_local[2]))


class IsaacPlacementReader:
    def __init__(self,stage,output,episode_id,config=DetectionConfig(),*,payload_variant='beef_box'):
        from pxr import Usd,UsdPhysics,PhysxSchema,PhysicsSchemaTools
        from omni.physx import get_physx_simulation_interface
        from omni.physx.bindings._physx import ContactEventType
        self.stage=stage;self.output=Path(output);self.output.mkdir(parents=True,exist_ok=False)
        self.variant=payload_variant;self.payload=payload_for(payload_variant)
        self.geometry=Geometry(stage,payload_variant=payload_variant);self.interface=get_physx_simulation_interface()
        self.detector=PhysicalDetector(episode_id,config);self.closed=False;self.step=-1
        self.history=deque(maxlen=steps_for_seconds(1.0))
        self.detector_offset=0
        self.execution_started=False
        self.latest=None
        with Usd.EditContext(stage,stage.GetSessionLayer()):
            for path in (PLATE,self.payload):
                PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath(path)).CreateThresholdAttr(0.)
        def sleeping(stage_id,actor):
            prim=stage.GetPrimAtPath(PhysicsSchemaTools.intToSdfPath(actor))
            if not prim:raise RuntimeError('Missing contact actor')
            return bool(not prim.HasAPI(UsdPhysics.RigidBodyAPI) or self.interface.is_sleeping(stage_id,actor))
        self.stream=ContactStream(PhysicsSchemaTools.intToSdfPath,sleeping,
            (ContactEventType.CONTACT_FOUND,ContactEventType.CONTACT_PERSIST,ContactEventType.CONTACT_LOST),self.geometry.top,
            payload_variant=payload_variant)
        self.hashes={l.realPath:hashlib.sha256(Path(l.realPath).read_bytes()).hexdigest()
                     for l in stage.GetUsedLayers() if l.realPath and Path(l.realPath).is_file()}
        self.trace=(self.output/'physical_facts.jsonl').open('x')
        (self.output/'detection_profile.json').write_text(json.dumps(dict(task_id='10.2',
            config=asdict(config),target=DESTINATION,payload_variant=self.variant,payload_instance=self.payload,
            region='full actual tabletop, entire plate footprint',
            table_top_m=self.geometry.top,physics_modified=False,
            instrumentation='session-layer contact reports only; no control writes'),indent=2)+'\n')

    def initialize(self,world):
        from evaluation.physics_readers import ReadOnlyRigidBody
        self.world=world;self.views={};self.last_time=float(world.current_time)
        for role,path in (('plate',PLATE),('payload',self.payload)):
            view=ReadOnlyRigidBody(prim_path=path)
            view.initialize();self.views[role]=view
        self.com=np.asarray(self.views['payload'].get_com()[0]).reshape(3)
        self.force_audit=None
        if os.environ.get('ARENA_FORCE_AUDIT')=='1':
            from .force_audit import ForceAudit
            self.force_audit=ForceAudit(self.output,self.views,self.detector.config.physics_dt_s)

    def after_step(self):
        from pxr import UsdGeom
        if self.closed:raise RuntimeError('Reader already closed')
        now=float(self.world.current_time)
        if abs(now-self.last_time-self.detector.config.physics_dt_s)>1e-6:
            raise RuntimeError('Missed/repeated physical step')
        if UsdGeom.Xformable(self.stage.GetPrimAtPath(DESTINATION)).ComputeLocalToWorldTransform(0)!=self.geometry.table_tf:
            raise RuntimeError('Target table moved; cached scoring region invalid')
        poses={}
        for role,view in self.views.items():
            pos,q=view.get_world_pose()
            if not np.isfinite(pos).all():raise ValueError('Nonfinite body position')
            rotation(q)
            poses[role]=dict(position_m=pos.tolist(),quaternion_wxyz=q.tolist())
        geometric=self.geometry.facts(poses,self.com)
        up=rotation(poses['plate']['quaternion_wxyz']).apply([0,0,1])
        report=(self.interface.get_full_contact_report() if self.force_audit is not None
                and hasattr(self.interface,'get_full_contact_report') else self.interface.get_contact_report())
        contacts=self.stream.read(report,up)
        self.step+=1
        if self.force_audit is not None:
            self.force_audit.record(report,self.stream,self.views,poses,self.step,now,
                                    'execution' if self.execution_started else 'settle')
        facts=PhysicalFacts(self.step-self.detector_offset,now,float(poses['plate']['position_m'][2]),
            float(poses['payload']['position_m'][2]),geometric['payload_z_in_plate_m'],
            contacts['plate_grasped'],contacts['plate_on_table'],geometric['plate_in_region'],
            bool(contacts['payload_supported_by_plate'] and geometric['payload_inside_plate'] and geometric['payload_above_plate']),
            geometric['payload_inside_plate'],geometric['payload_above_plate'],contacts['payload_grasped'],
            contacts['payload_on_other_support'])
        result=self.detector.observe(facts)
        self.latest=dict(result,poses=poses,contacts=contacts,reader_step=self.step,
                         payload_variant=self.variant,payload_instance=self.payload,
                         phase='execution' if self.execution_started else 'settle')
        self.history.append(self.latest)
        self.trace.write(json.dumps(self.latest,allow_nan=False)+'\n')
        self.trace.flush();self.last_time=now

    def begin_episode(self):
        """Drop history during reset is not a task penalty. Seed actual state."""
        if self.execution_started or self.latest is None:
            raise RuntimeError('Invalid evaluation start')
        self.execution_started=True
        self.detector_offset=self.step
        self.detector=PhysicalDetector(self.detector.scorer.episode_id,self.detector.config)
        f=PhysicalFacts(**dict(self.latest['facts'],step=0))
        initial=self.detector.observe(f)
        self.latest=dict(self.latest,**initial,phase='episode_start')

    def close(self,completed):
        if self.closed:return
        self.closed=True;self.trace.close()
        if getattr(self,'force_audit',None) is not None:self.force_audit.close()
        unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in self.hashes.items())
        result=dict(completed=bool(completed and unchanged),source_layers_unchanged=unchanged,
                    physics_samples=self.step+1,score=None)
        if completed and unchanged:result['score']=self.detector.finalize()
        (self.output/'score_result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        if not unchanged:raise RuntimeError('Source layer changed')
