"""Privileged task-10.1 Isaac reader. No motion, physics tuning or HTTP input.

Instantiate before World.reset to enable reports; initialize views after reset;
call after_step after EVERY completed physics step. Contact data is polled from
PhysX's current-step buffer. Missing active reports fail closed; sleeping pairs
are retained only with explicit PhysX sleep evidence, never invented impulses.
"""
from collections import deque
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.transform import Rotation

from . import initial_state as m, process_rules as p

HOLD_FAILURE_PENALTY = p.HOLD_FAILURE_PENALTY
READER_SCHEMA = "arena.microwave_take.isaac_reader.v1"
PADS = {hand: tuple(m.ROBOT+f"/gripper_{hand}_{part}_link4/collisions/{part}_link4/mesh"
                   for part in ("inner", "outer")) for hand in ("l", "r")}


def under(path, root):
    return path == root or path.startswith(root + "/")


def rotation(q):
    a = np.asarray(q, dtype=float)
    if a.shape != (4,) or not np.isfinite(a).all() or abs(np.linalg.norm(a)-1) > 1e-4:
        raise ValueError("Invalid measured quaternion")
    return Rotation.from_quat([*a[1:], a[0]])


def transform(vertices, pose):
    return rotation(pose['quaternion_wxyz']).apply(vertices) + pose['position_m']


def inverse(points, pose):
    return rotation(pose['quaternion_wxyz']).inv().apply(np.asarray(points)-pose['position_m'])


def extreme_vertices(vertices):
    """Exact support-set optimization for rigid min/max, NOT a collision hull.

    Linear extrema over a point set occur on its convex hull. Only measurement
    work is reduced; the complete visual/physical meshes remain untouched.
    Degenerate inputs retain all points. No Qhull jitter/mesh simplification.
    """
    try:
        return vertices[ConvexHull(vertices).vertices]
    except QhullError:
        return vertices


def score_contacts(rows, microwave_pose):
    if not rows:
        return ()
    positions = inverse([v['position_m'] for v in rows], microwave_pose)
    impulses = np.linalg.norm([v['impulse_Ns'] for v in rows], axis=1)
    return tuple(p.ContactPoint(*v['colliders'], tuple(pos), float(impulse))
                 for v, pos, impulse in zip(rows, positions, impulses))


class ContactClassifier:
    """Partition measured pairs before expensive geometry; never filter physics.

    Collider paths and the fixed wall registry are invariant for this reader.
    Every raw point was validated in contact_rows before entering this class.
    Zero-impulse persistent wall contacts remain eligible to maintain events.
    """
    def __init__(self, wall_paths, payload_root=m.PAYLOAD):
        self.payload_root = payload_root
        self.walls = frozenset(wall_paths)
        if self.walls != frozenset(p.INNER_WALL_FACES):
            raise ValueError('Incomplete wall registry')
        self.pairs = {}

    def split(self, rows):
        plate_rows, wall_rows = [], []
        for row in rows:
            pair = tuple(row['colliders'])
            if pair not in self.pairs:
                if len(pair) != 2 or any(not isinstance(x, str) or not x.startswith('/') for x in pair):
                    raise ValueError('Invalid measured collider pair')
                a, b = pair
                plate = under(a, m.PLATE) or under(b, m.PLATE)
                wall = any(w in self.walls and any(under(actor, root) for root in (m.ROBOT, m.PLATE, self.payload_root))
                           for w, actor in ((a, b), (b, a)))
                self.pairs[pair] = plate, wall
            plate, wall = self.pairs[pair]
            if plate: plate_rows.append(row)
            if wall: wall_rows.append(row)
        return plate_rows, wall_rows


def pack_native_contact(paths, value):
    # carb.Float3's sequence/NumPy conversion is substantially more expensive
    # than its named components. Read each native property once; same values.
    pos,normal,impulse=value.position,value.normal,value.impulse
    return dict(colliders=paths,position_m=[pos.x,pos.y,pos.z],
                normal=[normal.x,normal.y,normal.z],
                impulse_Ns=[impulse.x,impulse.y,impulse.z],
                separation_m=float(value.separation),cached_sleeping=False)


def validate_contact_points(rows):
    if not rows:return
    values=np.asarray([v['position_m']+v['normal']+v['impulse_Ns']+[v['separation_m']] for v in rows])
    if not np.isfinite(values).all():raise ValueError('Nonfinite contact')
    positive=np.linalg.norm(values[:,6:9],axis=1)>p.IMPULSE_EPSILON_NS
    for row,loaded in zip(rows,positive):row['had_positive_impulse']=bool(loaded)


def opening_direction(lower, upper):
    """Convert the asset's signed joint axis to positive opening, never abs()."""
    if lower is not None and upper is not None:
        if lower < -60 and abs(upper) < 1e-6:return -1.
        if abs(lower) < 1e-6 and upper > 60:return 1.
    raise ValueError('Unknown door closed limit/opening direction')


def force_on(row, root):
    """PhysX normal points from collider1 to collider0; never a model value."""
    sign = 1 if under(row['colliders'][0], root) else -1
    return np.asarray(row['impulse_Ns']) * sign


def supported(row):
    # Live rows already have a validated load bit. Sleeping history retains
    # that bit but has zero *current* impulse. Old standalone diagnostic inputs
    # without the bit still use the previous numeric calculation.
    if type(row.get('had_positive_impulse')) is bool:
        return row['had_positive_impulse']
    return np.linalg.norm(row['impulse_Ns']) > p.IMPULSE_EPSILON_NS


def carrying_facts(rows, plate_pose, payload_pose, payload_com_local, plate_footprint, payload_root=m.PAYLOAD):
    """Measured opposed pad loads + no stand support; physical box support.

    No commanded closure or arbitrary pose-drift/angle success gate. The box
    COM projection must remain within the plate footprint, with support from
    the plate. This is a rigid-object reader, not a deformable grasp model.
    """
    loads = {hand: [np.zeros(3), np.zeros(3)] for hand in PADS}
    normals = {hand: [[], []] for hand in PADS}
    stand = False
    box_support = False
    plate_up = rotation(plate_pose['quaternion_wxyz']).apply([0,0,1])
    for row in rows:
        a, b = row['colliders']
        if not supported(row):
            continue
        if any(under(x, m.PLATE) for x in (a,b)):
            other = b if under(a, m.PLATE) else a
            if under(other, m.STAND):
                stand = True
            for hand, pads in PADS.items():
                if other in pads:
                    index = pads.index(other)
                    loads[hand][index] += force_on(row, m.PLATE)
                    normals[hand][index].append(np.asarray(row['normal']) * (1 if under(a,m.PLATE) else -1))
            if under(other, payload_root):
                normal = np.asarray(row['normal']) * (1 if under(a,payload_root) else -1)
                if np.dot(normal, plate_up) > 0:
                    box_support = True
    opposed = {}
    for hand in PADS:
        ns = normals[hand]
        # Positive, opposing contact normals establish two-sided compression.
        # Sleeping evidence may preserve normals, but is never logged as force.
        opposed[hand] = bool(ns[0] and ns[1] and
                            np.dot(np.sum(ns[0],axis=0),np.sum(ns[1],axis=0)) < 0)
    com_world = transform(np.asarray([payload_com_local]), payload_pose)[0]
    com_plate = inverse([com_world], plate_pose)[0]
    inside = bool(np.all(plate_footprint[:,:2] @ com_plate[:2] + plate_footprint[:,2] <= 1e-7))
    above = bool(com_plate[2] > 0)
    held = any(opposed.values()) and not stand
    return dict(plate_grasped_by_robot=bool(any(opposed.values())),
                plate_held_by_robot=bool(held), payload_on_plate=bool(box_support and inside and above),
                opposed_pad_contact=opposed, pad_impulses_Ns={h:[v.tolist() for v in vs] for h,vs in loads.items()},
                plate_supported_by_stand=stand, payload_supported_by_plate=box_support,
                payload_com_inside_plate=inside, payload_com_above_plate=above,
                payload_com_in_plate_m=com_plate.tolist())


class Geometry:
    def __init__(self, stage):
        from pxr import Gf, Usd, UsdGeom, UsdPhysics
        m.verify_plate_asset()
        self.plan = m.plan_from_stage(stage)
        if UsdGeom.GetStageUpAxis(stage) != 'Z' or abs(UsdGeom.GetStageMetersPerUnit(stage)-1) > 1e-9:
            raise ValueError('Expected Z-up metre geometry')
        cache = UsdGeom.XformCache()
        mw_tf = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(m.MICROWAVE))
        mw_inv = mw_tf.GetInverse()
        def fixed_pose(tf):
            tr=Gf.Transform(tf); q=tr.GetRotation().GetQuat()
            return dict(position_m=list(tr.GetTranslation()),quaternion_wxyz=[q.GetReal(),*q.GetImaginary()])
        self.microwave_pose = fixed_pose(mw_tf)
        hinge=stage.GetPrimAtPath(m.DOOR_JOINT)
        self.door_opening_sign=opening_direction(hinge.GetAttribute('physics:lowerLimit').Get(),
                                                 hinge.GetAttribute('physics:upperLimit').Get())
        self.stand_pose = fixed_pose(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(m.STAND)))
        def box(path):
            prim = stage.GetPrimAtPath(path)
            if not prim.IsA(UsdGeom.Cube) or not prim.HasAPI(UsdPhysics.CollisionAPI):
                raise ValueError('Unknown cavity collider '+path)
            if prim.GetAttribute('physics:collisionEnabled').Get() is not True:
                raise ValueError('Disabled cavity collider '+path)
            xf=cache.GetLocalToWorldTransform(prim)*mw_inv
            axes=np.asarray(xf)[:3,:3]
            if np.any(np.sum(abs(axes)>1e-7,axis=0)!=1) or np.any(np.sum(abs(axes)>1e-7,axis=1)!=1):
                raise ValueError('Wall is not axis-aligned in microwave frame')
            s=UsdGeom.Cube(prim).GetSizeAttr().Get()/2
            pts=np.array([list(xf.Transform(Gf.Vec3d(x,y,z))) for x in (-s,s) for y in (-s,s) for z in (-s,s)])
            return tuple(pts.min(axis=0)),tuple(pts.max(axis=0))
        self.walls=tuple(p.WallBox(path,*box(path)) for path in p.INNER_WALL_FACES)
        # Read while CLOSED, then retain this plane in the rigid oven frame.
        self.front_y=float(box(m.MICROWAVE+'/Asset/door/col_door')[1][1])
        by_name={w.collider.rsplit('/',1)[1]:w for w in self.walls}
        self.lo=np.array([by_name['col_left'].maximum_m[0],self.front_y,by_name['col_bottom'].maximum_m[2]])
        self.hi=np.array([by_name['col_right'].minimum_m[0],by_name['col_back'].minimum_m[1],by_name['col_top'].minimum_m[2]])
        if not np.all(self.hi>self.lo):raise ValueError('Invalid cavity')
        self.vertices={}
        for role,root in [('plate',m.PLATE+'/Asset'),('payload',self.plan.bindings['payload'])]:
            tf=cache.GetLocalToWorldTransform(stage.GetPrimAtPath(root)); inv=tf.GetInverse()
            axes=np.asarray(tf)[:3,:3]; scale=np.linalg.norm(axes,axis=1)
            if not np.allclose((axes/scale[:,None])@(axes/scale[:,None]).T,np.eye(3),atol=1e-7):
                raise ValueError('Sheared target body')
            vertices=[]
            for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
                if not prim.IsA(UsdGeom.Mesh):continue
                visible=UsdGeom.Imageable(prim).ComputeVisibility()!='invisible'
                collision=prim.HasAPI(UsdPhysics.CollisionAPI) and prim.GetAttribute('physics:collisionEnabled').Get() is not False
                if not (visible or collision):continue
                local=cache.GetLocalToWorldTransform(prim)*inv
                vertices.extend([np.asarray(local.Transform(Gf.Vec3d(*v)))*scale for v in prim.GetAttribute('points').Get()])
            data=np.asarray(vertices)
            if data.ndim!=2 or data.shape[1]!=3 or not np.isfinite(data).all():raise ValueError('Missing target geometry')
            self.vertices[role]=data
        self.plate_footprint=ConvexHull(self.vertices['plate'][:,:2]).equations
        self.extrema={role:extreme_vertices(v) for role,v in self.vertices.items()}

    def contents(self, poses):
        result={}
        for role,vertices in self.extrema.items():
            local=inverse(transform(vertices,poses[role]),self.microwave_pose)
            low,high=local.min(axis=0),local.max(axis=0)
            result[role]=dict(front_clearance_m=float(self.front_y-high[1]),minimum_m=low.tolist(),maximum_m=high.tolist(),
                              inside=bool(np.all(low>=self.lo) and np.all(high<=self.hi)),vertices=len(self.vertices[role]))
        return result


class IsaacProcessReader:
    def __init__(self, stage, output, episode_id, *, record_raw_contacts=False):
        from pxr import Usd, UsdPhysics, PhysxSchema
        from omni.physx import get_physx_simulation_interface
        self.stage=stage;self.output=Path(output);self.episode_id=episode_id
        self.plan=m.plan_from_stage(stage)
        self.bindings=self.plan.bindings
        self.payload=self.bindings['payload']
        self.geometry=Geometry(stage)
        self.interface=get_physx_simulation_interface()
        self.report_bodies=[]
        with Usd.EditContext(stage,stage.GetSessionLayer()):
            for root in (m.ROBOT,m.MICROWAVE,m.PLATE,self.payload):
                for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
                    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                        PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
                        self.report_bodies.append(str(prim.GetPath()))
        self.layers={l.realPath:hashlib.sha256(Path(l.realPath).read_bytes()).hexdigest()
                     for l in stage.GetUsedLayers() if l.realPath and Path(l.realPath).is_file()}
        from .contact_recording import ContactRecorder
        self.contact_recorder=ContactRecorder(self.output,record_raw_contacts=record_raw_contacts)
        self.trace_file=open(self.output/'process_physics.jsonl','x')
        self.scores_file=open(self.output/'process_scores.jsonl','x')
        self.active={};self.step=-1;self.last_time=None;self.scorer=None;self.history=deque(maxlen=121)
        self.collider_paths={}
        self.contact_classifier=ContactClassifier(p.INNER_WALL_FACES,self.payload)
        self.views={};self.previous_held=False;self.previous_payload=True;self.row=None;self.initial_plate_bottom=None

    def initialize(self, world):
        from isaacsim.core.prims import SingleArticulation
        from evaluation.physics_readers import ReadOnlyRigidBody
        from arena_api import microwave_physics
        self.world=world;self.robot=world.scene.get_object('arena_robot')
        # Preserve one second of initial evidence when physics frequency changes.
        self.history=deque(maxlen=round(1/float(world.get_physics_dt()))+1)
        self.microwave=SingleArticulation(prim_path=microwave_physics.ROOT,name='process_microwave')
        self.microwave.initialize()
        if self.microwave.dof_names!=['door_hinge']:raise ValueError('Unknown microwave joint order')
        for role,path in [('plate',m.PLATE+'/Asset'),('payload',self.payload)]:
            view=ReadOnlyRigidBody(prim_path=path);view.initialize();self.views[role]=view
        self.payload_com=np.asarray(self.views['payload'].get_com()[0]).reshape(3)
        self.initial_mw=self.pose(self.microwave)
        self.last_time=float(world.current_time)

    @staticmethod
    def pose(body):
        pos,quat=body.get_world_pose()
        if not np.isfinite(pos).all():raise ValueError('Nonfinite physical pose')
        rotation(quat)
        return dict(position_m=np.asarray(pos).tolist(),quaternion_wxyz=np.asarray(quat).tolist())

    def contact_rows(self):
        from pxr import PhysicsSchemaTools, UsdPhysics
        from omni.physx.bindings._physx import ContactEventType
        report=self.interface.get_contact_report()
        if not isinstance(report,tuple) or len(report)<2:raise RuntimeError('Unavailable current-step contact buffer')
        headers,data=report[:2];seen=set();rows=[]
        # Only actor sleep queries are cached, and only within THIS step. An
        # actor waking next frame must still reject missing contact evidence.
        sleeping={}
        for h in headers:
            paths=[]
            for v in (h.collider0,h.collider1):
                key_path=int(v)
                if key_path not in self.collider_paths:
                    self.collider_paths[key_path]=str(PhysicsSchemaTools.intToSdfPath(v))
                paths.append(self.collider_paths[key_path])
            relevant=any(under(x,m.PLATE) or under(x,self.payload) for x in paths) or (
                any(under(x,m.ROBOT) for x in paths) and not all(under(x,m.ROBOT) for x in paths))
            if not relevant:continue
            key=tuple(sorted(paths));seen.add(key)
            if h.type==ContactEventType.CONTACT_LOST:
                self.active.pop(key,None);continue
            if h.type not in (ContactEventType.CONTACT_FOUND,ContactEventType.CONTACT_PERSIST):
                raise RuntimeError('Unknown contact event')
            current=[]
            for i in range(h.contact_data_offset,h.contact_data_offset+h.num_contact_data):
                current.append(pack_native_contact(paths,data[i]))
            if not current:raise RuntimeError('Active contact has no point data')
            self.active[key]=(current,int(h.stage_id),[int(h.actor0),int(h.actor1)])
            rows.extend(current)
        validate_contact_points(rows)
        for key,(cached,stage_id,actors) in self.active.items():
            if key in seen:continue
            asleep=[]
            for actor in actors:
                actor_key=(stage_id,actor)
                if actor_key not in sleeping:
                    prim=self.stage.GetPrimAtPath(PhysicsSchemaTools.intToSdfPath(actor))
                    if not prim:raise RuntimeError('Unresolved contact actor')
                    dynamic=prim.HasAPI(UsdPhysics.RigidBodyAPI)
                    sleeping[actor_key]=not dynamic or self.interface.is_sleeping(stage_id,actor)
                asleep.append(sleeping[actor_key])
            if not all(asleep):raise RuntimeError('Missing current contact report for an awake pair: '+str(key))
            rows.extend(dict(row,cached_sleeping=True,impulse_Ns=[0.,0.,0.]) for row in cached)
        return rows

    def after_step(self):
        now=float(self.world.current_time)
        if abs(now-self.last_time-float(self.world.get_physics_dt()))>1e-6:raise RuntimeError('Reader missed or repeated a physics step')
        self.last_time=now;self.step+=1
        rows=self.contact_rows()
        plate_rows, wall_rows=self.contact_classifier.split(rows)
        poses={role:self.pose(view) for role,view in self.views.items()}
        mw_pose=self.pose(self.microwave)
        if math.dist(mw_pose['position_m'],self.initial_mw['position_m'])>.001 or (
            rotation(mw_pose['quaternion_wxyz'])*rotation(self.initial_mw['quaternion_wxyz']).inv()).magnitude()>1e-4:
            raise RuntimeError('Fixed microwave moved; cached opening geometry invalid')
        contents=self.geometry.contents(poses)
        carry=carrying_facts(plate_rows,poses['plate'],poses['payload'],self.payload_com,self.geometry.plate_footprint,self.payload)
        bottom=float(transform(self.geometry.extrema['plate'],poses['plate'])[:,2].min())
        carry['plate_bottom_world_z_m']=bottom
        # 0.1 mm is a numerical resolution guard, not an imposed task lift height.
        carry['plate_lifted_from_support']=bool(self.initial_plate_bottom is not None and
            bottom>self.initial_plate_bottom+.0001 and not carry['plate_supported_by_stand'])
        dropped=bool((self.previous_held and not carry['plate_held_by_robot']) or
                     (self.previous_payload and not carry['payload_on_plate']))
        self.previous_held=carry['plate_held_by_robot'];self.previous_payload=carry['payload_on_plate']
        names=self.robot.dof_names
        row=dict(reader_schema=READER_SCHEMA,reader_step=self.step,time_s=now,source='isaac_physics',
                 episode_id=self.episode_id,bodies=poses,robot=self.pose(self.robot),microwave=mw_pose,
                 door_joint_angle_deg=float(np.degrees(self.microwave.get_joint_positions()[0])),
                 door_angle_deg=self.geometry.door_opening_sign*float(np.degrees(self.microwave.get_joint_positions()[0])),
                 joint_positions=dict(zip(names,self.robot.get_joint_positions().tolist())),
                 joint_velocities=dict(zip(names,self.robot.get_joint_velocities().tolist())),
                 containment=contents,carry=carry,dropped_since_last_frame=dropped,
                 contact_points=len(rows),sleep_cached_points=sum(r['cached_sleeping'] for r in rows))
        if not np.isfinite([row['door_angle_deg'],*row['joint_positions'].values(),*row['joint_velocities'].values()]).all():
            raise ValueError('Nonfinite measured joints')
        row['contact_summary']=self.contact_recorder.record(self.step,now,rows)
        self.trace_file.write(json.dumps(row,allow_nan=False)+'\n')
        self.history.append((row,rows));self.row=row
        if self.scorer:self.score_row(row,wall_rows,classified=True)
        for f in (self.trace_file,self.scores_file):f.flush()

    def initial_gate(self, metadata):
        from pxr import Usd, UsdPhysics
        plan=m.plan_from_stage(self.stage);row=self.row
        if len(self.history)<self.history.maxlen:raise RuntimeError('Need one full second of settled evidence')
        pairs=[r for _,rows in self.history for r in rows]
        def pair(a,b):return any(supported(r) and ((under(r['colliders'][0],a) and under(r['colliders'][1],b)) or
                                (under(r['colliders'][1],a) and under(r['colliders'][0],b))) for r in pairs)
        def dyn(path):
            rb=UsdPhysics.RigidBodyAPI(self.stage.GetPrimAtPath(path))
            return bool(rb and rb.GetRigidBodyEnabledAttr().Get() and not rb.GetKinematicEnabledAttr().Get())
        robot_rows=[r for r in pairs if any(under(x,m.ROBOT) for x in r['colliders'])]
        settled=all(np.ptp([r['bodies'][role]['position_m'] for r,_ in self.history],axis=0).max()<.001 for role in self.views)
        settled=settled and all(max(abs(r['joint_velocities'][n]) for r,_ in self.history)<(.003 if q.unit=='m' else .05)
                                for n,q in plan.joint_targets.items())
        hashes=plan.required_asset_sha256;m.verify_plate_asset()
        if plan.payload_variant == 'leftover_box':
            from .payload_assets import verify_leftover_asset
            verify_leftover_asset(self.stage)
        m.verify_stand_geometry(self.stage)
        m.verify_plate_geometry(self.stage)
        unchanged=all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==h for path,h in self.layers.items())
        payload_collision_root = self.payload + '/ContactSurface' if plan.payload_variant == 'beef_box' else self.payload
        colliders=[prim for root in (m.PLATE+'/Asset/ContactSurface',payload_collision_root)
                   for prim in Usd.PrimRange(self.stage.GetPrimAtPath(root)) if prim.HasAPI(UsdPhysics.CollisionAPI)
                   and prim.GetAttribute('physics:collisionEnabled').Get() is not False]
        expected_colliders = 877 if plan.payload_variant == 'beef_box' else 2
        fridge=metadata['l1_fridge_policy']
        facts=dict(settled=bool(settled),physics_valid=not metadata['physics_issues'],microwave_fixed=True,
                   door_passive_dynamic=dyn(m.MICROWAVE+'/Asset/door') and not self.stage.GetPrimAtPath(m.DOOR_JOINT).HasAPI(UsdPhysics.DriveAPI,'angular'),
                   plate_dynamic=dyn(m.PLATE+'/Asset'),payload_dynamic=dyn(self.payload),
                   collisions_enabled=len(colliders)==expected_colliders and all(v.GetAttribute('physics:collisionEnabled').Get() is True for v in colliders),
                   self_collision_enabled=self.stage.GetPrimAtPath(m.ROBOT).GetAttribute('physxArticulation:enabledSelfCollisions').Get() is True,
                   plate_supported_by_stand=pair(m.PLATE,m.STAND),payload_supported_by_plate=pair(self.payload,m.PLATE),
                   plate_inside_cavity=row['containment']['plate']['inside'],payload_inside_cavity=row['containment']['payload']['inside'],
                   other_payloads_inactive=all(not self.stage.GetPrimAtPath(x) or not self.stage.GetPrimAtPath(x).IsActive() for x in plan.other_payloads),
                   robot_not_holding_objects=not any(any(under(x,m.PLATE) or under(x,self.payload) for x in r['colliders']) for r in robot_rows),
                   robot_clear_of_environment=not robot_rows,
                   no_initial_interpenetration=bool(pairs) and min(r['separation_m'] for r in pairs)>=-.0001,
                   no_runtime_assistance=not fridge['events'] and not fridge['docked_object_count'] and not fridge['tray_dock_enabled'],
                   task_assets_unchanged=unchanged)
        def pose(v):return m.Pose(tuple(v['position_m']),float(rotation(v['quaternion_wxyz']).as_euler('xyz',degrees=True)[2]))
        obs=m.InitialObservation(self.episode_id,'isaac_physics',0,m.TASK_ID,self.bindings,
              dict(robot=pose(row['robot']),microwave=pose(row['microwave']),stand=pose(self.geometry.stand_pose),
                   **{role:pose(v) for role,v in row['bodies'].items()}),
              {n:m.JointTarget(row['joint_positions'][n],q.unit) for n,q in plan.joint_targets.items()},
              row['door_angle_deg'],{k:bool(v) for k,v in facts.items()},hashes)
        check=m.check_initial_state(plan,obs)
        receipt=dict(check=check,measured_facts=obs.facts,scene_layer_sha256=self.layers,
                     time_s=row['time_s'],reader_step=self.step,real_measurements=True)
        (self.output/'process_initial_acceptance.json').write_text(json.dumps(receipt,indent=2)+'\n')
        if not check['initial_state_checks_passed']:raise RuntimeError('Live initial acceptance failed: '+str(check['errors']))
        # Public, task-owned replay compatibility. No absolute workspace paths:
        # recording on another host must not move objects or override this reset.
        # Rendering noise/settle time are deliberately not part of the identity.
        portable_plan = plan.to_dict()
        portable_plan.pop('task_asset_layer', None)
        metadata['replay_reset_contract'] = dict(schema='microwave_take.reset.v1',
            plan=portable_plan)
        self.scorer=self.make_scorer()
        self.initial_plate_bottom=row['carry']['plate_bottom_world_z_m']
        # Accepted reset is a point in time, not a drop during earlier settling.
        self.row=dict(row,dropped_since_last_frame=False)
        self.score_row(self.row,self.history[-1][1])

    def make_scorer(self):
        return p.ProcessScorer(self.episode_id,float(self.world.get_physics_dt()),self.geometry.walls,initial_state_accepted=True,
                              hold_failure_penalty=HOLD_FAILURE_PENALTY,bindings=self.bindings)

    def score_row(self,row,rows,*,classified=False):
        if not classified:
            _, rows=self.contact_classifier.split(rows)
        contacts=score_contacts(rows,self.geometry.microwave_pose)
        f=p.ProcessFrame(self.episode_id,self.scorer.last_step+1,row['time_s'],row['door_angle_deg'],
            row['containment']['plate']['front_clearance_m'],row['containment']['payload']['front_clearance_m'],
            row['carry']['plate_held_by_robot'],row['carry']['payload_on_plate'],row['dropped_since_last_frame'],
            contacts,True,'isaac_physics',m.TASK_ID,dict(self.bindings),
            plate_grasped_by_robot=row['carry']['plate_grasped_by_robot'],
            plate_lifted_from_support=row['carry']['plate_lifted_from_support'])
        result=self.scorer.observe(f)
        self.scores_file.write(json.dumps(dict(reader_step=row['reader_step'],time_s=row['time_s'],score=result),allow_nan=False)+'\n')

    def close(self, completed):
        if self.scorer:
            result=self.scorer.finalize() if completed else self.scorer.summary()
            report=dict(reader_schema=READER_SCHEMA,isaac_stream_connected=True,
                        public_api_connected=False,physics_steps_completed=completed,
                        physical_run_completed=False,physics_shutdown_verified=False,
                        full_task_validated=False,task_success=None,score=result,
                        events=self.scorer.event_log(),physics_frames=self.step+1,
                        contact_recording=self.contact_recorder.report(),
                        scene_layer_sha256=self.layers,hold_failure_penalty=HOLD_FAILURE_PENALTY)
            (self.output/'process_result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        self.contact_recorder.close()
        for f in (self.trace_file,self.scores_file):f.close()
