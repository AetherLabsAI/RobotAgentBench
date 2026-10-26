"""Native room-bound adapter; original geometric/contact predicates preserved.
Methods ported from air_fryer/isaac_evidence.py; path bindings are instance local.
Source reader remains unchanged. No model actions, resets or pose writes.
"""
from collections import deque
from dataclasses import asdict
from pathlib import Path
import hashlib, importlib.util, json, math
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema
from evaluation.tasks.air_fryer.isaac_evidence import (
    body_colliders, hull2d, polygons_separate, rotation, Contacts,
    GraspDetector, DetectionConfig, PADS, HANDS, under, serializable,
    PHYSICS_DT, SUBSTEPS, CONTROL_DT, INITIAL_WINDOW_S,
    UP_NORMAL_Z, COUNTER_SUPPORT_NORMAL_Z,
    CakeFacts, TrayFacts, Facts,
    ON_FLOOR, ON_WALL, ON_BODY, ON_GROUND, ON_PLATE, ON_COUNTER, ON_OTHER)
from .composite import Binding, Frame, AirFryerComposite

REPO = Path(__file__).resolve().parents[4]
ENTRY = REPO / 'all_assets/air_fryer/asset.usda'
BEHAVIOR = ENTRY.parent / 'behaviors/rail_release145.py'

def physical_signature(stage, root):
    """Flattened room snapshots have no reference arcs: compare composed content."""
    result = {}
    for p in Usd.PrimRange(stage.GetPrimAtPath(root)):
        suffix=str(p.GetPath())[len(root):]
        values={}
        for a in p.GetAttributes():
            name=a.GetName()
            relevant=(name.startswith(('physics:','physx','limit:','drive:','arena:rail','xformOp'))
                      or name in ('points','faceVertexCounts','faceVertexIndices','extent','size','radius','height','axis'))
            if relevant and not (not suffix and name.startswith('xformOp')):
                values[name]=str(a.Get())
        for rel in p.GetRelationships():
            if rel.GetName().startswith('physics:'):
                values[rel.GetName()]=[str(v).replace(root,'<asset>',1) for v in rel.GetTargets()]
        if values or p.HasAPI(UsdPhysics.CollisionAPI) or p.HasAPI(UsdPhysics.RigidBodyAPI):
            result[suffix]=dict(type=p.GetTypeName(),schemas=list(p.GetAppliedSchemas()),active=p.IsActive(),values=values)
    return result

def bind_room(stage, manifest, variant):
    if variant not in ('sugar', 'no_sugar'):
        raise ValueError('Explicit target variant required')
    if UsdGeom.GetStageMetersPerUnit(stage) != 1 or UsdGeom.GetStageUpAxis(stage) != 'Z':
        raise ValueError('Expected metres/Z-up')
    def object_path(asset_id):
        objects = [o for o in manifest['objects'] if o['id'] == asset_id]
        if len(objects) != 1:
            raise ValueError('Ambiguous or missing object: ' + asset_id)
        obj = objects[0]
        expected = REPO / 'all_assets' / asset_id / 'asset.usda'
        if Path(obj['entry']).resolve() != expected.resolve():
            raise ValueError('Object must reference all_assets: ' + asset_id)
        root = obj['path'] + '/Asset'
        prim = stage.GetPrimAtPath(root)
        if not prim or not prim.IsActive():
            raise ValueError('Missing actual asset root: ' + root)
        # Check composed provenance, not just the external JSON claim.
        from pxr import Sdf
        references = {str(Path(Sdf.ComputeAssetPathRelativeToLayer(s.layer, ref.assetPath)).resolve())
                      for s in prim.GetPrimStack() for ref in s.referenceList.GetAppliedItems() if ref.assetPath}
        if str(expected.resolve()) not in references:
            asset_stage=Usd.Stage.Open(str(expected))
            actual=physical_signature(stage,root)
            approved=physical_signature(asset_stage,str(asset_stage.GetDefaultPrim().GetPath()))
            if actual != approved:
                differences=[k for k in set(actual)|set(approved) if actual.get(k)!=approved.get(k)]
                raise ValueError('Flattened snapshot differs from all_assets: '+asset_id+' '+str(differences[:12]))
        return root
    fryer = object_path('air_fryer')
    yellow = object_path('cake_plain_yellow') + '/Object'
    white = object_path('cake_frosted_white') + '/Object'
    plate = object_path('paper_plate_task2')
    from openend.tasks.scene_requirements.source_geometry import floor_paths
    floors = floor_paths(stage)
    if len(floors) != 1:
        raise ValueError('Unique physical floor required for fryer')
    paths = dict(fryer=fryer, tray=fryer+'/Centered/Tray', body=fryer+'/Centered/Body',
                 joint=fryer+'/Centered/Joints/DrawerSlideOldParams',
                 basket_floor=fryer+'/Centered/Tray/Collision/BasketFloor',
                 target=yellow if variant=='no_sugar' else white,
                 other=white if variant=='no_sugar' else yellow,
                 plate=plate, plate_support=plate+'/Support',
                 counter=manifest['supports']['fryer']['path'],
                 floor=floors[0], robot='/World/Robot')
    for key, path in paths.items():
        p = stage.GetPrimAtPath(path)
        if not p or not p.IsActive(): raise ValueError('Missing binding: '+key)
    for key in ('target','other','tray','body'):
        if not stage.GetPrimAtPath(paths[key]).HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError('Not a rigid body: '+key)
    for key in ('counter','plate_support','basket_floor','floor'):
        if not any(p.HasAPI(UsdPhysics.CollisionAPI) and p.GetAttribute('physics:collisionEnabled').Get() is not False
                   for p in Usd.PrimRange(stage.GetPrimAtPath(paths[key]))):
            raise ValueError('No support collider: '+key)
    return paths

def instrument(stage, paths):
    if not stage.GetSessionLayer().anonymous:
        raise ValueError('Anonymous episode session required')
    bodies = [stage.GetPrimAtPath(paths[k]) for k in ('target','other','tray','body')]
    bodies += [p for p in Usd.PrimRange(stage.GetPrimAtPath(paths['robot'])) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    for p in bodies:
        a = p.GetAttribute('physxContactReport:threshold')
        if a and a.HasAuthoredValueOpinion() and a.Get() != 0:
            raise ValueError('Conflicting instrumentation')
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for p in bodies:
            PhysxSchema.PhysxContactReportAPI.Apply(p).CreateThresholdAttr(0.)
    return dict(contact_bodies=len(bodies), threshold=0., physical_parameters_changed=False)

class NativeBody:
    def __init__(self, view, path):
        self.view=view.create_rigid_body_view(path)
        if self.view.count != 1 or list(self.view.prim_paths) != [path]:
            raise ValueError('Wrong native rigid body: '+path)
    def get_world_poses(self):
        t=np.asarray(self.view.get_transforms()).copy()
        return t[:,:3],t[:,[6,3,4,5]]
    def get_world_pose(self):
        p,q=self.get_world_poses()
        return p[0],q[0]
    def get_velocities(self):
        return np.asarray(self.view.get_velocities()).copy()

class AssetRail:
    def __init__(self, stage, paths, views):
        module_spec=importlib.util.spec_from_file_location('approved_fryer_rail',BEHAVIOR)
        module=importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(module)
        self.impl=module.RailRelease145(stage,paths['fryer'],views[paths['tray']],views[paths['body']])
        # Fresh enabled scene only; no restoration/reset in the adapter.
        self.impl.bind_closed_zero()
    @property
    def released(self): return self.impl.released
    @property
    def events(self): return self.impl.events
    def step(self, opening, physics_step, time_s):
        was=self.released
        self.impl.step(physics_step,time_s)
        return self.released and not was

class Geometry:
    """Basket contour, drawer axis, whole-basket outline and cake hulls from the composed scene (before physics)."""
    def __init__(self, stage, paths):
        TRAY, TARGET, OTHER, JOINT, BODY = (paths[k] for k in ('tray', 'target', 'other', 'joint', 'body'))
        from scipy.spatial import ConvexHull
        from pxr import Gf, UsdPhysics
        tray = body_colliders(stage, TRAY)
        floor = tray.get('/Collision/BasketFloor')
        if floor is None:
            raise ValueError('Missing active basket floor')
        centre = (floor.min(axis=0) + floor.max(axis=0))[:2] / 2
        inner, tops = [], []
        walls = [v for name, v in tray.items() if name.startswith('/Collision/MatchedBasket/')]
        if not walls:
            raise ValueError('Missing approved inner basket contour')
        for v in walls:
            xy = np.unique(v[:, :2], axis=0)
            if len(xy) != 4:
                raise ValueError('Unexpected basket wall topology')
            inner.extend(xy[np.argsort(((xy - centre) ** 2).sum(axis=1))[:2]])
            tops.append(v[:, 2].max())
        if len(inner) < 6:
            raise ValueError('Incomplete basket contour')
        self.planes = ConvexHull(np.asarray(inner)).equations
        self.bottom, self.top = float(floor[:, 2].min()), float(min(tops))
        every = np.vstack(list(tray.values()))
        self.tray_hull = every[ConvexHull(every).vertices]          # 3D hull of the whole basket (handle included), tray frame
        self.cake_hull = {}
        for root in (TARGET, OTHER):
            v = np.vstack(list(body_colliders(stage, root).values()))
            if not np.isfinite(v).all():
                raise ValueError('Missing cake geometry')
            self.cake_hull[root] = v[ConvexHull(v).vertices]
        joint = stage.GetPrimAtPath(JOINT)
        if (not joint or not joint.IsA(UsdPhysics.Joint) or joint.GetAttribute('limit:transX:physics:low').Get() != 0
                or [str(p) for p in joint.GetRelationship('physics:body0').GetTargets()] != [BODY]
                or [str(p) for p in joint.GetRelationship('physics:body1').GetTargets()] != [TRAY]):
            raise ValueError('Unexpected physical drawer joint')
        self.joint_limit_high = float(joint.GetAttribute('limit:transX:physics:high').Get())
        self.joint_enabled = joint.GetAttribute('physics:jointEnabled').Get() is not False
        self.p0 = np.asarray(joint.GetAttribute('physics:localPos0').Get(), dtype=float)
        self.p1 = np.asarray(joint.GetAttribute('physics:localPos1').Get(), dtype=float)
        q = joint.GetAttribute('physics:localRot0').Get()
        self.axis = np.asarray(q.Transform(Gf.Vec3f(1, 0, 0)), dtype=float)

    def opening(self, body, tray):
        bp, bq = body
        tp, tq = tray
        br, tr = rotation(bq), rotation(tq)
        return float(np.dot(np.asarray(tp) + tr.apply(self.p1) - np.asarray(bp) - br.apply(self.p0), br.apply(self.axis)))

    def inside(self, cake, tray_pose, cake_pose):
        tp, tq = tray_pose
        cp, cq = cake_pose
        local = rotation(tq).inv().apply(rotation(cq).apply(self.cake_hull[cake]) + np.asarray(cp) - np.asarray(tp))
        return bool(np.all(local[:, :2] @ self.planes[:, :2].T + self.planes[:, 2] <= 1e-7)
                    and local[:, 2].min() >= self.bottom - 1e-7 and local[:, 2].max() <= self.top + 1e-7)

    def outside(self, cake, tray_pose, cake_pose):
        tp, tq = tray_pose
        cp, cq = cake_pose
        tray_xy = hull2d(rotation(tq).apply(self.tray_hull) + np.asarray(tp))
        cake_xy = hull2d(rotation(cq).apply(self.cake_hull[cake]) + np.asarray(cp))
        return polygons_separate(tray_xy, cake_xy)

    def description(self):
        return dict(source='collision geometry of the composed scene', basket_floor_z_m=self.bottom, basket_wall_top_z_m=self.top,
                    inner_contour_planes=len(self.planes), basket_hull_vertices=len(self.tray_hull),
                    cake_hull_vertices={k.rsplit('/', 1)[-1]: len(v) for k, v in self.cake_hull.items()},
                    joint_limit_high_m=None if math.isinf(self.joint_limit_high) else self.joint_limit_high, joint_enabled_at_start=self.joint_enabled,
                    drawer_axis_body_frame=self.axis.tolist())


class NativeAirFryer:
    provenance = 'native_physx'
    def __init__(self, stage, manifest, output, episode_id, instance_id, variant):
        self.stage=stage; self.paths=bind_room(stage,manifest,variant)
        self.variant=variant; self.output=Path(output)
        self.output.mkdir(exist_ok=False)
        self.binding=Binding(episode_id,instance_id,tuple(sorted({
            'fryer':self.paths['fryer'],'basket':self.paths['tray'],
            'target':self.paths['target'],'other':self.paths['other'],
            'plate':self.paths['plate'],'counter':self.paths['counter']}.items())))
        self.geometry=Geometry(stage,self.paths)
        if not self.geometry.joint_enabled: raise ValueError('Initial rail disabled')
        if abs(self.geometry.joint_limit_high-.146)>1e-7: raise ValueError('Wrong all_assets rail')
        self.cakes=(self.paths['target'],self.paths['other'])
        self.config=DetectionConfig()
        self.detector=GraspDetector((self.paths['target'],),self.config)
        self.step=-1; self.start_step=None; self.initial_position=None
        self.latest=None; self.closed=False; self.diag=None; self.rail=None
        self.history=deque(maxlen=int(math.ceil(INITIAL_WINDOW_S/PHYSICS_DT-1e-9))+2)
        self.reset_accumulators()
        self.trace=(self.output/'physical_facts.jsonl').open('x')
        self.frames=(self.output/'control_facts.jsonl').open('x')
        self.consumer=None
        self.hashes={l.realPath:hashlib.sha256(Path(l.realPath).read_bytes()).hexdigest()
                     for l in stage.GetUsedLayers() if l.realPath}
        self.hashes[str(BEHAVIOR)]=hashlib.sha256(BEHAVIOR.read_bytes()).hexdigest()
        self.instrumentation=instrument(stage,self.paths)

    def initialize(self, world, simulation_view, *, allow_idle_60hz=False):
        if hasattr(self,'world'): raise RuntimeError('Bind once; no phase resets')
        self.allow_idle_60hz = allow_idle_60hz
        self.contiguous_240_steps = 0
        allowed = (PHYSICS_DT, 1/60) if allow_idle_60hz else (PHYSICS_DT,)
        if not any(abs(world.get_physics_dt()-dt)<1e-10 for dt in allowed):
            raise ValueError('Expected approved scene clock')
        from pxr import PhysicsSchemaTools
        from omni.physx import get_physx_simulation_interface
        from omni.physx.bindings._physx import ContactEventType
        self.world=world; self.last_time=float(world.current_time)
        self.interface=get_physx_simulation_interface()
        self.views={p:NativeBody(simulation_view,p) for p in (*self.cakes,self.paths['tray'],self.paths['body'],*HANDS.values())}
        def sleeping(stage_id,actor):
            p=self.stage.GetPrimAtPath(PhysicsSchemaTools.intToSdfPath(actor))
            if not p: raise RuntimeError('Unknown contact actor')
            if not p.HasAPI(UsdPhysics.RigidBodyAPI) or p.GetAttribute('physics:kinematicEnabled').Get() is True: return True
            return bool(self.interface.is_sleeping(stage_id,actor))
        self.contacts=Contacts(PhysicsSchemaTools.intToSdfPath,sleeping,
            (ContactEventType.CONTACT_FOUND,ContactEventType.CONTACT_PERSIST,ContactEventType.CONTACT_LOST),
            roots=(*self.cakes,self.paths['tray']))
        self.rail=AssetRail(self.stage,self.paths,self.views)

    def after_physics(self, helper_report):
        if helper_report['events'] or helper_report['docked_object_count'] or helper_report['tray_dock_enabled']:
            raise RuntimeError('Manipulation helper invalidates evidence')
        dt = self.world.get_physics_dt()
        if abs(dt-PHYSICS_DT)>1e-10 and not (self.allow_idle_60hz and abs(dt-1/60)<1e-10
                and (self.consumer is None or self.consumer.termination is not None)):
            raise RuntimeError('Active fryer needs existing 240Hz clock')
        self.after_step()

    def accept_initial(self):
        if self.consumer is not None or self.start_step is not None:
            raise RuntimeError('Initial acceptance once only')
        rows=list(self.history); p=self.paths; now=self.latest
        if not rows or self.contiguous_240_steps < round(1/PHYSICS_DT): raise RuntimeError('Need original 1s settling at 240Hz')
        span=lambda obj: max(math.dist(a['poses'][obj]['position_m'],b['poses'][obj]['position_m']) for a in rows for b in rows)
        checks=dict(history_complete=len(rows)==self.history.maxlen,
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
        self.consumer=AirFryerComposite(self.binding,self.variant)
        return self.score_control()

    def score_control(self):
        if self.consumer is None: raise RuntimeError('Not physically accepted')
        f=self.frame()
        return self.consumer.observe(Frame(self.binding,f))

    def reset_accumulators(self):
        TARGET, OTHER, TRAY, BODY, JOINT, BASKET_FLOOR, PLATE, PLATE_SUPPORT, COUNTER, FLOOR, ROBOT = (self.paths[k] for k in ('target', 'other', 'tray', 'body', 'joint', 'basket_floor', 'plate', 'plate_support', 'counter', 'floor', 'robot'))
        self.window = dict(ground={o: False for o in (TARGET, OTHER, TRAY)}, basket={o: False for o in self.cakes}, wall={o: False for o in self.cakes}, tray_body=False)

    def summarize(self):
        TARGET, OTHER, TRAY, BODY, JOINT, BASKET_FLOOR, PLATE, PLATE_SUPPORT, COUNTER, FLOOR, ROBOT = (self.paths[k] for k in ('target', 'other', 'tray', 'body', 'joint', 'basket_floor', 'plate', 'plate_support', 'counter', 'floor', 'robot'))
        result = {o: dict(robot=False, supports=set(), ground=False, basket=False, wall=False) for o in self.cakes}
        tray = dict(robot=False, body=False, counter=False, ground=False)
        grips = {o: {h: [[0., 0., 0.], [0., 0., 0.]] for h in PADS} for o in (*self.cakes, TRAY)}
        touched = {o: {h: [False, False] for h in PADS} for o in (*self.cakes, TRAY)}
        for paths, samples in self.contacts.pairs():
            for obj in self.cakes:
                index = 0 if under(paths[0], obj) else (1 if under(paths[1], obj) else None)
                if index is None:
                    continue
                other = paths[1 - index]
                sign = 1 if index == 0 else -1
                up = any(sign * n[2] > UP_NORMAL_Z for n, _ in samples)
                flags = result[obj]
                if under(other, ROBOT):
                    flags['robot'] = True
                    for hand, pads in PADS.items():
                        for k, pad in enumerate(pads):
                            if under(other, pad):
                                touched[obj][hand][k] = True
                                for n, _ in samples:
                                    for a in range(3):
                                        grips[obj][hand][k][a] += sign * n[a]
                elif under(other, TRAY):
                    flags['basket'] = True
                    flags['wall'] |= not under(other, BASKET_FLOOR)
                    if up:
                        flags['supports'].add(ON_FLOOR if under(other, BASKET_FLOOR) else ON_WALL)
                elif under(other, BODY):
                    if up:
                        flags['supports'].add(ON_BODY)
                elif under(other, FLOOR):
                    flags['ground'] = True
                    if up:
                        flags['supports'].add(ON_GROUND)
                elif under(other, PLATE) or under(other, PLATE_SUPPORT):
                    if up:
                        flags['supports'].add(ON_PLATE)
                elif up:
                    flags['supports'].add(ON_COUNTER if under(other, COUNTER) else ON_OTHER)
            index = 0 if under(paths[0], TRAY) else (1 if under(paths[1], TRAY) else None)
            if index is not None and not any(under(paths[1 - index], c) for c in self.cakes):
                other = paths[1 - index]
                sign = 1 if index == 0 else -1
                if under(other, ROBOT):
                    tray['robot'] = True
                    for hand, pads in PADS.items():
                        for k, pad in enumerate(pads):
                            if under(other, pad):
                                touched[TRAY][hand][k] = True
                                for n, _ in samples:
                                    for a in range(3):
                                        grips[TRAY][hand][k][a] += sign * n[a]
                elif under(other, BODY):
                    tray['body'] = True
                elif under(other, COUNTER):
                    tray['counter'] |= any(sign * n[2] > COUNTER_SUPPORT_NORMAL_Z for n, _ in samples)
                elif under(other, FLOOR):
                    tray['ground'] = True
        bilateral = {o: {h: bool(all(touched[o][h]) and sum(a * b for a, b in zip(*grips[o][h])) < 0) for h in PADS} for o in touched}
        return result, tray, bilateral

    def after_step(self):
        TARGET, OTHER, TRAY, BODY, JOINT, BASKET_FLOOR, PLATE, PLATE_SUPPORT, COUNTER, FLOOR, ROBOT = (self.paths[k] for k in ('target', 'other', 'tray', 'body', 'joint', 'basket_floor', 'plate', 'plate_support', 'counter', 'floor', 'robot'))
        if self.closed:
            raise RuntimeError('Reader closed')
        now = float(self.world.current_time)
        dt = float(self.world.get_physics_dt())
        if abs(now - self.last_time - dt) > 1e-6:
            raise RuntimeError('Missing physical substep')
        if abs(dt-PHYSICS_DT)<1e-10:
            self.contiguous_240_steps += 1
        else:
            self.contiguous_240_steps = 0
            self.history.clear()
        poses = {}
        for path, view in self.views.items():
            p, q = view.get_world_pose()
            if np.asarray(p).shape != (3,) or not np.isfinite(p).all():
                raise ValueError('Invalid physical position')
            rotation(q)
            poses[path] = dict(position_m=[float(v) for v in p], quaternion_wxyz=[float(v) for v in q])
        self.contacts.ingest(self.interface.get_contact_report())
        if self.diag is not None:
            rows = [dict(a=paths[0], b=paths[1], impulse=max(s for _, s in samples), n=len(samples))
                    for paths, samples in self.contacts.pairs() if any(under(p, ROBOT) for p in paths)]
            if rows:
                self.diag.write(json.dumps(dict(step=self.step + 1, time_s=now, pairs=rows)) + '\n')
        contact, tray, bilateral = self.summarize()
        grasped, grasp_details = self.detector.observe(now, poses, {TARGET: bilateral[TARGET]})
        body_pose = (poses[BODY]['position_m'], poses[BODY]['quaternion_wxyz'])
        tray_pose = (poses[TRAY]['position_m'], poses[TRAY]['quaternion_wxyz'])
        opening = self.geometry.opening(body_pose, tray_pose)
        released_now = self.rail.step(opening, self.step + 1, now) if self.rail is not None else False
        for obj in self.cakes:
            self.window['ground'][obj] |= contact[obj]['ground']
            self.window['basket'][obj] |= contact[obj]['basket']
            self.window['wall'][obj] |= contact[obj]['wall']
        self.window['ground'][TRAY] |= tray['ground']
        self.window['tray_body'] |= tray['body']
        self.step += 1
        self.latest = dict(
            time_s=now, reader_step=self.step, poses=poses,
            cakes={o: dict(robot=contact[o]['robot'], supports=sorted(contact[o]['supports']), ground=contact[o]['ground'], basket=contact[o]['basket'], wall=contact[o]['wall'],
                           bilateral=bilateral[o], grasped=bool(grasped.get(o, False)), tilt_deg=self.tilt(poses[o]['quaternion_wxyz']),
                           inside=self.geometry.inside(o, tray_pose, (poses[o]['position_m'], poses[o]['quaternion_wxyz'])),
                           outside=self.geometry.outside(o, tray_pose, (poses[o]['position_m'], poses[o]['quaternion_wxyz']))) for o in self.cakes},
            tray=dict(opening_m=opening, robot=tray['robot'], body=tray['body'], counter=tray['counter'], ground=tray['ground'], bilateral=bilateral[TRAY],
                      tilt_deg=self.tilt(poses[TRAY]['quaternion_wxyz']), joint_released=bool(self.rail.released) if self.rail else (not self.geometry.joint_enabled),
                      released_now=released_now),
            grasp_details=grasp_details)
        self.history.append(self.latest)
        self.trace.write(json.dumps(serializable(self.latest), allow_nan=False) + '\n')
        self.last_time = now

    @staticmethod
    def tilt(quaternion):
        up = rotation(quaternion).apply([0., 0., 1.])
        return float(np.degrees(np.arccos(np.clip(up[2], -1., 1.))))

    def frame(self):
        TARGET, OTHER, TRAY, BODY, JOINT, BASKET_FLOOR, PLATE, PLATE_SUPPORT, COUNTER, FLOOR, ROBOT = (self.paths[k] for k in ('target', 'other', 'tray', 'body', 'joint', 'basket_floor', 'plate', 'plate_support', 'counter', 'floor', 'robot'))
        """Facts of the control frame that ends at the latest substep."""
        if self.start_step is None:
            raise RuntimeError('Initial state not accepted')
        delta = self.step - self.start_step
        if delta % SUBSTEPS:
            raise RuntimeError('Control frame is not aligned with the physics substeps')
        row, window = self.latest, self.window
        cakes = {}
        for obj in self.cakes:
            c, pose = row['cakes'][obj], row['poses'][obj]
            cakes[obj] = CakeFacts(
                position=tuple(pose['position_m']), quaternion=tuple(pose['quaternion_wxyz']), tilt_deg=c['tilt_deg'],
                pads_bilateral=bool(any(c['bilateral'].values()) and c['robot']), grasped=bool(c['grasped'] and c['robot']),
                robot_contact=bool(c['robot']), supports=frozenset(c['supports']), ground_contact=bool(window['ground'][obj] or c['ground']),
                basket_contact=bool(window['basket'][obj] or c['basket']), inside_basket=bool(c['inside']), outside_basket=bool(c['outside']),
                displacement_m=float(np.linalg.norm(np.asarray(pose['position_m']) - self.initial_position[obj])))
        t, pose = row['tray'], row['poses'][TRAY]
        tray = TrayFacts(position=tuple(pose['position_m']), quaternion=tuple(pose['quaternion_wxyz']), tilt_deg=t['tilt_deg'], opening_m=float(t['opening_m']),
                         body_contact=bool(t['body']), counter_supported=bool(t['counter']), robot_contact=bool(t['robot']),
                         pads_bilateral=bool(any(t['bilateral'].values()) and t['robot']), ground_contact=bool(window['ground'][TRAY] or t['ground']),
                         joint_released=bool(t['joint_released']),
                         displacement_m=float(np.linalg.norm(np.asarray(pose['position_m']) - self.initial_position[TRAY])))
        index = delta // SUBSTEPS
        facts = Facts(step=index, time_s=index * CONTROL_DT, target=cakes[TARGET], other=cakes[OTHER], tray=tray)
        self.reset_accumulators()
        self.frames.write(json.dumps(serializable(dict(facts.as_dict(), simulation_time_s=row['time_s'], reader_step=self.step)), allow_nan=False) + '\n')
        self.frames.flush()
        self.trace.flush()
        return facts

    def begin_episode(self):
        TARGET, OTHER, TRAY, BODY, JOINT, BASKET_FLOOR, PLATE, PLATE_SUPPORT, COUNTER, FLOOR, ROBOT = (self.paths[k] for k in ('target', 'other', 'tray', 'body', 'joint', 'basket_floor', 'plate', 'plate_support', 'counter', 'floor', 'robot'))
        if self.start_step is not None or self.latest is None:
            raise RuntimeError('Invalid initial acceptance')
        self.start_step = self.step
        self.initial_position = {o: np.asarray(self.latest['poses'][o]['position_m'], dtype=float) for o in (*self.cakes, TRAY)}
        self.reset_accumulators()

    def close(self):
        TARGET, OTHER, TRAY, BODY, JOINT, BASKET_FLOOR, PLATE, PLATE_SUPPORT, COUNTER, FLOOR, ROBOT = (self.paths[k] for k in ('target', 'other', 'tray', 'body', 'joint', 'basket_floor', 'plate', 'plate_support', 'counter', 'floor', 'robot'))
        if not self.closed:
            self.trace.close()
            self.frames.close()
            if self.diag is not None:
                self.diag.close()
            if self.rail is not None:
                (self.output / 'rail_release.json').write_text(json.dumps(dict(released=self.rail.released, events=self.rail.events), indent=2) + '\n')
            self.closed = True
