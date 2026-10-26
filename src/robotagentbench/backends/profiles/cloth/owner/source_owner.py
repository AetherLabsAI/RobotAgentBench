"""Original-source physical owner plus the already approved cloth facts/rules.

No simulator/robot/cloth creation, rendering, state writes or parameter changes.
The source initializer supplies its own live objects and its exact substep.
The first validation mode is deliberately camera-free, not a model benchmark.
It cannot claim that the separate image synchronization issue is resolved.
"""
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import numpy as np

from control_session import ClothControlSession
from native_reader import NativeSample
from source_controls import SourceControls


def accepted_session_result(session):
    result = session.result()
    if (not session.closed or session.faulted or result.get('status') != 'finished'
            or result.get('evaluation_valid') is not True
            or result.get('scoring_available') is not True):
        raise RuntimeError('Native session did not close with valid physical evidence')
    return result


class OriginalOwner:
    def __init__(self, *, output, task_name, prepared, setting, policy, protocol,
                 source_root, diagnostic_controls=2, source_commands=None, session_handler=None,
                 camera_regression=False, isolated_camera_regression=False, sdk_site=None):
        if task_name not in ('collect_tablecloth', 'spread_tablecloth', 'long_task_18'):
            raise ValueError('Explicit approved cloth task required')
        if type(diagnostic_controls) is not int or not 1 <= diagnostic_controls <= 600:
            raise ValueError('Bounded diagnostic action source required')
        self.output = Path(output)
        self.task_name, self.prepared, self.setting = task_name, prepared, setting
        self.policy, self.protocol, self.source_root = policy, protocol, Path(source_root)
        self.control_count, self.source_commands = diagnostic_controls, source_commands
        if session_handler is not None and not callable(session_handler):
            raise ValueError('Session handler must be a trusted owner callback')
        self.session_handler = session_handler
        if type(camera_regression) is not bool:
            raise ValueError('Trusted explicit camera regression mode required')
        self.camera_regression, self.cameras = camera_regression, None
        if type(isolated_camera_regression) is not bool or (camera_regression and isolated_camera_regression):
            raise ValueError('Select at most one explicit observation owner')
        self.isolated_camera_regression, self.sdk_site = isolated_camera_regression, sdk_site
        self.observation_regression = camera_regression or isolated_camera_regression
        self.episode_id = str(uuid4())
        self.session = None
        self.fact_readers = {}
        self.report = dict(schema='robotagentbench.original_cloth_owner.v18',
            task_name=task_name, episode_id=self.episode_id,
            status='starting', native_initial_state_accepted=False,
            scorer_executed=False, scoring_available=False, agent_benchmark_result=False,
            physical_parameters_changed=False, scoring_changed=False,
            original_source_init_used=True, original_substep_used=True,
            fixed_world_anchor_preserved=True, post_start_state_writes=0,
            camera_resources_created=False, image_sync_accepted=False,
            public_rpc_accepted=False, native_shutdown_complete=False)

    def prepare_observations(self, stage, app):
        if not self.camera_regression:
            raise RuntimeError('Unexpected camera setup in camera-free source regression')
        from source_cameras import SourceCameras
        try:
            self.cameras = SourceCameras(stage=stage, app=app, output=self.output/'observations')
        except BaseException as exc:
            self.report.update(status='error', evaluation_valid=False, score=None,
                observation_setup_error=repr(exc))
            self.save()
            raise
        self.report.update(camera_resources_created=True,
            camera_resources_created_before_physics=True,
            camera_prewarm_updates=self.cameras.prewarm_updates)
        self.save()

    def save(self):
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output/'source_owner_report.json').write_text(
            json.dumps(self.report, indent=2, allow_nan=False)+'\n')

    def _clock(self):
        sim = self.native['sim']
        index, raw = int(sim.current_time_step_index), float(sim.current_time)
        if abs(raw-index/60) > 4*np.finfo(np.float32).eps*max(1., abs(raw)):
            raise ValueError('Original measured clock is not 60Hz')
        return index, index/60

    def _snapshot(self):
        n = self.native
        before = self._clock()
        chunks = [v.get_positions().detach().cpu().numpy().reshape(-1, 3).copy()
                  for v in n['cloth_views']]
        velocities = [v.get_velocities().detach().cpu().numpy().reshape(-1, 3).copy()
                      for v in n['cloth_views']]
        joints = n['art'].get_joint_positions().detach().cpu().numpy().copy()
        jd = n['art'].get_joint_velocities().detach().cpu().numpy().copy()
        if (any(q.shape != (count, 3) for q, count in zip(chunks, n['particle_counts']))
                or any(q.shape != v.shape for q, v in zip(chunks, velocities))
                or any(not np.isfinite(x).all() for x in [*chunks, *velocities, joints, jd])):
            raise ValueError('Invalid original native readback, no stale-state substitution')
        art_view = n['art']._articulation_view
        body_pose = art_view._physics_view.get_link_transforms().detach().cpu().numpy()[0].copy()
        body_velocity = art_view._physics_view.get_link_velocities().detach().cpu().numpy()[0].copy()
        root_pose = art_view._physics_view.get_root_transforms().detach().cpu().numpy()[0].copy()
        if not all(np.isfinite(x).all() for x in (body_pose, body_velocity, root_pose)):
            raise ValueError('Invalid original native articulation transforms')
        if self._clock() != before:
            raise RuntimeError('Read-only snapshot advanced physics')
        return dict(physics_step=before[0], time_s=before[1], positions=np.concatenate(chunks),
            velocities=np.concatenate(velocities), joints=joints, joint_velocities=jd,
            body_names=list(art_view.body_names), body_pose=body_pose,
            body_velocity=body_velocity, root_pose=root_pose)

    def _advance(self, targets):
        before = self._clock()[0]
        self.advance_original(targets)
        if self._clock()[0] != before+1:
            raise RuntimeError('Original substep must advance precisely once')
        self.current = self._snapshot()
        return self.current

    def _source_controls(self):
        from pxr import UsdPhysics
        from robotagentbench.interfaces import robot_profiles as rp
        n = self.native
        units, limits, active = {}, {}, []
        for name in n['names']:
            prim = n['stage'].GetPrimAtPath('/World/Robot/joints/'+name)
            if not prim:
                raise ValueError('Original joint identity missing: '+name)
            linear = prim.IsA(UsdPhysics.PrismaticJoint)
            axis = 'linear' if linear else 'angular'
            units[name] = 'm' if linear else 'rad'
            factor = 1 if linear else math.pi/180
            if prim.HasAPI(UsdPhysics.DriveAPI, axis):
                active.append(name)
            lo, hi = (prim.GetAttribute('physics:'+part+'Limit').Get()
                      for part in ('lower', 'upper'))
            if lo is not None and hi is not None:
                limits[name] = (float(lo)*factor, float(hi)*factor)
        frozen = dict(zip(n['command_names'], map(float, n['action']['joint_target'][0])))
        adapter = rp.NativeAdapter(rp.MOBILE22, n['names'], units, active,
            {name: frozen[name] for name in active}, limits,
            mimic_followers=rp.inspect_mimic_followers(n['stage']))
        adapter.followers = dict(rp.FOLLOWERS)
        return SourceControls(command_names=n['command_names'], driven=n['driven'],
            initial_targets=n['action']['joint_target'][0], adapter=adapter)

    def _bind_facts(self, history):
        from approved_rules import Binding
        from composition import ClothComposition
        from contact_geometry import NativePadGeometry, surface_proximity
        from facts_reader import ClothFactsReader
        from mesh import split_materials
        from native_geometry import (cooked_pad_geometry, table_geometry,
                                     cooked_table_geometry, floor_top_triangles)
        from placement import BasketProfile, convex_polygon, material_stability
        from solids import StaticSolid
        n = self.native
        self.materials = split_materials(dict(initial_cloth_q=n['initial'],
            uv=n['uv'], faces=n['faces'], cloth_id=n['cloth_ids']))
        # Geometry bindings are read-only and never install/replace a collider.
        pads = {p: NativePadGeometry(rows) for p, rows in cooked_pad_geometry(n['stage']).items()}
        table_info = table_geometry(n['stage'])
        table = StaticSolid(path='/World/Assets/CoffeeTable', pieces_world_m=cooked_table_geometry(n['stage']))
        basket_source = json.loads((self.source_root/
            'code/tablecloth_workflow/native_teleop/lay_20260924/basket_profile_split_colliders.json').read_text())
        basket = StaticSolid(path='/World/Assets/TableclothBasket',
            pieces_world_m=[r['world_vertices_m'] for r in basket_source['colliders']])
        rim = max(p['rows'][-1][0] for p in basket_source['profiles'])
        basket_profile = BasketProfile(basket_source, rim_z=rim)
        floor_triangles = floor_top_triangles(n['stage'])
        floor_contact = lambda q: surface_proximity(q, floor_triangles,
            shell_m=self.policy['reused_source_detection']['support_face_tolerance_m'])
        vertices = np.asarray(table_info['world_vertices_m'])
        top = table_info['top_z_m']
        polygon = convex_polygon(vertices[abs(vertices[:, 2]-top) < 1e-9, :2])
        self.targets = ({'old': 0} if self.task_name == 'collect_tablecloth' else
            {'new': 1} if self.task_name == 'spread_tablecloth' else {'old': 0, 'new': 1})
        bindings, facts, stability = {}, {}, {}
        for label, cid in self.targets.items():
            mesh = self.materials[cid][0]
            mask = n['cloth_ids'] == cid
            stability[label] = material_stability(mesh,
                np.stack([s['positions'][mask] for s in history]),
                position_span_m=self.policy['cloth_stability']['position_span_m'],
                area_fraction_min=self.policy['cloth_stability']['material_area_fraction_min'])
            if not stability[label]['stable']:
                raise ValueError('Original cloth initial stability failed: '+label)
            kind = 'collect_tablecloth' if label == 'old' else 'spread_tablecloth'
            src, dst = (table, basket) if label == 'old' else (basket, table)
            path = f'/World/TableclothPhysics/Cloth_{cid}'
            binding = Binding(kind, self.episode_id, path, src.path, dst.path)
            bindings[label] = binding
            reader = SimpleNamespace(episode_id=self.episode_id, body_path=path,
                                     mesh=mesh, close=lambda: None)
            self.fact_readers[label] = ClothFactsReader(binding=binding,
                native_reader=reader, source_solid=src, destination_solid=dst,
                table_z=top, table_polygon_xy=polygon, basket_profile=basket_profile,
                pad_geometry=pads, policy=self.policy, floor_contact=floor_contact,
                initial_source_crease=(self.setting['clean_initial_recipe']['fold_material_v_m']
                                       if label == 'new' else None))
            facts[label] = self.fact_readers[label].start(sample=self._sample(cid), body_poses=self._poses())
        score = ClothComposition(task_name=self.task_name, episode_id=self.episode_id,
            old_binding=bindings.get('old'), new_binding=bindings.get('new'))
        score.observe(facts)
        self.report.update(initial_stability=stability, initial_facts={k: asdict(v) for k, v in facts.items()})
        return score

    def _sample(self, cid):
        mesh = self.materials[cid][0]
        mask = self.native['cloth_ids'] == cid
        return NativeSample(self.episode_id, f'/World/TableclothPhysics/Cloth_{cid}',
            mesh.identity, self.current['physics_step'], self.current['time_s'],
            self.current['positions'][mask].copy(), self.current['velocities'][mask].copy())

    def _poses(self):
        required = {f'/World/Robot/gripper_{h}_{s}_link4' for h in ('l', 'r') for s in ('inner', 'outer')}
        required |= {f'/World/Robot/gripper_{h}_base_link' for h in ('l', 'r')}
        return {path: self.current['body_pose'][self.current['body_names'].index(path.rsplit('/', 1)[1])].copy()
                for path in required}

    def _facts(self):
        return {label: self.fact_readers[label].observe(sample=self._sample(cid), body_poses=self._poses())
                for label, cid in self.targets.items()}

    def run(self, native, advance_original):
        self.native, self.advance_original = native, advance_original
        try:
            self.report['status'] = 'original_initialization_complete'
            self.save()
            if native['fps'] != 20 or native['args'].substeps != 3 or native['source_frequency'] != 60:
                raise ValueError('Original 60Hz/20Hz source contract required')
            for key, source_key in (('initial_cloth_q', 'initial'), ('uv', 'uv'),
                                    ('faces', 'faces'), ('cloth_id', 'cloth_ids')):
                if not np.array_equal(self.prepared[key], native[source_key]):
                    raise ValueError('Original source initialization differs from prepared input: '+key)
            if (list(map(str, self.prepared['joint_names'])) != native['command_names']
                    or not np.array_equal(self.prepared['initial_joint_targets'], native['action']['joint_target'][0])):
                raise ValueError('Original source frozen commands differ')
            if self.protocol['settle_seconds'] != 3.:
                raise ValueError('Original protocol settling contract required')
            controls = self._source_controls()
            frozen = controls.source_targets(native['action']['joint_target'][0])
            history = []
            self.current = self._snapshot()
            self.report['status'] = 'settling_original_source'
            self.save()
            for _ in range(int(self.protocol['settle_seconds']*60)):
                history.append(self._advance(frozen))
                history = history[-16:]  # 15 intervals = the approved 0.25s window.
            self.origin = self.current['physics_step']
            self.report['status'] = 'binding_approved_facts'
            self.save()
            score = self._bind_facts(history)
            # Read-only binding must not change the source's live state or clock.
            verified = self._snapshot()
            for key in ('positions', 'velocities', 'joints', 'joint_velocities'):
                if not np.array_equal(verified[key], self.current[key]):
                    raise RuntimeError('Read-only facts binding changed native state')
            owner = self
            class Clock:
                hz = 60
                @property
                def current_time(self):
                    return (owner._clock()[0]-owner.origin)/60
            instruction = (self.setting['instruction'] if self.task_name != 'long_task_18' else
                'Collect and fold the old tablecloth into the cloth basket, then spread the clean tablecloth over the coffee table and release it.')
            self.session = ClothControlSession(composition=score, clock=Clock(),
                read_native_facts=self._facts, instruction=instruction,
                initial_acceptance=dict(native_initial_state_accepted=True, source_profile_verified=True))
            self.report.update(native_initial_state_accepted=True, scorer_executed=True,
                scoring_available=True, settle_substeps=180, native_origin_step=self.origin,
                ready_physical_q_sha256=hashlib.sha256(self.current['positions'].tobytes()).hexdigest())
            if self.isolated_camera_regression:
                from source_visual_client import SourceVisualClient
                self.cameras = SourceVisualClient(owner=self, sdk_site=self.sdk_site)
                self.report.update(observation_owner='isolated_render_only_subprocess',
                    physical_process_render_calls=0, renderer_native_rigid_count=len(self.cameras.paths))
            if self.session_handler is not None:
                self.report['status']='source_native_session_admitted'
                self.save()  # Persist admission before any potentially long operator action source.
                self.session_handler(self, controls)
                final = accepted_session_result(self.session)
                self.report.update(status='source_native_session_closed',
                    final_result=final, native_session_transport_accepted=True,
                    public_rpc_accepted=False)
                self.save()
                return self.report
            initial = self.current
            if self.observation_regression:
                if self.cameras is None:
                    raise RuntimeError('Required native observation owner is missing')
                self.cameras.capture(self)
            samples, fact_rows, executed = [], [], 0
            commands = (np.tile(native['action']['joint_target'][0], (self.control_count, 1))
                        if self.source_commands is None else np.asarray(self.source_commands))
            if commands.ndim != 2 or not 1 <= len(commands) <= 600:
                raise ValueError('Choose a bounded explicit diagnostic fragment')
            # Validate the entire chosen fragment before the first drive write.
            validated = [controls.source_targets(row) for row in commands]
            for targets in validated:
                self.session.begin_control()
                for _ in range(3):
                    samples.append(self._advance(targets))
                    self.session.after_physics(dict(events=[], docked_object_count=0, tray_dock_enabled=False))
                    fact_rows.append({k: asdict(v) for k, v in self._facts().items()})
                self.session.finish_control(completed_substeps=3)
                executed += 1
                if self.observation_regression:
                    self.cameras.capture(self)
                if self.session.result()['status'] != 'running':
                    break
            self.session.close()
            from placement import material_stability
            sanity = {}
            if self.source_commands is None:
                for cid, (mesh, _) in self.materials.items():
                    mask = native['cloth_ids'] == cid
                    sanity[str(cid)] = material_stability(mesh,
                        np.stack([initial['positions'][mask], *[row['positions'][mask] for row in samples]]),
                        position_span_m=self.policy['cloth_stability']['position_span_m'],
                        area_fraction_min=self.policy['cloth_stability']['material_area_fraction_min'])
            np.savez_compressed(self.output/'source_physical_fragment.npz',
                positions=np.stack([initial['positions'], *[r['positions'] for r in samples]]),
                velocities=np.stack([initial['velocities'], *[r['velocities'] for r in samples]]),
                joints=np.stack([initial['joints'], *[r['joints'] for r in samples]]),
                physics_step=np.array([initial['physics_step'], *[r['physics_step'] for r in samples]]),
                uv=native['uv'], faces=native['faces'], cloth_id=native['cloth_ids'],
                body_pose=np.stack([initial['body_pose'], *[r['body_pose'] for r in samples]]),
                body_names=initial['body_names'])
            self.report.update(status='source_native_fragment_complete', diagnostic_controls_executed=executed,
                final_result=self.session.result(), facts=fact_rows, physical_substeps=len(samples),
                native_final_time_s=self.session.clock.current_time,
                guarded_camera_captures=self.cameras.count if self.cameras else 0,
                native_hold_sanity=sanity,
                native_hold_sanity_accepted=bool(sanity and all(row['stable'] for row in sanity.values())),
                diagnostic_action_source=('unchanged_frozen_native34_holds' if self.source_commands is None
                                          else 'explicit_source_native34_fragment'),
                diagnostic_hold_duration_s=len(samples)/60,
                full_manipulation_success_claim=False, cameras_still_require_separate_validation=True)
            self.save()
            return self.report
        except BaseException as exc:
            if self.session is not None:
                self.session._fault(exc)
            self.report.update(status='error', scoring_available=False,
                evaluation_valid=False, score=None, error=repr(exc))
            self.save()
            raise

    def close(self):
        try:
            if self.session is not None:
                self.session.close()
            for reader in self.fact_readers.values():
                reader.close()
            if self.cameras is not None:
                self.cameras.close()
                if self.isolated_camera_regression:
                    self.report['render_process_shutdown_complete'] = self.cameras.shutdown_complete
        except BaseException as exc:
            self.report.update(status='error', cleanup_error=repr(exc))
            raise
        finally:
            self.save()

    def native_stopped(self):
        self.report['native_shutdown_complete'] = True
        self.save()

    def native_stop_failed(self, exc):
        self.report.update(native_shutdown_complete=False, native_stop_error=repr(exc))
        self.save()
