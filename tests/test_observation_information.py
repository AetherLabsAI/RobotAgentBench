"""Public observation tests; no simulator, policy invocation, or physical actions."""
import base64
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

try:
    import numpy as np
    import PIL
except ImportError:
    np = None

from robotagentbench.interfaces.action_contract import public_observation
from robotagentbench.interfaces.observation_extras import MAP_SCHEMA, MAP_SOURCE, public_navigation
from robotagentbench.runtime import action_host, coding_action_host
from robotagentbench.runtime.action_driver import NativeDriver as ActionDriver
from robotagentbench.runtime.coding_driver import NativeDriver as CodingDriver
from robotagentbench.runtime.navigation import ExploredMapObservation
from robotagentbench.runtime.observation_host import ObservationHost
from robotagentbench.simulation.admission import admit_prepared_launch
from robotagentbench.simulation.original_backend import bind_original_backend, ROOM_OWNER
from admission_fixtures import AdmissionFixture
from native_fixtures import make_host, png, request
from source_fixtures import SourceFixture


def packet(step):
    return dict(schema=MAP_SCHEMA, step=step, map_png=png(60, 60),
        origin_m=[-1., -1.], resolution_m=.04, shape=[20, 20], explored_fraction=.2,
        map_source=MAP_SOURCE, premapped=False, task_labels=False, body_footprint_radius_m=.3)


def map_fixture(directory):
    """Small deterministic grid and a measured-state head-camera model."""
    directory = Path(directory)
    shape = (80, 80)
    blocked = np.zeros(shape, dtype=bool)
    blocked[0,:] = blocked[-1,:] = blocked[:,0] = blocked[:,-1] = True
    # Wall limits visibility; the full room is never the returned image.
    blocked[:,55] = True
    height = np.where(blocked, 2.1, 0.).astype(np.float32)
    voxel_shape = (44, 65, 65)
    np.savez_compressed(directory/'maps.npz', blocked=blocked, height=height,
        floor=~blocked, voxels=np.packbits(np.zeros(voxel_shape, dtype=bool)),
        voxel_shape=voxel_shape, voxel_size=.05)
    (directory/'scene.json').write_text(json.dumps(dict(spawn=[0.,0.,0.], robot_radius_m=.2,
        map=dict(origin=[-1.6,-1.6], resolution=.04, shape=list(shape)))))
    local = np.array([[0.,0.,-1.,0.],[1.,0.,0.,0.],[0.,-1.,0.,1.5],[0.,0.,0.,1.]])
    (directory/'robot_model.json').write_text(json.dumps(dict(root_world=np.eye(4).tolist(),
        links=dict(base_link=np.eye(4).tolist()), joints=[], cameras=dict(Head_Camera=dict(
            link='base_link', local=local.tolist(), horizontal_aperture=20.,
            vertical_aperture=15., focal_length=20.)))))


class ObservationInformationTests(unittest.TestCase):
    def test_both_control_paths_return_context_and_map_without_control_steps(self):
        for module, driver in ((action_host, ActionDriver), (coding_action_host, CodingDriver)):
            with self.subTest(driver=driver.__name__):
                host, session = make_host(module.SessionHost, driver)
                host.driver.context = 'Alex prefers coffee. Keep the complete supplied memory.'
                calls = []
                def navigation(row):
                    calls.append(row['step'])
                    return packet(row['step'])
                host.driver.navigation = navigation
                before = session.clock.current_time
                ready = host.ready()
                self.assertEqual(ready['observation']['navigation'], packet(0))
                self.assertEqual(ready['observation']['context'], host.driver.context)
                reply = host.execute(request())
                self.assertEqual(reply['observation']['navigation']['step'], 1)
                self.assertEqual(reply['actions_executed'], 1)
                self.assertAlmostEqual(session.clock.current_time-before, .05)
                self.assertEqual(calls, [0, 1])

    def test_robot_only_reads_have_context_without_cameras_or_map(self):
        host, session = make_host(coding_action_host.SessionHost, CodingDriver)
        host.driver.context = 'Complete public memory'
        host.driver.navigation = lambda row: self.fail('Robot-only read rendered a map')
        before = deepcopy(session.trace)
        row = host.observe(robot_only=True)
        self.assertEqual(row['context'], host.driver.context)
        self.assertNotIn('images', row)
        self.assertNotIn('navigation', row)
        self.assertEqual(session.trace, before)

    def test_public_filter_keeps_only_exact_map_packet_and_context_text(self):
        host, _ = make_host(action_host.SessionHost, ActionDriver)
        row = host.observe()
        row.update(context='User preference', navigation=packet(row['step']),
                   answers={'target_pose': [1,2,3]}, task_scores=[99])
        filtered = public_observation(row)
        self.assertEqual(filtered['context'], 'User preference')
        self.assertEqual(filtered['navigation'], row['navigation'])
        self.assertNotIn('answers', filtered)
        self.assertNotIn('task_scores', filtered)
        filtered['navigation']['origin_m'][0] = 99
        self.assertEqual(row['navigation']['origin_m'], [-1.,-1.])
        row['context'] = {'memory': 'public', 'answer': 'not public'}
        with self.assertRaises(ValueError):
            public_observation(row)

    def test_malformed_stale_and_privileged_map_packets_are_rejected(self):
        for update in (dict(step=1), dict(premapped=True), dict(task_labels=True),
                       dict(full_grid=[]), dict(resolution_m=float('nan')),
                       dict(explored_fraction=1.1), dict(shape=[1,2]),
                       dict(map_png='not-a-png'), dict(target_pose=[1,2,3])):
            with self.subTest(update=list(update)), self.assertRaises(ValueError):
                public_navigation(dict(packet(0), **update), 0)

    def test_map_callback_may_not_advance_physics(self):
        host, session = make_host(action_host.SessionHost, ActionDriver)
        def stepping(row):
            session.clock.current_time += .01
            return packet(row['step'])
        host.driver.navigation = stepping
        with self.assertRaises(RuntimeError):
            host.observe()
        self.assertTrue(session.faulted)

    @unittest.skipIf(np is None, 'Map-render test requires the native numpy/Pillow dependencies')
    def test_original_exploration_is_partial_monotone_and_episode_local(self):
        with TemporaryDirectory() as directory:
            map_fixture(directory)
            provider = ExploredMapObservation(directory)
            host, session = make_host(action_host.SessionHost, ActionDriver)
            row = host.observe()
            a = provider(row)
            known = provider.explored.known.copy()
            self.assertGreater(a['explored_fraction'], 0.)
            self.assertLess(a['explored_fraction'], 1.)
            self.assertFalse(a['premapped'])
            self.assertFalse(a['task_labels'])
            self.assertEqual(provider(row), a)
            self.assertTrue(np.array_equal(known, provider.explored.known))
            self.assertEqual(session.clock.current_time, 0.)
            row2 = deepcopy(row)
            row2.update(step=1, time_s=.05)
            row2['base_state']['yaw_world_unwrapped_rad'] = np.pi
            row2['base_state']['orientation_world_xyzw'] = [0.,0.,1.,0.]
            b = provider(row2)
            self.assertGreaterEqual(b['explored_fraction'], a['explored_fraction'])
            self.assertNotEqual(b['map_png'], a['map_png'])
            with self.assertRaises(ValueError):
                provider(row)
            with self.assertRaises(ValueError):
                provider(dict(row2, episode_id='another-episode'))
            altered = deepcopy(row2)
            altered['joints'][next(iter(altered['joints']))] += .1
            with self.assertRaises(ValueError):
                provider(altered)

    def test_legacy_broker_filter_cannot_drop_public_information(self):
        raw, session = make_host(coding_action_host.SessionHost, CodingDriver)
        class Legacy:
            def __init__(self):
                self.session = session
            def ready(self):
                return dict(op='ready', contract={}, observation=raw.observe())
            def observe(self):
                return raw.observe()
            def exchange(self, command):
                reply = raw.exchange(command)
                if 'observation' in reply:
                    reply['observation'].pop('context', None)
                return reply
            def close(self):
                return raw.close()
        outer = ObservationHost(Legacy(), context='Full household memory',
                                navigation=lambda row: packet(row['step']))
        ready = outer.ready()
        self.assertIn('navigation', ready['contract']['observation_fields'])
        self.assertIn('context', ready['contract']['observation_fields'])
        before = len([r for r in session.trace if r[0] == 'capture'])
        reply = outer.exchange(request())
        self.assertEqual(reply['observation']['context'], 'Full household memory')
        self.assertEqual(reply['observation']['navigation']['step'], 1)
        self.assertEqual(len([r for r in session.trace if r[0] == 'capture'])-before, 1)
        self.assertNotIn('observation', outer.exchange({'op':'close'}))

    def test_admission_takes_context_only_from_trusted_backend_not_request(self):
        with TemporaryDirectory() as directory:
            f = AdmissionFixture(directory)
            f.request.write_text(json.dumps({'task_name':'test_task', 'context':'not-authoritative'}))
            f.backend = replace(f.backend, public_context=lambda launch, plan: 'Owner-selected memory')
            admitted = admit_prepared_launch(f.launch, f.output, backend=f.backend, workspace=f.workspace)
            self.assertEqual(admitted.context, 'Owner-selected memory')
            self.assertEqual(admitted.plan['score_policy'], {'maximum':17})
            self.assertFalse(f.output.exists())
            f.backend = replace(f.backend, public_context=lambda launch, plan: {'answer':'forbidden'})
            with self.assertRaises(ValueError):
                admit_prepared_launch(f.launch, f.output, backend=f.backend, workspace=f.workspace)

    def test_original_public_text_readers_are_used_without_private_bindings(self):
        with TemporaryDirectory() as directory:
            fixture = SourceFixture(directory)
            root = ROOM_OWNER.rsplit('.',1)[0]
            for suffix in ('.frozen_context.bundle', '.task_discovery_v1.inputs'):
                fixture.write((root+suffix).replace('.', '/')+'.py',
                    'def model_input(prompt_id):\n'
                    '    return {"prompt":"Scenario request", "context":"Complete original memory"}\n')
            with bind_original_backend(fixture.tree()) as backend:
                for name in ('openend', 'task_discovery'):
                    self.assertEqual(backend.public_context({'prompt_id':'test'}, {'task_name':name}),
                                     'Complete original memory')
                self.assertEqual(backend.public_context({}, dict(task_name='select_drink',
                    public_question={'context':'Diet context'}, target_bindings={'hidden':'not public'})),
                    'Diet context')
                self.assertEqual(backend.public_context({}, dict(task_name='select_trash',
                    selection={'public':{'context':'Classification rules'},'answer':'not public'})),
                    'Classification rules')
                self.assertEqual(backend.public_context({}, {'task_name':'prepare_coffee'}), '')


if __name__ == '__main__':
    unittest.main()
