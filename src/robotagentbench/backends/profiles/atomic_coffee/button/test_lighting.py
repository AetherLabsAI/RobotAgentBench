import math
from pathlib import Path
import unittest
from .cycle import LightingCycle, MULTIPLIER, TEMPERATURE_K, INTERVAL_S, FLASH_S, FLASH_MULTIPLIER


class ScheduleTests(unittest.TestCase):
    def test_initial_is_normal_and_unarmed(self):
        c = LightingCycle(7)
        self.assertEqual(c.tick(100), [])
        self.assertEqual(c.state['mode'], 'normal')
        self.assertEqual(c.state['multiplier'], 1.)
        c.arm(100)
        self.assertTrue(101 <= c.deadline <= 103)
        self.assertEqual(c.tick(100), [])

    def test_reproducible_and_seeded(self):
        cycles = [LightingCycle(x) for x in (9, 9, 10)]
        for c in cycles:
            c.arm(4)
            for step in range(240 * 30):
                c.tick(4 + step / 240)
        self.assertEqual(cycles[0].events, cycles[1].events)
        self.assertNotEqual(cycles[0].events, cycles[2].events)

    def test_ranges_and_flash_clear(self):
        for seed in range(30):
            c = LightingCycle(seed)
            c.arm(1)
            c.tick(68)
            changes = [e for e in c.events if e['kind'] == 'light_changed']
            self.assertGreater(len(changes), 10)
            previous = 1
            for e in changes:
                s = e['state']
                self.assertTrue(MULTIPLIER[0] <= s['multiplier'] <= MULTIPLIER[1])
                self.assertTrue(TEMPERATURE_K[0] <= s['temperature_k'] <= TEMPERATURE_K[1])
                interval = e['scheduled_time_s'] - previous
                self.assertTrue(INTERVAL_S[0] - 1e-10 <= interval <= INTERVAL_S[1] + 1e-10)
                previous = e['scheduled_time_s']
                if s['flash']:
                    self.assertTrue(FLASH_MULTIPLIER[0] <= s['flash_multiplier'] <= FLASH_MULTIPLIER[1])
                    duration = e['flash_end_s'] - previous
                    self.assertTrue(FLASH_S[0] - 1e-10 <= duration <= FLASH_S[1] + 1e-10)
                    if e['flash_end_s'] <= 68:
                        self.assertTrue(any(x['kind'] == 'flash_ended' and
                                            x['scheduled_time_s'] == e['flash_end_s'] for x in c.events))
                else:
                    self.assertIsNone(s['flash_multiplier'])

    def test_existing_schedule_and_environment_samples_preserved(self):
        import importlib.util
        original = Path(__file__).parent/'before_strong_flash_002/cycle.py'
        spec = importlib.util.spec_from_file_location('previous_lighting_cycle', original)
        previous = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(previous)
        for seed in (0,7,23):
            old, new = previous.LightingCycle(seed), LightingCycle(seed)
            old.arm(1); new.arm(1)
            old.tick(68); new.tick(68)
            self.assertEqual(len(old.events), len(new.events))
            for a,b in zip(old.events,new.events):
                for key in ('kind','scheduled_time_s','next_change_s','flash_end_s'):
                    self.assertEqual(a[key],b[key])
                for key in ('mode','multiplier','temperature_k','side','flash'):
                    self.assertEqual(a['state'][key],b['state'][key])

    def test_fine_and_coarse_ticks_have_same_schedule(self):
        fine, coarse = LightingCycle(5), LightingCycle(5)
        fine.arm(0); coarse.arm(0)
        for i in range(2400):
            fine.tick(i / 240)
        coarse.tick(2399 / 240)
        strip = lambda c: [{k: v for k, v in e.items() if k != 'observed_time_s'} for e in c.events]
        self.assertEqual(strip(fine), strip(coarse))
        self.assertEqual(fine.state, coarse.state)

    def test_invalid_clock_and_lifecycle(self):
        for seed in (None, True, -1, 1.5):
            with self.assertRaises(ValueError):
                LightingCycle(seed)
        c = LightingCycle(1)
        for now in (-1, math.nan, math.inf, True):
            with self.assertRaises(ValueError):
                c.tick(now)
        c.arm(3)
        with self.assertRaises(RuntimeError):
            c.arm(4)
        with self.assertRaises(ValueError):
            c.tick(2)
        c.close()
        with self.assertRaises(RuntimeError):
            c.tick(5)


class UsdTests(unittest.TestCase):
    def scene(self):
        from pxr import Usd, UsdGeom, UsdLux, Gf
        from .lighting import CAMERA
        s = Usd.Stage.CreateInMemory()
        cam = UsdGeom.Camera.Define(s, CAMERA)
        cam.AddTranslateOp().Set(Gf.Vec3d(0, 0, 2))
        cam.CreateClippingRangeAttr().Set(Gf.Vec2f(.01,1000))
        UsdGeom.Cube.Define(s, '/World/Target')
        UsdGeom.Cube.Define(s, '/World/Task75/Machine/Body')
        for i in range(2):
            light = UsdLux.RectLight.Define(s, '/World/Lamp%d' % i)
            light.CreateIntensityAttr(1000 + i * 1000)
            light.AddTranslateOp().Set(Gf.Vec3d(i, 4, 4))
        return s

    def test_only_lights_change_and_restore(self):
        from pxr import Usd, UsdGeom, Gf
        from .lighting import SceneLighting, CAMERA
        s = self.scene()
        root = s.GetRootLayer().ExportToString()
        camera = str(UsdGeom.XformCache().GetLocalToWorldTransform(s.GetPrimAtPath(CAMERA)))
        session = s.GetSessionLayer()
        with Usd.EditContext(s, session):
            # Preserve an unrelated concurrent session opinion on close.
            UsdGeom.Xformable(s.GetPrimAtPath('/World/Target')).AddTranslateOp().Set(Gf.Vec3d(1, 2, 3))
            s.GetPrimAtPath('/World/Lamp0').GetAttribute('inputs:intensity').Set(432.)
        light = SceneLighting(s, 0)
        light.arm(1)
        self.assertFalse(light.saved)
        modes = set()
        for i in range(2400):
            light.tick(1+i/240)
            modes.add(light.cycle.state['mode'])
        self.assertTrue({'ambient', 'side', 'camera_flash'} <= modes)
        self.assertEqual(root, s.GetRootLayer().ExportToString())
        self.assertEqual(camera, str(UsdGeom.XformCache().GetLocalToWorldTransform(s.GetPrimAtPath(CAMERA))))
        allowed = ('inputs:intensity', 'inputs:enableColorTemperature', 'inputs:colorTemperature',
                   'xformOpOrder', 'xformOp:transform:robustLighting')
        self.assertTrue(all(p.split('.')[-1] in allowed for p in light.saved))
        light.close(); light.close()
        self.assertEqual(s.GetPrimAtPath('/World/Lamp0').GetAttribute('inputs:intensity').Get(), 432.)
        self.assertFalse(s.GetPrimAtPath('/World/Lamp1').GetAttribute('xformOp:transform:robustLighting'))
        self.assertEqual(root, s.GetRootLayer().ExportToString())
        self.assertEqual(list(UsdGeom.XformCache().GetLocalToWorldTransform(s.GetPrimAtPath('/World/Target')).ExtractTranslation()), [1,2,3])

    def test_original_poses_and_nonlight_parameters_unchanged(self):
        from .lighting import SceneLighting
        from .initial import physical_properties
        s = self.scene()
        before = physical_properties(s)
        controller = SceneLighting(s, 8)
        controller.arm(0)
        controller.tick(25)
        self.assertEqual(physical_properties(s), before)
        self.assertEqual(controller.snapshot()['camera_parameter_writes'], 0)
        controller.close()

    def test_strong_flash_applies_only_to_selected_lamp_then_clears(self):
        from pxr import UsdLux
        from .lighting import SceneLighting
        s=self.scene()
        c=SceneLighting(s,7)
        c.arm(0)
        for step in range(2400):
            c.tick(step/240)
            if c.cycle.state['flash']:
                break
        self.assertTrue(c.cycle.state['flash'])
        state=c.cycle.state
        selected=c.emitter['path']
        for row in c.records:
            actual=UsdLux.LightAPI(s.GetPrimAtPath(row['path'])).GetIntensityAttr().Get()
            expected=row['intensity']*(state['flash_multiplier'] if row['path']==selected else state['multiplier'])
            self.assertAlmostEqual(actual, expected, delta=expected*1e-6)
        c.tick(c.cycle.flash_end)
        self.assertFalse(c.cycle.state['flash'])
        self.assertIsNone(c.cycle.state['flash_multiplier'])
        for row in c.records:
            actual=UsdLux.LightAPI(s.GetPrimAtPath(row['path'])).GetIntensityAttr().Get()
            expected=row['intensity']*c.cycle.state['multiplier']
            self.assertAlmostEqual(actual,expected,delta=expected*1e-6)
        c.close()

    def test_missing_or_animated_lights_rejected(self):
        from pxr import Usd, UsdGeom
        from .lighting import SceneLighting, CAMERA
        s = Usd.Stage.CreateInMemory()
        UsdGeom.Camera.Define(s, CAMERA)
        with self.assertRaises(ValueError):
            SceneLighting(s, 0)
        s = self.scene()
        s.GetPrimAtPath('/World/Lamp0').GetAttribute('inputs:intensity').Set(100., 1.)
        with self.assertRaises(ValueError):
            SceneLighting(s, 0)

    def test_real_emitter_aims_at_camera_without_moving_camera(self):
        from pxr import UsdGeom, Gf
        from .lighting import SceneLighting, CAMERA
        s = self.scene()
        controller = SceneLighting(s, 0)
        controller.arm(0)
        observed = set()
        for step in range(2400):
            controller.tick(step/240)
            mode = controller.cycle.state['mode']
            if mode not in ('side', 'camera_flash') or mode in observed:
                continue
            observed.add(mode)
            cache = UsdGeom.XformCache()
            camera = cache.GetLocalToWorldTransform(s.GetPrimAtPath(CAMERA)).ExtractTranslation()
            lamp = cache.GetLocalToWorldTransform(s.GetPrimAtPath(controller.emitter['path']))
            origin = lamp.ExtractTranslation()
            delta = camera-origin
            if mode=='side':
                self.assertAlmostEqual(delta.GetLength(), controller.original_distance)
            else:
                self.assertLess(delta.GetLength(),controller.placement_events[-1]['first_obstruction_m'])
                self.assertGreater(origin[2],0.)
            self.assertGreater(Gf.Dot(delta.GetNormalized(), lamp.TransformDir(Gf.Vec3d(0,0,-1)).GetNormalized()), .999999)
        self.assertEqual(observed, {'side','camera_flash'})
        controller.close()


class ClearPositionTests(unittest.TestCase):
    def test_midpoint_before_first_obstruction_not_below_floor(self):
        from .clear_position import choose
        r=choose((2.1635,.09175,1.4158),(2.55,.145,1.07),
                 [('machine',(2.4,.01,.92),(2.71,.28,1.22)),
                  ('floor',(-10,-10,-.1),(10,10,0))],.01)
        self.assertTrue(r['center_ray_unobstructed'])
        self.assertGreater(r['position_m'][2],1.22)
        self.assertLess(r['distance_from_camera_m'],r['first_obstruction_m'])

    def test_scene_box_is_used_even_if_it_is_not_machine(self):
        from .clear_position import choose
        r=choose((0,0,0),(10,0,0),[('block',(1,-1,-1),(2,1,1))],.1)
        self.assertEqual(r['first_obstruction_path'],'block')
        self.assertAlmostEqual(r['position_m'][0],.55)

    def test_no_clear_space_is_rejected_not_teleported(self):
        from .clear_position import choose
        with self.assertRaises(ValueError):
            choose((0,0,0),(10,0,0),[('block',(-1,-1,-1),(1,1,1))],.1)

    def test_parallel_rays_and_boxes_behind_camera(self):
        from .clear_position import ray_entry
        self.assertIsNone(ray_entry((0,0,0),(1,0,0),(1,1,1),(2,2,2)))
        self.assertIsNone(ray_entry((0,0,0),(1,0,0),(-2,-1,-1),(-1,1,1)))
        self.assertEqual(ray_entry((0,0,0),(1,0,0),(1,-1,-1),(2,1,1)),1.)


class SourceTests(unittest.TestCase):
    def test_unmodified_rule_and_evidence_hooks(self):
        from .runtime import LightingButtonRuntime
        from evaluation.tasks.coffee_machine.api_runtime import CoffeeMachineRuntime
        for name in ('collect_evidence', 'evaluate', 'check_success', 'check_failure', 'finalize',
                     'on_physics_ready', 'robot_initialization'):
            self.assertIs(getattr(LightingButtonRuntime, name), getattr(CoffeeMachineRuntime, name))

    def test_definition_is_not_registration(self):
        import json
        d = json.loads((Path(__file__).parent/'definition.json').read_text())
        self.assertFalse(d['globally_registered'])
        self.assertIsNone(d['runtime_id'])
        self.assertEqual(d['base_api'], '1-3')


class GPUAdmissionTests(unittest.TestCase):
    def result(self,memory=2,util=0,compute='',pmon='4 - - - - - - - - - -\n',index=4,approved=True):
        from unittest.mock import patch
        from .record_preview import idle_gpu
        with patch('subprocess.check_output',side_effect=[f'GPU-test, {memory}, {util}',compute,pmon]):
            return idle_gpu(index,approved)

    def test_only_explicit_gpu4_residue_is_accepted(self):
        self.assertTrue(self.result()['authorized_idle_gpu4_residue_2mib'])
        with self.assertRaises(RuntimeError):
            self.result(approved=False)
        with self.assertRaises(ValueError):
            self.result(index=0)

    def test_compute_or_graphics_processes_always_rejected(self):
        with self.assertRaises(RuntimeError):
            self.result(compute='1234, python')
        with self.assertRaises(RuntimeError):
            self.result(pmon='4 1234 G 0 0 - - - - app\n')

    def test_active_or_more_memory_is_rejected(self):
        with self.assertRaises(RuntimeError):
            self.result(memory=3)
        with self.assertRaises(RuntimeError):
            self.result(util=1)


if __name__ == '__main__':
    unittest.main()
