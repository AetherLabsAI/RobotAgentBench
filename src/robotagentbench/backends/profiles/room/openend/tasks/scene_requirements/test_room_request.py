import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from room_request import resolve
from generate_room import choose_plan

def request(**kw):
    return dict(schema='openend.scene_requirements.request.v2',seed=42,
                tasks=[dict(id='coffee',type='coffee')],extras=[],**kw)

class RoomRequestTests(unittest.TestCase):
    def test_specified(self):
        r=resolve(request(layout_id=4))
        self.assertEqual(r['layout_candidates'],[4])
        self.assertEqual(r['layout_selection']['mode'],'specified')
    def test_random_repeatable_all_layouts_once(self):
        a=resolve(request());b=resolve(request())
        self.assertEqual(a['layout_candidates'],b['layout_candidates'])
        self.assertEqual(sorted(a['layout_candidates']),list(range(1,51)))
    def test_random_null(self):
        self.assertEqual(resolve(request(layout_id=None))['layout_candidates'],resolve(request())['layout_candidates'])
    def test_invalid_layout(self):
        for n in (0,51,True,'1'):
            with self.assertRaises(ValueError):resolve(request(layout_id=n))
    def test_no_unsupported_target_claim(self):
        r=request();r['tasks']=[dict(id='unknown',type='unsupported_task')]
        with self.assertRaises(ValueError):resolve(r)
    def test_portable_only_keeps_random_microwave_infrastructure(self):
        r=request();r['tasks']=[dict(id='wallet',type='wallet_transfer')]
        with tempfile.TemporaryDirectory() as folder:
            out=choose_plan(resolve(r),Path(folder))
        self.assertEqual([p['id'] for p in out['plan']['placements']],['microwave'])
        self.assertEqual(out['plan']['placements'][0]['task_id'],'infrastructure_microwave')
        self.assertEqual([t['type'] for t in out['requirements']['portable_inventory']['tasks']],['wallet_transfer'])
        self.assertEqual(len(out['living_plan']['small_table']),1)
    def test_all_tasks_have_exact_inventory(self):
        r=json.loads((Path(__file__).parent/'example_all_tasks.json').read_text())
        with tempfile.TemporaryDirectory() as folder:
            out=choose_plan(resolve(r),Path(folder))
        inv=out['requirements']['portable_inventory']
        self.assertEqual(len(inv['tasks']),10)
        self.assertEqual(len(out['living_plan']['objects']),len(inv['objects']))
        self.assertEqual({p['id'] for p in out['plan']['placements']},{'coffee','fryer','microwave'})
        self.assertTrue(2<=next(t['target_count'] for t in inv['tasks'] if t['type']=='cloth_clear')<=4)
    def test_fixed_coordinate_needs_layout(self):
        r=request();r['tasks'][0]['placement']={'world_xy_m':[1,2]}
        with self.assertRaises(ValueError):resolve(r)
    def test_retries_next_layout_on_placement_failure(self):
        r=resolve(request());first,second=r['layout_candidates'][:2]
        class Response:
            stderr=''
            def __init__(self,ok):
                self.returncode=0 if ok else 2
                self.stdout=json.dumps(dict(plan=dict(status='planar_candidate' if ok else 'constraints_unsatisfied',candidate_counts={}),layout_selection={}))
        with tempfile.TemporaryDirectory() as folder,patch('generate_room.subprocess.run',side_effect=[Response(False),Response(True)]):
            out=choose_plan(r,Path(folder))
        self.assertEqual(out['layout_selection']['selected'],second)
        self.assertEqual(out['layout_selection']['rerolls'],1)
        self.assertEqual([x['layout_id'] for x in out['layout_selection']['attempts']],[first,second])
    def test_specified_failure_does_not_retry(self):
        r=resolve(request(layout_id=1))
        class Response:
            returncode=2;stderr='';stdout=json.dumps(dict(plan=dict(status='constraints_unsatisfied',candidate_counts={})))
        with tempfile.TemporaryDirectory() as folder,patch('generate_room.subprocess.run',return_value=Response()) as runner:
            with self.assertRaises(ValueError):choose_plan(r,Path(folder))
            self.assertEqual(runner.call_count,1)
    def test_actual_layout_surface_fallback(self):
        r=request();r['tasks'][0]['placement']={'surface_group':'island'}
        with tempfile.TemporaryDirectory() as folder:
            out=choose_plan(resolve(r),Path(folder))
        self.assertEqual(out['layout_selection']['attempts'][0]['layout_id'],26)
        self.assertEqual(out['layout_selection']['attempts'][0]['missing_surface_groups'],['island'])
        self.assertGreater(out['layout_selection']['rerolls'],0)
        self.assertEqual(out['plan']['status'],'planar_candidate')
    def test_unknown_surface_group_is_error_not_fifty_retries(self):
        r=request();r['tasks'][0]['placement']={'surface_group':'made_up'}
        with self.assertRaises(ValueError):resolve(r)

if __name__=='__main__':unittest.main()
