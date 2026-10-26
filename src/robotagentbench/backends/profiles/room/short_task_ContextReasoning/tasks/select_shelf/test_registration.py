"""CPU public-registry and lifecycle tests; never label synthetic contacts native."""
import json
from pathlib import Path
import unittest
from copy import deepcopy
from short_tasks import registry
from .task import RULE_VERSION, ShelfSelectionEvaluator
from .inventory_plan import plan_inventory
from .bank import BANK_DIR
from short_task_ContextReasoning.tasks.select_sofa.task import SofaSelectionEvaluator, ASSETS
from short_task_ContextReasoning.tasks.select_trash.native import SelectionEpisode

def request(name,question=None,seed=2):
    version='context_reasoning.'+('select_shelf' if name in ('select_drink','select_snack') else name)+'.v1'
    return dict(schema='openend.short_task.request.v1',task_name=name,rule_version=version,
        seed=seed,layout_id=3,options={} if question is None else dict(question_id=question))

def manifest_for(plan):
    name=plan['task_name'];seed=plan['request']['seed']
    if name in ('select_drink','select_snack'):
        inv=plan_inventory(plan['public_question']['question_id'],seed)
        rows=[dict(id=o['product_family'],instance_id=o['instance_id'],
            path='/World/RoomEpisode/ContextShelf_'+o['product_family'],
            source='snack_shelf',task_role='touch_candidate',task_id=name)
            for l in inv['inventory']['layers'] for o in l['objects']]
    else:
        rows=[dict(id=a,instance_id='item_'+str(i),path='/World/ContextSofa/item_'+str(i),
            source='sofa',task_role='selection_candidate',task_id=name) for i,a in enumerate(ASSETS.values())]
    return dict(seed=seed,layout_id=3,objects=rows,tasks=[dict(id=name)])

class RegisteredTests(unittest.TestCase):
    def test_source_requests_and_dispatch(self):
        from .registered_prepare import source_request
        from short_tasks.agent_probe_v1 import prepare_task
        from unittest.mock import patch
        for name in ('select_sofa','select_drink','select_snack'):
            req=request(name)
            base=source_request(req)
            self.assertTrue(registry.resolve(base)['available'])
            self.assertEqual(base['seed'],req['seed'])
            self.assertEqual(base['layout_id'],req['layout_id'])
            module='short_task_ContextReasoning.tasks.select_shelf.registered_prepare.prepare_registered'
            with patch(module,return_value={'delegated':name}) as hook:
                self.assertEqual(prepare_task.prepare(req,'unused','unused','unused'),{'delegated':name})
                hook.assert_called_once_with(req,'unused','unused','unused')

    def test_four_registered(self):
        self.assertEqual(set(registry.CONTEXT_TASKS),{'select_trash','select_sofa','select_drink','select_snack'})
        for name in registry.CONTEXT_TASKS:
            plan=registry.resolve(request(name))
            self.assertEqual(plan['score_policy']['maximum'],1)
            self.assertNotIn('interaction_policy',plan);self.assertNotIn('navigation_policy',plan)
            registry.verify_sources(plan)

    def test_all_400_questions_bind_and_terminate(self):
        banks=[('select_sofa',json.loads((Path(__file__).parents[1]/'select_sofa/question_bank_v1/public_questions.json').read_text())),
               (None,json.loads((BANK_DIR/'public_questions.json').read_text()))]
        count=0
        for family,bank in banks:
            for q in bank['questions']:
                name=family or {'drink':'select_drink','snack':'select_snack'}[q['kind']]
                plan=registry.resolve(request(name,q['id']))
                bound=registry.bind(plan,manifest_for(plan));selection=bound['selection']
                target=selection['target_instance_id'];wrong=next(r['instance_id'] for r in selection['candidates'] if r['instance_id']!=target)
                for touched,score,reason in (([],0,None),([target],1,'success'),([wrong],0,'wrong_candidate_touch'),([wrong,target],0,'wrong_candidate_touch')):
                    cls=SofaSelectionEvaluator if name=='select_sofa' else ShelfSelectionEvaluator
                    evaluator=cls('test',selection);evaluator.arm(0.,initial_touches=[])
                    value=evaluator.observe(episode_id='test',time_s=.05,touched=touched)
                    frame=registry.normalize(bound,{name:value})
                    frame.update(task_name=name,episode_id='test',rule_version=bound['rule_version'])
                    ledger=SelectionEpisode(task_name=name,episode_id='test',actual_layout_id=3,
                        rule_version=bound['rule_version'],start_time_s=0,target_roles={},maximum_score=1)
                    result=ledger.observe(frame,simulation_time_s=.05)
                    self.assertEqual(result['scores']['total'],score)
                    self.assertEqual(result['status'],'running' if reason is None else 'finished')
                    if reason: self.assertEqual(result['termination_reason'],reason)
                count+=1
        self.assertEqual(count,400)

    def test_wrong_kind_and_duplicate_fail_closed(self):
        with self.assertRaises(Exception):registry.resolve(request('select_snack','R001'))
        plan=registry.resolve(request('select_drink','R001'));manifest=manifest_for(plan)
        manifest['objects'].append(deepcopy(manifest['objects'][0]))
        with self.assertRaises(Exception):registry.bind(plan,manifest)

if __name__=='__main__':unittest.main()
