"""CPU invariants for the human-approved reset; no SDK or GPU imports."""
import ast
from copy import deepcopy
from pathlib import Path
import unittest

import numpy as np

from robotagentbench.simulation.cloth import randomization as reset
from robotagentbench.simulation.cloth.variation_native import extend_initialization, INITIAL_HOOK, SCENE_HOOK, _check_scene_edits


class ClothRandomizationTests(unittest.TestCase):
    def test_seed_and_range(self):
        for task in reset.TARGETS:
            values=[reset.draw(task,seed) for seed in range(1,31)]
            self.assertEqual(values,[reset.draw(task,seed) for seed in range(1,31)])
            self.assertEqual(len({repr(v) for v in values}),30)
            for value in values:reset.validate_poses(task,value)
        for seed in (None,0,-1,True,2**32):
            with self.assertRaises(ValueError):reset.draw('spread_tablecloth',seed)

    def test_rigid_shape_and_height(self):
        q=np.array([[0.,0.,.2],[.1,0.,.3],[0.,.1,.2],[1.,1.,.2],[1.1,1.,.3],[1.,1.1,.2]])
        ids=np.array([0,0,0,1,1,1]);old=q.copy()
        poses=reset.draw('long_task_18',4)
        moved=reset.transform_initial(q,ids,'long_task_18',poses)
        np.testing.assert_array_equal(q,old)
        np.testing.assert_array_equal(q[:,2],moved[:,2])
        for cid in (0,1):
            a=q[ids==cid];b=moved[ids==cid]
            np.testing.assert_allclose(np.linalg.norm(a[:,None]-a[None,:],axis=-1),
                                       np.linalg.norm(b[:,None]-b[None,:],axis=-1),atol=1e-14)
        collect=reset.transform_initial(q,ids,'collect_tablecloth',reset.draw('collect_tablecloth',4))
        np.testing.assert_array_equal(collect[ids==1],q[ids==1])

    def test_rejected_candidate_does_not_change_seed(self):
        arrays=dict(initial_cloth_q=np.array([[0.,0.,.2],[.1,0.,.2],[0.,.1,.2]]),
                    cloth_id=np.ones(3,dtype=np.int8),uv=np.zeros((3,2)),faces=np.array([[0,1,2]]),
                    joint_names=np.array(['joint']),initial_joint_targets=np.array([0.]))
        calls=[]
        def gate(q):
            calls.append(q.copy());return dict(accepted=len(calls)==3,reasons=['geometry'] if len(calls)<3 else [])
        value,record=reset.choose_initial(arrays,'spread_tablecloth',1,geometry_gate=gate)
        self.assertEqual(record['seed'],1);self.assertEqual(record['attempt'],2)
        self.assertEqual(len(record['rejected_candidates']),2)
        np.testing.assert_array_equal(value['initial_cloth_q'],reset.transform_initial(
            arrays['initial_cloth_q'],arrays['cloth_id'],'spread_tablecloth',reset.draw('spread_tablecloth',1,2)))
        self.assertFalse(record['native_initial_state_accepted'])

    def test_out_of_range_and_unknown_fields_rejected(self):
        for key,val in [('yaw_deg',10.01),('translation_xy_m',[.03001,0.])]:
            poses=reset.draw('spread_tablecloth',2);poses['1'][key]=val
            with self.assertRaises(ValueError):reset.validate_poses('spread_tablecloth',poses)
        poses=reset.draw('spread_tablecloth',2);poses['1']['height']=1
        with self.assertRaises(ValueError):reset.validate_poses('spread_tablecloth',poses)

    def test_original_source_extension_reversible(self):
        native=Path(__file__).resolve().parents[2]/'backends/profiles/cloth/original/code/tablecloth_workflow/native_teleop/lay_20260924'
        for name in ('native_measured_mirrored_lay_v50.py','native_removal_continuous_transport_v54.py'):
            original=(native/name).read_text();extended=extend_initialization(original)
            self.assertEqual(extended.replace(INITIAL_HOOK,'',1).replace(SCENE_HOOK,'',1),original)
            ast.parse(extended)
            self.assertLess(extended.index(INITIAL_HOOK),extended.index('initial_self = audit_self_intersections'))
            self.assertLess(extended.index(SCENE_HOOK),extended.index('existing_cloth ='))

    def test_scene_guard_rejects_physics_and_light_changes(self):
        old={'/World/A':dict(type='Cube',schemas=['CollisionAPI'],relationships={},
                             attributes={'physics:collisionEnabled':'True','xformOp:scale':'1','xformOp:translate':'0'})}
        after=deepcopy(old);after['/World/A']['attributes']['xformOp:translate']='1'
        self.assertEqual(len(_check_scene_edits(old,after,{'/World/A'},'/wall')),1)
        for name in ('physics:collisionEnabled','xformOp:scale','inputs:intensity'):
            after=deepcopy(old);after['/World/A']['attributes'][name]='changed'
            with self.assertRaises(ValueError):_check_scene_edits(old,after,{'/World/A'},'/wall')


if __name__=='__main__':unittest.main()
