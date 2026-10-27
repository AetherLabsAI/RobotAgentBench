"""CPU-only exact source and original projection equivalence, not model inference."""
import ast
from copy import deepcopy
import hashlib
import importlib.util
from importlib.machinery import SourceFileLoader
from pathlib import Path
import sys
import unittest

from . import _server_projection as portable
from .test_contract import worker_contract
from .test_contract import observation

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'private/server_source'


class Canonical(ast.NodeTransformer):
    def visit_Attribute(self,node):
        node=self.generic_visit(node)
        if isinstance(node.value,ast.Name)and node.value.id=='self':
            return ast.copy_location(ast.Name(id=node.attr,ctx=node.ctx),node)
        return node

    def visit_Call(self,node):
        node=self.generic_visit(node)
        if (isinstance(node.func,ast.Name)and node.func.id=='str'and len(node.args)==1
                and isinstance(node.args[0],ast.Name)and node.args[0].id=='Q'):
            return ast.copy_location(ast.Name(id='OPERATOR_VLM_PATH',ctx=ast.Load()),node)
        return node

    def visit_Subscript(self,node):
        node=self.generic_visit(node)
        if isinstance(node.value,ast.Name)and node.value.id=='config'and isinstance(node.slice,ast.Constant)and node.slice.value=='vlm':
            return ast.copy_location(ast.Name(id='OPERATOR_VLM_PATH',ctx=ast.Load()),node)
        return node


def assignment(tree,name):
    values=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign)and any(
            isinstance(target,ast.Name)and target.id==name or
            isinstance(target,ast.Attribute)and target.attr==name for target in node.targets):
            values.append(ast.dump(Canonical().visit(deepcopy(node.value)),include_attributes=False))
    if not values:raise AssertionError('Original key expression required: '+name)
    return values


@unittest.skipUnless((SOURCE/'serve_intern.py.private').is_file(),'Exact private source-review snapshots are not distributed in wheels')
class ServerParity(unittest.TestCase):
    def test_exact_action_adapter_and_pending_commit(self):
        expected='6939b1be0848261afb03829f4c29de57d7ff58d6b3f3d3b69ca7493a5202ad45'
        for path in [SOURCE/'action_adapter.py.private',ROOT/'_server_projection.py']:
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),expected)
        path=SOURCE/'action_adapter.py.private'
        spec=importlib.util.spec_from_loader('server_original_action_adapter',SourceFileLoader('server_original_action_adapter',str(path)))
        original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
        c=worker_contract();c['joint_limits'][c['action_order'][18]]=[0.,.7853981633974483]
        c['joint_limits'][c['action_order'][19]]=[0.,.7853981633974483]
        left,right=original.ActionAdapter(c),portable.ActionAdapter(c)
        actions=[]
        pairs=[(.001,.005),(.0010000001,.005000001),(.18,.25),(-.18,.25),
               (0.,0.),(.006,0.),(0.,.015),(-.5,-.7)]
        for index in range(50):
            row=[0.]*22;row[18]=[-.01,0.,.7853981633974483,.8][index%4]
            row[19]=row[18];row[20:]=pairs[index%len(pairs)];actions.append(row)
        for committed in (0,17,50):
            a,pa=left.project(actions);b,pb=right.project(actions)
            self.assertEqual(a,b);self.assertEqual(pa,pb)
            self.assertEqual(left.pending,right.pending)
            left.commit(committed);right.commit(committed)
            self.assertEqual(left.previous,right.previous);self.assertEqual(left.pending,right.pending)
        bad=deepcopy(actions);bad[-1][-1]=float('nan')
        for adapter in(left,right):
            with self.assertRaises(ValueError):adapter.project(bad)
            self.assertIsNone(adapter.pending)
        self.assertNotIn('torch',sys.modules)


class ServerTransportSeam(unittest.TestCase):
    def test_public_binding_full_chunk_projection_and_ack(self):
        from .server_transport import rollout
        ready=dict(contract=worker_contract(),observation=observation())
        ready['observation']['private_answer']={'do_not_forward':True}
        calls=[]
        class Model:
            def exchange(self,request):
                calls.append(request)
                if 'private_answer'in request['observation']:raise AssertionError('Private metadata leaked')
                return dict(op='actions',prompt=request['observation']['instruction'],
                    episode_id=request['observation']['episode_id'],step=0,
                    actions=[[0.]*20+[.0001,.0002]for _ in range(50)])
        class Owner:
            def exchange(self,request):
                if len(request['actions'])!=50:raise AssertionError('Changed chunk size')
                if any(row[20:]!=[0.,0.]for row in request['actions']):raise AssertionError('Original deadband not applied')
                return dict(actions_executed=50,observation=observation(50),episode_finished=True)
        traces=[]
        result=rollout(Owner(),Model(),ready,task_id='select_drink',max_calls=1,
            on_event=lambda *row:traces.append(row))
        self.assertEqual(result['actions_executed'],50)
        self.assertEqual(calls[0]['training_task_index'],'public:select_drink')
        self.assertEqual(calls[0]['observation']['context'],'Public synthetic context')
        self.assertEqual([x[0]for x in traces],['prediction','projection','reply'])
        self.assertNotIn('torch',sys.modules)

    @unittest.skipUnless((SOURCE/'serve_intern.py.private').is_file(),'Exact private source-review snapshots are not distributed in wheels')
    def test_original_numeric_and_image_transform_expressions(self):
        source=SOURCE/'serve_intern.py.private'
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),
            'a2e030601ee9531de6780a3a5220eb482ff523507acb440246590f56b9a5b3ea')
        original=ast.parse(source.read_text());new=ast.parse((ROOT/'server_policy.py').read_text())
        for name in('mapping','state','sample','rgb','batch','actions','transform'):
            self.assertEqual(assignment(original,name),assignment(new,name),name)
        self.assertNotIn('torch',sys.modules)


if __name__=='__main__':unittest.main()
