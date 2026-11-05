"""Installing the drawer slide on an open stage must not recompose that stage from its root."""
import ast
import importlib.util
import unittest

from robotagentbench.backends import sources


def task_file(name):
    return sources.root('room')/'short_tasks/tasks/sofa_to_drawer'/name


def copy_slide():
    tree = ast.parse(task_file('prepare_scene.py').read_text(encoding='utf-8'))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_copy_slide')
    namespace = {}
    exec(compile(ast.Module([function], []), 'prepare_scene.py', 'exec'), namespace)
    return namespace['_copy_slide']


class OpenStageDrawerTests(unittest.TestCase):
    def test_runtime_hook_says_the_stage_is_open(self):
        tree = ast.parse(task_file('scene.py').read_text(encoding='utf-8'))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and getattr(node.func, 'id', None) == 'add_existing_drawer']
        self.assertEqual(len(calls), 1)
        self.assertEqual([(k.arg, k.value.value) for k in calls[0].keywords], [('stage_is_open', True)])

    @unittest.skipUnless(importlib.util.find_spec('pxr'), 'requires USD')
    def test_copy_composes_like_the_sublayer_and_keeps_other_opinions(self):
        from pxr import Sdf, Usd
        base = Sdf.Layer.CreateAnonymous('.usda')
        base.ImportFromString("""#usda 1.0
def Xform "World"
{
    def Xform "Table"
    {
        def Xform "Box"
        {
            double weight = 1
        }
    }
}
""")
        slide = Sdf.Layer.CreateAnonymous('.usda')
        slide.ImportFromString("""#usda 1.0
over "World"
{
    over "Table"
    {
        over "Box" (
            prepend apiSchemas = ["PhysicsRigidBodyAPI"]
        )
        {
            double weight = 2

            def Scope "Part"
            {
                int size = 3
            }
        }

        def Scope "Joint"
        {
            rel body = </World/Table/Box>
        }
    }
}
""")

        def facts(stage):
            return {str(prim.GetPath()): (prim.GetAppliedSchemas(), {a.GetName(): a.Get() for a in prim.GetAttributes()},
                                          {r.GetName(): r.GetTargets() for r in prim.GetRelationships()})
                    for prim in stage.Traverse()}

        layered = Usd.Stage.Open(base, Sdf.Layer.CreateAnonymous('.usda'))
        layered.GetSessionLayer().subLayerPaths = [slide.identifier]
        copied = Usd.Stage.Open(base, Sdf.Layer.CreateAnonymous('.usda'))
        session = copied.GetSessionLayer()
        session.ImportFromString("""#usda 1.0
over "World"
{
    def Scope "Other"
    {
        int kept = 1
    }
}
""")
        expected = facts(layered)
        with Sdf.ChangeBlock():
            for prim in slide.rootPrims:
                copy_slide()(slide, session, prim.path)
        self.assertEqual(session.subLayerPaths, [])
        self.assertEqual(session.GetPrimAtPath('/World/Other').properties['kept'].default, 1)
        found = facts(copied)
        found.pop('/World/Other')
        self.assertEqual(found, expected)


if __name__ == '__main__':
    unittest.main()
