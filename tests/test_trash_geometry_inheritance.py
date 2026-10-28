"""Exercise the actual overriding trash constructor before native scene checks."""
import ast
from copy import deepcopy
import importlib.util
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from robotagentbench.backends import root
from test_cap_source_binding import Stage
from test_furniture_source_binding import TABLE, TRASH, CLOTH, TRAY, WALLET, WORKSPACE, SELECTED_WALLET

SUBJECTS = {family: root(family) / prefix / 'openend/tasks/trash_transfer/geometry.py'
            for family, prefix in (('room', ''), ('failure_recovery', 'generic'))}


def reader(family):
    prefix = root(family) / ('generic' if family == 'failure_recovery' else '')
    common = ast.parse((prefix / 'openend/tasks/all_objects_integration/geometry.py').read_text())
    nodes = [n for n in common.body if isinstance(n, ast.FunctionDef)
             and n.name in ('under', '_source_object_roots')]
    parent = next(n for n in common.body if isinstance(n, ast.ClassDef) and n.name == 'Geometry')
    parent.body = [n for n in parent.body if isinstance(n, ast.FunctionDef) and n.name == 'source_furniture']
    nodes.append(parent)
    child = ast.parse(SUBJECTS[family].read_text())
    nodes.append(next(n for n in child.body if isinstance(n, ast.ClassDef) and n.name == 'TrashGeometry'))
    registration = prefix / 'openend/tasks/trash_transfer/registration.py'
    spec = importlib.util.spec_from_file_location('pure_trash_registration_' + family, registration)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    namespace = dict(resolve_object=module.resolve_object)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 str(SUBJECTS[family]), 'exec'), namespace)
    return namespace['TrashGeometry']


class TrashInheritance(unittest.TestCase):
    def test_real_overriding_constructor_initializes_inherited_source_binding(self):
        # Deliberately stop at the original first native-scene guard after the
        # real constructor's identity initialization. This is not admission.
        modules = {'pxr': SimpleNamespace(Usd=object(), UsdPhysics=object(),
            UsdGeom=SimpleNamespace(GetStageUpAxis=lambda _: 'unsupported', GetStageMetersPerUnit=lambda _: 1)),
            'evaluation.tasks.trash_disposal.bin_geometry': SimpleNamespace(SceneGeometry=object())}
        for family in SUBJECTS:
            cls = reader(family)
            for support in (CLOTH, TRAY, WALLET):
                with self.subTest(family=family, support=support), patch.dict(sys.modules, modules):
                    manifest = dict(trash_semantics_version=1, objects=[
                        dict(id='paper_ball', path=TRASH, source='coffee_table.objects', trash_category='paper'),
                        *deepcopy(WORKSPACE), deepcopy(SELECTED_WALLET)],
                        task_destinations=[dict(role='ordinary_trash_bin', path='/World/Assets/TrashBinRed')])
                    parameters = dict(target_path=TRASH, category='paper', destination_path='/World/Assets/TrashBinRed')
                    before = deepcopy(manifest)
                    value = cls.__new__(cls)
                    with self.assertRaisesRegex(ValueError, 'Metre Z-up scene required'):
                        value.__init__(object(), manifest, parameters)
                    self.assertTrue(value._source_object_roots)
                    self.assertNotIn(TRASH, value._source_object_roots)
                    supports = frozenset((TABLE, support)); value.stage = Stage(supports)
                    self.assertEqual(value.source_furniture(supports), '/World/Assets/CoffeeTable')
                    self.assertEqual(manifest, before)
                    with self.assertRaisesRegex(ValueError, 'Unknown measured support'):
                        value.source_furniture(supports | {'/World/Unknown/Collider'})
                    foreign = '/World/Assets/ForeignTable/Collider'; value.stage = Stage(supports | {foreign})
                    with self.assertRaisesRegex(ValueError, 'Ambiguous source furniture'):
                        value.source_furniture(supports | {foreign})


if __name__ == '__main__':
    unittest.main()
