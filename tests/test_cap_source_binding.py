"""CPU regression fixtures for measured cap/background-object contacts."""
import ast
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from robotagentbench.backends import root


def geometry_methods(relative='openend/tasks/all_objects_integration/geometry.py'):
    path = root('room') / relative
    tree = ast.parse(path.read_text(), filename=str(path))
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in ('under', '_source_object_roots')]
    geometry = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Geometry')
    geometry.body = [n for n in geometry.body if isinstance(n, ast.FunctionDef)
                     and n.name == 'source_furniture']
    functions.append(geometry)
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])),
                 str(path), 'exec'), namespace)
    return namespace


class Prim:
    def __init__(self, path):
        self.path = path

    def __bool__(self):
        return bool(self.path)

    def GetPath(self):
        return self.path

    def GetParent(self):
        return Prim(self.path.rsplit('/', 1)[0] or '/') if self.path != '/' else Prim('')


class Stage:
    def __init__(self, paths):
        self.paths = set(paths)

    def GetPrimAtPath(self, path):
        return Prim(path if path in self.paths else '')


SOFA_CONTACT = ('/World/Assets/Sofa/Obj/Meshes/FAB_converted_model/root/'
                'GLTF_SceneRootNode/__seat_sofa_0/Object_4/Object_0')
TARGET = '/World/RoomEpisode/Selected_hang_cap__000'
FIXTURES = (
    # Exact support paths from the two saved, originally accepted cap initials.
    ('paper_ball', ('/World/Assets/SeatOverlay', SOFA_CONTACT,
                   '/World/RoomEpisode/SofaBackground_paper_ball/Asset/Geom/body/visual')),
    ('remote_tv', (SOFA_CONTACT,
                  '/World/RoomEpisode/SofaBackground_remote_tv/Asset/Obj/Meshes/Sketchfab_model/'
                  '_db9e04fd6ec4b6ea5b0bbe7adfe5564_fbx/RootNode/RemoteControl/'
                  'RemoteControl_RemoteControl_0/RemoteControl_RemoteControl_0')),
)


class CapSourceBindingTests(unittest.TestCase):
    def setUp(self):
        self.api = geometry_methods()
        self.pxr = patch.dict(sys.modules, {'pxr': SimpleNamespace(UsdPhysics=object())})
        self.pxr.start()
        self.addCleanup(self.pxr.stop)

    def bound(self, name, supports):
        object_root = '/World/RoomEpisode/SofaBackground_' + name
        manifest = {'objects': [
            {'path': TARGET, 'source': 'sofa'},
            {'path': object_root, 'source': 'sofa', 'role': 'background',
             'task_role': 'background', 'scoring_target': False},
        ]}
        geometry = self.api['Geometry'].__new__(self.api['Geometry'])
        geometry.stage = Stage(supports)
        geometry._source_object_roots = self.api['_source_object_roots'](manifest, TARGET, 'hang')
        return geometry, manifest, object_root

    def test_measured_background_contacts_do_not_become_furniture(self):
        for name, recorded in FIXTURES:
            with self.subTest(object=name):
                supports = frozenset(recorded)
                geometry, manifest, _ = self.bound(name, supports)
                before = deepcopy(manifest)
                self.assertEqual(geometry.source_furniture(supports), '/World/Assets/Sofa')
                self.assertEqual(manifest, before)
                self.assertEqual(supports, frozenset(recorded))

    def test_unknown_and_foreign_supports_are_not_suppressed(self):
        for name, recorded in FIXTURES:
            supports = frozenset(recorded)
            geometry, _, object_root = self.bound(name, supports)
            for unknown in ('/World/RoomEpisode/SofaUnknown/Collider', object_root + '/MissingCollider'):
                with self.subTest(object=name, unknown=unknown):
                    with self.assertRaisesRegex(ValueError, 'Unknown measured support'):
                        geometry.source_furniture(supports | {unknown})
            for foreign in ('/World/Assets/OtherTable/Collider', object_root + '_unlisted/Collider'):
                geometry.stage = Stage(supports | {foreign})
                with self.assertRaisesRegex(ValueError, 'Ambiguous source furniture'):
                    geometry.source_furniture(supports | {foreign})

    def test_object_only_contact_does_not_invent_sofa_contact(self):
        for name, recorded in FIXTURES:
            geometry, _, object_root = self.bound(name, recorded)
            supports = frozenset(p for p in recorded if self.api['under'](p, object_root))
            geometry.stage = Stage(supports)
            with self.assertRaisesRegex(ValueError, 'Ambiguous source furniture'):
                geometry.source_furniture(supports)

    def test_only_explicit_nontarget_background_objects_are_selected(self):
        geometry, manifest, object_root = self.bound(*FIXTURES[0])
        select = self.api['_source_object_roots']
        self.assertEqual(select(manifest, TARGET, 'hang'), frozenset({object_root}))
        for mode in ('bag', 'surface', 'drawer', 'trash'):
            self.assertEqual(select(manifest, TARGET, mode), frozenset({object_root}))
        for field, value in (('role', 'target'), ('task_role', 'target'),
                             ('scoring_target', True), ('source', 'table')):
            altered = deepcopy(manifest)
            altered['objects'][1][field] = value
            self.assertEqual(select(altered, TARGET, 'hang'), frozenset())


if __name__ == '__main__':
    unittest.main()
