"""Minimal measured-support fixtures; no simulator, SDK or scoring execution."""
from copy import deepcopy
import hashlib
import json
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from robotagentbench.backends import root
from test_cap_source_binding import geometry_methods, Stage, SOFA_CONTACT


COMMON = 'openend/tasks/all_objects_integration/geometry.py'
COMBINATION = 'short_tasks/tasks/sofa_to_drawer/native_geometry.py'
ROOM = '/World/RoomEpisode/'
TABLE = '/World/Assets/CoffeeTable/Mesh'
TRASH = ROOM + 'Selected_trash_group__000'
CAP = ROOM + 'Selected_hang_cap__000'
REMOTE = ROOM + 'Selected_sofa_to_coffee_table__000'
BOOK = ROOM + 'SofaBackground_book_red/Asset/Obj/geom'
CLOTH = ROOM + 'Tablecloth/Asset/Object'
TRAY = (ROOM + 'WoodenTray/Asset/Obj/Meshes/Sketchfab_model/'
        '_99fa031bb144e19baa97558c9a1ba91_fbx/RootNode/OPM0047/'
        'OPM0047_OPM0047_0/OPM0047_OPM0047_0')
WALLET = ROOM + 'Selected_pack_wallet__000/Asset/Obj/Geom/Mesh'


def background(name):
    return dict(path=ROOM + 'SofaBackground_' + name, source='sofa',
                role='background', task_role='background', scoring_target=False)


WORKSPACE = [dict(id='tablecloth_dirty', path=ROOM + 'Tablecloth', role='workspace_support'),
             dict(id='japanese_tray', path=ROOM + 'WoodenTray', role='workspace_support')]
SELECTED_WALLET = dict(id='wallet', path=ROOM + 'Selected_pack_wallet__000',
                      source='coffee_table.objects', task_id='pack_wallet',
                      instance_id='pack_wallet--000')

# Original measured paths, with only the identity fields used by the resolver.
# The sofa fixture exercises the room combination reader also used by Recovery;
# it does not claim that the separately frozen Recovery profile is updated.
FIXTURES = (
    ('sofa08', COMBINATION, REMOTE, 'sofa', 'surface', [background('book_red')],
     [SOFA_CONTACT, BOOK], '/World/Assets/Sofa'),
    ('trash04', COMMON, TRASH, 'coffee_table.objects', 'trash', WORKSPACE,
     [TABLE, CLOTH], '/World/Assets/CoffeeTable'),
    ('cap14', COMMON, CAP, 'sofa', 'hang', [background('power_bank')],
     ['/World/Assets/SeatOverlay', ROOM + 'SofaBackground_power_bank/Asset/Body'],
     '/World/Assets/Sofa'),
    ('cap20', COMMON, CAP, 'sofa', 'hang', [background('book_red')],
     [SOFA_CONTACT, BOOK], '/World/Assets/Sofa'),
    ('long17-04', COMMON, TRASH, 'coffee_table.objects', 'trash', WORKSPACE,
     [TABLE, CLOTH], '/World/Assets/CoffeeTable'),
    ('long17-12', COMMON, TRASH, 'coffee_table.objects', 'trash', WORKSPACE,
     [TABLE, TRAY], '/World/Assets/CoffeeTable'),
    ('long17-24', COMMON, TRASH, 'coffee_table.objects', 'trash', [SELECTED_WALLET],
     [TABLE, WALLET], '/World/Assets/CoffeeTable'),
)


class FurnitureSourceBindingTests(unittest.TestCase):
    def setUp(self):
        self.pxr = patch.dict(sys.modules, {'pxr': SimpleNamespace(UsdPhysics=object())})
        self.pxr.start()
        self.addCleanup(self.pxr.stop)

    def bound(self, fixture, extra=()):
        _, relative, target, source, mode, objects, supports, _ = fixture
        api = geometry_methods(relative)
        manifest = dict(objects=[dict(path=target, source=source),
                                 *deepcopy(objects), *deepcopy(extra)])
        value = api['Geometry'].__new__(api['Geometry'])
        value.stage = Stage(supports)
        value._source_object_roots = api['_source_object_roots'](manifest, target, mode)
        return api, value, manifest

    def test_seven_measured_fixtures_preserve_raw_supports(self):
        for fixture in FIXTURES:
            with self.subTest(case=fixture[0]):
                _, value, manifest = self.bound(fixture)
                before = deepcopy(manifest)
                supports = frozenset(fixture[6])
                self.assertEqual(value.source_furniture(supports), fixture[7])
                self.assertEqual(manifest, before)
                self.assertEqual(supports, frozenset(fixture[6]))

    def test_unknown_foreign_and_object_only_contacts_still_reject(self):
        for fixture in FIXTURES:
            with self.subTest(case=fixture[0]):
                api, value, _ = self.bound(fixture)
                supports = frozenset(fixture[6])
                declared = next(iter(value._source_object_roots))
                for unknown in ('/World/RoomEpisode/Unknown/Collider', declared + '/MissingCollider'):
                    with self.assertRaisesRegex(ValueError, 'Unknown measured support'):
                        value.source_furniture(supports | {unknown})
                for foreign in ('/World/Assets/OtherTable/Collider', declared + '_unlisted/Collider'):
                    value.stage = Stage(supports | {foreign})
                    with self.assertRaisesRegex(ValueError, 'Ambiguous source furniture'):
                        value.source_furniture(supports | {foreign})
                only_objects = frozenset(p for p in supports if any(
                    api['under'](p, r) for r in value._source_object_roots))
                value.stage = Stage(only_objects)
                with self.assertRaisesRegex(ValueError, 'Ambiguous source furniture'):
                    value.source_furniture(only_objects)

    def test_selected_object_requires_exact_source_and_explicit_task_instance(self):
        fixture = FIXTURES[-1]
        for change in (dict(source='other'), dict(task_id=None), dict(task_id=' '),
                       dict(task_id=4), dict(instance_id=None), dict(instance_id=' ')):
            with self.subTest(change=change):
                altered = list(fixture)
                altered[5] = [dict(SELECTED_WALLET, **change)]
                _, value, _ = self.bound(altered)
                self.assertFalse(value._source_object_roots)
                with self.assertRaisesRegex(ValueError, 'Ambiguous source furniture'):
                    value.source_furniture(frozenset(fixture[6]))
        for relative in (COMMON, COMBINATION):
            select = geometry_methods(relative)['_source_object_roots']
            with self.assertRaisesRegex(ValueError, 'Exact scene-owned target'):
                select(dict(objects=[]), TRASH, 'trash')
            self.assertFalse(select(dict(objects=[dict(SELECTED_WALLET)]),
                                    SELECTED_WALLET['path'], 'bag'))

    def test_workspace_anchor_identity_is_not_a_general_support_whitelist(self):
        fixture = FIXTURES[1]
        for objects in ([dict(id='other', path=ROOM + 'Tablecloth', role='workspace_support')],
                        [dict(id='tablecloth_dirty', path=ROOM + 'Tablecloth', role='other')]):
            altered = list(fixture); altered[5] = objects
            _, value, _ = self.bound(altered)
            with self.assertRaisesRegex(ValueError, 'Ambiguous source furniture'):
                value.source_furniture(frozenset(fixture[6]))

    def test_live_component_source_pins_follow_only_the_room_readers(self):
        room = root('room')
        names = ('clear_table_trash', 'hang_cap', 'pack_object',
                 'sofa_to_coffee_table', 'sofa_to_drawer')
        for name in names:
            data = json.loads((room / 'short_tasks/tasks' / name / 'component_sources.json').read_text())
            sources = data.get('source_files', data.get('sources'))
            for relative in (COMMON, COMBINATION):
                if relative in sources:
                    self.assertEqual(sources[relative], hashlib.sha256((room / relative).read_bytes()).hexdigest())


if __name__ == '__main__':
    unittest.main()
