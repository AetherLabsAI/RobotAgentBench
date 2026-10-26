"""CPU-only tests; no simulator, asset directory, or model service is required."""
from dataclasses import FrozenInstanceError
import json
import unittest

from robotagentbench.tasks.catalog import CatalogError, load_catalog, parse_catalogs


class CatalogTests(unittest.TestCase):
    def test_current_standalone_identifiers_are_exposed(self):
        catalog = load_catalog("standalone")
        self.assertIn("prepare_coffee", catalog.identifiers)
        self.assertIn("wipe_table", catalog.identifiers)
        self.assertIn("microwave_heat_take", catalog.identifiers)
        self.assertIn("failure_recovery_shelf_drink", catalog.identifiers)
        self.assertEqual(catalog.entry_kind, "task types")

    def test_openend_is_a_component_catalog(self):
        catalog = load_catalog("openend")
        self.assertIn("coffee", catalog.identifiers)
        self.assertIn("umbrella_transfer", catalog.identifiers)
        self.assertEqual(catalog.entry_kind, "component types")
        self.assertEqual(catalog.as_dict()["count"], len(catalog.identifiers))

    def test_catalog_is_immutable_and_returned_json_is_detached(self):
        catalog = load_catalog("standalone")
        with self.assertRaises(FrozenInstanceError):
            catalog.name = "other"
        view = catalog.as_dict()
        view["identifiers"].clear()
        self.assertTrue(catalog.identifiers)

    def test_unknown_catalog_is_rejected(self):
        with self.assertRaises(CatalogError):
            load_catalog("../../private")

    def test_invalid_data_is_rejected(self):
        valid = {"schema_version": "robotagentbench.catalog.v1",
                 "catalogs": {"standalone": ["task"], "openend": ["component"]}}
        invalid = [None, {}, {**valid, "schema_version": "unknown"},
                   {**valid, "extra": True},
                   {**valid, "catalogs": {"standalone": ["task"]}},
                   {**valid, "catalogs": {"standalone": ["task", "task"], "openend": ["component"]}},
                   {**valid, "catalogs": {"standalone": ["../task"], "openend": ["component"]}},
                   {**valid, "catalogs": {"standalone": [], "openend": ["component"]}},
                   {**valid, "catalogs": {"standalone": [True], "openend": ["component"]}}]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(CatalogError):
                parse_catalogs(json.dumps(value))
        with self.assertRaises(CatalogError):
            parse_catalogs('{"schema_version":"a","schema_version":"b"}')


if __name__ == "__main__":
    unittest.main()
