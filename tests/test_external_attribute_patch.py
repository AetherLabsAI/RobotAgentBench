"""Tiny CPU-only contracts for exact version-1 and version-2 scene patches.

These tests create Sdf layers only. They need the optional USD Python bindings,
not Isaac Sim, a renderer, assets, model weights, or a generated room.
"""
import copy
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from robotagentbench.simulation.saved_instance_release import original_scene_patch as reader

try:
    from pxr import Gf, Sdf, Vt
except ImportError:
    Sdf = None


@unittest.skipIf(Sdf is None, "USD Python bindings are not installed")
class ExternalAttributePatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.runtime = self.root / "Runtime"
        self.assets = self.root / "Assets"
        self.runtime.mkdir()
        self.assets.mkdir()
        self.base_path = self.root / "base.usdc"
        self.base = Sdf.Layer.CreateNew(str(self.base_path))
        Sdf.CreatePrimInLayer(self.base, "/World/Box/Geom")
        self.base.Save()

        self.resource_path = self.root / "asset.usdc"
        self.resource_layer = Sdf.Layer.CreateNew(str(self.resource_path))
        prim = Sdf.CreatePrimInLayer(self.resource_layer, "/Resource/Geom")
        attr = Sdf.AttributeSpec(prim, "samples", Sdf.ValueTypeNames.Float3Array)
        attr.default = Vt.Vec3fArray([Gf.Vec3f(i / 4, 1, 2) for i in range(1025)])
        attr.SetInfo("documentation", "Exact typed fixture; no physics")
        self.resource_layer.Save()
        self.original = Sdf.Layer.CreateAnonymous(".usda")
        self.original.TransferContent(self.base)
        self.assertTrue(Sdf.CopySpec(
            self.resource_layer, "/Resource/Geom.samples",
            self.original, "/World/Box/Geom.samples"))
        original_path = self.root / "original.usdc"
        self.original.Export(str(original_path))

        self.key = "libraries/test/asset.usdc"
        actual = self.runtime / self.key
        actual.parent.mkdir(parents=True)
        shutil.copyfile(self.resource_path, actual)
        digest = reader.sha(self.resource_path)
        object_path = "objects/" + digest[:2] + "/" + digest
        obj = self.assets / object_path
        obj.parent.mkdir(parents=True)
        shutil.copyfile(actual, obj)
        index = dict(
            schema="robotagentbench.asset_index.v1", version="0.1.7",
            objects={digest: dict(bytes=actual.stat().st_size, sha256=digest, path=object_path)},
            locations={self.key: digest})
        self.index_path = self.assets / "resource-index-v17.json"
        self.index_path.write_text(json.dumps(index))
        self.metadata = dict(
            version="0.1.7", assets=str(self.assets),
            resources={self.key: str(actual)}, library_files={self.key: digest})
        self.delta_path = self.root / "delta.usdc"
        self.delta = Sdf.Layer.CreateNew(str(self.delta_path))
        self.delta.Save()
        self.legacy = dict(
            schema=reader.SCHEMA, original_path="/original/room.usdc",
            original_scene_sha256=reader.sha(original_path),
            base_resource="libraries/test/base.usdc", base_sha256=reader.sha(self.base_path),
            delta_sha256=reader.sha(self.delta_path), removed_specs=[],
            changed_specs=["/World/Box/Geom.samples"], borrowed_specs={},
            root_order=["/World"], root_fields=list(self.delta.pseudoRoot.ListInfoKeys()),
            child_orders={"/World": ["/World/Box"], "/World/Box": ["/World/Box/Geom"],
                          "/World/Box/Geom": []},
            property_orders={"/World": [], "/World/Box": [],
                             "/World/Box/Geom": ["/World/Box/Geom.samples"]})
        names, signature = reader.external_v2.attribute_identity(attr)
        self.recipe = dict(
            self.legacy, schema=reader.external_v2.SCHEMA,
            asset_index=dict(path="resource-index-v17.json", sha256=reader.sha(self.index_path)),
            external_attribute_specs=[dict(
                target_path="/World/Box/Geom.samples", resource=self.key,
                resource_sha256=digest, source_path="/Resource/Geom.samples",
                field_names=names, fields_sha256=signature)])
        immutable = [self.base_path, self.resource_path, original_path, actual, obj, self.index_path]
        self.immutable = {path: reader.sha(path) for path in immutable}
        self.addCleanup(self.assert_inputs_unchanged)

    def assert_inputs_unchanged(self):
        for path, digest in self.immutable.items():
            self.assertEqual(reader.sha(path), digest)

    def restore(self, recipe=None, metadata=None, delta=None):
        return reader.restore_layer(
            self.recipe if recipe is None else recipe,
            base=self.base_path, delta=self.delta_path if delta is None else delta,
            output=self.root / "restored.usdc", runtime=self.runtime,
            metadata=self.metadata if metadata is None else metadata)

    def assert_exact(self, output):
        layer = Sdf.Layer.FindOrOpen(str(output))
        expected_paths, actual_paths = [], []
        self.original.Traverse("/", lambda p: expected_paths.append(str(p)))
        layer.Traverse("/", lambda p: actual_paths.append(str(p)))
        self.assertEqual(actual_paths, expected_paths)
        for path in expected_paths:
            expected = self.original.GetObjectAtPath(path)
            actual = layer.GetObjectAtPath(path)
            self.assertEqual(set(actual.ListInfoKeys()), set(expected.ListInfoKeys()))
            for key in expected.ListInfoKeys():
                self.assertEqual(actual.GetInfo(key), expected.GetInfo(key), (path, key))
            if isinstance(expected, Sdf.PrimSpec):
                self.assertEqual(list(actual.nameChildren.keys()), list(expected.nameChildren.keys()))
                self.assertEqual(list(actual.properties.keys()), list(expected.properties.keys()))

    def rejected(self, change, *, metadata=None, delta=None):
        recipe = copy.deepcopy(self.recipe)
        change(recipe)
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.restore(recipe, metadata=metadata, delta=delta)

    def test_v1_unchanged_exact_reconstruction(self):
        Sdf.CreatePrimInLayer(self.delta, "/World/Box/Geom")
        self.assertTrue(Sdf.CopySpec(
            self.resource_layer, "/Resource/Geom.samples",
            self.delta, "/World/Box/Geom.samples"))
        self.delta.Save()
        recipe = dict(self.legacy, delta_sha256=reader.sha(self.delta_path),
                      root_fields=list(self.delta.pseudoRoot.ListInfoKeys()))
        self.assert_exact(self.restore(recipe))

    def test_v2_index_runtime_exact_attribute_reconstruction(self):
        self.assert_exact(self.restore())

    def test_unknown_recipe_field(self):
        self.rejected(lambda r: r.update(unknown=True))

    def test_unknown_attribute_override(self):
        self.rejected(lambda r: r["external_attribute_specs"][0].update(default=[]))

    def test_duplicate_target(self):
        self.rejected(lambda r: r["external_attribute_specs"].append(
            copy.deepcopy(r["external_attribute_specs"][0])))

    def test_illegal_target(self):
        def change(recipe):
            recipe["changed_specs"] = ["relative.samples"]
            recipe["external_attribute_specs"][0]["target_path"] = "relative.samples"
        self.rejected(change)

    def test_undeclared_target(self):
        self.rejected(lambda r: r.update(changed_specs=[]))

    def test_wrong_resource_sha(self):
        self.rejected(lambda r: r["external_attribute_specs"][0].update(resource_sha256="0" * 64))

    def test_unknown_resource(self):
        self.rejected(lambda r: r["external_attribute_specs"][0].update(
            resource="libraries/unknown.usdc"))

    def test_missing_source_spec(self):
        self.rejected(lambda r: r["external_attribute_specs"][0].update(
            source_path="/Resource/Geom.absent"))

    def test_prim_not_attribute(self):
        self.rejected(lambda r: r["external_attribute_specs"][0].update(
            source_path="/Resource/Geom"))

    def test_undeclared_authored_field(self):
        self.rejected(lambda r: r["external_attribute_specs"][0]["field_names"].append(
            "zzzUndeclared"))

    def test_wrong_field_signature(self):
        self.rejected(lambda r: r["external_attribute_specs"][0].update(fields_sha256="1" * 64))

    def test_runtime_pin_mismatch(self):
        metadata = copy.deepcopy(self.metadata)
        metadata["library_files"][self.key] = "0" * 64
        self.rejected(lambda r: None, metadata=metadata)

    def test_resource_outside_runtime(self):
        metadata = copy.deepcopy(self.metadata)
        metadata["resources"][self.key] = str(self.resource_path)
        self.rejected(lambda r: None, metadata=metadata)

    def test_delta_cannot_override_external(self):
        # CopySpec needs a destination parent even for this deliberately invalid delta.
        Sdf.CreatePrimInLayer(self.delta, "/World/Box/Geom")
        self.assertTrue(Sdf.CopySpec(
            self.resource_layer, "/Resource/Geom.samples",
            self.delta, "/World/Box/Geom.samples"))
        self.delta.Save()
        self.rejected(lambda r: r.update(delta_sha256=reader.sha(self.delta_path)))
        self.assertFalse((self.root / "restored.usdc").exists())


if __name__ == "__main__":
    unittest.main()
