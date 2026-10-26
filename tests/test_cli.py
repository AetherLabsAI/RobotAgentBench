"""Tests of the installed CLI's read-only commands and error boundary."""
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
import hashlib
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
import unittest

from robotagentbench.cli import main


class CliTests(unittest.TestCase):
    def dependency_config(self, root):
        asset = root/'resource.txt'
        asset.write_text('reviewed resource\n')
        value = dict(schema='robotagentbench.runtime_dependencies.v1',
            roots=dict(assets=str(root)), resources=dict(example=dict(root='assets',path=asset.name,
                sha256=hashlib.sha256(asset.read_bytes()).hexdigest(),kind='asset')),legacy_sources={})
        config = root/'dependencies.json';config.write_text(json.dumps(value))
        return config, asset, value

    def test_dependency_check_json_is_read_only_and_does_not_load_native_modules(self):
        with tempfile.TemporaryDirectory() as folder:
            config,asset,_=self.dependency_config(Path(folder).resolve())
            before={p.name:p.read_bytes() for p in config.parent.iterdir()}
            loaded=set(sys.modules);output=StringIO()
            with redirect_stdout(output):
                code=main(['check-dependencies',str(config),'--format','json'])
            self.assertEqual(code,0)
            self.assertEqual(json.loads(output.getvalue()),dict(verified=True,file_count=1,kinds=dict(asset=1)))
            self.assertEqual(before,{p.name:p.read_bytes() for p in config.parent.iterdir()})
            forbidden={'isaacsim','pxr','omni','carb','numpy','PIL','short_tasks','openend','arena_api'}
            self.assertFalse({name.split('.')[0] for name in set(sys.modules)-loaded}&forbidden)

    def test_dependency_check_text(self):
        with tempfile.TemporaryDirectory() as folder:
            config,_,_=self.dependency_config(Path(folder).resolve());output=StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(['check-dependencies',str(config)]),0)
            self.assertEqual(output.getvalue(),'Verified 1 configured resources\n')

    def test_dependency_mismatch_is_error_not_score_and_does_not_leak_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            config,asset,_=self.dependency_config(Path(folder).resolve());asset.write_text('changed')
            output,error=StringIO(),StringIO()
            with redirect_stdout(output),redirect_stderr(error):
                self.assertEqual(main(['check-dependencies',str(config)]),2)
            self.assertFalse(output.getvalue());self.assertNotIn(folder,error.getvalue())
            self.assertNotIn('Traceback',error.getvalue());self.assertIn('failed verification',error.getvalue())
            self.assertEqual(asset.read_text(),'changed')

    def test_duplicate_dependency_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            config,_,_=self.dependency_config(Path(folder).resolve())
            config.write_text('{"resources":{},"resources":{}}')
            with redirect_stderr(StringIO()):
                self.assertEqual(main(['check-dependencies',str(config)]),2)

    def test_invalid_dependency_shapes_are_sanitized(self):
        with tempfile.TemporaryDirectory() as folder:
            config,_,value=self.dependency_config(Path(folder).resolve())
            for bad in (None,[],{**value,'resources':[]},{**value,'resources':{'bad':None}},{**value,'schema':'unknown'}):
                with self.subTest(value=bad),redirect_stderr(StringIO()):
                    config.write_text(json.dumps(bad));self.assertEqual(main(['check-dependencies',str(config)]),2)

    def test_missing_dependency_config_is_sanitized(self):
        error=StringIO()
        with tempfile.TemporaryDirectory() as folder,redirect_stderr(error):
            self.assertEqual(main(['check-dependencies',str(Path(folder)/'absent.json')]),2)
        self.assertNotIn(folder,error.getvalue())

    def test_json_catalog(self):
        output = StringIO()
        with redirect_stdout(output):
            result = main(["catalog", "standalone", "--format", "json"])
        self.assertEqual(result, 0)
        value = json.loads(output.getvalue())
        self.assertEqual(value["catalog"], "standalone")
        self.assertEqual(value["count"], len(value["identifiers"]))

    def test_text_component_catalog(self):
        output = StringIO()
        with redirect_stdout(output):
            result = main(["catalog", "openend"])
        self.assertEqual(result, 0)
        self.assertIn("registered component types", output.getvalue())

    def test_help_and_version(self):
        for args in (["--help"], ["--version"], ["catalog", "--help"]):
            with self.subTest(args=args), redirect_stdout(StringIO()), self.assertRaises(SystemExit) as exc:
                main(args)
            self.assertEqual(exc.exception.code, 0)

    def test_invalid_commands_are_errors_not_fake_results(self):
        for args in ([], ["evaluate"], ["catalog", "unknown"], ["catalog", "standalone", "--unknown"]):
            with self.subTest(args=args), redirect_stderr(StringIO()), self.assertRaises(SystemExit) as exc:
                main(args)
            self.assertEqual(exc.exception.code, 2)

    def test_resource_errors_do_not_print_private_paths(self):
        output, error = StringIO(), StringIO()
        with patch("robotagentbench.cli.load_catalog", side_effect=OSError("/private/internal/catalog.json")), \
                redirect_stdout(output), redirect_stderr(error):
            result = main(["catalog", "standalone"])
        self.assertEqual(result, 2)
        self.assertFalse(output.getvalue())
        self.assertNotIn("/private", error.getvalue())
        self.assertNotIn("Traceback", error.getvalue())


if __name__ == "__main__":
    unittest.main()
