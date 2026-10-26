"""CPU rules/source verification; only a task-owned JSON artifact is written."""
import argparse
import hashlib
import io
import importlib.util
import json
from pathlib import Path
import sys
import unittest


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    report = Path(args.report).absolute()
    if here not in report.parents or report.exists():
        raise ValueError("Fresh task-owned report path required")
    if any(p.is_symlink() for p in [report, *report.parents] if p != here.parent):
        raise ValueError("Symlinked report path rejected")
    baseline = json.loads((here/"source_manifest.json").read_text())
    source = here.parents[2]
    current_record = here/"component_sources.json"
    pins = (json.loads(current_record.read_text())["sources"] if current_record.exists()
            else baseline["files"])
    before = {rel: hashlib.sha256((source/rel).read_bytes()).hexdigest() for rel in pins}
    if before != pins:
        raise ValueError("Source changed: review rather than silently repin")
    sys.path.insert(0, str(source))
    stream = io.StringIO()
    # Only current task tests; preserved historical drafts are not current
    # acceptance coverage and must not inflate this count.
    suite = unittest.TestSuite()
    for path in sorted(here.glob("test_*.py")):
        name = "heat_take_validation_"+path.stem
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    after = {rel: hashlib.sha256((source/rel).read_bytes()).hexdigest()
             for rel in pins}
    payload = dict(schema="arena.short_microwave_heat_take.cpu_validation.v1",
        rules_tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
        skipped=len(result.skipped), source_pins=len(before), sources_unchanged=before == after,
        global_registration="see_integration_receipt" if current_record.exists() else "not_registered", model_calls=0,
        initial_state_validation="not_run", native_validation="not_run",
        stdout=stream.getvalue(), passed=result.wasSuccessful() and before == after)
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("x") as handle:
        json.dump(payload, handle, indent=2); handle.write("\n")
    print(json.dumps({k:v for k,v in payload.items() if k != "stdout"}, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
