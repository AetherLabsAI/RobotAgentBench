"""Task-owned CPU source/contract tests and actual room preparation."""
from pathlib import Path
import hashlib
import importlib.util
import io
import json
import sys
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(HERE)]


def write(name, value):
    folder = HERE/"integration_v1/validation"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder/name).open("x") as f:
        json.dump(value, f, indent=2, default=str); f.write("\n")


def old_plans():
    from short_tasks import registry
    from short_tasks.agent_probe_v1.probe_random_layout import request_for
    out = {}
    for name in registry.TASK_NAMES:
        if name == "microwave_heat_take":
            continue
        try:
            out[name] = dict(plan=registry.resolve(request_for(name)))
        except Exception as exc:
            out[name] = dict(existing_error_type=type(exc).__name__, existing_error=str(exc))
    return out


mode = sys.argv[1]
if mode == "baseline":
    out = old_plans(); write("old_plans_before.json", out)
    print(json.dumps(dict(old_task_count=len(out))))
elif mode in ("contracts", "contracts_002", "contracts_003"):
    from short_tasks import registry
    before = json.loads((HERE/"integration_v1/validation/old_plans_before.json").read_text())
    assert before == old_plans(), "Old task definitions changed"
    stream, suite = io.StringIO(), unittest.TestSuite()
    for p in sorted(HERE.glob("test_*.py")):
        spec = importlib.util.spec_from_file_location("heat_take_"+p.stem, p)
        module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    request = json.loads((HERE/"request.json").read_text())
    plan = registry.resolve(request)
    source = registry.verify_sources(plan)
    audit = {}
    for name in registry.TASK_NAMES:
        try:
            request_old = request if name == "microwave_heat_take" else __import__(
                "short_tasks.agent_probe_v1.probe_random_layout", fromlist=["request_for"]).request_for(name)
            p = registry.resolve(request_old)
            if p["available"]: audit[name] = registry.verify_sources(p)
        except Exception as exc:
            audit[name] = dict(existing_unavailable_or_error=str(exc))
    write(mode+".json", dict(passed=result.wasSuccessful(), tests_run=result.testsRun,
        failures=len(result.failures), errors=len(result.errors), stdout=stream.getvalue(),
        old_task_count=len(before), old_definitions_identical=True, source=source,
        existing_source_audit=audit, maximum=14, native_physics_validated=False))
    print(stream.getvalue()); print(json.dumps(dict(passed=result.wasSuccessful(), source=source)))
    raise SystemExit(0 if result.wasSuccessful() else 1)
elif mode.startswith("prepare_"):
    from short_tasks.agent_probe_v1.prepare_task import prepare
    from openend.tasks.layout_repair_v1.ready_envelope import REFERENCE
    case = mode.replace("prepare", "cpu", 1)
    inputs = HERE/"execution"/(case+"_inputs"); inputs.mkdir(parents=True, exist_ok=False)
    ready = inputs/"ready_pose.json"
    ready.write_text(json.dumps(json.loads((REFERENCE/"navigation_result.json").read_text())["navigation_posture"]))
    fixture = ROOT/"short_tasks/agent_probe_v1/cap_v1/native_tools/openend_control_adapter_v1/task_discovery_v1/four_scenarios_min10_v1/operation_v1/fixtures/prepare_to_leave/launch.json"
    sensor = inputs/"sensor.json"
    sensor.write_text(json.dumps(dict(sensor=json.loads(fixture.read_text())["sensor"])))
    report = prepare(json.loads((HERE/"request.json").read_text()), ready, sensor, HERE/"execution"/case)
    launch = json.loads(Path(report["launch"]).read_text())
    manifest = json.loads(Path(launch["manifest"]).read_text())
    write(case+".json", dict(report=report, layout_id=manifest["layout_id"],
        scene=manifest["scene"], scene_sha256=hashlib.sha256(Path(manifest["scene"]).read_bytes()).hexdigest(),
        tasks=manifest["tasks"], maximum=14, native_physics_validated=False))
    print(json.dumps(report))
else:
    raise SystemExit("Unknown check")
