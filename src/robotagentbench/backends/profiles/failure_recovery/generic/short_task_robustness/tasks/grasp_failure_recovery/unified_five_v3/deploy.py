"""Exact approved shared-name wiring; immutable staging, guarded publication."""
import argparse
from copy import deepcopy
import difflib
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import time
from .. import sim2_runner as io
from ..closeout_v1.candidate_runner import command as composed, CANON, PACKAGE

ROOT = io.PROJECT / "short_tasks/agent_probe_v1/deployment_generic_20261010/source"
TARGET = ROOT / "short_task_robustness/tasks/grasp_failure_recovery"
OUT = io.HOST / "unified_five_v3"
ENTRY = "short_tasks/registry.py"
RECORDS = ("public_component_sources.json", "closeout_v1/component_sources.json")
NEW = ("failure_recovery_sofa", "failure_recovery_table_waste_empty_grasp", "failure_recovery_table_waste_tipped_cup")


def sha(path):
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stage():
    if io.HERE != io.HOST or io.HOST.resolve() != io.HOST or TARGET.is_symlink():
        raise ValueError("Exact task/full-tree paths required")
    if (OUT / "stage.json").exists():
        raise ValueError("Immutable stage exists")
    old = json.loads((TARGET / RECORDS[0]).read_text())
    for group, root in (("shared_sources", ROOT), ("task_sources", TARGET), ("frozen_sources", io.PROJECT / "api-protocol-st18-20261013-frozen4")):
        for name, expected in old[group].items():
            if sha(root / name) != expected:
                raise ValueError("Published dependency drift: " + name)
    original = (ROOT / ENTRY).read_text()
    changed = original.replace('"failure_recovery_shelf_drink", "failure_recovery_coffee_bag",\n)',
        '"failure_recovery_shelf_drink", "failure_recovery_coffee_bag",\n    ' + ", ".join(repr(n) for n in NEW) + ",\n)")
    changed = changed.replace("frozenset(('failure_recovery_shelf_drink', 'failure_recovery_coffee_bag'))",
        "frozenset(('failure_recovery_shelf_drink', 'failure_recovery_coffee_bag', " + ", ".join(repr(n) for n in NEW) + "))")
    before = """        plan['no_progress_timeout_s'] = None
        plan['physics_policy'] = 'original_atomic_cpu_fixed_240hz'
        return plan
    if request['task_name'] in ATOMIC_COFFEE_TASKS:"""
    after = """        if plan['failure_recovery_variant'] in ('shelf_drink', 'coffee_bag'):
            plan['no_progress_timeout_s'] = None
            plan['physics_policy'] = 'original_atomic_cpu_fixed_240hz'
        else:
            plan.update(plan['recovery_original_execution'])
        return plan
    if request['task_name'] in ATOMIC_COFFEE_TASKS:"""
    if changed.count(before) != 1:
        raise ValueError("Exact existing family dispatch required")
    changed = changed.replace(before, after)
    if any(n not in changed for n in NEW) or changed == original:
        raise ValueError("Five names not staged")
    before_path = OUT / "source_before" / ENTRY
    if before_path.read_text() != original:
        raise ValueError("Shared entry changed since read-only snapshot")
    after_path = OUT / "candidate" / ENTRY
    after_path.parent.mkdir(parents=True, exist_ok=True)
    after_path.write_text(changed)
    (OUT / "registry.diff").write_text("".join(difflib.unified_diff(original.splitlines(True), changed.splitlines(True), fromfile="before/" + ENTRY, tofile="after/" + ENTRY)))
    sources = set(old["task_sources"])
    closeout_runtime = {"closeout_v1/" + n for n in (
        "__init__.py", "adapter.py", "sampler.py", "public_adapter.py", "registration.py",
        "atomic_owner_v2.py", "room_owner_v2.py")}
    # Do not replace historical fleet/diagnostic launchers that are not imported
    # by the runtime. Their old artifacts and staged hashes remain untouched.
    sources.update(closeout_runtime)
    sources.update(str(p.relative_to(io.HOST)) for p in (io.HOST / "tipped_open_v1").glob("*.py"))
    sources.update((
        "frozen_initials_v3/banks/remaining_exact_training_v1/bank.json",
        "frozen_initials_v3/compile/sofa_offline_adapt_v3/pool.json",
        "frozen_initials_v2/compile/cup_training_exact8_v6/bank_v2.json",
        "tipped_open_v1/definition.json", "tipped_open_v1/compile/elevated_open_v2/bank.json"))
    sources.update(str(p.relative_to(io.HOST)) for p in OUT.glob("*.py"))
    record = deepcopy(old)
    record["task_sources"] = {}
    record["shared_sources"][ENTRY] = sha(after_path)
    record["schema"] = "arena.failure_recovery.unified_five_sources.v3"
    for name in sorted(sources):
        source = OUT / "candidate/public_adapter.py" if name == "public_adapter.py" else io.HOST / name
        if source.is_symlink() or not source.is_file():
            raise ValueError("Real task source required: " + name)
        if name in old["task_sources"] and name != "public_adapter.py" and sha(source) != old["task_sources"][name]:
            raise ValueError("Unapproved old task source drift: " + name)
        dest = TARGET / name
        if name != "public_adapter.py" and dest.exists() and sha(dest) != sha(source):
            raise ValueError("Installed candidate drift: " + name)
        record["task_sources"][name] = sha(source)
    for name in ("public_adapter.py", *RECORDS):
        target = OUT / "before" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(TARGET / name, target)
    for name in RECORDS:
        target = OUT / "candidate" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        io.save(target, record)
    updates, drift, seen = {}, [], set()
    for group in ("short_tasks/tasks", "short_task_ContextReasoning/tasks", "short_task_robustness/tasks"):
        for path in (ROOT / group).glob("*/component_sources.json"):
            actual = path.resolve()
            if actual in seen:
                continue
            seen.add(actual)
            if not actual.is_relative_to(ROOT):
                raise ValueError("Source record outside full tree")
            data = json.loads(path.read_text())
            new = deepcopy(data)
            for field in ("sources", "source_files", "files_sha256"):
                if ENTRY not in data.get(field, {}):
                    continue
                if data[field][ENTRY] != sha(before_path):
                    drift.append(dict(record=str(path), expected=data[field][ENTRY], actual=sha(before_path)))
                    continue
                new[field][ENTRY] = sha(after_path)
            if new == data:
                continue
            name = str(path.relative_to(ROOT))
            text = path.read_text().replace(sha(before_path), sha(after_path))
            if json.loads(text) != new:
                raise ValueError("Only registry fingerprint updates allowed")
            backup, candidate = OUT / "records_before" / name, OUT / "records_after" / name
            backup.parent.mkdir(parents=True, exist_ok=True)
            candidate.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup)
            candidate.write_text(text)
            updates[name] = dict(before=sha(path), after=sha(candidate), real_path=str(actual))
    io.save(OUT / "stage.json", dict(shared_entry=ENTRY, before=sha(before_path), after=sha(after_path),
        task_before={n: sha(TARGET / n) for n in ("public_adapter.py", *RECORDS)},
        task_sources=record["task_sources"], shared_sources_before=old["shared_sources"],
        active_records=updates, preexisting_drift=drift, shared_published=False,
        only_approved_shared_name_and_execution_wiring=True, models_called=False))


def command(module, args, key, *, candidate, installed=False):
    cmd = composed("shelf_drink", module, args, "fivev3_" + key, candidate=False)
    mounts = []
    if candidate:
        mounts += ["--mount", f"type=bind,src={OUT}/candidate/{ENTRY},dst={io.CURRENT}/{ENTRY},readonly"]
        for name in ("public_adapter.py", *RECORDS):
            mounts += ["--mount", f"type=bind,src={OUT}/candidate/{name},dst={CANON}/{name},readonly"]
        stage_data = json.loads((OUT / "stage.json").read_text())
        for name, row in stage_data["active_records"].items():
            dest = Path(io.CURRENT) / Path(row["real_path"]).relative_to(ROOT)
            mounts += ["--mount", f"type=bind,src={OUT}/records_after/{name},dst={dest},readonly"]
    if installed:
        token = f"type=bind,src={io.HOST},dst={CANON}"
        index = cmd.index(token)
        del cmd[index - 1:index + 1]
        home = Path(next(x.split("=", 1)[1] for x in cmd if x.startswith("HOME=")))
        cache = home.parent
        for relative in (Path("unified_five_v3/runs"), cache.relative_to(CANON)):
            (TARGET / relative).mkdir(parents=True, exist_ok=True)
        mounts += ["--mount", f"type=bind,src={OUT}/runs,dst={CANON}/unified_five_v3/runs",
            "--mount", f"type=bind,src={io.HOST/cache.relative_to(CANON)},dst={cache}"]
    cmd[cmd.index("--entrypoint"):cmd.index("--entrypoint")] = mounts
    return cmd


def run(mode, key, name=None, seed=1, candidate=False, installed=False, prepared=None):
    out = OUT / "runs" / key
    out.mkdir(parents=True, exist_ok=False)
    guest = CANON / "unified_five_v3/runs" / key
    if mode == "prepare":
        mod = "short_tasks.agent_probe_v1.prepare_task"
        args = ["--task", name, "--seed", str(seed), "--output", str(guest / "prepared")]
    else:
        mod = PACKAGE + ".unified_five_v3.checks"
        args = [mode, "--output", str(guest / "result.json")]
        if prepared:
            args += ["--launch", str(CANON / "unified_five_v3/runs" / prepared / "prepared/launch.json")]
    cmd = command(mod, args, key, candidate=candidate, installed=installed)
    io.save(out / "command.json", dict(command=cmd, models_called=False, gpu_used=False,
        candidate=candidate, installed_without_task_source_overlay=installed, started=time.time()))
    start = time.time()
    with (out / "stdout.log").open("x") as stdout, (out / "stderr.log").open("x") as stderr:
        rc = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr).returncode
    io.save(out / "exit.json", dict(returncode=rc, wall_s=time.time() - start))
    if rc:
        raise RuntimeError("Preserved failed run: " + key)


def publish():
    data = json.loads((OUT / "stage.json").read_text())
    baseline = json.loads((OUT / "runs/baseline/result.json").read_text())
    after = json.loads((OUT / "runs/candidate_baseline_v2/result.json").read_text())
    if baseline != after or not json.loads((OUT / "runs/candidate_contracts_v4/result.json").read_text())["passed"]:
        raise ValueError("Existing-task regression or failed contracts")
    for name in ("shelf_drink", "coffee_bag", "sofa", "table_waste_empty_grasp", "table_waste_tipped_cup"):
        key = "admit_" + name + "_v3"
        if not json.loads((OUT / "runs" / key / "result.json").read_text())["passed"]:
            raise ValueError("Five actual launch admissions required")
    if sha(ROOT / ENTRY) != data["before"]:
        raise ValueError("Concurrent registry change")
    for n, h in data["task_before"].items():
        if sha(TARGET / n) != h or sha(io.HOST / n) != h:
            raise ValueError("Concurrent task router/record change")
    for n, h in data["shared_sources_before"].items():
        if sha(ROOT / n) != h:
            raise ValueError("Concurrent shared source change: " + n)
    for n, row in data["active_records"].items():
        if sha(ROOT / n) != row["before"]:
            raise ValueError("Concurrent source-record change")
    # Install task-only dependencies before switching the shared registry.
    for n, h in data["task_sources"].items():
        source = OUT / "candidate/public_adapter.py" if n == "public_adapter.py" else io.HOST / n
        if sha(source) != h:
            raise ValueError("Staged task dependency changed: " + n)
        if n == "public_adapter.py":
            continue
        dest = TARGET / n
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and sha(dest) != h:
            raise ValueError("Installed dependency changed: " + n)
        if not dest.exists():
            shutil.copy2(source, dest)
    changes = [(ROOT / ENTRY, OUT / "candidate" / ENTRY)]
    changes += [(ROOT / n, OUT / "records_after" / n) for n in data["active_records"]]
    for base in (io.HOST, TARGET):
        changes += [(base / n, OUT / "candidate" / n) for n in ("public_adapter.py", *RECORDS)]
    for dest, source in changes:
        pending = dest.with_name(dest.name + ".failure5_pending")
        shutil.copy2(source, pending)
        pending.replace(dest)
    io.save(OUT / "published.json", dict(published=True, time=time.time(),
        task_names=["failure_recovery_" + n for n in ("shelf_drink", "coffee_bag", "sofa", "table_waste_empty_grasp", "table_waste_tipped_cup")],
        shared_files_changed=[ENTRY], source_records_updated=len(data["active_records"]),
        ordinary_plans_unchanged=len(baseline), physics_score_control_timeouts_unchanged=True,
        source_hashes_verified=all(sha(TARGET / n) == h for n, h in data["task_sources"].items()),
        models_called=False, full_recovery_success_rate=None, waiting_for_later_model_integration=True))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("stage", "baseline", "contracts", "prepare", "admit", "publish"))
    p.add_argument("--key")
    p.add_argument("--name")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--candidate", action="store_true")
    p.add_argument("--installed", action="store_true")
    p.add_argument("--prepared")
    a = p.parse_args()
    if a.mode == "stage":
        stage()
    elif a.mode == "publish":
        with (OUT / "publication.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            publish()
    else:
        run(a.mode, a.key, a.name, a.seed, a.candidate, a.installed, a.prepared)
