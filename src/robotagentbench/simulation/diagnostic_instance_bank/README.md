# Saved diagnostic instances

Saved inputs cover Task Discovery, Context Reasoning, Failure Recovery and
Robustness. Selecting or loading an instance verifies its hashes without
resampling a scene, starting Isaac Sim or calling a model.

## Load an existing instance

```sh
python -m robotagentbench.simulation.diagnostic_instance_bank \
  --bank /absolute/path/to/bank --task select_drink --seed 1
```

Add `--model-input` to return only the saved public instruction and context.
Evaluator bindings and answer keys are not included in model input. Task
Discovery also requires `--scenario`: `prepare_dinner`, `host_guests`,
`prepare_to_leave`, or `tidy_room`.

`actual_seed` is the input to the original generator; `episode_ordinal` is the
selected configuration number, 1–30. Use the catalog's saved seed when loading.
The release configurations are separate from the paper's evaluation inputs.
Distinctness is based on physical object transforms, robot initial states and
task bindings, not merely different seed labels or filenames.

Saved-configuration preparation and model evaluation are separate operations.
Preparation status is not a task-success result or a model score. Scoring is
performed by the original task owner during execution.

## Original preparation

`prepare.py` invokes the original task preparers and source-integrity checks.
It does not change task scoring, physics, sampling ranges or admission gates.
Only accepted, distinct configurations enter the release catalog.

- Task Discovery reuses the corresponding saved OpenEnd room, robot initial
  state, object bindings and public context without generating another room.
- Context Reasoning uses original public question IDs and the original seeded
  physical candidate sampler. Answers are not used to select configurations.
- Failure Recovery retains the original pose-library eligibility checks.
- Shelf lighting derives from a saved ordinary-shelf instance using the
  original `prepare_existing` helper, preserving geometry and robot state.
- Shelf relocation uses the formal source-pinned Gatorade request in
  `relocation_request.py`. Its task, target, layout and medium-distance option
  are preserved; only the seed changes.

The small original Recovery preparation dependencies are distributed in
`recovery_sources.py` with per-file SHA-256 pins. An operator preparing a new
bank-owned source view can install them with:

```python
from robotagentbench.simulation.diagnostic_instance_bank.source_install import (
    install_original_recovery_dependencies,
)

receipt = install_original_recovery_dependencies(bank_root, source_root)
```

The installer only accepts the task-owned bank directory, verifies existing
files and does not overwrite different source bytes. Room, robot, product and
pose assets are installed separately using the release asset packages.

## Shelf relocation resource bindings

Source placeholders must not shadow the separately installed original
resources. `relocation_resources.py` lists the eleven original file aliases,
their Runtime-relative locations and asset hashes. For a configured Runtime:

```python
from robotagentbench.simulation.diagnostic_instance_bank.relocation_resources import (
    original_resource_aliases,
)

readonly_files = original_resource_aliases(runtime_root)
```

The returned rows have `source`, `target` and `read_only=True`. Mount these
specific files after their source-tree parents when assembling a preparation
namespace. Their logical target names are container compatibility paths, not
required host directories. The assets already ship in assets-0.1.0 and are not
duplicated in the initial-state configuration bank.

The relocation generator uses a separately recorded current-source baseline
and the original shared-source integrity checks. `relocation_sdk.py` reads the
recorded effective native contact offset for the exact pinned SDK and human
asset; it never sets collision parameters or substitutes a guessed margin.

For a fresh independent relocation source profile:

```python
from robotagentbench.simulation.diagnostic_instance_bank.source_install import (
    export_original_relocation_profile,
)

profile = export_original_relocation_profile(
    runtime_root, fresh_output_root,
    sdk_site=operator_sdk_site,
    logical_source_root=operator_source_root,
)
```

Pass `fresh_output_root` directly to the normal `--runtime` option. It contains
standard `runtime.json`, updated source pins, rebased resource paths and the
same `deployment_release/source` directory layout. No manual worker metadata
merge is needed. The export includes the reviewed sources, SDK-read
compatibility code, exact measured offset data and source-pinned request. It
preserves the original Runtime metadata and source bytes. Place the fresh
output on the same filesystem as the Runtime: unchanged source and resource
files are hardlinked once per shared Runtime, not copied for every
configuration. The SDK paths come from the operator contract; the export does
not install or start an SDK.

Serving an existing saved instance does not invoke these preparation helpers.
