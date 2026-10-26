# Dependency configuration

The core package uses the Python standard library. Native task execution adds
four resource groups:

| Group | Contents |
| --- | --- |
| Task source | Bundled registry, scene setup, native fact readers and scorers |
| Native runtime | Interpreter, Isaac Sim, collision extensions and shared libraries |
| Assets | Room/robot/object geometry, USD layers, textures, calibration and initial-state data |
| Code tools | Optional original broker, sandbox and starter resources |

Task source is included in the wheel. Scene assets are distributed through the
versioned asset release; the simulator and native libraries are installed
separately. Model clients and weights are not core package dependencies. See [installation](INSTALL.md) for the native
baseline and [third-party notices](THIRD_PARTY.md) for licensing boundaries.
The [task resource guide](RESOURCES.md) separates shared resources from each
task's conditional asset and initialization requirements.

## File identities

`RuntimeDependencies.from_config(config)` loads operator-owned JSON with schema
`robotagentbench.runtime_dependencies.v1` and exactly these fields:

| Field | Meaning |
| --- | --- |
| `schema` | Schema identifier above |
| `roots` | Named, existing canonical installation directories |
| `resources` | Resource identifiers mapped to `root`, relative `path`, `sha256` and `kind` |
| `legacy_sources` | Original absolute source names mapped individually to resource identifiers |

Each resource kind is `source`, `native` or `asset`. Pins must come from reviewed
source/asset records, not be regenerated to accept a mismatch. Local installation
paths and credentials do not belong in the repository.

```sh
robotagentbench check-dependencies /path/to/dependencies.json --format json
```

The check rejects missing/changed files, symlinks, path escapes and unresolved
Git LFS pointer text. It never downloads files, searches alternate installations,
rewrites manifests or refreshes hashes. Resolve LFS content only to the original
recorded object identity and size.

`verify_original_sources()` checks an original source record against the explicit
`legacy_sources` bindings. These aliases do not globally intercept filesystem
calls or make every embedded path portable; the selected owner must use its
reviewed bindings.

## Task routes and imports

`BackendPaths.from_config(dependencies, config)` accepts
`robotagentbench.backend_paths.v1` with exactly `schema`, `common` and `tasks`.
Every common/task group contains explicit `source`, `assets`, `native` and
`coding` lists. All registered task routes must be present; empty phase lists do
not enable native execution. Resource identifiers refer to the separate file
configuration, not to paths supplied by an agent.

`NativeSourceTree` separately pins imports from the reviewed task source root.
Each imported file must match both its source pin and its resource binding.
Unreviewed installed modules and bytecode caches cannot replace task code.
Use one source binding for the lifetime of one dedicated worker process.

Pass the configured `paths` to `bind_original_backend` or
`run_original_worker`. Factories check their required phases before delegating
to the task owner. Recovery and atomic-coffee owners require separately verified
fresh-interpreter handoffs; paths alone do not enable those owners.

## Interpreter and loader profile

`NativeEnvironment.from_config(dependencies, config)` accepts
`robotagentbench.native_environment.v1` with these fields:

| Fields | Meaning |
| --- | --- |
| `launcher`, `binary` | Launcher path and hash-pinned interpreter resource |
| `python_version`, `platform`, `machine` | Exact interpreter ABI/platform selection |
| `packages` | Exact reviewed distribution versions |
| `required_resources` | Prerequisite resource identifiers |
| `metadata_paths`, `python_paths`, `library_paths` | Ordered package and loader directories |
| `plugin_paths`, `asset_paths`, `material_paths` | USD plugin, asset and material directories |

All fields, including `schema`, are required. `python_version` is a two-integer
list. Each directory entry has exactly `root` and `path`; unused lists are empty.
A launcher may be a leaf symlink only if it resolves to the exact pinned binary.
The profile excludes inherited Python/loader/USD injection paths rather than
appending them to a reviewed installation.

`probe()` checks the interpreter ABI and package versions in an isolated
process, without initializing the simulator.

## Asset closure and source revisions

A flat scene file is not a complete asset package. Include nested references,
sublayers, payloads, textures, native material resolvers and reference collision
surfaces required by the composed task. Initial posture banks, calibration and
source-episode bindings can remain runtime prerequisites after flattening.
Scene preparation may additionally use task-specific Python/JavaScript helpers;
those versions and subprocess resources must be explicit.

Preserve original geometry, units, physical parameters and file identities.
Unused references require evidence from the actual composed scene, not blanket
exceptions based on filenames. Available bytes do not establish redistribution
rights.

Select compatible scorer, fact-schema and evidence-reader revisions together.
An approved upstream revision gets its own reviewed profile; old records remain
intact. Admission never automatically adopts a newer file or falls back after a
hash mismatch.

The runtime records setup errors separately from scored episodes.
