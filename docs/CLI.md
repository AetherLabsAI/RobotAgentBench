# Command-line interface

Install the package from the repository root:

```sh
python -m pip install .
```

The console command and module entrypoint provide the same interface:

```sh
robotagentbench --help
python -m robotagentbench --help
robotagentbench --version
```

## Query the task catalogs

```sh
robotagentbench catalog standalone
robotagentbench catalog openend
robotagentbench catalog standalone --format json
```

`standalone` enumerates the registered standalone task types, including task
variants. `openend` enumerates component types that can appear in an OpenEnd
scenario; it is not a list of scenario episodes.

Catalog counts are registration counts, not benchmark score denominators.
Listing a task does not initialize it, load its assets, or assert that a task
instance is ready for evaluation. Catalog queries do not import Isaac, contact
model services, or advance physics. The list order is not an execution order.

## Bundled task implementations

```sh
robotagentbench backends
robotagentbench backends --profile room --verify
robotagentbench backends --profile cloth --root
```

## Asset releases

```sh
robotagentbench assets fetch resources/assets-0.1.0.json --cache downloads
robotagentbench assets install resources/assets-0.1.0.json --cache downloads --destination Assets
robotagentbench assets configure resources/assets-0.1.0.json --destination Assets --runtime Runtime --profile room
robotagentbench assets verify resources/assets-0.1.0.json --destination Assets
```

See [assets](ASSETS.md) for offline installation and version checks.

## Verify configured dependencies

```sh
robotagentbench check-dependencies /path/to/dependencies.json
robotagentbench check-dependencies /path/to/dependencies.json --format json
```

The input uses the `robotagentbench.runtime_dependencies.v1` schema described in
[dependency configuration](DEPENDENCIES.md). Duplicate JSON keys are rejected.
The command checks every configured file against its reviewed SHA-256, rejects
changed/missing files, symbolic links and unresolved Git LFS pointers, and returns
only counts by resource kind. It does not print configuration contents or local
paths in an error.

No task modules or simulator are loaded, and no source pins are generated or
updated. A successful check means the listed files match the supplied pins; it
does not authenticate the operator's configuration, prove dependency closure,
admit a task, advance physics or compute scores.

Example JSON output:

```json
{
  "verified": true,
  "file_count": 3,
  "kinds": {"asset": 1, "native": 1, "source": 1}
}
```

## Native tablecloth operator

```sh
robotagentbench cloth /config/cloth.json --task spread_tablecloth
robotagentbench cloth /config/cloth.json --task collect_tablecloth --verify
robotagentbench cloth /config/cloth.json --task long_task_18 --prepare /inputs/new-launch.json
robotagentbench cloth /config/cloth.json --task long_task_18 --execute /inputs/new-launch.json --output /outputs/new-run
```

The default describes the selected task without starting a process or querying
a GPU. `--verify` checks reviewed resources; `--prepare` exclusively creates a
new launch. Only explicit `--execute` performs the trusted native handoff, using
the configured original owner and common JSONL action stream. It requires an
independent native job with read-only resources, the shared GPU lease and outer
watchdog. See [native tablecloth tasks](CLOTH.md) for the configuration contract.

## Recovery and coffee-disturbance operators

```sh
robotagentbench specialized /config/native.json --family failure_recovery --task failure_recovery_sofa
robotagentbench specialized /config/native.json --family atomic_coffee --task coffee_bag_interference --verify
robotagentbench specialized /config/native.json --family atomic_coffee --task coffee_bag_interference --prepare /inputs/new-dispatch.json
robotagentbench specialized /config/native.json --family atomic_coffee --task coffee_bag_interference --execute --launch /inputs/launch.json --dispatch /inputs/new-dispatch.json --output /outputs/new-run
```

Default, verification and preparation modes do not start native execution.
Preparation seals existing original inputs; it does not generate or admit a
scene. Explicit execution requires the original task-specific environment and
an independent supervisor. See [specialized owners](SPECIALIZED.md) for source,
dependency, import-precedence and lifecycle requirements.

## Exit status

- `0`: the requested inspection, verification, or preparation completed, or a
  native execution returned its original successful lifecycle exit.
- `2`: arguments, catalog/configuration data or a required resource failed checks.

Native operators use the corresponding bundled task implementation, asset
release and interpreter profile.
