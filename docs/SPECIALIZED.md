# Recovery and coffee-disturbance owners

These task families use their original task-specific native interpreters, not
the generic room worker. The public operator verifies the configured source
profile and hands off through the existing `run_original_worker` interface.
It does not implement task scoring, generate scenes, or drive a robot itself.

| Family | Task identifiers |
| :-- | :-- |
| `atomic_coffee` | `coffee_bag_interference`, `coffee_machine_push_interference`, `coffee_bag_then_machine_interference`, `coffee_button_lighting_interference` |
| `failure_recovery` | `failure_recovery_shelf_drink`, `failure_recovery_coffee_bag`, `failure_recovery_sofa`, `failure_recovery_table_waste_empty_grasp`, `failure_recovery_table_waste_tipped_cup` |

## Configure an independent worker

The operator JSON is trusted deployment configuration, never an agent input.
It uses schema `robotagentbench.specialized_operator.v1` and these fields:

- `public_root`: canonical directory containing the installed package.
- `dependencies`: a complete [runtime dependency configuration](DEPENDENCIES.md).
- `environment`: the reviewed native interpreter and SDK profile.
- `workspace`: canonical `input_roots` and shared `lock_root` directories.
- `gpu`: explicitly authorized exclusive device index.
- `families`: separate `atomic_coffee` and `failure_recovery` configurations.
- `outer_lifecycle`: the fixed contract in `simulation/specialized/config.py`.

Each family specifies its audited `manifest`, pinned `bindings` JSON resource,
independent `source_root`, and complete `tasks` mapping. Each task specifies an
existing prepared `launch`, all pinned `inputs`, its `required_resources`, and
original-owner `output_parent`. Outputs must be fresh direct children of that
parent. Preparation does not manufacture missing inputs or resample an episode.

The binding schema is `robotagentbench.specialized_bindings.v1`. It contains
`family`, all manifest `source_resources`, exact `startup_resources`, original
`namespace_mounts`, ordered `import_roots`, and fixed adapter `entry_file`.
The package's compact `identities.json` anchors the two audited manifests; the
external manifests retain their full source tables. Verification checks every
required source, without regenerating or refreshing its pin.

Atomic coffee additionally requires fourteen original startup resources:
the refrigerator controller, environment, hull, manifest and tuning, the salad
model and six textures, and the original shelf-support collision and alignment
layers. Their exact bytes and logical read paths must be bound read-only.
These are not unconditional cup/sofa Recovery dependencies; Recovery resource
closure is specified separately for each task. Static USD dependencies alone
do not cover files opened dynamically by startup policies.

Source precedence is task-family-specific. Recovery retains its current
Generic-first driver baseline. Atomic coffee retains its original
button-parent/Frozen/Generic order and separate public robot-contract package.
The operator checks this order and preserves it across the original second
interpreter launch. Do not globally reorder sources to resolve an import error.

## Inspect and prepare

```sh
robotagentbench specialized /config/native.json --family atomic_coffee --task coffee_machine_push_interference
robotagentbench specialized /config/native.json --family atomic_coffee --task coffee_machine_push_interference --verify
robotagentbench specialized /config/native.json --family atomic_coffee --task coffee_machine_push_interference --prepare /inputs/new-dispatch.json
```

Default mode describes the fixed route. Verification checks configured files
and interpreter resources without importing Isaac or querying a GPU.
Preparation exclusively creates a descriptor sealing an **existing** original
launch and its input hashes. It does not execute original admission or prove
physical initial acceptance.

## Execute explicitly

```sh
robotagentbench specialized /config/native.json --family atomic_coffee --task coffee_machine_push_interference --execute --launch /inputs/original-launch.json --dispatch /inputs/new-dispatch.json --output /outputs/new-run
```

A dedicated external supervisor must enforce the actual shared exclusive GPU
lease, persistent one-dispatch identity, read-only source/asset namespaces,
narrow task-owned writable mounts, the configured 600-second validation
watchdog, and verified process/container/GPU cleanup. Configuration flags alone
do not implement that supervisor. Never retry a failed or uncertain action.

Preserve stdin/stdout as live JSONL pipes through every interpreter and container
boundary. Docker hosts need `--interactive` without a TTY; a host-side stdin
pipe alone does not connect through `docker run`. The public in-container
operator neither launches Docker nor changes host permissions.

Each task owner writes its process scores, completion status and termination
reason. The operator waits for native shutdown before acknowledging closure.

Retain each owner's original observation profile as well. The atomic-coffee
owner returns three 640 × 480 camera images; the generic room profile's
1280 × 800 head and 848 × 480 wrist sizes must not be silently substituted.
