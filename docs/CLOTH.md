# Native tablecloth tasks

The native cloth operator connects the original GPU cloth environment to the
same observation/action protocol used by the evaluation core. It does not
generate robot actions or replace the original initializer, scoring, physics,
camera configuration, or progress timeout.

| Task | Runtime identifier | Original source selection |
| --- | --- | --- |
| Spread the tablecloth | `spread_tablecloth` | Spread initial state and laying entry |
| Collect the tablecloth | `collect_tablecloth` | Collection initial state and removal entry |
| Replace the tablecloth | `long_task_18` | Two-cloth scene; collect before spreading |

Cloth tasks use their own GPU environment. They cannot fall back to the ordinary
CPU-physics scene owner. The original fixed world anchor is retained, so both
mobile-base commands must be zero. Each 20 Hz command advances the original
three 60 Hz physical substeps. Observations use the original head camera and
both wrist cameras.

## Required configuration

The operator JSON has the `robotagentbench.cloth_operator.v1` schema and these
fields:

- `public_root`: the canonical installed directory containing `robotagentbench`.
- `workspace`: explicit `input_roots`, `output_root`, and shared `lock_root`.
- `native`: existing `dependencies` and `profile` objects, as described in
  [dependency configuration](DEPENDENCIES.md) and [installation](INSTALL.md).
- `gpu`: an explicit device index; GPU sharing exceptions are not supported.
- `launch_templates`: exactly the three runtime identifiers above. Each entry
  contains reviewed `directories`, `files`, and SHA-256 `hashes`.
- `outer_lifecycle`: `validation_watchdog_s: 600`, `exclusive_gpu: true`,
  `read_only_inputs: true`, `no_automatic_retry: true`, and
  `owned_cleanup_required: true`.

Each template binds the original source and asset bundles, prepared initial
state and setting, derived collision geometry and its original export receipt,
protocol and trial records, source owner, geometry libraries, and SDK. It also
pins this installed operator and public runtime. Hashes must come from reviewed
resource records; admission never refreshes them from arbitrary current files.

The original source's required logical paths are retained as exact read-only
compatibility mounts and aliases. The fixed recipes are defined in
`simulation.cloth.config`; preparation verifies them but does not create host
mounts or modify shared assets. Source-owner code is included in the `cloth`
backend profile. Initial states, scene assets and geometry are distributed in
the separate [asset release](ASSETS.md).

The outer job supervisor must enforce the declared watchdog, read-only mounts,
host/GPU identity, and exact-owned container cleanup. Declaring this contract in
JSON is not evidence that those host protections exist. All jobs sharing a GPU
must use the same cooperative lock directory.

## Inspect and prepare

The default command only describes the selected task. It does not query a GPU,
start a child process, import Isaac, or initialize a scene.

```sh
robotagentbench cloth /config/cloth.json --task spread_tablecloth
robotagentbench cloth /config/cloth.json --task collect_tablecloth --verify
robotagentbench cloth /config/cloth.json --task long_task_18 --prepare /inputs/new-launch.json
```

Verification checks the configured files and environment bindings. Preparation
creates one fresh launch and refuses to overwrite an existing file. It neither
admits a physical initial state nor establishes task success.

## Execute

Inside the configured independent native job, explicitly select execution:

```sh
robotagentbench cloth /config/cloth.json --task long_task_18 \
  --execute /inputs/new-launch.json --output /outputs/new-run
```

This entry constructs the trusted binding internally, starts a fresh native
interpreter, and forwards the common JSONL action protocol over stdin/stdout.
The configuration cannot select a Python factory, diagnostic policy, or scoring
override. The physical owner performs original initialization and measured
initial-state acceptance before exposing observations or accepting actions.

The supervisor forwards requests once and waits for actual physical and render
child shutdown before releasing its lease and sending the terminal `closed`
acknowledgement. Missing cleanup, an uncertain action, or an interrupted job is
not a successful result and must not trigger automatic action replay.

The task owner computes process scores and completion status, using its shared
120-second simulation-time no-progress timeout.
