# Task backend source

Task implementations are included in the repository and Python distribution,
under `src/robotagentbench/backends/profiles/`. You do not need access to the
authors' server or a separate private task-code checkout.

| Profile | Contents |
| --- | --- |
| `room` | Short-task initialization and randomization, component scoring, long-task composition, Task Discovery, Context Reasoning, shelf disturbances, and OpenEnd |
| `atomic_coffee` | Bean-bag movement, coffee-machine movement, their combination, and button lighting |
| `failure_recovery` | Shelf drink, bean bag, sofa item, missed cup grasp, and tipped cup |
| `cloth` | Cloth initialization, physical-state reading, grasp/fold/coverage/placement rules, and the native cloth owner |

Inspect the profiles without loading the simulator:

```sh
robotagentbench backends
robotagentbench backends --profile room --verify
robotagentbench backends --profile cloth --root
```

The source manifest records each file's exact version. The task backends remain
separate because their native interpreters, initialization and source-loading
contracts differ. The evaluation core selects the appropriate task owner rather
than merging these environments.

Some source records retain historical absolute names as resource identifiers.
Runtime configuration maps them to local resources inside the job's namespace;
they are not remote directories users must obtain or create on their host.
Do not edit scoring code or asset files to make a mismatched release load.

Scene geometry, textures, initial-state arrays, and large layout or collision
data are distributed separately. See [assets](ASSETS.md),
[installation](INSTALL.md), and [runtime configuration](DEPENDENCIES.md).
