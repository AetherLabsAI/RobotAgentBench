# Saved initial configurations

Install the shared assets once, then choose one of 30 saved initial
configurations for a task or scenario. Different agents can use the same saved
configuration. Running an evaluation does not select another initial state.

## Evaluation sets

| Family | Groups | Configurations per group | Total |
| --- | ---: | ---: | ---: |
| Manipulation: short tasks | 18 | 30 | 540 |
| Manipulation: long tasks | 8 | 30 | 240 |
| Task Discovery | 4 scenarios | 30 | 120 |
| Context Reasoning | 4 tasks | 30 | 120 |
| Failure Recovery | 5 tasks | 30 | 150 |
| Robustness | 6 variants | 30 | 180 |
| OpenEnd | 4 scenarios | 30 | 120 |
| **Total** | **49** | **30** | **1,470** |

These release configurations are for reproducible evaluation. They are **not
the exact instances used for the paper's reported results**. Report the code,
asset and configuration-bank versions, model checkpoint, task identifiers and
episode numbers with new results.

Episode numbers 1–30 identify saved configurations. Each also records the
generator's `actual_seed`; accepted seeds need not be consecutive.

## OpenEnd daily situations (0.2.1)

All four OpenEnd scenarios have **30 configurations each**: six context/scene
variants, each paired with five spatial initial states. The agent receives
the household memory, a high-level request, and **Today's situation**. Public
context contains non-visual background information, not an inventory of the
current scene or a description of its cleanliness.

| Scenario | Examples of current circumstances |
| --- | --- |
| Preparing dinner | A different person comes home; a dinner plan is cancelled; the usual coffee is skipped tonight |
| Hosting guests | A visitor changes; two guests want different refreshments; a guest arrives after training |
| Preparing to leave | Rain during an outdoor trip; dry weather; an updated forecast; rain only after the person returns |
| Tidying the room | The household wants the shared living area ready for use again; current objects and cleanliness must be observed |

Stains, empty containers, used packaging, and whether a surface is already clean
are not stated in the daily context. The agent must inspect camera observations
and apply the household's general storage and cleaning rules. Matched variants
with and without a stain use identical public text; all six tidying variants
also share the same public request and context.

Rain during an outdoor trip adds the existing umbrella-packing task. A cancelled
rain forecast or rain after returning home does not. Weather is contextual
information, not a simulated rain effect. Existing task scoring and physical
parameters are unchanged; the task composition determines the available points.

Within each scenario, episodes 1–6 cover the six variants; the same order
repeats in 7–12, 13–18, 19–24, and 25–30 with different spatial seeds. These
are not thirty distinct stories: some variants differ only in observable scene
state. Version 0.2.1 corrects the public context in 40 OpenEnd records without
changing any scene state, task target, scoring rule, or physical parameter.
The other 1,430 state files are byte-identical to 0.2.0; the 1,350 non-OpenEnd
records remain byte-identical to 0.1.0. Older releases remain available.

## Scene-composition recipes (0.2.2)

Version 0.2.2 includes the original scene-composition recipes for all 30
long-task tray-stocking configurations and all 30 shelf-lighting configurations.
It preserves every sampled physical state, seed, public context, and scoring
target from 0.2.1; the other 1,410 configuration records are byte-identical.
No room geometry or texture assets are added to the configuration download.

## Original spawn receipts (0.2.3)

Version 0.2.3 includes the original spawn receipts for all 30 configurations
of `long_task_16`, `long_task_17`, `long_task_19`, `long_task_20`, `long_task_21`,
and `long_task_22`. Their original scene and robot-base bindings are retained;
running a saved configuration does not resample the robot's starting position.
All physical states, seeds, public contexts, and scoring targets are unchanged
from 0.2.2, and the other 1,290 configuration records are byte-identical.

## Current configuration bank (0.2.17)

Version 0.2.17 replaces six saved initial states using original sampling and
admission rules. All six replacements passed their task-owned native initial
admission checks and saved-state restoration/readonly verification.

| Task identifier | Episode | Actual seed |
| --- | ---: | ---: |
| `microwave_heat_take` | 11 | 34 |
| `task_discovery--prepare_dinner` | 17 | 131 |
| `failure_recovery_sofa` | 6 | 83 |
| `prepare_coffee` | 27 | 39 |
| `bake_bread` | 6 | 31 |
| `carry_meal_tray` | 6 | 31 |

The bank contains 49 groups of 30 configurations (1,470 total). All saved
initial states have passing original task-owned admission records; initial
validity is not task completion or model success. The other **1,464 state
records are byte-identical to 0.2.16**, including its previous sofa replacements.
Public contexts, task definitions, scoring rules, physical parameters, shared
assets 0.1.7, and all 50 room layouts are unchanged.

Use evaluator commit `e8448f732572cacf330554c3620d6f25341be81f` or later;
the saved-scene reader supports lossless external attribute records. The
approximately **260 MB** download references five new 0.2.17
archives and two unchanged historical archives. Use the complete manifest;
do not rename or concatenate archive parts. These release configurations are
not the exact evaluation instances used for the paper's results.

## Configuration bank 0.2.16

Version 0.2.16 replaces two sofa Failure Recovery initial states using the
original sampling and admission rules. Both replacements passed original
native initial admission and saved-state restoration.

| Task identifier | Episode | Actual seed |
| --- | ---: | ---: |
| `failure_recovery_sofa` | 8 | 96 |
| `failure_recovery_sofa` | 30 | 110 |

The bank remains at 49 groups of 30 configurations (1,470 total). The other
1,468 state records are byte-identical to 0.2.15, including its three previous
replacements. Public contexts, task definitions, scoring rules, physical
parameters, and shared assets 0.1.7 are unchanged. The approximately **259 MB**
download references four new 0.2.16 archives and three unchanged historical
archives; do not rename or concatenate them. Use source commit
`f86a908d709fb94dcfc970be8c5da93305be9968` or later for nested saved-scene
dependency support. These public configurations differ from the paper's
evaluation instances. Historical releases remain available.

## Configuration bank 0.2.15

Version 0.2.15 replaces three saved initial states using original-rule seeded
candidates that passed their original native initial admission.

| Task identifier | Episode | Actual seed |
| --- | ---: | ---: |
| `coffee_bag_then_machine_interference` | 21 | 204 |
| `coffee_button_lighting_interference` | 14 | 73 |
| `failure_recovery_sofa` | 19 | 94 |

The bank remains at 49 groups of 30 configurations (1,470 total). The other
1,467 state records are byte-identical to 0.2.14. Public contexts, task
definitions, scoring rules, physical parameters, and shared assets 0.1.7 are
unchanged. The manifest references five new 0.2.15 archives and two unchanged
archives from earlier releases; do not rename or concatenate them.
Historical releases remain available. Use source commit
`f86a908d709fb94dcfc970be8c5da93305be9968` or later for nested saved-scene
dependency support. These public configurations differ from the paper's
evaluation instances.

## Configuration bank 0.2.14

Version 0.2.14 replaces fourteen saved initial states using original-rule seeded
candidates that passed their original native initial admission: ten coffee-
interference configurations and four OpenEnd episode configurations (three leaving
episodes and one tidying episode).

| Task identifier | Episode | Actual seed |
| --- | ---: | ---: |
| `coffee_bag_interference` | 21 | 171 |
| `coffee_button_lighting_interference` | 25 | 221 |
| `coffee_button_lighting_interference` | 28 | 231 |
| `coffee_machine_push_interference` | 10 | 241 |
| `coffee_machine_push_interference` | 12 | 251 |
| `coffee_machine_push_interference` | 13 | 261 |
| `coffee_machine_push_interference` | 14 | 271 |
| `coffee_machine_push_interference` | 23 | 281 |
| `coffee_machine_push_interference` | 28 | 291 |
| `coffee_machine_push_interference` | 30 | 311 |
| `prepare_to_leave` | 3 | 351 |
| `prepare_to_leave` | 17 | 371 |
| `prepare_to_leave` | 22 | 391 |
| `tidy_room` | 26 | 181 |

The bank remains at 49 groups of 30 configurations (1,470 total). The other
1,456 state records are byte-identical to 0.2.13. Public contexts, task
definitions, scoring rules, physical parameters, and shared assets 0.1.7 are
unchanged. The manifest references six new 0.2.14 archives and one unchanged
archive from an earlier release; do not rename or concatenate them.
Historical releases remain available.

## Configuration bank 0.2.13

Version 0.2.13 replaces two saved initial states using original-rule seeded
candidates that passed their original native initial admission:

| Task identifier | Episode | Actual seed |
| --- | ---: | ---: |
| `coffee_machine_push_interference` | 4 | 31 |
| `coffee_button_lighting_interference` | 21 | 211 |

The bank remains at 49 groups of 30 configurations (1,470 total). The other
1,468 state records are byte-identical to 0.2.12. Public contexts, task
definitions, scoring rules, physical parameters, and shared assets 0.1.7 are
unchanged. The manifest references four new 0.2.13 archives and three unchanged
archives from earlier releases; do not rename or concatenate them. Historical
releases remain available.

## Configuration bank 0.2.12

Version 0.2.12 replaces four saved initial states using original-rule seeded
candidates that passed their original native initial admission:

| Task identifier | Episode | Actual seed |
| --- | ---: | ---: |
| `tidy_room` | 2 | 141 |
| `coffee_machine_push_interference` | 5 | 41 |
| `task_discovery--prepare_dinner` | 26 | 71 |
| `task_discovery--prepare_to_leave` | 17 | 41 |

The bank remains at 49 groups of 30 configurations (1,470 total). The other
1,466 state records, including all previous replacements, are byte-identical
to 0.2.11. Public contexts, task definitions, scoring rules, physical parameters,
and shared assets 0.1.7 are unchanged. The complete manifest reuses one unchanged
archive part from 0.2.8 alongside six new 0.2.12 parts; do not rename or
concatenate them. Historical releases remain available.

## Configuration bank 0.2.11

Version 0.2.11 replaces eight saved initial states using original-rule seeded
candidates that passed their original native initial admission:

| Task identifier | Episode | Actual seed |
| --- | ---: | ---: |
| `host_guests` | 17 | 31 |
| `host_guests` | 26 | 41 |
| `prepare_dinner` | 26 | 61 |
| `tidy_room` | 17 | 151 |
| `task_discovery--host_guests` | 17 | 101 |
| `task_discovery--host_guests` | 26 | 111 |
| `task_discovery--prepare_to_leave` | 2 | 121 |
| `task_discovery--tidy_room` | 17 | 31 |

The bank remains at 49 groups of 30 configurations (1,470 total). The other
1,462 state records, including the nine 0.2.9 and three 0.2.10 replacements,
are byte-identical to 0.2.10. Public contexts, task definitions, scoring rules,
physical parameters, and shared assets 0.1.7 are unchanged. The complete manifest
reuses unchanged archive parts from 0.2.8 and 0.2.10 alongside the new 0.2.11
parts; do not rename or concatenate them. Historical releases remain available.

## Configuration bank 0.2.10

Version 0.2.10 replaces three saved initial states using original-rule seeded
candidates that passed their original native initial admission:

| Task identifier | Episode | Actual seed |
| --- | ---: | ---: |
| `prepare_to_leave` | 2 | 31 |
| `coffee_button_lighting_interference` | 13 | 61 |
| `coffee_button_lighting_interference` | 18 | 81 |

The bank remains at 49 groups of 30 configurations (1,470 total). The other
1,467 state records, including the nine 0.2.9 replacements, are byte-identical
to 0.2.9. Public contexts, task definitions, scoring rules, physical parameters,
and shared assets 0.1.7 are unchanged. The complete manifest reuses unchanged
archive parts from 0.2.8 and 0.2.9 alongside the new 0.2.10 parts; do not rename
or concatenate them. Historical releases remain available.

## Configuration bank 0.2.9

Version 0.2.9 replaces nine saved initial states using original-rule seeded
candidates that passed their original native initial admission:

| Task identifier | Episode | Actual seed |
| --- | ---: | ---: |
| `long_task_17` | 26 | 31 |
| `long_task_19` | 21 | 31 |
| `long_task_21` | 19 | 31 |
| `long_task_22` | 25 | 31 |
| `prepare_dinner` | 3 | 51 |
| `clear_table_trash` | 23 | 32 |
| `microwave_heat_take` | 14 | 41 |
| `microwave_heat_take` | 17 | 51 |
| `microwave_heat_take` | 28 | 61 |

The bank remains at 49 groups of 30 configurations (1,470 total). The other
1,461 state records are byte-identical to 0.2.8. Public contexts, task definitions,
scoring rules, and physical parameters are unchanged. Shared assets remain at
0.1.7. The download reuses four unchanged 0.2.8 archive parts and provides three
new 0.2.9 parts; use the complete manifest rather than renaming or concatenating
parts. Historical releases remain available.

## Configuration bank 0.2.8

Version 0.2.8 replaces `coffee_bag_then_machine_interference` episode 7 (seed 71),
`coffee_button_lighting_interference` episode 8 (seed 51), and
`failure_recovery_sofa` episode 3 (seed 66). Each passed its original
native initial admission. The other 1,467 state records are byte-identical
to 0.2.7. Public contexts, task definitions, scoring, and physical parameters
are unchanged. Shared assets 0.1.7 provide the additional compact lossless
scene dependency; all earlier asset archives remain unchanged.

## Configuration bank 0.2.7

Version 0.2.7 replaces `coffee_bag_interference` episode 6 (seed 51)
and `coffee_bag_then_machine_interference` episode 6 (seed 66). Both
passed their original native initial admission. The other 1,468 state
records are byte-identical to 0.2.6. Public contexts, task definitions,
scoring rules, and physical parameters are unchanged. Shared assets
remain at version 0.1.6; earlier releases remain available.

## Configuration bank 0.2.6

Version 0.2.6 replaces three saved initial states using original-rule seeded
candidates that passed native initial admission: `coffee_button_lighting_interference`
episode 4 (seed 41), `coffee_bag_then_machine_interference` episode 5 (seed 51),
and `failure_recovery_sofa` episode 2 (seed 58). Public requests and contexts,
task definitions, scoring rules, and physical parameters are unchanged.
The other 1,467 state records are byte-identical to 0.2.5, including its
`host_guests` and `task_discovery--tidy_room` episode-2 replacements. Earlier
versions remain available.
Room bases are stored once in the shared asset release;
saved configurations contain compact lossless patches rather than complete
room copies.

## What is stored

The configuration bank contains small state records: the selected shared
layout and asset identifiers, task parameters, object transforms, robot base
pose and joint initialization, task-owned initial-state attachments, and
compact lossless scene patches. Cloth tasks retain their required material-state
arrays. Public scenario text
is stored separately from evaluator bindings.

It does **not** distribute a complete room/mesh/texture copy per configuration.
Room geometry, furniture, lights, robot and task objects come from the shared
[asset release](ASSETS.md). The native environment assembles a local scene
from those resources and the saved state when preparing a run. These generated
runtime files are not additional downloads.

## 1. Install code and shared assets

Follow [Installation](INSTALL.md), then configure a fresh task workspace:

Use configuration bank **0.2.17** with shared assets **0.1.7**, which include the
complete library of 50 original room layouts and scene-preparation dependencies. The bank's
original base-asset provenance remains 0.1.0; the current asset release preserves
those resources and adds the shared layouts, path bindings, and dependencies.

```sh
robotagentbench assets fetch resources/assets-0.1.7.json --cache downloads
robotagentbench assets install resources/assets-0.1.7.json --cache downloads --destination Assets
robotagentbench assets configure resources/assets-0.1.7.json --destination Assets --runtime Runtime --profile room
```

Use the task's matching profile for dedicated cloth or specialized owners;
do not replace those owners with the ordinary room worker.

## 2. Download the configuration bank

The bank is approximately **260 MB** across seven archives; the shared simulation
assets are installed separately and are not duplicated in this download.

```sh
python -m robotagentbench.simulation.saved_instance_release.download \
  --manifest resources/saved-instances-0.2.17.json \
  --cache downloads/configurations --destination /absolute/path/to/SavedConfigurations
```

The manifest pins archive sizes, member hashes and an immutable Hugging Face
revision. Downloads reuse verified archive parts. Installation requires a new
destination and rejects unexpected members or unsafe paths. For offline
installation, supply the archives at their manifest-relative cache paths and
add `--offline`.

When upgrading from an earlier release, use a new destination directory; do not overlay an
existing configuration bank. Keep the earlier bank if it is used by a saved run.

## 3. Restore a selected initial state

For the model walkthroughs, use the [shared Docker setup](MODEL_QUICKSTART.md)
and let the image supply the native SDK:

```bash
"$RAB_PY" -m robotagentbench.model_examples.docker_workflow restore \
  --runtime "$RAB_ROOT/Runtime" --image robotagentbench:native \
  --release "$RAB_ROOT/SavedConfigurations/release.json" \
  --task prepare_coffee --episode 1 --output "$RAB_ROOT/Restored/coffee01"
```

For Codex, stage the coding Runtime first and use that Runtime here, as shown
in its walkthrough. Restoration is CPU-only and does not start physics.

Advanced operators with an existing matching native Python/SDK environment
can instead invoke the original restorer directly:

```sh
python -m robotagentbench.simulation.saved_instance_release.restore \
  --release /absolute/path/to/SavedConfigurations/release.json \
  --runtime /absolute/path/to/Runtime \
  --task prepare_coffee --episode 1 \
  --output /absolute/path/to/Restored/coffee01 \
  --sdk-site /absolute/path/to/native/site-packages --execute
```

Omit `--execute` to describe the selection without writing files or importing
USD. Execution prepares the scene from the saved recipe and restores the saved
initial state; it does not run a model, start physics, or resample the episode.
Use the native CPython 3.11 environment specified in [Installation](INSTALL.md).
For containers, run this command inside the configured source/asset namespace.

For cloth and specialized tasks, also pass `--owner-config OPERATOR` with the
matching dedicated environment configuration from [Cloth](CLOTH.md) or
[specialized owners](SPECIALIZED.md). Restoration reuses that owner's native
settings and verification; it does not substitute the ordinary room owner.

The output contains `selection.json` and a restoration receipt. Generated task
files are placed automatically in the original task owner's fresh Runtime
directory. Existing output directories are never overwritten. Keep the
receipt with the evaluation result.

## 4. Run an agent

Use the Runtime and prepared launch from `selection.json` with the task's
original worker. Where provided, forward the selection's read-only input roots
and hash-bound public instruction/context. Operator metadata, task manifests,
answer bindings and private evaluator state are not policy observations.

For the ordinary restored room/short-task format, `runtime`, `source_root`,
`launch` and `public_input_binding` are **top-level fields** in `selection.json`.
There is no `operator.public_input_binding` nesting in this format. The model
walkthrough helper reads those fields and supplies the trusted source root and
public-text binding automatically; do not substitute a target/answer file.

- [InternVLA-A1.5 example](../src/robotagentbench/model_examples/internvla/README.md)
- [Codex Robo Agent example](../src/robotagentbench/model_examples/codex/README.md)
- [Cloth owner](CLOTH.md) and [specialized owners](SPECIALIZED.md)

Use a fresh run output, the host's shared exclusive-GPU lock directory, and
normal worker shutdown. Restoring an initial state is not a task result;
process scores and completion come from verified physical execution by the
original task owner.
