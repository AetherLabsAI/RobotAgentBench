# Task resource requirements

The core package has no third-party Python runtime dependencies. Native tasks
need an external simulator installation, a bundled task-source profile, and
the matching asset release. This guide identifies the resource families used
by each task. For download and local workspace setup, see [assets](ASSETS.md).
Configure native file identities as described in [dependency configuration](DEPENDENCIES.md).

## Shared resources

| Resource | When required |
| --- | --- |
| Isaac Sim, its interpreter, extensions, native libraries and material resolvers | Native execution; use the selected environment profile, not packages from an unrelated Python installation |
| Robot articulation, meshes, collision geometry and calibration | Every physical task; the AgiBot robot asset is obtained separately |
| Selected room layout, static geometry, surfaces and physics layers | Tasks using that layout; specialized owners retain their original scene preparation |
| Registry, scene preparation, native control, fact readers and scorers | The selected original task owner; retain its source-record pins |
| Initial-state, posture and source-episode resources | Where required by the task's preparation or failed-start recipe |
| Code-tool broker and sandbox resources | Only when using the original code-tool execution mode |

Model weights, service credentials and agent clients are not dependencies of
the core evaluator. Demonstration videos are not needed merely to query the
catalog. Source-episode metadata used for initialization can still be a native
task prerequisite; do not remove it just because it came from a demonstration.

## Per-task resources

Entries below are runtime identifiers, not paper numbering or episode counts.
They add to the shared resources above. Object variants and quantities are
selected by the episode, not by copying the whole asset library.

| Task identifier(s) | Task-specific resource families |
| --- | --- |
| `prepare_coffee` | Coffee machine and controls, cup, bean bag, bean/fluid resources and station reference geometry |
| `bake_bread` | Articulated air fryer and basket, selected pastry and plate |
| `stock_wooden_tray` | Selected shelf products, shelf and small wooden tray |
| `retrieve_from_drawer`, `sofa_to_drawer` | Selected belongings, articulated drawer, source/support surfaces |
| `hang_cap` | Cap and coat rack with valid support geometry |
| `clear_table_trash` | Selected waste objects and assigned bins |
| `fridge_to_heated_box` | Articulated refrigerator and microwave, selected food box, heating/control resources and any selected carrier plate |
| `microwave_heat_take` | Microwave, selected payload and plate variant, heating/control resources and extraction initial state |
| `microwave_to_meal_tray` (legacy component, not one of the 18 short tasks) | Heated-payload initial-state contract, microwave, payload, optional plate and large meal tray; no standalone benchmark launch is provided |
| `spread_tablecloth`, `collect_tablecloth`, `clear_tablecloth` | Selected cloth, its physics resources, support surfaces and any required storage container; preserve the selected owner's execution/scoring availability |
| `pack_object` | Selected belonging, open tote bag and source/support surfaces |
| `prepare_cutlery` | Requested cutlery instances, source surface and large meal tray |
| `carry_meal_tray` | Meal tray, selected contents, source and destination supports |
| `sofa_to_coffee_table` | Selected sofa object, sofa and designated tabletop |
| `table_to_storage_basket` | Selected table object, source table and item-storage basket |
| `stock_wooden_tray_interference` | Shelf pickup resources plus the human-arm intervention model, rig and motion resources |
| `stock_wooden_tray_lighting` | Shelf pickup resources and native scene-light configuration |
| `select_drink`, `select_snack` | Active shelf-product inventory, context inputs and separately provisioned evaluator selection data |
| `select_trash` | Active waste-selection catalog, candidate geometry/materials, context inputs and evaluator selection data |
| `select_sofa` | Candidate belongings, sofa, context inputs and evaluator selection data |
| `coffee_bag_interference` | Original atomic-coffee source/interpreter profile, bean-pouring scene and bag-intervention resources |
| `coffee_machine_push_interference` | Original atomic-coffee profile, pouring scene and machine-intervention resources |
| `coffee_bag_then_machine_interference` | Original atomic-coffee profile and both intervention resource sets |
| `coffee_button_lighting_interference` | Original atomic-coffee profile, button/dispensing scene and native lighting resources |
| `failure_recovery_shelf_drink` | Original recovery owner, shelf-drink source episode and failed-grasp initialization |
| `failure_recovery_coffee_bag` | Original recovery/coffee-worker profiles, bean scene and failed-grasp initialization |
| `failure_recovery_sofa` | Original recovery owner, selected sofa object and failed-grasp initialization |
| `failure_recovery_table_waste_empty_grasp`, `failure_recovery_table_waste_tipped_cup` | Original recovery owner, cup/bin resources and the selected failed-state recipe |
| `wipe_table` | Brush, stain grid/textures, support surface and wiping-state resources |
| `openend`, `task_discovery` | Selected layout and episode-specific union of objects, appliances, destinations, context and component resources |
| `long_task_15`, `long_task_16`, `long_task_17`, `long_task_18`, `long_task_19`, `long_task_20`, `long_task_21`, `long_task_22` | Selected layout and the union of the long task's component resources and initial conditions |

Long-task identifiers above retain their implementation numbering even when
the paper numbers the eight long tasks from 1 to 8.
The three native cloth routes use their own [operator entry](CLOTH.md), bundled
`cloth` source profile, and cloth state/geometry data from the asset release.

## Constructing a minimum installation

Use the selected owner's source record and prepared-scene manifest, then follow
composed USD references, sublayers, payloads, textures and material dependencies.
Retain required shared geometry and preparation helpers. An inventory of an
asset directory is not a minimum closure: backups, previews and inactive catalog
revisions are not automatically needed, while a shared texture outside a task
directory can be essential.

Deduplicate identical resources in an installation manifest without changing
their original identities or required aliases. Keep different source profiles
separate even when relative filenames match. Never reduce a source record by
dropping a file still checked by the original owner, or refresh a hash to accept
a newer scorer. Scorer, fact schema and evidence reader must form a compatible
reviewed bundle.

The asset manifest identifies the versioned archive members. Scene preparation
selects the resources needed by the episode. Evaluator-private answers and
target bindings are not agent observations. For resource licensing and
distribution terms, see [third-party notices](THIRD_PARTY.md).
