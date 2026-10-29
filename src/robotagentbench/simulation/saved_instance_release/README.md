# Saved initial configurations

The configuration release stores selected task parameters, asset choices,
object transforms, robot base/joint state, and existing placement recipes. It
references shared room bases and bundled task source, with compact lossless
scene patches where required, instead of shipping a full room USD for every
episode. Cloth states include their recorded small
NPZ initial-state arrays. These pregenerated configurations are separate from
the evaluation instances used in the paper.

Use configuration bank **0.2.17** with evaluator commit
`e8448f732572cacf330554c3620d6f25341be81f` or later.

## Download

Install the matching Python package and base assets first, then download and
safely install the independent, checksum-pinned gzip archives:

```bash
python -m robotagentbench.simulation.saved_instance_release.download \
  --manifest resources/saved-instances-0.2.17.json \
  --cache .cache/saved-instances --destination /your/SavedConfigurations
```

Use `--offline` with already downloaded archives. No manual concatenation or
unrestricted extraction is required. Existing destinations are refused.

## Restore a saved configuration

After configuring the appropriate local Runtime and native dependencies:

```bash
python -m robotagentbench.simulation.saved_instance_release \
  --release /your/SavedConfigurations/release.json \
  --task prepare_coffee --episode 1 --runtime /your/Runtime \
  --output /your/restore-coffee-01 --sdk-site /your/isaac/site-packages \
  --execute
```

Without `--execute`, the command describes the selection without creating
outputs or importing USD. Restoration uses the saved placement recipe and
recorded final initial state. It does not ask the room planner to select a new
layout, task composition, or initial pose. The original composer may perform
its existing seeded component construction; saved transforms and joint state
remain authoritative. CPU preparation is not a physics run.

The generated selection points to the original task owner's prepared launch,
manifest, public-input binding, and source Runtime. Use the model examples to
run it. Dedicated cloth and specialized failed-state/intervention owners retain
their own configured entry; a generic room worker is never silently substituted.
Only `public_input` is model-visible. Targets, task bindings and other evaluator
metadata are operator inputs, not observations.

Episode ordinal and original accepted generation seed are recorded separately.
Configuration identity comes from the saved numeric state and asset/layout
choices, not a changed filename or seed. Preparation does not claim a completed
robot task or model success.

The older `Builder` and `materialize` Python functions remain for legacy snapshot
readers. They are not the published compact format or recommended workflow.
