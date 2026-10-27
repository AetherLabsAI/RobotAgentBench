# Saved cloth instances

The cloth operator supports short tasks `spread_tablecloth` and
`collect_tablecloth`, plus composite task `long_task_18` (paper long task 4).

Instance preparation is separate from execution:

```bash
python -m robotagentbench.simulation.cloth.launcher OPERATOR.json \
  --task spread_tablecloth --prepare-variants NEW_BANK --seed 1 --count 30

python -m robotagentbench.simulation.cloth.instance_bank NEW_BANK/index.json --seed 7

python -m robotagentbench.simulation.cloth.launcher OPERATOR.json \
  --task spread_tablecloth --execute NEW_BANK/seed-0000000007/launch.json \
  --output NEW_RUN
```

`OPERATOR.json` is the installed trusted native-environment configuration.
The bank must be inside its configured input roots, and the run must be inside
its output root. Preparation and lookup do not start Isaac or query a GPU.
Execution uses the existing exclusive-device owner and native initial checks.

Each saved instance contains the exact initial material nodes, its seed,
planar cloth transforms, existing room-layout and wall-colour choices,
source hashes, and the launch configuration. Loading does not regenerate them.

The approved cloth reset is at most 3 cm independently along X and Y and at
most 10 degrees of yaw. It preserves cloth height, original folded/unfolded
shape, UVs, material topology, robot state, and task workstation. Geometry
screening rejects tabletop overhang and crossings of the original basket
walls. The original basket initialization includes an above-rim drape; that
shape is retained. Background furniture layouts and wall colours come from
the existing house catalog. No lights, physical properties, scoring rules,
camera settings, control rates, or timeouts change.

CPU-prepared instances remain subject to the original native settling and
admission checks when executed; preparation is not an episode success result.
