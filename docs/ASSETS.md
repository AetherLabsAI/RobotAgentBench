# Simulation assets

The asset release contains rooms, objects, robot resources, materials,
initial-state arrays and the layout/collision data needed by the task backends.
Task code is included in the Python package; Isaac Sim and its SDK are installed
separately.

Asset repository: [DylanHel/RobotAgentBench-assets](https://huggingface.co/datasets/DylanHel/RobotAgentBench-assets).
The manifest shipped with the code pins the exact dataset revision and file hashes.

## Install

Place all archives from the matching asset release in `downloads`. From the
code repository root, run:

```sh
robotagentbench assets install resources/assets-0.1.0.json \
  --cache downloads --destination Assets
```

No network access is needed for installation. The destination must be new.
The installer checks every
archive member and rejects unexpected files, path traversal and symbolic links.
Existing installations are never overwritten.

The installation contains a deduplicated `objects/` store and the resource index
selected by the release manifest, mapping task resources to exact content versions.
Use runtime configuration to assemble a task's local source and asset namespace.

The full Context Reasoning question banks and standard answers are included for
reproduction. Answer files are evaluator resources, not agent observations.

## Downloading a hosted release

For a manifest with a configured Hugging Face dataset and immutable revision:

```sh
robotagentbench assets fetch resources/assets-0.1.0.json --cache downloads
```

The downloader verifies archive size and SHA-256 and reuses valid cached files.
An interrupted download is retained as `.partial` for inspection.

## Configure a task workspace

```sh
robotagentbench assets configure resources/assets-0.1.0.json \
  --destination Assets --runtime Runtime --profile room
```

Select `room`, `atomic_coffee`, `failure_recovery`, or `cloth`. This creates a
fresh workspace containing the bundled source, selected resource bindings and
`Runtime/runtime.json`. It does not alter the installed package or an existing
simulation environment. Historical resource names are confined to the job's
namespace; they do not require access to the original server.

## Verify an installation

```sh
robotagentbench assets verify resources/assets-0.1.0.json --destination Assets
```

Code and assets must use matching release versions. Keep the selected manifest
with evaluation results so that the same source and resource versions can be
identified later. Asset and third-party terms are separate from the code
license. Team-authored resources use CC BY 4.0, while third-party resources
retain their own terms. The manifest includes a license archive so the notices
accompany automatic downloads and installation; see [third-party notices](THIRD_PARTY.md).
