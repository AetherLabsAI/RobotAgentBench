<div align="center">

<h1><img src="docs/images/robotagentbench-banner.png" alt="RobotAgentBench — Towards Open-world Robot Autonomy" width="100%"></h1>

A benchmark for household robot agents, from individual manipulation tasks to open-ended assistance.

[Website](https://aetherlabsai.github.io/RobotAgentBench/) · [Assets on Hugging Face](https://huggingface.co/datasets/DylanHel/RobotAgentBench-assets) · [Quick start](#quick-start) · [Benchmark](docs/BENCHMARK.md) · [Documentation](#documentation)

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square)](docs/INSTALL.md)
[![Isaac Sim](https://img.shields.io/badge/Native_profile-Isaac_Sim_5.1-76B900?style=flat-square)](docs/INSTALL.md#native-environment)
[![License: Apache 2.0](https://img.shields.io/badge/Code-Apache_2.0-0F766E?style=flat-square)](LICENSE)
[![Assets](https://img.shields.io/badge/Hugging_Face-Assets-FFD21E?style=flat-square)](https://huggingface.co/datasets/DylanHel/RobotAgentBench-assets)

</div>

<!-- overview-video:start -->
<p align="center">
  <a href="docs/media/overview.mp4"><img src="docs/images/robotagentbench-demo-poster.png" alt="Watch the RobotAgentBench demo" width="100%"></a><br>
  <strong><a href="docs/media/overview.mp4">▶ Watch RobotAgentBench in action</a></strong>
</p>
<!-- overview-video:end -->

## Overview

**RobotAgentBench** is a simulation benchmark for household robot agents. Using the **AgiBot G1** embodiment, agents observe the home, identify what needs to be done, plan their actions, and execute them through a shared observation/action interface.

The benchmark spans everyday manipulation, task discovery, context reasoning, failure recovery, robustness, and open-ended scenarios. Task-specific process scores measure verified physical progress, including partial completion.

This repository contains the task backends, initialization and randomization, scoring, and evaluation runtime. Versioned simulation assets are available separately on [Hugging Face](https://huggingface.co/datasets/DylanHel/RobotAgentBench-assets).

<p align="center">
  <a href="docs/images/robotagentbench-overview.png"><img src="docs/images/robotagentbench-overview.png" alt="RobotAgentBench overview: task-level capabilities and scenario-level household assistance" width="100%"></a>
</p>

## Benchmark at a glance

| Evaluation | Scope | What it measures |
| :-- | :-- | :-- |
| **Manipulation** | 18 short tasks | Executing explicit household manipulation instructions |
| **Long tasks** | 8 composite tasks | Coordinating several subgoals in one continuous scene |
| **Task Discovery** | Scenario-conditioned exploration | Identifying and touching task-relevant objects |
| **Context Reasoning** | Drink, snack, waste, and sofa-object selection | Choosing the correct object from context |
| **Failure Recovery** | Predefined failed-action states | Recognizing a failed state and adapting subsequent actions |
| **Robustness** | Object movement and lighting disturbances | Continuing the objective as execution conditions change |
| **OpenEnd** | Dinner, guests, leaving, and tidying | Inferring, planning, and executing tasks from a high-level request |

The **18 short tasks** contribute to the Manipulation score. The **8 long tasks** are evaluated separately and do not contribute to that score.

See the [benchmark guide](docs/BENCHMARK.md) for the task list and scoring distinctions.

## Quick start

### 1. Install the code

Use **Python 3.11 or newer** and check out the evaluation code's `main` branch:

```sh
git clone --branch main https://github.com/AetherLabsAI/RobotAgentBench.git
cd RobotAgentBench
python3 -m venv .venv
. .venv/bin/activate
python -m pip install .
python -m robotagentbench --help
robotagentbench catalog standalone
```

The core package and catalog inspection do not require Isaac Sim, a GPU, or model credentials. Native simulation uses a separate environment described below.

### 2. Download assets

The [asset release](https://huggingface.co/datasets/DylanHel/RobotAgentBench-assets) is approximately **4.55 GB**, split into **18 archives**, including the license archive. It contains scene and robot resources, task objects, initialization data, and the full Context Reasoning question banks and standard answers.

```sh
robotagentbench assets fetch resources/assets-0.1.0.json --cache downloads
robotagentbench assets install resources/assets-0.1.0.json --cache downloads --destination Assets
robotagentbench assets configure resources/assets-0.1.0.json --destination Assets --runtime Runtime --profile room
```

The manifest pins an immutable asset revision and verifies file hashes. The commands create fresh `Assets` and `Runtime` directories; existing installations are not overwritten. For offline setup, place all archive parts in `downloads` and skip `fetch`. See the [asset guide](docs/ASSETS.md) for details.

Standard answers are evaluator data and are not included in agent observations.

### 3. Prepare and run an example

Native execution requires **Linux x86_64**, **Docker**, **NVIDIA Container Toolkit**, and an **Isaac Sim-compatible NVIDIA GPU**. Build the pinned native environment, then prepare the coffee task:

```sh
docker build -f docker/Dockerfile -t robotagentbench:native .
python -m robotagentbench.namespace --runtime Runtime --image robotagentbench:native --plan
python -m robotagentbench.namespace --runtime Runtime --image robotagentbench:native --prepare --name coffee_01
```

Review the [NVIDIA license setup](docs/EXAMPLE.md#4-start-the-task). If you accept the applicable terms, start the task worker:

```sh
export OMNI_KIT_ACCEPT_EULA=YES
mkdir -p gpu-locks
python -m robotagentbench.namespace --runtime Runtime --image robotagentbench:native \
  --run --name coffee_01 --gpu 0 --lock-root gpu-locks
```

The worker exposes a JSONL observation/action interface for an agent or recorded action source. Starting the worker does not run a policy automatically. Follow the [coffee walkthrough](docs/EXAMPLE.md) for interaction examples and the [runtime guide](docs/RUNTIME.md) for agent integration. Use the same GPU-lock directory for all jobs on a host.

## Evaluation interface

Agents receive synchronized camera observations and robot-state feedback, and submit physical action chunks. The standard room profile uses one head camera and two wrist cameras with a 20 Hz control interface. Each task handles its own process scores, completion conditions, and termination.

Results include process scores, completion status, and a termination reason. Setup and runtime errors are reported separately from scored episodes. Camera, physics, and execution settings for specialized tasks are described in the [task backend guides](docs/BACKENDS.md).

## Documentation

| Guide | Start here to… |
| :-- | :-- |
| [Benchmark](docs/BENCHMARK.md) | Understand task families, scenarios, and evaluation scope |
| [Installation](docs/INSTALL.md) | Install the evaluator and native dependencies |
| [Coffee example](docs/EXAMPLE.md) | Install assets, prepare an episode, and connect to the task worker |
| [Task backends](docs/BACKENDS.md) | Locate task implementations and execution profiles |
| [Assets](docs/ASSETS.md) | Download, install, and verify a versioned asset release |
| [CLI reference](docs/CLI.md) | Inspect catalogs, verify resources, and invoke configured native operators |
| [Tablecloth tasks](docs/CLOTH.md) | Configure the separate cloth environment |
| [Recovery and coffee variants](docs/SPECIALIZED.md) | Configure specialized execution environments |
| [Runtime](docs/RUNTIME.md) | Integrate workers, observations, actions, scoring, and shutdown |
| [Licenses and notices](docs/THIRD_PARTY.md) | Read asset and dependency licensing terms |

## Repository structure

```text
src/robotagentbench/
├── tasks/          Task and OpenEnd component catalogs
├── backends/       Task implementations and versioned source profiles
├── interfaces/     Robot state and action contracts
├── runtime/        Action transport and session adapters
└── simulation/     Source admission, native workers, and resource lifecycle
docs/               Setup, benchmark, integration, and resource guides
tests/              CPU contract and lifecycle tests
```

## Contributing

For questions and bug reports, use [GitHub Issues](https://github.com/AetherLabsAI/RobotAgentBench/issues). Include the task identifier, code revision, execution profile, and a minimal reproduction. Do not include credentials or sensitive configuration.

Run the CPU tests from the repository root:

```sh
python -m unittest discover -s tests
```

## License

- **Evaluation code:** [Apache-2.0](LICENSE).
- **Team-authored assets and task data:** [CC BY 4.0](https://huggingface.co/datasets/DylanHel/RobotAgentBench-assets/blob/main/LICENSE.md).
- **Third-party materials:** retain their original terms, including CC BY-NC-SA 4.0 for the AgiBot GenieSim robot resources. See [third-party notices](docs/THIRD_PARTY.md) and the [asset notices](https://huggingface.co/datasets/DylanHel/RobotAgentBench-assets/blob/main/NOTICE.md).
