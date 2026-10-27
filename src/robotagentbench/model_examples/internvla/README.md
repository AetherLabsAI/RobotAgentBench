# Run InternVLA-A1.5-RAB

This walkthrough runs the published G1 policy on **Prepare coffee, saved
episode 1**. Complete [shared setup](../../../../docs/MODEL_QUICKSTART.md) first
and keep its shell variables (`RAB_ROOT`, `RAB_PY`, `RAB_LOCKS`, `DOCKER_HOST`).
Isaac runs in Docker; the model runs in a separate Conda environment on a
different GPU. Neither task scoring nor model preprocessing is changed.

## 1. Install the model environment

Use the pinned [upstream implementation](https://github.com/InternRobotics/InternVLA-A-series/tree/e6fc904f9edbfb14532e97095fc2372202517f76).
The commands explicitly name the clone directory and check out the matching
revision. CUDA 12.8 build tools must be available for the attention kernels.

```bash
cd "$RAB_ROOT"
mkdir -p third_party
git clone https://github.com/InternRobotics/InternVLA-A-series.git third_party/InternVLA-A-series
git -C third_party/InternVLA-A-series checkout --detach e6fc904f9edbfb14532e97095fc2372202517f76

conda create -y -n rab-internvla python=3.11
conda activate rab-internvla
export RAB_MODEL_PY="$CONDA_PREFIX/bin/python"
python -m pip install --upgrade pip
conda install -c conda-forge ffmpeg svt-av1 -y
python -m pip install torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install transformers==5.2.0
python -m pip install -e "$RAB_ROOT/third_party/InternVLA-A-series"
python -m pip install flash-attn==2.8.3 flash-linear-attention==0.5.0 causal-conv1d==1.6.1 \
  --no-build-isolation

# Apply upstream's Transformers replacements only in this new model environment.
RAB_TRANSFORMERS_DIR="$(python -c 'import transformers; print(next(iter(transformers.__path__)))')"
for policy in pi0 pi05 internvla_a1_5; do
  cp -r "$RAB_ROOT/third_party/InternVLA-A-series/src/lerobot/policies/$policy/transformers_replace/models" \
    "$RAB_TRANSFORMERS_DIR/"
done
python -m pip install "$RAB_ROOT"
python -m pip check
```

Do not proceed if dependency/kernel installation fails. Follow the pinned
[upstream installation notes](https://github.com/InternRobotics/InternVLA-A-series/blob/e6fc904f9edbfb14532e97095fc2372202517f76/tutorials/installation.md)
for the matching CUDA/PyTorch build. Wan/VGM training weights and training
datasets are not required for this action-only example.

## 2. Download the fine-tuned model

```bash
cd "$RAB_ROOT"
"$RAB_PY" -m pip install huggingface_hub
"$RAB_ROOT/.venv/bin/hf" download AetherLabs-AI/InternVLA-A1.5-RAB \
  --revision c158066761879f59224a9f939c237b6f9ac3e687 \
  --local-dir "$RAB_ROOT/weights/InternVLA-A1.5-RAB"
(cd "$RAB_ROOT/weights/InternVLA-A1.5-RAB" && sha256sum -c SHA256SUMS)
```

The [model release](https://huggingface.co/AetherLabs-AI/InternVLA-A1.5-RAB)
includes `config.json`, `model.safetensors`, `stats.json`,
`control_schema.json` and `vlm/`. The **downloaded directory itself** is the
checkpoint directory. The helper uses its matching `G1_omnipicker` statistics
and control schema; no `G1/pretrained_model` subdirectory is needed.

## 3. Restore one saved episode

```bash
"$RAB_PY" -m robotagentbench.model_examples.docker_workflow restore \
  --runtime "$RAB_ROOT/Runtime" --image robotagentbench:native \
  --release "$RAB_ROOT/SavedConfigurations/release.json" \
  --task prepare_coffee --episode 1 --output "$RAB_ROOT/Restored/internvla-coffee01"
```

This uses the native SDK **inside the image**, without starting physics or
sampling another initial state. It writes `selection.json`; task files stay
in the Runtime's source directory. Change `--episode` to select another of the
30 release configurations, using a new restore output directory.

## 4. Configure the worker and model

Choose free GPUs: the example uses **GPU 0 for Isaac and GPU 1 for InternVLA**.
The model GPU must have more than 20,000 MiB free. Both use `RAB_LOCKS`.

```bash
"$RAB_PY" -m robotagentbench.model_examples.docker_workflow configure \
  --agent internvla --image robotagentbench:native \
  --selection "$RAB_ROOT/Restored/internvla-coffee01/selection.json" \
  --name internvla_coffee01 --gpu 0 --model-gpu 1 --lock-root "$RAB_LOCKS" \
  --model-python "$RAB_MODEL_PY" --model-dir "$RAB_ROOT/weights/InternVLA-A1.5-RAB" \
  --destination "$RAB_ROOT/model-configs/internvla-coffee01"
```

This verifies model files, prepares fresh image-owned native caches, and writes
both configurations. It reads the selected Runtime, launch and top-level
`public_input_binding` automatically. No model inference or simulation occurs.
For a one-chunk connection check, add `--max-calls 1` **when configuring a new
run**; this is not a complete task evaluation.

## 5. Inspect, then run

```bash
# No simulator or model starts here.
"$RAB_PY" -m robotagentbench.model_examples.docker_workflow run \
  "$RAB_ROOT/model-configs/internvla-coffee01/run.json"

# After accepting the NVIDIA terms in shared setup:
"$RAB_PY" -m robotagentbench.model_examples.docker_workflow run \
  "$RAB_ROOT/model-configs/internvla-coffee01/run.json" --execute
```

The printed output directory contains `report.json`, model predictions,
acknowledgments and `native/native_report.json`. The original native report
determines process score, success and termination. Use a fresh configuration
name/output for another run; do not restart a dispatched run.

## Model interface and scope

Inputs are the public instruction, three full native RGB views and a measured
23-dimensional robot state. The policy predicts 50-step chunks, decoded into
22 physical action channels through the original 20 Hz adapter. Map and
separate context fields remain available from the worker but are not encoded
as extra inputs by this policy. See [interface details](SERVER_REUSE.md).

This example uses the standard room/short-task owner. Dedicated cloth and
specialized Recovery/Robustness tasks retain their own bindings. Advanced
operators with an existing host SDK can use [native setup](NATIVE_SETUP.md).

The release connection check executed one real 50-action chunk and closed
normally; it was not a full-task success result. The upstream implementation
retains [CC BY-NC-SA 4.0](https://github.com/InternRobotics/InternVLA-A-series/blob/e6fc904f9edbfb14532e97095fc2372202517f76/LICENSE);
downloaded model components retain the licenses in their model card.
