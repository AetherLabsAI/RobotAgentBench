"""Shared Family-4 initial-state draw used by task-owned entrypoints.

The benchmark task key is supplied by the per-task wrapper.  Historical
generator labels are passed through only as producer-routing provenance; they
are never treated as benchmark identities.
"""

from __future__ import annotations

import numpy as np


REFERENCE_SCALES = {
    "base_dxy_m": 0.05,
    "base_dyaw_deg": 5.0,
    "arm_joint_deg": 4.0,
}


def sample(seed: int, scales: dict[str, float] | None = None) -> dict:
    values = dict(REFERENCE_SCALES)
    if scales:
        values.update(scales)
    rng = np.random.default_rng(seed)
    return {
        "seed": int(seed),
        "base_dxy_m": rng.uniform(
            -values["base_dxy_m"], values["base_dxy_m"], 2,
        ).round(4).tolist(),
        "base_dyaw_deg": round(float(rng.uniform(
            -values["base_dyaw_deg"], values["base_dyaw_deg"])), 3),
        "arm_noise_deg": round(float(values["arm_joint_deg"]), 3),
        "arm_noise_seed": int(seed) * 7 + 1,
    }


def generator_args(draw: dict) -> list[str]:
    dx, dy = draw["base_dxy_m"]
    return [
        f"--base-dxy-m={dx},{dy}",
        f"--base-dyaw-deg={draw['base_dyaw_deg']}",
        f"--arm-noise-deg={draw['arm_noise_deg']}",
        f"--arm-noise-seed={draw['arm_noise_seed']}",
    ]


def sample_and_apply(
    *, task_key: str, producer_template_label: str,
    seed: int, mode: str, runtime: object,
) -> dict:
    draw = sample(seed)
    arguments = generator_args(draw)
    if runtime is not None and hasattr(runtime, "run_generator"):
        result = runtime.run_generator(
            task_key=task_key,
            producer_template_label=producer_template_label,
            extra_args=arguments,
            draw=draw,
        )
    else:
        result = {"deferred": True, "generator_args": arguments}
    return {
        "benchmark_task_key": task_key,
        "producer_template_label": producer_template_label,
        "draw": draw,
        "mode": mode,
        "result": result,
    }
