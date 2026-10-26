#!/usr/bin/env python3
"""Read-only discovery links for owner-managed Kitchen producers."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


PATH = Path(__file__).with_name("external_kitchen_producers.json")
SCHEMA = "real2sim.external_kitchen_producer_links.v1"


def load(path: Path = PATH) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != SCHEMA or not isinstance(value.get("links"), list):
        raise ValueError("invalid external Kitchen producer links")
    seen: set[str] = set()
    for link in value["links"]:
        tasks = tuple(map(str, link.get("benchmark_tasks", ())))
        if not tasks or seen.intersection(tasks):
            raise ValueError("external producer task links must be non-empty and unique")
        seen.update(tasks)
        if link.get("canonical_scene_compatibility") not in {
            "pending_replay", "passed", "failed"
        }:
            raise ValueError("invalid canonical scene compatibility status")
        if not link.get("host") or not link.get("workspace") or not link.get("entrypoints"):
            raise ValueError("external producer link lacks a source location")
    return value


LINKS = load()["links"]


def for_task(task_id: str) -> list[dict[str, Any]]:
    task = str(task_id)
    return [
        copy.deepcopy(link)
        for link in LINKS
        if task in set(map(str, link["benchmark_tasks"]))
    ]


def summary() -> dict[str, Any]:
    tasks = sorted(
        {str(task) for link in LINKS for task in link["benchmark_tasks"]},
        key=lambda value: tuple(map(int, value.split("."))),
    )
    return {
        "schema": SCHEMA,
        "link_count": len(LINKS),
        "task_count": len(tasks),
        "tasks": tasks,
        "canonical_scene_passed": sorted(
            {
                str(task)
                for link in LINKS
                if link["canonical_scene_compatibility"] == "passed"
                for task in link["benchmark_tasks"]
            }
        ),
        "execution_policy": "read_only_never_import_or_execute",
    }
