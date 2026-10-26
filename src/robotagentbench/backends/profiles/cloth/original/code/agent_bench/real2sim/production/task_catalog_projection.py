#!/usr/bin/env python3
"""Build the benchmark catalog view from the raw Google Sheet rows.

The Sheet stays untouched. Family 7 disposal has four end-to-end pairs and
Family 8 cloth clearing has three. Each pick row is folded into its place row.
This is a small, visible product decision rather than a second semantic lock.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Mapping


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_SNAPSHOT = HERE / "google_sheet_tasks_current.json"
DEFAULT_MARKDOWN = REPO / "docs" / "ALL_TASKS_AND_INSTRUCTIONS.md"


FAMILY7_MERGES = {
    "7.2": {
        "source_rows": ("7.1", "7.2"),
        "action": "抓取使用过的纸巾并投入普通垃圾桶",
        "object_or_destination": "使用过的纸巾@茶几→普通垃圾桶",
    },
    "7.4": {
        "source_rows": ("7.3", "7.4"),
        "action": "抓取空食品包装并投入普通垃圾桶",
        "object_or_destination": "空食品包装@茶几→普通垃圾桶",
    },
    "7.8": {
        "source_rows": ("7.7", "7.8"),
        "action": "抓取空易拉罐并投入可回收垃圾桶",
        "object_or_destination": "空易拉罐@茶几→可回收垃圾桶",
    },
    "7.10": {
        "source_rows": ("7.9", "7.10"),
        "action": "抓取空一次性杯子并投入普通垃圾桶",
        "object_or_destination": "空一次性杯子@茶几→普通垃圾桶",
    },
}
FAMILY8_MERGES = {
    task: {
        "source_rows": (pick, task),
        "action": f"逐件抓取桌布上的{label}并放到桌布左侧临时桌面区域",
        "object_or_destination": f"{label}@桌布→桌布左侧桌面临时区",
    }
    for pick, task, label in (
        ("8.1", "8.2", "空杯子或容器"),
        ("8.3", "8.4", "书籍"),
        ("8.5", "8.6", "零食/个人物品"),
    )
}
END_TO_END_MERGES = {**FAMILY7_MERGES, **FAMILY8_MERGES}
MERGED_SOURCE_TO_TASK = {
    source: task
    for task, rule in END_TO_END_MERGES.items()
    for source in rule["source_rows"][:-1]
}


def project_sheet(sheet: Mapping[str, Any]) -> dict[str, Any]:
    """Return the projected catalog view while retaining Sheet provenance."""
    output = copy.deepcopy(dict(sheet))
    source_rows = {
        str(row["benchmark_subtask"]): dict(row)
        for row in sheet.get("tasks", ())
    }
    rows = []
    for raw in sheet.get("tasks", ()):
        row = dict(raw)
        task = str(row["benchmark_subtask"])
        if task in MERGED_SOURCE_TO_TASK:
            continue
        rule = END_TO_END_MERGES.get(task)
        if rule is not None:
            missing = [
                source for source in rule["source_rows"]
                if source not in source_rows
            ]
            if missing:
                raise ValueError(
                    f"E2E merge {task} is missing Sheet rows {missing}"
                )
            row["action"] = rule["action"]
            row["object_or_destination"] = rule["object_or_destination"]
            row["source_sheet_rows"] = list(rule["source_rows"])
            row["task_granularity"] = "end_to_end"
        else:
            row["source_sheet_rows"] = [task]
            row["task_granularity"] = "sheet_row"
        rows.append(row)
    output["source_task_count"] = len(sheet.get("tasks", ()))
    output["task_count"] = len(rows)
    output["tasks"] = rows
    output["catalog_projection"] = {
        "family8_end_to_end_merges": {
            task: list(rule["source_rows"])
            for task, rule in FAMILY8_MERGES.items()
        },
        "family7_end_to_end_merges": {
            task: list(rule["source_rows"])
            for task, rule in FAMILY7_MERGES.items()
        }
    }
    return output


def _cell(value: Any) -> str:
    return str(value if value not in (None, "") else "—").replace("|", "\\|")


def render_markdown(catalog: Mapping[str, Any]) -> str:
    """Render the current catalog; progress remains in the JSON snapshot."""
    tasks = list(catalog.get("tasks", ()))
    families: dict[int, int] = {}
    for task in tasks:
        family = int(task["benchmark_family"])
        families[family] = families.get(family, 0) + 1
    sheet_url = (
        "https://docs.google.com/spreadsheets/d/"
        + str(catalog.get("source_spreadsheet_id", ""))
        + "/edit"
    )
    family_summary = ", ".join(
        f"Family {family}: {count}" for family, count in sorted(families.items())
    )
    lines = [
        "# 当前 Benchmark task 清单",
        "",
        "> 本页由 `task_catalog_projection.py` 从 Google Sheet 只读快照生成。",
        "> benchmark task 是完整定义；real robot 数据只作为可选母数据或参考。",
        "",
        f"- Sheet：[`{_cell(catalog.get('source_sheet_name'))}`]({sheet_url})",
        f"- 快照日期：`{_cell(catalog.get('snapshot_date'))}`",
        f"- Sheet 原始行：{int(catalog.get('source_task_count', 0))}",
        f"- 当前完整 task：{len(tasks)}",
        f"- 分族数量：{family_summary}",
        "- Family 7 合并：`7.1+7.2→7.2`、`7.3+7.4→7.4`、"
        "`7.7+7.8→7.8`、`7.9+7.10→7.10`。",
        "",
        "- Family 8 合并：`8.1+8.2→8.2`、`8.3+8.4→8.4`、`8.5+8.6→8.6`。",
        "",
        "运行状态、owner 脚本和 real-data 覆盖不改变本页的 task 定义。统一管理方式见 "
        "[`TASK_MANAGEMENT.md`](TASK_MANAGEMENT.md)。",
        "",
        "| ID | Task | 完整动作 | 物体/目的地 | Owner | Sheet 行 |",
        "|---|---|---|---|---|---|",
    ]
    for task in tasks:
        lines.append(
            "| " + " | ".join((
                _cell(task.get("benchmark_subtask")),
                _cell(task.get("task_name")),
                _cell(task.get("action")),
                _cell(task.get("object_or_destination")),
                _cell(task.get("owner")),
                _cell(", ".join(map(str, task.get("source_sheet_rows", ())))),
            )) + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    source = json.loads(args.snapshot.read_text(encoding="utf-8"))
    catalog = project_sheet(source)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(render_markdown(catalog), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
