#!/usr/bin/env python3
"""Compose several task state contributions into one House episode plan.

Owner initializers remain single-task producers.  They may contribute robot,
prim and articulation state here; this module merges those contributions once
before a shared session is opened.  It deliberately contains no owner script
runner and no second task catalog.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from . import benchmark_eval_registry
    from . import house_layout
    from .house_task_environment import resolve as resolve_environment
except ImportError:
    import benchmark_eval_registry
    import house_layout
    from house_task_environment import resolve as resolve_environment


REPO = Path(__file__).resolve().parents[2]
BASELINE_MANIFEST = REPO / "real2sim/teleop/scene_baseline_manifest_v1.json"
SCHEMA = "real2sim.house_episode_plan.v1"


class EpisodeConflict(ValueError):
    pass


def _layout_recipe(value: str) -> house_layout.HouseLayout | None:
    if str(value) == "canonical":
        return None
    return house_layout.layout_for_id(str(value))


def _transformed_robot(
    robot: Mapping[str, Any], recipe: house_layout.HouseLayout,
    frame: str,
) -> dict[str, Any]:
    result = copy.deepcopy(dict(robot))
    if "base_xy_yaw" in result:
        value = result["base_xy_yaw"]
        if not isinstance(value, Sequence) or len(value) != 3:
            raise ValueError("robot.base_xy_yaw must contain x, y and yaw degrees")
        transform = house_layout.transform_for_frame(recipe, frame)
        xy = house_layout.transform_points(value[:2], transform)
        result["base_xy_yaw"] = [
            float(xy[0]), float(xy[1]), float(value[2]) + transform.yaw_deg,
        ]
    result["layout_frame"] = frame
    return result


def _transformed_prim_state(
    state: Mapping[str, Any], recipe: house_layout.HouseLayout,
    default_frame: str,
) -> dict[str, Any]:
    result = copy.deepcopy(dict(state))
    frame = str(result.pop("layout_frame", default_frame))
    transform = house_layout.transform_for_frame(recipe, frame)
    if "translate_xyz_m" in result:
        result["translate_xyz_m"] = house_layout.transform_points(
            result["translate_xyz_m"], transform).tolist()
    if "orient_wxyz" in result:
        result["orient_wxyz"] = house_layout.transform_quaternions_wxyz(
            result["orient_wxyz"], transform).tolist()
    if "yaw_deg" in result:
        result["yaw_deg"] = float(result["yaw_deg"]) + transform.yaw_deg
    if "velocity_mps_radps" in result:
        velocity = result["velocity_mps_radps"]
        if not isinstance(velocity, Sequence) or len(velocity) != 6:
            raise ValueError("prim velocity_mps_radps must contain six values")
        result["velocity_mps_radps"] = [
            *house_layout.rotate_vectors(velocity[:3], transform).tolist(),
            *house_layout.rotate_vectors(velocity[3:], transform).tolist(),
        ]
    result["layout_frame_applied"] = frame
    return result


def _transform_contribution(
    contribution: Mapping[str, Any], recipe: house_layout.HouseLayout,
    frame: str,
) -> dict[str, Any]:
    result = copy.deepcopy(dict(contribution))
    result["prims"] = {
        str(path): _transformed_prim_state(state, recipe, frame)
        for path, state in contribution.get("prims", {}).items()
    }
    if "robot" in contribution:
        result["robot"] = _transformed_robot(
            contribution["robot"], recipe, frame)
    return result


def _same(left: Any, right: Any) -> bool:
    """Compare JSON-like or NumPy-backed state without inventing identities."""
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _same(left[key], right[key]) for key in left
        )
    if (
        isinstance(left, Sequence)
        and isinstance(right, Sequence)
        and not isinstance(left, (str, bytes))
        and not isinstance(right, (str, bytes))
    ):
        return len(left) == len(right) and all(
            _same(a, b) for a, b in zip(left, right)
        )
    try:
        import numpy as np

        if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
            return bool(np.array_equal(left, right))
    except ImportError:
        pass
    try:
        result = left == right
        return bool(result)
    except (TypeError, ValueError):
        return False


def _merge_named(
    target: dict[str, Any], source: Mapping[str, Any], *, kind: str, task_id: str,
) -> None:
    for key, value in source.items():
        name = str(key)
        if name in target and not _same(target[name], value):
            raise EpisodeConflict(
                f"{kind} {name!r} has conflicting state from task {task_id}"
            )
        target[name] = value


def _active_conflicts(
    prims: Mapping[str, Any], *, manifest_path: Path = BASELINE_MANIFEST,
) -> list[dict[str, Any]]:
    active = {
        str(path)
        for path, state in prims.items()
        if isinstance(state, Mapping) and state.get("active") is True
    }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return [
        dict(pair)
        for pair in manifest.get("interpenetrating_pairs", ())
        if str(pair.get("a")) in active and str(pair.get("b")) in active
    ]


def compose_episode(
    task_ids: Sequence[str], *,
    contributions: Mapping[str, Mapping[str, Any]] | None = None,
    episode_destinations: Mapping[str, Mapping[str, Any]] | None = None,
    layout: str = "canonical",
) -> dict[str, Any]:
    """Merge task state without executing or rewriting owner code."""
    ordered = [str(task_id) for task_id in task_ids]
    if not ordered or len(ordered) != len(set(ordered)):
        raise ValueError("task_ids must be a non-empty unique ordered list")
    contributions = {
        str(task_id): dict(value)
        for task_id, value in (contributions or {}).items()
    }
    episode_destinations = {
        str(task_id): dict(value)
        for task_id, value in (episode_destinations or {}).items()
    }
    recipe = _layout_recipe(str(layout))
    unknown_contributions = set(contributions) - set(ordered)
    if unknown_contributions:
        raise ValueError(
            "contributions contain tasks outside task_ids: "
            + ", ".join(sorted(unknown_contributions))
        )
    unknown_destinations = set(episode_destinations) - set(ordered)
    if unknown_destinations:
        raise ValueError(
            "episode_destinations contain tasks outside task_ids: "
            + ", ".join(sorted(unknown_destinations))
        )

    profiles: dict[str, dict[str, Any]] = {}
    choices: dict[str, str] = {}
    prims: dict[str, Any] = {}
    articulations: dict[str, Any] = {}
    robot: Any = None
    complete: list[str] = []
    for task_id in ordered:
        task = dict(benchmark_eval_registry.resolve(task_id)["task"])
        if task_id in episode_destinations:
            task["episode_destination"] = episode_destinations[task_id]
        profile = resolve_environment(task)
        destination = profile.get("destination_selection", {})
        if (
            destination.get("required") is True
            and destination.get("runtime_supported") is not True
        ):
            detail = destination.get("blocker") or (
                "select one of "
                + ", ".join(map(str, destination.get("allowed_modes", ())))
            )
            raise ValueError(
                f"task {task_id} requires an executable episode destination: "
                + str(detail)
            )
        if recipe is not None:
            station = house_layout.station_pose(
                profile["robot_station"]["key"], recipe)
            profile["robot_station"] = {
                **profile["robot_station"],
                "xy_m": list(station.xy_m),
                "yaw_deg": station.yaw_deg,
                "quat_wxyz": list(station.quat_wxyz),
                "layout_id": recipe.layout_id,
            }
        profiles[task_id] = profile
        for key, value in profile["scenario_choices"].items():
            if key in choices and choices[key] != value:
                raise EpisodeConflict(
                    f"scenario choice {key!r} cannot be both "
                    f"{choices[key]!r} and {value!r}"
                )
            choices[key] = value
        contribution = contributions.get(task_id)
        if contribution is None:
            continue
        if recipe is not None:
            contribution = _transform_contribution(
                contribution, recipe, profile["layout_frame"])
        _merge_named(
            prims, contribution.get("prims", {}),
            kind="prim", task_id=task_id,
        )
        _merge_named(
            articulations, contribution.get("articulations", {}),
            kind="articulation", task_id=task_id,
        )
        _merge_named(
            choices, contribution.get("choice_groups", {}),
            kind="scenario choice", task_id=task_id,
        )
        if "robot" in contribution:
            if robot is not None and not _same(robot, contribution["robot"]):
                raise EpisodeConflict(
                    "an episode has one frame-zero robot state; task "
                    f"{task_id} supplied a different one"
                )
            robot = contribution["robot"]
        complete.append(task_id)

    if robot is None:
        first = profiles[ordered[0]]["robot_station"]
        robot = {
            "base_xy_yaw": [*first["xy_m"], first["yaw_deg"]],
            "source": "first task canonical station",
        }
    conflicts = _active_conflicts(prims)
    if conflicts:
        pairs = [f"{row['a']} <> {row['b']}" for row in conflicts]
        raise EpisodeConflict(
            "active prims occupy mutually exclusive baseline poses: "
            + "; ".join(pairs)
        )
    return {
        "schema": SCHEMA,
        "task_ids": ordered,
        "layout": str(layout),
        "layout_status": recipe.status if recipe is not None else "canonical",
        "robot": robot,
        "prims": prims,
        "articulations": articulations,
        "choice_groups": choices,
        "episode_destinations": {
            task_id: profiles[task_id]["destination_selection"]["value"]
            for task_id in ordered
            if profiles[task_id]["destination_selection"]["selected"] is True
        },
        "task_environments": profiles,
        "segment_stations": {
            task_id: profiles[task_id]["robot_station"] for task_id in ordered
        },
        "contributions_present": complete,
        "ready_to_materialize": len(complete) == len(ordered),
        "owner_scripts_executed": False,
        "planned_baseline_restore_count": 1,
    }


def _write_episode_state(
    plan: Mapping[str, Any], opened: Mapping[str, Any], task_ids: Sequence[str],
) -> dict[str, Any]:
    """Write one claimed plan; contribution transforms are world-frame values."""

    from pxr import Gf, Sdf, Usd, UsdGeom

    scene = Path(str(opened["scene"]))
    prim_states = {
        str(path): dict(state)
        for path, state in plan.get("prims", {}).items()
    }
    articulation_states = {
        str(path): dict(state)
        for path, state in plan.get("articulations", {}).items()
    }
    recipe = _layout_recipe(str(plan.get("layout", "canonical")))
    if recipe is not None:
        first_frame = str(
            plan["task_environments"][task_ids[0]]["layout_frame"])
        house_layout.apply_scene_overlay(
            scene, recipe, frame=first_frame, robot_path=None)

    stage = Usd.Stage.Open(str(scene))
    if stage is None:
        raise RuntimeError(f"could not open House working scene {scene}")

    def sequence(value: Any, length: int, label: str) -> list[float]:
        if (
            not isinstance(value, Sequence)
            or isinstance(value, (str, bytes))
            or len(value) != length
        ):
            raise ValueError(f"{label} must contain {length} numbers")
        return [float(item) for item in value]

    def set_vec3(attribute: Any, value: Any, label: str) -> None:
        x, y, z = sequence(value, 3, label)
        vector = (
            Gf.Vec3d(x, y, z)
            if str(attribute.GetTypeName()) == "double3"
            else Gf.Vec3f(x, y, z)
        )
        if not attribute.Set(vector):
            raise RuntimeError(f"could not write {label}")

    def apply_world_transform(
        path: str, prim: Any, state: Mapping[str, Any],
    ) -> None:
        """Convert the composed world pose to the prim parent's local frame."""

        if not any(
            key in state for key in ("translate_xyz_m", "orient_wxyz", "yaw_deg")
        ):
            return
        xformable = UsdGeom.Xformable(prim)
        world = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        current = Gf.Transform(world)
        translation = current.GetTranslation()
        rotation = current.GetRotation()
        if "translate_xyz_m" in state:
            x, y, z = sequence(
                state["translate_xyz_m"], 3, f"{path}.translate_xyz_m")
            translation = Gf.Vec3d(x, y, z)
        if "orient_wxyz" in state:
            w, x, y, z = sequence(
                state["orient_wxyz"], 4, f"{path}.orient_wxyz")
            rotation = Gf.Rotation(Gf.Quatd(w, x, y, z))
        elif "yaw_deg" in state:
            rotation = Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), float(state["yaw_deg"]))

        desired = Gf.Transform()
        desired.SetTranslation(translation)
        desired.SetRotation(rotation)
        desired.SetScale(current.GetScale())
        parent_world = xformable.ComputeParentToWorldTransform(
            Usd.TimeCode.Default())
        local = desired.GetMatrix() * parent_world.GetInverse()
        operation = xformable.AddTransformOp(
            precision=UsdGeom.XformOp.PrecisionDouble,
            opSuffix="episodeState",
        )
        if not xformable.SetXformOpOrder([operation]) or not operation.Set(local):
            raise RuntimeError(f"could not author world-frame episode pose for {path}")

    def apply_prim_state(path: str, state: Mapping[str, Any]) -> None:
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            raise ValueError(f"episode state names missing prim {path}")
        if "active" in state:
            prim.SetActive(bool(state["active"]))
        apply_world_transform(path, prim, state)
        if "velocity_mps_radps" in state:
            velocity = sequence(
                state["velocity_mps_radps"], 6, f"{path}.velocity")
            for attribute_name, values in (
                ("physics:velocity", velocity[:3]),
                ("physics:angularVelocity", velocity[3:]),
            ):
                attribute = prim.GetAttribute(attribute_name)
                if not attribute:
                    attribute = prim.CreateAttribute(
                        attribute_name, Sdf.ValueTypeNames.Vector3f)
                set_vec3(attribute, values, f"{path}.{attribute_name}")

    for path, state in prim_states.items():
        apply_prim_state(path, state)

    robot = dict(plan.get("robot", {}))
    base = robot.get("base_xy_yaw")
    if base is not None:
        x, y, yaw = sequence(base, 3, "robot.base_xy_yaw")
        robot_prim = stage.GetPrimAtPath("/World/Robot")
        if not robot_prim or not robot_prim.IsValid():
            raise ValueError("House scene has no /World/Robot")
        robot_world = UsdGeom.Xformable(robot_prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default())
        apply_prim_state("/World/Robot", {
            "translate_xyz_m": [x, y, float(robot_world.ExtractTranslation()[2])],
            "yaw_deg": yaw,
        })

    for path, state in articulation_states.items():
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            raise ValueError(f"episode articulation names missing joint {path}")
        kind = str(state.get("kind", "linear"))
        if kind not in {"linear", "angular"}:
            raise ValueError(f"{path}.kind must be linear or angular")
        attribute_name = f"state:{kind}:physics:position"
        attribute = prim.GetAttribute(attribute_name)
        if not attribute:
            attribute = prim.CreateAttribute(
                attribute_name, Sdf.ValueTypeNames.Float)
        if not attribute.Set(float(state["position"])):
            raise RuntimeError(f"could not write {path} articulation position")

    if not stage.GetRootLayer().Save():
        raise RuntimeError(f"could not save House episode scene {scene}")
    return {
        "schema": "real2sim.house_episode_materialization.v1",
        "task_ids": list(task_ids),
        "layout": str(plan.get("layout", "canonical")),
        "baseline_restore_count": 1,
        "prim_state_count": len(prim_states),
        "articulation_state_count": len(articulation_states),
        "robot_state_written": base is not None,
        "scene_saved": True,
        "physics_started": False,
        "pose_coordinates": "world",
    }


def materialize_episode(
    plan: Mapping[str, Any], *, session_name: str,
    recover_stale: bool = False,
) -> dict[str, Any]:
    """Restore once, write one composed frame-zero state and keep it claimed.

    Owner scripts produce contributions; this adapter never imports or runs
    them.  A failed write releases the session and keeps a diagnostic snapshot
    instead of leaving a false owner claim behind.
    """

    if plan.get("schema") != SCHEMA:
        raise ValueError(f"expected episode plan schema {SCHEMA}")
    if plan.get("ready_to_materialize") is not True:
        raise ValueError("episode plan is missing one or more task contributions")
    task_ids = [str(value) for value in plan.get("task_ids", ())]
    if not task_ids:
        raise ValueError("episode plan has no tasks")

    teleop = REPO / "real2sim" / "teleop"
    import sys
    if str(teleop) not in sys.path:
        sys.path.insert(0, str(teleop))
    import session_scene

    prim_states = {
        str(path): dict(state)
        for path, state in plan.get("prims", {}).items()
    }
    activate = sorted(
        path for path, state in prim_states.items()
        if state.get("active") is True
    )
    opened = session_scene.open_session(
        session_name,
        activate,
        target_prim="",
        recover_stale=recover_stale,
        task_ids=task_ids,
        layout=str(plan.get("layout", "canonical")),
        choice_groups=dict(plan.get("choice_groups", {})),
    )
    try:
        receipt = _write_episode_state(plan, opened, task_ids)
        session_scene.record_episode_materialization(session_name, receipt)
    except BaseException as error:
        try:
            session_scene.close_session(session_name, keep=True)
        except Exception as cleanup_error:
            error.add_note(
                "failed to release House session after materialization error: "
                + str(cleanup_error)
            )
        raise
    return {
        "session": opened,
        "receipt": receipt,
        "scene": str(opened["scene"]),
        "session_remains_open": True,
    }
