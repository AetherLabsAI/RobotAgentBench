"""Validate pick scene structure and reviewed physical calibration values.
Git owns versioned source identity.  These helpers therefore inspect the
authored USDA fields and calibration values that affect physical replay
directly; they do not create a second source identity.
"""
from __future__ import annotations

import json
import os
import re


PICK_SCENE_PROVENANCE_SCHEMA = "real2sim.pick_scene_provenance.v1"
SEAT_RISER_PATH = "/World/Assets/SeatRiser"
ROBOT_REFERENCE = (
    "../source/geniesim/assets/robot/G1_omnipicker/robot.usda"
)
TARGET_REFERENCES = {
    "GlassesCase": "./usdz/glasses_case.usdz",
    "PaperCup": "./usdz/paper_cup.usdz",
    "Remote": "./usdz/remote_tv.usdz",
    "Umbrella": "./usdz/umbrella.usdz",
    "Wallet": "./usdz/wallet.usda",
}
TASK53_CONTACT_SCENE_CALIBRATION_FILENAME = (
    "task53_contact_scene_calibration_v2.json")
TASK53_CONTACT_SCENE_CALIBRATION_SCHEMA = (
    "real2sim.task53_contact_scene_calibration.v2")
TASK53_SUPPORT_AUTHORITY = "task53_real_rgb_rounded_cushion_v2"
TASK53_SUPPORT_PATH = "/World/Assets/CushionOnBook"
TASK53_SUPPORT_GEOMETRY_VERSION = "task53_rounded_pillow_v2"
TASK53_TARGET_COLLISION_GEOMETRY_VERSION = "task53_wallet_box_v1"
TASK53_TARGET_COLLISION_PROXY_PATH = (
    "/World/Assets/Wallet/Task53CollisionBox")
TASK53_CALIBRATION_EPISODES = (
    "7217cfe2-16a4-4b14-b82f-f09bd65cb9dc",
    "f913c9e0-5e5c-4f64-98d4-7b046de95453",
    "1b241e0d-33d4-4d46-9eb4-09a788370c02",
)
TASK59_SUPPORT_CALIBRATION_FILENAME = (
    "task59_support_calibration_v1.json")
TASK59_SUPPORT_CALIBRATION_SCHEMA = (
    "real2sim.task59_support_calibration.v1")
TASK59_SUPPORT_METHOD_VERSION = (
    "depth-only-dominant-horizontal-plane.v1")
TASK59_SUPPORT_AUTHORITY = "task59_rgbd_tabletop_v1"
TASK59_AUTHORIZED_SUPPORT_MIN_M = 0.510
TASK59_AUTHORIZED_SUPPORT_MAX_M = 0.530
TASK59_SUPPORT_UNCERTAINTY_M = 0.010
TASK59_CALIBRATION_EPISODES = (
    "d15fc3f0-595d-4df1-b279-fb0ce29a7832",
    "d6baa895-493c-4e6b-842d-bbcbe697b3db",
    "da0f93de-e7cf-45d7-8e71-d2854de7b79a",
)


def _read_snapshot(path):
    with open(path, "rb") as stream:
        return stream.read()


def _finite_number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be one finite number")
    value = float(value)
    if value != value or abs(value) == float("inf"):
        raise ValueError(f"{label} must be one finite number")
    return value


def _finite_tuple(value, length, label):
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(
            f"{label} must contain exactly {length} finite numbers")
    return tuple(
        _finite_number(item, f"{label}[{index}]")
        for index, item in enumerate(value)
    )


def task53_contact_scene_calibration_identity(path):
    """Validate Task53's reviewed real-initial-state calibration structure."""
    payload = _read_snapshot(path)
    try:
        manifest = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "Task53 contact-scene calibration must be UTF-8 JSON") from exc
    if not isinstance(manifest, dict):
        raise ValueError(
            "Task53 contact-scene calibration must be one JSON object")
    for key, expected in (
        ("schema", TASK53_CONTACT_SCENE_CALIBRATION_SCHEMA),
        ("task_id", "53"),
        ("target_body_path", "/World/Assets/Wallet"),
        ("support_body_path", TASK53_SUPPORT_PATH),
        ("support_mode", TASK53_SUPPORT_AUTHORITY),
        ("coordinate_frame", "sim_scene_world_from_robot_base"),
    ):
        if manifest.get(key) != expected:
            raise ValueError(
                f"Task53 contact-scene calibration {key} is missing/wrong")
    if manifest.get("policy") != {
        "scope_task_ids": ["53"],
        "contact_context_scope": "per_episode",
        "missing_episode_policy": "fail_closed",
        "forbid_cross_episode_pose_reuse": True,
        "forbid_invisible_replacement_support": True,
        "requires_three_primary_camera_initial_review": True,
        "publication_scope":
            "pick_contact_v1_and_three_view_compare_video",
    }:
        raise ValueError(
            "Task53 contact-scene calibration policy is missing/wrong")
    target = manifest.get("target")
    support = manifest.get("support")
    if (
        not isinstance(target, dict)
        or target.get("asset_reference") != TARGET_REFERENCES["Wallet"]
        or target.get("pose") != "flat"
        or target.get("dynamic") is not True
        or _finite_tuple(
            target.get("scale_xyz"), 3, "Task53 Wallet scale")
        != (0.78, 1.14, 1.0)
        or _finite_tuple(
            target.get("scaled_size_m"), 3, "Task53 Wallet size")
        != (0.1287, 0.1482, 0.04)
    ):
        raise ValueError(
            "Task53 target calibration is missing/inconsistent")
    if support != {
        "source_asset_reference": "./usdz/cushions_set.usdz",
        "geometry_version": TASK53_SUPPORT_GEOMETRY_VERSION,
        "size_xyz_m": [0.46, 0.36, 0.1],
        "flat_top_size_xy_m": [0.43, 0.33],
        "relation": "directly_supports_target",
        "required": True,
        "active": True,
        "kinematic": True,
        "rigid_body_enabled": True,
        "collision_required": True,
        "visible": True,
    }:
        raise ValueError(
            "Task53 visible support calibration is missing/inconsistent")

    episodes = manifest.get("episodes")
    if (
        not isinstance(episodes, dict)
        or tuple(episodes) != TASK53_CALIBRATION_EPISODES
    ):
        raise ValueError(
            "Task53 contact-scene episodes/order are not canonical")
    normalized = {}
    for episode_uid, episode in episodes.items():
        if (
            not isinstance(episode, dict)
            or episode.get("source_episode_uuid") != episode_uid
            or episode.get("active_hand") != "r"
        ):
            raise ValueError(
                f"Task53 episode {episode_uid!r} identity/hand is invalid")
        source = episode.get("source_initial_state")
        scene = episode.get("scene_state")
        controls = episode.get("contact_controls")
        if (
            not isinstance(source, dict)
            or source.get("frame_index") != 0
            or source.get("head_decoded_frame0_shape")
            != [800, 1280, 3]
            or not isinstance(source.get("review"), str)
            or not source["review"].strip()
        ):
            raise ValueError(
                f"Task53 episode {episode_uid!r} source review is invalid")
        for hand in ("left", "right"):
            filename = source.get(f"hand_{hand}_frame0_filename")
            if not isinstance(filename, str) or not re.fullmatch(
                    r"[0-9]{16,20}\.jpg", filename):
                raise ValueError(
                    f"Task53 episode {episode_uid!r} has invalid "
                    f"hand_{hand} frame filename")
        if not isinstance(scene, dict) or not isinstance(controls, dict):
            raise ValueError(
                f"Task53 episode {episode_uid!r} state/controls are missing")
        wallet_translate = _finite_tuple(
            scene.get("wallet_translate_xyz_m"), 3,
            f"Task53 episode {episode_uid} Wallet translation")
        wallet_yaw = _finite_number(
            scene.get("wallet_yaw_deg"),
            f"Task53 episode {episode_uid} Wallet yaw")
        grasp_local_offset = _finite_tuple(
            scene.get("grasp_local_offset_xy_m"), 2,
            f"Task53 episode {episode_uid} grasp-local offset")
        cushion_translate = _finite_tuple(
            scene.get("cushion_translate_xyz_m"), 3,
            f"Task53 episode {episode_uid} cushion translation")
        cushion_orient = _finite_tuple(
            scene.get("cushion_orient_wxyz"), 4,
            f"Task53 episode {episode_uid} cushion orientation")
        support_top = _finite_number(
            scene.get("support_top_m"),
            f"Task53 episode {episode_uid} support top")
        normalized_controls = {
            key: _finite_number(
                controls.get(key),
                f"Task53 episode {episode_uid} {key}")
            for key in (
                "bite_z_m",
                "pad_axis_offset_m",
                "grip_past_rad",
                "static_friction",
                "dynamic_friction",
                "torsional_patch_radius_m",
                "min_torsional_patch_radius_m",
            )
        }
        if (
            not -180.0 <= wallet_yaw <= 180.0
            or abs(wallet_translate[2] - support_top) > 1e-8
            or not (
                abs(grasp_local_offset[0]) <= 0.0825
                and abs(grasp_local_offset[1]) <= 0.065
            )
            or abs(normalized_controls["pad_axis_offset_m"]) > 0.025
            or not (
                0.0 <= normalized_controls[
                    "min_torsional_patch_radius_m"]
                <= normalized_controls["torsional_patch_radius_m"]
                <= 0.015
            )
            or (
                normalized_controls["static_friction"] == 0.0
            ) != (
                normalized_controls["dynamic_friction"] == 0.0
            )
            or (
                normalized_controls["static_friction"]
                and not (
                    1.0 <= normalized_controls["dynamic_friction"]
                    <= normalized_controls["static_friction"] <= 4.0
                )
            )
        ):
            raise ValueError(
                f"Task53 episode {episode_uid!r} scene/contact values "
                "are inconsistent")
        normalized[episode_uid] = {
            "episode_uid": episode_uid,
            "active_hand": "r",
            "source_initial_state": dict(source),
            "wallet_translate_xyz_m": wallet_translate,
            "wallet_yaw_deg": wallet_yaw,
            "grasp_local_offset_xy_m": grasp_local_offset,
            "cushion_translate_xyz_m": cushion_translate,
            "cushion_orient_wxyz": cushion_orient,
            "support_top_m": support_top,
            "contact_controls": normalized_controls,
        }
    return {
        "schema": TASK53_CONTACT_SCENE_CALIBRATION_SCHEMA,
        "manifest_revision": TASK53_CONTACT_SCENE_CALIBRATION_SCHEMA,
        "support_authority": TASK53_SUPPORT_AUTHORITY,
        "target_scale_xyz": (0.78, 1.14, 1.0),
        "support_reference": support["source_asset_reference"],
        "support_geometry_version": support["geometry_version"],
        "support_size_xyz_m": tuple(support["size_xyz_m"]),
        "episodes": normalized,
    }


def task53_episode_calibration(calibration, episode_uid):
    """Resolve one exact Task53 episode, failing closed on missing/reused UID."""
    if not isinstance(calibration, dict):
        raise ValueError(
            "Task53 requires its pinned contact-scene calibration")
    try:
        episode = calibration["episodes"][str(episode_uid)]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "Task53 has no reviewed contact scene for episode "
            f"{episode_uid!r}") from exc
    if episode.get("episode_uid") != str(episode_uid):
        raise ValueError(
            "Task53 calibration episode identity is inconsistent")
    return episode


def validate_task53_contact_controls(
        calibration, episode_uid, *, bite_z_m, pad_axis_offset_m,
        grip_past_rad, static_friction, dynamic_friction,
        torsional_patch_radius_m, min_torsional_patch_radius_m):
    """Require runtime/init controls to equal the episode's frozen values."""
    episode = task53_episode_calibration(calibration, episode_uid)
    actual = {
        "bite_z_m": bite_z_m,
        "pad_axis_offset_m": pad_axis_offset_m,
        "grip_past_rad": grip_past_rad,
        "static_friction": static_friction,
        "dynamic_friction": dynamic_friction,
        "torsional_patch_radius_m": torsional_patch_radius_m,
        "min_torsional_patch_radius_m": min_torsional_patch_radius_m,
    }
    expected = episode["contact_controls"]
    mismatches = []
    for key, value in actual.items():
        value = _finite_number(value, f"Task53 runtime {key}")
        if abs(value - float(expected[key])) > 1e-12:
            mismatches.append(
                f"{key}={value:g} expected {float(expected[key]):g}")
    if mismatches:
        raise ValueError(
            "Task53 runtime controls do not match episode calibration: "
            + ", ".join(mismatches))
    return episode


def task59_support_calibration_identity(path):
    """Validate the reviewed three-probe real-depth calibration structure."""
    payload = _read_snapshot(path)
    try:
        manifest = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "Task59 support calibration must be UTF-8 JSON") from exc
    if not isinstance(manifest, dict):
        raise ValueError(
            "Task59 support calibration must be one JSON object")
    if manifest.get("schema") != TASK59_SUPPORT_CALIBRATION_SCHEMA:
        raise ValueError(
            "Task59 support calibration schema is missing/wrong")
    method_version = manifest.get("method_version")
    if method_version != TASK59_SUPPORT_METHOD_VERSION:
        raise ValueError(
            "Task59 support calibration method_version is invalid")
    for key, expected in (
        ("task_id", "59"),
        ("target_body_path", "/World/Assets/GlassesCase"),
        ("support_body_path", SEAT_RISER_PATH),
        ("support_mode", TASK59_SUPPORT_AUTHORITY),
        ("coordinate_frame", "sim_scene_world_from_robot_base"),
    ):
        if manifest.get(key) != expected:
            raise ValueError(
                f"Task59 support calibration {key} is missing/wrong")
    authorization = manifest.get("authorization")
    if authorization != {
        "task_ids": ["59"],
        "requires_constant_support_top": True,
        "forbids_episode_dependent_support_height": True,
        "requires_explicit_active_collision": True,
        "publication_scope": "pick_contact_v1_target_support_only",
    }:
        raise ValueError(
            "Task59 support calibration authorization is missing/wrong")
    camera_model = manifest.get("camera_model")
    if not isinstance(camera_model, dict):
        raise ValueError(
            "Task59 support calibration camera_model is missing")
    probes = manifest.get("probes")
    if not isinstance(probes, list) or len(probes) != 3:
        raise ValueError(
            "Task59 support calibration requires exactly three probes")
    seen = []
    normalized_probes = []
    for probe in probes:
        if not isinstance(probe, dict):
            raise ValueError(
                "Task59 support calibration probe must be an object")
        episode_uid = probe.get("episode_uid")
        if not isinstance(episode_uid, str):
            raise ValueError(
                "Task59 support calibration probe has no episode_uid")
        if probe.get("raw_streams_aligned") is not False:
            raise ValueError(
                f"Task59 calibration probe {episode_uid!r} must preserve "
                "the recorded unaligned-stream limitation")
        table_plane = probe.get("table_plane")
        if not isinstance(table_plane, dict):
            raise ValueError(
                f"Task59 calibration probe {episode_uid!r} has no table "
                "plane")
        table_median = table_plane.get("median_m")
        if (
            isinstance(table_median, bool)
            or not isinstance(table_median, (int, float))
            or not float(table_median) == float(table_median)
            or abs(float(table_median)) == float("inf")
        ):
            raise ValueError(
                f"Task59 calibration probe {episode_uid!r} has invalid "
                "table median")
        seen.append(episode_uid)
        normalized_probes.append({
            "episode_uid": episode_uid,
            "table_median_m": float(table_median),
        })
    if tuple(seen) != TASK59_CALIBRATION_EPISODES:
        raise ValueError(
            "Task59 support calibration probes/order are not canonical")

    aggregate = manifest.get("aggregate")
    if not isinstance(aggregate, dict):
        raise ValueError(
            "Task59 support calibration aggregate is missing")
    values = {}
    for key in (
            "support_top_m",
            "uncertainty_m",
            "authorized_min_m",
            "authorized_max_m"):
        value = aggregate.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"Task59 support calibration {key} is invalid")
        value = float(value)
        if value != value or abs(value) == float("inf"):
            raise ValueError(
                f"Task59 support calibration {key} is invalid")
        values[key] = value
    table_medians = [
        probe["table_median_m"] for probe in normalized_probes]
    sorted_medians = sorted(table_medians)
    expected_top = sorted_medians[len(sorted_medians) // 2]
    expected_spread = max(table_medians) - min(table_medians)
    spread = aggregate.get("max_interprobe_spread_m")
    if (
        isinstance(spread, bool)
        or not isinstance(spread, (int, float))
        or not float(spread) == float(spread)
        or abs(float(spread)) == float("inf")
    ):
        raise ValueError(
            "Task59 support calibration spread is invalid")
    if (
        values["authorized_min_m"] != TASK59_AUTHORIZED_SUPPORT_MIN_M
        or values["authorized_max_m"] != TASK59_AUTHORIZED_SUPPORT_MAX_M
        or values["uncertainty_m"] != TASK59_SUPPORT_UNCERTAINTY_M
        or not (
            values["authorized_min_m"]
            <= values["support_top_m"]
            <= values["authorized_max_m"]
        )
        or abs(values["support_top_m"] - expected_top) > 1e-9
        or abs(float(spread) - expected_spread) > 1e-9
        or expected_spread > 0.003
    ):
        raise ValueError(
            "Task59 support calibration bounds/uncertainty are inconsistent")
    return {
        "schema": TASK59_SUPPORT_CALIBRATION_SCHEMA,
        "method_version": method_version,
        "manifest_revision": TASK59_SUPPORT_CALIBRATION_SCHEMA,
        "probes": normalized_probes,
        "support_authority": TASK59_SUPPORT_AUTHORITY,
        "max_interprobe_spread_m": float(spread),
        **values,
    }


def _scene_prim_header(scene_text, prim_type, prim_name):
    pattern = re.compile(
        rf"\bdef\s+{re.escape(prim_type)}\s+"
        rf'"{re.escape(prim_name)}"\s*\((.*?)\)\s*\{{',
        re.DOTALL,
    )
    match = pattern.search(scene_text)
    if not match:
        raise ValueError(
            f"scene is missing {prim_type} {prim_name!r}")
    return match.group(1)


def _scene_prim_prefix(scene_text, prim_type, prim_name):
    """Return attributes before the prim's first nested definition.

    The relevant target and SeatRiser physics opinions are authored directly
    on their prims.  Limiting the check to the prefix prevents an unrelated
    descendant from satisfying a required opinion.
    """
    header = re.search(
        rf"\bdef\s+{re.escape(prim_type)}\s+"
        rf'"{re.escape(prim_name)}"\s*(?:\(.*?\))?\s*\{{',
        scene_text,
        re.DOTALL,
    )
    if not header:
        raise ValueError(
            f"scene is missing {prim_type} {prim_name!r}")
    tail = scene_text[header.end():]
    nested = re.search(r"\n\s*(?:def|over|class)\s+", tail)
    return tail[:nested.start()] if nested else tail


def _usd_tokens(scene_text):
    """Lex the USDA constructs needed for conservative prim scoping.

    Comments, quoted strings and asset paths are handled as opaque tokens so
    braces or fake ``def`` text inside them cannot alter scope.  Unsupported
    unterminated literals fail closed instead of falling back to substring
    guesses.
    """
    tokens = []
    index = 0
    size = len(scene_text)
    while index < size:
        char = scene_text[index]
        if char.isspace():
            index += 1
            continue
        if char == "#":
            newline = scene_text.find("\n", index + 1)
            index = size if newline < 0 else newline + 1
            continue
        if char == '"':
            start = index
            triple = scene_text.startswith('"""', index)
            delimiter = '"""' if triple else '"'
            index += len(delimiter)
            value_start = index
            while index < size:
                if scene_text[index] == "\\":
                    index += 2
                    if index > size:
                        raise ValueError(
                            "scene has an unterminated escaped USDA string")
                    continue
                if scene_text.startswith(delimiter, index):
                    value = scene_text[value_start:index]
                    index += len(delimiter)
                    tokens.append(("string", value, start, index))
                    break
                if not triple and scene_text[index] in "\r\n":
                    raise ValueError(
                        "scene has an unterminated USDA string")
                index += 1
            else:
                raise ValueError("scene has an unterminated USDA string")
            continue
        if char == "@":
            start = index
            index += 1
            value_start = index
            while index < size:
                if scene_text[index] == "\\":
                    index += 2
                    if index > size:
                        raise ValueError(
                            "scene has an unterminated escaped asset path")
                    continue
                if scene_text[index] == "@":
                    value = scene_text[value_start:index]
                    index += 1
                    tokens.append(("asset", value, start, index))
                    break
                index += 1
            else:
                raise ValueError(
                    "scene has an unterminated USDA asset path")
            continue
        if char.isalpha() or char == "_":
            start = index
            index += 1
            while (
                index < size
                and (
                    scene_text[index].isalnum()
                    or scene_text[index] in "_:"
                )
            ):
                index += 1
            tokens.append(
                ("word", scene_text[start:index], start, index))
            continue
        tokens.append(("symbol", char, index, index + 1))
        index += 1
    return tokens


def _usd_prim_declarations(scene_text):
    """Return conservatively scoped prim declarations from one USDA layer."""
    tokens = _usd_tokens(scene_text)
    brace_stack = []
    brace_pairs = {}
    containers = {}
    for token_index, token in enumerate(tokens):
        containers[token_index] = (
            brace_stack[-1] if brace_stack else None)
        if token[:2] == ("symbol", "{"):
            brace_stack.append(token_index)
        elif token[:2] == ("symbol", "}"):
            if not brace_stack:
                raise ValueError(
                    "scene has an unmatched closing brace")
            opening = brace_stack.pop()
            brace_pairs[opening] = token_index
    if brace_stack:
        raise ValueError("scene has an unterminated brace scope")

    declarations = []
    for token_index, token in enumerate(tokens):
        if (
            token[0] != "word"
            or token[1] not in ("def", "over", "class")
        ):
            continue
        cursor = token_index + 1
        if cursor >= len(tokens):
            continue
        prim_type = None
        if (
            tokens[cursor][0] == "word"
            and cursor + 1 < len(tokens)
            and tokens[cursor + 1][0] == "string"
        ):
            prim_type = tokens[cursor][1]
            cursor += 1
        if tokens[cursor][0] != "string":
            continue
        name_token = tokens[cursor]
        cursor += 1
        if (
            cursor < len(tokens)
            and tokens[cursor][:2] == ("symbol", "(")
        ):
            paren_depth = 0
            while cursor < len(tokens):
                current = tokens[cursor]
                if current[:2] == ("symbol", "("):
                    paren_depth += 1
                elif current[:2] == ("symbol", ")"):
                    paren_depth -= 1
                    if paren_depth == 0:
                        cursor += 1
                        break
                cursor += 1
            else:
                raise ValueError(
                    "scene has unterminated prim metadata")
        if (
            cursor >= len(tokens)
            or tokens[cursor][:2] != ("symbol", "{")
        ):
            continue
        close_token_index = brace_pairs.get(cursor)
        if close_token_index is None:
            raise ValueError("scene has an unterminated prim scope")
        declarations.append({
            "specifier": token[1],
            "prim_type": prim_type,
            "name": name_token[1],
            "start_token_index": token_index,
            "open_token_index": cursor,
            "close_token_index": close_token_index,
            "container_open_token_index": containers[token_index],
            "start": token[2],
            "name_end": name_token[3],
            "open": tokens[cursor][2],
            "open_end": tokens[cursor][3],
            "close": tokens[close_token_index][2],
        })

    by_open = {
        declaration["open_token_index"]: declaration
        for declaration in declarations
    }
    for declaration in declarations:
        path = []
        current = declaration
        visited = set()
        while current is not None:
            opening = current["open_token_index"]
            if opening in visited:
                raise ValueError("scene has cyclic prim scope metadata")
            visited.add(opening)
            path.append(current["name"])
            container = current["container_open_token_index"]
            if container is None:
                current = None
            else:
                current = by_open.get(container)
                if current is None:
                    path = None
                    break
        declaration["path"] = (
            tuple(reversed(path)) if path is not None else None)
    return tokens, declarations


def _scene_prim_scope(scene_text, prim_type, prim_path):
    """Return one exact prim subtree, rejecting ambiguous/non-prim nesting."""
    prim_path = tuple(prim_path)
    tokens, declarations = _usd_prim_declarations(scene_text)
    matches = [
        declaration
        for declaration in declarations
        if (
            declaration["specifier"] == "def"
            and declaration["prim_type"] == prim_type
            and declaration["path"] == prim_path
        )
    ]
    display_path = "/" + "/".join(prim_path)
    if not matches:
        raise ValueError(
            f"scene is missing {prim_type} {display_path!r}")
    if len(matches) != 1:
        raise ValueError(
            f"scene has ambiguous {prim_type} {display_path!r}")
    declaration = matches[0]
    direct_children = [
        child
        for child in declarations
        if (
            child["container_open_token_index"]
            == declaration["open_token_index"]
        )
    ]
    prefix_end = min(
        (child["start"] for child in direct_children),
        default=declaration["close"],
    )
    subtree_assets = tuple(
        token[1]
        for token_index, token in enumerate(tokens)
        if (
            token[0] == "asset"
            and declaration["start_token_index"]
            <= token_index
            <= declaration["close_token_index"]
        )
    )
    return {
        "header": scene_text[
            declaration["name_end"]:declaration["open"]],
        "prefix": scene_text[
            declaration["open_end"]:prefix_end],
        "subtree": scene_text[
            declaration["open_end"]:declaration["close"]],
        "asset_paths": subtree_assets,
    }


def _parse_numeric_tuple(text, attribute, length):
    number = r"([-+0-9.eE]+)"
    match = re.search(
        rf"\b{re.escape(attribute)}\s*=\s*\(\s*"
        + r"\s*,\s*".join(number for _ in range(length))
        + r"\s*\)",
        text,
    )
    if not match:
        raise ValueError(
            f"scene has no parseable authored {attribute}")
    try:
        values = tuple(float(value) for value in match.groups())
    except ValueError as exc:
        raise ValueError(
            f"scene has no parseable authored {attribute}") from exc
    if not all(
            value == value and abs(value) != float("inf")
            for value in values):
        raise ValueError(
            f"scene authored {attribute} must be finite")
    return values


def _task53_scene_state(scene_payload):
    """Extract exact authored Wallet/cushion state from Task53's root layer."""
    try:
        scene_text = scene_payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(
            "Task53 provenance requires one UTF-8 USDA root scene") from exc
    wallet = _scene_prim_scope(
        scene_text, "Xform", ("World", "Assets", "Wallet"))
    cushion = _scene_prim_scope(
        scene_text, "Xform", ("World", "Assets", "CushionOnBook"))
    pillow = _scene_prim_scope(
        scene_text, "Mesh",
        ("World", "Assets", "CushionOnBook", "Task53Pillow"))
    wallet_box = _scene_prim_scope(
        scene_text, "Cube",
        ("World", "Assets", "Wallet", "Task53CollisionBox"))
    active_match = re.search(
        r"\bactive\s*=\s*(true|false|0|1)\b",
        cushion["header"],
        re.IGNORECASE,
    )
    if not active_match:
        raise ValueError(
            "Task53 CushionOnBook must explicitly author active state")
    cushion_active = active_match.group(1).lower() in ("true", "1")
    if "PhysicsRigidBodyAPI" not in cushion["header"]:
        raise ValueError(
            "Task53 CushionOnBook must author PhysicsRigidBodyAPI")
    for attribute in ("physics:kinematicEnabled",
                      "physics:rigidBodyEnabled"):
        if not re.search(
                rf"\b(?:custom\s+)?bool\s+{re.escape(attribute)}\s*=\s*"
                r"(?:1|true)\b",
                cushion["prefix"],
                re.IGNORECASE):
            raise ValueError(
                f"Task53 CushionOnBook must author {attribute}=true")
    if "./usdz/cushions_set.usdz" not in cushion["asset_paths"]:
        raise ValueError(
            "Task53 CushionOnBook does not reference the canonical cushion "
            "asset")
    geometry_match = re.search(
        r'\bcustom\s+string\s+real2sim:geometryVersion\s*=\s*"([^"]+)"',
        cushion["prefix"],
    )
    if (
        not geometry_match
        or geometry_match.group(1) != TASK53_SUPPORT_GEOMETRY_VERSION
        or "PhysicsCollisionAPI" not in pillow["header"]
        or "PhysicsMeshCollisionAPI" not in pillow["header"]
        or "material:binding:physics" not in pillow["prefix"]
        or not re.search(
            r'\bphysics:approximation\s*=\s*"convexHull"',
            pillow["prefix"],
        )
    ):
        raise ValueError(
            "Task53 CushionOnBook has no exact rounded-pillow "
            "collision/material geometry")
    wallet_collision_match = re.search(
        r'\bcustom\s+string\s+real2sim:collisionGeometryVersion'
        r'\s*=\s*"([^"]+)"',
        wallet["prefix"],
    )
    if (
        not wallet_collision_match
        or wallet_collision_match.group(1)
        != TASK53_TARGET_COLLISION_GEOMETRY_VERSION
        or "PhysicsCollisionAPI" not in wallet_box["header"]
        or not re.search(
            r"\bbool\s+physics:collisionEnabled\s*=\s*(?:1|true)\b",
            wallet_box["prefix"],
            re.IGNORECASE,
        )
        or not re.search(
            r'\btoken\s+visibility\s*=\s*"invisible"',
            wallet_box["prefix"],
        )
        or not re.search(
            r'\bdouble\s+size\s*=\s*2(?:\.0+)?\b',
            wallet_box["prefix"],
        )
        or _parse_numeric_tuple(
            wallet_box["prefix"], "xformOp:translate", 3)
        != (0.0, 0.0, 0.02)
        or _parse_numeric_tuple(
            wallet_box["prefix"], "xformOp:scale", 3)
        != (0.0825, 0.065, 0.02)
        or not re.search(
            r'over\s+"Mesh".*?\bbool\s+physics:collisionEnabled'
            r"\s*=\s*(?:0|false)\b",
            wallet["subtree"],
            re.IGNORECASE | re.DOTALL,
        )
    ):
        raise ValueError(
            "Task53 Wallet has no exact visible-envelope box collision LOD")
    return {
        "task53_wallet_translate_xyz": _parse_numeric_tuple(
            wallet["prefix"], "xformOp:translate", 3),
        "task53_wallet_rotate_xyz_deg": _parse_numeric_tuple(
            wallet["prefix"], "xformOp:rotateXYZ", 3),
        "task53_wallet_scale_xyz": _parse_numeric_tuple(
            wallet["prefix"], "xformOp:scale", 3),
        "task53_wallet_collision_geometry_version": (
            TASK53_TARGET_COLLISION_GEOMETRY_VERSION),
        "task53_wallet_collision_proxy_path": (
            TASK53_TARGET_COLLISION_PROXY_PATH),
        "task53_wallet_source_collision_disabled": True,
        "task53_cushion_path": TASK53_SUPPORT_PATH,
        "task53_cushion_active": cushion_active,
        "task53_cushion_kinematic_enabled": True,
        "task53_cushion_rigid_body_enabled": True,
        "task53_cushion_collision_authored": True,
        "task53_cushion_reference": "./usdz/cushions_set.usdz",
        "task53_cushion_geometry_version": (
            TASK53_SUPPORT_GEOMETRY_VERSION),
        "task53_cushion_translate_xyz": _parse_numeric_tuple(
            cushion["prefix"], "xformOp:translate", 3),
        "task53_cushion_orient_wxyz": _parse_numeric_tuple(
            cushion["prefix"], "xformOp:orient", 4),
    }


def _validate_scene_contract(scene_payload, target_prim):
    try:
        scene_text = scene_payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(
            "pick provenance requires one UTF-8 USDA root scene") from exc
    if target_prim not in TARGET_REFERENCES:
        raise ValueError(
            f"unsupported pick provenance target {target_prim!r}")

    target_scope = _scene_prim_scope(
        scene_text, "Xform", ("World", "Assets", target_prim))
    target_header = target_scope["header"]
    if "PhysicsRigidBodyAPI" not in target_header:
        raise ValueError(
            f"{target_prim} is not authored as a rigid body")
    target_prefix = target_scope["prefix"]
    if not re.search(
            r"\b(?:custom\s+)?bool\s+physics:kinematicEnabled\s*=\s*"
            r"(?:0|false)\b", target_prefix):
        raise ValueError(
            f"{target_prim} must author physics:kinematicEnabled=false")
    target_reference_path = TARGET_REFERENCES[target_prim]
    target_reference = f"@{target_reference_path}@"
    if target_reference_path not in target_scope["asset_paths"]:
        raise ValueError(
            f"scene does not reference canonical {target_prim} asset "
            f"{target_reference}")

    riser_header = _scene_prim_header(
        scene_text, "Cube", "SeatRiser")
    if "PhysicsCollisionAPI" not in riser_header:
        raise ValueError(
            "SeatRiser must author PhysicsCollisionAPI")
    riser_prefix = _scene_prim_prefix(
        scene_text, "Cube", "SeatRiser")
    for required in (
            "material:binding:physics",
            "xformOp:scale",
            "xformOp:translate"):
        if required not in riser_prefix:
            raise ValueError(
                f"SeatRiser is missing authored {required}")
    active_match = re.search(
        r"\bactive\s*=\s*(true|false|0|1)\b",
        riser_header,
        re.IGNORECASE,
    )
    if not active_match:
        raise ValueError(
            "SeatRiser must explicitly author active state")
    seat_riser_active = active_match.group(1).lower() in ("true", "1")
    scale_match = re.search(
        r"\bxformOp:scale\s*=\s*\(\s*"
        r"([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*,\s*"
        r"([-+0-9.eE]+)\s*\)",
        riser_prefix,
    )
    if not scale_match:
        raise ValueError(
            "SeatRiser has no parseable authored scale")
    try:
        seat_riser_scale = tuple(
            float(value) for value in scale_match.groups())
    except ValueError as exc:
        raise ValueError(
            "SeatRiser has no parseable authored scale") from exc
    if (
        not all(value == value and abs(value) != float("inf")
                for value in seat_riser_scale)
        or any(value <= 0.0 for value in seat_riser_scale)
    ):
        raise ValueError(
            "SeatRiser authored scale must be finite and positive")
    translate_match = re.search(
        r"\bxformOp:translate\s*=\s*\(\s*"
        r"([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*,\s*"
        r"([-+0-9.eE]+)\s*\)",
        riser_prefix,
    )
    if not translate_match:
        raise ValueError(
            "SeatRiser has no parseable authored translation")
    try:
        seat_riser_translate = tuple(
            float(value) for value in translate_match.groups())
    except ValueError as exc:
        raise ValueError(
            "SeatRiser has no parseable authored translation") from exc
    if not all(
            value == value and abs(value) != float("inf")
            for value in seat_riser_translate):
        raise ValueError(
            "SeatRiser authored translation must be finite")

    robot_header = _scene_prim_header(scene_text, "Xform", "Robot")
    robot_reference = f"@{ROBOT_REFERENCE}@"
    if robot_reference not in robot_header:
        raise ValueError(
            "scene does not reference the canonical G1 omnipicker USD")
    return {
        "seat_riser_active": seat_riser_active,
        "seat_riser_scale_xyz": seat_riser_scale,
        "seat_riser_translate_xyz": seat_riser_translate,
        "seat_riser_support_height_m": (
            float(seat_riser_scale[2])
            if seat_riser_active else 0.0),
        "seat_riser_support_top_m": (
            float(
                seat_riser_translate[2]
                + 0.5 * seat_riser_scale[2])
            if seat_riser_active else 0.0),
        "support_mode": (
            "seat_riser" if seat_riser_active else "native_surface"),
    }


def _authorize_runtime_support(identity, calibration, episode_uid):
    """Attach support facts by comparing parsed fields and reviewed values."""

    identity = dict(identity)
    task_id = str(identity.get("task_id"))
    if task_id == "59":
        if not identity.get("seat_riser_active"):
            identity["support_authority"] = "native_surface"
            return identity
        if not isinstance(calibration, dict):
            raise ValueError(
                "Task59 active SeatRiser requires reviewed depth calibration")
        support_top = float(identity["seat_riser_support_top_m"])
        if not (
            float(calibration["authorized_min_m"])
            <= support_top
            <= float(calibration["authorized_max_m"])
        ):
            raise ValueError(
                "Task59 SeatRiser support top is outside calibration bounds")
        if abs(support_top - float(calibration["support_top_m"])) > 1e-9:
            raise ValueError(
                "Task59 SeatRiser support top differs from calibration")
        identity.update({
            "support_authority": TASK59_SUPPORT_AUTHORITY,
            "authorized_support_top_m": float(
                calibration["support_top_m"]),
            "authorized_support_uncertainty_m": float(
                calibration["uncertainty_m"]),
            "authorized_support_min_m": float(
                calibration["authorized_min_m"]),
            "authorized_support_max_m": float(
                calibration["authorized_max_m"]),
        })
        return identity

    if task_id != "53":
        identity["support_authority"] = "scene_fields"
        return identity

    episode = task53_episode_calibration(calibration, episode_uid)
    if identity.get("seat_riser_active"):
        raise ValueError(
            "Task53 visible-cushion scene forbids an active SeatRiser")
    if (
        not identity.get("task53_cushion_active")
        or not identity.get("task53_cushion_kinematic_enabled")
        or not identity.get("task53_cushion_rigid_body_enabled")
        or not identity.get("task53_cushion_collision_authored")
        or identity.get("task53_cushion_path") != TASK53_SUPPORT_PATH
        or identity.get("task53_cushion_reference")
        != calibration["support_reference"]
        or identity.get("task53_cushion_geometry_version")
        != calibration["support_geometry_version"]
        or identity.get("task53_wallet_collision_geometry_version")
        != TASK53_TARGET_COLLISION_GEOMETRY_VERSION
        or identity.get("task53_wallet_collision_proxy_path")
        != TASK53_TARGET_COLLISION_PROXY_PATH
        or identity.get("task53_wallet_source_collision_disabled") is not True
    ):
        raise ValueError(
            "Task53 scene lacks its explicit visible colliding cushion")

    def require_close(label, actual, expected, tolerance=5e-8):
        actual = tuple(float(value) for value in actual)
        expected = tuple(float(value) for value in expected)
        if len(actual) != len(expected) or any(
            abs(left - right) > tolerance
            for left, right in zip(actual, expected)
        ):
            raise ValueError(
                f"Task53 {label} does not match episode {episode_uid}")

    require_close(
        "Wallet scale",
        identity["task53_wallet_scale_xyz"],
        calibration["target_scale_xyz"],
    )
    require_close(
        "Wallet translation",
        identity["task53_wallet_translate_xyz"],
        episode["wallet_translate_xyz_m"],
    )
    require_close(
        "Wallet flat/yaw pose",
        identity["task53_wallet_rotate_xyz_deg"],
        (0.0, 0.0, episode["wallet_yaw_deg"]),
        tolerance=1e-6,
    )
    require_close(
        "cushion translation",
        identity["task53_cushion_translate_xyz"],
        episode["cushion_translate_xyz_m"],
    )
    require_close(
        "cushion orientation",
        identity["task53_cushion_orient_wxyz"],
        episode["cushion_orient_wxyz"],
    )
    if abs(
        float(identity["task53_wallet_translate_xyz"][2])
        - float(episode["support_top_m"])
    ) > 1e-8:
        raise ValueError(
            "Task53 Wallet bottom does not meet the calibrated cushion top")
    identity.update({
        "support_mode": "visible_cushion",
        "support_authority": TASK53_SUPPORT_AUTHORITY,
        "authorized_episode_uid": str(episode_uid),
        "authorized_support_path": TASK53_SUPPORT_PATH,
        "authorized_support_top_m": float(episode["support_top_m"]),
    })
    return identity


def scene_runtime_facts(
        scene_path, *, task_id, target_prim, support_calibration=None,
        episode_uid=None):
    """Validate and return the scene facts that affect physical scoring.

    Versioned file identity comes from Git.  This function compares authored
    USDA fields and calibration values directly.
    """

    scene_payload = _read_snapshot(os.path.abspath(scene_path))
    facts = _validate_scene_contract(scene_payload, str(target_prim))
    if str(task_id) == "53":
        facts.update(_task53_scene_state(scene_payload))
    identity = {
        "schema": PICK_SCENE_PROVENANCE_SCHEMA,
        "task_id": str(task_id),
        "target_body_path": f"/World/Assets/{str(target_prim)}",
        "target_reference": TARGET_REFERENCES[str(target_prim)],
        "target_kinematic_enabled": False,
        "seat_riser_path": SEAT_RISER_PATH,
        "robot_reference": ROBOT_REFERENCE,
        **facts,
    }
    return _authorize_runtime_support(
        identity, support_calibration, episode_uid)


def native_wallet_scene_identity(scene_path):
    """Validate owner-authored native support; this is not a grasp success claim."""
    from pxr import Usd, UsdGeom, UsdPhysics
    stage = Usd.Stage.Open(str(scene_path))
    if not stage:
        raise ValueError('Native Wallet scene cannot be opened')
    wallet = stage.GetPrimAtPath('/World/Assets/Wallet')
    if not wallet or not wallet.IsActive():
        raise ValueError('Native Wallet target is missing')
    branch = wallet.GetAttribute('real2sim:nativeWalletSupport').Get()
    if branch not in ('seat', 'cushion'):
        raise ValueError('Native Wallet support branch must be authored before physics')
    seed = wallet.GetAttribute('real2sim:nativeWalletSeed').Get()
    if not isinstance(seed, int):
        raise ValueError('Native Wallet episode seed is missing')
    for name in ('SeatRiser', 'SeatOverlay'):
        prim = stage.GetPrimAtPath('/World/Assets/' + name)
        if prim and prim.IsActive():
            raise ValueError('Native Wallet forbids hidden support ' + name)
    def dynamic(prim):
        api = UsdPhysics.RigidBodyAPI(prim)
        if not api or api.GetKinematicEnabledAttr().Get() or not api.GetRigidBodyEnabledAttr().Get():
            raise ValueError('Native Wallet target and cushion must be dynamic: ' + str(prim.GetPath()))
    dynamic(wallet)
    cushion = stage.GetPrimAtPath('/World/Assets/CushionOnBook')
    if branch == 'cushion':
        if not cushion or not cushion.IsActive():
            raise ValueError('Required real cushion is missing')
        dynamic(cushion)
        obj = cushion.GetChild('Obj')
        if not obj or not obj.IsActive():
            raise ValueError('Native cushion source Obj is missing')
        references = [ref.assetPath for prim in Usd.PrimRange(obj)
                      for spec in prim.GetPrimStack()
                      for ref in spec.referenceList.GetAddedOrExplicitItems()]
        if not any(path.replace('\\', '/').endswith('usdz/cushions_set.usdz') for path in references):
            raise ValueError('Cushion must retain its real cushions_set.usdz reference')
        visible = 0
        for prim in Usd.PrimRange(cushion):
            if prim.GetName() == 'Task53Pillow' and prim.IsActive():
                raise ValueError('Synthetic Task53Pillow is forbidden')
            if not prim.IsA(UsdGeom.Mesh):
                continue
            shown = UsdGeom.Imageable(prim).ComputeVisibility() != 'invisible'
            collision = UsdPhysics.CollisionAPI(prim)
            enabled = bool(collision and collision.GetCollisionEnabledAttr().Get())
            if shown:
                visible += 1
                approximation = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
                if not enabled or approximation != 'convexDecomposition':
                    raise ValueError('Real cushion visible collision must preserve its shape')
            else:
                raise ValueError('Invisible cushion mesh must be inactive to exclude its mass')
        if not visible:
            raise ValueError('Real cushion has no visible mesh')
    elif cushion and cushion.IsActive():
        raise ValueError('Direct-seat branch must retain its sampled cushion absence')
    sofa = stage.GetPrimAtPath('/World/Assets/Sofa')
    if not sofa or not sofa.IsActive() or not any(
            prim.IsA(UsdGeom.Mesh)
            and UsdGeom.Imageable(prim).ComputeVisibility() != 'invisible'
            and UsdPhysics.CollisionAPI(prim)
            and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
            for prim in Usd.PrimRange(sofa)):
        raise ValueError('Native Wallet requires the visible colliding sofa')
    return dict(support_authority='native_house_wallet_support_v1',
                authorized_support_path='/World/Assets/CushionOnBook' if branch == 'cushion' else '/World/Assets/Sofa',
                branch=branch, seed=seed, historical_episode_required=False,
                physics_success_verified=False)
