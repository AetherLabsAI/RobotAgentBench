"""Immutable Task 29 destination geometry.

The Task 29 initializer emits the composed destination description at
``init_result["placement"]["task_bins"]["destination"]``.  This module
normalizes that JSON-shaped value into frozen Python objects and rejects
partial, ambiguous, or internally inconsistent geometry.

Only the teal destination is part of this contract.  In particular, no global
scorer bin table and no red distractor geometry is consulted.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
from numbers import Real


TASK_ID = "29"
CONTRACT_SCHEMA = "real2sim.task29_destination_contract.v2"
SOURCE_PATH = "init_result.placement.task_bins.destination"

DESTINATION_SLOT = "destination"
DESTINATION_NAME = "teal"
DESTINATION_ROLE = "calibrated_destination_seed"
DESTINATION_BODY_PATH = "/World/Assets/TrashBinTeal"
CALIBRATED_CENTER_XY = (-2.60, -0.75)
ENABLED_COLLIDER_PATH = (
    "/World/Assets/TrashBinTeal/geo/Cube_022"
)
DISABLED_COLLIDER_PATH = (
    "/World/Assets/TrashBinTeal/geo/Trash_Bag/Trash_Bag_002"
)
BOTTOM_COLLISION_LOD_PATH = (
    "/World/Assets/TrashBinTeal/geo/Real2SimBottomCollisionLOD"
)
EXPECTED_COLLIDER_PATHS = frozenset(
    (
        ENABLED_COLLIDER_PATH,
        DISABLED_COLLIDER_PATH,
        BOTTOM_COLLISION_LOD_PATH,
    )
)
EXPECTED_ENABLED_COLLIDER_PATHS = frozenset(
    (ENABLED_COLLIDER_PATH, BOTTOM_COLLISION_LOD_PATH)
)

GEOMETRY_DESCRIPTOR_SCHEMA = "task29.composed_bin_geometry.v2"
BOUNDS_KIND = "visible_destination_collider_world_aabb"
CALIBRATION_BASIS = "raw_rgbd_frame0_backprojection_seed"
RIM_DEFINITION = "collision_aabb_max_z"
BOTTOM_COLLISION_LOD_SCHEMA = "task29.geometry_derived_bottom_lod.v1"
BOTTOM_COLLISION_LOD_THICKNESS_M = 0.006
BOTTOM_COLLISION_LOD_FACE_VERTEX_COUNT = 16
BOTTOM_COLLISION_LOD_EXTRUSION_DIRECTION = "world_negative_z"
BOTTOM_COLLISION_LOD_TOPOLOGY = "closed_extruded_polygon_v1"
BOTTOM_COLLISION_LOD_APPROXIMATION = "convexHull"
BOTTOM_COLLISION_LOD_PURPOSE = "guide"
BOTTOM_COLLISION_LOD_VISIBILITY = "invisible"
BOTTOM_FACE_TOLERANCE_M = 1e-4

# The initializer stores AABB components rounded to six decimal places and the
# calibration error rounded to nine.  Two micrometres safely covers composing
# those rounded values while remaining far below the centimetre calibration
# acceptance threshold.
CONSISTENCY_TOLERANCE_M = 2e-6
MAX_CENTER_ERROR_M = 0.010
QUATERNION_NORM_TOLERANCE = 1e-5


class Task29DestinationContractError(ValueError):
    """The initializer result is not a valid Task 29 destination contract."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _require_mapping(
    value: object,
    field: str,
    keys: frozenset[str],
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise Task29DestinationContractError(f"{field} must be a mapping")
    actual = frozenset(value)
    if actual != keys:
        missing = sorted(keys - actual)
        extra = sorted(actual - keys)
        raise Task29DestinationContractError(
            f"{field} fields are not canonical: "
            f"missing={missing!r}, extra={extra!r}"
        )
    return value


def _finite_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise Task29DestinationContractError(
            f"{field} must be a finite JSON number"
        )
    result = float(value)
    if not math.isfinite(result):
        raise Task29DestinationContractError(f"{field} must be finite")
    # Normalize signed zero so canonical structural output stays readable.
    return 0.0 if result == 0.0 else result


def _vector(
    value: object,
    field: str,
    length: int,
    *,
    positive: bool = False,
) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise Task29DestinationContractError(
            f"{field} must contain exactly {length} numbers"
        )
    result = tuple(
        _finite_float(component, f"{field}[{index}]")
        for index, component in enumerate(value)
    )
    if positive and not all(component > 0.0 for component in result):
        raise Task29DestinationContractError(
            f"{field} components must all be positive"
        )
    return result


def _close(
    actual: float,
    expected: float,
    field: str,
    *,
    tolerance: float = CONSISTENCY_TOLERANCE_M,
) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        raise Task29DestinationContractError(
            f"{field} is inconsistent: {actual!r} != {expected!r}"
        )


@dataclass(frozen=True, slots=True)
class RootTransform:
    """The complete canonical transform stack on the teal bin root."""

    translate: tuple[float, float, float]
    orient_wxyz: tuple[float, float, float, float]
    scale: tuple[float, float, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "translate": list(self.translate),
            "orient_wxyz": list(self.orient_wxyz),
            "scale": list(self.scale),
        }


@dataclass(frozen=True, slots=True)
class ColliderState:
    """One authored collider and its complete active/enabled state."""

    path: str
    active: bool
    collision_enabled: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "active": self.active,
            "collision_enabled": self.collision_enabled,
        }


@dataclass(frozen=True, slots=True)
class WorldAabb:
    """Internally consistent composed world-space AABB."""

    minimum: tuple[float, float, float]
    maximum: tuple[float, float, float]
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    bottom_z: float
    rim_z: float

    def to_dict(self) -> dict[str, object]:
        return {
            "minimum": list(self.minimum),
            "maximum": list(self.maximum),
            "center": list(self.center),
            "size": list(self.size),
            "bottom_z": self.bottom_z,
            "rim_z": self.rim_z,
        }


@dataclass(frozen=True, slots=True)
class GeometryDescriptor:
    """Meaning and explicit limitations of the composed geometry."""

    schema: str
    bounds_kind: str
    calibration_basis: str
    rim_definition: str
    interior_geometry_asserted: bool
    scoring_polygon_asserted: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "bounds_kind": self.bounds_kind,
            "calibration_basis": self.calibration_basis,
            "rim_definition": self.rim_definition,
            "interior_geometry_asserted": self.interior_geometry_asserted,
            "scoring_polygon_asserted": self.scoring_polygon_asserted,
        }


@dataclass(frozen=True, slots=True)
class BottomCollisionLod:
    """Exact finite collision refinement derived from the visible bottom face."""

    schema: str
    path: str
    source_collider_path: str
    source_face_vertex_count: int
    top_face_world_xyz: tuple[tuple[float, float, float], ...]
    bottom_face_world_xyz: tuple[tuple[float, float, float], ...]
    thickness_m: float
    extrusion_direction: str
    topology: str
    approximation: str
    purpose: str
    visibility: str
    double_sided: bool
    collision_enabled: bool
    rigid_body: bool
    enabled_from_frame: int
    source_double_sided: bool
    source_approximation: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "path": self.path,
            "source_collider_path": self.source_collider_path,
            "source_face_vertex_count": self.source_face_vertex_count,
            "top_face_world_xyz": [
                list(point) for point in self.top_face_world_xyz
            ],
            "bottom_face_world_xyz": [
                list(point) for point in self.bottom_face_world_xyz
            ],
            "thickness_m": self.thickness_m,
            "extrusion_direction": self.extrusion_direction,
            "topology": self.topology,
            "approximation": self.approximation,
            "purpose": self.purpose,
            "visibility": self.visibility,
            "double_sided": self.double_sided,
            "collision_enabled": self.collision_enabled,
            "rigid_body": self.rigid_body,
            "enabled_from_frame": self.enabled_from_frame,
            "source_double_sided": self.source_double_sided,
            "source_approximation": self.source_approximation,
        }


@dataclass(frozen=True, slots=True)
class ConservativeHorizontalZone:
    """A destination AABB eroded by an object's horizontal half-extent."""

    minimum_xy: tuple[float, float]
    maximum_xy: tuple[float, float]
    center_xy: tuple[float, float]
    size_xy: tuple[float, float]
    object_half_extent_xy: tuple[float, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "minimum_xy": list(self.minimum_xy),
            "maximum_xy": list(self.maximum_xy),
            "center_xy": list(self.center_xy),
            "size_xy": list(self.size_xy),
            "object_half_extent_xy": list(self.object_half_extent_xy),
        }


@dataclass(frozen=True, slots=True)
class Task29DestinationContract:
    """Frozen normalized contract for Task 29's teal destination."""

    schema: str
    task_id: str
    source_path: str
    slot: str
    name: str
    role: str
    body_path: str
    body_active: bool
    root_transform: RootTransform
    collider_path: str
    collider_states: tuple[ColliderState, ...]
    calibrated_center_xy: tuple[float, float]
    center_error_m: float
    maximum_center_error_m: float
    world_aabb: WorldAabb
    bottom_collision_lod: BottomCollisionLod
    geometry_descriptor: GeometryDescriptor

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "task_id": self.task_id,
            "source_path": self.source_path,
            "slot": self.slot,
            "name": self.name,
            "role": self.role,
            "body_path": self.body_path,
            "body_active": self.body_active,
            "root_transform": self.root_transform.to_dict(),
            "collider_path": self.collider_path,
            "collider_states": [
                state.to_dict() for state in self.collider_states
            ],
            "calibrated_center_xy": list(self.calibrated_center_xy),
            "center_error_m": self.center_error_m,
            "maximum_center_error_m": self.maximum_center_error_m,
            "world_aabb": self.world_aabb.to_dict(),
            "bottom_collision_lod": self.bottom_collision_lod.to_dict(),
            "geometry_descriptor": self.geometry_descriptor.to_dict(),
        }

    def canonical_json_bytes(self) -> bytes:
        """Return deterministic bytes for direct comparison or export."""

        return _canonical_json(self.to_dict())

    def conservative_horizontal_zone(
        self,
        object_horizontal_half_extent: object,
    ) -> ConservativeHorizontalZone:
        """Erode this destination's composed AABB for an object footprint."""

        return conservative_horizontal_zone(
            self.world_aabb,
            object_horizontal_half_extent,
        )


_DESTINATION_FIELDS = frozenset(
    {
        "slot",
        "name",
        "role",
        "body_path",
        "body_active",
        "root_transform",
        "collider_path",
        "collider_states",
        "calibrated_center_xy",
        "center_error_m",
        "bounds",
        "bottom_collision_lod",
        "geometry_descriptor",
    }
)
_ROOT_TRANSFORM_FIELDS = frozenset(
    {"translate", "orient_wxyz", "scale"}
)
_COLLIDER_STATE_FIELDS = frozenset(
    {"path", "active", "collision_enabled"}
)
_BOUNDS_FIELDS = frozenset(
    {"minimum", "maximum", "center", "size", "bottom_z", "rim_z"}
)
_GEOMETRY_DESCRIPTOR_FIELDS = frozenset(
    {
        "schema",
        "bounds_kind",
        "calibration_basis",
        "rim_definition",
        "interior_geometry_asserted",
        "scoring_polygon_asserted",
    }
)
_BOTTOM_COLLISION_LOD_FIELDS = frozenset(
    {
        "schema",
        "path",
        "source_collider_path",
        "source_face_vertex_count",
        "top_face_world_xyz",
        "bottom_face_world_xyz",
        "thickness_m",
        "extrusion_direction",
        "topology",
        "approximation",
        "purpose",
        "visibility",
        "double_sided",
        "collision_enabled",
        "rigid_body",
        "enabled_from_frame",
        "source_double_sided",
        "source_approximation",
    }
)


def _validate_task_identity(init_result: Mapping[str, object]) -> None:
    """Bind the legacy initializer result to Task 29 without inference by bin."""

    if init_result.get("ok") is not True:
        raise Task29DestinationContractError(
            "Task 29 init_result must have ok=true"
        )
    if init_result.get("target") != "PaperCup":
        raise Task29DestinationContractError(
            "Task 29 init_result target must be PaperCup"
        )
    required_count = init_result.get("required_count")
    if type(required_count) is not int or required_count != 3:
        raise Task29DestinationContractError(
            "Task 29 init_result required_count must be integer 3"
        )
    if init_result.get("probe_scope") != "single_cup_partial":
        raise Task29DestinationContractError(
            "Task 29 init_result probe_scope is not canonical"
        )
    if "task_id" in init_result and init_result["task_id"] != TASK_ID:
        raise Task29DestinationContractError(
            "init_result task_id does not identify Task 29"
        )


def _parse_root_transform(value: object) -> RootTransform:
    root = _require_mapping(
        value,
        "destination.root_transform",
        _ROOT_TRANSFORM_FIELDS,
    )
    translate = _vector(
        root["translate"], "destination.root_transform.translate", 3
    )
    orient = _vector(
        root["orient_wxyz"],
        "destination.root_transform.orient_wxyz",
        4,
    )
    norm = math.sqrt(sum(component * component for component in orient))
    if not math.isclose(
        norm,
        1.0,
        rel_tol=0.0,
        abs_tol=QUATERNION_NORM_TOLERANCE,
    ):
        raise Task29DestinationContractError(
            "destination root quaternion must have unit norm"
        )
    scale = _vector(
        root["scale"],
        "destination.root_transform.scale",
        3,
        positive=True,
    )
    return RootTransform(
        translate=translate,  # type: ignore[arg-type]
        orient_wxyz=orient,  # type: ignore[arg-type]
        scale=scale,  # type: ignore[arg-type]
    )


def _parse_collider_states(value: object) -> tuple[ColliderState, ...]:
    if not isinstance(value, (list, tuple)):
        raise Task29DestinationContractError(
            "destination.collider_states must be a list"
        )
    states: list[ColliderState] = []
    seen: set[str] = set()
    for index, raw_state in enumerate(value):
        state = _require_mapping(
            raw_state,
            f"destination.collider_states[{index}]",
            _COLLIDER_STATE_FIELDS,
        )
        path = state["path"]
        if not isinstance(path, str) or path in seen:
            raise Task29DestinationContractError(
                "destination collider paths must be unique strings"
            )
        active = state["active"]
        enabled = state["collision_enabled"]
        if type(active) is not bool or type(enabled) is not bool:
            raise Task29DestinationContractError(
                "destination collider active/enabled states must be booleans"
            )
        if active is not True:
            raise Task29DestinationContractError(
                f"destination collider is not active: {path}"
            )
        seen.add(path)
        states.append(
            ColliderState(
                path=path,
                active=active,
                collision_enabled=enabled,
            )
        )
    if seen != EXPECTED_COLLIDER_PATHS:
        raise Task29DestinationContractError(
            "destination collider_states do not completely describe the "
            "canonical teal bin colliders"
        )
    states.sort(key=lambda state: state.path)
    enabled_paths = frozenset(
        state.path for state in states if state.collision_enabled
    )
    if enabled_paths != EXPECTED_ENABLED_COLLIDER_PATHS:
        raise Task29DestinationContractError(
            "destination requires exactly the canonical enabled colliders"
        )
    return tuple(states)


def _parse_world_aabb(value: object) -> WorldAabb:
    bounds = _require_mapping(
        value,
        "destination.bounds",
        _BOUNDS_FIELDS,
    )
    minimum = _vector(
        bounds["minimum"], "destination.bounds.minimum", 3
    )
    maximum = _vector(
        bounds["maximum"], "destination.bounds.maximum", 3
    )
    center = _vector(bounds["center"], "destination.bounds.center", 3)
    size = _vector(
        bounds["size"], "destination.bounds.size", 3, positive=True
    )
    bottom_z = _finite_float(
        bounds["bottom_z"], "destination.bounds.bottom_z"
    )
    rim_z = _finite_float(bounds["rim_z"], "destination.bounds.rim_z")

    for axis in range(3):
        if not minimum[axis] < maximum[axis]:
            raise Task29DestinationContractError(
                f"destination world AABB axis {axis} has no positive width"
            )
        expected_center = (minimum[axis] + maximum[axis]) / 2.0
        expected_size = maximum[axis] - minimum[axis]
        _close(
            center[axis],
            expected_center,
            f"destination.bounds.center[{axis}]",
        )
        _close(
            size[axis],
            expected_size,
            f"destination.bounds.size[{axis}]",
        )
    _close(bottom_z, minimum[2], "destination.bounds.bottom_z")
    _close(rim_z, maximum[2], "destination.bounds.rim_z")

    return WorldAabb(
        minimum=minimum,  # type: ignore[arg-type]
        maximum=maximum,  # type: ignore[arg-type]
        center=center,  # type: ignore[arg-type]
        size=size,  # type: ignore[arg-type]
        bottom_z=bottom_z,
        rim_z=rim_z,
    )


def _parse_geometry_descriptor(value: object) -> GeometryDescriptor:
    descriptor = _require_mapping(
        value,
        "destination.geometry_descriptor",
        _GEOMETRY_DESCRIPTOR_FIELDS,
    )
    expected_strings = {
        "schema": GEOMETRY_DESCRIPTOR_SCHEMA,
        "bounds_kind": BOUNDS_KIND,
        "calibration_basis": CALIBRATION_BASIS,
        "rim_definition": RIM_DEFINITION,
    }
    for field, expected in expected_strings.items():
        if descriptor[field] != expected:
            raise Task29DestinationContractError(
                f"destination geometry descriptor {field} is not canonical"
            )
    if descriptor["interior_geometry_asserted"] is not False:
        raise Task29DestinationContractError(
            "Task 29 destination interior geometry must remain unasserted"
        )
    if descriptor["scoring_polygon_asserted"] is not False:
        raise Task29DestinationContractError(
            "Task 29 destination scoring polygon must remain unasserted"
        )
    return GeometryDescriptor(
        schema=GEOMETRY_DESCRIPTOR_SCHEMA,
        bounds_kind=BOUNDS_KIND,
        calibration_basis=CALIBRATION_BASIS,
        rim_definition=RIM_DEFINITION,
        interior_geometry_asserted=False,
        scoring_polygon_asserted=False,
    )


def _point_sequence(
    value: object,
    field: str,
) -> tuple[tuple[float, float, float], ...]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != BOTTOM_COLLISION_LOD_FACE_VERTEX_COUNT
    ):
        raise Task29DestinationContractError(
            f"{field} must contain exactly "
            f"{BOTTOM_COLLISION_LOD_FACE_VERTEX_COUNT} XYZ points"
        )
    return tuple(
        _vector(point, f"{field}[{index}]", 3)  # type: ignore[arg-type]
        for index, point in enumerate(value)
    )


def _validate_convex_xy_polygon(
    points: tuple[tuple[float, float, float], ...],
) -> None:
    xy = tuple((point[0], point[1]) for point in points)
    if len(set(xy)) != len(xy):
        raise Task29DestinationContractError(
            "destination bottom LOD top face has duplicate XY vertices"
        )
    cross_products = []
    for index in range(len(xy)):
        previous = xy[index - 1]
        current = xy[index]
        following = xy[(index + 1) % len(xy)]
        cross_products.append(
            (current[0] - previous[0]) * (following[1] - current[1])
            - (current[1] - previous[1]) * (following[0] - current[0])
        )
    significant = [
        value for value in cross_products if abs(value) > 1e-10
    ]
    if not significant or not (
        all(value > 0.0 for value in significant)
        or all(value < 0.0 for value in significant)
    ):
        raise Task29DestinationContractError(
            "destination bottom LOD top face must be a convex XY polygon"
        )
    twice_area = sum(
        xy[index][0] * xy[(index + 1) % len(xy)][1]
        - xy[(index + 1) % len(xy)][0] * xy[index][1]
        for index in range(len(xy))
    )
    if abs(twice_area) <= 1e-8:
        raise Task29DestinationContractError(
            "destination bottom LOD top face is degenerate"
        )


def _parse_bottom_collision_lod(
    value: object,
    world_aabb: WorldAabb,
) -> BottomCollisionLod:
    lod = _require_mapping(
        value,
        "destination.bottom_collision_lod",
        _BOTTOM_COLLISION_LOD_FIELDS,
    )
    expected_values = {
        "schema": BOTTOM_COLLISION_LOD_SCHEMA,
        "path": BOTTOM_COLLISION_LOD_PATH,
        "source_collider_path": ENABLED_COLLIDER_PATH,
        "source_face_vertex_count": BOTTOM_COLLISION_LOD_FACE_VERTEX_COUNT,
        "extrusion_direction": BOTTOM_COLLISION_LOD_EXTRUSION_DIRECTION,
        "topology": BOTTOM_COLLISION_LOD_TOPOLOGY,
        "approximation": BOTTOM_COLLISION_LOD_APPROXIMATION,
        "purpose": BOTTOM_COLLISION_LOD_PURPOSE,
        "visibility": BOTTOM_COLLISION_LOD_VISIBILITY,
        "double_sided": False,
        "collision_enabled": True,
        "rigid_body": False,
        "enabled_from_frame": 0,
        "source_double_sided": True,
        "source_approximation": "none",
    }
    for field, expected in expected_values.items():
        if lod[field] != expected or type(lod[field]) is not type(expected):
            raise Task29DestinationContractError(
                f"destination bottom collision LOD {field} is not canonical"
            )

    thickness_m = _finite_float(
        lod["thickness_m"],
        "destination.bottom_collision_lod.thickness_m",
    )
    _close(
        thickness_m,
        BOTTOM_COLLISION_LOD_THICKNESS_M,
        "destination.bottom_collision_lod.thickness_m",
    )
    top = _point_sequence(
        lod["top_face_world_xyz"],
        "destination.bottom_collision_lod.top_face_world_xyz",
    )
    bottom = _point_sequence(
        lod["bottom_face_world_xyz"],
        "destination.bottom_collision_lod.bottom_face_world_xyz",
    )
    _validate_convex_xy_polygon(top)

    for index, (top_point, bottom_point) in enumerate(zip(top, bottom)):
        if abs(top_point[2] - world_aabb.bottom_z) > BOTTOM_FACE_TOLERANCE_M:
            raise Task29DestinationContractError(
                "destination bottom LOD top face is not on the visible "
                f"collider bottom at vertex {index}"
            )
        for axis in range(2):
            if (
                top_point[axis]
                < world_aabb.minimum[axis] - BOTTOM_FACE_TOLERANCE_M
                or top_point[axis]
                > world_aabb.maximum[axis] + BOTTOM_FACE_TOLERANCE_M
            ):
                raise Task29DestinationContractError(
                    "destination bottom LOD escaped the visible collider "
                    f"XY bounds at vertex {index}"
                )
            _close(
                bottom_point[axis],
                top_point[axis],
                "destination bottom LOD vertical extrusion "
                f"vertex {index} axis {axis}",
            )
        _close(
            bottom_point[2],
            top_point[2] - thickness_m,
            "destination bottom LOD thickness "
            f"at vertex {index}",
        )

    return BottomCollisionLod(
        schema=BOTTOM_COLLISION_LOD_SCHEMA,
        path=BOTTOM_COLLISION_LOD_PATH,
        source_collider_path=ENABLED_COLLIDER_PATH,
        source_face_vertex_count=BOTTOM_COLLISION_LOD_FACE_VERTEX_COUNT,
        top_face_world_xyz=top,
        bottom_face_world_xyz=bottom,
        thickness_m=thickness_m,
        extrusion_direction=BOTTOM_COLLISION_LOD_EXTRUSION_DIRECTION,
        topology=BOTTOM_COLLISION_LOD_TOPOLOGY,
        approximation=BOTTOM_COLLISION_LOD_APPROXIMATION,
        purpose=BOTTOM_COLLISION_LOD_PURPOSE,
        visibility=BOTTOM_COLLISION_LOD_VISIBILITY,
        double_sided=False,
        collision_enabled=True,
        rigid_body=False,
        enabled_from_frame=0,
        source_double_sided=True,
        source_approximation="none",
    )


def normalize_task29_destination_contract(
    init_result: object,
) -> Task29DestinationContract:
    """Validate and freeze a Task 29 initializer destination.

    The outer Task 29 markers are mandatory because identifying the task from a
    teal-bin path alone would allow a result from another throw task to be
    rebound.  A future explicit ``task_id`` field is accepted only when it is
    exactly ``"29"``.
    """

    if not isinstance(init_result, Mapping):
        raise Task29DestinationContractError(
            "Task 29 init_result must be a mapping"
        )
    _validate_task_identity(init_result)

    placement = init_result.get("placement")
    if not isinstance(placement, Mapping):
        raise Task29DestinationContractError(
            "init_result.placement must be a mapping"
        )
    task_bins = placement.get("task_bins")
    if not isinstance(task_bins, Mapping):
        raise Task29DestinationContractError(
            "init_result.placement.task_bins must be a mapping"
        )
    destination = _require_mapping(
        task_bins.get(DESTINATION_SLOT),
        SOURCE_PATH,
        _DESTINATION_FIELDS,
    )

    exact_values = {
        "slot": DESTINATION_SLOT,
        "name": DESTINATION_NAME,
        "role": DESTINATION_ROLE,
        "body_path": DESTINATION_BODY_PATH,
        "collider_path": ENABLED_COLLIDER_PATH,
    }
    for field, expected in exact_values.items():
        if destination[field] != expected:
            raise Task29DestinationContractError(
                f"Task 29 destination {field} is not canonical"
            )
    if destination["body_active"] is not True:
        raise Task29DestinationContractError(
            "Task 29 destination body must be active"
        )

    root_transform = _parse_root_transform(destination["root_transform"])
    collider_states = _parse_collider_states(
        destination["collider_states"]
    )
    world_aabb = _parse_world_aabb(destination["bounds"])
    bottom_collision_lod = _parse_bottom_collision_lod(
        destination["bottom_collision_lod"],
        world_aabb,
    )
    calibrated_center_xy = _vector(
        destination["calibrated_center_xy"],
        "destination.calibrated_center_xy",
        2,
    )
    for axis, expected in enumerate(CALIBRATED_CENTER_XY):
        _close(
            calibrated_center_xy[axis],
            expected,
            f"destination.calibrated_center_xy[{axis}]",
        )
    center_error_m = _finite_float(
        destination["center_error_m"],
        "destination.center_error_m",
    )
    if center_error_m < 0.0:
        raise Task29DestinationContractError(
            "destination.center_error_m must be non-negative"
        )
    recomputed_center_error = math.hypot(
        world_aabb.center[0] - calibrated_center_xy[0],
        world_aabb.center[1] - calibrated_center_xy[1],
    )
    _close(
        center_error_m,
        recomputed_center_error,
        "destination.center_error_m",
    )
    if recomputed_center_error > (
        MAX_CENTER_ERROR_M + CONSISTENCY_TOLERANCE_M
    ):
        raise Task29DestinationContractError(
            "destination composed AABB center exceeds calibration tolerance"
        )

    geometry_descriptor = _parse_geometry_descriptor(
        destination["geometry_descriptor"]
    )
    return Task29DestinationContract(
        schema=CONTRACT_SCHEMA,
        task_id=TASK_ID,
        source_path=SOURCE_PATH,
        slot=DESTINATION_SLOT,
        name=DESTINATION_NAME,
        role=DESTINATION_ROLE,
        body_path=DESTINATION_BODY_PATH,
        body_active=True,
        root_transform=root_transform,
        collider_path=ENABLED_COLLIDER_PATH,
        collider_states=collider_states,
        calibrated_center_xy=calibrated_center_xy,  # type: ignore[arg-type]
        center_error_m=center_error_m,
        maximum_center_error_m=MAX_CENTER_ERROR_M,
        world_aabb=world_aabb,
        bottom_collision_lod=bottom_collision_lod,
        geometry_descriptor=geometry_descriptor,
    )


def canonical_destination_json_bytes(init_result: object) -> bytes:
    """Validate ``init_result`` and return canonical destination bytes."""

    return normalize_task29_destination_contract(
        init_result
    ).canonical_json_bytes()


def conservative_horizontal_zone(
    world_aabb: WorldAabb,
    object_horizontal_half_extent: object,
) -> ConservativeHorizontalZone:
    """Erode a composed destination AABB by an object's XY half-extent.

    This is deliberately only an AABB containment test.  Task 29 currently
    declares neither the bin's interior geometry nor a scoring polygon, so the
    returned rectangle must not be represented as either one.
    """

    if not isinstance(world_aabb, WorldAabb):
        raise Task29DestinationContractError(
            "world_aabb must be a normalized composed WorldAabb"
        )
    half_extent = _vector(
        object_horizontal_half_extent,
        "object_horizontal_half_extent",
        2,
        positive=True,
    )
    minimum_xy = tuple(
        world_aabb.minimum[axis] + half_extent[axis]
        for axis in range(2)
    )
    maximum_xy = tuple(
        world_aabb.maximum[axis] - half_extent[axis]
        for axis in range(2)
    )
    size_xy = tuple(
        maximum_xy[axis] - minimum_xy[axis] for axis in range(2)
    )
    if not all(width > 0.0 for width in size_xy):
        raise Task29DestinationContractError(
            "object horizontal half-extent collapses the conservative "
            "destination zone"
        )
    center_xy = tuple(
        (minimum_xy[axis] + maximum_xy[axis]) / 2.0
        for axis in range(2)
    )
    return ConservativeHorizontalZone(
        minimum_xy=minimum_xy,  # type: ignore[arg-type]
        maximum_xy=maximum_xy,  # type: ignore[arg-type]
        center_xy=center_xy,  # type: ignore[arg-type]
        size_xy=size_xy,  # type: ignore[arg-type]
        object_half_extent_xy=half_extent,  # type: ignore[arg-type]
    )


__all__ = [
    "BOTTOM_COLLISION_LOD_APPROXIMATION",
    "BOTTOM_COLLISION_LOD_EXTRUSION_DIRECTION",
    "BOTTOM_COLLISION_LOD_FACE_VERTEX_COUNT",
    "BOTTOM_COLLISION_LOD_PATH",
    "BOTTOM_COLLISION_LOD_PURPOSE",
    "BOTTOM_COLLISION_LOD_SCHEMA",
    "BOTTOM_COLLISION_LOD_THICKNESS_M",
    "BOTTOM_COLLISION_LOD_TOPOLOGY",
    "BOTTOM_COLLISION_LOD_VISIBILITY",
    "BOUNDS_KIND",
    "BottomCollisionLod",
    "CALIBRATED_CENTER_XY",
    "CONTRACT_SCHEMA",
    "ConservativeHorizontalZone",
    "DESTINATION_BODY_PATH",
    "ENABLED_COLLIDER_PATH",
    "GEOMETRY_DESCRIPTOR_SCHEMA",
    "GeometryDescriptor",
    "MAX_CENTER_ERROR_M",
    "RootTransform",
    "SOURCE_PATH",
    "TASK_ID",
    "Task29DestinationContract",
    "Task29DestinationContractError",
    "WorldAabb",
    "canonical_destination_json_bytes",
    "conservative_horizontal_zone",
    "normalize_task29_destination_contract",
]
