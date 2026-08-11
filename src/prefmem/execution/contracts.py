"""Strict contracts for the lower-level execution boundary.

Gemma is allowed to choose only a small symbolic manipulation program.  It
never supplies trusted identifiers, actuator values, or absolute world-frame
coordinates.  Host code validates this module's contracts before perception or
motion is allowed to start.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Mapping, Sequence


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _reject_unknown(payload: Mapping[str, Any], allowed: set[str], name: str) -> None:
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"{name} contains unknown fields: {', '.join(sorted(unknown))}")


class AnchorKind(str, Enum):
    TOP_CENTER = "top_center"
    SURFACE_CENTER = "surface_center"
    CENTER = "center"


class SpatialRelation(str, Enum):
    ON_TOP = "on_top"


class WaypointKind(str, Enum):
    APPROACH_SOURCE = "approach_source"
    GRASP_SOURCE = "grasp_source"
    LIFT = "lift"
    APPROACH_TARGET = "approach_target"
    PLACE_TARGET = "place_target"
    RETREAT = "retreat"


class GripperCommand(str, Enum):
    OPEN = "open"
    CLOSE = "close"
    HOLD = "hold"


REQUIRED_PICK_PLACE_WAYPOINTS = (
    WaypointKind.APPROACH_SOURCE,
    WaypointKind.GRASP_SOURCE,
    WaypointKind.LIFT,
    WaypointKind.APPROACH_TARGET,
    WaypointKind.PLACE_TARGET,
    WaypointKind.RETREAT,
)


@dataclass(frozen=True, slots=True)
class ObjectReference:
    query: str
    anchor: AnchorKind

    def __post_init__(self) -> None:
        object.__setattr__(self, "query", _text(self.query, "query"))
        try:
            anchor = AnchorKind(self.anchor)
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid object anchor: {self.anchor!r}") from error
        object.__setattr__(self, "anchor", anchor)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], *, name: str) -> ObjectReference:
        payload = _mapping(payload, name)
        _reject_unknown(payload, {"query", "anchor"}, name)
        return cls(query=payload.get("query"), anchor=payload.get("anchor"))

    def to_dict(self) -> dict[str, str]:
        return {"query": self.query, "anchor": self.anchor.value}


@dataclass(frozen=True, slots=True)
class WaypointSpec:
    kind: WaypointKind
    clearance_m: float = 0.0
    gripper: GripperCommand = GripperCommand.HOLD

    def __post_init__(self) -> None:
        try:
            kind = WaypointKind(self.kind)
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid waypoint kind: {self.kind!r}") from error
        try:
            gripper = GripperCommand(self.gripper)
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid gripper command: {self.gripper!r}") from error
        clearance = _finite_float(self.clearance_m, "clearance_m")
        if not 0.0 <= clearance <= 0.30:
            raise ValueError("clearance_m must be between 0 and 0.30")
        clearance_stage = kind in {
            WaypointKind.APPROACH_SOURCE,
            WaypointKind.LIFT,
            WaypointKind.APPROACH_TARGET,
            WaypointKind.RETREAT,
        }
        if clearance_stage and not 0.05 <= clearance <= 0.20:
            raise ValueError(
                f"{kind.value} clearance_m must be between 0.05 and 0.20"
            )
        if not clearance_stage and clearance != 0.0:
            raise ValueError(f"{kind.value} clearance_m must be zero")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "gripper", gripper)
        object.__setattr__(self, "clearance_m", clearance)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> WaypointSpec:
        payload = _mapping(payload, "waypoint")
        _reject_unknown(payload, {"kind", "clearance_m", "gripper"}, "waypoint")
        return cls(
            kind=payload.get("kind"),
            clearance_m=payload.get("clearance_m", 0.0),
            gripper=payload.get("gripper", "hold"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "clearance_m": self.clearance_m,
            "gripper": self.gripper.value,
        }


@dataclass(frozen=True, slots=True)
class ManipulationProgram:
    """A model-authored symbolic program accepted by the deterministic host."""

    source: ObjectReference
    target: ObjectReference
    relation: SpatialRelation
    waypoints: tuple[WaypointSpec, ...]
    schema_version: int = 1
    skill: str = "pick_place"
    orientation: str = "tool_down"

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("schema_version must be 1")
        if self.skill != "pick_place":
            raise ValueError("the initial executor supports only pick_place")
        if self.orientation != "tool_down":
            raise ValueError("the initial executor supports only tool_down orientation")
        if not isinstance(self.source, ObjectReference):
            raise ValueError("source must be an ObjectReference")
        if not isinstance(self.target, ObjectReference):
            raise ValueError("target must be an ObjectReference")
        if self.source.anchor is not AnchorKind.TOP_CENTER:
            raise ValueError("pick_place source anchor must be top_center")
        if self.target.anchor not in {
            AnchorKind.TOP_CENTER,
            AnchorKind.SURFACE_CENTER,
        }:
            raise ValueError(
                "pick_place target anchor must be top_center or surface_center"
            )
        if self.source.query.casefold() == self.target.query.casefold():
            raise ValueError("source and target queries must be different")
        try:
            relation = SpatialRelation(self.relation)
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid spatial relation: {self.relation!r}") from error
        waypoint_tuple = tuple(self.waypoints)
        if not all(isinstance(item, WaypointSpec) for item in waypoint_tuple):
            raise ValueError("waypoints must contain WaypointSpec objects")
        actual = tuple(item.kind for item in waypoint_tuple)
        if actual != REQUIRED_PICK_PLACE_WAYPOINTS:
            raise ValueError(
                "pick_place waypoints must use the exact safe six-stage sequence"
            )
        expected_grippers = (
            GripperCommand.OPEN,
            GripperCommand.CLOSE,
            GripperCommand.HOLD,
            GripperCommand.HOLD,
            GripperCommand.OPEN,
            GripperCommand.HOLD,
        )
        if tuple(item.gripper for item in waypoint_tuple) != expected_grippers:
            raise ValueError("pick_place waypoint gripper commands are invalid")
        object.__setattr__(self, "relation", relation)
        object.__setattr__(self, "waypoints", waypoint_tuple)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ManipulationProgram:
        payload = _mapping(payload, "manipulation program")
        _reject_unknown(
            payload,
            {
                "schema_version",
                "skill",
                "source",
                "target",
                "relation",
                "orientation",
                "waypoints",
            },
            "manipulation program",
        )
        raw_waypoints = payload.get("waypoints", ())
        if isinstance(raw_waypoints, (str, bytes)) or not isinstance(
            raw_waypoints, Sequence
        ):
            raise ValueError("waypoints must be an array")
        return cls(
            schema_version=payload.get("schema_version"),
            skill=payload.get("skill"),
            source=ObjectReference.from_dict(payload.get("source"), name="source"),
            target=ObjectReference.from_dict(payload.get("target"), name="target"),
            relation=payload.get("relation"),
            orientation=payload.get("orientation"),
            waypoints=tuple(WaypointSpec.from_dict(item) for item in raw_waypoints),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "skill": self.skill,
            "source": self.source.to_dict(),
            "target": self.target.to_dict(),
            "relation": self.relation.value,
            "orientation": self.orientation,
            "waypoints": [item.to_dict() for item in self.waypoints],
        }


class ExecutionState(str, Enum):
    PREPARING = "PREPARING"
    RUNNING = "RUNNING"
    SETTLED = "SETTLED"
    FAULT = "FAULT"
    CANCELLED = "CANCELLED"


class ExecutionErrorKind(str, Enum):
    COMPILER = "COMPILER_ERROR"
    FRAME = "FRAME_ERROR"
    GROUNDING = "GROUNDING_ERROR"
    CONTROL = "CONTROL_ERROR"
    UNKNOWN = "UNKNOWN_ERROR"


@dataclass(frozen=True, slots=True)
class ExecutionEvent:
    publication_id: str
    state: ExecutionState
    message: str
    observed_at: float
    error_kind: ExecutionErrorKind | None = None
    exception_type: str | None = None
    traceback_text: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "publication_id", _text(self.publication_id, "publication_id")
        )
        try:
            state = ExecutionState(self.state)
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid execution state: {self.state!r}") from error
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "message", _text(self.message, "message"))
        object.__setattr__(
            self, "observed_at", _finite_float(self.observed_at, "observed_at")
        )
        if self.error_kind is not None:
            try:
                error_kind = ExecutionErrorKind(self.error_kind)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid execution error kind: {self.error_kind!r}"
                ) from error
            if state is not ExecutionState.FAULT:
                raise ValueError("error_kind is permitted only for FAULT events")
            object.__setattr__(self, "error_kind", error_kind)
        for name in ("exception_type", "traceback_text"):
            value = getattr(self, name)
            if value is not None:
                value = str(value).strip()
                object.__setattr__(self, name, value or None)


@dataclass(frozen=True, slots=True)
class SceneChangeEvent:
    publication_id: str
    description: str
    observed_at: float
    frame_sequence: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "publication_id", _text(self.publication_id, "publication_id")
        )
        object.__setattr__(
            self, "description", _text(self.description, "description")
        )
        object.__setattr__(
            self, "observed_at", _finite_float(self.observed_at, "observed_at")
        )
        if self.frame_sequence is not None and (
            isinstance(self.frame_sequence, bool)
            or not isinstance(self.frame_sequence, int)
            or self.frame_sequence < 0
        ):
            raise ValueError("frame_sequence must be a non-negative integer or None")
