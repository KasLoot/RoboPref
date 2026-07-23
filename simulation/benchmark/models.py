"""Immutable, dependency-light data contracts for simulation benchmarks.

The benchmark manifests deliberately separate the desired predicates from the
predicates measured in the final state.  This makes success labels auditable
without asking the same vision-language model that is under evaluation to
judge its own output.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Mapping, Self


SCHEMA_VERSION = "1.0"
COMMON_OUTCOMES = ("success", "wrong_complete", "partial", "near_miss")
OPTIONAL_OUTCOMES = ("unknown", "unsafe")
SUPPORTED_OUTCOMES = COMMON_OUTCOMES + OPTIONAL_OUTCOMES

GroundTruthScalar = str | int | float | bool | None
GroundTruthPairs = tuple[tuple[str, GroundTruthScalar], ...]

_SCENARIO_ID_RE = re.compile(r"^ep-[0-9a-f]{24}$")
_GROUP_ID_RE = re.compile(r"^(?:cfg|grp)-[0-9a-f]{24}$")


def _normalise_pairs(
    pairs: GroundTruthPairs | Mapping[str, GroundTruthScalar],
    *,
    field_name: str,
) -> GroundTruthPairs:
    items = pairs.items() if isinstance(pairs, Mapping) else pairs
    normalised: list[tuple[str, GroundTruthScalar]] = []
    seen: set[str] = set()
    for key, value in items:
        if not isinstance(key, str) or not key:
            raise ValueError(f"{field_name} keys must be non-empty strings")
        if key in seen:
            raise ValueError(f"{field_name} contains duplicate key {key!r}")
        if not isinstance(value, (str, int, float, bool, type(None))):
            raise TypeError(f"{field_name}[{key!r}] must be a JSON scalar")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{field_name}[{key!r}] must be finite")
        seen.add(key)
        normalised.append((key, value))
    return tuple(sorted(normalised))


def _float_tuple(
    values: tuple[float, ...],
    *,
    length: int,
    field_name: str,
) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != length:
        raise ValueError(f"{field_name} must contain {length} values")
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{field_name} must contain only finite values")
    return result


@dataclass(frozen=True, slots=True)
class PredicateSpec:
    """One desired or measured relation in a scenario.

    ``value`` is the desired truth value in ``goal_predicates`` and the
    measured truth value in ``final_predicates``.  ``margin`` is optional
    signed evaluator evidence: positive is inside tolerance, negative is
    outside tolerance.  It is especially useful for near-miss cases.
    """

    name: str
    arguments: tuple[str, ...] = ()
    value: bool = True
    observable: bool = True
    margin: float | None = None
    details: GroundTruthPairs = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("predicate name must be a non-empty string")
        arguments = tuple(self.arguments)
        if any(not isinstance(argument, str) or not argument for argument in arguments):
            raise ValueError("predicate arguments must be non-empty strings")
        if not isinstance(self.value, bool):
            raise TypeError("predicate value must be a bool")
        if not isinstance(self.observable, bool):
            raise TypeError("predicate observable must be a bool")
        if self.margin is not None and not math.isfinite(float(self.margin)):
            raise ValueError("predicate margin must be finite")
        object.__setattr__(self, "arguments", arguments)
        object.__setattr__(
            self,
            "margin",
            None if self.margin is None else float(self.margin),
        )
        object.__setattr__(
            self,
            "details",
            _normalise_pairs(self.details, field_name="predicate details"),
        )

    @property
    def key(self) -> tuple[str, tuple[str, ...]]:
        """Return the identity of a predicate independently of its truth value."""

        return self.name, self.arguments

    def __str__(self) -> str:
        """Return canonical identity and truth, excluding evaluator metadata."""

        arguments = ",".join(self.arguments)
        truth = "true" if self.value else "false"
        return f"{self.name.upper()}({arguments})={truth}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "arguments": list(self.arguments),
            "value": self.value,
            "observable": self.observable,
            "margin": self.margin,
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        return cls(
            name=str(data["name"]),
            arguments=tuple(str(item) for item in data.get("arguments", ())),
            value=data.get("value", True),
            observable=data.get("observable", True),
            margin=data.get("margin"),
            details=data.get("details", {}),
        )


@dataclass(frozen=True, slots=True)
class ObjectState:
    """Serializable state for one object at one point in an episode.

    Object identifiers are intentionally opaque (for example ``obj-001``).
    Semantic properties such as colour or role live in ``attributes`` so IDs
    remain stable when attributes or target instructions change.
    """

    object_id: str
    object_type: str
    position_m: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    size_m: tuple[float, float, float] = (0.01, 0.01, 0.01)
    attributes: GroundTruthPairs = ()
    state: GroundTruthPairs = ()

    def __post_init__(self) -> None:
        if not isinstance(self.object_id, str) or not self.object_id:
            raise ValueError("object_id must be a non-empty string")
        if not isinstance(self.object_type, str) or not self.object_type:
            raise ValueError("object_type must be a non-empty string")
        position = _float_tuple(
            self.position_m,
            length=3,
            field_name="position_m",
        )
        orientation = _float_tuple(
            self.orientation_xyzw,
            length=4,
            field_name="orientation_xyzw",
        )
        if not math.isclose(
            math.sqrt(sum(value * value for value in orientation)),
            1.0,
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            raise ValueError("orientation_xyzw must be a unit quaternion")
        size = _float_tuple(self.size_m, length=3, field_name="size_m")
        if any(value <= 0.0 for value in size):
            raise ValueError("size_m values must be positive")
        object.__setattr__(self, "position_m", position)
        object.__setattr__(self, "orientation_xyzw", orientation)
        object.__setattr__(self, "size_m", size)
        object.__setattr__(
            self,
            "attributes",
            _normalise_pairs(self.attributes, field_name="object attributes"),
        )
        object.__setattr__(
            self,
            "state",
            _normalise_pairs(self.state, field_name="object state"),
        )

    def attribute(self, name: str, default: GroundTruthScalar = None) -> GroundTruthScalar:
        return dict(self.attributes).get(name, default)

    def state_value(self, name: str, default: GroundTruthScalar = None) -> GroundTruthScalar:
        return dict(self.state).get(name, default)

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "object_type": self.object_type,
            "position_m": list(self.position_m),
            "orientation_xyzw": list(self.orientation_xyzw),
            "size_m": list(self.size_m),
            "attributes": dict(self.attributes),
            "state": dict(self.state),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        return cls(
            object_id=str(data["object_id"]),
            object_type=str(data["object_type"]),
            position_m=tuple(data["position_m"]),
            orientation_xyzw=tuple(
                data.get("orientation_xyzw", (0.0, 0.0, 0.0, 1.0))
            ),
            size_m=tuple(data.get("size_m", (0.01, 0.01, 0.01))),
            attributes=data.get("attributes", {}),
            state=data.get("state", {}),
        )


@dataclass(frozen=True, slots=True)
class ScenarioSpec:
    """Complete, immutable ground truth for one counterfactual episode."""

    scenario_id: str
    counterfactual_group_id: str
    family: str
    scene_variant: str
    target_id: str
    outcome: str
    seed: int
    instruction: str
    target_description: str
    outcome_description: str
    goal_predicates: tuple[PredicateSpec, ...]
    final_predicates: tuple[PredicateSpec, ...]
    initial_objects: tuple[ObjectState, ...]
    final_objects: tuple[ObjectState, ...]
    expected_success: bool
    expected_validator_label: str
    failure_mode: str | None
    evidence: GroundTruthPairs = ()
    notes: tuple[str, ...] = ()
    observation_status: str = "observable"
    occluded_object_ids: tuple[str, ...] = ()
    control_kind: str | None = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            not isinstance(self.scenario_id, str)
            or not _SCENARIO_ID_RE.fullmatch(self.scenario_id)
        ):
            raise ValueError("scenario_id must have the form ep-<24 lowercase hex>")
        if (
            not isinstance(self.counterfactual_group_id, str)
            or not _GROUP_ID_RE.fullmatch(self.counterfactual_group_id)
        ):
            raise ValueError(
                "counterfactual_group_id must have the form "
                "cfg-<24 lowercase hex> or grp-<24 lowercase hex>"
            )
        for field_name in (
            "family",
            "scene_variant",
            "target_id",
            "instruction",
            "target_description",
            "outcome_description",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field_name} must be a non-empty string")
        if self.outcome not in SUPPORTED_OUTCOMES:
            raise ValueError(f"unsupported outcome {self.outcome!r}")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError("seed must be an int")
        if not isinstance(self.expected_success, bool):
            raise TypeError("expected_success must be a bool")
        if self.expected_validator_label not in {
            "SUCCESS",
            "PARTIAL",
            "FAILURE",
            "UNKNOWN",
            "NOT_RUN",
        }:
            raise ValueError(
                "expected_validator_label must be SUCCESS, PARTIAL, FAILURE, "
                "UNKNOWN, or NOT_RUN"
            )
        if self.outcome == "unknown":
            if self.expected_validator_label != "UNKNOWN":
                raise ValueError("unknown outcomes must expect an UNKNOWN validation")
            if not self.expected_success:
                raise ValueError("unknown outcomes retain a physically successful state")
            if self.observation_status != "occluded":
                raise ValueError("unknown outcomes must declare an occluded observation")
        elif self.outcome == "unsafe":
            if self.expected_validator_label != "NOT_RUN":
                raise ValueError(
                    "unsafe execution must be stopped before Validator is called"
                )
            if self.expected_success:
                raise ValueError("unsafe outcomes cannot be successful")
            if self.observation_status != "observable":
                raise ValueError("unsafe outcomes must declare an observable endpoint")
        else:
            expected_label = (
                "SUCCESS"
                if self.expected_success
                else "PARTIAL"
                if self.outcome == "partial"
                else "FAILURE"
            )
            if self.expected_validator_label != expected_label:
                raise ValueError(
                    "expected_validator_label conflicts with expected_success"
                )
            if self.observation_status != "observable":
                raise ValueError(
                    "non-unknown outcomes must declare an observable observation"
                )
        if self.observation_status not in {"observable", "occluded"}:
            raise ValueError("unsupported observation_status")
        if self.expected_success and self.failure_mode is not None:
            raise ValueError("successful scenarios cannot have a failure_mode")
        if not self.expected_success and not self.failure_mode:
            raise ValueError("failed scenarios must identify a failure_mode")

        goals = tuple(self.goal_predicates)
        finals = tuple(self.final_predicates)
        initial_objects = tuple(self.initial_objects)
        final_objects = tuple(self.final_objects)
        notes = tuple(self.notes)
        occluded_object_ids = tuple(self.occluded_object_ids)
        if not goals:
            raise ValueError("goal_predicates cannot be empty")
        if not initial_objects:
            raise ValueError("initial_objects cannot be empty")
        if any(not isinstance(note, str) or not note for note in notes):
            raise ValueError("notes must contain non-empty strings")

        goal_map = {predicate.key: predicate for predicate in goals}
        final_map = {predicate.key: predicate for predicate in finals}
        if len(goal_map) != len(goals):
            raise ValueError("goal_predicates contains duplicate predicate keys")
        if len(final_map) != len(finals):
            raise ValueError("final_predicates contains duplicate predicate keys")
        missing = goal_map.keys() - final_map.keys()
        if missing:
            raise ValueError(f"final_predicates omits goal predicates: {sorted(missing)!r}")
        goals_satisfied = all(
            final_map[key].value == goal.value for key, goal in goal_map.items()
        )
        if goals_satisfied != self.expected_success:
            raise ValueError(
                "expected_success conflicts with measured goal predicates"
            )

        initial_ids = [item.object_id for item in initial_objects]
        final_ids = [item.object_id for item in final_objects]
        if len(set(initial_ids)) != len(initial_ids):
            raise ValueError("initial_objects contains duplicate object IDs")
        if len(set(final_ids)) != len(final_ids):
            raise ValueError("final_objects contains duplicate object IDs")
        if set(initial_ids) != set(final_ids):
            raise ValueError("initial and final object ID sets must match")
        unknown_occluded_ids = set(occluded_object_ids) - set(final_ids)
        if unknown_occluded_ids:
            raise ValueError(
                "occluded_object_ids contains unknown objects: "
                f"{sorted(unknown_occluded_ids)!r}"
            )
        if self.outcome == "unknown" and not occluded_object_ids:
            raise ValueError("unknown outcomes must identify occluded objects")
        if self.outcome != "unknown" and occluded_object_ids:
            raise ValueError("only unknown outcomes may identify occluded objects")
        if self.control_kind is not None and (
            not isinstance(self.control_kind, str) or not self.control_kind
        ):
            raise ValueError("control_kind must be null or a non-empty string")

        object.__setattr__(self, "goal_predicates", goals)
        object.__setattr__(self, "final_predicates", finals)
        object.__setattr__(self, "initial_objects", initial_objects)
        object.__setattr__(self, "final_objects", final_objects)
        object.__setattr__(
            self,
            "evidence",
            _normalise_pairs(self.evidence, field_name="scenario evidence"),
        )
        object.__setattr__(self, "notes", notes)
        object.__setattr__(self, "occluded_object_ids", occluded_object_ids)

    @property
    def goals_satisfied(self) -> bool:
        final_by_key = {predicate.key: predicate for predicate in self.final_predicates}
        return all(
            final_by_key[goal.key].value == goal.value
            for goal in self.goal_predicates
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable manifest with explicit ground truth."""

        return {
            "schema_version": self.schema_version,
            "scenario_id": self.scenario_id,
            "counterfactual_group_id": self.counterfactual_group_id,
            "family": self.family,
            "scene_variant": self.scene_variant,
            "target_id": self.target_id,
            "outcome": self.outcome,
            "seed": self.seed,
            "instruction": self.instruction,
            "target_description": self.target_description,
            "outcome_description": self.outcome_description,
            "goal_predicates": [
                predicate.to_dict() for predicate in self.goal_predicates
            ],
            "final_predicates": [
                predicate.to_dict() for predicate in self.final_predicates
            ],
            "initial_objects": [item.to_dict() for item in self.initial_objects],
            "final_objects": [item.to_dict() for item in self.final_objects],
            "expected_success": self.expected_success,
            "expected_validator_label": self.expected_validator_label,
            "failure_mode": self.failure_mode,
            "evidence": dict(self.evidence),
            "notes": list(self.notes),
            "observation_status": self.observation_status,
            "occluded_object_ids": list(self.occluded_object_ids),
            "control_kind": self.control_kind,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        return cls(
            scenario_id=str(data["scenario_id"]),
            counterfactual_group_id=str(data["counterfactual_group_id"]),
            family=str(data["family"]),
            scene_variant=str(data["scene_variant"]),
            target_id=str(data["target_id"]),
            outcome=str(data["outcome"]),
            seed=data["seed"],
            instruction=str(data["instruction"]),
            target_description=str(data["target_description"]),
            outcome_description=str(data["outcome_description"]),
            goal_predicates=tuple(
                PredicateSpec.from_dict(item)
                for item in data["goal_predicates"]
            ),
            final_predicates=tuple(
                PredicateSpec.from_dict(item)
                for item in data["final_predicates"]
            ),
            initial_objects=tuple(
                ObjectState.from_dict(item) for item in data["initial_objects"]
            ),
            final_objects=tuple(
                ObjectState.from_dict(item) for item in data["final_objects"]
            ),
            expected_success=data["expected_success"],
            expected_validator_label=str(data["expected_validator_label"]),
            failure_mode=data.get("failure_mode"),
            evidence=data.get("evidence", {}),
            notes=tuple(str(item) for item in data.get("notes", ())),
            observation_status=str(data.get("observation_status", "observable")),
            occluded_object_ids=tuple(
                str(item) for item in data.get("occluded_object_ids", ())
            ),
            control_kind=(
                str(data["control_kind"])
                if data.get("control_kind") is not None
                else None
            ),
            schema_version=str(data.get("schema_version", SCHEMA_VERSION)),
        )


@dataclass(frozen=True, slots=True)
class FamilyDefinition:
    """Static coverage contract for a task family."""

    family: str
    description: str
    scene_variants: tuple[str, ...]
    target_ids: tuple[str, ...]
    outcomes: tuple[str, ...] = COMMON_OUTCOMES
    evaluation_dimensions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in ("family", "description"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field_name} must be a non-empty string")
        for field_name in (
            "scene_variants",
            "target_ids",
            "outcomes",
            "evaluation_dimensions",
        ):
            values = tuple(getattr(self, field_name))
            if not values or any(
                not isinstance(value, str) or not value for value in values
            ):
                raise ValueError(f"{field_name} must contain non-empty strings")
            if len(set(values)) != len(values):
                raise ValueError(f"{field_name} cannot contain duplicates")
            object.__setattr__(self, field_name, values)
        unknown = set(self.outcomes) - set(SUPPORTED_OUTCOMES)
        if unknown:
            raise ValueError(f"unsupported family outcomes: {sorted(unknown)!r}")
        missing_common = set(COMMON_OUTCOMES) - set(self.outcomes)
        if missing_common:
            raise ValueError(
                f"family omits common outcomes: {sorted(missing_common)!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "description": self.description,
            "scene_variants": list(self.scene_variants),
            "target_ids": list(self.target_ids),
            "outcomes": list(self.outcomes),
            "evaluation_dimensions": list(self.evaluation_dimensions),
        }

    @property
    def targets(self) -> tuple[str, ...]:
        """Public alias used by catalog consumers."""

        return self.target_ids
