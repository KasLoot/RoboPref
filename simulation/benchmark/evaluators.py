from __future__ import annotations

import math
from dataclasses import replace
from typing import Iterable, Mapping, Sequence

from .models import GroundTruthPairs, ObjectState, PredicateSpec


def _pairs(value: GroundTruthPairs) -> dict[str, str | int | float | bool | None]:
    return dict(value)


def _distance_xy(first: ObjectState, second: ObjectState) -> float:
    return math.dist(first.position_m[:2], second.position_m[:2])


def _yaw_radians(item: ObjectState) -> float:
    x, y, z, w = item.orientation_xyzw
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _result(
    goal: PredicateSpec,
    *,
    value: bool,
    margin: float | None,
    observable: bool,
    evidence_source: str,
) -> PredicateSpec:
    details = dict(goal.details)
    details["evidence_source"] = evidence_source
    return replace(
        goal,
        value=value,
        margin=margin,
        observable=observable and goal.observable,
        details=tuple(details.items()),
    )


def _safe_execution(
    goal: PredicateSpec,
    evidence: Mapping[str, object],
    *,
    observable: bool,
) -> PredicateSpec:
    details = _pairs(goal.details)
    if "minimum_edge_support_fraction" in details:
        minimum = float(details["minimum_edge_support_fraction"])
        actual = float(evidence.get("edge_support_fraction", 1.0))
        margin = actual - minimum
    else:
        limit = float(
            evidence.get(
                "safety_limit_n",
                details.get("max_contact_force_n", 35.0),
            )
        )
        actual = float(evidence.get("peak_contact_force_n", 0.0))
        margin = limit - actual
    return _result(
        goal,
        value=margin >= 0.0,
        margin=margin,
        observable=observable,
        evidence_source="scripted_execution_fixture",
    )


def _evaluate_block(
    goal: PredicateSpec,
    objects: Mapping[str, ObjectState],
    evidence: Mapping[str, object],
    *,
    observable: bool,
) -> PredicateSpec:
    name = goal.name.lower()
    details = _pairs(goal.details)
    if name == "safe_execution":
        return _safe_execution(goal, evidence, observable=observable)
    if name == "supported_by":
        upper, lower = (objects[object_id] for object_id in goal.arguments)
        tolerance_xy = float(details.get("max_xy_offset_m", 0.012))
        xy_error = _distance_xy(upper, lower)
        expected_z = (upper.size_m[2] + lower.size_m[2]) / 2.0
        z_error = abs((upper.position_m[2] - lower.position_m[2]) - expected_z)
        margin = min(tolerance_xy - xy_error, 0.008 - z_error)
        return _result(
            goal,
            value=margin >= 0.0,
            margin=margin,
            observable=observable,
            evidence_source="derived_object_geometry",
        )
    if name == "inside_stack_zone":
        item = objects[goal.arguments[0]]
        radius = float(details.get("radius_m", 0.04))
        distance = math.dist(item.position_m[:2], (0.25, -0.12))
        return _result(
            goal,
            value=distance <= radius,
            margin=radius - distance,
            observable=observable,
            evidence_source="derived_object_geometry",
        )
    selected = [objects[object_id] for object_id in goal.arguments]
    if name == "vertically_aligned":
        tolerance = float(details.get("max_axis_deviation_m", 0.012))
        bottom = min(selected, key=lambda item: item.position_m[2])
        error = max(_distance_xy(item, bottom) for item in selected)
        return _result(
            goal,
            value=error <= tolerance,
            margin=tolerance - error,
            observable=observable,
            evidence_source="derived_object_geometry",
        )
    if name == "stable_stack":
        ordered = sorted(selected, key=lambda item: item.position_m[2])
        if len(ordered) < 2:
            stable = False
            margin = -1.0
        else:
            margins: list[float] = []
            for lower, upper in zip(ordered, ordered[1:]):
                support_radius = min(lower.size_m[0], lower.size_m[1]) / 2.0
                xy_margin = support_radius - _distance_xy(upper, lower)
                expected_z = (upper.size_m[2] + lower.size_m[2]) / 2.0
                z_margin = 0.008 - abs(
                    (upper.position_m[2] - lower.position_m[2]) - expected_z
                )
                margins.extend((xy_margin, z_margin))
            margin = min(margins)
            stable = margin >= 0.0
        return _result(
            goal,
            value=stable,
            margin=margin,
            observable=observable,
            evidence_source="derived_quasistatic_geometry",
        )
    raise ValueError(f"Unsupported block predicate: {goal.name}")


def _evaluate_category(
    goal: PredicateSpec,
    objects: Mapping[str, ObjectState],
    evidence: Mapping[str, object],
    *,
    observable: bool,
) -> PredicateSpec:
    name = goal.name.lower()
    if name == "safe_execution":
        return _safe_execution(goal, evidence, observable=observable)
    if name == "inside_sort_zone":
        item = objects[goal.arguments[0]]
        side = goal.arguments[1].split(":", 1)[-1]
        minimum = float(_pairs(goal.details).get("minimum_abs_y_m", 0.04))
        signed_y = -item.position_m[1] if side == "left" else item.position_m[1]
        placed = item.state_value("zone") == side
        margin = signed_y - minimum
        return _result(
            goal,
            value=placed and margin >= 0.0,
            margin=margin,
            observable=observable,
            evidence_source="derived_object_geometry_and_state",
        )
    if name == "all_items_assigned":
        assigned = [
            objects[object_id].state_value("zone") in {"left", "right"}
            for object_id in goal.arguments
        ]
        count = sum(assigned)
        return _result(
            goal,
            value=all(assigned),
            margin=float(count - len(assigned)),
            observable=observable,
            evidence_source="derived_object_state",
        )
    raise ValueError(f"Unsupported category predicate: {goal.name}")


def _evaluate_place(
    goal: PredicateSpec,
    objects: Mapping[str, ObjectState],
    evidence: Mapping[str, object],
    *,
    observable: bool,
) -> PredicateSpec:
    name = goal.name.lower()
    if name == "safe_execution":
        return _safe_execution(goal, evidence, observable=observable)
    if name == "at_anchor":
        item = objects[goal.arguments[0]]
        tolerance = float(_pairs(goal.details).get("max_distance_m", 0.015))
        distance = math.dist(item.position_m[:2], (0.25, 0.0))
        return _result(
            goal,
            value=distance <= tolerance,
            margin=tolerance - distance,
            observable=observable,
            evidence_source="derived_object_geometry",
        )
    if name == "on_relative_side":
        item, reference = (objects[object_id] for object_id in goal.arguments[:2])
        side = goal.arguments[2]
        signed_delta = (
            reference.position_m[1] - item.position_m[1]
            if side == "left"
            else item.position_m[1] - reference.position_m[1]
        )
        margin = signed_delta - 0.04
        placed = item.state_value("placement") == "place_setting"
        return _result(
            goal,
            value=placed and margin >= 0.0,
            margin=margin,
            observable=observable,
            evidence_source="derived_object_geometry_and_state",
        )
    if name == "blade_faces":
        knife, plate = (
            objects[object_id] for object_id in goal.arguments[:2]
        )
        max_error = float(_pairs(goal.details).get("max_angular_error_deg", 12.0))
        yaw = _yaw_radians(knife)
        # The renderer defines the blade as local +Y. Rotate that direction into
        # world XY and compare it with the vector from knife to plate.
        blade_direction = (-math.sin(yaw), math.cos(yaw))
        toward_plate = (
            plate.position_m[0] - knife.position_m[0],
            plate.position_m[1] - knife.position_m[1],
        )
        length = math.hypot(*toward_plate)
        if length <= 1e-12:
            angular_error = 180.0
        else:
            dot = (
                blade_direction[0] * toward_plate[0]
                + blade_direction[1] * toward_plate[1]
            ) / length
            angular_error = math.degrees(
                math.acos(max(-1.0, min(1.0, dot)))
            )
        inward = angular_error <= max_error
        return _result(
            goal,
            value=inward,
            margin=max_error - angular_error,
            observable=observable,
            evidence_source="derived_object_geometry_and_orientation",
        )
    if name == "at_relative_corner":
        item, reference = (objects[object_id] for object_id in goal.arguments[:2])
        corner = goal.arguments[2]
        side = corner.rsplit("_", 1)[-1]
        lateral = (
            reference.position_m[1] - item.position_m[1]
            if side == "left"
            else item.position_m[1] - reference.position_m[1]
        )
        upper = reference.position_m[0] - item.position_m[0]
        margin = min(lateral - 0.04, upper - 0.03)
        placed = item.state_value("placement") == "place_setting"
        return _result(
            goal,
            value=placed and margin >= 0.0,
            margin=margin,
            observable=observable,
            evidence_source="derived_object_geometry_and_state",
        )
    if name == "on_outer_side":
        napkin, fork = (objects[object_id] for object_id in goal.arguments[:2])
        side = goal.arguments[2]
        delta = (
            fork.position_m[1] - napkin.position_m[1]
            if side == "left"
            else napkin.position_m[1] - fork.position_m[1]
        )
        margin = delta - 0.025
        placed = napkin.state_value("placement") == "place_setting"
        return _result(
            goal,
            value=placed and margin >= 0.0,
            margin=margin,
            observable=observable,
            evidence_source="derived_object_geometry_and_state",
        )
    if name == "all_place_setting_items_placed":
        placed = [
            objects[object_id].state_value("placement") == "place_setting"
            for object_id in goal.arguments
        ]
        return _result(
            goal,
            value=all(placed),
            margin=float(sum(placed) - len(placed)),
            observable=observable,
            evidence_source="derived_object_state",
        )
    raise ValueError(f"Unsupported place-setting predicate: {goal.name}")


def evaluate_goal_predicates(
    family: str,
    goals: Sequence[PredicateSpec],
    final_objects: Iterable[ObjectState],
    evidence_pairs: GroundTruthPairs,
    *,
    observable: bool,
) -> tuple[PredicateSpec, ...]:
    """Independently evaluate authored goals against terminal state.

    Visual/spatial truth is derived from object state. Safety truth uses explicitly
    marked scripted execution fixtures because the endpoint generator is not a
    physics rollout.
    """

    objects = {item.object_id: item for item in final_objects}
    evidence: Mapping[str, object] = dict(evidence_pairs)
    evaluators = {
        "block_stack": _evaluate_block,
        "category_sort": _evaluate_category,
        "place_setting": _evaluate_place,
    }
    try:
        evaluator = evaluators[family]
    except KeyError as error:
        raise ValueError(f"Unsupported benchmark family: {family}") from error
    return tuple(
        evaluator(
            goal,
            objects,
            evidence,
            observable=observable,
        )
        for goal in goals
    )
