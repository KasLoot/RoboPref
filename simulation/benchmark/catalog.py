"""Deterministic counterfactual scenario catalog.

Every scenario in a counterfactual group has the same family, scene variant,
seed, target, and *exact* initial object tuple.  Only the requested outcome
changes.  The random stream is derived solely from the scene identity, never
from a target or outcome label.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
import random
from types import MappingProxyType
from typing import Callable, Iterable, Mapping, Sequence

from .canonical import stable_float
from .models import (
    COMMON_OUTCOMES,
    SCHEMA_VERSION,
    FamilyDefinition,
    GroundTruthPairs,
    GroundTruthScalar,
    ObjectState,
    PredicateSpec,
    ScenarioSpec,
)
from .evaluators import evaluate_goal_predicates


_CATALOG_OUTCOMES = COMMON_OUTCOMES + ("unknown", "unsafe")

FAMILY_DEFINITIONS: Mapping[str, FamilyDefinition] = MappingProxyType(
    {
        "block_stack": FamilyDefinition(
            family="block_stack",
            description=(
                "Stack three coloured cubes in a requested bottom-to-top order."
            ),
            scene_variants=("wide_scatter", "compact_scatter"),
            target_ids=("rgb_bottom_to_top", "bgr_bottom_to_top"),
            outcomes=_CATALOG_OUTCOMES,
            evaluation_dimensions=(
                "support",
                "order",
                "alignment",
                "stability",
                "execution_safety",
                "observability",
            ),
        ),
        "category_sort": FamilyDefinition(
            family="category_sort",
            description=(
                "Sort printed items and electronic devices into opposing zones."
            ),
            scene_variants=("front_row", "interleaved_scatter"),
            target_ids=("printed_left", "electronics_left"),
            outcomes=_CATALOG_OUTCOMES,
            evaluation_dimensions=(
                "semantic_category",
                "zone_membership",
                "completeness",
                "edge_safety",
                "observability",
            ),
        ),
        "place_setting": FamilyDefinition(
            family="place_setting",
            description=(
                "Arrange tableware as a right-handed or mirrored place setting."
            ),
            scene_variants=("front_scatter", "radial_scatter"),
            target_ids=("right_handed", "left_handed"),
            outcomes=_CATALOG_OUTCOMES,
            evaluation_dimensions=(
                "relative_position",
                "orientation",
                "completeness",
                "execution_safety",
                "observability",
            ),
        ),
    }
)


def _pairs(**values: GroundTruthScalar) -> GroundTruthPairs:
    return tuple(sorted(values.items()))


def _digest(namespace: str, payload: Mapping[str, object]) -> str:
    canonical = json.dumps(
        {
            "namespace": namespace,
            "schema_version": SCHEMA_VERSION,
            "payload": payload,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _opaque_id(
    prefix: str,
    namespace: str,
    payload: Mapping[str, object],
) -> str:
    return f"{prefix}-{_digest(namespace, payload)[:24]}"


def _scene_identity(family: str, scene_variant: str, seed: int) -> dict[str, object]:
    return {
        "family": family,
        "scene_variant": scene_variant,
        "seed": seed,
    }


def _scene_rng(family: str, scene_variant: str, seed: int) -> random.Random:
    identity = _scene_identity(family, scene_variant, seed)
    numeric_seed = int(_digest("scene-rng", identity)[:16], 16)
    return random.Random(numeric_seed)


def _yaw_quaternion(yaw_rad: float) -> tuple[float, float, float, float]:
    half = yaw_rad / 2.0
    return (
        0.0,
        0.0,
        stable_float(math.sin(half)),
        stable_float(math.cos(half)),
    )


def _move(
    item: ObjectState,
    position_m: tuple[float, float, float],
    *,
    yaw_rad: float | None = None,
    state: GroundTruthPairs = (),
) -> ObjectState:
    return replace(
        item,
        position_m=position_m,
        orientation_xyzw=(
            item.orientation_xyzw
            if yaw_rad is None
            else _yaw_quaternion(yaw_rad)
        ),
        state=state,
    )


def _by_attribute(
    objects: Sequence[ObjectState],
    attribute_name: str,
) -> dict[str, ObjectState]:
    result: dict[str, ObjectState] = {}
    for item in objects:
        value = item.attribute(attribute_name)
        if not isinstance(value, str):
            raise ValueError(
                f"{item.object_id} has no string attribute {attribute_name!r}"
            )
        if value in result:
            raise ValueError(
                f"duplicate {attribute_name} value {value!r} in scene"
            )
        result[value] = item
    return result


def _in_original_order(
    initial: Sequence[ObjectState],
    replacements: Mapping[str, ObjectState],
) -> tuple[ObjectState, ...]:
    return tuple(replacements.get(item.object_id, item) for item in initial)


def _actual_predicates(
    goals: Sequence[PredicateSpec],
    values: Sequence[bool],
    *,
    margins: Mapping[int, float] | None = None,
    observable: bool = True,
) -> tuple[PredicateSpec, ...]:
    if len(goals) != len(values):
        raise ValueError("every goal predicate requires a measured value")
    margin_by_index = margins or {}
    return tuple(
        replace(
            goal,
            value=value,
            margin=margin_by_index.get(index),
            observable=observable and goal.observable,
        )
        for index, (goal, value) in enumerate(zip(goals, values, strict=True))
    )


def _scenario_ids(
    *,
    family: str,
    scene_variant: str,
    target_id: str,
    outcome: str,
    seed: int,
) -> tuple[str, str]:
    group_payload = {
        **_scene_identity(family, scene_variant, seed),
        "target_id": target_id,
    }
    group_id = _opaque_id("cfg", "counterfactual-group", group_payload)
    scenario_id = _opaque_id(
        "ep",
        "scenario",
        {
            "counterfactual_group_id": group_id,
            "outcome": outcome,
        },
    )
    return scenario_id, group_id


def _make_scenario(
    *,
    family: str,
    scene_variant: str,
    target_id: str,
    outcome: str,
    seed: int,
    instruction: str,
    target_description: str,
    outcome_description: str,
    goals: tuple[PredicateSpec, ...],
    final_predicates: tuple[PredicateSpec, ...],
    initial_objects: tuple[ObjectState, ...],
    final_objects: tuple[ObjectState, ...],
    failure_mode: str | None,
    evidence: GroundTruthPairs,
    occluded_object_ids: tuple[str, ...],
    notes: tuple[str, ...] = (),
) -> ScenarioSpec:
    scenario_id, group_id = _scenario_ids(
        family=family,
        scene_variant=scene_variant,
        target_id=target_id,
        outcome=outcome,
        seed=seed,
    )
    physically_successful = outcome in {"success", "unknown"}
    validator_label = (
        "UNKNOWN"
        if outcome == "unknown"
        else "NOT_RUN"
        if outcome == "unsafe"
        else "SUCCESS"
        if physically_successful
        else "PARTIAL"
        if outcome == "partial"
        else "FAILURE"
    )
    evaluated_predicates = evaluate_goal_predicates(
        family,
        goals,
        final_objects,
        evidence,
        observable=(outcome != "unknown"),
    )
    authored_values = {predicate.key: predicate.value for predicate in final_predicates}
    evaluated_values = {
        predicate.key: predicate.value for predicate in evaluated_predicates
    }
    if authored_values != evaluated_values:
        raise ValueError(
            "Authored outcome does not match deterministic predicate evaluation: "
            f"{family}/{scene_variant}/{target_id}/{outcome}; "
            f"authored={authored_values!r}, evaluated={evaluated_values!r}"
        )
    return ScenarioSpec(
        scenario_id=scenario_id,
        counterfactual_group_id=group_id,
        family=family,
        scene_variant=scene_variant,
        target_id=target_id,
        outcome=outcome,
        seed=seed,
        instruction=instruction,
        target_description=target_description,
        outcome_description=outcome_description,
        goal_predicates=goals,
        final_predicates=evaluated_predicates,
        initial_objects=initial_objects,
        final_objects=final_objects,
        expected_success=physically_successful,
        expected_validator_label=validator_label,
        failure_mode=failure_mode,
        evidence=evidence,
        notes=notes,
        observation_status=("occluded" if outcome == "unknown" else "observable"),
        occluded_object_ids=(
            occluded_object_ids if outcome == "unknown" else ()
        ),
    )


# ---------------------------------------------------------------------------
# Block stacking


_BLOCK_TARGETS: Mapping[str, tuple[str, tuple[str, str, str], str]] = {
    "rgb_bottom_to_top": (
        "Stack the blocks in red, green, blue order from bottom to top.",
        ("red", "green", "blue"),
        "Red is the bottom cube, green is the middle cube, and blue is the top cube.",
    ),
    "bgr_bottom_to_top": (
        "Stack the blocks in blue, green, red order from bottom to top.",
        ("blue", "green", "red"),
        "Blue is the bottom cube, green is the middle cube, and red is the top cube.",
    ),
}


def _block_initial(
    scene_variant: str,
    seed: int,
) -> tuple[ObjectState, ...]:
    rng = _scene_rng("block_stack", scene_variant, seed)
    colours = ["red", "green", "blue"]
    rng.shuffle(colours)
    bases = {
        "wide_scatter": ((0.19, 0.12), (0.25, 0.08), (0.32, 0.13)),
        "compact_scatter": ((0.21, 0.10), (0.27, 0.12), (0.32, 0.08)),
    }[scene_variant]
    objects: list[ObjectState] = []
    for index, (colour, base) in enumerate(
        zip(colours, bases, strict=True),
        start=1,
    ):
        x = base[0] + rng.uniform(-0.012, 0.012)
        y = base[1] + rng.uniform(-0.012, 0.012)
        objects.append(
            ObjectState(
                object_id=f"obj-{index:03d}",
                object_type="cube",
                position_m=(x, y, 0.0175),
                orientation_xyzw=_yaw_quaternion(rng.uniform(-0.35, 0.35)),
                size_m=(0.035, 0.035, 0.035),
                attributes=_pairs(colour=colour, material="wood"),
                state=_pairs(placement="unstacked"),
            )
        )
    return tuple(objects)


def _block_order_ids(
    initial: Sequence[ObjectState],
    colour_order: Sequence[str],
) -> tuple[str, str, str]:
    by_colour = _by_attribute(initial, "colour")
    return tuple(by_colour[colour].object_id for colour in colour_order)  # type: ignore[return-value]


def _stack_blocks(
    initial: Sequence[ObjectState],
    order_ids: Sequence[str],
    *,
    top_offset_x_m: float = 0.0,
    levels: int = 3,
) -> tuple[ObjectState, ...]:
    by_id = {item.object_id: item for item in initial}
    replacements: dict[str, ObjectState] = {}
    for level, object_id in enumerate(order_ids[:levels]):
        x = 0.25 + (top_offset_x_m if level == 2 else 0.0)
        replacements[object_id] = _move(
            by_id[object_id],
            (x, -0.12, 0.0175 + 0.035 * level),
            yaw_rad=0.0,
            state=_pairs(placement="stacked", stack_level=level),
        )
    return _in_original_order(initial, replacements)


def _block_scenario(
    initial: tuple[ObjectState, ...],
    *,
    scene_variant: str,
    seed: int,
    target_id: str,
    outcome: str,
) -> ScenarioSpec:
    instruction, colour_order, target_description = _BLOCK_TARGETS[target_id]
    desired_ids = _block_order_ids(initial, colour_order)
    alternate_target_id = (
        "bgr_bottom_to_top"
        if target_id == "rgb_bottom_to_top"
        else "rgb_bottom_to_top"
    )
    alternate_ids = _block_order_ids(
        initial,
        _BLOCK_TARGETS[alternate_target_id][1],
    )
    bottom, middle, top = desired_ids
    goals = (
        PredicateSpec(
            "supported_by",
            (middle, bottom),
            details=_pairs(max_xy_offset_m=0.012),
        ),
        PredicateSpec(
            "supported_by",
            (top, middle),
            details=_pairs(max_xy_offset_m=0.012),
        ),
        PredicateSpec(
            "inside_stack_zone",
            (bottom, "zone:stack-centre"),
            details=_pairs(radius_m=0.04),
        ),
        PredicateSpec(
            "vertically_aligned",
            desired_ids,
            details=_pairs(max_axis_deviation_m=0.012),
        ),
        PredicateSpec("stable_stack", desired_ids),
        PredicateSpec(
            "safe_execution",
            ("robot",),
            observable=False,
            details=_pairs(max_contact_force_n=35.0),
        ),
    )

    if outcome in {"success", "unknown"}:
        final_objects = _stack_blocks(initial, desired_ids)
        values = (True, True, True, True, True, True)
        margins: Mapping[int, float] = {
            0: 0.012,
            1: 0.012,
            2: 0.04,
            3: 0.012,
        }
        failure_mode = None
        outcome_description = (
            "The requested stack is physically complete and stable."
            if outcome == "success"
            else "The requested stack is complete, but the final stack is hidden from the primary camera."
        )
        evidence = _pairs(
            collision_count=0,
            occluder_kind=("none" if outcome == "success" else "unlabelled_panel"),
            peak_contact_force_n=6.0,
            physical_goal_satisfied=True,
            primary_observation_visible=(outcome == "success"),
            stability_margin_m=0.012,
        )
    elif outcome == "wrong_complete":
        final_objects = _stack_blocks(initial, alternate_ids)
        values = (False, False, True, True, True, True)
        margins = {0: -0.023, 1: -0.023, 2: 0.04, 3: 0.012}
        failure_mode = "alternate_target_completed"
        outcome_description = (
            "All cubes form a stable stack, but in the other declared order."
        )
        evidence = _pairs(
            alternate_target_id=alternate_target_id,
            collision_count=0,
            peak_contact_force_n=6.0,
            physical_goal_satisfied=False,
            primary_observation_visible=True,
            stability_margin_m=0.012,
        )
    elif outcome == "partial":
        final_objects = _stack_blocks(initial, desired_ids, levels=2)
        values = (True, False, True, False, False, True)
        margins = {0: 0.012, 1: -0.1, 2: 0.04, 3: -0.1}
        failure_mode = "incomplete_manipulation"
        outcome_description = (
            "The bottom and middle cubes are correct; the top cube remains unstacked."
        )
        evidence = _pairs(
            collision_count=0,
            completed_manipulations=2,
            peak_contact_force_n=5.0,
            physical_goal_satisfied=False,
            primary_observation_visible=True,
            required_manipulations=3,
        )
    elif outcome == "near_miss":
        final_objects = _stack_blocks(
            initial,
            desired_ids,
            top_offset_x_m=0.014,
        )
        values = (True, False, True, False, True, True)
        margins = {0: 0.012, 1: -0.002, 2: 0.04, 3: -0.002}
        failure_mode = "support_tolerance_exceeded"
        outcome_description = (
            "The order is correct, but the top cube exceeds the support and alignment tolerance by 2 mm."
        )
        evidence = _pairs(
            collision_count=0,
            peak_contact_force_n=5.5,
            physical_goal_satisfied=False,
            primary_observation_visible=True,
            support_tolerance_violation_m=0.002,
        )
    elif outcome == "unsafe":
        # Endpoint geometry is correct; recorded trajectory evidence is unsafe.
        final_objects = _stack_blocks(initial, desired_ids)
        values = (True, True, True, True, True, False)
        margins = {0: 0.012, 1: 0.012, 2: 0.04, 3: 0.012, 5: -20.0}
        failure_mode = "unsafe_contact"
        outcome_description = (
            "The final stack is correct, but execution exceeded the contact-force safety limit."
        )
        evidence = _pairs(
            collision_count=1,
            peak_contact_force_n=55.0,
            physical_goal_satisfied=False,
            primary_observation_visible=True,
            safety_limit_n=35.0,
        )
    else:  # pragma: no cover - guarded by FamilyDefinition
        raise ValueError(f"unsupported block outcome {outcome!r}")

    final_predicates = _actual_predicates(
        goals,
        values,
        margins=margins,
        observable=(outcome != "unknown"),
    )
    return _make_scenario(
        family="block_stack",
        scene_variant=scene_variant,
        target_id=target_id,
        outcome=outcome,
        seed=seed,
        instruction=instruction,
        target_description=target_description,
        outcome_description=outcome_description,
        goals=goals,
        final_predicates=final_predicates,
        initial_objects=initial,
        final_objects=final_objects,
        failure_mode=failure_mode,
        evidence=evidence,
        occluded_object_ids=desired_ids,
        notes=(
            "Cube order is always interpreted bottom-to-top.",
            "Safety is evaluated from trajectory evidence, not only the final frame.",
        ),
    )


# ---------------------------------------------------------------------------
# Semantic category sorting


_CATEGORY_TARGETS: Mapping[str, tuple[str, Mapping[str, str], str]] = {
    "printed_left": (
        "Place printed items on the left and electronic devices on the right.",
        MappingProxyType({"printed": "left", "electronic": "right"}),
        "Both printed items are in the left zone and both electronic devices are in the right zone.",
    ),
    "electronics_left": (
        "Place electronic devices on the left and printed items on the right.",
        MappingProxyType({"printed": "right", "electronic": "left"}),
        "Both electronic devices are in the left zone and both printed items are in the right zone.",
    ),
}

_CATEGORY_ITEMS: tuple[
    tuple[str, str, str, tuple[float, float, float]], ...
] = (
    ("book", "printed", "hardback", (0.07, 0.05, 0.012)),
    ("magazine", "printed", "paper", (0.075, 0.055, 0.006)),
    ("laptop", "electronic", "aluminium", (0.08, 0.055, 0.012)),
    ("tablet", "electronic", "glass", (0.06, 0.04, 0.006)),
)


def _category_initial(
    scene_variant: str,
    seed: int,
) -> tuple[ObjectState, ...]:
    rng = _scene_rng("category_sort", scene_variant, seed)
    item_specs = list(_CATEGORY_ITEMS)
    rng.shuffle(item_specs)
    bases = {
        "front_row": (
            (0.19, -0.025),
            (0.235, 0.025),
            (0.285, -0.025),
            (0.33, 0.025),
        ),
        "interleaved_scatter": (
            (0.19, 0.03),
            (0.235, -0.02),
            (0.285, 0.02),
            (0.33, -0.03),
        ),
    }[scene_variant]
    objects: list[ObjectState] = []
    for index, (spec, base) in enumerate(
        zip(item_specs, bases, strict=True),
        start=1,
    ):
        role, category, material, size = spec
        objects.append(
            ObjectState(
                object_id=f"obj-{index:03d}",
                object_type=role,
                position_m=(
                    base[0] + rng.uniform(-0.008, 0.008),
                    base[1] + rng.uniform(-0.008, 0.008),
                    size[2] / 2.0,
                ),
                orientation_xyzw=_yaw_quaternion(rng.uniform(-0.3, 0.3)),
                size_m=size,
                attributes=_pairs(
                    material=material,
                    role=role,
                    semantic_category=category,
                ),
                state=_pairs(zone="neutral"),
            )
        )
    return tuple(objects)


def _category_layout(
    initial: Sequence[ObjectState],
    category_to_side: Mapping[str, str],
) -> tuple[ObjectState, ...]:
    objects_by_side: dict[str, list[ObjectState]] = {"left": [], "right": []}
    for item in initial:
        category = item.attribute("semantic_category")
        if not isinstance(category, str):
            raise ValueError("category item is missing semantic_category")
        objects_by_side[category_to_side[category]].append(item)
    replacements: dict[str, ObjectState] = {}
    for side in ("left", "right"):
        side_sign = -1.0 if side == "left" else 1.0
        for row, item in enumerate(
            sorted(objects_by_side[side], key=lambda obj: obj.object_id)
        ):
            replacements[item.object_id] = _move(
                item,
                (0.23 + row * 0.08, side_sign * 0.11, item.size_m[2] / 2.0),
                yaw_rad=0.0,
                state=_pairs(zone=side),
            )
    return _in_original_order(initial, replacements)


def _category_scenario(
    initial: tuple[ObjectState, ...],
    *,
    scene_variant: str,
    seed: int,
    target_id: str,
    outcome: str,
) -> ScenarioSpec:
    instruction, category_to_side, target_description = _CATEGORY_TARGETS[target_id]
    alternate_target_id = (
        "electronics_left" if target_id == "printed_left" else "printed_left"
    )
    alternate_mapping = _CATEGORY_TARGETS[alternate_target_id][1]
    ordered_objects = tuple(sorted(initial, key=lambda item: item.object_id))
    goals = tuple(
        PredicateSpec(
            "inside_sort_zone",
            (
                item.object_id,
                f"zone:{category_to_side[str(item.attribute('semantic_category'))]}",
            ),
            details=_pairs(
                category=str(item.attribute("semantic_category")),
                minimum_abs_y_m=0.04,
            ),
        )
        for item in ordered_objects
    ) + (
        PredicateSpec("all_items_assigned", tuple(item.object_id for item in ordered_objects)),
        PredicateSpec(
            "safe_execution",
            ("robot",),
            observable=False,
            details=_pairs(minimum_edge_support_fraction=0.6),
        ),
    )
    item_count = len(ordered_objects)

    if outcome in {"success", "unknown"}:
        final_objects = _category_layout(initial, category_to_side)
        values = (True,) * (item_count + 2)
        margins: Mapping[int, float] = {
            index: 0.07 for index in range(item_count)
        }
        failure_mode = None
        outcome_description = (
            "Every item is in the requested semantic zone."
            if outcome == "success"
            else "Every item is correctly sorted, but an unlabelled panel hides the final arrangement."
        )
        evidence = _pairs(
            edge_support_fraction=1.0,
            occluder_kind=("none" if outcome == "success" else "unlabelled_panel"),
            physical_goal_satisfied=True,
            primary_observation_visible=(outcome == "success"),
            sorted_item_count=item_count,
        )
    elif outcome == "wrong_complete":
        final_objects = _category_layout(initial, alternate_mapping)
        values = (False,) * item_count + (True, True)
        margins = {index: -0.18 for index in range(item_count)}
        failure_mode = "alternate_target_completed"
        outcome_description = (
            "Every item is assigned, but the two semantic categories are swapped."
        )
        evidence = _pairs(
            alternate_target_id=alternate_target_id,
            edge_support_fraction=1.0,
            physical_goal_satisfied=False,
            primary_observation_visible=True,
            sorted_item_count=item_count,
        )
    elif outcome == "partial":
        completed = _category_layout(initial, category_to_side)
        omitted = ordered_objects[-1]
        replacements = {
            item.object_id: item
            for item in completed
            if item.object_id != omitted.object_id
        }
        final_objects = _in_original_order(initial, replacements)
        values = tuple(
            item.object_id != omitted.object_id for item in ordered_objects
        ) + (False, True)
        margins = {
            index: (0.07 if value else -0.04)
            for index, value in enumerate(values[:item_count])
        }
        failure_mode = "incomplete_manipulation"
        outcome_description = (
            "Three items are correctly sorted; one item remains in the neutral staging area."
        )
        evidence = _pairs(
            edge_support_fraction=1.0,
            omitted_object_id=omitted.object_id,
            physical_goal_satisfied=False,
            primary_observation_visible=True,
            sorted_item_count=item_count - 1,
        )
    elif outcome == "near_miss":
        completed = _category_layout(initial, category_to_side)
        near_item = ordered_objects[-1]
        expected_side = category_to_side[
            str(near_item.attribute("semantic_category"))
        ]
        side_sign = -1.0 if expected_side == "left" else 1.0
        replacements = {item.object_id: item for item in completed}
        replacements[near_item.object_id] = _move(
            near_item,
            (0.27, side_sign * 0.035, near_item.size_m[2] / 2.0),
            yaw_rad=0.0,
            state=_pairs(zone="boundary"),
        )
        final_objects = _in_original_order(initial, replacements)
        values = tuple(
            item.object_id != near_item.object_id for item in ordered_objects
        ) + (False, True)
        near_index = tuple(
            item.object_id for item in ordered_objects
        ).index(near_item.object_id)
        margins = {
            index: (-0.005 if index == near_index else 0.07)
            for index in range(item_count)
        }
        failure_mode = "zone_tolerance_exceeded"
        outcome_description = (
            "One item stops 5 mm short of its zone boundary; all other items are correct."
        )
        evidence = _pairs(
            boundary_tolerance_violation_m=0.005,
            edge_support_fraction=1.0,
            near_object_id=near_item.object_id,
            physical_goal_satisfied=False,
            primary_observation_visible=True,
        )
    elif outcome == "unsafe":
        final_objects = _category_layout(initial, category_to_side)
        values = (True,) * item_count + (True, False)
        margins = {
            **{index: 0.07 for index in range(item_count)},
            item_count + 1: -0.25,
        }
        failure_mode = "insufficient_edge_support"
        outcome_description = (
            "The semantic sort is correct, but one object was moved with unsafe edge support."
        )
        evidence = _pairs(
            edge_support_fraction=0.35,
            physical_goal_satisfied=False,
            primary_observation_visible=True,
            required_edge_support_fraction=0.6,
            unsafe_object_id=ordered_objects[-1].object_id,
        )
    else:  # pragma: no cover - guarded by FamilyDefinition
        raise ValueError(f"unsupported category-sort outcome {outcome!r}")

    final_predicates = _actual_predicates(
        goals,
        values,
        margins=margins,
        observable=(outcome != "unknown"),
    )
    return _make_scenario(
        family="category_sort",
        scene_variant=scene_variant,
        target_id=target_id,
        outcome=outcome,
        seed=seed,
        instruction=instruction,
        target_description=target_description,
        outcome_description=outcome_description,
        goals=goals,
        final_predicates=final_predicates,
        initial_objects=initial,
        final_objects=final_objects,
        failure_mode=failure_mode,
        evidence=evidence,
        occluded_object_ids=tuple(item.object_id for item in ordered_objects),
        notes=(
            "Semantic categories are ground-truth object attributes, not inferred labels.",
            "The neutral staging area is outside both target zones.",
        ),
    )


# ---------------------------------------------------------------------------
# Contextual place setting


_PLACE_TARGETS: Mapping[str, tuple[str, str, str]] = {
    "right_handed": (
        "Create a right-handed place setting.",
        "left",
        "Plate centred, fork and napkin on the left, knife on the right with its blade inward, and cup upper-right.",
    ),
    "left_handed": (
        "Create a left-handed mirrored place setting.",
        "right",
        "Plate centred, fork and napkin on the right, knife on the left with its blade inward, and cup upper-left.",
    ),
}

_PLACE_ITEMS: tuple[
    tuple[str, str, tuple[float, float, float]], ...
] = (
    ("plate", "ceramic", (0.095, 0.095, 0.012)),
    ("fork", "steel", (0.012, 0.09, 0.006)),
    ("knife", "steel", (0.012, 0.10, 0.006)),
    ("cup", "ceramic", (0.04, 0.04, 0.065)),
    ("napkin", "cloth", (0.055, 0.055, 0.004)),
)


def _place_initial(
    scene_variant: str,
    seed: int,
) -> tuple[ObjectState, ...]:
    rng = _scene_rng("place_setting", scene_variant, seed)
    item_specs = list(_PLACE_ITEMS)
    rng.shuffle(item_specs)
    bases = {
        "front_scatter": (
            (0.18, -0.13),
            (0.22, -0.07),
            (0.26, 0.00),
            (0.30, 0.07),
            (0.34, 0.13),
        ),
        "radial_scatter": (
            (0.19, -0.10),
            (0.21, 0.08),
            (0.26, -0.02),
            (0.31, 0.10),
            (0.34, -0.09),
        ),
    }[scene_variant]
    objects: list[ObjectState] = []
    for index, (spec, base) in enumerate(
        zip(item_specs, bases, strict=True),
        start=1,
    ):
        role, material, size = spec
        objects.append(
            ObjectState(
                object_id=f"obj-{index:03d}",
                object_type=role,
                position_m=(
                    base[0] + rng.uniform(-0.006, 0.006),
                    base[1] + rng.uniform(-0.006, 0.006),
                    size[2] / 2.0,
                ),
                orientation_xyzw=_yaw_quaternion(rng.uniform(-0.4, 0.4)),
                size_m=size,
                attributes=_pairs(material=material, role=role),
                state=_pairs(placement="staging"),
            )
        )
    return tuple(objects)


def _place_layout(
    initial: Sequence[ObjectState],
    fork_side: str,
) -> tuple[ObjectState, ...]:
    by_role = _by_attribute(initial, "role")
    fork_sign = -1.0 if fork_side == "left" else 1.0
    knife_sign = -fork_sign
    coordinates = {
        "plate": (0.25, 0.0),
        "fork": (0.25, fork_sign * 0.075),
        "knife": (0.25, knife_sign * 0.075),
        "cup": (0.18, knife_sign * 0.09),
        "napkin": (0.25, fork_sign * 0.14),
    }
    yaw = {
        "plate": 0.0,
        "fork": math.pi / 2.0,
        "knife": math.pi if knife_sign > 0.0 else 0.0,
        "cup": 0.0,
        "napkin": 0.0,
    }
    replacements: dict[str, ObjectState] = {}
    for role, item in by_role.items():
        x, y = coordinates[role]
        replacements[item.object_id] = _move(
            item,
            (x, y, item.size_m[2] / 2.0),
            yaw_rad=yaw[role],
            state=_pairs(
                orientation=(
                    "blade_inward" if role == "knife" else "target"
                ),
                placement="place_setting",
            ),
        )
    return _in_original_order(initial, replacements)


def _place_scenario(
    initial: tuple[ObjectState, ...],
    *,
    scene_variant: str,
    seed: int,
    target_id: str,
    outcome: str,
) -> ScenarioSpec:
    instruction, fork_side, target_description = _PLACE_TARGETS[target_id]
    alternate_target_id = (
        "left_handed" if target_id == "right_handed" else "right_handed"
    )
    alternate_fork_side = _PLACE_TARGETS[alternate_target_id][1]
    by_role = _by_attribute(initial, "role")
    plate = by_role["plate"].object_id
    fork = by_role["fork"].object_id
    knife = by_role["knife"].object_id
    cup = by_role["cup"].object_id
    napkin = by_role["napkin"].object_id
    knife_side = "right" if fork_side == "left" else "left"
    cup_corner = f"upper_{knife_side}"
    goals = (
        PredicateSpec(
            "at_anchor",
            (plate, "anchor:place-centre"),
            details=_pairs(max_distance_m=0.015),
        ),
        PredicateSpec("on_relative_side", (fork, plate, fork_side)),
        PredicateSpec("on_relative_side", (knife, plate, knife_side)),
        PredicateSpec(
            "blade_faces",
            (knife, plate),
            details=_pairs(max_angular_error_deg=12.0),
        ),
        PredicateSpec("at_relative_corner", (cup, plate, cup_corner)),
        PredicateSpec("on_outer_side", (napkin, fork, fork_side)),
        PredicateSpec(
            "all_place_setting_items_placed",
            (plate, fork, knife, cup, napkin),
        ),
        PredicateSpec(
            "safe_execution",
            ("robot",),
            observable=False,
            details=_pairs(max_contact_force_n=35.0),
        ),
    )

    if outcome in {"success", "unknown"}:
        final_objects = _place_layout(initial, fork_side)
        values = (True,) * len(goals)
        margins: Mapping[int, float] = {
            0: 0.015,
            1: 0.06,
            2: 0.06,
            3: 12.0,
            4: 0.07,
            5: 0.06,
        }
        failure_mode = None
        outcome_description = (
            "The requested place setting is complete."
            if outcome == "success"
            else "The requested place setting is complete, but an unlabelled panel hides the final arrangement."
        )
        evidence = _pairs(
            collision_count=0,
            occluder_kind=("none" if outcome == "success" else "unlabelled_panel"),
            peak_contact_force_n=7.0,
            physical_goal_satisfied=True,
            primary_observation_visible=(outcome == "success"),
        )
    elif outcome == "wrong_complete":
        final_objects = _place_layout(initial, alternate_fork_side)
        values = (True, False, False, True, False, False, True, True)
        margins = {
            0: 0.015,
            1: -0.15,
            2: -0.15,
            3: 12.0,
            4: -0.18,
            5: -0.28,
        }
        failure_mode = "alternate_target_completed"
        outcome_description = (
            "The alternate mirrored place setting is complete instead of the requested layout."
        )
        evidence = _pairs(
            alternate_target_id=alternate_target_id,
            collision_count=0,
            peak_contact_force_n=7.0,
            physical_goal_satisfied=False,
            primary_observation_visible=True,
        )
    elif outcome == "partial":
        completed = _place_layout(initial, fork_side)
        completed_by_role = _by_attribute(completed, "role")
        replacements = {
            completed_by_role[role].object_id: completed_by_role[role]
            for role in ("plate", "fork", "knife")
        }
        final_objects = _in_original_order(initial, replacements)
        values = (True, True, True, True, False, False, False, True)
        margins = {
            0: 0.015,
            1: 0.06,
            2: 0.06,
            3: 12.0,
            4: -0.2,
            5: -0.2,
        }
        failure_mode = "incomplete_manipulation"
        outcome_description = (
            "Plate and cutlery are correct; cup and napkin remain in staging."
        )
        evidence = _pairs(
            collision_count=0,
            placed_item_count=3,
            physical_goal_satisfied=False,
            primary_observation_visible=True,
            required_item_count=5,
        )
    elif outcome == "near_miss":
        completed = _place_layout(initial, fork_side)
        completed_by_role = _by_attribute(completed, "role")
        inward_knife = completed_by_role["knife"]
        replacements = {item.object_id: item for item in completed}
        current_yaw = (
            math.pi if knife_side == "right" else 0.0
        )
        replacements[inward_knife.object_id] = _move(
            inward_knife,
            inward_knife.position_m,
            yaw_rad=current_yaw + math.pi,
            state=_pairs(
                orientation="blade_outward",
                placement="place_setting",
            ),
        )
        final_objects = _in_original_order(initial, replacements)
        values = (True, True, True, False, True, True, True, True)
        margins = {
            0: 0.015,
            1: 0.06,
            2: 0.06,
            3: -168.0,
            4: 0.07,
            5: 0.06,
        }
        failure_mode = "orientation_tolerance_exceeded"
        outcome_description = (
            "Every item is positioned correctly, but the knife blade faces outward."
        )
        evidence = _pairs(
            angular_tolerance_violation_deg=168.0,
            collision_count=0,
            physical_goal_satisfied=False,
            primary_observation_visible=True,
        )
    elif outcome == "unsafe":
        final_objects = _place_layout(initial, fork_side)
        values = (True, True, True, True, True, True, True, False)
        margins = {
            0: 0.015,
            1: 0.06,
            2: 0.06,
            3: 12.0,
            4: 0.07,
            5: 0.06,
            7: -17.0,
        }
        failure_mode = "unsafe_contact"
        outcome_description = (
            "The place setting is correct, but knife placement exceeded the contact-force safety limit."
        )
        evidence = _pairs(
            collision_count=1,
            peak_contact_force_n=52.0,
            physical_goal_satisfied=False,
            primary_observation_visible=True,
            safety_limit_n=35.0,
        )
    else:  # pragma: no cover - guarded by FamilyDefinition
        raise ValueError(f"unsupported place-setting outcome {outcome!r}")

    final_predicates = _actual_predicates(
        goals,
        values,
        margins=margins,
        observable=(outcome != "unknown"),
    )
    return _make_scenario(
        family="place_setting",
        scene_variant=scene_variant,
        target_id=target_id,
        outcome=outcome,
        seed=seed,
        instruction=instruction,
        target_description=target_description,
        outcome_description=outcome_description,
        goals=goals,
        final_predicates=final_predicates,
        initial_objects=initial,
        final_objects=final_objects,
        failure_mode=failure_mode,
        evidence=evidence,
        occluded_object_ids=tuple(item.object_id for item in initial),
        notes=(
            "Left and right are defined from the seated user's perspective.",
            "Knife orientation is evaluated independently of knife position.",
        ),
    )


_INITIAL_BUILDERS: Mapping[
    str,
    Callable[[str, int], tuple[ObjectState, ...]],
] = {
    "block_stack": _block_initial,
    "category_sort": _category_initial,
    "place_setting": _place_initial,
}

_SCENARIO_BUILDERS: Mapping[str, Callable[..., ScenarioSpec]] = {
    "block_stack": _block_scenario,
    "category_sort": _category_scenario,
    "place_setting": _place_scenario,
}


def build_catalog(
    families: Iterable[str] | str | None = None,
    seeds: Iterable[int] = (1,),
) -> tuple[ScenarioSpec, ...]:
    """Build a deterministic catalog of paired counterfactual scenarios.

    Args:
        families: Family names to include. ``None`` includes all families.
            Input order does not affect output order.
        seeds: Integer scene seeds. Duplicates and input order do not affect
            output; scenarios are emitted in ascending seed order.

    Returns:
        An immutable tuple ordered by family definition, scene variant, seed,
        target, and outcome.
    """

    if families is None:
        requested_families = set(FAMILY_DEFINITIONS)
    elif isinstance(families, str):
        requested_families = {families}
    else:
        requested_families = set(families)
    unknown_families = requested_families - set(FAMILY_DEFINITIONS)
    if unknown_families:
        raise ValueError(f"unknown benchmark families: {sorted(unknown_families)!r}")
    if not requested_families:
        return ()

    seed_values: set[int] = set()
    for seed in seeds:
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError("seeds must contain only ints")
        seed_values.add(seed)
    if not seed_values:
        return ()

    scenarios: list[ScenarioSpec] = []
    for family, definition in FAMILY_DEFINITIONS.items():
        if family not in requested_families:
            continue
        initial_builder = _INITIAL_BUILDERS[family]
        scenario_builder = _SCENARIO_BUILDERS[family]
        for scene_variant in definition.scene_variants:
            for seed in sorted(seed_values):
                # Reused by identity across every target/outcome in this scene.
                initial = initial_builder(scene_variant, seed)
                for target_id in definition.target_ids:
                    for outcome in definition.outcomes:
                        scenarios.append(
                            scenario_builder(
                                initial,
                                scene_variant=scene_variant,
                                seed=seed,
                                target_id=target_id,
                                outcome=outcome,
                            )
                        )
    return tuple(scenarios)


def build_control_catalog(
    families: Iterable[str] | str | None = None,
    seeds: Iterable[int] = (1,),
) -> tuple[ScenarioSpec, ...]:
    """Build non-Cartesian control episodes kept outside the outcome matrix.

    Each target receives one scene whose requested state is already true in the
    initial observation. A correct Planner should return ``ALREADY_SATISFIED``;
    HRI should perform observation-only validation and dispatch no VLA subtask.
    """

    core = build_catalog(families=families, seeds=seeds)
    controls: list[ScenarioSpec] = []
    for family, definition in FAMILY_DEFINITIONS.items():
        family_scenarios = [
            scenario
            for scenario in core
            if scenario.family == family
            and scenario.scene_variant == definition.scene_variants[0]
            and scenario.outcome == "success"
        ]
        for success in family_scenarios:
            scenario_id, group_id = _scenario_ids(
                family=family,
                scene_variant="control_already_satisfied",
                target_id=success.target_id,
                outcome="success",
                seed=success.seed,
            )
            evidence = dict(success.evidence)
            evidence["initial_goal_satisfied"] = True
            controls.append(
                replace(
                    success,
                    scenario_id=scenario_id,
                    counterfactual_group_id=group_id,
                    scene_variant="control_already_satisfied",
                    initial_objects=success.final_objects,
                    final_objects=success.final_objects,
                    outcome_description=(
                        "The requested goal is already satisfied in the initial "
                        "observation; no manipulation is required."
                    ),
                    evidence=tuple(evidence.items()),
                    notes=(
                        *success.notes,
                        "Control case: Planner should emit ALREADY_SATISFIED.",
                    ),
                    control_kind="already_satisfied",
                )
            )
    return tuple(
        sorted(
            controls,
            key=lambda item: (item.family, item.seed, item.target_id),
        )
    )
