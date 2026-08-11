"""Deterministic Layer-B component catalogue and strict scorers."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


COMPONENT_TARGETS = {
    "hri_planner": 500,
    "memory": 300,
    "monitor_validator": 1000,
    "grounding_execution": 500,
}
SPLIT_COUNTS = {
    "hri_planner": {"dev": 100, "pilot": 100, "locked": 300},
    "memory": {"dev": 60, "pilot": 60, "locked": 180},
    "monitor_validator": {"dev": 200, "pilot": 200, "locked": 600},
    "grounding_execution": {"dev": 100, "pilot": 100, "locked": 300},
}
STRATA = {
    "hri_planner": (
        "clear_goal",
        "ambiguous_goal",
        "duplicate_reference",
        "constraint_conflict",
        "supported_primitive",
        "unsupported_primitive",
        "already_satisfied",
        "missing_object",
        "exact_revision",
        "open_closed_distractors",
    ),
    "memory": (
        "remember",
        "retrieve",
        "update",
        "forget",
        "one_run_override",
        "old_new_conflict",
        "irrelevant_neighbour",
        "paraphrase",
        "delayed_retrieval",
        "no_consent",
        "confirmation_only",
        "long_store",
        "transfer",
    ),
    "monitor_validator": (
        "met",
        "not_met",
        "unknown",
        "partial_occlusion",
        "complete_occlusion",
        "blur_lighting",
        "transient_contradiction",
        "persistent_contradiction",
        "moved_source_target",
        "fallen_stack",
        "protected_displacement",
        "visible_emergency_proxy",
    ),
    "grounding_execution": (
        "colour_reference",
        "spatial_reference",
        "duplicate_objects",
        "flat_target",
        "three_d_target",
        "similar_source_target",
        "workspace_edge",
        "unreachable",
        "corrupted_depth",
        "sparse_depth",
        "post_grounding_motion",
        "post_cancel_restart",
        "stack_height_stress",
    ),
}


class ComponentSchemaError(ValueError):
    """A component item or prediction violates its frozen schema."""


def _seed(component: str, split: str, index: int) -> int:
    digest = hashlib.sha256(
        f"component-v1:{component}:{split}:{index}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big")


@dataclass(frozen=True, slots=True)
class ComponentItem:
    item_id: str
    component: str
    split: str
    stratum: str
    template_id: str
    seed: int
    inputs: Mapping[str, Any]
    expected: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _expected(component: str, stratum: str, index: int) -> dict[str, Any]:
    if component == "hri_planner":
        clarification = stratum in {
            "ambiguous_goal",
            "duplicate_reference",
            "constraint_conflict",
        }
        supported = stratum != "unsupported_primitive"
        return {
            "clarification_required": clarification,
            "supported": supported,
            "decision": (
                "BLOCKED"
                if not supported
                else ("CLARIFY" if clarification else "READY")
            ),
            "schema_valid": True,
        }
    if component == "memory":
        operation = stratum if stratum in {"remember", "retrieve", "update", "forget"} else "retrieve"
        authorized = stratum not in {"no_consent", "confirmation_only"}
        should_write = operation in {"remember", "update", "forget"} and authorized
        return {
            "operation": operation,
            "authorized": authorized,
            "should_write": should_write,
            "maximum_writes": 1 if should_write else 0,
        }
    if component == "monitor_validator":
        if stratum in {"unknown", "complete_occlusion", "blur_lighting"}:
            label = "UNKNOWN"
        elif stratum in {
            "not_met",
            "persistent_contradiction",
            "moved_source_target",
            "fallen_stack",
            "protected_displacement",
            "visible_emergency_proxy",
        }:
            label = "NOT_MET"
        else:
            label = "MET" if index % 2 == 0 else "UNKNOWN"
        return {
            "criterion_state": label,
            "appropriate_abstention": label == "UNKNOWN",
            "emergency": stratum == "visible_emergency_proxy",
        }
    reachable = stratum not in {"unreachable", "workspace_edge"} or index % 2 == 0
    depth_valid = stratum not in {"corrupted_depth", "sparse_depth"}
    return {
        "reachable": reachable,
        "depth_valid": depth_valid,
        "expected_status": "SUCCESS" if reachable and depth_valid else "REJECT",
        "coordinates_from_model_forbidden": True,
    }


def build_component_catalogue() -> tuple[ComponentItem, ...]:
    items: list[ComponentItem] = []
    for component, split_counts in SPLIT_COUNTS.items():
        ordinal = 0
        for split in ("dev", "pilot", "locked"):
            for split_index in range(split_counts[split]):
                stratum = STRATA[component][split_index % len(STRATA[component])]
                item_seed = _seed(component, split, split_index)
                item_id = f"{component.upper()}-{ordinal + 1:04d}"
                template_id = f"{component}:{split}:{stratum}:{split_index:04d}"
                inputs = {
                    "semantic_template": template_id,
                    "language_variant": f"{split}-language-{item_seed % 97:02d}",
                    "layout_variant": f"{split}-layout-{item_seed % 89:02d}",
                    "visual_condition": stratum,
                    "history_length": int(item_seed % 41),
                }
                items.append(
                    ComponentItem(
                        item_id=item_id,
                        component=component,
                        split=split,
                        stratum=stratum,
                        template_id=template_id,
                        seed=item_seed,
                        inputs=inputs,
                        expected=_expected(component, stratum, split_index),
                    )
                )
                ordinal += 1
    validate_component_catalogue(items)
    return tuple(items)


def validate_component_catalogue(items: Sequence[ComponentItem]) -> None:
    counts = Counter(item.component for item in items)
    if counts != Counter(COMPONENT_TARGETS):
        raise ComponentSchemaError(
            f"component target mismatch: {dict(counts)} != {COMPONENT_TARGETS}"
        )
    ids = [item.item_id for item in items]
    templates = [item.template_id for item in items]
    if len(ids) != len(set(ids)) or len(templates) != len(set(templates)):
        raise ComponentSchemaError("item and semantic-template IDs must be unique")
    for component, expected_splits in SPLIT_COUNTS.items():
        actual = Counter(
            item.split for item in items if item.component == component
        )
        if actual != Counter(expected_splits):
            raise ComponentSchemaError(
                f"{component} split mismatch: {dict(actual)}"
            )
        if not all(item.stratum in STRATA[component] for item in items if item.component == component):
            raise ComponentSchemaError(f"{component} contains an unknown stratum")


def score_prediction(
    item: ComponentItem,
    prediction: Mapping[str, Any],
) -> dict[str, Any]:
    """Strict exact-field scoring without aggregating away error causes."""

    if not isinstance(prediction, Mapping):
        raise ComponentSchemaError("prediction must be an object")
    expected_keys = set(item.expected)
    if set(prediction) != expected_keys:
        return {
            "item_id": item.item_id,
            "passed": False,
            "schema_valid": False,
            "mismatches": {
                "missing": sorted(expected_keys - set(prediction)),
                "unknown": sorted(set(prediction) - expected_keys),
            },
        }
    mismatches = {
        key: {"expected": item.expected[key], "actual": prediction[key]}
        for key in sorted(expected_keys)
        if prediction[key] != item.expected[key]
    }
    return {
        "item_id": item.item_id,
        "passed": not mismatches,
        "schema_valid": True,
        "mismatches": mismatches,
    }


def catalogue_payload(items: Sequence[ComponentItem] | None = None) -> dict[str, Any]:
    materialized = tuple(items) if items is not None else build_component_catalogue()
    validate_component_catalogue(materialized)
    return {
        "schema_version": 1,
        "generator": "experiments.harness.components:build_component_catalogue",
        "targets": COMPONENT_TARGETS,
        "split_counts": SPLIT_COUNTS,
        "items": [item.to_dict() for item in materialized],
    }


def write_component_catalogue(path: str | Path) -> str:
    payload = catalogue_payload()
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(rendered, encoding="utf-8")
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


__all__ = [
    "COMPONENT_TARGETS",
    "SPLIT_COUNTS",
    "STRATA",
    "ComponentItem",
    "ComponentSchemaError",
    "build_component_catalogue",
    "catalogue_payload",
    "score_prediction",
    "validate_component_catalogue",
    "write_component_catalogue",
]
