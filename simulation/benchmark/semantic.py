"""Semantic normalization for benchmark conversations.

This module is deliberately independent from the benchmark scorer.  It turns
the oracle and planner predicate schemas into structured values, while retaining
the information needed to distinguish a wrong predicate from an ungrounded or
ambiguous natural-language argument.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Iterable, Mapping, Sequence


_MISSING = object()
_PREDICATE_PATTERN = re.compile(
    r"^\s*([A-Za-z][A-Za-z0-9_-]*)\s*\((.*)\)\s*"
    r"(?:=\s*(true|false))?\s*$",
    re.IGNORECASE,
)
_PREDICATE_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")
_TEXT_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_ARTICLE_PATTERN = re.compile(r"^(?:the|a|an)\s+")
_POSITION_NAMES = ("bottom", "middle", "top")
_COLOUR_ALIASES = {
    "r": "red",
    "red": "red",
    "g": "green",
    "green": "green",
    "b": "blue",
    "blue": "blue",
}
_CATEGORY_ALIASES = {
    "printed": "printed",
    "print": "printed",
    "book": "printed",
    "books": "printed",
    "electronic": "electronic",
    "electronics": "electronic",
    "device": "electronic",
    "devices": "electronic",
}

# Argument order has semantic meaning unless a predicate is explicitly listed
# here.  In particular, relations such as SUPPORTED_BY and ON_RELATIVE_SIDE are
# intentionally absent.
_UNORDERED_ARGUMENT_PREDICATES = frozenset(
    {
        "all_items_assigned",
        "all_place_setting_items_placed",
        "stable_stack",
        "vertically_aligned",
    }
)


@dataclass(frozen=True, order=True, slots=True)
class CanonicalPredicate:
    """A normalized predicate that preserves arguments, multiplicity, and truth."""

    name: str
    arguments: tuple[str, ...] = ()
    truth: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Canonical predicate name must not be empty.")
        if any(not argument for argument in self.arguments):
            raise ValueError("Canonical predicate arguments must not be empty.")
        if not isinstance(self.truth, bool):
            raise TypeError("Canonical predicate truth must be a bool.")

    @property
    def canonical(self) -> str:
        arguments = ",".join(self.arguments)
        truth = "true" if self.truth else "false"
        return f"{self.name.upper()}({arguments})={truth}"

    def __str__(self) -> str:
        return self.canonical

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "arguments": list(self.arguments),
            "truth": self.truth,
            "canonical": self.canonical,
        }


@dataclass(frozen=True, slots=True)
class PredicateDiff:
    """A multiplicity-preserving semantic comparison with grounding evidence."""

    expected: tuple[CanonicalPredicate, ...]
    actual: tuple[CanonicalPredicate, ...]
    matched: tuple[CanonicalPredicate, ...]
    missing: tuple[CanonicalPredicate, ...]
    extra: tuple[CanonicalPredicate, ...]
    unresolved: tuple[Mapping[str, Any], ...] = ()
    ambiguous: tuple[Mapping[str, Any], ...] = ()
    issues: tuple[str, ...] = ()

    @property
    def precision(self) -> float:
        if self.actual:
            return len(self.matched) / len(self.actual)
        return 1.0 if not self.expected else 0.0

    @property
    def recall(self) -> float:
        if self.expected:
            return len(self.matched) / len(self.expected)
        return 1.0

    @property
    def f1(self) -> float:
        denominator = self.precision + self.recall
        if denominator == 0.0:
            return 0.0
        return 2.0 * self.precision * self.recall / denominator

    @property
    def exact(self) -> bool:
        return not (
            self.missing
            or self.extra
            or self.unresolved
            or self.ambiguous
            or self.issues
        )

    def to_dict(self) -> dict[str, Any]:
        def predicates(
            values: tuple[CanonicalPredicate, ...],
        ) -> list[dict[str, Any]]:
            return [value.to_dict() for value in values]

        return {
            "expected": predicates(self.expected),
            "actual": predicates(self.actual),
            "matched": predicates(self.matched),
            "missing": predicates(self.missing),
            "extra": predicates(self.extra),
            "unresolved": [dict(item) for item in self.unresolved],
            "ambiguous": [dict(item) for item in self.ambiguous],
            "issues": list(self.issues),
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "exact": self.exact,
        }


def canonical_text(value: Any) -> str:
    """Return a deterministic comparison form for short semantic labels."""

    normalized = unicodedata.normalize("NFKC", str(value)).casefold()
    return " ".join(_TEXT_TOKEN_PATTERN.findall(normalized))


def _canonical_symbol(value: Any) -> str:
    return "_".join(_TEXT_TOKEN_PATTERN.findall(canonical_text(value)))


def _nested(
    value: Mapping[str, Any],
    *paths: tuple[str, ...],
    default: Any = _MISSING,
) -> Any:
    for path in paths:
        current: Any = value
        for key in path:
            if not isinstance(current, Mapping) or key not in current:
                break
            current = current[key]
        else:
            return current
    return default


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    )


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        converted = to_dict()
        return converted if isinstance(converted, Mapping) else None
    if is_dataclass(value) and not isinstance(value, type):
        converted = asdict(value)
        return converted if isinstance(converted, Mapping) else None
    return None


def _mapping_values(
    value: Any,
    *,
    parent_key: str = "",
) -> Iterable[tuple[str, Any]]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_part = _canonical_symbol(key)
            path = "_".join(part for part in (parent_key, key_part) if part)
            yield path, item
            yield from _mapping_values(item, parent_key=path)
    elif _is_sequence(value):
        for item in value:
            yield parent_key, item
            yield from _mapping_values(item, parent_key=parent_key)


def _normalise_colour(value: Any) -> str | None:
    text = canonical_text(value)
    if text in _COLOUR_ALIASES:
        return _COLOUR_ALIASES[text]
    matches = {
        _COLOUR_ALIASES[token]
        for token in text.split()
        if token in _COLOUR_ALIASES
    }
    return next(iter(matches)) if len(matches) == 1 else None


def _colour_sequence(value: Any) -> tuple[str, ...] | None:
    if _is_sequence(value):
        colours = tuple(_normalise_colour(item) for item in value)
        if len(colours) == 3 and all(colours):
            return tuple(str(colour) for colour in colours)
        return None
    text = canonical_text(value)
    if text in {"rgb", "r g b"}:
        return ("red", "green", "blue")
    if text in {"bgr", "b g r"}:
        return ("blue", "green", "red")
    colours = tuple(
        _COLOUR_ALIASES[token]
        for token in text.split()
        if token in {"red", "green", "blue"}
    )
    if len(colours) == 3 and len(set(colours)) == 3:
        return colours
    return None


def _block_target(contract: Mapping[str, Any]) -> str | None:
    parameters = contract.get("parameters", {})
    positions: dict[str, str] = {}
    declared_order: tuple[str, ...] | None = None
    for key, value in _mapping_values(parameters):
        leaf = key.rsplit("_", 1)[-1]
        if leaf in _POSITION_NAMES:
            colour = _normalise_colour(value)
            if colour is not None:
                positions[leaf] = colour
        if any(
            marker in key
            for marker in (
                "order",
                "bottom_to_top",
                "colour_positions",
                "color_positions",
            )
        ):
            candidate = _colour_sequence(value)
            if candidate is not None:
                declared_order = candidate
    if set(positions) == set(_POSITION_NAMES):
        declared_order = tuple(positions[position] for position in _POSITION_NAMES)

    text = canonical_text(contract.get("confirmed_intent", ""))
    tokens = text.split()
    if declared_order is None:
        text_positions: dict[str, str] = {}
        filler = r"(?:\s+(?:is|at|on|in|as|the|a|cube|block)){0,5}\s+"
        for colour, position in re.findall(
            rf"\b(red|green|blue)\b{filler}\b(bottom|middle|top)\b",
            text,
        ):
            text_positions[position] = colour
        for position, colour in re.findall(
            rf"\b(bottom|middle|top)\b{filler}\b(red|green|blue)\b",
            text,
        ):
            text_positions.setdefault(position, colour)
        if set(text_positions) == set(_POSITION_NAMES):
            declared_order = tuple(
                text_positions[position] for position in _POSITION_NAMES
            )
    if declared_order is None and (
        "bottom to top" in text or "bottom up" in text
    ):
        declared_order = _colour_sequence(text)
    if declared_order is None:
        for acronym, order in (
            ("rgb", ("red", "green", "blue")),
            ("bgr", ("blue", "green", "red")),
        ):
            if acronym in tokens:
                declared_order = order
                break
    return {
        ("red", "green", "blue"): "rgb_bottom_to_top",
        ("blue", "green", "red"): "bgr_bottom_to_top",
    }.get(declared_order)


def _normalise_side(value: Any) -> str | None:
    text = canonical_text(value)
    if text in {"left", "left side", "on left", "on the left"}:
        return "left"
    if text in {"right", "right side", "on right", "on the right"}:
        return "right"
    return None


def _normalise_category(value: Any) -> str | None:
    matches = {
        _CATEGORY_ALIASES[token]
        for token in canonical_text(value).split()
        if token in _CATEGORY_ALIASES
    }
    return next(iter(matches)) if len(matches) == 1 else None


def _nearest_token_value(
    tokens: Sequence[str],
    source_aliases: Mapping[str, str],
    target_values: set[str],
    *,
    maximum_distance: int = 6,
) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for index, token in enumerate(tokens):
        source = source_aliases.get(token)
        if source is None:
            continue
        following = [
            (other_index - index, other_token)
            for other_index, other_token in enumerate(tokens)
            if other_token in target_values
            and 0 < other_index - index <= maximum_distance
        ]
        if following:
            _, target = min(following)
            assignments[source] = target
            continue
        preceding = [
            (index - other_index, other_token)
            for other_index, other_token in enumerate(tokens)
            if other_token in target_values
            and 0 < index - other_index <= maximum_distance
        ]
        if preceding:
            _, target = min(preceding)
            assignments[source] = target
    return assignments


def _category_target(contract: Mapping[str, Any]) -> str | None:
    assignments: dict[str, str] = {}
    for key, value in _mapping_values(contract.get("parameters", {})):
        category = _normalise_category(key)
        side = _normalise_side(value)
        if category and side:
            assignments[category] = side
            continue
        side = _normalise_side(key)
        if side:
            category = _normalise_category(value)
            if category:
                assignments[category] = side

    text = canonical_text(contract.get("confirmed_intent", ""))
    if len(assignments) < 2:
        assignments.update(
            _nearest_token_value(
                text.split(),
                _CATEGORY_ALIASES,
                {"left", "right"},
            )
        )
    if assignments == {"printed": "left", "electronic": "right"}:
        return "printed_left"
    if assignments == {"printed": "right", "electronic": "left"}:
        return "electronics_left"
    return None


def _place_target(contract: Mapping[str, Any]) -> str | None:
    for key, value in _mapping_values(contract.get("parameters", {})):
        if "hand" in key or "setting" in key or "target" in key:
            text = canonical_text(value)
            if "right handed" in text or text == "right":
                return "right_handed"
            if "left handed" in text or text == "left":
                return "left_handed"
    text = canonical_text(contract.get("confirmed_intent", ""))
    if "right handed" in text:
        return "right_handed"
    if "left handed" in text:
        return "left_handed"
    return None


_FAMILY_TARGETS = {
    "block_stack": frozenset({"rgb_bottom_to_top", "bgr_bottom_to_top"}),
    "category_sort": frozenset({"printed_left", "electronics_left"}),
    "place_setting": frozenset({"right_handed", "left_handed"}),
}


@dataclass(frozen=True, slots=True)
class SubtaskTargetInference:
    """Conservative target evidence extracted from executable subtasks."""

    family: str
    candidate_target_ids: tuple[str, ...]
    recognized_action_count: int
    auxiliary_action_count: int
    issues: tuple[str, ...]
    evidence: tuple[Mapping[str, Any], ...]

    @property
    def target_id(self) -> str | None:
        if (
            self.recognized_action_count > 0
            and not self.issues
            and len(self.candidate_target_ids) == 1
        ):
            return self.candidate_target_ids[0]
        return None

    def compatible(self, target_id: str) -> bool:
        return (
            self.recognized_action_count > 0
            and not self.issues
            and target_id in self.candidate_target_ids
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "candidate_target_ids": list(self.candidate_target_ids),
            "target_id": self.target_id,
            "recognized_action_count": self.recognized_action_count,
            "auxiliary_action_count": self.auxiliary_action_count,
            "issues": list(self.issues),
            "evidence": [dict(item) for item in self.evidence],
        }


_NO_ACTION_PATTERNS = (
    re.compile(r"\bdo nothing\b"),
    re.compile(r"\bno action\b"),
    re.compile(r"\bno manipulation\b"),
    re.compile(r"\bno (?:move|movement) (?:is )?required\b"),
    re.compile(r"\b(?:wait|idle)\b"),
    re.compile(r"\bleave\b.*\b(?:unchanged|as is)\b"),
    re.compile(r"\balready (?:satisfied|complete|correct)\b"),
    re.compile(r"\bobservation only\b"),
)
_AUXILIARY_ACTION_PATTERN = re.compile(
    r"\b(?:clear|remove|unstack|move away|set aside|discard)\b"
)


def _scene_objects(manifest: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    values = _nested(manifest, ("scene", "initial_objects"), default=())
    return tuple(item for item in values if isinstance(item, Mapping)) if _is_sequence(values) else ()


def _object_semantic_value(
    manifest: Mapping[str, Any],
    reference: Any,
    attribute: str,
) -> str | None:
    reference_text = _strip_article(reference)
    if not reference_text:
        return None
    matches: set[str] = set()
    for item in _scene_objects(manifest):
        attributes = item.get("attributes", {})
        attributes = attributes if isinstance(attributes, Mapping) else {}
        object_id = canonical_text(item.get("object_id", ""))
        object_type = canonical_text(item.get("object_type", ""))
        role = canonical_text(attributes.get("role", ""))
        colour = canonical_text(
            attributes.get("colour", attributes.get("color", ""))
        )
        category = canonical_text(
            attributes.get(
                "semantic_category",
                attributes.get("category", ""),
            )
        )
        aliases = {
            object_id,
            object_type,
            role,
            colour,
            category,
            f"{colour} {object_type}".strip(),
            f"{category} {object_type}".strip(),
            f"{category} {role}".strip(),
        }
        aliases.discard("")
        if reference_text in aliases:
            value = {
                "colour": colour,
                "category": category,
                "role": role or object_type,
            }.get(attribute, "")
            if value:
                matches.add(value)
    return next(iter(matches)) if len(matches) == 1 else None


def _structured_reference(
    subtask: Mapping[str, Any],
    *keys: str,
) -> Any:
    for key in keys:
        value = subtask.get(key, _MISSING)
        if value is not _MISSING and str(value).strip():
            return value
    return ""


def _block_action_candidates(
    manifest: Mapping[str, Any],
    subtask: Mapping[str, Any],
    text: str,
) -> tuple[set[str] | None, Mapping[str, Any]]:
    order = _colour_sequence(text)
    order_target = {
        ("red", "green", "blue"): "rgb_bottom_to_top",
        ("blue", "green", "red"): "bgr_bottom_to_top",
    }.get(order)
    if order_target:
        return {order_target}, {"kind": "declared_stack_order", "order": list(order or ())}

    source_reference = _structured_reference(subtask, "target", "source")
    destination_reference = _structured_reference(subtask, "destination")
    source = _normalise_colour(source_reference) or _object_semantic_value(
        manifest, source_reference, "colour"
    )
    destination = _normalise_colour(destination_reference) or _object_semantic_value(
        manifest, destination_reference, "colour"
    )
    if source is None:
        match = re.search(
            r"\b(?:pick(?: up)?|move|place|stack)\b(?:\s+the)?\s+"
            r"(red|green|blue)\b",
            text,
        )
        source = match.group(1) if match else None
    if destination is None:
        match = re.search(
            r"\b(?:on top of|onto|above|supported by)\b(?:\s+the)?\s+"
            r"(red|green|blue)\b",
            text,
        )
        destination = match.group(1) if match else None

    relation_targets = {
        ("green", "red"): "rgb_bottom_to_top",
        ("blue", "green"): "rgb_bottom_to_top",
        ("green", "blue"): "bgr_bottom_to_top",
        ("red", "green"): "bgr_bottom_to_top",
    }
    relation_target = relation_targets.get((source, destination))
    if relation_target:
        return {relation_target}, {
            "kind": "support_relation",
            "source": source,
            "destination": destination,
        }
    destination_text = canonical_text(destination_reference)
    base_destination = any(
        marker in destination_text or marker in text
        for marker in ("stack centre", "stack center", "stack zone", "table base")
    )
    if source in {"red", "blue"} and base_destination:
        target = (
            "rgb_bottom_to_top" if source == "red" else "bgr_bottom_to_top"
        )
        return {target}, {"kind": "base_placement", "source": source}
    return None, {"kind": "unrecognized_block_action"}


def _category_for_reference(
    manifest: Mapping[str, Any],
    value: Any,
) -> str | None:
    return _normalise_category(value) or _object_semantic_value(
        manifest, value, "category"
    )


def _category_action_candidates(
    manifest: Mapping[str, Any],
    subtask: Mapping[str, Any],
    text: str,
) -> tuple[set[str] | None, Mapping[str, Any]]:
    whole_target = _category_target(
        {"confirmed_intent": text, "parameters": {}}
    )
    if whole_target:
        return {whole_target}, {"kind": "declared_category_layout"}

    source_reference = _structured_reference(subtask, "target", "source")
    destination_reference = _structured_reference(subtask, "destination")
    category = _category_for_reference(manifest, source_reference)
    side = _normalise_side(destination_reference)
    assignments: dict[str, str] = {}
    if category and side:
        assignments[category] = side
    else:
        assignments.update(
            _nearest_token_value(
                text.split(),
                _CATEGORY_ALIASES,
                {"left", "right"},
            )
        )
    candidates = {
        {
            ("printed", "left"): "printed_left",
            ("printed", "right"): "electronics_left",
            ("electronic", "right"): "printed_left",
            ("electronic", "left"): "electronics_left",
        }[item]
        for item in assignments.items()
        if item
        in {
            ("printed", "left"),
            ("printed", "right"),
            ("electronic", "left"),
            ("electronic", "right"),
        }
    }
    if candidates:
        return candidates, {
            "kind": "category_placement",
            "assignments": dict(sorted(assignments.items())),
        }
    return None, {"kind": "unrecognized_category_action"}


def _place_role(
    manifest: Mapping[str, Any],
    value: Any,
) -> str | None:
    text = canonical_text(value)
    for role in ("fork", "knife", "cup", "mug", "napkin", "plate"):
        if role in text.split():
            return "cup" if role == "mug" else role
    role = _object_semantic_value(manifest, value, "role")
    return "cup" if role == "mug" else role


def _place_action_candidates(
    manifest: Mapping[str, Any],
    subtask: Mapping[str, Any],
    text: str,
) -> tuple[set[str] | None, Mapping[str, Any]]:
    whole_target = _place_target(
        {"confirmed_intent": text, "parameters": {}}
    )
    if whole_target:
        return {whole_target}, {"kind": "declared_handedness"}

    source_reference = _structured_reference(subtask, "target", "source")
    destination_reference = _structured_reference(subtask, "destination")
    role = _place_role(manifest, source_reference)
    if role is None:
        role = _place_role(manifest, text)
    destination_text = canonical_text(destination_reference) or text
    side = _normalise_side(destination_reference)
    if side is None:
        side_matches = [
            item for item in ("left", "right") if item in destination_text.split()
        ]
        side = side_matches[0] if len(side_matches) == 1 else None
    decisive = {
        ("fork", "left"): "right_handed",
        ("fork", "right"): "left_handed",
        ("knife", "right"): "right_handed",
        ("knife", "left"): "left_handed",
        ("cup", "right"): "right_handed",
        ("cup", "left"): "left_handed",
        ("napkin", "left"): "right_handed",
        ("napkin", "right"): "left_handed",
    }.get((role, side))
    if decisive:
        return {decisive}, {
            "kind": "place_setting_relation",
            "role": role,
            "side": side,
        }
    neutral = (
        role == "plate"
        and any(marker in destination_text for marker in ("center", "centre"))
    ) or (
        role == "knife"
        and "blade" in text
        and any(marker in text for marker in ("toward plate", "towards plate", "facing plate", "inward"))
    )
    if neutral:
        return set(_FAMILY_TARGETS["place_setting"]), {
            "kind": "place_setting_neutral_relation",
            "role": role,
        }
    return None, {"kind": "unrecognized_place_setting_action"}


def infer_subtask_target(
    manifest: Mapping[str, Any],
    subtasks: Any,
) -> SubtaskTargetInference:
    """Infer intended task target only from executable action evidence."""

    family = str(_nested(manifest, ("scene", "family"), default=""))
    family_targets = set(_FAMILY_TARGETS.get(family, ()))
    issues: list[str] = []
    evidence: list[Mapping[str, Any]] = []
    recognized = 0
    auxiliary = 0
    candidates = set(family_targets)
    if not family_targets:
        issues.append(f"unsupported task family: {family or '<missing>'}")
    if not _is_sequence(subtasks):
        issues.append("subtasks must be a non-string sequence")
        subtasks = ()
    if not subtasks:
        issues.append("subtasks contain no executable actions")

    parser = {
        "block_stack": _block_action_candidates,
        "category_sort": _category_action_candidates,
        "place_setting": _place_action_candidates,
    }.get(family)
    for index, raw_subtask in enumerate(subtasks, start=1):
        subtask = _as_mapping(raw_subtask)
        if subtask is None:
            issues.append(f"subtask {index} is not an object")
            continue
        instruction = str(subtask.get("task_instruction", "")).strip()
        text = canonical_text(instruction)
        if not text:
            issues.append(f"subtask {index} has no task_instruction")
            continue
        if any(pattern.search(text) for pattern in _NO_ACTION_PATTERNS):
            issues.append(f"subtask {index} declares no executable action")
            evidence.append({"subtask": index, "kind": "no_action"})
            continue
        parsed, detail = parser(manifest, subtask, text) if parser else (None, {})
        if parsed:
            recognized += 1
            overlap = candidates & parsed
            evidence.append(
                {
                    "subtask": index,
                    **dict(detail),
                    "candidate_target_ids": sorted(parsed),
                }
            )
            if overlap:
                candidates = overlap
            else:
                candidates.clear()
                issues.append(f"subtask {index} conflicts with earlier actions")
            continue
        if _AUXILIARY_ACTION_PATTERN.search(text):
            auxiliary += 1
            evidence.append({"subtask": index, "kind": "auxiliary_action"})
            continue
        issues.append(f"subtask {index} is unrelated or semantically ungrounded")
        evidence.append({"subtask": index, **dict(detail)})
    if recognized == 0:
        issues.append("no target-defining action was recognized")
    return SubtaskTargetInference(
        family=family,
        candidate_target_ids=tuple(sorted(candidates)),
        recognized_action_count=recognized,
        auxiliary_action_count=auxiliary,
        issues=tuple(issues),
        evidence=tuple(evidence),
    )


def infer_target_id(
    manifest: Mapping[str, Any],
    contract_or_text: Mapping[str, Any] | str | Any,
) -> str | None:
    """Infer the target selected by a task contract or confirmed-intent string."""

    contract = _as_mapping(contract_or_text)
    if contract is None:
        contract = {"confirmed_intent": str(contract_or_text)}

    family = str(_nested(manifest, ("scene", "family"), default=""))
    expected_targets = _FAMILY_TARGETS.get(family, frozenset())
    expected = _nested(manifest, ("target", "target_id"), default="")
    expected_id = str(expected).strip()

    for key, value in _mapping_values(contract.get("parameters", {})):
        if key.endswith("target_id"):
            declared = _canonical_symbol(value)
            if declared in expected_targets or (
                declared and declared == expected_id
            ):
                return declared

    inference = {
        "block_stack": _block_target,
        "category_sort": _category_target,
        "place_setting": _place_target,
    }.get(family)
    if inference is not None:
        inferred = inference(contract)
        if inferred is not None:
            return inferred

    instruction = _nested(manifest, ("target", "instruction"), default="")
    intent = contract.get(
        "confirmed_intent",
        contract.get("instruction", contract.get("intent", "")),
    )
    if canonical_text(instruction) == canonical_text(intent):
        return expected_id or None
    return None


def _object_aliases(
    manifest: Mapping[str, Any],
) -> dict[str, frozenset[str]]:
    """Return every semantic alias and all object IDs it could denote."""

    objects = _nested(manifest, ("scene", "initial_objects"), default=())
    if not _is_sequence(objects):
        return {}

    candidates: dict[str, set[str]] = defaultdict(set)
    for item in objects:
        if not isinstance(item, Mapping):
            continue
        object_id = str(item.get("object_id", "")).strip()
        if not object_id:
            continue
        object_type = canonical_text(item.get("object_type", ""))
        attributes = item.get("attributes", {})
        attributes = attributes if isinstance(attributes, Mapping) else {}
        role = canonical_text(attributes.get("role", ""))
        colour = canonical_text(
            attributes.get("colour", attributes.get("color", ""))
        )
        category = canonical_text(
            attributes.get(
                "semantic_category",
                attributes.get("category", ""),
            )
        )

        aliases = {
            canonical_text(object_id),
            object_type,
            role,
            colour,
            category,
        }
        nouns = {value for value in (object_type, role) if value}
        descriptors = {value for value in (colour, category) if value}
        aliases.update(
            f"{descriptor} {noun}"
            for descriptor in descriptors
            for noun in nouns
        )
        if colour:
            aliases.update({f"{colour} block", f"{colour} cube"})
        for alias in aliases:
            if alias:
                candidates[alias].add(object_id)
    return {
        alias: frozenset(sorted(object_ids))
        for alias, object_ids in candidates.items()
    }


def _strip_article(value: Any) -> str:
    return _ARTICLE_PATTERN.sub("", canonical_text(value))


def _canonical_argument_symbol(value: Any) -> str:
    text = _strip_article(value)
    for namespace in ("zone", "anchor"):
        prefix = f"{namespace} "
        suffix = f" {namespace}"
        if text.startswith(prefix):
            return f"{namespace}:{_canonical_symbol(text[len(prefix):])}"
        if text.endswith(suffix):
            return f"{namespace}:{_canonical_symbol(text[:-len(suffix)])}"
    return _canonical_symbol(text)


def _expected_constant_aliases(
    expected: Sequence[CanonicalPredicate],
    object_ids: set[str],
) -> dict[str, frozenset[str]]:
    candidates: dict[str, set[str]] = defaultdict(set)
    for predicate in expected:
        for argument in predicate.arguments:
            if argument in object_ids:
                continue
            candidates[canonical_text(argument)].add(argument)
            if ":" in argument:
                namespace, value = argument.split(":", 1)
                candidates[canonical_text(value)].add(argument)
                candidates[canonical_text(f"{namespace} {value}")].add(argument)
                candidates[canonical_text(f"{value} {namespace}")].add(argument)
    return {
        alias: frozenset(sorted(values))
        for alias, values in candidates.items()
        if alias
    }


def _truth_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = canonical_text(value)
    if text == "true":
        return True
    if text == "false":
        return False
    raise ValueError(f"Predicate truth must be true or false, not {value!r}.")


def _argument_issue(
    *,
    condition_index: int,
    condition_id: str,
    predicate: str,
    argument_index: int,
    argument: Any,
    candidates: Sequence[str] = (),
) -> dict[str, Any]:
    issue = {
        "condition_index": condition_index,
        "condition_id": condition_id,
        "predicate": predicate,
        "argument_index": argument_index,
        "argument": str(argument),
        "canonical_text": _strip_article(argument),
    }
    if candidates:
        issue["candidates"] = list(candidates)
    return issue


def _resolve_argument(
    value: Any,
    *,
    object_aliases: Mapping[str, frozenset[str]],
    constant_aliases: Mapping[str, frozenset[str]],
    trusted: bool,
    condition_index: int,
    condition_id: str,
    predicate: str,
    argument_index: int,
) -> tuple[str, str | None, dict[str, Any] | None]:
    text = _strip_article(value)
    candidates = object_aliases.get(text, frozenset())
    if len(candidates) == 1:
        return next(iter(candidates)), None, None
    if len(candidates) > 1:
        issue = _argument_issue(
            condition_index=condition_index,
            condition_id=condition_id,
            predicate=predicate,
            argument_index=argument_index,
            argument=value,
            candidates=sorted(candidates),
        )
        return f"?ambiguous:{_canonical_symbol(text)}", "ambiguous", issue

    constants = constant_aliases.get(text, frozenset())
    if len(constants) == 1:
        return next(iter(constants)), None, None
    if len(constants) > 1:
        issue = _argument_issue(
            condition_index=condition_index,
            condition_id=condition_id,
            predicate=predicate,
            argument_index=argument_index,
            argument=value,
            candidates=sorted(constants),
        )
        return f"?ambiguous:{_canonical_symbol(text)}", "ambiguous", issue

    symbol = _canonical_argument_symbol(value)
    if trusted:
        return symbol, None, None
    issue = _argument_issue(
        condition_index=condition_index,
        condition_id=condition_id,
        predicate=predicate,
        argument_index=argument_index,
        argument=value,
    )
    return f"?unresolved:{symbol}", "unresolved", issue


def _parse_predicate(
    predicate: Any,
    arguments: Any,
    *,
    object_aliases: Mapping[str, frozenset[str]],
    constant_aliases: Mapping[str, frozenset[str]],
    trusted: bool,
    condition_index: int,
    condition_id: str,
    truth_override: Any = _MISSING,
) -> tuple[
    CanonicalPredicate | None,
    tuple[Mapping[str, Any], ...],
    tuple[Mapping[str, Any], ...],
    str | None,
]:
    text = str(predicate).strip()
    match = _PREDICATE_PATTERN.fullmatch(text)
    parsed_arguments: list[Any]
    if match:
        name = _canonical_symbol(match.group(1))
        raw_arguments = match.group(2).strip()
        parsed_arguments = (
            [part.strip() for part in raw_arguments.split(",")]
            if raw_arguments
            else []
        )
        truth = (
            match.group(3).casefold() == "true"
            if match.group(3) is not None
            else (
                _truth_value(truth_override)
                if truth_override is not _MISSING
                else True
            )
        )
    else:
        if not _PREDICATE_NAME_PATTERN.fullmatch(text):
            return None, (), (), f"Malformed predicate {predicate!r}."
        name = _canonical_symbol(text)
        if not _is_sequence(arguments):
            return (
                None,
                (),
                (),
                f"Predicate {predicate!r} arguments must be a list.",
            )
        parsed_arguments = list(arguments)
        truth = (
            _truth_value(truth_override)
            if truth_override is not _MISSING
            else True
        )

    resolved: list[str] = []
    unresolved: list[Mapping[str, Any]] = []
    ambiguous: list[Mapping[str, Any]] = []
    for argument_index, argument in enumerate(parsed_arguments):
        canonical, issue_kind, issue = _resolve_argument(
            argument,
            object_aliases=object_aliases,
            constant_aliases=constant_aliases,
            trusted=trusted,
            condition_index=condition_index,
            condition_id=condition_id,
            predicate=name,
            argument_index=argument_index,
        )
        resolved.append(canonical)
        if issue_kind == "unresolved" and issue is not None:
            unresolved.append(issue)
        elif issue_kind == "ambiguous" and issue is not None:
            ambiguous.append(issue)
    canonical_arguments = tuple(resolved)
    if name in _UNORDERED_ARGUMENT_PREDICATES:
        canonical_arguments = tuple(sorted(canonical_arguments))
    return (
        CanonicalPredicate(name, canonical_arguments, truth),
        tuple(unresolved),
        tuple(ambiguous),
        None,
    )


def canonical_goal_predicates(
    manifest: Mapping[str, Any],
) -> tuple[CanonicalPredicate, ...]:
    """Return the manifest target goals in deterministic structured form."""

    raw_goals = _nested(manifest, ("target", "goal_predicates"), default=())
    if not _is_sequence(raw_goals):
        raise ValueError("Manifest target goal_predicates must be a list.")
    aliases = _object_aliases(manifest)
    parsed: list[CanonicalPredicate] = []
    for index, raw_goal in enumerate(raw_goals):
        goal_mapping = _as_mapping(raw_goal)
        if goal_mapping is not None:
            predicate = goal_mapping.get(
                "predicate",
                goal_mapping.get("name", ""),
            )
            arguments = goal_mapping.get("arguments", ())
            truth = goal_mapping.get(
                "truth",
                goal_mapping.get("value", _MISSING),
            )
        else:
            predicate = raw_goal
            arguments = ()
            truth = _MISSING
        canonical, unresolved, ambiguous, issue = _parse_predicate(
            predicate,
            arguments,
            object_aliases=aliases,
            constant_aliases={},
            trusted=True,
            condition_index=index,
            condition_id=f"oracle-{index + 1}",
            truth_override=truth,
        )
        if canonical is None or unresolved or ambiguous or issue:
            reason = issue or "oracle predicate argument could not be grounded"
            raise ValueError(
                f"Manifest goal predicate {index + 1} is invalid: {reason}"
            )
        parsed.append(canonical)
    return tuple(sorted(parsed))


def planner_predicate_diff(
    manifest: Mapping[str, Any],
    validation_spec: Any,
) -> PredicateDiff:
    """Compare planner goal conditions with the manifest's oracle predicates."""

    expected = canonical_goal_predicates(manifest)
    aliases = _object_aliases(manifest)
    object_ids = {
        object_id
        for candidates in aliases.values()
        for object_id in candidates
    }
    constants = _expected_constant_aliases(expected, object_ids)

    spec = _as_mapping(validation_spec)
    issues: list[str] = []
    unresolved: list[Mapping[str, Any]] = []
    ambiguous: list[Mapping[str, Any]] = []
    actual: list[CanonicalPredicate] = []
    if spec is None:
        issues.append("validation_spec is missing or is not an object")
        conditions: Sequence[Any] = ()
    else:
        raw_conditions = spec.get("goal_conditions")
        if not _is_sequence(raw_conditions):
            issues.append("goal_conditions is missing or is not a list")
            conditions = ()
        else:
            conditions = raw_conditions

    for index, raw_condition in enumerate(conditions):
        condition = _as_mapping(raw_condition)
        if condition is None:
            issues.append(f"goal condition {index + 1} is not an object")
            continue
        condition_id = str(condition.get("id") or index + 1)
        predicate = condition.get("predicate")
        if predicate is None or not str(predicate).strip():
            issues.append(
                f"goal condition {condition_id!r} has no structured predicate"
            )
            continue
        truth = _MISSING
        for key in ("truth", "value", "expected"):
            if key in condition:
                truth = condition[key]
                break
        try:
            canonical, unresolved_items, ambiguous_items, issue = (
                _parse_predicate(
                    predicate,
                    condition.get("arguments", ()),
                    object_aliases=aliases,
                    constant_aliases=constants,
                    trusted=False,
                    condition_index=index,
                    condition_id=condition_id,
                    truth_override=truth,
                )
            )
        except ValueError as error:
            canonical = None
            unresolved_items = ()
            ambiguous_items = ()
            issue = str(error)
        unresolved.extend(unresolved_items)
        ambiguous.extend(ambiguous_items)
        if issue is not None:
            issues.append(f"goal condition {condition_id!r}: {issue}")
        if canonical is not None:
            actual.append(canonical)

    expected_counter = Counter(expected)
    actual_counter = Counter(actual)
    matched_counter = expected_counter & actual_counter
    missing_counter = expected_counter - actual_counter
    extra_counter = actual_counter - expected_counter

    return PredicateDiff(
        expected=tuple(sorted(expected_counter.elements())),
        actual=tuple(sorted(actual_counter.elements())),
        matched=tuple(sorted(matched_counter.elements())),
        missing=tuple(sorted(missing_counter.elements())),
        extra=tuple(sorted(extra_counter.elements())),
        unresolved=tuple(unresolved),
        ambiguous=tuple(ambiguous),
        issues=tuple(issues),
    )


__all__ = [
    "CanonicalPredicate",
    "PredicateDiff",
    "canonical_goal_predicates",
    "canonical_text",
    "infer_target_id",
    "planner_predicate_diff",
]
