from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


_MISSING = object()
_PREDICATE_PATTERN = re.compile(
    r"^\s*([A-Za-z][A-Za-z0-9_-]*)\s*\((.*)\)\s*(?:=\s*(true|false))?\s*$",
    re.IGNORECASE,
)
_TEXT_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_COLOUR_ALIASES = {
    "r": "red",
    "red": "red",
    "g": "green",
    "green": "green",
    "b": "blue",
    "blue": "blue",
}
_POSITION_NAMES = ("bottom", "middle", "top")
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
_UNORDERED_ARGUMENT_PREDICATES = frozenset(
    {
        "all_items_assigned",
        "all_place_setting_items_placed",
        "stable_stack",
        "vertically_aligned",
    }
)


@dataclass(frozen=True, slots=True)
class ScoreReport:
    scenario_id: str
    passed: bool
    checks: tuple[dict[str, Any], ...]

    @property
    def correct(self) -> int:
        return sum(
            bool(check["passed"])
            for check in self.checks
            if check.get("evaluated") is True
        )

    @property
    def total(self) -> int:
        return sum(check.get("evaluated") is True for check in self.checks)

    @property
    def skipped(self) -> int:
        return sum(check.get("evaluated") is not True for check in self.checks)


def _nested(
    value: Mapping[str, Any], *paths: tuple[str, ...], default: Any = _MISSING
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


def _canonical_text(value: Any) -> str:
    """Return a deterministic comparison form for short semantic labels."""

    normalised = unicodedata.normalize("NFKC", str(value)).casefold()
    return " ".join(_TEXT_TOKEN_PATTERN.findall(normalised))


def _canonical_symbol(value: Any) -> str:
    return "_".join(_TEXT_TOKEN_PATTERN.findall(_canonical_text(value)))


def _mapping_values(
    value: Any,
    *,
    parent_key: str = "",
) -> Iterable[tuple[str, Any]]:
    """Yield recursively flattened task parameters with canonical key paths."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            key_part = _canonical_symbol(key)
            path = "_".join(part for part in (parent_key, key_part) if part)
            yield path, item
            yield from _mapping_values(item, parent_key=path)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield parent_key, item
            yield from _mapping_values(item, parent_key=parent_key)


def _normalise_colour(value: Any) -> str | None:
    text = _canonical_text(value)
    if text in _COLOUR_ALIASES:
        return _COLOUR_ALIASES[text]
    tokens = text.split()
    matches = {
        _COLOUR_ALIASES[token]
        for token in tokens
        if token in _COLOUR_ALIASES
    }
    return next(iter(matches)) if len(matches) == 1 else None


def _colour_sequence(value: Any) -> tuple[str, ...] | None:
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        colours = tuple(_normalise_colour(item) for item in value)
        if all(colours) and len(colours) == 3:
            return tuple(str(colour) for colour in colours)
        return None
    text = _canonical_text(value)
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


def _block_target_from_contract(contract: Mapping[str, Any]) -> str | None:
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
            for marker in ("order", "bottom_to_top", "colour_positions", "color_positions")
        ):
            candidate = _colour_sequence(value)
            if candidate is not None:
                declared_order = candidate
    if set(positions) == set(_POSITION_NAMES):
        declared_order = tuple(positions[position] for position in _POSITION_NAMES)

    text = _canonical_text(contract.get("confirmed_intent", ""))
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
    text = _canonical_text(value)
    if text in {"left", "left side", "on left", "on the left"}:
        return "left"
    if text in {"right", "right side", "on right", "on the right"}:
        return "right"
    return None


def _normalise_category(value: Any) -> str | None:
    tokens = _canonical_text(value).split()
    matches = {
        _CATEGORY_ALIASES[token]
        for token in tokens
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


def _category_target_from_contract(contract: Mapping[str, Any]) -> str | None:
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

    text = _canonical_text(contract.get("confirmed_intent", ""))
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


def _place_target_from_contract(contract: Mapping[str, Any]) -> str | None:
    for key, value in _mapping_values(contract.get("parameters", {})):
        if "hand" in key or "setting" in key or "target" in key:
            text = _canonical_text(value)
            if "right handed" in text or text == "right":
                return "right_handed"
            if "left handed" in text or text == "left":
                return "left_handed"
    text = _canonical_text(contract.get("confirmed_intent", ""))
    if "right handed" in text:
        return "right_handed"
    if "left handed" in text:
        return "left_handed"
    return None


def _infer_target_id(
    manifest: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> str | None:
    target = manifest.get("target")
    expected_target = (
        str(target.get("target_id", "")).strip()
        if isinstance(target, Mapping)
        else ""
    )
    for key, value in _mapping_values(contract.get("parameters", {})):
        if key.endswith("target_id") and str(value).strip() == expected_target:
            return expected_target

    family = _nested(manifest, ("scene", "family"), default="")
    if family == "block_stack":
        return _block_target_from_contract(contract)
    if family == "category_sort":
        return _category_target_from_contract(contract)
    if family == "place_setting":
        return _place_target_from_contract(contract)

    instruction = (
        target.get("instruction", "") if isinstance(target, Mapping) else ""
    )
    actual_intent = contract.get("confirmed_intent", "")
    if _canonical_text(instruction) == _canonical_text(actual_intent):
        return expected_target or None
    return None


def _object_aliases(manifest: Mapping[str, Any]) -> dict[str, str]:
    """Map unique semantic object labels in an agent schema to oracle object IDs."""

    initial_objects = _nested(
        manifest, ("scene", "initial_objects"), default=()
    )
    if not isinstance(initial_objects, Sequence) or isinstance(
        initial_objects, (str, bytes, bytearray)
    ):
        return {}
    candidates: list[tuple[str, str]] = []
    for item in initial_objects:
        if not isinstance(item, Mapping):
            continue
        object_id = str(item.get("object_id", "")).strip()
        if not object_id:
            continue
        aliases = {_canonical_text(object_id)}
        object_type = _canonical_text(item.get("object_type", ""))
        if object_type:
            aliases.add(object_type)
        attributes = item.get("attributes", {})
        if isinstance(attributes, Mapping):
            role = _canonical_text(attributes.get("role", ""))
            colour = _canonical_text(
                attributes.get("colour", attributes.get("color", ""))
            )
            if role:
                aliases.add(role)
            if colour:
                aliases.update(
                    {
                        colour,
                        f"{colour} block",
                        f"{colour} cube",
                    }
                )
        candidates.extend(
            (alias, object_id) for alias in aliases if alias
        )
    counts = Counter(alias for alias, _ in candidates)
    return {
        alias: object_id
        for alias, object_id in candidates
        if counts[alias] == 1
    }


def _canonical_argument(value: Any, aliases: Mapping[str, str]) -> str:
    text = _canonical_text(value)
    without_article = re.sub(r"^(?:the|a|an)\s+", "", text)
    if text in aliases:
        return aliases[text]
    if without_article in aliases:
        return aliases[without_article]
    for prefix in ("zone ", "anchor "):
        if without_article.startswith(prefix):
            return prefix.rstrip() + ":" + _canonical_symbol(
                without_article[len(prefix) :]
            )
    return _canonical_symbol(without_article)


def _parse_predicate(
    predicate: Any,
    arguments: Any,
    aliases: Mapping[str, str],
) -> tuple[str, tuple[str, ...], bool] | None:
    text = str(predicate).strip()
    match = _PREDICATE_PATTERN.fullmatch(text)
    truth = True
    parsed_arguments: list[Any] | None = None
    if match:
        name = _canonical_symbol(match.group(1))
        raw_arguments = match.group(2).strip()
        parsed_arguments = (
            [part.strip() for part in raw_arguments.split(",")]
            if raw_arguments
            else []
        )
        if match.group(3) is not None:
            truth = match.group(3).casefold() == "true"
    else:
        name = _canonical_symbol(text)
    if not name:
        return None
    if parsed_arguments is None:
        if not isinstance(arguments, Sequence) or isinstance(
            arguments, (str, bytes, bytearray)
        ):
            return None
        parsed_arguments = list(arguments)
    canonical_arguments = tuple(
        _canonical_argument(item, aliases) for item in parsed_arguments
    )
    if name in _UNORDERED_ARGUMENT_PREDICATES:
        canonical_arguments = tuple(sorted(canonical_arguments))
    return name, canonical_arguments, truth


def _predicate_display(
    predicate: tuple[str, tuple[str, ...], bool],
) -> str:
    name, arguments, truth = predicate
    return (
        f"{name.upper()}({','.join(arguments)})="
        f"{'true' if truth else 'false'}"
    )


def _oracle_goal_predicates(
    manifest: Mapping[str, Any],
) -> tuple[str, ...]:
    target = manifest.get("target")
    if not isinstance(target, Mapping):
        return ()
    raw_goals = target.get("goal_predicates")
    if not isinstance(raw_goals, Sequence) or isinstance(
        raw_goals, (str, bytes, bytearray)
    ):
        return ()
    aliases = _object_aliases(manifest)
    parsed: list[tuple[str, tuple[str, ...], bool]] = []
    for raw_goal in raw_goals:
        predicate = _parse_predicate(raw_goal, (), aliases)
        if predicate is None:
            raise ValueError(
                f"Manifest target contains malformed goal predicate {raw_goal!r}."
            )
        parsed.append(predicate)
    return tuple(sorted(_predicate_display(item) for item in parsed))


def _planner_goal_predicates(
    validation_spec: Any,
    manifest: Mapping[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if not isinstance(validation_spec, Mapping):
        return (), ("validation_spec is missing",)
    conditions = validation_spec.get("goal_conditions")
    if not isinstance(conditions, Sequence) or isinstance(
        conditions, (str, bytes, bytearray)
    ):
        return (), ("goal_conditions is missing or is not a list",)
    aliases = _object_aliases(manifest)
    parsed: list[str] = []
    issues: list[str] = []
    for index, condition in enumerate(conditions):
        if not isinstance(condition, Mapping):
            issues.append(f"goal condition {index + 1} is not an object")
            continue
        predicate = condition.get("predicate")
        if predicate is None or not str(predicate).strip():
            goal_id = str(condition.get("id") or index + 1)
            issues.append(f"goal condition {goal_id!r} has no structured predicate")
            continue
        canonical = _parse_predicate(
            predicate,
            condition.get("arguments", ()),
            aliases,
        )
        if canonical is None:
            goal_id = str(condition.get("id") or index + 1)
            issues.append(f"goal condition {goal_id!r} has a malformed predicate")
            continue
        parsed.append(_predicate_display(canonical))
    return tuple(sorted(parsed)), tuple(issues)


def score_agent_result(
    manifest: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    allow_partial: bool = False,
) -> ScoreReport:
    """Score a structured PrefMem result against hidden benchmark expectations.

    This function is intentionally one-way: it consumes oracle truth only after an
    agent run and returns metrics.  Its output must not be fed back as HRI context.
    """

    scenario_id = str(manifest.get("scenario_id", ""))
    expectations = manifest.get("benchmark_expectations", {})
    if not isinstance(expectations, Mapping):
        raise ValueError("Manifest benchmark_expectations must be an object.")

    validator_expected = _nested(expectations, ("validator", "outcome"))
    validator_called_expected = _nested(expectations, ("validator", "called"))
    complete_expected = _nested(
        expectations, ("validator", "task_complete")
    )
    execution_expected = _nested(
        expectations, ("execution", "expected_status")
    )
    dispatches_expected = _nested(
        expectations, ("execution", "expected_vla_dispatches")
    )
    planner_status_expected = _nested(
        expectations, ("planner", "expected_status")
    )
    hri_mode_expected = _nested(
        expectations,
        ("hri", "accepted_response_modes"),
        ("hri", "response_mode"),
    )
    hri_outcome_expected = _nested(expectations, ("hri", "terminal_outcome"))
    recovery_expected = _nested(expectations, ("recovery",))
    history_delta_expected = _nested(
        expectations, ("history", "expected_terminal_record_delta")
    )
    preference_delta_expected = _nested(
        expectations,
        ("preference", "expected_record_delta_without_explicit_consent"),
    )

    task = result.get("task") if isinstance(result.get("task"), Mapping) else {}
    attempts = task.get("attempts") if isinstance(task, Mapping) else None
    attempt_plans: list[Mapping[str, Any]] = []
    last_attempt: Mapping[str, Any] = {}
    if isinstance(attempts, list):
        for attempt in attempts:
            if isinstance(attempt, Mapping) and isinstance(
                attempt.get("plan"), Mapping
            ):
                attempt_plans.append(attempt["plan"])
        for attempt in reversed(attempts):
            if isinstance(attempt, Mapping):
                last_attempt = attempt
                break
    semantic_plan: Mapping[str, Any] = (
        attempt_plans[0] if attempt_plans else {}
    )
    resolved_task = _nested(
        result,
        ("resolved_task",),
        ("task_contract",),
        ("hri", "task_contract"),
        default=_MISSING,
    )
    if not isinstance(resolved_task, Mapping):
        resolved_task = _MISSING
    validation_spec = _nested(
        semantic_plan,
        ("validation_spec",),
        default=_MISSING,
    )
    validation = _nested(
        result,
        ("validator",),
        ("validation",),
        default=_MISSING,
    )
    if validation is _MISSING:
        validation = last_attempt.get("validation", _MISSING)
    if validation is _MISSING and (
        "outcome" in result or "task_complete" in result
    ):
        validation = result
    validator_called_actual = isinstance(validation, Mapping)
    validator_actual = (
        validation.get("outcome", _MISSING)
        if isinstance(validation, Mapping)
        else None
    )
    complete_actual = (
        validation.get("task_complete", _MISSING)
        if isinstance(validation, Mapping)
        else None
    )
    execution = _nested(result, ("execution",), default=_MISSING)
    if execution is _MISSING:
        execution = last_attempt.get("execution", _MISSING)
    execution_actual = _nested(
        execution if isinstance(execution, Mapping) else {},
        ("execution", "status"),
        ("status",),
        default=_MISSING,
    )
    hri_mode_actual = _nested(result, ("hri", "mode"))
    hri_outcome_actual = _nested(
        result,
        ("task", "outcome"),
        ("hri", "report", "outcome"),
    )
    first_attempt: Mapping[str, Any] = {}
    if isinstance(attempts, list):
        first_attempt = next(
            (
                attempt
                for attempt in attempts
                if isinstance(attempt, Mapping)
                and (
                    "validation_assurance" in attempt
                    or "execution_assurance" in attempt
                )
            ),
            {},
        )
    planner_status_actual = _nested(
        first_attempt or last_attempt,
        ("plan", "planning_status"),
    )
    dispatches_actual: Any = _MISSING
    if isinstance(attempts, list):
        dispatches_actual = 0
        for attempt in attempts:
            if not isinstance(attempt, Mapping):
                continue
            execution_result = attempt.get("execution")
            if not isinstance(execution_result, Mapping):
                continue
            subtask_results = execution_result.get("subtask_results", [])
            if isinstance(subtask_results, list):
                dispatches_actual += len(subtask_results)
    recovery_actual = _nested(
        first_attempt or last_attempt,
        ("validation_assurance", "next_action"),
        ("execution_assurance", "next_action"),
        default=_MISSING,
    )
    if recovery_actual is _MISSING:
        recovery_actual = _nested(result, ("task", "next_action"))
    history_delta_actual = _nested(
        result,
        ("memory", "history_delta"),
        ("history_delta",),
    )
    if (
        result.get("history_checkpointed") is True
        and history_delta_expected == 1
        and (
            history_delta_actual is _MISSING
            or history_delta_actual == 0
        )
    ):
        # A MEMORY_CONFIRM response has durably checkpointed the completed
        # episode in the history outbox. The repository append happens when the
        # user answers the optional memory question, but the task record is
        # already recoverable and counts as this command's terminal record.
        history_delta_actual = 1
    preference_delta_actual = _nested(
        result,
        ("memory", "preference_delta"),
        ("preference_delta",),
    )
    if preference_delta_actual is _MISSING and "memory" in result:
        memory_result = result.get("memory")
        preference_delta_actual = (
            1
            if isinstance(memory_result, Mapping)
            and memory_result.get("committed") is True
            else 0
        )

    checks: list[dict[str, Any]] = []

    def add_check(name: str, expected: Any, actual: Any) -> None:
        if expected is _MISSING:
            return
        if actual is _MISSING:
            checks.append(
                {
                    "name": name,
                    "expected": expected,
                    "actual": None,
                    "evaluated": False,
                    "passed": None,
                }
            )
            return
        checks.append(
            {
                "name": name,
                "expected": expected,
                "actual": actual,
                "evaluated": True,
                "passed": actual == expected,
            }
        )

    def add_required_check(
        name: str,
        expected: Any,
        actual: Any,
        *,
        passed: bool,
        details: Any = _MISSING,
    ) -> None:
        """Add an oracle check whose absent agent evidence is a scored failure."""

        check = {
            "name": name,
            "expected": expected,
            "actual": None if actual is _MISSING else actual,
            "evaluated": True,
            "passed": bool(passed),
        }
        if details is not _MISSING:
            check["details"] = details
        checks.append(check)

    target = manifest.get("target")
    if isinstance(target, Mapping) and str(target.get("target_id", "")).strip():
        expected_target_id = str(target["target_id"]).strip()
        inferred_target_id = (
            _infer_target_id(manifest, resolved_task)
            if isinstance(resolved_task, Mapping)
            else None
        )
        add_required_check(
            "hri_resolved_target",
            expected_target_id,
            inferred_target_id,
            passed=inferred_target_id == expected_target_id,
        )

        resolved_intent = (
            str(resolved_task.get("confirmed_intent", "")).strip()
            if isinstance(resolved_task, Mapping)
            else ""
        )
        planner_intent = (
            str(validation_spec.get("confirmed_intent", "")).strip()
            if isinstance(validation_spec, Mapping)
            else ""
        )
        add_required_check(
            "planner_confirmed_intent",
            resolved_intent or None,
            planner_intent or None,
            passed=(
                bool(resolved_intent)
                and bool(planner_intent)
                and _canonical_text(resolved_intent)
                == _canonical_text(planner_intent)
            ),
        )

        expected_goals = _oracle_goal_predicates(manifest)
        if expected_goals:
            actual_goals, goal_issues = _planner_goal_predicates(
                validation_spec,
                manifest,
            )
            add_required_check(
                "planner_goal_predicates",
                list(expected_goals),
                list(actual_goals),
                passed=not goal_issues and actual_goals == expected_goals,
                details=(
                    {"issues": list(goal_issues)}
                    if goal_issues
                    else _MISSING
                ),
            )

    if isinstance(hri_mode_expected, (list, tuple, set)):
        expected_modes = list(hri_mode_expected)
        checks.append(
            {
                "name": "hri_response_mode",
                "expected": expected_modes,
                "actual": (
                    None if hri_mode_actual is _MISSING else hri_mode_actual
                ),
                "evaluated": hri_mode_actual is not _MISSING,
                "passed": (
                    None
                    if hri_mode_actual is _MISSING
                    else hri_mode_actual in expected_modes
                ),
            }
        )
    else:
        add_check("hri_response_mode", hri_mode_expected, hri_mode_actual)
    add_check("hri_terminal_outcome", hri_outcome_expected, hri_outcome_actual)
    add_check(
        "validator_called",
        validator_called_expected,
        validator_called_actual,
    )
    add_check("validator_outcome", validator_expected, validator_actual)
    add_check("task_complete", complete_expected, complete_actual)
    add_check(
        "planner_status",
        planner_status_expected,
        planner_status_actual,
    )
    add_check("execution_status", execution_expected, execution_actual)
    add_check("vla_dispatches", dispatches_expected, dispatches_actual)
    add_check("recovery", recovery_expected, recovery_actual)
    add_check("history_delta", history_delta_expected, history_delta_actual)
    add_check(
        "preference_delta_without_consent",
        preference_delta_expected,
        preference_delta_actual,
    )

    evaluated = [check for check in checks if check.get("evaluated") is True]
    skipped = [check for check in checks if check.get("evaluated") is not True]
    return ScoreReport(
        scenario_id=scenario_id,
        passed=(
            bool(evaluated)
            and all(check["passed"] for check in evaluated)
            and (allow_partial or not skipped)
        ),
        checks=tuple(checks),
    )
