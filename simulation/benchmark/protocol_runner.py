from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping


SUPPORTED_EXPECTATIONS = frozenset(
    {
        "active_equivalent_preferences",
        "active_rgb_preference_preserved",
        "clarification_required",
        "dispatches",
        "foreign_preference_refs",
        "history_delta",
        "history_retrieval_required",
        "min_dispatches",
        "pending_question_cleared",
        "planner_status",
        "post_task_preference_question",
        "preference_delta",
        "preference_retrieval_required",
        "uses_one_off_override",
        "validator_outcome",
    }
)


@dataclass(frozen=True, slots=True)
class ProtocolStepReport:
    index: int
    message: str
    result: dict[str, Any]
    history_delta: int | None
    preference_delta: int | None
    checks: tuple[dict[str, Any], ...]

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(
            check["evaluated"] and check["passed"] for check in self.checks
        )


@dataclass(frozen=True, slots=True)
class ProtocolReport:
    protocol_id: str
    steps: tuple[ProtocolStepReport, ...]

    @property
    def passed(self) -> bool:
        return bool(self.steps) and all(step.passed for step in self.steps)


@dataclass(frozen=True, slots=True)
class _MemorySnapshot:
    histories: tuple[dict[str, Any], ...]
    preferences: tuple[dict[str, Any], ...]
    outbox_sources: tuple[dict[str, Any], ...]


def _memory_snapshot(orchestrator: Any, user_id: str) -> _MemorySnapshot | None:
    memory_agent = getattr(orchestrator, "memory_agent", None)
    history = getattr(memory_agent, "history_repository", None)
    preferences = getattr(memory_agent, "preference_repository", None)
    list_history = getattr(history, "list_episodes", None)
    list_preferences = getattr(preferences, "list_preferences", None)
    if not callable(list_history) or not callable(list_preferences):
        return None

    outbox_sources: list[dict[str, Any]] = []
    outbox = getattr(orchestrator, "history_outbox", None)
    list_pending = getattr(outbox, "list_pending", None)
    if callable(list_pending):
        for pending in list_pending(user_id):
            if isinstance(pending, Mapping) and isinstance(
                pending.get("source"), Mapping
            ):
                outbox_sources.append(copy.deepcopy(dict(pending["source"])))

    return _MemorySnapshot(
        histories=tuple(
            copy.deepcopy(item)
            for item in list_history(user_id)
            if isinstance(item, dict)
        ),
        preferences=tuple(
            copy.deepcopy(item)
            for item in list_preferences(user_id)
            if isinstance(item, dict)
        ),
        outbox_sources=tuple(outbox_sources),
    )


def _require_fresh_snapshot(
    snapshot: _MemorySnapshot | None,
    *,
    user_id: str,
) -> None:
    if (
        snapshot is None
        or snapshot.histories
        or snapshot.preferences
        or snapshot.outbox_sources
    ):
        raise ValueError(
            "Protocol requires a fresh per-user history, preference, and outbox "
            f"namespace for user {user_id!r} before its declared fixture."
        )


def _attempts(result: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    task = result.get("task")
    raw = task.get("attempts", []) if isinstance(task, Mapping) else []
    return [item for item in raw if isinstance(item, Mapping)]


def _new_episode_source(
    before: _MemorySnapshot | None,
    after: _MemorySnapshot | None,
) -> Mapping[str, Any]:
    if before is None or after is None:
        return {}
    before_history_ids = {
        str(item.get("episode_id", "")) for item in before.histories
    }
    for episode in after.histories:
        if str(episode.get("episode_id", "")) not in before_history_ids:
            return episode
    before_outbox_ids = {
        str(item.get("episode_id", "")) for item in before.outbox_sources
    }
    for source in after.outbox_sources:
        if str(source.get("episode_id", "")) not in before_outbox_ids:
            return source
    return {}


def _references(result: Mapping[str, Any], source: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    hri = result.get("hri") if isinstance(result.get("hri"), Mapping) else {}
    trace = hri.get("trace") if isinstance(hri.get("trace"), Mapping) else {}
    history_refs = {
        str(item)
        for item in trace.get("history_refs", [])
        if str(item).strip()
    }
    preference_refs = {
        str(item)
        for item in trace.get("memory_refs", [])
        if str(item).strip()
    }
    resolved_task = (
        result.get("resolved_task")
        if isinstance(result.get("resolved_task"), Mapping)
        else source.get("resolved_task")
        if isinstance(source.get("resolved_task"), Mapping)
        else {}
    )
    preference_refs.update(
        str(item)
        for item in resolved_task.get("preference_refs", [])
        if str(item).strip()
    )
    return history_refs, preference_refs


_COLOUR = re.compile(r"\b(red|green|blue)\b", re.IGNORECASE)


def _colour_token(value: Any) -> str | None:
    match = _COLOUR.search(str(value))
    return match.group(1).lower() if match else None


def _colour_order(value: Any) -> tuple[str, str, str] | None:
    """Extract a structured three-colour order for out-of-band assertions.

    This is an oracle-side invariant check, not the retrieval mechanism. Runtime
    preference matching and compaction remain VLM-reasoned.
    """

    if isinstance(value, Mapping):
        if all(key in value for key in ("bottom", "middle", "top")):
            order = tuple(
                _colour_token(value[key]) for key in ("bottom", "middle", "top")
            )
            if all(order):
                return order  # type: ignore[return-value]
        for key in (
            "order_bottom_to_top",
            "bottom_to_top",
            "order",
            "structured_value",
            "value",
            "parameters",
            "preference",
        ):
            if key in value:
                order = _colour_order(value[key])
                if order is not None:
                    return order
        for nested in value.values():
            order = _colour_order(nested)
            if order is not None:
                return order
    elif isinstance(value, (list, tuple)) and len(value) == 3:
        order = tuple(_colour_token(item) for item in value)
        if all(order):
            return order  # type: ignore[return-value]
    return None


def _rgb_preferences(
    snapshot: _MemorySnapshot | None,
) -> dict[str, dict[str, Any]]:
    if snapshot is None:
        return {}
    return {
        str(item.get("id", "")): item
        for item in snapshot.preferences
        if str(item.get("id", "")).strip()
        and _colour_order(item) == ("red", "green", "blue")
    }


def _rgb_preserved(
    before: _MemorySnapshot | None,
    after: _MemorySnapshot | None,
) -> bool | None:
    if before is None or after is None:
        return None
    before_rgb = _rgb_preferences(before)
    after_rgb = _rgb_preferences(after)
    return bool(before_rgb) and all(
        preference_id in after_rgb
        and after_rgb[preference_id] == before_record
        for preference_id, before_record in before_rgb.items()
    )


def _actual_values(
    result: Mapping[str, Any],
    orchestrator: Any,
    *,
    before: _MemorySnapshot | None,
    after: _MemorySnapshot | None,
) -> dict[str, Any]:
    attempts = _attempts(result)
    first_plan = next(
        (
            attempt.get("plan")
            for attempt in attempts
            if isinstance(attempt.get("plan"), Mapping)
        ),
        {},
    )
    validations = [
        attempt["validation"]
        for attempt in attempts
        if isinstance(attempt.get("validation"), Mapping)
    ]
    dispatches = 0
    for attempt in attempts:
        execution = attempt.get("execution")
        if not isinstance(execution, Mapping):
            continue
        subtask_results = execution.get("subtask_results", [])
        if isinstance(subtask_results, list):
            dispatches += len(subtask_results)

    hri = result.get("hri") if isinstance(result.get("hri"), Mapping) else {}
    mode = str(hri.get("mode", ""))
    source = _new_episode_source(before, after)
    history_refs, preference_refs = _references(result, source)
    resolved_task = (
        result.get("resolved_task")
        if isinstance(result.get("resolved_task"), Mapping)
        else source.get("resolved_task")
        if isinstance(source.get("resolved_task"), Mapping)
        else {}
    )
    resolved_order = _colour_order(resolved_task.get("parameters", {}))
    rgb_preserved = _rgb_preserved(before, after)
    current_ids = (
        {str(item.get("id", "")) for item in after.preferences}
        if after is not None
        else None
    )
    history_delta = (
        len(after.histories) - len(before.histories)
        if before is not None and after is not None
        else None
    )
    preference_delta = (
        len(after.preferences) - len(before.preferences)
        if before is not None and after is not None
        else None
    )
    return {
        "active_equivalent_preferences": (
            len(_rgb_preferences(after)) if after is not None else None
        ),
        "active_rgb_preference_preserved": rgb_preserved,
        "clarification_required": mode in {"ASK", "CONFIRM"},
        "dispatches": dispatches,
        "foreign_preference_refs": (
            len(preference_refs - current_ids)
            if current_ids is not None
            else None
        ),
        "history_delta": history_delta,
        "history_retrieval_required": bool(history_refs),
        "min_dispatches": dispatches,
        "pending_question_cleared": getattr(
            orchestrator, "pending_question", None
        )
        is None,
        "planner_status": (
            str(first_plan.get("planning_status", ""))
            if isinstance(first_plan, Mapping)
            else ""
        ),
        "post_task_preference_question": mode == "MEMORY_CONFIRM",
        "preference_delta": preference_delta,
        "preference_retrieval_required": bool(preference_refs),
        "uses_one_off_override": (
            resolved_order == ("blue", "green", "red")
            and rgb_preserved is True
            and preference_delta == 0
        ),
        "validator_outcome": (
            str(validations[-1].get("outcome", "")) if validations else None
        ),
    }


def _checks(
    expected: Mapping[str, Any], actual: Mapping[str, Any]
) -> tuple[dict[str, Any], ...]:
    unknown = set(expected) - SUPPORTED_EXPECTATIONS
    if unknown:
        raise ValueError(
            f"Unsupported protocol expectation keys: {sorted(unknown)}"
        )
    checks: list[dict[str, Any]] = []
    for name, expected_value in expected.items():
        actual_value = actual.get(name)
        if actual_value is None:
            checks.append(
                {
                    "name": name,
                    "expected": copy.deepcopy(expected_value),
                    "actual": None,
                    "evaluated": False,
                    "passed": None,
                }
            )
            continue
        if name == "min_dispatches":
            if isinstance(expected_value, bool) or not isinstance(
                expected_value, int
            ):
                raise ValueError("min_dispatches must be an integer.")
            passed = (
                isinstance(actual_value, int)
                and not isinstance(actual_value, bool)
                and actual_value >= expected_value
            )
        else:
            passed = actual_value == expected_value
        checks.append(
            {
                "name": name,
                "expected": copy.deepcopy(expected_value),
                "actual": copy.deepcopy(actual_value),
                "evaluated": True,
                "passed": passed,
            }
        )
    return tuple(checks)


def run_memory_protocol(
    protocol: Mapping[str, Any],
    orchestrator: Any,
    *,
    resolve_scenario: Callable[[Mapping[str, Any]], str | Path],
    apply_fixture: Callable[[str, Mapping[str, Any], Any], None] | None = None,
    fixtures: Mapping[str, Mapping[str, Any]] | None = None,
) -> ProtocolReport:
    """Execute a semantic protocol through the real HRI state machine.

    Each step is exactly one user turn. ``resolve_scenario`` binds a selector to
    an opaque generated episode path. ``apply_fixture`` is explicit because fixture
    approval must go through the experiment's consent-aware memory API rather than
    direct JSON mutation.
    """

    protocol_id = str(protocol.get("protocol_id", ""))
    user_id = str(protocol.get("user_id", ""))
    if not protocol_id or not user_id:
        raise ValueError("Protocol requires protocol_id and user_id.")
    configured_user = str(getattr(orchestrator.config, "user_id", ""))
    if configured_user != user_id:
        raise ValueError(
            f"Protocol user {user_id!r} does not match orchestrator user "
            f"{configured_user!r}."
        )

    initial_snapshot = _memory_snapshot(orchestrator, user_id)
    requires_fresh = protocol.get("requires_fresh_memory", True) is True
    if requires_fresh:
        _require_fresh_snapshot(
            initial_snapshot,
            user_id=user_id,
        )

    fixture_id = protocol.get("initial_memory_fixture")
    if fixture_id is not None:
        if apply_fixture is None or fixtures is None:
            raise ValueError(
                "Protocol requires an initial fixture; provide fixtures and "
                "apply_fixture."
            )
        fixture = fixtures.get(str(fixture_id))
        if fixture is None:
            raise ValueError(f"Unknown memory fixture: {fixture_id!r}")
        fixture_user = str(fixture.get("user_id", "")).strip()
        if requires_fresh and fixture_user and fixture_user != user_id:
            _require_fresh_snapshot(
                _memory_snapshot(orchestrator, fixture_user),
                user_id=fixture_user,
            )
        apply_fixture(
            str(fixture_id),
            copy.deepcopy(fixture),
            orchestrator,
        )

    raw_steps = protocol.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError("Protocol requires at least one step.")

    reports: list[ProtocolStepReport] = []
    for index, step in enumerate(raw_steps, start=1):
        if not isinstance(step, Mapping):
            raise ValueError(f"Protocol step {index} must be an object.")
        selector = step.get("scenario_selector")
        if selector is not None and not isinstance(selector, Mapping):
            raise ValueError(
                f"Protocol step {index} scenario_selector must be an object."
            )
        if isinstance(selector, Mapping):
            episode_path = resolve_scenario(selector)
            orchestrator.switch_dataset(str(episode_path))

        query = str(step.get("query", "")).strip()
        reply = str(step.get("reply", "")).strip()
        if bool(query) == bool(reply):
            raise ValueError(
                f"Protocol step {index} must contain exactly one query or reply."
            )
        message = query or reply
        expected = step.get("expected")
        if not isinstance(expected, Mapping) or not expected:
            raise ValueError(
                f"Protocol step {index} expected must be a non-empty object."
            )
        unknown = set(expected) - SUPPORTED_EXPECTATIONS
        if unknown:
            raise ValueError(
                f"Protocol step {index} has unsupported expectations: "
                f"{sorted(unknown)}"
            )

        before = _memory_snapshot(orchestrator, user_id)
        result = orchestrator.handle_user_message(message)
        after = _memory_snapshot(orchestrator, user_id)
        history_delta = (
            len(after.histories) - len(before.histories)
            if before is not None and after is not None
            else None
        )
        preference_delta = (
            len(after.preferences) - len(before.preferences)
            if before is not None and after is not None
            else None
        )
        actual = _actual_values(
            result,
            orchestrator,
            before=before,
            after=after,
        )
        reports.append(
            ProtocolStepReport(
                index=index,
                message=message,
                result=copy.deepcopy(dict(result)),
                history_delta=history_delta,
                preference_delta=preference_delta,
                checks=_checks(expected, actual),
            )
        )
    return ProtocolReport(protocol_id=protocol_id, steps=tuple(reports))
