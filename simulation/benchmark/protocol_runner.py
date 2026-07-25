from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from .ablations import (
    MemoryContextDelivery,
    MemoryContextRecordingAgent,
    record_memory_context,
)

from .scorer import ScoreReport, score_agent_result


SUPPORTED_EXPECTATIONS = frozenset(
    {
        "active_equivalent_preferences",
        "active_rgb_preference_preserved",
        "clarification_required",
        "dispatches",
        "foreign_history_refs",
        "foreign_preference_refs",
        "history_delta",
        "history_retrieval_required",
        "min_dispatches",
        "memory_context_ownership_verified",
        "pending_question_cleared",
        "planner_status",
        "post_task_preference_question",
        "preference_delta",
        "preference_retrieval_required",
        "task_semantic_score",
        "uses_one_off_override",
        "validator_outcome",
    }
)

MANDATORY_MEMORY_CONTEXT_EXPECTATIONS = {
    "memory_context_ownership_verified": True,
    "foreign_history_refs": 0,
    "foreign_preference_refs": 0,
}


@dataclass(frozen=True, slots=True)
class ProtocolStepReport:
    index: int
    message: str
    result: dict[str, Any]
    history_delta: int | None
    preference_delta: int | None
    checks: tuple[dict[str, Any], ...]
    delivered_memory_context: dict[str, Any] | None = None
    task_score_required: bool = False
    task_score: dict[str, Any] | None = None

    @property
    def passed(self) -> bool:
        protocol_checks_pass = bool(self.checks) and all(
            check["evaluated"] and check["passed"] for check in self.checks
        )
        task_score_pass = (
            not self.task_score_required
            or (
                isinstance(self.task_score, dict)
                and self.task_score.get("passed") is True
            )
        )
        return protocol_checks_pass and task_score_pass


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


ManifestLoader = Callable[[str | Path], Mapping[str, Any]]
_TASK_SEMANTIC_EXCLUDED_CHECKS = frozenset(
    {
        "history_delta",
        "preference_delta_without_consent",
    }
)


def _load_manifest(episode_path: str | Path) -> Mapping[str, Any]:
    path = Path(episode_path) / "manifest.json"
    try:
        with path.open(encoding="utf-8") as stream:
            manifest = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Could not load protocol scoring manifest {path}: {error}"
        ) from error
    if not isinstance(manifest, dict):
        raise ValueError(f"Protocol scoring manifest must be an object: {path}")
    return manifest


def _score_dict(report: ScoreReport) -> dict[str, Any]:
    semantic_checks = [
        check
        for check in report.checks
        if str(check.get("name", "")) not in _TASK_SEMANTIC_EXCLUDED_CHECKS
    ]
    semantic_passed = bool(semantic_checks) and all(
        check.get("evaluated") is True and check.get("passed") is True
        for check in semantic_checks
    )
    return {
        "scenario_id": report.scenario_id,
        # The mandatory task-semantic gate excludes memory deltas because the
        # protocol layer scores those explicitly and memory ablations suppress
        # them by design. The unmodified strict scorer result remains available
        # as full_passed with every raw check below.
        "passed": semantic_passed,
        "semantic_passed": semantic_passed,
        "full_passed": report.passed,
        "correct": report.correct,
        "total": report.total,
        "skipped": report.skipped,
        "semantic_excluded_checks": sorted(_TASK_SEMANTIC_EXCLUDED_CHECKS),
        "semantic_checks": [copy.deepcopy(check) for check in semantic_checks],
        "checks": [copy.deepcopy(check) for check in report.checks],
    }


def _has_task_evidence(result: Mapping[str, Any]) -> bool:
    if _attempts(result):
        return True
    for key in ("validator", "validation", "execution"):
        if isinstance(result.get(key), Mapping):
            return True
    return False


def _score_task_result(
    manifest: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    history_delta: int | None,
    preference_delta: int | None,
) -> dict[str, Any]:
    """Score a completed turn after HRI returns, without mutating its result."""

    scoring_result = copy.deepcopy(dict(result))
    raw_memory = scoring_result.get("memory")
    memory = (
        copy.deepcopy(dict(raw_memory))
        if isinstance(raw_memory, Mapping)
        else {}
    )
    memory["history_delta"] = history_delta
    memory["preference_delta"] = preference_delta
    scoring_result["memory"] = memory
    return _score_dict(
        score_agent_result(
            manifest,
            scoring_result,
            allow_partial=False,
        )
    )


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
    delivery: MemoryContextDelivery | None,
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
    resolved_task = (
        result.get("resolved_task")
        if isinstance(result.get("resolved_task"), Mapping)
        else source.get("resolved_task")
        if isinstance(source.get("resolved_task"), Mapping)
        else {}
    )
    resolved_order = _colour_order(resolved_task.get("parameters", {}))
    rgb_preserved = _rgb_preserved(before, after)
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
        "foreign_history_refs": (
            len(delivery.foreign_history_ids)
            if delivery is not None
            else None
        ),
        "foreign_preference_refs": (
            len(delivery.foreign_preference_ids)
            if delivery is not None
            else None
        ),
        "history_delta": history_delta,
        "history_retrieval_required": (
            bool(delivery.owned_history_ids)
            if delivery is not None
            else None
        ),
        "memory_context_ownership_verified": (
            delivery.history_ownership_verified
            and delivery.preference_ownership_verified
            if delivery is not None
            else None
        ),
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
        "preference_retrieval_required": (
            bool(delivery.owned_preference_ids)
            if delivery is not None
            else None
        ),
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
    load_manifest: ManifestLoader | None = None,
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

    # Install the transparent observer only after consent fixtures have seeded
    # the underlying real Memory Agent. This preserves an identical starting
    # repository state across full and memory-ablation conditions.
    memory_recorder: MemoryContextRecordingAgent = record_memory_context(
        getattr(orchestrator, "memory_agent", None)
    )
    orchestrator.memory_agent = memory_recorder
    raw_steps = protocol.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError("Protocol requires at least one step.")

    manifest_loader = load_manifest or _load_manifest
    reports: list[ProtocolStepReport] = []
    active_episode_path: str | Path | None = None
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
            active_episode_path = episode_path

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
        expected_task_score = expected.get("task_semantic_score")
        if expected_task_score is not None and not isinstance(
            expected_task_score, bool
        ):
            raise ValueError("task_semantic_score must be a bool.")

        before = _memory_snapshot(orchestrator, user_id)
        attached_context = copy.deepcopy(
            getattr(orchestrator, "history_context", None)
        )
        memory_recorder.begin_evaluation_turn()
        # No oracle manifest is loaded before this call. The orchestrator sees
        # only the selected packet through its normal opaque dataset boundary.
        result = orchestrator.handle_user_message(message)
        delivery = memory_recorder.delivery_for_current_turn(
            attached_context=attached_context,
            user_id=user_id,
        )
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
            delivery=delivery,
        )
        task_score_required = (
            expected_task_score is True or _has_task_evidence(result)
        )
        task_score: dict[str, Any] | None = None
        if task_score_required:
            if active_episode_path is None:
                raise ValueError(
                    f"Protocol step {index} requires task semantic scoring but "
                    "has no active scenario."
                )
            # This is intentionally the first oracle access in the turn. It is
            # out-of-band and occurs only after HRI has returned and memory
            # deltas have been measured.
            manifest = manifest_loader(active_episode_path)
            task_score = _score_task_result(
                manifest,
                result,
                history_delta=history_delta,
                preference_delta=preference_delta,
            )
            actual["task_semantic_score"] = task_score["passed"]
        elif "task_semantic_score" in expected:
            actual["task_semantic_score"] = False

        effective_expected = copy.deepcopy(dict(expected))
        for name, required_value in (
            MANDATORY_MEMORY_CONTEXT_EXPECTATIONS.items()
        ):
            if (
                name in effective_expected
                and effective_expected[name] != required_value
            ):
                raise ValueError(
                    f"Protocol step {index} cannot weaken mandatory memory "
                    f"context expectation {name!r}={required_value!r}."
                )
            effective_expected[name] = required_value
        if task_score_required:
            # Even a hand-authored protocol that omitted the expectation cannot
            # allow an executed task to bypass the strict semantic scorer.
            effective_expected["task_semantic_score"] = True
        reports.append(
            ProtocolStepReport(
                index=index,
                message=message,
                result=copy.deepcopy(dict(result)),
                history_delta=history_delta,
                preference_delta=preference_delta,
                checks=_checks(effective_expected, actual),
                delivered_memory_context=(
                    delivery.to_dict() if delivery is not None else None
                ),
                task_score_required=task_score_required,
                task_score=copy.deepcopy(task_score),
            )
        )
    return ProtocolReport(protocol_id=protocol_id, steps=tuple(reports))
