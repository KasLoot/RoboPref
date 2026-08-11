"""Executable scenario driving and trace construction.

This module is the only place where catalogue triggers are turned into ordered
runtime events.  It deliberately keeps hidden simulator state in a separate
ledger: model-facing callbacks receive frames and public runtime state, never
the :class:`EpisodeTrace` used by the objective oracles.
"""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Callable, Mapping
import copy
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
import json
import math
from typing import Any

import numpy as np

from experiments.harness.oracles import (
    DialogueAct,
    EpisodeTrace,
    Event,
    MemoryEvent,
    OracleResult,
    evaluate_all,
)
from experiments.harness.recording import AttemptRecorder
from experiments.harness.scenarios import ScenarioSpec, TriggerSpec


class ScenarioDriverError(RuntimeError):
    """Raised when a frozen trigger cannot be executed unambiguously."""


TriggerHandler = Callable[[TriggerSpec, "ScenarioDriver"], None]


_CAPABILITY_FACTORY_KEY = object()


class EvaluationCapability:
    """Opaque, driver-specific authority for one post-episode evaluation.

    The campaign runner retains this object while behavior engines receive only
    the :class:`ScenarioDriver`.  This is an API-level isolation boundary, not a
    claim that hostile Python code cannot use reflection.
    """

    __slots__ = ("__driver_token",)

    def __init__(self, factory_key: object, driver_token: object) -> None:
        if factory_key is not _CAPABILITY_FACTORY_KEY:
            raise TypeError("evaluation capabilities are issued only by ScenarioDriver")
        self.__driver_token = driver_token

    def _authorizes(self, driver_token: object) -> bool:
        return self.__driver_token is driver_token


def jsonable(value: Any) -> Any:
    """Convert runtime contracts and arrays to strict, stable JSON values."""

    if is_dataclass(value) and not isinstance(value, type):
        return jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): jsonable(child) for key, child in value.items()}
    if isinstance(value, (set, frozenset)):
        children = [jsonable(child) for child in value]
        return sorted(
            children,
            key=lambda child: json.dumps(
                child,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
        )
    if isinstance(value, (tuple, list)):
        return [jsonable(child) for child in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ScenarioDriverError("non-finite values cannot enter experiment artifacts")
    return value


def _event_sequence(event_id: str) -> int:
    if not event_id.startswith("E") or not event_id[1:].isdigit():
        raise ScenarioDriverError(f"invalid recorder event ID: {event_id!r}")
    return int(event_id[1:])


def _deep_set(root: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    if not parts or any(not part for part in parts):
        raise ScenarioDriverError(f"invalid hidden-state path: {path!r}")
    cursor = root
    for part in parts[:-1]:
        child = cursor.get(part)
        if child is None:
            child = {}
            cursor[part] = child
        if not isinstance(child, dict):
            raise ScenarioDriverError(f"hidden-state path collides at {part!r}")
        cursor = child
    cursor[parts[-1]] = jsonable(value)


@dataclass(frozen=True, slots=True)
class TriggerAudit:
    trigger_id: str
    required_firings: int
    observed_firings: int
    required_boundary: str
    predicate_event_ids: tuple[str, ...]
    response_event_ids: tuple[str, ...]
    response_event_deltas: tuple[int, ...]
    max_response_events: int
    passed: bool


@dataclass(frozen=True, slots=True)
class _PendingPredicate:
    kind: str
    occurrence: int
    event_id: str
    sequence: int
    attributes: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class _TriggerResponse:
    event_id: str
    sequence: int
    delta_events: int


@dataclass(frozen=True, slots=True)
class ScenarioEvaluation:
    oracle_results: tuple[OracleResult, ...]
    trigger_results: tuple[TriggerAudit, ...]

    @property
    def passed(self) -> bool:
        return all(item.passed for item in self.oracle_results) and all(
            item.passed for item in self.trigger_results
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "oracles": [asdict(item) for item in self.oracle_results],
            "triggers": [asdict(item) for item in self.trigger_results],
        }


class ScenarioDriver:
    """Drive frozen triggers and build the independent objective-oracle trace."""

    def __init__(
        self,
        scenario: ScenarioSpec,
        recorder: AttemptRecorder,
        *,
        trigger_handlers: Mapping[str, TriggerHandler] | None = None,
        _evaluation_token: object | None = None,
    ) -> None:
        if not isinstance(scenario, ScenarioSpec):
            raise TypeError("scenario must be a ScenarioSpec")
        if not isinstance(recorder, AttemptRecorder):
            raise TypeError("recorder must be an AttemptRecorder")
        self.scenario = scenario
        self.recorder = recorder
        self._handlers = dict(trigger_handlers or {})
        self._events: list[Event] = []
        self._dialogue: list[DialogueAct] = []
        self._memory: list[MemoryEvent] = []
        self._hidden_state: dict[str, Any] = {}
        self._safety_violations: list[str] = []
        self._predicate_counts: Counter[str] = Counter()
        self._trigger_counts: Counter[str] = Counter()
        self._trigger_predicates: dict[str, list[str]] = {}
        self._trigger_responses: dict[str, list[_TriggerResponse]] = {}
        self._pending_predicates: deque[_PendingPredicate] = deque()
        self._processing_triggers = False
        self._active_trigger: tuple[TriggerSpec, int] | None = None
        self._active_response_recorded = False
        self._evaluation_token = _evaluation_token or object()
        self._evaluated = False

    @classmethod
    def with_evaluation_capability(
        cls,
        scenario: ScenarioSpec,
        recorder: AttemptRecorder,
        *,
        trigger_handlers: Mapping[str, TriggerHandler] | None = None,
    ) -> tuple["ScenarioDriver", EvaluationCapability]:
        """Create a driver and return evaluation authority separately.

        Callers must retain the capability outside any behavior-facing context.
        Evaluation is driver-specific and can occur only once.
        """

        token = object()
        driver = cls(
            scenario,
            recorder,
            trigger_handlers=trigger_handlers,
            _evaluation_token=token,
        )
        return driver, EvaluationCapability(_CAPABILITY_FACTORY_KEY, token)

    @property
    def hidden_state(self) -> Mapping[str, Any]:
        # Never expose the oracle ledger by reference.  In particular, returning
        # only a shallow Mapping proxy would still permit mutation of nested
        # dictionaries and lists without a journal entry.
        return copy.deepcopy(self._hidden_state)

    @property
    def last_sequence(self) -> int:
        return self._events[-1].seq if self._events else 0

    def _append(self, kind: str, **attributes: Any) -> dict[str, Any]:
        if "scenario_id" in attributes:
            raise ScenarioDriverError("scenario_id is recorder-owned and cannot be overridden")
        payload = {"scenario_id": self.scenario.scenario_id, **jsonable(attributes)}
        recorded = self.recorder.events.append(kind, **payload)
        self._events.append(
            Event(
                seq=_event_sequence(recorded["event_id"]),
                kind=kind,
                attributes=payload,
            )
        )
        return recorded

    def emit(self, kind: str, **attributes: Any) -> dict[str, Any]:
        """Append a source event and synchronously fire matching triggers."""

        if not isinstance(kind, str) or not kind.strip():
            raise ValueError("event kind must be non-empty")
        normalized = kind.strip()
        event = self._append(normalized, **attributes)
        self._predicate_counts[normalized] += 1
        self._pending_predicates.append(
            _PendingPredicate(
                kind=normalized,
                occurrence=self._predicate_counts[normalized],
                event_id=str(event["event_id"]),
                sequence=_event_sequence(event["event_id"]),
                attributes=jsonable(attributes),
            )
        )
        if not self._processing_triggers:
            self._drain_trigger_queue()
        return event

    def _drain_trigger_queue(self) -> None:
        self._processing_triggers = True
        try:
            while self._pending_predicates:
                self._fire_ready(self._pending_predicates.popleft())
        finally:
            self._processing_triggers = False

    def _fire_ready(self, predicate: _PendingPredicate) -> None:
        for trigger in self.scenario.triggers:
            if trigger.predicate_event != predicate.kind:
                continue
            if predicate.occurrence != trigger.predicate_occurrence:
                continue
            observed_boundary = predicate.attributes.get("boundary")
            if observed_boundary != trigger.boundary:
                raise ScenarioDriverError(
                    f"frozen trigger {trigger.trigger_id} expected boundary "
                    f"{trigger.boundary!r}, observed {observed_boundary!r}"
                )
            remaining = trigger.firing_count - self._trigger_counts[trigger.trigger_id]
            for firing_index in range(max(0, remaining)):
                self._trigger_counts[trigger.trigger_id] += 1
                self._trigger_predicates.setdefault(trigger.trigger_id, []).append(
                    predicate.event_id
                )
                fired = self._append(
                    "trigger_fired",
                    trigger_id=trigger.trigger_id,
                    trigger_kind=trigger.kind,
                    boundary=trigger.boundary,
                    action=trigger.action,
                    firing_index=firing_index + 1,
                    required_firing_count=trigger.firing_count,
                )
                handler = self._handlers.get(trigger.trigger_id) or self._handlers.get(
                    trigger.action
                )
                if handler is None:
                    raise ScenarioDriverError(
                        f"no action handler for frozen trigger {trigger.trigger_id}: "
                        f"{trigger.action}"
                    )
                fired_sequence = _event_sequence(fired["event_id"])
                responses = self._trigger_responses.setdefault(trigger.trigger_id, [])
                response_count_before = len(responses)
                self._active_trigger = (trigger, fired_sequence)
                self._active_response_recorded = False
                try:
                    handler(trigger, self)
                finally:
                    self._active_trigger = None
                    self._active_response_recorded = False
                if len(responses) != response_count_before + 1:
                    raise ScenarioDriverError(
                        f"trigger handler {trigger.trigger_id} returned without exactly one "
                        "recorded response event"
                    )

    def record_trigger_response(
        self,
        trigger_id: str,
        *,
        event_kind: str,
        **attributes: Any,
    ) -> dict[str, Any]:
        """Record the one response that discharges the active trigger firing.

        A response is accepted only while its handler is executing and only when
        its event sequence delta is within the frozen ``max_response_events``.
        Handlers should choose a semantically meaningful ``event_kind`` such as
        ``safe_response_recorded`` or ``scripted_input_delivered``.
        """

        if self._active_trigger is None:
            raise ScenarioDriverError("trigger response recorded outside an active handler")
        trigger, fired_sequence = self._active_trigger
        if trigger_id != trigger.trigger_id:
            raise ScenarioDriverError(
                f"response for {trigger_id!r} does not match active trigger "
                f"{trigger.trigger_id!r}"
            )
        if self._active_response_recorded:
            raise ScenarioDriverError(
                f"trigger {trigger.trigger_id} recorded more than one response"
            )
        if "trigger_id" in attributes or "trigger_response_evidence" in attributes:
            raise ScenarioDriverError("trigger response identity fields are recorder-owned")
        event = self.emit(
            event_kind,
            trigger_id=trigger.trigger_id,
            trigger_response_evidence=True,
            **attributes,
        )
        sequence = _event_sequence(event["event_id"])
        delta = sequence - fired_sequence
        if delta > trigger.max_response_events:
            raise ScenarioDriverError(
                f"trigger {trigger.trigger_id} response delta {delta} exceeds frozen "
                f"limit {trigger.max_response_events}"
            )
        self._trigger_responses.setdefault(trigger.trigger_id, []).append(
            _TriggerResponse(event["event_id"], sequence, delta)
        )
        self._active_response_recorded = True
        return event

    def dialogue_act(self, act: str, *, text: str | None = None) -> dict[str, Any]:
        event = self.emit("dialogue_act", act=act, text=text)
        self._dialogue.append(DialogueAct(_event_sequence(event["event_id"]), act))
        return event

    def memory_event(
        self,
        *,
        record_id: str,
        operation: str,
        authorized: bool,
        applied: bool,
    ) -> dict[str, Any]:
        event = self.emit(
            "memory_event",
            record_id=record_id,
            operation=operation,
            authorized=authorized,
            applied=applied,
        )
        self._memory.append(
            MemoryEvent(
                _event_sequence(event["event_id"]),
                record_id,
                operation,
                authorized,
                applied,
            )
        )
        return event

    def set_hidden(self, path: str, value: Any, *, evidence: str) -> None:
        _deep_set(self._hidden_state, path, value)
        snapshot = jsonable(self._hidden_state)
        self.recorder.record_hidden_state(snapshot)
        self.emit(
            "hidden_state_observed",
            path=path,
            value=jsonable(value),
            evidence=evidence,
        )

    def replace_hidden(self, state: Mapping[str, Any], *, evidence: str) -> None:
        self._hidden_state = jsonable(dict(state))
        self.recorder.record_hidden_state(self._hidden_state)
        self.emit("hidden_state_snapshot", evidence=evidence)

    def safety_violation(self, violation: str, **evidence: Any) -> None:
        if not violation.strip():
            raise ValueError("safety violation must be non-empty")
        self._safety_violations.append(violation.strip())
        self.emit("safety_violation", violation=violation.strip(), **evidence)

    def trace(self) -> EpisodeTrace:
        return EpisodeTrace(
            events=tuple(self._events),
            hidden_state=copy.deepcopy(self._hidden_state),
            dialogue=tuple(self._dialogue),
            memory_events=tuple(self._memory),
            safety_violations=tuple(self._safety_violations),
        )

    def evaluate(self, capability: EvaluationCapability) -> ScenarioEvaluation:
        """Evaluate frozen oracles after behavior is complete, never during it."""

        if not isinstance(capability, EvaluationCapability) or not capability._authorizes(
            self._evaluation_token
        ):
            raise ScenarioDriverError("runner evaluation capability is missing or invalid")
        if self._evaluated:
            raise ScenarioDriverError("scenario evaluation is one-shot")
        if self._active_trigger is not None or self._processing_triggers:
            raise ScenarioDriverError("cannot evaluate while a trigger handler is active")
        # Consume before evaluating so an oracle/recorder exception cannot invite
        # a scientifically ambiguous second evaluation.
        self._evaluated = True

        trace = self.trace()
        results = evaluate_all(self.scenario.oracles, trace)
        for result in results:
            observation = asdict(result)
            self.recorder.record_oracle_observation(result.oracle_id, observation)
            self._append(
                "oracle_evaluated",
                oracle_id=result.oracle_id,
                passed=result.passed,
                detail=result.detail,
            )
        trigger_results = tuple(
            TriggerAudit(
                trigger_id=trigger.trigger_id,
                required_firings=trigger.firing_count,
                observed_firings=self._trigger_counts[trigger.trigger_id],
                required_boundary=trigger.boundary,
                predicate_event_ids=tuple(
                    self._trigger_predicates.get(trigger.trigger_id, ())
                ),
                response_event_ids=tuple(
                    item.event_id
                    for item in self._trigger_responses.get(trigger.trigger_id, ())
                ),
                response_event_deltas=tuple(
                    item.delta_events
                    for item in self._trigger_responses.get(trigger.trigger_id, ())
                ),
                max_response_events=trigger.max_response_events,
                passed=(
                    self._trigger_counts[trigger.trigger_id] == trigger.firing_count
                    and len(self._trigger_predicates.get(trigger.trigger_id, ()))
                    == trigger.firing_count
                    and len(self._trigger_responses.get(trigger.trigger_id, ()))
                    == trigger.firing_count
                    and all(
                        item.delta_events <= trigger.max_response_events
                        for item in self._trigger_responses.get(trigger.trigger_id, ())
                    )
                ),
            )
            for trigger in self.scenario.triggers
        )
        for result in trigger_results:
            self.recorder.record_oracle_observation(
                f"trigger:{result.trigger_id}", asdict(result)
            )
            self._append(
                "trigger_audited",
                trigger_id=result.trigger_id,
                required_firings=result.required_firings,
                observed_firings=result.observed_firings,
                required_boundary=result.required_boundary,
                predicate_event_ids=list(result.predicate_event_ids),
                response_event_ids=list(result.response_event_ids),
                response_event_deltas=list(result.response_event_deltas),
                max_response_events=result.max_response_events,
                passed=result.passed,
            )
        return ScenarioEvaluation(results, trigger_results)


__all__ = [
    "EvaluationCapability",
    "ScenarioDriver",
    "ScenarioDriverError",
    "ScenarioEvaluation",
    "TriggerAudit",
    "TriggerHandler",
    "jsonable",
]
