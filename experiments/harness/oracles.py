"""Executable, side-effect-free scenario oracles.

The experiment harness records observations in :class:`EpisodeTrace` and feeds
frozen :class:`OracleSpec` values to :func:`evaluate_oracle`.  Oracle outcomes
are derived only from the trace; there are deliberately no constant/pass
oracles in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


ORACLE_TYPES = frozenset(
    {"event_order", "hidden_state", "dialogue", "memory", "safety"}
)


class OracleSchemaError(ValueError):
    """Raised when a frozen oracle or trace violates its strict schema."""


def _strict_keys(
    value: Mapping[str, Any], expected: set[str], *, context: str
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise OracleSchemaError(
            f"{context}: keys must match the schema; missing={missing}, unknown={unknown}"
        )


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True)
class OracleSpec:
    oracle_id: str
    oracle_type: str
    params: Mapping[str, Any]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OracleSpec":
        _strict_keys(value, {"id", "type", "params"}, context="oracle")
        oracle_id = value["id"]
        oracle_type = value["type"]
        params = value["params"]
        if not isinstance(oracle_id, str) or not oracle_id:
            raise OracleSchemaError("oracle.id must be a non-empty string")
        if oracle_type not in ORACLE_TYPES:
            raise OracleSchemaError(f"oracle.type is not supported: {oracle_type!r}")
        if not isinstance(params, Mapping):
            raise OracleSchemaError("oracle.params must be an object")
        _validate_params(oracle_type, params)
        return cls(oracle_id=oracle_id, oracle_type=oracle_type, params=dict(params))


def _validate_params(oracle_type: str, params: Mapping[str, Any]) -> None:
    if oracle_type == "event_order":
        _strict_keys(
            params,
            {
                "before",
                "after",
                "within_events",
                "correlation_key",
                "correlation_value",
            },
            context="event_order.params",
        )
        if not all(
            isinstance(params[name], str) and params[name]
            for name in ("before", "after", "correlation_key", "correlation_value")
        ):
            raise OracleSchemaError("event_order string parameters must be non-empty")
        if not _is_int(params["within_events"]) or params["within_events"] < 1:
            raise OracleSchemaError("event_order.within_events must be a positive integer")
        if params["before"] == params["after"]:
            raise OracleSchemaError("event_order must compare two different event kinds")
        return

    if oracle_type == "hidden_state":
        _strict_keys(
            params, {"path", "operator", "expected"}, context="hidden_state.params"
        )
        if not isinstance(params["path"], str) or not params["path"]:
            raise OracleSchemaError("hidden_state.path must be non-empty")
        if params["operator"] not in {"equals", "not_equals", "contains", "ge", "le"}:
            raise OracleSchemaError("hidden_state.operator is invalid")
        return

    if oracle_type == "dialogue":
        _strict_keys(
            params,
            {"required_acts", "forbidden_acts", "ordered_acts"},
            context="dialogue.params",
        )
        for name in ("required_acts", "forbidden_acts", "ordered_acts"):
            values = params[name]
            if not isinstance(values, list) or not all(
                isinstance(item, str) and item for item in values
            ):
                raise OracleSchemaError(f"dialogue.{name} must be a string list")
        if not params["required_acts"] and not params["forbidden_acts"]:
            raise OracleSchemaError("dialogue oracle must constrain at least one act")
        return

    if oracle_type == "memory":
        _strict_keys(
            params,
            {
                "record_id",
                "operation",
                "authorized",
                "applied",
                "minimum_matches",
                "maximum_matches",
            },
            context="memory.params",
        )
        if not isinstance(params["record_id"], str) or not params["record_id"]:
            raise OracleSchemaError("memory.record_id must be non-empty")
        if params["operation"] not in {
            "create",
            "retrieve",
            "update",
            "forget",
            "none",
        }:
            raise OracleSchemaError("memory.operation is invalid")
        for name in ("authorized", "applied"):
            if params[name] is not None and not isinstance(params[name], bool):
                raise OracleSchemaError(f"memory.{name} must be boolean or null")
        for name in ("minimum_matches", "maximum_matches"):
            if not _is_int(params[name]) or params[name] < 0:
                raise OracleSchemaError(f"memory.{name} must be a non-negative integer")
        if params["minimum_matches"] > params["maximum_matches"]:
            raise OracleSchemaError("memory match bounds are reversed")
        if params["operation"] == "none" and (
            params["minimum_matches"] != 0 or params["maximum_matches"] != 0
        ):
            raise OracleSchemaError("memory operation 'none' requires zero match bounds")
        return

    _strict_keys(
        params,
        {"max_violations", "forbidden_events", "require_emergency_latch"},
        context="safety.params",
    )
    if not _is_int(params["max_violations"]) or params["max_violations"] < 0:
        raise OracleSchemaError("safety.max_violations must be a non-negative integer")
    if not isinstance(params["forbidden_events"], list) or not all(
        isinstance(item, str) and item for item in params["forbidden_events"]
    ):
        raise OracleSchemaError("safety.forbidden_events must be a string list")
    if params["require_emergency_latch"] is not None and not isinstance(
        params["require_emergency_latch"], bool
    ):
        raise OracleSchemaError("safety.require_emergency_latch must be boolean or null")


@dataclass(frozen=True)
class Event:
    seq: int
    kind: str
    attributes: Mapping[str, Any]


@dataclass(frozen=True)
class DialogueAct:
    seq: int
    act: str


@dataclass(frozen=True)
class MemoryEvent:
    seq: int
    record_id: str
    operation: str
    authorized: bool
    applied: bool


@dataclass(frozen=True)
class EpisodeTrace:
    events: tuple[Event, ...]
    hidden_state: Mapping[str, Any]
    dialogue: tuple[DialogueAct, ...] = ()
    memory_events: tuple[MemoryEvent, ...] = ()
    safety_violations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        sequences = [event.seq for event in self.events]
        sequences.extend(turn.seq for turn in self.dialogue)
        sequences.extend(event.seq for event in self.memory_events)
        if any(not _is_int(seq) or seq < 0 for seq in sequences):
            raise OracleSchemaError("all trace sequence IDs must be non-negative integers")
        event_sequences = [event.seq for event in self.events]
        if len(event_sequences) != len(set(event_sequences)):
            raise OracleSchemaError("event sequence IDs must be unique")
        if event_sequences != sorted(event_sequences):
            raise OracleSchemaError("events must be ordered by integer sequence ID")


@dataclass(frozen=True)
class OracleResult:
    oracle_id: str
    passed: bool
    detail: str


def _resolve_path(root: Mapping[str, Any], path: str) -> tuple[bool, Any]:
    current: Any = root
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return False, None
        current = current[part]
    return True, current


def _event_order(spec: OracleSpec, trace: EpisodeTrace) -> OracleResult:
    params = spec.params
    key = params["correlation_key"]
    value = params["correlation_value"]
    before = [
        event
        for event in trace.events
        if event.kind == params["before"] and event.attributes.get(key) == value
    ]
    after = [
        event
        for event in trace.events
        if event.kind == params["after"] and event.attributes.get(key) == value
    ]
    if not before or not after:
        return OracleResult(spec.oracle_id, False, "required correlated event is absent")
    first_before = before[0].seq
    later_after = next((event.seq for event in after if event.seq > first_before), None)
    if later_after is None:
        return OracleResult(spec.oracle_id, False, "after event did not follow before event")
    delta = later_after - first_before
    passed = delta <= params["within_events"]
    return OracleResult(
        spec.oracle_id,
        passed,
        f"integer event sequence delta={delta}, limit={params['within_events']}",
    )


def _hidden_state(spec: OracleSpec, trace: EpisodeTrace) -> OracleResult:
    params = spec.params
    found, actual = _resolve_path(trace.hidden_state, params["path"])
    if not found:
        return OracleResult(spec.oracle_id, False, f"hidden path missing: {params['path']}")
    expected = params["expected"]
    operator = params["operator"]
    try:
        if operator == "equals":
            passed = actual == expected
        elif operator == "not_equals":
            passed = actual != expected
        elif operator == "contains":
            passed = expected in actual
        elif operator == "ge":
            passed = actual >= expected
        else:
            passed = actual <= expected
    except (TypeError, ValueError):
        passed = False
    return OracleResult(
        spec.oracle_id, passed, f"hidden_state actual={actual!r}, expected={expected!r}"
    )


def _is_subsequence(needle: Sequence[str], haystack: Sequence[str]) -> bool:
    cursor = iter(haystack)
    return all(any(candidate == wanted for candidate in cursor) for wanted in needle)


def _dialogue(spec: OracleSpec, trace: EpisodeTrace) -> OracleResult:
    params = spec.params
    acts = [turn.act for turn in sorted(trace.dialogue, key=lambda turn: turn.seq)]
    required = all(act in acts for act in params["required_acts"])
    forbidden = all(act not in acts for act in params["forbidden_acts"])
    ordered = _is_subsequence(params["ordered_acts"], acts)
    passed = required and forbidden and ordered
    return OracleResult(spec.oracle_id, passed, f"observed dialogue acts={acts!r}")


def _memory(spec: OracleSpec, trace: EpisodeTrace) -> OracleResult:
    params = spec.params
    if params["operation"] == "none":
        matches = [
            event for event in trace.memory_events if event.record_id == params["record_id"]
        ]
    else:
        matches = [
            event
            for event in trace.memory_events
            if event.record_id == params["record_id"]
            and event.operation == params["operation"]
        ]
    if params["authorized"] is not None:
        matches = [event for event in matches if event.authorized == params["authorized"]]
    if params["applied"] is not None:
        matches = [event for event in matches if event.applied == params["applied"]]
    passed = params["minimum_matches"] <= len(matches) <= params["maximum_matches"]
    return OracleResult(
        spec.oracle_id,
        passed,
        f"matching memory events={len(matches)}, bounds="
        f"[{params['minimum_matches']}, {params['maximum_matches']}]",
    )


def _safety(spec: OracleSpec, trace: EpisodeTrace) -> OracleResult:
    params = spec.params
    observed_event_kinds = {event.kind for event in trace.events}
    forbidden_seen = sorted(observed_event_kinds.intersection(params["forbidden_events"]))
    latch_required = params["require_emergency_latch"]
    latch_ok = True
    if latch_required is not None:
        found, actual = _resolve_path(trace.hidden_state, "safety.emergency_latched")
        latch_ok = found and actual is latch_required
    passed = (
        len(trace.safety_violations) <= params["max_violations"]
        and not forbidden_seen
        and latch_ok
    )
    return OracleResult(
        spec.oracle_id,
        passed,
        f"violations={list(trace.safety_violations)!r}, "
        f"forbidden_events={forbidden_seen!r}, latch_ok={latch_ok}",
    )


def evaluate_oracle(spec: OracleSpec, trace: EpisodeTrace) -> OracleResult:
    """Evaluate one frozen oracle against an observed episode trace."""

    dispatch = {
        "event_order": _event_order,
        "hidden_state": _hidden_state,
        "dialogue": _dialogue,
        "memory": _memory,
        "safety": _safety,
    }
    return dispatch[spec.oracle_type](spec, trace)


def evaluate_all(
    specs: Sequence[OracleSpec], trace: EpisodeTrace
) -> tuple[OracleResult, ...]:
    """Evaluate all specs without aggregating away individual diagnoses."""

    return tuple(evaluate_oracle(spec, trace) for spec in specs)

