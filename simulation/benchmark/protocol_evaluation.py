"""Production-facing batch evaluation for PrefMem memory protocols.

This module deliberately keeps benchmark oracle data outside model context.  It
loads the semantic protocol bundle, resolves each scenario selector to an opaque
episode path, creates isolated memory stores for every protocol repetition, and
delegates protocol semantics to :func:`run_memory_protocol`.

It is an API module rather than a command-line entry point so experiments can
inject model/orchestrator factories while retaining one implementation of
isolation, fixture consent, scenario selection, and result persistence.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import re
import tempfile
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from agents.configs import PrefMemConfig
from dataset.benchmark import BenchmarkEpisode
from memory.models import ConsentEvidence

from .ablations import MEMORY_MODES, MemoryAblationAgent
from .evaluation import _agent_failure_evidence
from .executor import BenchmarkEpisodeExecutor
from .provenance import (
    benchmark_content_sha256,
    callable_provenance,
    runtime_provenance,
)
from .protocol_runner import (
    MANDATORY_MEMORY_CONTEXT_EXPECTATIONS,
    SUPPORTED_EXPECTATIONS,
    ProtocolReport,
    _has_task_evidence,
    _score_task_result,
    run_memory_protocol,
)
from .validator import validate_benchmark


PROTOCOL_SCHEMA_VERSION = "robopref.memory-protocols.v1"
EVALUATION_SCHEMA_VERSION = "robopref.memory-protocol-evaluation.v1"
_SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9_.-]+")
_SUPPORTED_SELECTOR_KEYS = frozenset(
    {
        "control_kind",
        "counterfactual_group_id",
        "family",
        "outcome",
        "scenario_id",
        "scene_variant",
        "seed",
        "target_id",
        "variant",
    }
)
_PROTOCOL_TASK_SCORE_CHECKS = frozenset(
    {
        "hri_resolved_target",
        "planner_confirmed_intent",
        "planner_goal_predicates",
        "hri_response_mode",
        "hri_terminal_outcome",
        "validator_called",
        "validator_outcome",
        "task_complete",
        "planner_status",
        "execution_status",
        "vla_dispatches",
        "recovery",
        "history_delta",
        "preference_delta_without_consent",
    }
)
_MANDATORY_PROTOCOL_CHECKS = frozenset(
    {
        "memory_context_ownership_verified",
        "foreign_history_refs",
        "foreign_preference_refs",
    }
)
_BOOLEAN_EXPECTATIONS = frozenset(
    {
        "active_rgb_preference_preserved",
        "clarification_required",
        "history_retrieval_required",
        "memory_context_ownership_verified",
        "pending_question_cleared",
        "post_task_preference_question",
        "preference_retrieval_required",
        "task_semantic_score",
        "uses_one_off_override",
    }
)
_NONNEGATIVE_INTEGER_EXPECTATIONS = frozenset(
    {
        "active_equivalent_preferences",
        "dispatches",
        "foreign_history_refs",
        "foreign_preference_refs",
        "min_dispatches",
    }
)
_SIGNED_INTEGER_EXPECTATIONS = frozenset(
    {
        "history_delta",
        "preference_delta",
    }
)
_NONEMPTY_STRING_EXPECTATIONS = frozenset(
    {
        "planner_status",
        "validator_outcome",
    }
)
_EXECUTION_INDICATING_EXPECTATIONS = frozenset(
    {
        "dispatches",
        "min_dispatches",
        "planner_status",
        "post_task_preference_question",
        "uses_one_off_override",
        "validator_outcome",
    }
)
_PROTOCOL_BOOTSTRAP_REPLICATES = 2_000
_PROTOCOL_BOOTSTRAP_SEED = 20_260_725
_PROTOCOL_CLUSTER_UNIT = (
    "core:family_scene_variant_seed;"
    "control:family_scene_variant_seed_target;"
    "counterfactual_group_fallback"
)


class ProtocolEvaluationError(ValueError):
    """Raised when a batch cannot be set up without compromising its validity."""


@dataclass(frozen=True, slots=True)
class EpisodeRecord:
    path: Path
    scenario_id: str
    family: str
    variant: str
    seed: int
    counterfactual_group_id: str
    target_id: str
    outcome: str
    control_kind: str | None

    def selector_value(self, key: str) -> Any:
        if key == "scene_variant":
            key = "variant"
        return getattr(self, key)

    def public_selection(self) -> dict[str, Any]:
        """Return oracle-side provenance for the result log, never model input."""

        return {
            "scenario_id": self.scenario_id,
            "family": self.family,
            "scene_variant": self.variant,
            "seed": self.seed,
            "counterfactual_group_id": self.counterfactual_group_id,
            "target_id": self.target_id,
            "outcome": self.outcome,
            "control_kind": self.control_kind,
        }


@dataclass(frozen=True, slots=True)
class ProtocolRunContext:
    protocol: dict[str, Any]
    repetition: int
    run_directory: Path
    initial_episode_path: Path
    config: PrefMemConfig
    memory_mode: str
    model_seed: int | None


@dataclass(frozen=True, slots=True)
class ProtocolBatchResult:
    output_directory: Path
    results_path: Path
    summary_path: Path
    planned_runs: int
    executed_runs: int
    skipped_runs: int
    summary: dict[str, Any]


OrchestratorFactory = Callable[[ProtocolRunContext], Any]
FixtureApplier = Callable[[str, Mapping[str, Any], Any], None]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_component(value: str) -> str:
    safe = _SAFE_COMPONENT.sub("-", value.strip()).strip("-.")
    if not safe:
        raise ProtocolEvaluationError("Protocol IDs must contain a usable path component.")
    return safe


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _read_json_object(path: Path, *, description: str) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise ProtocolEvaluationError(
            f"{description} is not readable JSON: {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise ProtocolEvaluationError(f"{description} must contain a JSON object: {path}")
    return value


def load_protocol_bundle(path: str | Path) -> dict[str, Any]:
    """Load and structurally validate a generated memory-protocol bundle."""

    protocol_path = Path(path).expanduser().resolve()
    bundle = _read_json_object(protocol_path, description="Protocol bundle")
    if bundle.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        raise ProtocolEvaluationError(
            "Unsupported memory protocol schema: "
            f"{bundle.get('schema_version')!r}"
        )
    protocols = bundle.get("protocols")
    fixtures = bundle.get("fixtures", {})
    if not isinstance(protocols, list) or not protocols:
        raise ProtocolEvaluationError("Protocol bundle requires a non-empty protocols list.")
    if not isinstance(fixtures, dict):
        raise ProtocolEvaluationError("Protocol bundle fixtures must be an object.")

    seen: set[str] = set()
    for protocol in protocols:
        if not isinstance(protocol, dict):
            raise ProtocolEvaluationError("Every protocol must be an object.")
        protocol_id = str(protocol.get("protocol_id", "")).strip()
        if not protocol_id or protocol_id in seen:
            raise ProtocolEvaluationError(
                "Protocol IDs must be non-empty and unique."
            )
        _safe_component(protocol_id)
        seen.add(protocol_id)
        user_id = str(protocol.get("user_id", "")).strip()
        if not user_id:
            raise ProtocolEvaluationError(
                f"Protocol {protocol_id!r} requires a user_id."
            )
        requires_fresh = protocol.get("requires_fresh_memory", True)
        if not isinstance(requires_fresh, bool):
            raise ProtocolEvaluationError(
                f"Protocol {protocol_id!r} requires_fresh_memory must be bool."
            )
        fixture_id = protocol.get("initial_memory_fixture")
        if fixture_id is not None and str(fixture_id) not in fixtures:
            raise ProtocolEvaluationError(
                f"Protocol {protocol_id!r} references unknown fixture "
                f"{fixture_id!r}."
            )
        steps = protocol.get("steps")
        if not isinstance(steps, list) or not steps:
            raise ProtocolEvaluationError(
                f"Protocol {protocol_id!r} requires non-empty steps."
            )
        has_selector = False
        for index, step in enumerate(steps, start=1):
            if not isinstance(step, dict):
                raise ProtocolEvaluationError(
                    f"Protocol {protocol_id!r} step {index} must be an object."
                )
            query = str(step.get("query", "")).strip()
            reply = str(step.get("reply", "")).strip()
            if bool(query) == bool(reply):
                raise ProtocolEvaluationError(
                    f"Protocol {protocol_id!r} step {index} must contain "
                    "exactly one query or reply."
                )
            selector = step.get("scenario_selector")
            if selector is not None:
                if not isinstance(selector, dict) or not selector:
                    raise ProtocolEvaluationError(
                        f"Protocol {protocol_id!r} step {index} selector "
                        "must be a non-empty object."
                    )
                has_selector = True
            expected = step.get("expected")
            if not isinstance(expected, dict) or not expected:
                raise ProtocolEvaluationError(
                    f"Protocol {protocol_id!r} step {index} requires expected."
                )
            unknown = set(expected) - SUPPORTED_EXPECTATIONS
            if unknown:
                raise ProtocolEvaluationError(
                    f"Protocol {protocol_id!r} step {index} has unsupported "
                    f"expectations: {sorted(unknown)}"
                )
            for name, required in MANDATORY_MEMORY_CONTEXT_EXPECTATIONS.items():
                if name in expected and expected[name] != required:
                    raise ProtocolEvaluationError(
                        f"Protocol {protocol_id!r} step {index} weakens "
                        f"mandatory expectation {name!r}."
                    )
            for name in _BOOLEAN_EXPECTATIONS & set(expected):
                if not isinstance(expected[name], bool):
                    raise ProtocolEvaluationError(
                        f"{name} must be a bool."
                    )
            for name in _NONNEGATIVE_INTEGER_EXPECTATIONS & set(expected):
                value = expected[name]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                ):
                    raise ProtocolEvaluationError(
                        f"{name} must be non-negative integer."
                    )
            for name in _SIGNED_INTEGER_EXPECTATIONS & set(expected):
                value = expected[name]
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ProtocolEvaluationError(
                        f"{name} must be an integer."
                    )
            for name in _NONEMPTY_STRING_EXPECTATIONS & set(expected):
                if (
                    not isinstance(expected[name], str)
                    or not expected[name].strip()
                ):
                    raise ProtocolEvaluationError(
                        f"{name} must be a non-empty string."
                    )
            if "task_semantic_score" in expected and not has_selector:
                raise ProtocolEvaluationError(
                    f"Protocol {protocol_id!r} step {index} declares "
                    "task_semantic_score before any scenario selector."
                )
            execution_indicators = (
                _EXECUTION_INDICATING_EXPECTATIONS & set(expected)
            )
            if (
                execution_indicators
                and expected.get("task_semantic_score") is not True
            ):
                raise ProtocolEvaluationError(
                    f"Protocol {protocol_id!r} step {index} uses "
                    f"execution-indicating expectations "
                    f"{sorted(execution_indicators)} without "
                    "task_semantic_score=true."
                )
        if not has_selector:
            raise ProtocolEvaluationError(
                f"Protocol {protocol_id!r} requires a scenario selector."
            )

    for fixture_id, fixture in fixtures.items():
        if not isinstance(fixture_id, str) or not fixture_id.strip():
            raise ProtocolEvaluationError(
                "Fixture IDs must be non-empty strings."
            )
        if not isinstance(fixture, dict):
            raise ProtocolEvaluationError(
                f"Fixture {fixture_id!r} must be an object."
            )
        if fixture.get("fixture_kind") != "approved_preference_request":
            raise ProtocolEvaluationError(
                f"Fixture {fixture_id!r} has an unsupported kind."
            )
        request = fixture.get("preference_request")
        consent = fixture.get("consent")
        fixture_user = str(fixture.get("user_id", "")).strip()
        if (
            not isinstance(request, dict)
            or not isinstance(consent, dict)
            or not fixture_user
            or str(request.get("user_id", "")).strip() != fixture_user
            or consent.get("proposal") != request
            or consent.get("authorized_action")
            != str(request.get("requested_action", "")).upper()
        ):
            raise ProtocolEvaluationError(
                f"Fixture {fixture_id!r} has invalid user/consent binding."
            )
        try:
            ConsentEvidence(**copy.deepcopy(consent)).validate()
        except (TypeError, ValueError) as error:
            raise ProtocolEvaluationError(
                f"Fixture {fixture_id!r} has invalid consent: {error}"
            ) from error
    return copy.deepcopy(bundle)


class EpisodeCatalog:
    """Immutable deterministic selector over benchmark episode manifests."""

    def __init__(self, records: Sequence[EpisodeRecord]):
        ordered = sorted(
            records,
            key=lambda item: (
                item.family,
                item.target_id,
                item.outcome,
                item.control_kind or "",
                item.variant,
                item.seed,
                item.scenario_id,
            ),
        )
        if not ordered:
            raise ProtocolEvaluationError("Benchmark contains no episode manifests.")
        scenario_ids = [item.scenario_id for item in ordered]
        if len(scenario_ids) != len(set(scenario_ids)):
            raise ProtocolEvaluationError("Benchmark contains duplicate scenario IDs.")
        self.records = tuple(ordered)

    @classmethod
    def from_benchmark(cls, benchmark_root: str | Path) -> EpisodeCatalog:
        root = Path(benchmark_root).expanduser().resolve()
        index = _read_json_object(root / "index.json", description="Benchmark index")
        scenario_ids = index.get("scenario_ids")
        if not isinstance(scenario_ids, list) or not scenario_ids:
            raise ProtocolEvaluationError(
                "Benchmark index requires a non-empty scenario_ids list."
            )
        if any(not isinstance(item, str) or not item for item in scenario_ids):
            raise ProtocolEvaluationError("Benchmark scenario IDs must be non-empty strings.")
        if len(scenario_ids) != len(set(scenario_ids)):
            raise ProtocolEvaluationError("Benchmark index contains duplicate scenario IDs.")

        records: list[EpisodeRecord] = []
        for scenario_id in sorted(scenario_ids):
            episode_path = root / "episodes" / scenario_id
            manifest = _read_json_object(
                episode_path / "manifest.json",
                description=f"Manifest for {scenario_id}",
            )
            if manifest.get("scenario_id") != scenario_id:
                raise ProtocolEvaluationError(
                    f"Manifest scenario ID does not match its index entry: {scenario_id}"
                )
            scene = manifest.get("scene")
            target = manifest.get("target")
            if not isinstance(scene, Mapping) or not isinstance(target, Mapping):
                raise ProtocolEvaluationError(
                    f"Manifest scene and target must be objects: {scenario_id}"
                )
            seed = scene.get("seed")
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise ProtocolEvaluationError(
                    f"Manifest scene seed must be an integer: {scenario_id}"
                )
            control = scene.get("control_kind")
            records.append(
                EpisodeRecord(
                    path=episode_path.resolve(),
                    scenario_id=scenario_id,
                    family=str(scene.get("family", "")),
                    variant=str(scene.get("variant", "")),
                    seed=seed,
                    counterfactual_group_id=str(
                        scene.get("counterfactual_group_id", "")
                    ),
                    target_id=str(target.get("target_id", "")),
                    outcome=str(manifest.get("expected_outcome", "")),
                    control_kind=None if control is None else str(control),
                )
            )
        return cls(records)

    def compatible(self, selector: Mapping[str, Any]) -> tuple[EpisodeRecord, ...]:
        unknown = set(map(str, selector)) - _SUPPORTED_SELECTOR_KEYS
        if unknown:
            raise ProtocolEvaluationError(
                f"Unsupported scenario selector keys: {sorted(unknown)}"
            )
        if not selector:
            raise ProtocolEvaluationError("Scenario selectors cannot be empty.")

        matches: list[EpisodeRecord] = []
        for record in self.records:
            if all(
                record.selector_value(str(key)) == expected
                for key, expected in selector.items()
            ):
                matches.append(record)
        if not matches:
            raise ProtocolEvaluationError(
                "No benchmark episode matches selector "
                f"{json.dumps(dict(selector), sort_keys=True, default=str)}"
            )
        return tuple(
            sorted(
                matches,
                key=lambda item: (
                    item.variant,
                    item.seed,
                    item.scenario_id,
                ),
            )
        )


class RepetitionScenarioResolver:
    """Resolve a selector stably and rotate repetitions across all matches."""

    def __init__(self, catalog: EpisodeCatalog, repetition: int):
        if repetition <= 0:
            raise ProtocolEvaluationError("Repetitions are one-indexed and must be positive.")
        self.catalog = catalog
        self.repetition = repetition
        self._cache: dict[str, EpisodeRecord] = {}
        self.selections: list[dict[str, Any]] = []

    def __call__(self, selector: Mapping[str, Any]) -> Path:
        canonical = json.dumps(dict(selector), sort_keys=True, separators=(",", ":"))
        selected = self._cache.get(canonical)
        if selected is None:
            candidates = _balanced_candidate_order(
                self.catalog.compatible(selector)
            )
            candidate_index = (self.repetition - 1) % len(candidates)
            selected = candidates[candidate_index]
            self._cache[canonical] = selected
            self.selections.append(
                {
                    "selector": copy.deepcopy(dict(selector)),
                    "candidate_count": len(candidates),
                    "candidate_index": candidate_index,
                    **selected.public_selection(),
                }
            )
        return selected.path


def _balanced_candidate_order(
    candidates: Sequence[EpisodeRecord],
) -> tuple[EpisodeRecord, ...]:
    """Interleave variants and rotate seeds for balanced early repetitions."""

    groups: dict[str, list[EpisodeRecord]] = {}
    for candidate in candidates:
        groups.setdefault(candidate.variant, []).append(candidate)
    ordered_groups = [
        sorted(
            group,
            key=lambda item: (item.seed, item.scenario_id),
        )
        for _, group in sorted(groups.items())
    ]
    schedule: list[EpisodeRecord] = []
    maximum = max(map(len, ordered_groups), default=0)
    for round_index in range(maximum):
        for group_index, group in enumerate(ordered_groups):
            if round_index >= len(group):
                continue
            schedule.append(
                group[(round_index + group_index) % len(group)]
            )
    if len(schedule) != len(candidates) or len(
        {item.scenario_id for item in schedule}
    ) != len(candidates):
        raise ProtocolEvaluationError(
            "Balanced selector schedule did not preserve every candidate once."
        )
    return tuple(schedule)


def _preflight_selection_plans(
    protocols: Sequence[Mapping[str, Any]],
    *,
    catalog: EpisodeCatalog,
    repetitions: int,
) -> dict[tuple[str, int], list[dict[str, Any]]]:
    """Resolve the full selector schedule and reject physical-scene drift."""

    by_path = {record.path.resolve(): record for record in catalog.records}
    plans: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for protocol in protocols:
        protocol_id = str(protocol.get("protocol_id", ""))
        for repetition in range(1, repetitions + 1):
            resolver = RepetitionScenarioResolver(catalog, repetition)
            physical_scenes: set[tuple[str, ...]] = set()
            for step in protocol.get("steps", []):
                selector = (
                    step.get("scenario_selector")
                    if isinstance(step, Mapping)
                    else None
                )
                if not isinstance(selector, Mapping):
                    continue
                selected_path = resolver(selector).resolve()
                selected = by_path.get(selected_path)
                if selected is None:
                    raise ProtocolEvaluationError(
                        "Preflight selector resolved outside the benchmark "
                        f"catalog: {selected_path}"
                    )
                physical_scenes.add(
                    (
                        "family_scene_variant_seed",
                        selected.family,
                        selected.variant,
                        str(selected.seed),
                    )
                )
            if len(physical_scenes) > 1:
                raise ProtocolEvaluationError(
                    f"Protocol {protocol_id!r} repetition {repetition} "
                    "selectors drift across physical scenes: "
                    f"{sorted(physical_scenes)}"
                )
            plans[_run_key(protocol_id, repetition)] = copy.deepcopy(
                resolver.selections
            )
    return plans


def _first_selector(protocol: Mapping[str, Any]) -> Mapping[str, Any]:
    steps = protocol.get("steps")
    if not isinstance(steps, list):
        raise ProtocolEvaluationError("Protocol steps must be a list.")
    for step in steps:
        if isinstance(step, Mapping) and isinstance(
            step.get("scenario_selector"), Mapping
        ):
            return step["scenario_selector"]
    raise ProtocolEvaluationError(
        f"Protocol {protocol.get('protocol_id')!r} has no scenario selector."
    )


def apply_consent_fixture(
    fixture_id: str,
    fixture: Mapping[str, Any],
    orchestrator: Any,
) -> None:
    """Apply a named fixture through MemoryAgent's consent-gated update API."""

    if fixture.get("fixture_kind") != "approved_preference_request":
        raise ProtocolEvaluationError(
            f"Unsupported fixture kind for {fixture_id!r}: "
            f"{fixture.get('fixture_kind')!r}"
        )
    request = fixture.get("preference_request")
    raw_consent = fixture.get("consent")
    if not isinstance(request, dict) or not isinstance(raw_consent, dict):
        raise ProtocolEvaluationError(
            f"Fixture {fixture_id!r} requires preference_request and consent objects."
        )
    fixture_user = str(fixture.get("user_id", "")).strip()
    request_user = str(request.get("user_id", "")).strip()
    if not fixture_user or request_user != fixture_user:
        raise ProtocolEvaluationError(
            f"Fixture {fixture_id!r} user does not match its preference request."
        )
    if raw_consent.get("proposal") != request:
        raise ProtocolEvaluationError(
            f"Fixture {fixture_id!r} consent is not bound to the exact request."
        )
    requested_action = str(request.get("requested_action", "")).upper()
    if raw_consent.get("authorized_action") != requested_action:
        raise ProtocolEvaluationError(
            f"Fixture {fixture_id!r} consent action does not match the request."
        )
    try:
        consent = ConsentEvidence(**copy.deepcopy(raw_consent))
        consent.validate()
    except (TypeError, ValueError) as error:
        raise ProtocolEvaluationError(
            f"Fixture {fixture_id!r} contains invalid consent: {error}"
        ) from error

    memory_agent = getattr(orchestrator, "memory_agent", None)
    # Fixtures define the identical starting state for paired ablations.  Seed
    # that state through the real consent-aware agent, then let the adapter hide
    # or disable it during protocol turns according to the selected condition.
    if isinstance(memory_agent, MemoryAblationAgent):
        memory_agent = memory_agent.agent
    update = getattr(memory_agent, "update_preference_memory", None)
    if not callable(update):
        raise ProtocolEvaluationError(
            "Fixture application requires MemoryAgent.update_preference_memory."
        )
    transaction_digest = hashlib.sha256(
        json.dumps(
            {"fixture_id": fixture_id, "fixture": fixture},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()[:24]
    result = update(
        copy.deepcopy(request),
        consent=consent,
        transaction_id=f"fixture-{transaction_digest}",
    )
    if not isinstance(result, Mapping) or not isinstance(
        result.get("transaction"), Mapping
    ):
        raise ProtocolEvaluationError(
            f"Fixture {fixture_id!r} did not commit a preference transaction."
        )


def default_orchestrator_factory(context: ProtocolRunContext) -> Any:
    """Construct telemetry-enabled production PrefMem for one protocol run."""

    from .evaluation_runtime import build_evaluation_orchestrator

    benchmark_episode = BenchmarkEpisode.from_path(
        context.initial_episode_path,
        context.config.workspace_root,
    )
    executor = BenchmarkEpisodeExecutor(benchmark_episode)
    return build_evaluation_orchestrator(
        context.config,
        executor,
        context.run_directory,
        memory_mode=context.memory_mode,
    )


def _isolated_config(
    base_config: PrefMemConfig,
    *,
    user_id: str,
    initial_episode: Path,
    run_directory: Path,
    model_seed: int | None,
) -> PrefMemConfig:
    config = copy.deepcopy(base_config)
    memory_root = run_directory / "memory"
    config.user_id = user_id
    config.dataset_path = str(initial_episode)
    config.history_store_path = str(memory_root / "history.json")
    config.history_outbox_path = str(memory_root / "history_outbox.json")
    config.preference_store_path = str(memory_root / "preferences.json")
    if model_seed is not None:
        for name in ("hri", "memory", "planner", "validator"):
            getattr(config, name).seed = model_seed
    for value in (
        config.history_store_path,
        config.history_outbox_path,
        config.preference_store_path,
    ):
        if not _inside(Path(value), run_directory):
            raise ProtocolEvaluationError(
                "Evaluation memory stores must remain inside their isolated run directory."
            )
    return config


def _protocol_report_dict(report: ProtocolReport) -> dict[str, Any]:
    return {
        "protocol_id": report.protocol_id,
        "passed": report.passed,
        "steps": [
            {
                "index": step.index,
                "message": step.message,
                "passed": step.passed,
                "history_delta": step.history_delta,
                "preference_delta": step.preference_delta,
                "delivered_memory_context": copy.deepcopy(
                    step.delivered_memory_context
                ),
                "checks": [copy.deepcopy(check) for check in step.checks],
                "task_score_required": step.task_score_required,
                "task_score": copy.deepcopy(step.task_score),
                "result": copy.deepcopy(step.result),
            }
            for step in report.steps
        ],
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return asdict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    raise TypeError(f"Value of type {type(value).__name__} is not JSON serializable.")


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        json.dump(
            value,
            stream,
            ensure_ascii=False,
            sort_keys=True,
            default=_json_default,
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ProtocolEvaluationError(
                    f"Invalid protocol result JSONL at {path}:{line_number}: {error}"
                ) from error
            if not isinstance(value, dict):
                raise ProtocolEvaluationError(
                    f"Protocol result at {path}:{line_number} is not an object."
                )
            records.append(value)
    return records


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            json.dump(
                value,
                stream,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
                default=_json_default,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            temporary = stream.name
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_key(protocol_id: str, repetition: int) -> tuple[str, int]:
    return protocol_id, repetition


def _record_key(record: Mapping[str, Any]) -> tuple[str, int]:
    protocol_id = record.get("protocol_id")
    repetition = record.get("repetition")
    if (
        not isinstance(protocol_id, str)
        or not protocol_id
        or isinstance(repetition, bool)
        or not isinstance(repetition, int)
        or repetition <= 0
    ):
        raise ProtocolEvaluationError(
            "Every protocol JSONL resume key requires a non-empty string "
            "protocol_id and a positive integer repetition."
        )
    return _run_key(protocol_id, repetition)


def _check_evidence_passes(
    check: Any,
    *,
    key: tuple[str, int],
    label: str,
) -> bool:
    if not isinstance(check, Mapping):
        raise ProtocolEvaluationError(
            f"Protocol resume record {key!r} has incomplete {label} evidence."
        )
    name = str(check.get("name", "")).strip()
    evaluated = check.get("evaluated")
    passed = check.get("passed")
    if (
        not name
        or not isinstance(evaluated, bool)
        or (
            evaluated
            and not isinstance(passed, bool)
        )
        or (
            not evaluated
            and passed is not None
        )
    ):
        raise ProtocolEvaluationError(
            f"Protocol resume record {key!r} has malformed {label} evidence."
        )
    return evaluated and passed is True


def _strict_scalar_equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left is right
    return type(left) is type(right) and left == right


def _validate_report_evidence(
    report: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any],
    catalog: EpisodeCatalog,
    key: tuple[str, int],
) -> bool:
    protocol_id = key[0]
    protocol_steps = protocol.get("steps")
    report_steps = report.get("steps")
    if (
        report.get("protocol_id") != protocol_id
        or not isinstance(protocol_steps, list)
        or not protocol_steps
        or not isinstance(report_steps, list)
        or len(report_steps) != len(protocol_steps)
        or not report_steps
        or not isinstance(report.get("passed"), bool)
    ):
        raise ProtocolEvaluationError(
            f"Protocol resume record {key!r} has incomplete report.steps "
            "evidence."
        )

    computed_step_passes: list[bool] = []
    resolver = RepetitionScenarioResolver(catalog, key[1])
    active_episode_path: Path | None = None
    for position, (protocol_step, step) in enumerate(
        zip(protocol_steps, report_steps, strict=True),
        start=1,
    ):
        if not isinstance(protocol_step, Mapping) or not isinstance(step, Mapping):
            raise ProtocolEvaluationError(
                f"Protocol resume record {key!r} has incomplete report step "
                f"{position} evidence."
            )
        selector = protocol_step.get("scenario_selector")
        if isinstance(selector, Mapping):
            active_episode_path = resolver(selector)
        expected_message = str(
            protocol_step.get("query") or protocol_step.get("reply") or ""
        ).strip()
        checks = step.get("checks")
        step_index = step.get("index")
        if (
            isinstance(step_index, bool)
            or not isinstance(step_index, int)
            or step_index != position
            or step.get("message") != expected_message
            or not isinstance(step.get("passed"), bool)
            or not isinstance(checks, list)
            or not checks
            or not isinstance(step.get("result"), Mapping)
        ):
            raise ProtocolEvaluationError(
                f"Protocol resume record {key!r} has incomplete report step "
                f"{position}/check evidence."
            )

        declared = protocol_step.get("expected")
        task_score_required = step.get("task_score_required")
        task_score = step.get("task_score")
        if not isinstance(task_score_required, bool):
            raise ProtocolEvaluationError(
                f"Protocol resume record {key!r} has incomplete task-score "
                f"evidence at step {position}."
            )
        computed_task_score_required = (
            isinstance(declared, Mapping)
            and declared.get("task_semantic_score") is True
        ) or _has_task_evidence(step["result"])
        if task_score_required is not computed_task_score_required:
            raise ProtocolEvaluationError(
                f"Protocol resume record {key!r} has inconsistent "
                f"task_score_required evidence at step {position}."
            )
        if (
            isinstance(declared, Mapping)
            and declared.get("task_semantic_score") is True
            and not task_score_required
        ):
            raise ProtocolEvaluationError(
                f"Protocol resume record {key!r} omits required task-score "
                f"evidence at step {position}."
            )

        effective_expected = (
            copy.deepcopy(dict(declared))
            if isinstance(declared, Mapping)
            else {}
        )
        effective_expected.update(MANDATORY_MEMORY_CONTEXT_EXPECTATIONS)
        if task_score_required:
            effective_expected["task_semantic_score"] = True

        check_names = [
            str(check.get("name", "")).strip()
            if isinstance(check, Mapping)
            else ""
            for check in checks
        ]
        if (
            len(check_names) != len(set(check_names))
            or set(check_names) != set(effective_expected)
        ):
            raise ProtocolEvaluationError(
                f"Protocol resume record {key!r} has incomplete, duplicate, "
                f"or unexpected report step {position}/check evidence."
            )

        protocol_checks_pass = True
        for check in checks:
            stored_pass = _check_evidence_passes(
                check,
                key=key,
                label=f"step {position} check",
            )
            name = str(check["name"]).strip()
            expected_value = effective_expected[name]
            if (
                "expected" not in check
                or "actual" not in check
                or not _strict_scalar_equal(
                    check["expected"],
                    expected_value,
                )
            ):
                raise ProtocolEvaluationError(
                    f"Protocol resume record {key!r} has inconsistent "
                    f"expected evidence for step {position} check {name!r}."
                )
            evaluated = check["evaluated"] is True
            actual = check["actual"]
            if not evaluated:
                if actual is not None:
                    raise ProtocolEvaluationError(
                        f"Protocol resume record {key!r} has malformed actual "
                        f"evidence for step {position} check {name!r}."
                    )
                recomputed_pass = False
            elif name == "min_dispatches":
                recomputed_pass = (
                    isinstance(actual, int)
                    and not isinstance(actual, bool)
                    and actual >= expected_value
                )
            else:
                recomputed_pass = _strict_scalar_equal(
                    actual,
                    expected_value,
                )
            if stored_pass != (evaluated and recomputed_pass):
                raise ProtocolEvaluationError(
                    f"Protocol resume record {key!r} has inconsistent passed "
                    f"evidence for step {position} check {name!r}."
                )
            protocol_checks_pass = (
                evaluated and recomputed_pass and protocol_checks_pass
            )

        task_score_pass = not task_score_required
        if task_score_required:
            raw_checks = (
                task_score.get("checks")
                if isinstance(task_score, Mapping)
                else None
            )
            semantic_checks = (
                task_score.get("semantic_checks")
                if isinstance(task_score, Mapping)
                else None
            )
            if (
                "task_semantic_score" not in check_names
                or not isinstance(task_score, Mapping)
                or not isinstance(task_score.get("passed"), bool)
                or not isinstance(task_score.get("semantic_passed"), bool)
                or not isinstance(task_score.get("full_passed"), bool)
                or not isinstance(raw_checks, list)
                or not raw_checks
                or not isinstance(semantic_checks, list)
                or not semantic_checks
            ):
                raise ProtocolEvaluationError(
                    f"Protocol resume record {key!r} has incomplete task-score "
                    f"evidence at step {position}."
                )
            raw_pass = True
            for check in raw_checks:
                raw_pass = (
                    _check_evidence_passes(
                        check,
                        key=key,
                        label=f"step {position} raw task-score check",
                    )
                    and raw_pass
                )
            semantic_pass = True
            for check in semantic_checks:
                semantic_pass = (
                    _check_evidence_passes(
                        check,
                        key=key,
                        label=f"step {position} semantic task-score check",
                    )
                    and semantic_pass
                )
            if (
                task_score["full_passed"] is not raw_pass
                or task_score["semantic_passed"] is not semantic_pass
                or task_score["passed"] is not semantic_pass
            ):
                raise ProtocolEvaluationError(
                    f"Protocol resume record {key!r} has inconsistent "
                    f"task-score passed evidence at step {position}."
                )
            if active_episode_path is None:
                raise ProtocolEvaluationError(
                    f"Protocol resume record {key!r} has task-score evidence "
                    f"without an active scenario at step {position}."
                )
            manifest = _read_json_object(
                active_episode_path / "manifest.json",
                description=(
                    f"Resume scoring manifest for {key!r} step {position}"
                ),
            )
            rescored = _score_task_result(
                manifest,
                step["result"],
                history_delta=step.get("history_delta"),
                preference_delta=step.get("preference_delta"),
            )
            if task_score != rescored:
                raise ProtocolEvaluationError(
                    f"Protocol resume record {key!r} has task-score evidence "
                    f"that does not match oracle re-scoring at step {position}."
                )
            task_score_pass = task_score["passed"] is True
        elif task_score is not None or "task_semantic_score" in check_names:
            raise ProtocolEvaluationError(
                f"Protocol resume record {key!r} has inconsistent task-score "
                f"evidence at step {position}."
            )

        computed_step_pass = protocol_checks_pass and task_score_pass
        if step["passed"] is not computed_step_pass:
            raise ProtocolEvaluationError(
                f"Protocol resume record {key!r} has inconsistent report step "
                f"{position} passed evidence."
            )
        computed_step_passes.append(computed_step_pass)

    computed_report_pass = bool(computed_step_passes) and all(
        computed_step_passes
    )
    if report["passed"] is not computed_report_pass:
        raise ProtocolEvaluationError(
            f"Protocol resume record {key!r} has inconsistent report passed "
            "evidence."
        )
    return computed_report_pass


def _selection_integer_types_are_strict(value: Any) -> bool:
    if not isinstance(value, list):
        return False
    for selection in value:
        if not isinstance(selection, Mapping):
            return False
        for name in ("candidate_count", "candidate_index", "seed"):
            number = selection.get(name)
            if isinstance(number, bool) or not isinstance(number, int):
                return False
        if (
            selection["candidate_count"] <= 0
            or selection["candidate_index"] < 0
            or selection["candidate_index"] >= selection["candidate_count"]
        ):
            return False
    return True


def _validate_resume_record(
    record: Mapping[str, Any],
    *,
    protocols_by_id: Mapping[str, Mapping[str, Any]],
    repetitions: int,
    memory_mode: str,
    model_seed: int | None,
    catalog: EpisodeCatalog,
) -> tuple[str, int]:
    """Validate durable protocol evidence before accepting its resume key."""

    key = _record_key(record)
    protocol_id, repetition = key
    if record.get("schema_version") != EVALUATION_SCHEMA_VERSION:
        raise ProtocolEvaluationError(
            f"Protocol resume record {key!r} has an incompatible schema."
        )
    if not 1 <= repetition <= repetitions:
        raise ProtocolEvaluationError(
            f"Protocol resume record {key!r} is outside this evaluation."
        )
    protocol = protocols_by_id.get(protocol_id)
    if protocol is None:
        raise ProtocolEvaluationError(
            f"Protocol resume record {key!r} is not selected."
        )
    expected_seed = (
        model_seed + repetition - 1 if model_seed is not None else None
    )
    record_model_seed = record.get("model_seed")
    model_seed_valid = (
        record_model_seed is None
        if expected_seed is None
        else (
            isinstance(record_model_seed, int)
            and not isinstance(record_model_seed, bool)
            and record_model_seed == expected_seed
        )
    )
    if (
        record.get("user_id") != str(protocol.get("user_id", "")).strip()
        or record.get("memory_mode") != memory_mode
        or not model_seed_valid
    ):
        raise ProtocolEvaluationError(
            f"Protocol resume record {key!r} has inconsistent runtime metadata."
        )
    resolver = RepetitionScenarioResolver(catalog, repetition)
    for step in protocol.get("steps", []):
        selector = (
            step.get("scenario_selector")
            if isinstance(step, Mapping)
            else None
        )
        if isinstance(selector, Mapping):
            resolver(selector)
    status = record.get("status")
    passed = record.get("passed")
    if status not in {"passed", "failed", "error"} or not isinstance(
        passed, bool
    ):
        raise ProtocolEvaluationError(
            f"Protocol resume record {key!r} has invalid terminal status."
        )
    actual_selections = record.get("selections")
    planned_selections = record.get("planned_selections")
    if (
        not _selection_integer_types_are_strict(planned_selections)
        or planned_selections != resolver.selections
    ):
        raise ProtocolEvaluationError(
            f"Protocol resume record {key!r} has inconsistent planned "
            "selections."
        )
    selections_valid = (
        _selection_integer_types_are_strict(actual_selections)
        and (
            actual_selections == resolver.selections
            if status != "error"
            else actual_selections
            == resolver.selections[: len(actual_selections)]
        )
    )
    if not selections_valid:
        raise ProtocolEvaluationError(
            f"Protocol resume record {key!r} has inconsistent selections."
        )
    artifact_status = record.get("artifact_status")
    tracked = (
        artifact_status.get("tracked")
        if isinstance(artifact_status, Mapping)
        else None
    )
    complete = (
        artifact_status.get("complete")
        if isinstance(artifact_status, Mapping)
        else None
    )
    agent_failures = record.get("agent_failures")
    agent_failures_valid = (
        isinstance(agent_failures, list)
        and all(
            isinstance(failure, Mapping)
            and all(
                isinstance(failure.get(name), str)
                and bool(failure.get(name).strip())
                for name in (
                    "stage",
                    "error_type",
                    "classification",
                    "message",
                )
            )
            for failure in agent_failures
        )
    )
    if (
        not isinstance(artifact_status, Mapping)
        or not isinstance(tracked, bool)
        or (
            tracked
            and not isinstance(complete, bool)
        )
        or (
            not tracked
            and complete is not None
        )
        or not agent_failures_valid
    ):
        raise ProtocolEvaluationError(
            f"Protocol resume record {key!r} lacks artifact evidence."
        )
    if status == "error":
        error = record.get("error")
        if (
            passed
            or record.get("report") is not None
            or not isinstance(error, Mapping)
            or any(
                not isinstance(error.get(name), str)
                or not error.get(name).strip()
                for name in ("category", "type", "message", "traceback")
            )
            or record.get("failure_category") != error.get("category")
        ):
            raise ProtocolEvaluationError(
                f"Protocol resume error record {key!r} is malformed."
            )
    else:
        report = record.get("report")
        if (
            not isinstance(report, Mapping)
            or record.get("error") is not None
            or (status == "passed") is not passed
        ):
            raise ProtocolEvaluationError(
                f"Protocol resume record {key!r} has invalid report evidence."
            )
        report_passed = _validate_report_evidence(
            report,
            protocol=protocol,
            catalog=catalog,
            key=key,
        )
        expected_record_passed = report_passed and (
            not tracked or complete is True
        )
        if passed is not expected_record_passed:
            raise ProtocolEvaluationError(
                f"Protocol resume record {key!r} has inconsistent passed "
                "status."
            )
        expected_failure_category = (
            None
            if expected_record_passed
            else (
                "ARTIFACT_INCOMPLETE"
                if tracked and complete is False
                else "PROTOCOL_CHECK_FAILURE"
            )
        )
        if record.get("failure_category") != expected_failure_category:
            raise ProtocolEvaluationError(
                f"Protocol resume record {key!r} has inconsistent failure "
                "category."
            )
    return key


def _prepare_run_config(
    path: Path,
    payload: Mapping[str, Any],
    *,
    output_directory: Path,
) -> None:
    if path.is_file():
        existing = _read_json_object(path, description="Protocol run configuration")
        if existing.get("evaluation") != payload:
            raise ProtocolEvaluationError(
                "Output directory belongs to an incompatible protocol evaluation; "
                "choose a new output directory."
            )
        return
    if output_directory.exists() and any(output_directory.iterdir()):
        raise ProtocolEvaluationError(
            "Existing evaluation artifacts have no compatible run_config.json; "
            "choose a new output directory."
        )
    _write_json_atomic(
        path,
        {
            "created_at": _utc_now(),
            "evaluation": copy.deepcopy(dict(payload)),
        },
    )


def _failure_category(error: Exception) -> str:
    name = type(error).__name__
    if isinstance(error, TimeoutError) or "timeout" in name.lower():
        return "INFRASTRUCTURE_TIMEOUT"
    if name in {"HRIContractError"}:
        return "OUTPUT_CONTRACT"
    if name in {"MemoryAgentError", "MemoryRepositoryError"}:
        return "MEMORY"
    if name in {"DatasetEpisodeError", "BenchmarkEpisodeError"}:
        return "DATASET"
    if isinstance(error, OSError):
        return "PERSISTENCE"
    if isinstance(error, ProtocolEvaluationError):
        return "EVALUATION_CONTRACT"
    return "AGENT_RUNTIME"


def _evaluation_artifact_status(orchestrator: Any) -> dict[str, Any]:
    """Read production audit-stream status without requiring it from test fakes."""

    provider = getattr(orchestrator, "evaluation_artifact_status", None)
    if not callable(provider):
        return {"tracked": False, "complete": None}
    try:
        raw = provider()
    except Exception as error:
        return {
            "tracked": True,
            "complete": False,
            "status_error": {
                "type": type(error).__name__,
                "message": str(error),
            },
        }
    if not isinstance(raw, Mapping):
        return {
            "tracked": True,
            "complete": False,
            "status_error": {
                "type": "TypeError",
                "message": "Artifact status provider did not return an object.",
            },
        }
    status = copy.deepcopy(dict(raw))
    status["tracked"] = True
    status["complete"] = status.get("complete") is True
    return status


def _config_metadata(config: PrefMemConfig) -> dict[str, Any]:
    agents: dict[str, Any] = {}
    for name in ("hri", "memory", "planner", "validator"):
        agent = getattr(config, name)
        prompt = Path(agent.system_prompt_path)
        agents[name] = {
            "model": agent.model,
            "temperature": agent.temperature,
            "seed": agent.seed,
            "host": agent.host,
            "timeout_seconds": agent.timeout_seconds,
            "prompt_path": str(prompt),
            "prompt_sha256": _sha256(prompt) if prompt.is_file() else None,
        }
    return {
        "agents": agents,
        "max_replans": config.max_replans,
        "max_reobservations": config.max_reobservations,
        "recent_history_limit": config.recent_history_limit,
        "history_semantic_scan_limit": config.history_semantic_scan_limit,
        "semantic_history_limit": config.semantic_history_limit,
        "semantic_preference_limit": config.semantic_preference_limit,
        "semantic_preference_threshold": config.semantic_preference_threshold,
        "preference_proposal_min_episodes": config.preference_proposal_min_episodes,
        "preference_proposal_confidence": config.preference_proposal_confidence,
        "compact_preferences_after_write": config.compact_preferences_after_write,
        "minimum_planner_confidence": config.minimum_planner_confidence,
        "minimum_validator_confidence": config.minimum_validator_confidence,
        "vision": {
            "resize_images": config.vision.resize_images,
            "image_width": config.vision.image_width,
            "image_height": config.vision.image_height,
            "image_jpeg_quality": config.vision.image_jpeg_quality,
        },
    }


def _wilson_ci95(successes: int, total: int) -> dict[str, float] | None:
    if total <= 0:
        return None
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return {
        "confidence": 0.95,
        "lower": max(0.0, centre - radius),
        "upper": min(1.0, centre + radius),
    }


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _duration_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    durations = [
        float(record["duration_seconds"])
        for record in records
        if isinstance(record.get("duration_seconds"), (int, float))
        and not isinstance(record.get("duration_seconds"), bool)
    ]
    return {
        "observations": len(durations),
        "p50": _percentile(durations, 0.50),
        "p95": _percentile(durations, 0.95),
    }


def _physical_scene_cluster_unit(selection: Mapping[str, Any]) -> str | None:
    family = str(selection.get("family", "")).strip()
    variant = str(
        selection.get(
            "scene_variant",
            selection.get("variant", ""),
        )
    ).strip()
    seed = selection.get("seed")
    if (
        family
        and variant
        and isinstance(seed, int)
        and not isinstance(seed, bool)
    ):
        control_kind = selection.get("control_kind")
        if control_kind is not None:
            target_id = str(selection.get("target_id", "")).strip()
            if target_id:
                return json.dumps(
                    ["control", family, variant, seed, target_id],
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
        return json.dumps(
            ["core", family, variant, seed],
            ensure_ascii=False,
            separators=(",", ":"),
        )

    counterfactual_group_id = str(
        selection.get("counterfactual_group_id", "")
    ).strip()
    if counterfactual_group_id:
        return f"counterfactual_group_id:{counterfactual_group_id}"

    scenario_id = str(selection.get("scenario_id", "")).strip()
    return f"scenario_id:{scenario_id}" if scenario_id else None


def _record_planned_selections(
    record: Mapping[str, Any],
) -> list[Any]:
    planned = record.get("planned_selections")
    if isinstance(planned, list) and planned:
        return planned
    actual = record.get("selections")
    return actual if isinstance(actual, list) else []


def _protocol_scene_cluster_key(
    record: Mapping[str, Any],
) -> tuple[str, ...]:
    selections = _record_planned_selections(record)
    scene_units = {
        unit
        for selection in selections
        if isinstance(selection, Mapping)
        and (unit := _physical_scene_cluster_unit(selection)) is not None
    }
    if scene_units:
        return tuple(sorted(scene_units))
    return (
        f"unresolved:{record.get('protocol_id')}:{record.get('repetition')}",
    )


def _protocol_cluster_bootstrap_ci95(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    if not records:
        return None
    clusters: dict[tuple[str, ...], list[Mapping[str, Any]]] = {}
    for record in records:
        key = _protocol_scene_cluster_key(record)
        clusters.setdefault(key, []).append(record)
    keys = sorted(clusters)
    if len(keys) == 1:
        rate = sum(
            item.get("passed") is True for item in records
        ) / len(records)
        return {
            "confidence": 0.95,
            "lower": rate,
            "upper": rate,
            "method": "percentile_scene_cluster_bootstrap",
            "clusters": 1,
            "cluster_count": 1,
            "cluster_unit": _PROTOCOL_CLUSTER_UNIT,
            "replicates": _PROTOCOL_BOOTSTRAP_REPLICATES,
            "replicate_count": _PROTOCOL_BOOTSTRAP_REPLICATES,
            "random_seed": _PROTOCOL_BOOTSTRAP_SEED,
        }
    generator = random.Random(_PROTOCOL_BOOTSTRAP_SEED)
    rates: list[float] = []
    for _ in range(_PROTOCOL_BOOTSTRAP_REPLICATES):
        sample = [generator.choice(keys) for _ in keys]
        sampled_records = [
            record for key in sample for record in clusters[key]
        ]
        rates.append(
            sum(item.get("passed") is True for item in sampled_records)
            / len(sampled_records)
        )
    return {
        "confidence": 0.95,
        "lower": _percentile(rates, 0.025),
        "upper": _percentile(rates, 0.975),
        "method": "percentile_scene_cluster_bootstrap",
        "clusters": len(keys),
        "cluster_count": len(keys),
        "cluster_unit": _PROTOCOL_CLUSTER_UNIT,
        "replicates": _PROTOCOL_BOOTSTRAP_REPLICATES,
        "replicate_count": _PROTOCOL_BOOTSTRAP_REPLICATES,
        "random_seed": _PROTOCOL_BOOTSTRAP_SEED,
    }


def _error_categories(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    categories: dict[str, int] = {}
    for record in records:
        if record.get("status") != "error":
            continue
        error = record.get("error")
        category = (
            str(error.get("category", "")).strip()
            if isinstance(error, Mapping)
            else ""
        )
        key = category or "UNCLASSIFIED"
        categories[key] = categories.get(key, 0) + 1
    return dict(sorted(categories.items()))


def _empty_protocol_check_counts() -> dict[str, int]:
    return {
        "total": 0,
        "evaluated": 0,
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "expected_evidence": 0,
        "expected_evaluated": 0,
        "expected_passed": 0,
        "unexpected_evidence": 0,
    }


def _accumulate_protocol_check(
    checks: dict[str, dict[str, int]],
    *,
    name: str,
    check: Mapping[str, Any],
    expected_slot: bool,
) -> None:
    counts = checks.setdefault(name, _empty_protocol_check_counts())
    counts["total"] += 1
    evaluated = check.get("evaluated") is True
    passed = evaluated and check.get("passed") is True
    if not evaluated:
        counts["skipped"] += 1
    else:
        counts["evaluated"] += 1
        counts["passed" if passed else "failed"] += 1

    if expected_slot:
        counts["expected_evidence"] += 1
        if evaluated:
            counts["expected_evaluated"] += 1
            if passed:
                counts["expected_passed"] += 1
    else:
        counts["unexpected_evidence"] += 1


def _build_summary(
    *,
    records: Sequence[Mapping[str, Any]],
    benchmark_root: Path,
    protocol_path: Path,
    protocol_digest: str,
    index: Mapping[str, Any],
    repetitions: int,
    config: PrefMemConfig,
    results_path: Path,
    memory_mode: str,
    model_seed: int | None,
    planned_runs: int,
    executed_runs: int,
    skipped_runs: int,
    protocols: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_protocol: dict[str, dict[str, Any]] = {}
    records_by_protocol: dict[str, list[Mapping[str, Any]]] = {}
    checks: dict[str, dict[str, int]] = {}
    expected_checks: dict[str, int] = {}
    failure_categories: dict[str, int] = {}
    tracked_artifacts = 0
    complete_artifacts = 0
    agent_failure_classifications: dict[str, int] = {}
    agent_failure_stages: dict[str, int] = {}
    agent_failure_events = 0
    agent_failure_affected_trials = 0
    agent_failure_affected_classifications: dict[str, int] = {}
    agent_failure_affected_stages: dict[str, int] = {}
    protocols_by_id = {
        str(protocol.get("protocol_id", "")): protocol
        for protocol in protocols
    }
    for record in records:
        protocol_id = str(record["protocol_id"])
        protocol = protocols_by_id.get(protocol_id, {})
        for step in protocol.get("steps", []):
            if not isinstance(step, Mapping):
                continue
            expected = step.get("expected")
            if not isinstance(expected, Mapping):
                continue
            names = set(map(str, expected)) | set(
                _MANDATORY_PROTOCOL_CHECKS
            )
            for name in names:
                expected_checks[name] = expected_checks.get(name, 0) + 1
            if expected.get("task_semantic_score") is True:
                for name in _PROTOCOL_TASK_SCORE_CHECKS:
                    key = f"task_score.{name}"
                    expected_checks[key] = expected_checks.get(key, 0) + 1
        records_by_protocol.setdefault(protocol_id, []).append(record)
        item = by_protocol.setdefault(
            protocol_id,
            {
                "runs": 0,
                "passed": 0,
                "failed": 0,
                "errors": 0,
                "subagent_failures": {
                    "event_count": 0,
                    "affected_trial_count": 0,
                    "events_by_classification": {},
                    "events_by_stage": {},
                    "affected_trials_by_classification": {},
                    "affected_trials_by_stage": {},
                },
            },
        )
        item["runs"] += 1
        if record.get("status") == "error":
            item["errors"] += 1
        elif record.get("passed") is True:
            item["passed"] += 1
        else:
            item["failed"] += 1
        failure_category = str(record.get("failure_category", "")).strip()
        if failure_category:
            failure_categories[failure_category] = (
                failure_categories.get(failure_category, 0) + 1
            )
        failures = record.get("agent_failures")
        trial_failure_events = 0
        trial_failure_classifications: set[str] = set()
        trial_failure_stages: set[str] = set()
        protocol_failures = item["subagent_failures"]
        if isinstance(failures, list):
            for failure in failures:
                if not isinstance(failure, Mapping):
                    continue
                trial_failure_events += 1
                agent_failure_events += 1
                protocol_failures["event_count"] += 1
                classification = str(
                    failure.get("classification", "")
                ).strip()
                stage = str(failure.get("stage", "")).strip()
                if classification:
                    trial_failure_classifications.add(classification)
                    agent_failure_classifications[classification] = (
                        agent_failure_classifications.get(classification, 0)
                        + 1
                    )
                    events_by_classification = protocol_failures[
                        "events_by_classification"
                    ]
                    events_by_classification[classification] = (
                        events_by_classification.get(classification, 0) + 1
                    )
                if stage:
                    trial_failure_stages.add(stage)
                    agent_failure_stages[stage] = (
                        agent_failure_stages.get(stage, 0) + 1
                    )
                    events_by_stage = protocol_failures["events_by_stage"]
                    events_by_stage[stage] = (
                        events_by_stage.get(stage, 0) + 1
                    )
        if trial_failure_events:
            agent_failure_affected_trials += 1
            protocol_failures["affected_trial_count"] += 1
        for classification in trial_failure_classifications:
            agent_failure_affected_classifications[classification] = (
                agent_failure_affected_classifications.get(classification, 0)
                + 1
            )
            affected = protocol_failures[
                "affected_trials_by_classification"
            ]
            affected[classification] = affected.get(classification, 0) + 1
        for stage in trial_failure_stages:
            agent_failure_affected_stages[stage] = (
                agent_failure_affected_stages.get(stage, 0) + 1
            )
            affected = protocol_failures["affected_trials_by_stage"]
            affected[stage] = affected.get(stage, 0) + 1
        artifact_status = record.get("artifact_status")
        if (
            isinstance(artifact_status, Mapping)
            and artifact_status.get("tracked") is True
        ):
            tracked_artifacts += 1
            if artifact_status.get("complete") is True:
                complete_artifacts += 1
        report = record.get("report")
        if not isinstance(report, Mapping):
            continue
        protocol_steps = protocol.get("steps", [])
        for step in report.get("steps", []):
            if not isinstance(step, Mapping):
                continue
            step_index = step.get("index")
            protocol_step = (
                protocol_steps[step_index - 1]
                if (
                    isinstance(step_index, int)
                    and not isinstance(step_index, bool)
                    and 1 <= step_index <= len(protocol_steps)
                    and isinstance(protocol_steps[step_index - 1], Mapping)
                )
                else {}
            )
            declared_expected = protocol_step.get("expected")
            declared_regular = (
                set(map(str, declared_expected))
                | set(_MANDATORY_PROTOCOL_CHECKS)
                if isinstance(declared_expected, Mapping)
                else set()
            )
            declared_task = (
                set(_PROTOCOL_TASK_SCORE_CHECKS)
                if (
                    isinstance(declared_expected, Mapping)
                    and declared_expected.get("task_semantic_score") is True
                )
                else set()
            )
            seen_regular: set[str] = set()
            for check in step.get("checks", []):
                if not isinstance(check, Mapping):
                    continue
                name = str(check.get("name", ""))
                expected_slot = (
                    name in declared_regular and name not in seen_regular
                )
                if expected_slot:
                    seen_regular.add(name)
                _accumulate_protocol_check(
                    checks,
                    name=name,
                    check=check,
                    expected_slot=expected_slot,
                )
            task_score = step.get("task_score")
            if isinstance(task_score, Mapping):
                seen_task: set[str] = set()
                for check in task_score.get("checks", []):
                    if not isinstance(check, Mapping):
                        continue
                    raw_name = str(check.get("name", ""))
                    name = f"task_score.{raw_name}"
                    expected_slot = (
                        raw_name in declared_task
                        and raw_name not in seen_task
                    )
                    if expected_slot:
                        seen_task.add(raw_name)
                    _accumulate_protocol_check(
                        checks,
                        name=name,
                        check=check,
                        expected_slot=expected_slot,
                    )

    for protocol_id, protocol_records in records_by_protocol.items():
        item = by_protocol[protocol_id]
        protocol_runs = len(protocol_records)
        protocol_passed = int(item["passed"])
        item["pass_rate"] = (
            protocol_passed / protocol_runs if protocol_runs else 0.0
        )
        item["pass_rate_wilson_ci95"] = _wilson_ci95(
            protocol_passed,
            protocol_runs,
        )
        item["duration_seconds"] = _duration_summary(protocol_records)
        item["error_categories"] = _error_categories(protocol_records)
        item["pass_rate_scene_cluster_bootstrap_ci95"] = (
            _protocol_cluster_bootstrap_ci95(protocol_records)
        )
        failure_summary = item["subagent_failures"]
        failure_summary["affected_trial_rate"] = (
            failure_summary["affected_trial_count"] / protocol_runs
            if protocol_runs
            else None
        )
        for key in (
            "events_by_classification",
            "events_by_stage",
            "affected_trials_by_classification",
            "affected_trials_by_stage",
        ):
            failure_summary[key] = dict(
                sorted(failure_summary[key].items())
            )
        # Preserve the original names as explicit event-count aliases.
        failure_summary["by_classification"] = copy.deepcopy(
            failure_summary["events_by_classification"]
        )
        failure_summary["by_stage"] = copy.deepcopy(
            failure_summary["events_by_stage"]
        )
        failure_summary["events"] = {
            "total": failure_summary["event_count"],
            "by_classification": copy.deepcopy(
                failure_summary["events_by_classification"]
            ),
            "by_stage": copy.deepcopy(
                failure_summary["events_by_stage"]
            ),
        }
        failure_summary["affected_trials"] = {
            "total": failure_summary["affected_trial_count"],
            "rate": failure_summary["affected_trial_rate"],
            "by_classification": copy.deepcopy(
                failure_summary[
                    "affected_trials_by_classification"
                ]
            ),
            "by_stage": copy.deepcopy(
                failure_summary["affected_trials_by_stage"]
            ),
        }

    for name in sorted(set(checks) | set(expected_checks)):
        counts = checks.setdefault(
            name,
            _empty_protocol_check_counts(),
        )
        expected = expected_checks.get(name, 0)
        expected_evidence = counts["expected_evidence"]
        aligned_evidence = min(expected_evidence, expected)
        aligned_passed = min(
            counts["expected_passed"],
            aligned_evidence,
        )
        counts["expected"] = expected
        counts["aligned_expected_evidence"] = aligned_evidence
        counts["excess_evidence"] = (
            counts["unexpected_evidence"]
            + max(0, expected_evidence - expected)
        )
        counts["missing_evidence"] = max(0, expected - aligned_evidence)
        counts["coverage_rate"] = (
            aligned_evidence / expected if expected else None
        )
        counts["conditional_accuracy"] = (
            counts["expected_passed"] / counts["expected_evaluated"]
            if counts["expected_evaluated"]
            else None
        )
        counts["unconditional_accuracy"] = (
            aligned_passed / expected if expected else None
        )
        counts["all_evidence_conditional_accuracy"] = (
            counts["passed"] / counts["evaluated"]
            if counts["evaluated"]
            else None
        )

    run_count = len(records)
    passed = sum(record.get("passed") is True for record in records)
    errors = sum(record.get("status") == "error" for record in records)
    selection_variants: dict[str, int] = {}
    selection_seeds: dict[str, int] = {}
    selected_scenarios: set[str] = set()
    for record in records:
        selections = _record_planned_selections(record)
        for selection in selections:
            if not isinstance(selection, Mapping):
                continue
            variant = str(selection.get("scene_variant", "")).strip()
            seed = str(selection.get("seed", "")).strip()
            scenario_id = str(selection.get("scenario_id", "")).strip()
            if variant:
                selection_variants[variant] = (
                    selection_variants.get(variant, 0) + 1
                )
            if seed:
                selection_seeds[seed] = selection_seeds.get(seed, 0) + 1
            if scenario_id:
                selected_scenarios.add(scenario_id)
    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "benchmark_root": str(benchmark_root),
        "benchmark_catalog_digest": index.get("catalog_digest"),
        "protocol_path": str(protocol_path),
        "protocol_sha256": protocol_digest,
        "results_path": str(results_path),
        "repetitions": repetitions,
        "planned_runs": planned_runs,
        "executed_runs_this_invocation": executed_runs,
        "skipped_runs_this_invocation": skipped_runs,
        "memory_mode": memory_mode,
        "model_seed_base": model_seed,
        "model_seeds_by_repetition": (
            [model_seed + repetition for repetition in range(repetitions)]
            if model_seed is not None
            else None
        ),
        "run_count": run_count,
        "passed": passed,
        "failed": run_count - passed,
        "errors": errors,
        "pass_rate": passed / run_count if run_count else 0.0,
        "pass_rate_wilson_ci95": _wilson_ci95(passed, run_count),
        "pass_rate_scene_cluster_bootstrap_ci95": (
            _protocol_cluster_bootstrap_ci95(records)
        ),
        "duration_seconds": _duration_summary(records),
        "error_categories": _error_categories(records),
        "failure_categories": dict(sorted(failure_categories.items())),
        "subagent_failures": {
            "event_count": agent_failure_events,
            "affected_trial_count": agent_failure_affected_trials,
            "affected_trial_rate": (
                agent_failure_affected_trials / run_count
                if run_count
                else None
            ),
            "events_by_classification": dict(
                sorted(agent_failure_classifications.items())
            ),
            "events_by_stage": dict(sorted(agent_failure_stages.items())),
            "affected_trials_by_classification": dict(
                sorted(agent_failure_affected_classifications.items())
            ),
            "affected_trials_by_stage": dict(
                sorted(agent_failure_affected_stages.items())
            ),
            "by_classification": dict(
                sorted(agent_failure_classifications.items())
            ),
            "by_stage": dict(sorted(agent_failure_stages.items())),
            "events": {
                "total": agent_failure_events,
                "by_classification": dict(
                    sorted(agent_failure_classifications.items())
                ),
                "by_stage": dict(sorted(agent_failure_stages.items())),
            },
            "affected_trials": {
                "total": agent_failure_affected_trials,
                "rate": (
                    agent_failure_affected_trials / run_count
                    if run_count
                    else None
                ),
                "by_classification": dict(
                    sorted(
                        agent_failure_affected_classifications.items()
                    )
                ),
                "by_stage": dict(
                    sorted(agent_failure_affected_stages.items())
                ),
            },
        },
        "evaluation_artifacts": {
            "tracked_runs": tracked_artifacts,
            "complete_runs": complete_artifacts,
            "incomplete_runs": tracked_artifacts - complete_artifacts,
            "complete_rate": (
                complete_artifacts / tracked_artifacts
                if tracked_artifacts
                else None
            ),
        },
        "selection_coverage": {
            "unique_scenarios": len(selected_scenarios),
            "by_scene_variant": dict(sorted(selection_variants.items())),
            "by_seed": dict(sorted(selection_seeds.items())),
        },
        "protocols": by_protocol,
        "checks": checks,
        "config": _config_metadata(config),
    }


def evaluate_memory_protocols(
    benchmark_root: str | Path,
    output_directory: str | Path,
    *,
    repetitions: int = 1,
    protocol_ids: Sequence[str] | None = None,
    base_config: PrefMemConfig | None = None,
    orchestrator_factory: OrchestratorFactory | None = None,
    fixture_applier: FixtureApplier | None = None,
    protocols_path: str | Path | None = None,
    memory_mode: str = "full",
    model_seed: int | None = None,
    resume: bool = True,
) -> ProtocolBatchResult:
    """Execute generated memory protocols with fresh state per repetition.

    Every attempted ``(protocol_id, repetition)`` is durably appended to JSONL
    and becomes a resume key, including failed attempts. Run-level model,
    contract, fixture, and persistence failures do not prevent later protocols
    from running. Structural setup failures are raised before any model call.
    """

    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions <= 0:
        raise ProtocolEvaluationError("repetitions must be a positive integer.")
    normalized_memory_mode = str(memory_mode).strip().lower()
    if normalized_memory_mode not in MEMORY_MODES:
        raise ProtocolEvaluationError(
            f"Unsupported memory mode {memory_mode!r}; expected one of {MEMORY_MODES}."
        )
    if model_seed is not None and (
        isinstance(model_seed, bool)
        or not isinstance(model_seed, int)
        or model_seed < 0
    ):
        raise ProtocolEvaluationError(
            "model_seed must be a non-negative integer or null."
        )

    benchmark = Path(benchmark_root).expanduser().resolve()
    output = Path(output_directory).expanduser().resolve()
    if _inside(output, benchmark) or _inside(benchmark, output):
        raise ProtocolEvaluationError(
            "Evaluation output and immutable benchmark directories must be disjoint."
        )
    validation = validate_benchmark(benchmark)
    if not validation.valid:
        errors = list(validation.errors)
        detail = "\n".join(errors[:20])
        suffix = (
            f"\n... and {len(errors) - 20} more error(s)"
            if len(errors) > 20
            else ""
        )
        raise ProtocolEvaluationError(
            f"Benchmark validation failed with {len(errors)} error(s):\n"
            f"{detail}{suffix}"
        )
    protocol_path = (
        Path(protocols_path).expanduser().resolve()
        if protocols_path is not None
        else benchmark / "protocols.json"
    )
    bundle = load_protocol_bundle(protocol_path)
    catalog = EpisodeCatalog.from_benchmark(benchmark)
    index = _read_json_object(benchmark / "index.json", description="Benchmark index")

    selected_ids = None if protocol_ids is None else tuple(map(str, protocol_ids))
    if selected_ids is not None:
        if not selected_ids or len(selected_ids) != len(set(selected_ids)):
            raise ProtocolEvaluationError(
                "protocol_ids must be a non-empty sequence of unique IDs."
            )
        available = {
            str(item["protocol_id"])
            for item in bundle["protocols"]
            if isinstance(item, Mapping)
        }
        missing = set(selected_ids) - available
        if missing:
            raise ProtocolEvaluationError(
                f"Unknown requested protocol IDs: {sorted(missing)}"
            )
        order = {protocol_id: index for index, protocol_id in enumerate(selected_ids)}
        protocols = sorted(
            (
                copy.deepcopy(item)
                for item in bundle["protocols"]
                if str(item["protocol_id"]) in order
            ),
            key=lambda item: order[str(item["protocol_id"])],
        )
    else:
        protocols = copy.deepcopy(bundle["protocols"])

    # Resolve the complete repetition schedule before creating output
    # directories, repositories, consent fixtures, or model-backed agents.
    selection_plans = _preflight_selection_plans(
        protocols,
        catalog=catalog,
        repetitions=repetitions,
    )
    selection_plan_payload = [
        {
            "protocol_id": protocol_id,
            "repetition": repetition,
            "selections": selections,
        }
        for (protocol_id, repetition), selections in sorted(
            selection_plans.items()
        )
    ]
    selection_plan_sha256 = hashlib.sha256(
        json.dumps(
            selection_plan_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    config_template = copy.deepcopy(base_config or PrefMemConfig())
    factory = orchestrator_factory or default_orchestrator_factory
    apply_fixture = fixture_applier or apply_consent_fixture
    protocol_digest = _sha256(protocol_path)
    benchmark_content_digest = benchmark_content_sha256(
        benchmark,
        index.get("scenario_ids", []),
    )
    run_payload = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "runtime": runtime_provenance(Path(__file__).resolve().parents[2]),
        "benchmark_root": str(benchmark),
        "benchmark_catalog_digest": index.get("catalog_digest"),
        "benchmark_content_sha256": benchmark_content_digest,
        "protocol_path": str(protocol_path),
        "protocol_sha256": protocol_digest,
        "protocol_ids": [str(item["protocol_id"]) for item in protocols],
        "repetitions": repetitions,
        "memory_mode": normalized_memory_mode,
        "model_seed": model_seed,
        "config": _config_metadata(config_template),
        "orchestrator_factory": callable_provenance(factory),
        "fixture_applier": callable_provenance(apply_fixture),
        "selector_policy": "balanced-variant-seed-physical-scene-v3",
        "selection_plan_sha256": selection_plan_sha256,
    }

    results_path = output / "results.jsonl"
    summary_path = output / "summary.json"
    runs_root = output / "runs"
    run_config_path = output / "run_config.json"
    _prepare_run_config(
        run_config_path,
        run_payload,
        output_directory=output,
    )
    existing_records = _load_jsonl(results_path)
    if existing_records and not resume:
        raise ProtocolEvaluationError(
            "results.jsonl already exists and resume is disabled; choose a fresh "
            "output directory."
        )
    completed_keys: set[tuple[str, int]] = set()
    protocols_by_id = {
        str(protocol["protocol_id"]): protocol for protocol in protocols
    }
    for existing in existing_records:
        key = _validate_resume_record(
            existing,
            protocols_by_id=protocols_by_id,
            repetitions=repetitions,
            memory_mode=normalized_memory_mode,
            model_seed=model_seed,
            catalog=catalog,
        )
        if key in completed_keys:
            raise ProtocolEvaluationError(
                f"Duplicate protocol resume key in results.jsonl: {key!r}"
            )
        completed_keys.add(key)

    planned_keys = {
        _run_key(str(protocol["protocol_id"]), repetition)
        for protocol in protocols
        for repetition in range(1, repetitions + 1)
    }
    outside = completed_keys - planned_keys
    if outside:
        raise ProtocolEvaluationError(
            "results.jsonl contains runs outside the configured evaluation: "
            f"{sorted(outside)}"
        )
    runs_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = list(existing_records)
    executed_runs = 0
    skipped_runs = 0

    for protocol in protocols:
        protocol_id = str(protocol["protocol_id"])
        user_id = str(protocol.get("user_id", "")).strip()
        safe_id = _safe_component(protocol_id)
        for repetition in range(1, repetitions + 1):
            key = _run_key(protocol_id, repetition)
            if key in completed_keys:
                skipped_runs += 1
                continue
            trial_root = runs_root / safe_id / f"rep-{repetition:04d}"
            run_directory = trial_root / f"attempt-{uuid.uuid4().hex}"
            run_directory.mkdir(parents=True, exist_ok=False)
            resolver = RepetitionScenarioResolver(catalog, repetition)
            started_at = _utc_now()
            started = time.perf_counter()
            record: dict[str, Any] = {
                "schema_version": EVALUATION_SCHEMA_VERSION,
                "protocol_id": protocol_id,
                "repetition": repetition,
                "user_id": user_id,
                "memory_mode": normalized_memory_mode,
                "model_seed": (
                    model_seed + repetition - 1
                    if model_seed is not None
                    else None
                ),
                "started_at": started_at,
                "run_directory": str(run_directory),
                "status": "error",
                "passed": False,
                "selections": resolver.selections,
                "planned_selections": copy.deepcopy(selection_plans[key]),
                "report": None,
                "error": None,
                "failure_category": None,
                "artifact_status": {
                    "tracked": False,
                    "complete": None,
                },
                "agent_failures": [],
            }
            orchestrator: Any | None = None
            try:
                initial_episode = resolver(_first_selector(protocol))
                config = _isolated_config(
                    config_template,
                    user_id=user_id,
                    initial_episode=initial_episode,
                    run_directory=run_directory,
                    model_seed=record["model_seed"],
                )
                context = ProtocolRunContext(
                    protocol=copy.deepcopy(protocol),
                    repetition=repetition,
                    run_directory=run_directory,
                    initial_episode_path=initial_episode,
                    config=config,
                    memory_mode=normalized_memory_mode,
                    model_seed=record["model_seed"],
                )
                orchestrator = factory(context)
                report = run_memory_protocol(
                    protocol,
                    orchestrator,
                    resolve_scenario=resolver,
                    apply_fixture=apply_fixture,
                    fixtures=bundle["fixtures"],
                )
                artifact_status = _evaluation_artifact_status(orchestrator)
                artifacts_complete = (
                    artifact_status.get("tracked") is not True
                    or artifact_status.get("complete") is True
                )
                record["artifact_status"] = artifact_status
                record["status"] = (
                    "passed"
                    if report.passed and artifacts_complete
                    else "failed"
                )
                record["passed"] = report.passed and artifacts_complete
                if not artifacts_complete:
                    record["failure_category"] = "ARTIFACT_INCOMPLETE"
                elif not report.passed:
                    record["failure_category"] = "PROTOCOL_CHECK_FAILURE"
                record["report"] = _protocol_report_dict(report)
                record["agent_failures"] = _agent_failure_evidence(
                    record["report"]
                )
            except Exception as error:
                if orchestrator is not None:
                    record["artifact_status"] = (
                        _evaluation_artifact_status(orchestrator)
                    )
                record["error"] = {
                    "category": _failure_category(error),
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                }
                record["failure_category"] = record["error"]["category"]
            finally:
                record["selections"] = copy.deepcopy(resolver.selections)
                record["finished_at"] = _utc_now()
                record["duration_seconds"] = round(
                    time.perf_counter() - started,
                    6,
                )
                records.append(record)
                _append_jsonl(results_path, record)
                completed_keys.add(key)
                executed_runs += 1

    summary = _build_summary(
        records=records,
        benchmark_root=benchmark,
        protocol_path=protocol_path,
        protocol_digest=protocol_digest,
        index=index,
        repetitions=repetitions,
        config=config_template,
        results_path=results_path,
        memory_mode=normalized_memory_mode,
        model_seed=model_seed,
        planned_runs=len(planned_keys),
        executed_runs=executed_runs,
        skipped_runs=skipped_runs,
        protocols=protocols,
    )
    _write_json_atomic(summary_path, summary)
    return ProtocolBatchResult(
        output_directory=output,
        results_path=results_path,
        summary_path=summary_path,
        planned_runs=len(planned_keys),
        executed_runs=executed_runs,
        skipped_runs=skipped_runs,
        summary=summary,
    )
