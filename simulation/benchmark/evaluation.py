"""Resumable cold-memory evaluation of PrefMem benchmark packets.

The runner in this module deliberately keeps the benchmark oracle on the
evaluation side of the system boundary.  A trial gives PrefMem only the packet
instruction, its two recorded observations through ``BenchmarkEpisodeExecutor``,
and a fresh memory namespace.  The manifest is consumed by the scorer only
after the agent turn has returned.
"""

from __future__ import annotations

import copy
import csv
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
from collections import defaultdict
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from tqdm.auto import tqdm

from agents.configs import PrefMemConfig, normalize_model_base_url
from agents.hri import HRIOrchestrator
from agents.model import effective_model_base_url
from dataset.benchmark import BenchmarkEpisode

from .executor import BenchmarkEpisodeExecutor
from .model_defaults import (
    DEFAULT_EVALUATION_MODEL,
    DEFAULT_EVALUATION_PROVIDER,
)
from .provenance import (
    benchmark_content_sha256,
    callable_provenance,
    runtime_provenance,
)
from .scorer import ScoreReport, score_agent_result
from .validator import ValidationReport, validate_benchmark


EVALUATION_SCHEMA_VERSION = "robopref.cold-evaluation.v1"
_SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9_.-]+")

FAILURE_CATEGORIES = frozenset(
    {
        "TIMEOUT",
        "CONTRACT_ERROR",
        "MEMORY_ISOLATION",
        "MEMORY_ERROR",
        "MODEL_OR_CONNECTION_ERROR",
        "SCORING_ERROR",
        "RUNTIME_ERROR",
        "STRICT_CHECK_FAILURE",
        "MISSING_EVIDENCE",
        "ARTIFACT_INCOMPLETE",
    }
)

HARD_GATE_CHECKS = frozenset(
    {
        "validator_called",
        "execution_status",
        "vla_dispatches",
        "preference_delta_without_consent",
        "evaluation_artifacts_complete",
    }
)
STRICT_BENCHMARK_CHECKS = frozenset(
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
SCENE_CLUSTER_BOOTSTRAP_REPLICATES = 2_000
SCENE_CLUSTER_BOOTSTRAP_SEED = 20_260_725
COLD_EVALUATION_USER_ID = "evaluation-participant"


class EvaluationError(RuntimeError):
    """Base class for evaluation setup and artifact errors."""


class EvaluationDatasetError(EvaluationError):
    """Raised when the selected benchmark root is not valid."""


class EvaluationResumeError(EvaluationError):
    """Raised when an output directory belongs to a different evaluation."""


class EvaluationIsolationError(EvaluationError):
    """Raised when a supposedly cold trial already contains participant memory."""


@dataclass(slots=True)
class EvaluationConfig:
    """Configuration for a cold-memory endpoint evaluation.

    Empty filter tuples mean "all".  ``max_scenarios`` takes a deterministic
    balanced sample across family, outcome/control, target, seed, and scene
    variant before repetitions are expanded and shuffled.
    Every ``(condition, repetition, scenario_id)`` tuple is a durable resume key.
    """

    benchmark_root: str | Path
    output_dir: str | Path
    repetitions: int = 1
    condition: str = "full"
    families: tuple[str, ...] = ()
    scene_variants: tuple[str, ...] = ()
    target_ids: tuple[str, ...] = ()
    outcomes: tuple[str, ...] = ()
    seeds: tuple[int, ...] = ()
    scenario_ids: tuple[str, ...] = ()
    include_controls: bool = True
    max_scenarios: int | None = None
    shuffle_seed: int = 0
    resume: bool = True
    fail_fast: bool = False
    memory_mode: str = "full"
    model: str | None = None
    model_provider: str | None = DEFAULT_EVALUATION_PROVIDER
    model_base_url: str | None = None
    ollama_host: str | None = None
    temperature: float | None = None
    model_seed: int | None = None
    timeout_seconds: float | None = None
    resize_images: bool = False
    max_replans: int = 1
    max_reobservations: int = 1

    def __post_init__(self) -> None:
        self.benchmark_root = Path(self.benchmark_root).expanduser().resolve()
        self.output_dir = Path(self.output_dir).expanduser().resolve()
        self.condition = str(self.condition).strip()
        if not self.condition:
            raise ValueError("condition must be a non-empty string")
        self.model_provider = str(
            self.model_provider or DEFAULT_EVALUATION_PROVIDER
        ).strip().casefold()
        if self.model_provider not in {"ollama", "vllm"}:
            raise ValueError("model_provider must be 'ollama' or 'vllm'")
        if self.model is not None:
            self.model = str(self.model).strip()
            if not self.model:
                raise ValueError("model must be a non-empty string when provided")
        elif self.model_provider == DEFAULT_EVALUATION_PROVIDER:
            self.model = DEFAULT_EVALUATION_MODEL
        if self.model_base_url is not None:
            self.model_base_url = normalize_model_base_url(self.model_base_url)
        effective_provider = self.model_provider
        if self.model_base_url is not None and effective_provider != "vllm":
            raise ValueError("model_base_url requires model_provider='vllm'")
        if self.ollama_host is not None and effective_provider == "vllm":
            raise ValueError("ollama_host cannot be used with model_provider='vllm'")
        if self.repetitions <= 0:
            raise ValueError("repetitions must be positive")
        if self.max_scenarios is not None and self.max_scenarios <= 0:
            raise ValueError("max_scenarios must be positive when provided")
        if self.max_replans < 0 or self.max_reobservations < 0:
            raise ValueError("recovery budgets cannot be negative")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive when provided")
        if self.temperature is not None and self.temperature < 0:
            raise ValueError("temperature cannot be negative")
        if self.model_seed is not None and (
            not isinstance(self.model_seed, int)
            or isinstance(self.model_seed, bool)
            or self.model_seed < 0
        ):
            raise ValueError("model_seed must be a non-negative integer or null")
        from .ablations import MEMORY_MODES

        self.memory_mode = str(self.memory_mode).strip().lower()
        if self.memory_mode not in MEMORY_MODES:
            raise ValueError(
                f"memory_mode must be one of {', '.join(MEMORY_MODES)}"
            )
        for name in (
            "families",
            "scene_variants",
            "target_ids",
            "outcomes",
            "scenario_ids",
        ):
            values = tuple(str(value).strip() for value in getattr(self, name))
            if any(not value for value in values):
                raise ValueError(f"{name} cannot contain empty values")
            setattr(self, name, values)
        self.seeds = tuple(int(value) for value in self.seeds)


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    output_dir: Path
    selected_scenarios: int
    planned_trials: int
    executed_trials: int
    skipped_trials: int
    passed_trials: int
    failed_trials: int
    error_trials: int
    summary: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _EpisodeMetadata:
    scenario_id: str
    family: str
    scene_variant: str
    target_id: str
    outcome: str
    seed: int
    control_kind: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _SelectedEpisode:
    episode: BenchmarkEpisode
    metadata: _EpisodeMetadata


@dataclass(frozen=True, slots=True)
class _MemorySnapshot:
    available: bool
    history_ids: tuple[str, ...] = ()
    preference_ids: tuple[str, ...] = ()
    outbox_episode_ids: tuple[str, ...] = ()
    error: str | None = None

    @property
    def history_count(self) -> int:
        return len(self.history_ids)

    @property
    def preference_count(self) -> int:
        return len(self.preference_ids)

    @property
    def outbox_count(self) -> int:
        return len(self.outbox_episode_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "history_count": self.history_count,
            "preference_count": self.preference_count,
            "outbox_count": self.outbox_count,
            "history_ids": list(self.history_ids),
            "preference_ids": list(self.preference_ids),
            "outbox_episode_ids": list(self.outbox_episode_ids),
            "error": self.error,
        }


OrchestratorFactory = Callable[
    [PrefMemConfig, BenchmarkEpisode, BenchmarkEpisodeExecutor, Path, str],
    Any,
]


def default_orchestrator_factory(
    config: PrefMemConfig,
    episode: BenchmarkEpisode,
    executor: BenchmarkEpisodeExecutor,
    trial_dir: Path,
    user_id: str,
) -> HRIOrchestrator:
    """Construct the production orchestrator.

    The extra arguments are explicit to make offline test factories easy to
    write and to make the per-trial isolation boundary inspectable.
    """

    del episode, user_id
    from .evaluation_runtime import build_evaluation_orchestrator

    return build_evaluation_orchestrator(
        config,
        executor,
        trial_dir,
        memory_mode="full",
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _runtime_provenance() -> dict[str, Any]:
    """Return stable runtime metadata including uncommitted source identity."""

    return runtime_provenance(Path(__file__).resolve().parents[2])


def _slug(value: str) -> str:
    result = _SAFE_COMPONENT.sub("-", value.strip()).strip("-._")
    return result or "condition"


def _ensure_disjoint_roots(config: EvaluationConfig) -> None:
    benchmark_root = Path(config.benchmark_root)
    output_dir = Path(config.output_dir)
    if (
        benchmark_root == output_dir
        or benchmark_root in output_dir.parents
        or output_dir in benchmark_root.parents
    ):
        raise EvaluationError(
            "benchmark_root and output_dir must be disjoint: evaluation "
            "artifacts cannot be written inside the immutable benchmark tree, "
            "and the benchmark tree cannot be nested inside the output tree."
        )


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if is_dataclass(value):
        return _json_safe(asdict(value))
    return repr(value)


def _agent_failure_evidence(value: Any) -> list[dict[str, Any]]:
    """Collect structured sub-agent failures retained inside HRI results."""

    found: list[dict[str, Any]] = []
    seen_event_ids: set[str] = set()

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            if all(
                isinstance(item.get(name), str) and item.get(name)
                for name in ("stage", "error_type", "classification", "message")
            ):
                evidence: dict[str, Any] = {
                    name: str(item[name])
                    for name in (
                        "stage",
                        "error_type",
                        "classification",
                        "message",
                    )
                }
                event_id = item.get("event_id")
                if isinstance(event_id, str) and event_id.strip():
                    event_id = event_id.strip()
                    evidence = {
                        "event_id": event_id,
                        **(
                            {"attempt": item["attempt"]}
                            if isinstance(item.get("attempt"), int)
                            and not isinstance(item.get("attempt"), bool)
                            else {}
                        ),
                        **(
                            {"event": str(item["event"])}
                            if isinstance(item.get("event"), str)
                            and str(item["event"]).strip()
                            else {}
                        ),
                        **evidence,
                    }
                    if event_id in seen_event_ids:
                        return
                    seen_event_ids.add(event_id)
                    found.append(evidence)
                else:
                    # Legacy evidence has no event identity.  Do not collapse
                    # semantically identical failures from separate retries:
                    # without an ID there is no proof that two embeddings are
                    # the same event.
                    found.append(evidence)
            for nested in item.values():
                visit(nested)
        elif isinstance(item, (list, tuple)):
            for nested in item:
                visit(nested)

    visit(value)
    return found


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                _json_safe(value),
                stream,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(
        _json_safe(value),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(line)
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
                raise EvaluationResumeError(
                    f"Invalid JSONL at {path}:{line_number}: {error}"
                ) from error
            if not isinstance(value, dict):
                raise EvaluationResumeError(
                    f"JSONL record at {path}:{line_number} is not an object"
                )
            records.append(value)
    return records


def _trial_key(
    condition: str,
    repetition: int,
    scenario_id: str,
) -> tuple[str, int, str]:
    return condition, repetition, scenario_id


def _record_key(record: Mapping[str, Any]) -> tuple[str, int, str]:
    condition = record.get("condition")
    repetition = record.get("repetition")
    scenario_id = record.get("scenario_id")
    if (
        not isinstance(condition, str)
        or not condition.strip()
        or type(repetition) is not int
        or repetition <= 0
        or not isinstance(scenario_id, str)
        or not scenario_id.strip()
    ):
        raise EvaluationResumeError(
            "Every JSONL record must contain condition, repetition, and scenario_id"
        )
    return _trial_key(condition, repetition, scenario_id)


def _validated_resume_score(
    value: Any,
    *,
    key: tuple[str, int, str],
    label: str,
    expected_names: set[str],
) -> dict[str, Mapping[str, Any]]:
    """Validate one strict score payload and return checks keyed by name."""

    if not isinstance(value, Mapping):
        raise EvaluationResumeError(
            f"Resume record {key!r} has no valid {label} object."
        )
    checks = value.get("checks")
    if not isinstance(checks, list) or not checks:
        raise EvaluationResumeError(
            f"Resume record {key!r} has empty {label} check evidence."
        )
    by_name: dict[str, Mapping[str, Any]] = {}
    for index, check in enumerate(checks, start=1):
        if not isinstance(check, Mapping):
            raise EvaluationResumeError(
                f"Resume record {key!r} has an invalid {label} check "
                f"at position {index}."
            )
        if not all(
            name in check
            for name in ("name", "expected", "actual", "evaluated", "passed")
        ):
            raise EvaluationResumeError(
                f"Resume record {key!r} has an incomplete {label} check "
                f"at position {index}."
            )
        name = check.get("name")
        if not isinstance(name, str) or not name.strip():
            raise EvaluationResumeError(
                f"Resume record {key!r} has an unnamed {label} check."
            )
        name = name.strip()
        if name in by_name:
            raise EvaluationResumeError(
                f"Resume record {key!r} has duplicate {label} check "
                f"evidence for {name!r}."
            )
        if check.get("evaluated") is not True or not isinstance(
            check.get("passed"),
            bool,
        ):
            raise EvaluationResumeError(
                f"Resume record {key!r} has non-terminal {label} check "
                f"evidence for {name!r}."
            )
        by_name[name] = check

    actual_names = set(by_name)
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        unexpected = sorted(actual_names - expected_names)
        raise EvaluationResumeError(
            f"Resume record {key!r} has incomplete {label} check coverage "
            f"(missing={missing}, unexpected={unexpected})."
        )

    correct = sum(check.get("passed") is True for check in by_name.values())
    expected_passed = bool(by_name) and correct == len(by_name)
    integer_fields = {
        "correct": correct,
        "total": len(by_name),
        "skipped": 0,
    }
    if value.get("passed") is not expected_passed or any(
        type(value.get(name)) is not int or value.get(name) != expected
        for name, expected in integer_fields.items()
    ):
        raise EvaluationResumeError(
            f"Resume record {key!r} has inconsistent {label} totals."
        )
    return by_name


def _expected_score_exclusions(memory_mode: str) -> list[dict[str, str]]:
    expected: list[dict[str, str]] = []
    if memory_mode in {"no-memory", "preference-only"}:
        expected.append(
            {
                "check_name": "history_delta",
                "reason": (
                    "history writes are disabled by the "
                    f"{memory_mode} intervention"
                ),
            }
        )
    if memory_mode in {"no-memory", "history-only"}:
        expected.append(
            {
                "check_name": "preference_delta_without_consent",
                "reason": (
                    "preference writes are disabled by the "
                    f"{memory_mode} intervention"
                ),
            }
        )
    return expected


def _validated_artifact_status(
    value: Any,
    *,
    key: tuple[str, int, str],
) -> tuple[bool, bool | None]:
    if not isinstance(value, Mapping) or not isinstance(
        value.get("tracked"),
        bool,
    ):
        raise EvaluationResumeError(
            f"Resume record {key!r} has invalid artifact tracking evidence."
        )
    tracked = value["tracked"]
    complete = value.get("complete")
    if tracked:
        if not isinstance(complete, bool):
            raise EvaluationResumeError(
                f"Resume record {key!r} has no terminal tracked-artifact status."
            )
    elif complete is not None:
        raise EvaluationResumeError(
            f"Resume record {key!r} claims artifact completeness without "
            "tracking artifacts."
        )
    return tracked, complete


def _validated_memory_snapshot(
    value: Any,
    *,
    key: tuple[str, int, str],
    label: str,
) -> dict[str, int]:
    if not isinstance(value, Mapping) or value.get("available") is not True:
        raise EvaluationResumeError(
            f"Resume record {key!r} has no available {label} memory snapshot."
        )
    if value.get("error") is not None:
        raise EvaluationResumeError(
            f"Resume record {key!r} has an errored {label} memory snapshot."
        )
    counts: dict[str, int] = {}
    fields = (
        ("history", "history_ids"),
        ("preference", "preference_ids"),
        ("outbox", "outbox_episode_ids"),
    )
    for prefix, ids_name in fields:
        count_name = f"{prefix}_count"
        count = value.get(count_name)
        ids = value.get(ids_name)
        if (
            type(count) is not int
            or count < 0
            or not isinstance(ids, list)
            or any(not isinstance(item, str) or not item for item in ids)
            or len(ids) != len(set(ids))
            or count != len(ids)
        ):
            raise EvaluationResumeError(
                f"Resume record {key!r} has invalid {label} "
                f"{prefix} memory evidence."
            )
        counts[prefix] = count
    return counts


def _validate_completed_resume_evidence(
    record: Mapping[str, Any],
    *,
    key: tuple[str, int, str],
    memory_mode: str,
    selected: _SelectedEpisode,
) -> None:
    raw_checks = _validated_resume_score(
        record.get("raw_score"),
        key=key,
        label="raw_score",
        expected_names=set(STRICT_BENCHMARK_CHECKS),
    )
    expected_exclusions = _expected_score_exclusions(memory_mode)
    if record.get("score_exclusions") != expected_exclusions:
        raise EvaluationResumeError(
            f"Resume record {key!r} has invalid memory-ablation exclusions."
        )
    excluded_names = {
        item["check_name"] for item in expected_exclusions
    }

    tracked, complete = _validated_artifact_status(
        record.get("artifact_status"),
        key=key,
    )
    expected_score_names = set(STRICT_BENCHMARK_CHECKS) - excluded_names
    if tracked:
        expected_score_names.add("evaluation_artifacts_complete")
    score_checks = _validated_resume_score(
        record.get("score"),
        key=key,
        label="score",
        expected_names=expected_score_names,
    )
    for name in set(STRICT_BENCHMARK_CHECKS) - excluded_names:
        if score_checks[name] != raw_checks[name]:
            raise EvaluationResumeError(
                f"Resume record {key!r} altered retained check {name!r} "
                "during ablation scoring."
            )
    if tracked:
        artifact_check = score_checks["evaluation_artifacts_complete"]
        if (
            artifact_check.get("expected") is not True
            or artifact_check.get("actual") is not complete
            or artifact_check.get("passed") is not complete
        ):
            raise EvaluationResumeError(
                f"Resume record {key!r} has an inconsistent tracked-artifact gate."
            )

    score = record["score"]
    if record.get("passed") is not score.get("passed"):
        raise EvaluationResumeError(
            f"Resume record {key!r} has inconsistent terminal pass evidence."
        )

    memory = record.get("memory")
    if not isinstance(memory, Mapping):
        raise EvaluationResumeError(
            f"Resume record {key!r} lacks terminal memory evidence."
        )
    before = _validated_memory_snapshot(
        memory.get("before"),
        key=key,
        label="before",
    )
    after = _validated_memory_snapshot(
        memory.get("after"),
        key=key,
        label="after",
    )
    if any(before.values()):
        raise EvaluationResumeError(
            f"Resume record {key!r} did not start from cold memory."
        )
    deltas: dict[str, int] = {}
    for prefix in ("history", "preference", "outbox"):
        name = f"{prefix}_delta"
        value = memory.get(name)
        expected = after[prefix] - before[prefix]
        if type(value) is not int or value != expected:
            raise EvaluationResumeError(
                f"Resume record {key!r} has inconsistent {name!r} evidence."
            )
        deltas[prefix] = value
    if (
        memory_mode in {"no-memory", "preference-only"}
        and deltas["history"] != 0
    ):
        raise EvaluationResumeError(
            f"Resume record {key!r} wrote history in a disabled ablation."
        )
    if (
        memory_mode in {"no-memory", "history-only"}
        and deltas["preference"] != 0
    ):
        raise EvaluationResumeError(
            f"Resume record {key!r} wrote preferences in a disabled ablation."
        )

    agent_result = record.get("agent_result")
    if not isinstance(agent_result, Mapping):
        raise EvaluationResumeError(
            f"Resume record {key!r} has no replayable agent_result evidence."
        )
    scored_result = copy.deepcopy(dict(agent_result))
    scored_result["history_delta"] = deltas["history"]
    scored_result["preference_delta"] = deltas["preference"]
    try:
        recomputed_raw = score_agent_result(
            selected.episode.oracle(),
            scored_result,
            allow_partial=False,
        )
        recomputed_score, recomputed_exclusions = _ablation_adjusted_score(
            recomputed_raw,
            memory_mode,
        )
        recomputed_score = _with_artifact_completeness_check(
            recomputed_score,
            record["artifact_status"],
        )
    except Exception as error:
        raise EvaluationResumeError(
            f"Resume record {key!r} could not be rescored from its oracle "
            f"evidence: {error}"
        ) from error
    if record.get("raw_score") != _score_to_dict(recomputed_raw):
        raise EvaluationResumeError(
            f"Resume record {key!r} raw_score does not match oracle rescoring."
        )
    if record.get("score_exclusions") != recomputed_exclusions:
        raise EvaluationResumeError(
            f"Resume record {key!r} exclusions do not match oracle rescoring."
        )
    canonical_score = _score_to_dict(recomputed_score)
    if record.get("score") != canonical_score:
        raise EvaluationResumeError(
            f"Resume record {key!r} score does not match oracle rescoring."
        )
    if record.get("passed") is not canonical_score["passed"]:
        raise EvaluationResumeError(
            f"Resume record {key!r} terminal pass does not match oracle rescoring."
        )


def _validate_resume_record(
    record: Mapping[str, Any],
    *,
    config: EvaluationConfig,
    selected_by_id: Mapping[str, _SelectedEpisode],
) -> tuple[str, int, str]:
    """Fail closed when durable JSONL evidence is stale or malformed."""

    key = _record_key(record)
    condition, repetition, scenario_id = key
    if record.get("schema_version") != EVALUATION_SCHEMA_VERSION:
        raise EvaluationResumeError(
            f"Resume record {key!r} has an incompatible schema version."
        )
    if condition != config.condition or not 1 <= repetition <= config.repetitions:
        raise EvaluationResumeError(
            f"Resume record {key!r} is outside the configured evaluation."
        )
    selected = selected_by_id.get(scenario_id)
    if selected is None:
        raise EvaluationResumeError(
            f"Resume record {key!r} is not in the selected packet set."
        )
    for name, expected in selected.metadata.to_dict().items():
        actual = record.get(name)
        if actual != expected or type(actual) is not type(expected):
            raise EvaluationResumeError(
                f"Resume record {key!r} has inconsistent {name!r} metadata."
            )
    expected_seed = (
        config.model_seed + repetition - 1
        if config.model_seed is not None
        else None
    )
    actual_model_seed = record.get("model_seed")
    seed_type_valid = (
        actual_model_seed is None
        if expected_seed is None
        else type(actual_model_seed) is int
    )
    if (
        record.get("memory_mode") != config.memory_mode
        or not seed_type_valid
        or actual_model_seed != expected_seed
        or record.get("user_id") != COLD_EVALUATION_USER_ID
    ):
        raise EvaluationResumeError(
            f"Resume record {key!r} has inconsistent runtime metadata."
        )
    status = record.get("status")
    passed = record.get("passed")
    if status not in {"COMPLETED", "ERROR"} or not isinstance(passed, bool):
        raise EvaluationResumeError(
            f"Resume record {key!r} has an invalid terminal status."
        )
    if status == "ERROR":
        if passed or not isinstance(record.get("error"), Mapping):
            raise EvaluationResumeError(
                f"Resume error record {key!r} has invalid error evidence."
            )
        score = record.get("score")
        if (
            not isinstance(score, Mapping)
            or score.get("passed") is not False
            or score.get("correct") != 0
            or score.get("total") != 0
            or score.get("skipped") != 0
            or score.get("checks") != []
            or record.get("raw_score") is not None
            or record.get("score_exclusions") != []
        ):
            raise EvaluationResumeError(
                f"Resume error record {key!r} has invalid empty scoring evidence."
            )
    else:
        if record.get("error") is not None:
            raise EvaluationResumeError(
                f"Resume record {key!r} has invalid scoring evidence."
            )
        _validate_completed_resume_evidence(
            record,
            key=key,
            memory_mode=config.memory_mode,
            selected=selected,
        )
    if not isinstance(record.get("memory"), Mapping):
        raise EvaluationResumeError(
            f"Resume record {key!r} lacks memory evidence."
        )
    _validated_artifact_status(record.get("artifact_status"), key=key)
    failures = record.get("agent_failures")
    if not isinstance(failures, list) or any(
        not isinstance(failure, Mapping)
        or any(
            not isinstance(failure.get(name), str)
            or not str(failure.get(name)).strip()
            for name in ("stage", "error_type", "classification", "message")
        )
        for failure in failures
    ):
        raise EvaluationResumeError(
            f"Resume record {key!r} has invalid sub-agent failure evidence."
        )
    event_ids = [
        str(failure["event_id"]).strip()
        for failure in failures
        if isinstance(failure.get("event_id"), str)
        and str(failure["event_id"]).strip()
    ]
    if len(event_ids) != len(set(event_ids)):
        raise EvaluationResumeError(
            f"Resume record {key!r} duplicates a sub-agent failure event."
        )
    return key


def _manifest_metadata(episode: BenchmarkEpisode) -> _EpisodeMetadata:
    manifest = episode.manifest
    scene = manifest.get("scene")
    target = manifest.get("target")
    if not isinstance(scene, Mapping) or not isinstance(target, Mapping):
        raise EvaluationDatasetError(
            f"{episode.scenario_id}: manifest scene and target must be objects"
        )
    try:
        return _EpisodeMetadata(
            scenario_id=episode.scenario_id,
            family=str(scene["family"]),
            scene_variant=str(scene["variant"]),
            target_id=str(target["target_id"]),
            outcome=str(manifest["expected_outcome"]),
            seed=int(scene["seed"]),
            control_kind=(
                str(scene["control_kind"])
                if scene.get("control_kind") is not None
                else None
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise EvaluationDatasetError(
            f"{episode.scenario_id}: manifest selection metadata is incomplete"
        ) from error


def _matches(config: EvaluationConfig, item: _EpisodeMetadata) -> bool:
    return (
        (not config.families or item.family in config.families)
        and (
            not config.scene_variants
            or item.scene_variant in config.scene_variants
        )
        and (not config.target_ids or item.target_id in config.target_ids)
        and (not config.outcomes or item.outcome in config.outcomes)
        and (not config.seeds or item.seed in config.seeds)
        and (
            not config.scenario_ids
            or item.scenario_id in config.scenario_ids
        )
        and (config.include_controls or item.control_kind is None)
    )


def _stratified_prefix(
    selected: Sequence[_SelectedEpisode],
    limit: int,
) -> list[_SelectedEpisode]:
    """Take a deterministic, hierarchically balanced pilot sample.

    Family is the primary stratum so a prefix cannot be dominated by whichever
    family happens to sort first.  Within the least-represented family, the
    greedy choice balances outcome/control cells first, followed by target,
    dataset seed, and scene variant.  The opaque scenario ID is only a stable
    final tie-breaker.
    """

    remaining = list(selected)
    result: list[_SelectedEpisode] = []
    family_counts: dict[str, int] = defaultdict(int)
    outcome_kind_counts: dict[tuple[str, str, str], int] = defaultdict(int)
    target_counts: dict[tuple[str, str], int] = defaultdict(int)
    seed_counts: dict[tuple[str, int], int] = defaultdict(int)
    variant_counts: dict[tuple[str, str], int] = defaultdict(int)

    while remaining and len(result) < limit:
        available_families = {item.metadata.family for item in remaining}
        minimum_family_count = min(
            family_counts[family] for family in available_families
        )
        family = min(
            family
            for family in available_families
            if family_counts[family] == minimum_family_count
        )
        candidates = [
            item for item in remaining if item.metadata.family == family
        ]

        def balance_key(item: _SelectedEpisode) -> tuple[Any, ...]:
            metadata = item.metadata
            packet_kind = (
                metadata.control_kind
                if metadata.control_kind is not None
                else "core"
            )
            return (
                outcome_kind_counts[
                    (family, metadata.outcome, packet_kind)
                ],
                target_counts[(family, metadata.target_id)],
                seed_counts[(family, metadata.seed)],
                variant_counts[(family, metadata.scene_variant)],
                metadata.outcome,
                packet_kind,
                metadata.target_id,
                metadata.seed,
                metadata.scene_variant,
                metadata.scenario_id,
            )

        chosen = min(candidates, key=balance_key)
        result.append(chosen)
        remaining.remove(chosen)
        metadata = chosen.metadata
        packet_kind = (
            metadata.control_kind
            if metadata.control_kind is not None
            else "core"
        )
        family_counts[family] += 1
        outcome_kind_counts[(family, metadata.outcome, packet_kind)] += 1
        target_counts[(family, metadata.target_id)] += 1
        seed_counts[(family, metadata.seed)] += 1
        variant_counts[(family, metadata.scene_variant)] += 1
    return result


def _select_episodes(config: EvaluationConfig) -> list[_SelectedEpisode]:
    paths = sorted(Path(config.benchmark_root).rglob("manifest.json"))
    selected: list[_SelectedEpisode] = []
    for path in paths:
        episode = BenchmarkEpisode.from_path(path.parent, config.benchmark_root)
        metadata = _manifest_metadata(episode)
        if _matches(config, metadata):
            selected.append(_SelectedEpisode(episode=episode, metadata=metadata))
    selected.sort(key=lambda item: item.metadata.scenario_id)
    if config.max_scenarios is not None:
        selected = _stratified_prefix(selected, config.max_scenarios)
    if not selected:
        active_filters = {
            "families": list(config.families),
            "scene_variants": list(config.scene_variants),
            "target_ids": list(config.target_ids),
            "outcomes": list(config.outcomes),
            "seeds": list(config.seeds),
            "scenario_ids": list(config.scenario_ids),
            "include_controls": config.include_controls,
            "max_scenarios": config.max_scenarios,
        }
        raise EvaluationDatasetError(
            "Zero benchmark packets were selected from "
            f"{len(paths)} validated packet(s). Active filters: "
            f"{json.dumps(active_filters, sort_keys=True)}"
        )
    if config.scenario_ids:
        found = {item.metadata.scenario_id for item in selected}
        missing = sorted(set(config.scenario_ids) - found)
        if missing:
            raise EvaluationDatasetError(
                "Requested scenario IDs were not found or did not match the other "
                f"filters: {', '.join(missing)}"
            )
    return selected


def _snapshot_memory(orchestrator: Any, user_id: str) -> _MemorySnapshot:
    try:
        memory_agent = getattr(orchestrator, "memory_agent", None)
        history_repository = getattr(memory_agent, "history_repository", None)
        preference_repository = getattr(
            memory_agent, "preference_repository", None
        )
        history_reader = getattr(history_repository, "list_episodes", None)
        preference_reader = getattr(
            preference_repository, "list_preferences", None
        )
        if not callable(history_reader) or not callable(preference_reader):
            return _MemorySnapshot(
                available=False,
                error="orchestrator does not expose readable memory repositories",
            )

        histories = history_reader(user_id)
        try:
            preferences = preference_reader(user_id, include_inactive=True)
        except TypeError:
            # Lightweight injected repositories used by offline tests may only
            # implement the one-argument protocol.
            preferences = preference_reader(user_id)
        history_ids = tuple(
            sorted(
                str(item.get("episode_id", ""))
                for item in histories
                if isinstance(item, Mapping)
            )
        )
        preference_ids = tuple(
            sorted(
                str(item.get("id", ""))
                for item in preferences
                if isinstance(item, Mapping)
            )
        )

        outbox_ids: list[str] = []
        outbox = getattr(orchestrator, "history_outbox", None)
        outbox_reader = getattr(outbox, "list_pending", None)
        if callable(outbox_reader):
            for item in outbox_reader(user_id):
                if isinstance(item, Mapping):
                    source = item.get("source")
                    if isinstance(source, Mapping):
                        outbox_ids.append(str(source.get("episode_id", "")))
                    else:
                        outbox_ids.append(str(item.get("episode_id", "")))
        return _MemorySnapshot(
            available=True,
            history_ids=history_ids,
            preference_ids=preference_ids,
            outbox_episode_ids=tuple(sorted(outbox_ids)),
        )
    except Exception as error:
        return _MemorySnapshot(
            available=False,
            error=f"{type(error).__name__}: {error}",
        )


def _ensure_fresh(snapshot: _MemorySnapshot) -> None:
    if not snapshot.available:
        detail = f" ({snapshot.error})" if snapshot.error else ""
        raise EvaluationIsolationError(
            "Cold-memory isolation could not be verified because the "
            f"orchestrator's stores were not inspectable{detail}."
        )
    if (
        snapshot.history_count
        or snapshot.preference_count
        or snapshot.outbox_count
    ):
        raise EvaluationIsolationError(
            "Cold-memory trial started with non-empty history, preference, or "
            "history-outbox state."
        )


def _memory_delta(
    before: _MemorySnapshot,
    after: _MemorySnapshot,
) -> dict[str, int | None]:
    if not before.available or not after.available:
        return {
            "history_delta": None,
            "preference_delta": None,
            "outbox_delta": None,
        }
    return {
        "history_delta": after.history_count - before.history_count,
        "preference_delta": after.preference_count - before.preference_count,
        "outbox_delta": after.outbox_count - before.outbox_count,
    }


def _build_prefmem_config(
    evaluation: EvaluationConfig,
    episode: BenchmarkEpisode,
    trial_dir: Path,
    user_id: str,
    *,
    repetition: int = 1,
) -> PrefMemConfig:
    memory_dir = trial_dir / "memory"
    config = PrefMemConfig(
        workspace_root=str(evaluation.benchmark_root),
        dataset_path=str(episode.episode.directory),
        history_store_path=str(memory_dir / "history.json"),
        history_outbox_path=str(memory_dir / "history_outbox.json"),
        preference_store_path=str(memory_dir / "preferences.json"),
        user_id=user_id,
        max_replans=evaluation.max_replans,
        max_reobservations=evaluation.max_reobservations,
    )
    config.vision.resize_images = evaluation.resize_images
    for model_config in (
        config.hri,
        config.memory,
        config.planner,
        config.validator,
    ):
        if evaluation.model:
            model_config.model = evaluation.model
        if evaluation.model_provider:
            model_config.provider = evaluation.model_provider
        if evaluation.model_base_url:
            model_config.base_url = evaluation.model_base_url
        if evaluation.ollama_host:
            model_config.host = evaluation.ollama_host
        if evaluation.temperature is not None:
            model_config.temperature = evaluation.temperature
        if evaluation.model_seed is not None:
            model_config.seed = evaluation.model_seed + repetition - 1
        if evaluation.timeout_seconds is not None:
            model_config.timeout_seconds = evaluation.timeout_seconds
    return config


def _prompt_hashes(config: PrefMemConfig) -> dict[str, str | None]:
    hashes: dict[str, str | None] = {}
    for name in ("hri", "memory", "planner", "validator"):
        path = Path(getattr(config, name).system_prompt_path)
        try:
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            hashes[name] = None
    return hashes


def _config_payload(
    config: EvaluationConfig,
    selected: Sequence[_SelectedEpisode],
    orchestrator_factory: OrchestratorFactory,
) -> dict[str, Any]:
    probe = _build_prefmem_config(
        config,
        selected[0].episode,
        Path(config.output_dir) / ".configuration-probe",
        "configuration-probe",
    )
    index_path = Path(config.benchmark_root) / "index.json"
    dataset_digest = (
        hashlib.sha256(index_path.read_bytes()).hexdigest()
        if index_path.is_file()
        else None
    )
    selected_packet_digest = benchmark_content_sha256(
        config.benchmark_root,
        (item.metadata.scenario_id for item in selected),
    )
    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "runtime": _runtime_provenance(),
        "benchmark_root": str(config.benchmark_root),
        "dataset_index_sha256": dataset_digest,
        "selected_packets_sha256": selected_packet_digest,
        "condition": config.condition,
        "orchestrator_factory": callable_provenance(orchestrator_factory),
        "memory_mode": config.memory_mode,
        "repetitions": config.repetitions,
        "filters": {
            "families": list(config.families),
            "scene_variants": list(config.scene_variants),
            "target_ids": list(config.target_ids),
            "outcomes": list(config.outcomes),
            "seeds": list(config.seeds),
            "scenario_ids": list(config.scenario_ids),
            "include_controls": config.include_controls,
            "max_scenarios": config.max_scenarios,
        },
        "selected_scenario_ids": [
            item.metadata.scenario_id for item in selected
        ],
        "shuffle_seed": config.shuffle_seed,
        "selection_policy": (
            "balanced-family-outcome-control-target-seed-variant-v2"
            if config.max_scenarios is not None
            else "all-matching-scenario-id-order-v1"
        ),
        "agent": {
            "models": {
                name: getattr(probe, name).model
                for name in ("hri", "memory", "planner", "validator")
            },
            "providers": {
                name: getattr(probe, name).provider
                for name in ("hri", "memory", "planner", "validator")
            },
            "base_urls": {
                name: effective_model_base_url(getattr(probe, name))
                for name in ("hri", "memory", "planner", "validator")
            },
            "hosts": {
                name: getattr(probe, name).host
                for name in ("hri", "memory", "planner", "validator")
            },
            "timeouts_seconds": {
                name: getattr(probe, name).timeout_seconds
                for name in ("hri", "memory", "planner", "validator")
            },
            "temperatures": {
                name: getattr(probe, name).temperature
                for name in ("hri", "memory", "planner", "validator")
            },
            "model_seed_base": config.model_seed,
            "model_seeds_by_repetition": (
                [
                    config.model_seed + repetition - 1
                    for repetition in range(1, config.repetitions + 1)
                ]
                if config.model_seed is not None
                else [None for _ in range(config.repetitions)]
            ),
            "prompt_sha256": _prompt_hashes(probe),
            "resize_images": config.resize_images,
            "max_replans": config.max_replans,
            "max_reobservations": config.max_reobservations,
        },
        "scoring": {"allow_partial": False},
    }


def _prepare_run_config(
    config: EvaluationConfig,
    payload: Mapping[str, Any],
) -> None:
    path = Path(config.output_dir) / "run_config.json"
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise EvaluationResumeError(
                f"Could not read existing run configuration: {error}"
            ) from error
        existing_payload = (
            existing.get("evaluation")
            if isinstance(existing, Mapping)
            else None
        )
        if existing_payload != payload:
            raise EvaluationResumeError(
                "Output directory belongs to an incompatible evaluation "
                "configuration or lacks complete provenance. Use a new output "
                "directory."
            )
        return
    _atomic_json(
        path,
        {
            "created_at": _now(),
            "evaluation": payload,
        },
    )


def _score_to_dict(report: ScoreReport) -> dict[str, Any]:
    return {
        "passed": report.passed,
        "correct": report.correct,
        "total": report.total,
        "skipped": report.skipped,
        "checks": [copy.deepcopy(check) for check in report.checks],
    }


def _evaluation_artifact_status(orchestrator: Any) -> dict[str, Any]:
    provider = getattr(orchestrator, "evaluation_artifact_status", None)
    if not callable(provider):
        # Injectable offline factories are allowed to omit production telemetry.
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


def _with_artifact_completeness_check(
    score: ScoreReport,
    artifact_status: Mapping[str, Any],
) -> ScoreReport:
    if artifact_status.get("tracked") is not True:
        return score
    complete = artifact_status.get("complete") is True
    check = {
        "name": "evaluation_artifacts_complete",
        "expected": True,
        "actual": complete,
        "evaluated": True,
        "passed": complete,
    }
    return ScoreReport(
        scenario_id=score.scenario_id,
        passed=score.passed and complete,
        checks=tuple([*score.checks, check]),
    )


def _ablation_adjusted_score(
    raw_report: ScoreReport,
    memory_mode: str,
) -> tuple[ScoreReport, list[dict[str, str]]]:
    excluded_names: dict[str, str] = {}
    if memory_mode in {"no-memory", "preference-only"}:
        excluded_names["history_delta"] = (
            f"history writes are disabled by the {memory_mode} intervention"
        )
    if memory_mode in {"no-memory", "history-only"}:
        excluded_names["preference_delta_without_consent"] = (
            f"preference writes are disabled by the {memory_mode} intervention"
        )
    exclusions = [
        {"check_name": name, "reason": reason}
        for name, reason in excluded_names.items()
        if any(check.get("name") == name for check in raw_report.checks)
    ]
    retained = tuple(
        copy.deepcopy(check)
        for check in raw_report.checks
        if check.get("name") not in excluded_names
    )
    evaluated = [
        check for check in retained if check.get("evaluated") is True
    ]
    skipped = [
        check for check in retained if check.get("evaluated") is not True
    ]
    adjusted = ScoreReport(
        scenario_id=raw_report.scenario_id,
        passed=(
            bool(evaluated)
            and all(check.get("passed") is True for check in evaluated)
            and not skipped
        ),
        checks=retained,
    )
    return adjusted, exclusions


def _check_failure_category(name: str) -> str:
    if name == "evaluation_artifacts_complete":
        return "EVALUATION_ARTIFACT"
    if name.startswith("hri_"):
        return "HRI"
    if name.startswith("planner_"):
        return "PLANNER"
    if name in {"validator_called", "validator_outcome", "task_complete"}:
        return "VALIDATOR"
    if name in {"execution_status", "vla_dispatches"}:
        return "EXECUTION"
    if name == "recovery":
        return "RECOVERY"
    if name.startswith("history_") or name.startswith("preference_"):
        return "MEMORY"
    return "OTHER"


def _score_failure_details(score: ScoreReport) -> dict[str, Any]:
    failed = [
        str(check.get("name", ""))
        for check in score.checks
        if check.get("evaluated") is True and check.get("passed") is not True
    ]
    skipped = [
        str(check.get("name", ""))
        for check in score.checks
        if check.get("evaluated") is not True
    ]
    by_category: dict[str, list[str]] = {}
    for name in failed:
        by_category.setdefault(_check_failure_category(name), []).append(name)
    if "evaluation_artifacts_complete" in failed:
        primary = "ARTIFACT_INCOMPLETE"
    elif failed:
        primary = "STRICT_CHECK_FAILURE"
    elif skipped or not score.passed:
        primary = "MISSING_EVIDENCE"
    else:
        primary = None
    return {
        "failure_category": primary,
        "failure_categories": sorted(by_category),
        "failed_check_names": failed,
        "skipped_check_names": skipped,
        "failed_checks_by_category": {
            key: sorted(value) for key, value in sorted(by_category.items())
        },
    }


def _exception_failure_category(error: Exception, phase: str) -> str:
    name = type(error).__name__.casefold()
    message = str(error).casefold()
    if isinstance(error, EvaluationIsolationError):
        return "MEMORY_ISOLATION"
    if isinstance(error, TimeoutError) or "timeout" in name or "timed out" in message:
        return "TIMEOUT"
    if "contract" in name or "contract" in message:
        return "CONTRACT_ERROR"
    if phase == "scoring":
        return "SCORING_ERROR"
    if phase in {"memory_snapshot", "memory_isolation"}:
        return "MEMORY_ERROR"
    if any(
        marker in name or marker in message
        for marker in (
            "connection",
            "connecterror",
            "ollama",
            "openai",
            "vllm",
            "apierror",
            "apistatus",
            "ratelimit",
            "http",
            "model unavailable",
        )
    ):
        return "MODEL_OR_CONNECTION_ERROR"
    return "RUNTIME_ERROR"


def _trial_record(
    *,
    config: EvaluationConfig,
    selected: _SelectedEpisode,
    repetition: int,
    factory: OrchestratorFactory,
) -> dict[str, Any]:
    metadata = selected.metadata
    condition_slug = _slug(config.condition)
    trial_name = (
        f"{condition_slug}__r{repetition:03d}__{metadata.scenario_id}"
    )
    trial_root = Path(config.output_dir) / "trials" / trial_name
    attempt_dir = trial_root / f"attempt-{uuid.uuid4().hex}"
    attempt_dir.mkdir(parents=True, exist_ok=False)
    # Every cold attempt has isolated stores, so one fixed blinded participant
    # ID keeps paired conditions/model seeds prompt-identical without exposing a
    # scenario ID or introducing a random UUID confound.
    user_id = COLD_EVALUATION_USER_ID
    prefmem_config = _build_prefmem_config(
        config,
        selected.episode,
        attempt_dir,
        user_id,
        repetition=repetition,
    )
    executor = BenchmarkEpisodeExecutor(selected.episode)
    started_at = _now()
    started = time.perf_counter()
    runtime_result: Any = None
    before = _MemorySnapshot(available=False, error="trial did not start")
    after = before
    artifact_status: dict[str, Any] = {
        "tracked": False,
        "complete": None,
    }
    phase = "orchestrator_construction"
    record: dict[str, Any] = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "condition": config.condition,
        "repetition": repetition,
        **metadata.to_dict(),
        "status": "ERROR",
        "started_at": started_at,
        "trial_artifact_dir": str(attempt_dir),
        "user_id": user_id,
        "memory_mode": config.memory_mode,
        "model_seed": (
            config.model_seed + repetition - 1
            if config.model_seed is not None
            else None
        ),
    }
    try:
        orchestrator = factory(
            prefmem_config,
            selected.episode,
            executor,
            attempt_dir,
            user_id,
        )
        if config.memory_mode != "full":
            from .ablations import apply_memory_mode

            memory_agent = getattr(orchestrator, "memory_agent", None)
            if memory_agent is None:
                raise EvaluationIsolationError(
                    "The selected memory ablation requires an orchestrator "
                    "with a memory_agent."
                )
            orchestrator.memory_agent = apply_memory_mode(
                memory_agent,
                config.memory_mode,
            )
        phase = "memory_snapshot"
        before = _snapshot_memory(orchestrator, user_id)
        phase = "memory_isolation"
        _ensure_fresh(before)

        # This is the sole user turn in the cold endpoint study.  Clarification
        # is deliberately not auto-answered: an unnecessary ASK is a scored
        # HRI failure for the benchmark's canonical explicit instructions.
        phase = "agent_runtime"
        runtime_result = orchestrator.handle_user_message(
            selected.episode.instruction
        )
        phase = "memory_snapshot"
        after = _snapshot_memory(orchestrator, user_id)
        deltas = _memory_delta(before, after)
        if not isinstance(runtime_result, Mapping):
            raise TypeError("HRI orchestrator result must be an object")
        scored_result = copy.deepcopy(dict(runtime_result))
        scored_result["history_delta"] = deltas["history_delta"]
        scored_result["preference_delta"] = deltas["preference_delta"]

        # Oracle access occurs only after the agent has returned and is never
        # copied into an agent request or memory repository.
        phase = "scoring"
        raw_score = score_agent_result(
            selected.episode.oracle(),
            scored_result,
            allow_partial=False,
        )
        score, score_exclusions = _ablation_adjusted_score(
            raw_score,
            config.memory_mode,
        )
        artifact_status = _evaluation_artifact_status(orchestrator)
        score = _with_artifact_completeness_check(
            score,
            artifact_status,
        )
        failure_details = _score_failure_details(score)
        record.update(
            {
                "status": "COMPLETED",
                "passed": score.passed,
                "agent_result": _json_safe(runtime_result),
                "memory": {
                    "before": before.to_dict(),
                    "after": after.to_dict(),
                    **deltas,
                },
                "score": _score_to_dict(score),
                "raw_score": _score_to_dict(raw_score),
                "score_exclusions": score_exclusions,
                "artifact_status": artifact_status,
                "agent_failures": _agent_failure_evidence(runtime_result),
                **failure_details,
                "error": None,
            }
        )
    except Exception as error:
        after = (
            _snapshot_memory(locals()["orchestrator"], user_id)
            if "orchestrator" in locals()
            else after
        )
        if "orchestrator" in locals():
            artifact_status = _evaluation_artifact_status(orchestrator)
        failure_category = _exception_failure_category(error, phase)
        record.update(
            {
                "status": "ERROR",
                "passed": False,
                "agent_result": _json_safe(runtime_result),
                "memory": {
                    "before": before.to_dict(),
                    "after": after.to_dict(),
                    **_memory_delta(before, after),
                },
                "score": {
                    "passed": False,
                    "correct": 0,
                    "total": 0,
                    "skipped": 0,
                    "checks": [],
                },
                "raw_score": None,
                "score_exclusions": [],
                "artifact_status": artifact_status,
                "agent_failures": _agent_failure_evidence(runtime_result),
                "failure_category": failure_category,
                "failure_categories": [failure_category],
                "failed_check_names": [],
                "skipped_check_names": [],
                "failed_checks_by_category": {},
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                    "phase": phase,
                    "traceback": traceback.format_exc(),
                },
            }
        )
    record["finished_at"] = _now()
    record["duration_seconds"] = round(time.perf_counter() - started, 6)
    _atomic_json(attempt_dir / "result.json", record)
    return record


def _check_rows(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        base = {
            "condition": record.get("condition"),
            "repetition": record.get("repetition"),
            "scenario_id": record.get("scenario_id"),
            "family": record.get("family"),
            "scene_variant": record.get("scene_variant"),
            "target_id": record.get("target_id"),
            "outcome": record.get("outcome"),
            "seed": record.get("seed"),
            "control_kind": record.get("control_kind"),
            "trial_status": record.get("status"),
            "trial_passed": record.get("passed"),
            "failure_category": record.get("failure_category"),
        }
        score = record.get("score")
        checks = score.get("checks", []) if isinstance(score, Mapping) else []
        if not isinstance(checks, list) or not checks:
            rows.append(
                {
                    **base,
                    "check_name": "",
                    "expected": "",
                    "actual": "",
                    "evaluated": "",
                    "check_passed": "",
                    "hard_gate": "",
                }
            )
            continue
        for check in checks:
            if not isinstance(check, Mapping):
                continue
            rows.append(
                {
                    **base,
                    "check_name": check.get("name"),
                    "expected": json.dumps(
                        _json_safe(check.get("expected")),
                        sort_keys=True,
                        ensure_ascii=False,
                    ),
                    "actual": json.dumps(
                        _json_safe(check.get("actual")),
                        sort_keys=True,
                        ensure_ascii=False,
                    ),
                    "evaluated": check.get("evaluated"),
                    "check_passed": check.get("passed"),
                    "hard_gate": check.get("name") in HARD_GATE_CHECKS,
                }
            )
    return rows


def _write_checks_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    columns = [
        "condition",
        "repetition",
        "scenario_id",
        "family",
        "scene_variant",
        "target_id",
        "outcome",
        "seed",
        "control_kind",
        "trial_status",
        "trial_passed",
        "failure_category",
        "check_name",
        "expected",
        "actual",
        "evaluated",
        "check_passed",
        "hard_gate",
    ]
    rows = _check_rows(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


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


def _rate_metric(successes: int, total: int) -> dict[str, Any]:
    return {
        "successes": successes,
        "trials": total,
        "rate": successes / total if total else None,
        "wilson_ci95": _wilson_ci95(successes, total),
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


def _scene_cluster_bootstrap_ci95(
    records: Sequence[Mapping[str, Any]],
    *,
    replicate_count: int = SCENE_CLUSTER_BOOTSTRAP_REPLICATES,
    random_seed: int = SCENE_CLUSTER_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Bootstrap strict pass rate while keeping scene repetitions nested.

    Core counterfactual packets share one initial physical scene across targets
    and outcomes, so their cluster is ``(family, scene_variant, seed)``.
    Already-satisfied controls use a target-specific goal state as their
    initial observation, so ``target_id`` is additionally required for those
    packets. Resampling a cluster keeps every model repetition nested.
    """

    clusters: dict[tuple[str, ...], tuple[int, int]] = {}
    cluster_records: dict[tuple[str, ...], list[Mapping[str, Any]]] = {}
    for record in records:
        common = (
            str(record.get("family", "")),
            str(record.get("scene_variant", "")),
            str(record.get("seed", "")),
        )
        key = (
            (
                "control",
                *common,
                str(record.get("target_id", "")),
            )
            if record.get("control_kind") is not None
            else ("core", *common)
        )
        cluster_records.setdefault(key, []).append(record)
    for key, items in cluster_records.items():
        clusters[key] = (
            sum(item.get("passed") is True for item in items),
            len(items),
        )

    keys = sorted(clusters)
    if not keys or replicate_count <= 0:
        return {
            "method": "percentile_scene_cluster_bootstrap",
            "confidence": 0.95,
            "cluster_unit": [
                "packet_kind",
                "family",
                "scene_variant",
                "seed",
                "target_id_if_control",
            ],
            "cluster_count": len(keys),
            "replicate_count": 0,
            "random_seed": random_seed,
            "lower": None,
            "upper": None,
        }

    generator = random.Random(random_seed)
    estimates: list[float] = []
    for _ in range(replicate_count):
        sampled = [generator.choice(keys) for _ in range(len(keys))]
        successes = sum(clusters[key][0] for key in sampled)
        observations = sum(clusters[key][1] for key in sampled)
        estimates.append(successes / observations)
    return {
        "method": "percentile_scene_cluster_bootstrap",
        "confidence": 0.95,
        "cluster_unit": [
            "packet_kind",
            "family",
            "scene_variant",
            "seed",
            "target_id_if_control",
        ],
        "cluster_count": len(keys),
        "replicate_count": replicate_count,
        "random_seed": random_seed,
        "lower": _percentile(estimates, 0.025),
        "upper": _percentile(estimates, 0.975),
    }


def _record_checks(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    score = record.get("score")
    raw = score.get("checks", []) if isinstance(score, Mapping) else []
    if not isinstance(raw, list):
        return []
    return [check for check in raw if isinstance(check, Mapping)]


def _named_check(
    record: Mapping[str, Any],
    name: str,
) -> Mapping[str, Any] | None:
    return next(
        (
            check
            for check in _record_checks(record)
            if str(check.get("name", "")) == name
        ),
        None,
    )


def _summarize(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def bucket(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        completed = sum(item.get("status") == "COMPLETED" for item in items)
        errors = sum(item.get("status") == "ERROR" for item in items)
        passed = sum(item.get("passed") is True for item in items)
        durations = [
            float(item["duration_seconds"])
            for item in items
            if isinstance(item.get("duration_seconds"), (int, float))
            and not isinstance(item.get("duration_seconds"), bool)
        ]
        return {
            "trials": len(items),
            "completed": completed,
            "errors": errors,
            "passed": passed,
            "failed": len(items) - passed,
            "strict_pass_rate": passed / len(items) if items else None,
            "strict_pass_rate_wilson_ci95": _wilson_ci95(
                passed,
                len(items),
            ),
            "duration_seconds": {
                "observations": len(durations),
                "p50": _percentile(durations, 0.50),
                "p95": _percentile(durations, 0.95),
            },
        }

    dimensions: dict[str, dict[str, list[Mapping[str, Any]]]] = {
        "by_family": {},
        "by_outcome": {},
        "by_scene_variant": {},
        "by_target": {},
        "by_seed": {},
        "by_control_kind": {},
    }
    check_buckets: dict[str, list[Mapping[str, Any]]] = {}
    check_expected_counts: dict[str, int] = {}
    all_checks: list[Mapping[str, Any]] = []
    failure_categories: dict[str, int] = {}
    agent_failure_classifications: dict[str, int] = {}
    agent_failure_stages: dict[str, int] = {}
    agent_failure_trial_classifications: dict[str, int] = {}
    agent_failure_trial_stages: dict[str, int] = {}
    agent_failure_event_count = 0
    agent_failure_affected_trials = 0
    exclusion_count = 0
    complete_check_evidence_trials = 0
    for record in records:
        dimension_keys = {
            "by_family": str(record.get("family", "")),
            "by_outcome": str(record.get("outcome", "")),
            "by_scene_variant": str(record.get("scene_variant", "")),
            "by_target": str(record.get("target_id", "")),
            "by_seed": str(record.get("seed", "")),
            "by_control_kind": (
                str(record.get("control_kind"))
                if record.get("control_kind") is not None
                else "regular"
            ),
        }
        for dimension, key in dimension_keys.items():
            dimensions[dimension].setdefault(key, []).append(record)
        category = record.get("failure_category")
        if category:
            key = str(category)
            failure_categories[key] = failure_categories.get(key, 0) + 1
        failures = record.get("agent_failures")
        if isinstance(failures, list):
            trial_classifications: set[str] = set()
            trial_stages: set[str] = set()
            trial_event_count = 0
            for failure in failures:
                if not isinstance(failure, Mapping):
                    continue
                trial_event_count += 1
                classification = str(
                    failure.get("classification", "")
                ).strip()
                stage = str(failure.get("stage", "")).strip()
                if classification:
                    trial_classifications.add(classification)
                    agent_failure_classifications[classification] = (
                        agent_failure_classifications.get(classification, 0)
                        + 1
                    )
                if stage:
                    trial_stages.add(stage)
                    agent_failure_stages[stage] = (
                        agent_failure_stages.get(stage, 0) + 1
                    )
            agent_failure_event_count += trial_event_count
            if trial_event_count:
                agent_failure_affected_trials += 1
            for classification in trial_classifications:
                agent_failure_trial_classifications[classification] = (
                    agent_failure_trial_classifications.get(
                        classification,
                        0,
                    )
                    + 1
                )
            for stage in trial_stages:
                agent_failure_trial_stages[stage] = (
                    agent_failure_trial_stages.get(stage, 0) + 1
                )
        exclusions = record.get("score_exclusions")
        if isinstance(exclusions, list):
            exclusion_count += len(exclusions)
        record_checks = _record_checks(record)
        for check in record_checks:
            all_checks.append(check)
            check_buckets.setdefault(str(check.get("name", "")), []).append(
                check
            )
        expected_names = set(STRICT_BENCHMARK_CHECKS)
        memory_mode = str(record.get("memory_mode", "full"))
        if memory_mode in {"no-memory", "preference-only"}:
            expected_names.discard("history_delta")
        if memory_mode in {"no-memory", "history-only"}:
            expected_names.discard("preference_delta_without_consent")
        artifact_status = record.get("artifact_status")
        if (
            isinstance(artifact_status, Mapping)
            and artifact_status.get("tracked") is True
        ):
            expected_names.add("evaluation_artifacts_complete")
        for name in expected_names:
            check_expected_counts[name] = check_expected_counts.get(name, 0) + 1
        evaluated_names = {
            str(check.get("name", ""))
            for check in record_checks
            if check.get("evaluated") is True
        }
        if expected_names <= evaluated_names:
            complete_check_evidence_trials += 1

    check_summary: dict[str, Any] = {}
    for name in sorted(set(check_buckets) | set(check_expected_counts)):
        checks = check_buckets.get(name, [])
        evaluated = [item for item in checks if item.get("evaluated") is True]
        passed = sum(item.get("passed") is True for item in evaluated)
        expected_observations = check_expected_counts.get(name, len(checks))
        emitted_observations = min(len(checks), expected_observations)
        check_summary[name] = {
            "observations": len(checks),
            "expected_observations": expected_observations,
            "missing_evidence": max(
                0,
                expected_observations - emitted_observations,
            ),
            "coverage_rate": (
                emitted_observations / expected_observations
                if expected_observations
                else None
            ),
            "evaluated": len(evaluated),
            "skipped": len(checks) - len(evaluated),
            "passed": passed,
            "failed": len(evaluated) - passed,
            "accuracy": passed / len(evaluated) if evaluated else None,
            "accuracy_wilson_ci95": _wilson_ci95(passed, len(evaluated)),
            "unconditional_accuracy": (
                passed / expected_observations
                if expected_observations
                else None
            ),
            "unconditional_accuracy_wilson_ci95": _wilson_ci95(
                passed,
                expected_observations,
            ),
            "hard_gate": name in HARD_GATE_CHECKS,
        }

    evaluated_checks = [
        check for check in all_checks if check.get("evaluated") is True
    ]
    skipped_checks = [
        check for check in all_checks if check.get("evaluated") is not True
    ]
    passed_checks = sum(
        check.get("passed") is True for check in evaluated_checks
    )
    hard_gate_checks = [
        check
        for check in all_checks
        if str(check.get("name", "")) in HARD_GATE_CHECKS
    ]
    hard_gate_evaluated = [
        check for check in hard_gate_checks if check.get("evaluated") is True
    ]
    hard_gate_failed = [
        check
        for check in hard_gate_evaluated
        if check.get("passed") is not True
    ]

    unsafe_records = [
        record for record in records if record.get("outcome") == "unsafe"
    ]
    safe_abort_successes = 0
    validator_suppressed = 0
    validator_after_unsafe = 0
    for record in unsafe_records:
        required = [
            _named_check(record, name)
            for name in ("execution_status", "validator_called", "recovery")
        ]
        if all(
            check is not None and check.get("passed") is True
            for check in required
        ):
            safe_abort_successes += 1
        called = _named_check(record, "validator_called")
        if called is not None and called.get("actual") is False:
            validator_suppressed += 1
        if called is not None and called.get("actual") is True:
            validator_after_unsafe += 1

    validator_called_checks = [
        check
        for record in records
        if (check := _named_check(record, "validator_called")) is not None
        and check.get("evaluated") is True
    ]
    validator_called_passes = sum(
        check.get("passed") is True for check in validator_called_checks
    )
    validator_outcome_checks: list[Mapping[str, Any]] = []
    for record in records:
        # Outcome accuracy is conditional on the oracle requiring Validator to
        # run.  Unsafe packets deliberately suppress Validator and carry a
        # null expected outcome; counting their null/null contract check would
        # inflate both accuracy and the confusion matrix.
        called_check = _named_check(record, "validator_called")
        outcome_check = _named_check(record, "validator_outcome")
        if (
            called_check is not None
            and called_check.get("expected") is True
            and outcome_check is not None
            and outcome_check.get("evaluated") is True
        ):
            validator_outcome_checks.append(outcome_check)
    validator_outcome_passes = sum(
        check.get("passed") is True for check in validator_outcome_checks
    )
    validator_outcome_eligible = sum(
        str(record.get("outcome", "")).strip().casefold() != "unsafe"
        for record in records
    )
    validator_confusion: dict[str, dict[str, int]] = {}
    for check in validator_outcome_checks:
        expected = (
            "null" if check.get("expected") is None else str(check.get("expected"))
        )
        actual = (
            "null" if check.get("actual") is None else str(check.get("actual"))
        )
        row = validator_confusion.setdefault(expected, {})
        row[actual] = row.get(actual, 0) + 1
    false_task_completions = 0
    non_success_terminal_cases = 0
    task_completion_states = {
        "reported_true": 0,
        "reported_false": 0,
        "reported_none": 0,
        "missing_or_unevaluated": 0,
        "invalid": 0,
    }
    for record in records:
        expected_outcome = str(record.get("outcome", "")).strip().casefold()
        if not expected_outcome or expected_outcome == "success":
            continue

        non_success_terminal_cases += 1
        check = _named_check(record, "task_complete")
        if (
            check is None
            or check.get("evaluated") is not True
            or "actual" not in check
        ):
            task_completion_states["missing_or_unevaluated"] += 1
            continue

        actual = check["actual"]
        if actual is True:
            false_task_completions += 1
            task_completion_states["reported_true"] += 1
        elif actual is False:
            task_completion_states["reported_false"] += 1
        elif actual is None:
            task_completion_states["reported_none"] += 1
        else:
            task_completion_states["invalid"] += 1

    control_records = [
        record
        for record in records
        if record.get("control_kind") is not None
    ]
    zero_dispatch_successes = 0
    control_dispatch_violations = 0
    already_satisfied_successes = 0
    for record in control_records:
        dispatch = _named_check(record, "vla_dispatches")
        if dispatch is not None and dispatch.get("passed") is True:
            zero_dispatch_successes += 1
        if (
            dispatch is not None
            and dispatch.get("evaluated") is True
            and dispatch.get("actual") != 0
        ):
            control_dispatch_violations += 1
        planner = _named_check(record, "planner_status")
        if planner is not None and planner.get("passed") is True:
            already_satisfied_successes += 1

    dimension_summaries = {
        dimension: {
            key: bucket(items)
            for key, items in sorted(groups.items())
        }
        for dimension, groups in dimensions.items()
    }
    overall_summary = bucket(records)
    overall_summary[
        "strict_pass_rate_scene_cluster_bootstrap_ci95"
    ] = _scene_cluster_bootstrap_ci95(records)
    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "updated_at": _now(),
        "overall": overall_summary,
        **dimension_summaries,
        "failure_categories": dict(sorted(failure_categories.items())),
        "subagent_failures": {
            "failure_events": agent_failure_event_count,
            "affected_trials": agent_failure_affected_trials,
            "affected_trial_rate": (
                agent_failure_affected_trials / len(records)
                if records
                else None
            ),
            "events_by_classification": dict(
                sorted(agent_failure_classifications.items())
            ),
            "affected_trials_by_classification": dict(
                sorted(agent_failure_trial_classifications.items())
            ),
            "events_by_stage": dict(
                sorted(agent_failure_stages.items())
            ),
            "affected_trials_by_stage": dict(
                sorted(agent_failure_trial_stages.items())
            ),
            # Backwards-compatible aliases. These count failure events, not
            # trials; the affected-trial mappings above are the denominators
            # for prevalence reporting.
            "by_classification": dict(
                sorted(agent_failure_classifications.items())
            ),
            "by_stage": dict(sorted(agent_failure_stages.items())),
        },
        "check_totals": {
            "observations": len(all_checks),
            "expected_observations": sum(check_expected_counts.values()),
            "missing_evidence": max(
                0,
                sum(check_expected_counts.values()) - len(all_checks),
            ),
            "complete_evidence_trials": complete_check_evidence_trials,
            "incomplete_evidence_trials": (
                len(records) - complete_check_evidence_trials
            ),
            "evaluated": len(evaluated_checks),
            "skipped": len(skipped_checks),
            "passed": passed_checks,
            "failed": len(evaluated_checks) - passed_checks,
            "ablation_exclusions": exclusion_count,
            "hard_gate_observations": len(hard_gate_checks),
            "hard_gate_evaluated": len(hard_gate_evaluated),
            "hard_gate_skipped": len(hard_gate_checks)
            - len(hard_gate_evaluated),
            "hard_gate_failed": len(hard_gate_failed),
        },
        "headline": {
            "safety": {
                "unsafe_abort_correct": _rate_metric(
                    safe_abort_successes,
                    len(unsafe_records),
                ),
                "validator_suppressed_after_unsafe": _rate_metric(
                    validator_suppressed,
                    len(unsafe_records),
                ),
                "validator_called_after_unsafe": validator_after_unsafe,
            },
            "validator": {
                "call_decision_accuracy": _rate_metric(
                    validator_called_passes,
                    len(records),
                ),
                "outcome_accuracy": _rate_metric(
                    validator_outcome_passes,
                    validator_outcome_eligible,
                ),
                "confusion_matrix": {
                    expected: dict(sorted(actuals.items()))
                    for expected, actuals in sorted(
                        validator_confusion.items()
                    )
                },
                "false_task_completions": false_task_completions,
                "non_success_terminal_cases": non_success_terminal_cases,
                # Backward-compatible alias retained for existing analysis
                # scripts; its denominator now includes absent/unevaluated
                # task-complete evidence.
                "non_success_task_checks": non_success_terminal_cases,
                "false_task_completion_rate": _rate_metric(
                    false_task_completions,
                    non_success_terminal_cases,
                ),
                "task_completion_states": task_completion_states,
            },
            "controls": {
                "strict_pass": _rate_metric(
                    sum(
                        record.get("passed") is True
                        for record in control_records
                    ),
                    len(control_records),
                ),
                "zero_vla_dispatches": _rate_metric(
                    zero_dispatch_successes,
                    len(control_records),
                ),
                "already_satisfied_planner_status": _rate_metric(
                    already_satisfied_successes,
                    len(control_records),
                ),
                "dispatch_violations": control_dispatch_violations,
            },
        },
        "checks": check_summary,
    }


def _write_derived_artifacts(
    output_dir: Path,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    summary = _summarize(records)
    _write_checks_csv(output_dir / "checks.csv", records)
    _atomic_json(output_dir / "summary.json", summary)
    return summary


def run_cold_memory_evaluation(
    config: EvaluationConfig,
    *,
    orchestrator_factory: OrchestratorFactory = default_orchestrator_factory,
    show_progress: bool = False,
) -> EvaluationReport:
    """Run or resume a randomized, repeated cold-memory endpoint study.

    Dataset validation is mandatory.  Every attempted trial, including model
    exceptions and contract failures, is appended durably to JSONL and therefore
    will not be silently rerun on resume.
    """

    _ensure_disjoint_roots(config)
    validation: ValidationReport = validate_benchmark(config.benchmark_root)
    if not validation.valid:
        detail = "\n".join(validation.errors[:20])
        suffix = (
            f"\n... and {len(validation.errors) - 20} more errors"
            if len(validation.errors) > 20
            else ""
        )
        raise EvaluationDatasetError(
            f"Benchmark validation failed with {len(validation.errors)} "
            f"error(s):\n{detail}{suffix}"
        )

    selected = _select_episodes(config)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = _config_payload(config, selected, orchestrator_factory)
    _prepare_run_config(config, payload)

    jsonl_path = output_dir / "episode_results.jsonl"
    existing_records = _load_jsonl(jsonl_path)
    if existing_records and not config.resume:
        raise EvaluationResumeError(
            "episode_results.jsonl already exists and resume is disabled. "
            "Use a new output directory."
        )
    completed_keys: set[tuple[str, int, str]] = set()
    selected_by_id = {
        item.metadata.scenario_id: item for item in selected
    }
    for record in existing_records:
        key = _validate_resume_record(
            record,
            config=config,
            selected_by_id=selected_by_id,
        )
        if key in completed_keys:
            raise EvaluationResumeError(
                f"Duplicate resume key in episode_results.jsonl: {key!r}"
            )
        completed_keys.add(key)

    trials = [
        (repetition, item)
        for repetition in range(1, config.repetitions + 1)
        for item in selected
    ]
    random.Random(config.shuffle_seed).shuffle(trials)
    planned = len(trials)
    skipped = 0
    executed = 0
    progress = tqdm(
        total=planned,
        initial=len(completed_keys),
        desc="Cold evaluation",
        unit="trial",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    current_records = list(existing_records)
    try:
        for repetition, item in trials:
            key = _trial_key(
                config.condition,
                repetition,
                item.metadata.scenario_id,
            )
            if key in completed_keys:
                skipped += 1
                continue
            record = _trial_record(
                config=config,
                selected=item,
                repetition=repetition,
                factory=orchestrator_factory,
            )
            _append_jsonl(jsonl_path, record)
            current_records.append(record)
            completed_keys.add(key)
            executed += 1
            progress.update(1)
            if config.fail_fast and record["status"] == "ERROR":
                raise EvaluationError(
                    "Evaluation stopped after a trial error because fail_fast "
                    "is enabled."
                )
    finally:
        progress.close()
        summary = _write_derived_artifacts(output_dir, current_records)

    relevant_records = [
        record
        for record in current_records
        if str(record.get("condition")) == config.condition
        and str(record.get("scenario_id"))
        in {item.metadata.scenario_id for item in selected}
        and 1 <= int(record.get("repetition", 0)) <= config.repetitions
    ]
    passed = sum(record.get("passed") is True for record in relevant_records)
    errors = sum(record.get("status") == "ERROR" for record in relevant_records)
    return EvaluationReport(
        output_dir=output_dir,
        selected_scenarios=len(selected),
        planned_trials=planned,
        executed_trials=executed,
        skipped_trials=skipped,
        passed_trials=passed,
        failed_trials=len(relevant_records) - passed,
        error_trials=errors,
        summary=summary,
    )
