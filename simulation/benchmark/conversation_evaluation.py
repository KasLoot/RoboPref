"""Resumable batch evaluation for full PrefMem conversations."""

from __future__ import annotations

import json
import os
import tempfile
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from agents.hri import HRIContractError
from dataset.benchmark import BenchmarkEpisodeError

from .conversation_cases import (
    ConversationDatasetError,
    build_conversation_cases,
    case_matrix,
    load_episode_catalog,
)
from .conversation_metrics import build_conversation_summary
from .conversation_models import (
    CONVERSATION_SCHEMA_VERSION,
    ConversationBatchReport,
    BENCHMARK_RESULTS,
    RUN_STATUSES,
    ConversationEvaluationConfig,
    ConversationRunContext,
    ConversationCase,
    stable_digest,
)
from .conversation_runner import (
    ConversationOrchestratorFactory,
    ConversationRunError,
    run_conversation_case,
)
from .conversation_runtime import default_conversation_orchestrator_factory
from .conversation_scoring import score_conversation_run
from .provenance import (
    benchmark_content_sha256,
    callable_provenance,
    runtime_provenance,
)
from .validator import validate_benchmark


class ConversationEvaluationError(RuntimeError):
    pass

def _path(value: Any, *keys: str, default: Any = None) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current



def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as stream:
            json.dump(
                value,
                stream,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                default=str,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            temporary = stream.name
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary and Path(temporary).exists():
            Path(temporary).unlink()


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        json.dump(
            value,
            stream,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _load_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ConversationEvaluationError(
                    f"Malformed results.jsonl line {line_number}: {error}"
                ) from error
            if not isinstance(value, dict):
                raise ConversationEvaluationError(
                    f"results.jsonl line {line_number} is not an object."
                )
            records.append(value)
    return records


def _record_key(record: Mapping[str, Any]) -> tuple[str, int]:
    case = record.get("case")
    case_id = str(case.get("case_id", "")) if isinstance(case, Mapping) else ""
    repetition = record.get("repetition")
    if not case_id or type(repetition) is not int or repetition <= 0:
        raise ConversationEvaluationError(
            "Every result must contain case.case_id and a positive repetition."
        )
    return case_id, repetition


def _resume_index(
    records: list[dict[str, Any]],
    *,
    config_digest: str,
    cases_by_id: Mapping[str, ConversationCase],
    config: ConversationEvaluationConfig,
) -> dict[tuple[str, int], dict[str, Any]]:
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for record in records:
        if record.get("config_digest") != config_digest:
            raise ConversationEvaluationError(
                "Existing results were produced by a different semantic configuration. "
                "Choose a new output directory."
            )
        key = _record_key(record)
        if record.get("schema_version") != "robopref.conversation-run.v1":
            raise ConversationEvaluationError(
                f"Resume record {key} has an unsupported run schema."
            )
        case_id, repetition = key
        expected_case = cases_by_id.get(case_id)
        if expected_case is None or repetition > config.repetitions:
            raise ConversationEvaluationError(
                f"Resume record {key} is outside the current case plan."
            )
        embedded_case = record.get("case")
        if (
            not isinstance(embedded_case, Mapping)
            or stable_digest(dict(embedded_case)) != expected_case.digest
            or record.get("case_digest") != expected_case.digest
        ):
            raise ConversationEvaluationError(
                f"Resume record {key} does not match its immutable case plan."
            )
        expected_model_seed = (
            config.model_seed + repetition - 1
            if config.model_seed is not None
            else None
        )
        if record.get("model_seed") != expected_model_seed:
            raise ConversationEvaluationError(
                f"Resume record {key} has the wrong model seed."
            )
        run_status = str(record.get("run_status", ""))
        benchmark_result = str(record.get("benchmark_result", ""))
        if run_status not in RUN_STATUSES:
            raise ConversationEvaluationError(
                f"Resume record {key} has an invalid run status."
            )
        if benchmark_result not in BENCHMARK_RESULTS:
            raise ConversationEvaluationError(
                f"Resume record {key} has an invalid functional verdict."
            )
        valid_pair = (
            run_status in {"COMPLETED", "ARTIFACT_ERROR"}
            and benchmark_result in {"PASS", "FAIL"}
        ) or (
            run_status not in {"COMPLETED", "ARTIFACT_ERROR"}
            and benchmark_result == "NOT_SCORED"
        )
        if not valid_pair:
            raise ConversationEvaluationError(
                f"Resume record {key} has an inconsistent status/verdict pair."
            )
        if run_status in {"COMPLETED", "ARTIFACT_ERROR"}:
            commands = record.get("commands")
            if not isinstance(commands, list):
                raise ConversationEvaluationError(
                    f"Resume record {key} has no command ledger."
                )
            expected_ids = [item.command_id for item in expected_case.commands]
            actual_ids = [
                str(_path(item, "command", "command_id", default=""))
                for item in commands
                if isinstance(item, Mapping)
            ]
            if actual_ids != expected_ids:
                raise ConversationEvaluationError(
                    f"Resume record {key} has an incomplete command ledger."
                )
            if not isinstance(record.get("checks"), list):
                raise ConversationEvaluationError(
                    f"Resume record {key} has no check ledger."
                )
            artifact_complete = _path(
                record, "artifact_status", "complete", default=None
            )
            if run_status == "COMPLETED" and artifact_complete is not True:
                raise ConversationEvaluationError(
                    f"Resume record {key} claims completion without complete artifacts."
                )
            if run_status == "ARTIFACT_ERROR" and artifact_complete is True:
                raise ConversationEvaluationError(
                    f"Resume record {key} claims an artifact error with complete artifacts."
                )
        if key in result:
            raise ConversationEvaluationError(
                f"Duplicate result for {key[0]} repetition {key[1]}."
            )
        result[key] = record
    return result


def _exception_chain(error: BaseException) -> list[BaseException]:
    chain: list[BaseException] = []
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        next_error = current.__cause__
        if next_error is None and not current.__suppress_context__:
            next_error = current.__context__
        current = next_error
    return chain


def _classify_exception(error: BaseException) -> str:
    chain = _exception_chain(error)
    signatures = [
        f"{type(node).__name__} {node}".casefold()
        for node in chain
    ]
    timeout_names = {
        "apitimeouterror",
        "connecttimeout",
        "readtimeout",
        "timeout",
        "timeouterror",
    }
    if any(
        isinstance(node, TimeoutError)
        or type(node).__name__.casefold() in timeout_names
        or "timed out" in signature
        or "timeout" in signature
        for node, signature in zip(chain, signatures)
    ):
        return "TIMEOUT"
    if isinstance(
        error,
        (
            ConversationRunError,
            ConversationDatasetError,
            BenchmarkEpisodeError,
            KeyError,
        ),
    ):
        return "BENCHMARK_ERROR"
    infrastructure_markers = (
        "api connection",
        "connecterror",
        "connection error",
        "connection refused",
        "remotedisconnected",
        "server disconnected",
        "service unavailable",
    )
    if any(
        isinstance(node, OSError)
        or any(marker in signature for marker in infrastructure_markers)
        for node, signature in zip(chain, signatures)
    ):
        return "INFRASTRUCTURE_ERROR"
    return "AGENT_TERMINATED"


def _error_record(
    context: ConversationRunContext,
    error: BaseException,
    *,
    config_digest: str,
    started: float,
) -> dict[str, Any]:
    return {
        "schema_version": "robopref.conversation-run.v1",
        "case": context.case.to_dict(),
        "case_digest": context.case.digest,
        "config_digest": config_digest,
        "repetition": context.repetition,
        "model_seed": context.model_seed,
        "run_directory": str(context.run_directory),
        "run_status": _classify_exception(error),
        "benchmark_result": "NOT_SCORED",
        "primary_failure": {
            "name": "run_exception",
            "stage": "runtime",
            "status": "MISSING",
            "criticality": "HARD",
            "root_cause_code": type(error).__name__.upper(),
            "details": {
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        },
        "checks": [],
        "commands": [],
        "artifact_status": {
            "complete": False,
            "reason": "run terminated before artifact audit",
        },
        "duration_seconds": time.monotonic() - started,
    }


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]
def _next_run_directory(
    output: Path,
    case_id: str,
    repetition: int,
) -> Path:
    base = output / "runs" / case_id / f"rep-{repetition:03d}"
    if not base.exists():
        return base
    retry = 1
    while True:
        candidate = base.with_name(f"{base.name}-retry-{retry:03d}")
        if not candidate.exists():
            return candidate
        retry += 1



def evaluate_conversations(
    config: ConversationEvaluationConfig,
    *,
    orchestrator_factory: ConversationOrchestratorFactory = (
        default_conversation_orchestrator_factory
    ),
) -> ConversationBatchReport:
    """Validate, plan, run, score, and summarize an evaluation batch."""

    validation = validate_benchmark(config.benchmark_root)
    if not validation.valid:
        raise ConversationEvaluationError(
            "Benchmark validation failed before model calls: "
            + "; ".join(validation.errors)
        )
    catalog = load_episode_catalog(config.benchmark_root)
    episodes = {item.scenario_id: item for item in catalog}
    cases = build_conversation_cases(catalog, config)
    planned_runs = len(cases) * config.repetitions
    selected_scenario_ids = sorted(
        {
            identifier
            for case in cases
            for command in case.commands
            for identifier in (
                command.episode_id,
                command.recovery_episode_id,
            )
            if identifier is not None
        }
    )
    repository_root = _repository_root()
    provenance = runtime_provenance(repository_root)
    semantic_config = {
        **config.semantic_dict(),
        "case_digests": {case.case_id: case.digest for case in cases},
        "benchmark_content_sha256": benchmark_content_sha256(
            config.benchmark_root,
            selected_scenario_ids,
        ),
        "source_tree_sha256": provenance["source_tree_sha256"],
        "orchestrator_factory": callable_provenance(orchestrator_factory),
    }
    config_digest = stable_digest(semantic_config)
    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    results_path = output / "results.jsonl"
    summary_path = output / "summary.json"
    cases_path = output / "cases.json"
    experiment_path = output / "experiment.json"

    existing = _load_results(results_path)
    if existing and not config.resume:
        raise ConversationEvaluationError(
            "The output directory already has results and resume is disabled. "
            "Choose a new output directory."
        )
    resume_index = _resume_index(
        existing,
        config_digest=config_digest,
        cases_by_id={case.case_id: case for case in cases},
        config=config,
    )
    planned_keys = {
        (case.case_id, repetition)
        for case in cases
        for repetition in range(1, config.repetitions + 1)
    }
    extra = set(resume_index) - planned_keys
    if extra:
        raise ConversationEvaluationError(
            "Existing results contain runs outside the current plan."
        )

    _atomic_json(
        cases_path,
        {
            "schema_version": "robopref.conversation-case-plan.v1",
            "config_digest": config_digest,
            "matrix": case_matrix(cases),
            "cases": [case.to_dict() for case in cases],
        },
    )
    _atomic_json(
        experiment_path,
        {
            "schema_version": CONVERSATION_SCHEMA_VERSION,
            "config_digest": config_digest,
            "configuration": semantic_config,
            "batch_controls": {
                "output_dir": str(config.output_dir),
                "resume": config.resume,
                "fail_fast": config.fail_fast,
                "display_all": config.display_all,
            },
            "reporting_configuration": {
                "bootstrap_replicates": config.bootstrap_replicates,
                "bootstrap_seed": config.bootstrap_seed,
            },
            "provenance": provenance,
            "dataset_validation": validation.to_dict(),
            "case_matrix": case_matrix(cases),
            "planned_runs": planned_runs,
            "result_schema": "robopref.conversation-run.v1",
        },
    )

    records = list(existing)
    executed = 0
    resumed = len(existing)
    for case in cases:
        for repetition in range(1, config.repetitions + 1):
            key = (case.case_id, repetition)
            if key in resume_index:
                continue
            run_directory = _next_run_directory(
                output,
                case.case_id,
                repetition,
            )
            model_seed = (
                config.model_seed + repetition - 1
                if config.model_seed is not None
                else None
            )
            context = ConversationRunContext(
                case=case,
                repetition=repetition,
                run_directory=run_directory,
                config=config,
                model_seed=model_seed,
                episodes=episodes,
            )
            started = time.monotonic()
            try:
                raw = run_conversation_case(
                    context,
                    orchestrator_factory=orchestrator_factory,
                )
                record = score_conversation_run(
                    raw,
                    episodes,
                    near_miss_policy=config.near_miss_policy,
                )
                artifact_complete = _path(
                    record,
                    "artifact_status",
                    "complete",
                    default=None,
                )
                record["run_status"] = (
                    "ARTIFACT_ERROR"
                    if artifact_complete is not True
                    else "COMPLETED"
                )
                record["config_digest"] = config_digest
            except Exception as error:
                record = _error_record(
                    context,
                    error,
                    config_digest=config_digest,
                    started=started,
                )
            _append_jsonl(results_path, record)
            records.append(record)
            executed += 1
            if config.fail_fast and (
                record.get("run_status") != "COMPLETED"
                or record.get("benchmark_result") != "PASS"
            ):
                summary = build_conversation_summary(
                    records,
                    planned_runs=planned_runs,
                    planned_cases=cases,
                    repetitions=config.repetitions,
                    near_miss_policy=config.near_miss_policy,
                    bootstrap_replicates=config.bootstrap_replicates,
                    bootstrap_seed=config.bootstrap_seed,
                )
                summary["config_digest"] = config_digest
                summary["batch_complete"] = False
                _atomic_json(summary_path, summary)
                raise ConversationEvaluationError(
                    f"Fail-fast stopped after {case.case_id} repetition {repetition}."
                )

    summary = build_conversation_summary(
        records,
        planned_runs=planned_runs,
        planned_cases=cases,
        repetitions=config.repetitions,
        near_miss_policy=config.near_miss_policy,
        bootstrap_replicates=config.bootstrap_replicates,
        bootstrap_seed=config.bootstrap_seed,
    )
    summary["config_digest"] = config_digest
    summary["batch_complete"] = len(records) == planned_runs
    _atomic_json(summary_path, summary)

    statuses = Counter(str(item.get("run_status")) for item in records)
    passed = sum(
        item.get("run_status") == "COMPLETED"
        and item.get("benchmark_result") == "PASS"
        for item in records
    )
    failed = sum(
        item.get("run_status") == "COMPLETED"
        and item.get("benchmark_result") == "FAIL"
        for item in records
    )
    return ConversationBatchReport(
        output_dir=output,
        results_path=results_path,
        summary_path=summary_path,
        selected_cases=len(cases),
        planned_runs=planned_runs,
        executed_runs=executed,
        resumed_runs=resumed,
        passed_runs=passed,
        failed_runs=failed,
        agent_terminated_runs=statuses["AGENT_TERMINATED"],
        infrastructure_error_runs=statuses["INFRASTRUCTURE_ERROR"],
        benchmark_error_runs=statuses["BENCHMARK_ERROR"],
        timeout_runs=statuses["TIMEOUT"],
        artifact_error_runs=statuses["ARTIFACT_ERROR"],
    )


__all__ = [
    "ConversationEvaluationError",
    "evaluate_conversations",
]
