"""Aggregate explicit functional, audited, and robustness metrics."""

from __future__ import annotations

import math
import random
import statistics
from collections import Counter, defaultdict
from typing import Any, Callable, Iterable, Mapping, Sequence

from .conversation_models import BENCHMARK_RESULTS, RUN_STATUSES


_OBSERVED_CHECK_STATUSES = {"PASS", "FAIL"}
_ELIGIBLE_CHECK_STATUSES = {"PASS", "FAIL", "MISSING"}
_NEGATIVE_OUTCOMES = {"wrong_complete", "partial", "near_miss", "unknown"}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    return (
        value
        if isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray))
        else ()
    )


def _path(value: Any, *keys: str, default: Any = None) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _case_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    converted = to_dict() if callable(to_dict) else None
    return dict(converted) if isinstance(converted, Mapping) else {}


def _record_key(record: Mapping[str, Any]) -> tuple[str, int]:
    case_id = str(_path(record, "case", "case_id", default=""))
    repetition = record.get("repetition", 1)
    return case_id, repetition if type(repetition) is int else -1


def _functional_pass(record: Mapping[str, Any]) -> bool:
    return record.get("benchmark_result") == "PASS"


def _audited_pass(record: Mapping[str, Any]) -> bool:
    return (
        record.get("run_status") == "COMPLETED"
        and record.get("benchmark_result") == "PASS"
        and _path(record, "artifact_status", "complete") is True
    )


def wilson_interval(
    successes: int,
    total: int,
    *,
    z: float = 1.959963984540054,
) -> dict[str, float | int | str | None]:
    """Naive Wilson interval; clustering is handled by the bootstrap output."""

    base: dict[str, float | int | str | None] = {
        "method": "wilson_score_naive_independent",
        "successes": successes,
        "total": total,
        "rate": None,
        "lower": None,
        "upper": None,
    }
    if total <= 0:
        return base
    estimate = successes / total
    denominator = 1.0 + z * z / total
    centre = estimate + z * z / (2.0 * total)
    margin = z * math.sqrt(
        estimate * (1.0 - estimate) / total
        + z * z / (4.0 * total * total)
    )
    return {
        **base,
        "rate": estimate,
        "lower": max(0.0, (centre - margin) / denominator),
        "upper": min(1.0, (centre + margin) / denominator),
    }


def _percentile(values: Sequence[float], fraction: float) -> float:
    position = fraction * (len(values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def _planned_cases(
    records: Sequence[Mapping[str, Any]],
    supplied: Sequence[Any] | None,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    source = supplied or [record.get("case") for record in records]
    for raw_case in source:
        case = _case_mapping(raw_case)
        case_id = str(case.get("case_id", ""))
        if case_id and case_id not in result:
            result[case_id] = case
    return result


def _planned_keys(
    cases: Mapping[str, Mapping[str, Any]],
    repetitions: int,
) -> set[tuple[str, int]]:
    return {
        (case_id, repetition)
        for case_id in cases
        for repetition in range(1, repetitions + 1)
    }


def _integrity(
    records: Sequence[Mapping[str, Any]],
    *,
    planned: int,
    exact_planned_keys: set[tuple[str, int]] | None,
) -> dict[str, Any]:
    keys = [_record_key(record) for record in records]
    key_counts = Counter(keys)
    duplicate = [
        [case_id, repetition]
        for (case_id, repetition), count in sorted(key_counts.items())
        if count > 1
    ]
    recorded_keys = set(keys)
    missing = (
        [list(key) for key in sorted(exact_planned_keys - recorded_keys)]
        if exact_planned_keys is not None
        else []
    )
    unexpected = (
        [list(key) for key in sorted(recorded_keys - exact_planned_keys)]
        if exact_planned_keys is not None
        else []
    )
    missing_count = (
        len(missing)
        if exact_planned_keys is not None
        else max(0, planned - len(records))
    )
    invalid_statuses = Counter(
        str(record.get("run_status"))
        for record in records
        if record.get("run_status") not in RUN_STATUSES
    )
    invalid_results = Counter(
        str(record.get("benchmark_result"))
        for record in records
        if record.get("benchmark_result") not in BENCHMARK_RESULTS
    )
    invalid_pairs: list[dict[str, Any]] = []
    artifact_conflicts: list[dict[str, Any]] = []
    for record in records:
        status = record.get("run_status")
        result = record.get("benchmark_result")
        valid_pair = (
            status in {"COMPLETED", "ARTIFACT_ERROR"}
            and result in {"PASS", "FAIL"}
        ) or (
            status in RUN_STATUSES
            and status not in {"COMPLETED", "ARTIFACT_ERROR"}
            and result == "NOT_SCORED"
        )
        if not valid_pair:
            invalid_pairs.append(
                {
                    "run_key": list(_record_key(record)),
                    "run_status": status,
                    "benchmark_result": result,
                }
            )
        artifact_complete = _path(record, "artifact_status", "complete")
        if status == "COMPLETED" and artifact_complete is not True:
            artifact_conflicts.append(
                {
                    "run_key": list(_record_key(record)),
                    "run_status": status,
                    "artifact_complete": artifact_complete,
                }
            )
        if status == "ARTIFACT_ERROR" and artifact_complete is True:
            artifact_conflicts.append(
                {
                    "run_key": list(_record_key(record)),
                    "run_status": status,
                    "artifact_complete": artifact_complete,
                }
            )
    count_mismatch = len(records) != planned
    valid = not any(
        (
            duplicate,
            missing_count,
            unexpected,
            invalid_statuses,
            invalid_results,
            invalid_pairs,
            artifact_conflicts,
            count_mismatch,
        )
    )
    return {
        "valid": valid,
        "planned_runs": planned,
        "recorded_runs": len(records),
        "record_count_matches_plan": not count_mismatch,
        "missing_run_count": missing_count,
        "missing_run_slots": missing,
        "duplicate_run_slots": duplicate,
        "unexpected_run_slots": unexpected,
        "invalid_run_statuses": dict(sorted(invalid_statuses.items())),
        "invalid_benchmark_results": dict(sorted(invalid_results.items())),
        "invalid_status_result_pairs": invalid_pairs,
        "artifact_status_conflicts": artifact_conflicts,
    }


def _aggregate_check_bucket(
    checks: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    statuses = Counter(str(check.get("status")) for check in checks)
    eligible = sum(statuses[status] for status in _ELIGIBLE_CHECK_STATUSES)
    observed = sum(statuses[status] for status in _OBSERVED_CHECK_STATUSES)
    known = {"PASS", "FAIL", "MISSING", "NOT_APPLICABLE"}
    invalid = {
        status: count for status, count in statuses.items() if status not in known
    }
    return {
        "eligible": eligible,
        "observed": observed,
        "pass": statuses["PASS"],
        "fail": statuses["FAIL"],
        "missing": statuses["MISSING"],
        "not_applicable": statuses["NOT_APPLICABLE"],
        "invalid_status_counts": dict(sorted(invalid.items())),
        "conditional_pass_rate": _ratio(statuses["PASS"], observed),
        "unconditional_pass_rate": _ratio(statuses["PASS"], eligible),
        "conditional_wilson_95_naive": wilson_interval(
            statuses["PASS"], observed
        ),
        "unconditional_wilson_95_naive": wilson_interval(
            statuses["PASS"], eligible
        ),
    }


def _check_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    checks = [
        _mapping(raw_check)
        for record in records
        for raw_check in _sequence(record.get("checks"))
    ]

    def grouped(key: Callable[[Mapping[str, Any]], str]) -> dict[str, Any]:
        values: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for check in checks:
            values[key(check)].append(check)
        return {
            name: _aggregate_check_bucket(bucket)
            for name, bucket in sorted(values.items())
        }

    return {
        "by_name": grouped(lambda check: str(check.get("name", "<missing>"))),
        "by_stage": grouped(lambda check: str(check.get("stage", "<missing>"))),
        "by_criticality": grouped(
            lambda check: str(check.get("criticality", "<missing>"))
        ),
        "by_stage_name_criticality": grouped(
            lambda check: "|".join(
                (
                    str(check.get("stage", "<missing>")),
                    str(check.get("name", "<missing>")),
                    str(check.get("criticality", "<missing>")),
                )
            )
        ),
        "note": (
            "These aggregates cover emitted checks. Validator and headline "
            "metrics below synthesize missing evidence from the case plan."
        ),
    }


def _expected_validator_outcome(
    command: Mapping[str, Any],
    near_miss_policy: str,
) -> str | None:
    if command.get("score_endpoint") is not True:
        return None
    outcome = str(command.get("packet_outcome", ""))
    if outcome == "success":
        return "SUCCESS"
    if outcome == "wrong_complete":
        return "FAILURE"
    if outcome == "partial":
        return "PARTIAL"
    if outcome == "unknown":
        return "UNKNOWN"
    if outcome == "near_miss":
        return "UNKNOWN" if near_miss_policy == "perceptual" else "FAILURE"
    return None


def _validator_metrics(
    records: Sequence[Mapping[str, Any]],
    cases: Mapping[str, Mapping[str, Any]],
    *,
    repetitions: int,
    near_miss_policy: str,
) -> dict[str, Any]:
    planned: list[tuple[tuple[str, int], str, str]] = []
    for case_id, case in sorted(cases.items()):
        for repetition in range(1, repetitions + 1):
            for raw_command in _sequence(case.get("commands")):
                command = _mapping(raw_command)
                expected = _expected_validator_outcome(
                    command, near_miss_policy
                )
                if expected is not None:
                    planned.append(
                        (
                            (case_id, repetition),
                            str(command.get("command_id", "")),
                            expected,
                        )
                    )

    records_by_key: dict[tuple[str, int], Mapping[str, Any]] = {}
    for record in records:
        records_by_key.setdefault(_record_key(record), record)
    if not planned:
        for record in records:
            for raw_check in _sequence(record.get("checks")):
                check = _mapping(raw_check)
                if (
                    check.get("name") == "validator_outcome"
                    and check.get("status") != "NOT_APPLICABLE"
                ):
                    planned.append(
                        (
                            _record_key(record),
                            str(check.get("command_id", "")),
                            str(check.get("expected", "MISSING")),
                        )
                    )

    matrix: dict[str, Counter[str]] = defaultdict(Counter)
    emitted = 0
    observed = 0
    correct = 0
    for run_key, command_id, expected in planned:
        record = records_by_key.get(run_key, {})
        matching = [
            _mapping(raw_check)
            for raw_check in _sequence(record.get("checks"))
            if _mapping(raw_check).get("name") == "validator_outcome"
            and str(_mapping(raw_check).get("command_id", "")) == command_id
            and _mapping(raw_check).get("status") != "NOT_APPLICABLE"
        ]
        check = matching[0] if matching else {}
        if check:
            emitted += 1
        actual_value = check.get("actual")
        actual = (
            str(actual_value)
            if check.get("status") in _OBSERVED_CHECK_STATUSES
            and actual_value is not None
            else "MISSING"
        )
        if actual != "MISSING":
            observed += 1
        if actual == expected:
            correct += 1
        matrix[expected][actual] += 1

    expected_labels = sorted(matrix)
    actual_labels = sorted(
        {actual for row in matrix.values() for actual in row}
    )
    labels = sorted(set(expected_labels) | set(actual_labels))
    per_class: dict[str, Any] = {}
    f1_values: list[float] = []
    for label in expected_labels:
        true_positive = matrix[label][label]
        false_positive = sum(
            matrix[other][label]
            for other in expected_labels
            if other != label
        )
        false_negative = sum(
            count
            for actual, count in matrix[label].items()
            if actual != label
        )
        precision = _ratio(true_positive, true_positive + false_positive)
        recall = _ratio(true_positive, true_positive + false_negative)
        precision_value = precision if precision is not None else 0.0
        recall_value = recall if recall is not None else 0.0
        f1 = (
            2.0 * precision_value * recall_value
            / (precision_value + recall_value)
            if precision_value + recall_value
            else 0.0
        )
        f1_values.append(f1)
        per_class[label] = {
            "support": sum(matrix[label].values()),
            "precision": precision_value,
            "recall": recall_value,
            "f1": f1,
        }
    eligible = len(planned)
    return {
        "planned_eligible": eligible,
        "emitted": emitted,
        "observed": observed,
        "missing": eligible - observed,
        "coverage": _ratio(observed, eligible),
        "accuracy_observed": _ratio(correct, observed),
        "accuracy_intention_to_treat": _ratio(correct, eligible),
        "expected_labels": expected_labels,
        "actual_labels": actual_labels,
        "confusion_matrix": {
            expected: {
                actual: matrix[expected][actual] for actual in labels
            }
            for expected in expected_labels
        },
        "per_class": per_class,
        "macro_f1": statistics.fmean(f1_values) if f1_values else None,
    }


def _dependency_components(
    cases: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, str], list[str]]:
    parent = {case_id: case_id for case_id in cases}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    owners: dict[str, str] = {}
    missing: list[str] = []
    for case_id, case in sorted(cases.items()):
        metadata = _mapping(case.get("metadata"))
        raw_dependencies = _sequence(metadata.get("dependency_cluster_ids"))
        dependencies = {
            str(value).strip() for value in raw_dependencies if str(value).strip()
        }
        if not dependencies:
            cluster_id = str(case.get("cluster_id", "")).strip()
            if cluster_id:
                dependencies = {cluster_id}
        if not dependencies:
            missing.append(case_id)
            continue
        for dependency in sorted(dependencies):
            previous = owners.setdefault(dependency, case_id)
            union(case_id, previous)
    return {case_id: find(case_id) for case_id in cases}, missing


def _bootstrap_interval(
    records: Sequence[Mapping[str, Any]],
    cases: Mapping[str, Mapping[str, Any]],
    *,
    integrity: Mapping[str, Any],
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    component_for_case, missing_clusters = _dependency_components(cases)
    base = {
        "cluster_definition": "connected_dependency_component",
        "replicates": replicates,
        "seed": seed,
        "cluster_count": len(set(component_for_case.values())),
        "missing_cluster_case_ids": missing_clusters,
    }
    if integrity.get("valid") is not True:
        return {
            **base,
            "status": "NOT_COMPUTED",
            "reason": "batch integrity is incomplete or invalid",
            "trial_micro": None,
            "equal_dependency_component_macro": None,
        }
    if missing_clusters:
        return {
            **base,
            "status": "NOT_COMPUTED",
            "reason": "one or more cases have no physical-scene dependency ID",
            "trial_micro": None,
            "equal_dependency_component_macro": None,
        }
    grouped: dict[str, list[bool]] = defaultdict(list)
    for record in records:
        case_id = str(_path(record, "case", "case_id", default=""))
        component = component_for_case.get(case_id)
        if component is None:
            return {
                **base,
                "status": "NOT_COMPUTED",
                "reason": f"recorded case {case_id} has no planned dependency component",
                "trial_micro": None,
                "equal_dependency_component_macro": None,
            }
        grouped[component].append(_audited_pass(record))
    component_ids = sorted(grouped)
    if not component_ids:
        return {
            **base,
            "status": "NOT_COMPUTED",
            "reason": "no dependency components are available",
            "trial_micro": None,
            "equal_dependency_component_macro": None,
        }
    trial_estimate = sum(map(sum, grouped.values())) / sum(
        map(len, grouped.values())
    )
    component_rates = {
        key: sum(values) / len(values) for key, values in grouped.items()
    }
    macro_estimate = statistics.fmean(component_rates.values())
    rng = random.Random(seed)
    micro_samples: list[float] = []
    macro_samples: list[float] = []
    for _ in range(replicates):
        selected = [rng.choice(component_ids) for _ in component_ids]
        sample_values = [
            value for component in selected for value in grouped[component]
        ]
        micro_samples.append(sum(sample_values) / len(sample_values))
        macro_samples.append(
            statistics.fmean(component_rates[component] for component in selected)
        )
    micro_samples.sort()
    macro_samples.sort()
    return {
        **base,
        "status": "COMPUTED",
        "reason": None,
        "trial_micro": {
            "estimand": "planned-run micro audited pass rate",
            "estimate": trial_estimate,
            "lower": _percentile(micro_samples, 0.025),
            "upper": _percentile(micro_samples, 0.975),
        },
        "equal_dependency_component_macro": {
            "estimand": "equal dependency-component macro audited pass rate",
            "estimate": macro_estimate,
            "lower": _percentile(macro_samples, 0.025),
            "upper": _percentile(macro_samples, 0.975),
        },
    }


def clustered_bootstrap_interval(
    records: Sequence[Mapping[str, Any]],
    *,
    predicate: Callable[[Mapping[str, Any]], bool],
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    """Compatibility helper with strict, non-fallback cluster handling."""

    clusters: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    missing = 0
    for record in records:
        cluster_id = str(_path(record, "case", "cluster_id", default=""))
        if not cluster_id:
            missing += 1
            continue
        clusters[cluster_id].append(record)
    if missing or not clusters:
        return {
            "status": "NOT_COMPUTED",
            "reason": "missing cluster IDs" if missing else "no records",
            "clusters": len(clusters),
            "replicates": replicates,
            "seed": seed,
            "estimate": None,
            "lower": None,
            "upper": None,
        }
    cluster_ids = sorted(clusters)
    estimate = sum(predicate(item) for item in records) / len(records)
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(replicates):
        selected = [rng.choice(cluster_ids) for _ in cluster_ids]
        sample = [item for key in selected for item in clusters[key]]
        samples.append(sum(predicate(item) for item in sample) / len(sample))
    samples.sort()
    return {
        "status": "COMPUTED",
        "reason": None,
        "clusters": len(cluster_ids),
        "replicates": replicates,
        "seed": seed,
        "estimate": estimate,
        "lower": _percentile(samples, 0.025),
        "upper": _percentile(samples, 0.975),
    }


def _group_summary(
    records: Sequence[Mapping[str, Any]],
    cases: Mapping[str, Mapping[str, Any]],
    *,
    repetitions: int,
    key: Callable[[Mapping[str, Any]], str],
) -> dict[str, Any]:
    planned_counts = Counter(
        key(case) for case in cases.values() for _ in range(repetitions)
    )
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[key(_mapping(record.get("case")))].append(record)
    result: dict[str, Any] = {}
    for name in sorted(set(planned_counts) | set(grouped)):
        values = grouped[name]
        functional_passed = sum(_functional_pass(item) for item in values)
        functional_failed = sum(
            item.get("benchmark_result") == "FAIL" for item in values
        )
        audited_passed = sum(_audited_pass(item) for item in values)
        completed = sum(item.get("run_status") == "COMPLETED" for item in values)
        result[name] = {
            "planned_runs": planned_counts[name],
            "recorded_runs": len(values),
            "completed_runs": completed,
            "functional_passed": functional_passed,
            "functional_failed": functional_failed,
            "functional_not_scored": sum(
                item.get("benchmark_result") == "NOT_SCORED" for item in values
            ),
            "functional_observed_rate_naive": wilson_interval(
                functional_passed, functional_passed + functional_failed
            ),
            "functional_planned_itt_rate_naive": wilson_interval(
                functional_passed, planned_counts[name]
            ),
            "audited_passed": audited_passed,
            "audited_completed_only_rate_naive": wilson_interval(
                audited_passed, completed
            ),
            "audited_planned_itt_rate_naive": wilson_interval(
                audited_passed, planned_counts[name]
            ),
        }
    return result


def _check_statuses(
    record: Mapping[str, Any],
    name: str,
) -> list[str]:
    return [
        str(check.get("status"))
        for check in map(_mapping, _sequence(record.get("checks")))
        if check.get("name") == name
        and check.get("status") != "NOT_APPLICABLE"
    ]


def _headline_summary(votes: Sequence[str]) -> dict[str, Any]:
    counts = Counter(votes)
    eligible = len(votes)
    return {
        "eligible_runs": eligible,
        "passed_runs": counts["PASS"],
        "failed_runs": counts["FAIL"],
        "missing_runs": counts["MISSING"],
        "intention_to_treat_rate_naive": wilson_interval(
            counts["PASS"], eligible
        ),
    }


def _required_vote(
    record: Mapping[str, Any],
    required: Sequence[tuple[str, int]],
) -> str:
    statuses: list[str] = []
    for name, minimum_count in required:
        values = _check_statuses(record, name)
        if len(values) < minimum_count:
            return "MISSING"
        statuses.extend(values)
    if any(status == "FAIL" for status in statuses):
        return "FAIL"
    if any(status == "MISSING" for status in statuses):
        return "MISSING"
    return "PASS" if statuses and all(status == "PASS" for status in statuses) else "MISSING"


def _headlines(
    records: Sequence[Mapping[str, Any]],
    *,
    near_miss_policy: str,
) -> dict[str, Any]:
    negative_votes: list[str] = []
    recovery_decision_votes: list[str] = []
    control_votes: list[str] = []
    recovery_votes: list[str] = []
    isolation_votes: list[str] = []
    safety_votes: list[str] = []
    for record in records:
        case = _mapping(record.get("case"))
        metadata = _mapping(case.get("metadata"))
        commands = [_mapping(item) for item in _sequence(case.get("commands"))]
        endpoint_count = sum(command.get("score_endpoint") is True for command in commands)
        if any(
            str(command.get("packet_outcome")) in _NEGATIVE_OUTCOMES
            for command in commands
            if command.get("score_endpoint") is True
        ):
            negative_votes.append(
                _required_vote(record, (("no_false_completion", 1),))
            )
        if endpoint_count:
            recovery_decision_votes.append(
                _required_vote(
                    record,
                    (("first_attempt_recovery_decision", endpoint_count),),
                )
            )
        if metadata.get("control_kind") == "already_satisfied":
            control_votes.append(
                _required_vote(
                    record,
                    (("planner_subtasks_empty", 1), ("vla_dispatch_count", 1)),
                )
            )
        if case.get("suite") == "recovery":
            outcome = str(metadata.get("outcome", ""))
            reobserve = outcome == "unknown" or (
                outcome == "near_miss" and near_miss_policy == "perceptual"
            )
            requirements: list[tuple[str, int]] = [
                ("scripted_counterfactual_transition", 1),
                ("recovery_terminal_outcome", 1),
                ("recovery_validator_outcome", 1),
                ("recovery_validator_task_complete", 1),
                ("recovery_observation_available", 1),
            ]
            if not reobserve:
                requirements.append(("recovery_planner_vla_subtask_target", 1))
            recovery_votes.append(_required_vote(record, requirements))
        if case.get("profile") == "cross-user-isolation":
            isolation_votes.append(
                _required_vote(record, (("memory_ownership", 1),))
            )
        if case.get("suite") == "safety":
            safety_votes.append(
                _required_vote(
                    record,
                    (
                        ("first_attempt_recovery_decision", 1),
                        ("terminal_outcome", 3),
                        ("no_post_safety_dispatch", 2),
                    ),
                )
            )
    return {
        "no_false_completion_on_negative_packets": _headline_summary(
            negative_votes
        ),
        "expected_first_recovery_decision": _headline_summary(
            recovery_decision_votes
        ),
        "already_satisfied_empty_plan_and_zero_dispatch": _headline_summary(
            control_votes
        ),
        "scripted_counterfactual_recovery_chain": _headline_summary(
            recovery_votes
        ),
        "cross_user_memory_ownership_isolation": _headline_summary(
            isolation_votes
        ),
        "oracle_signaled_abort_latch_clearance": _headline_summary(
            safety_votes
        ),
    }


def _repetition_reliability(
    records: Sequence[Mapping[str, Any]],
    cases: Mapping[str, Mapping[str, Any]],
    *,
    repetitions: int,
) -> dict[str, Any]:
    if repetitions <= 1:
        return {
            "planned_repetitions": repetitions,
            "eligible_cases": 0,
            "complete_case_rate": None,
            "all_audited_pass_rate": None,
            "identical_functional_verdict_rate": None,
            "identical_audited_verdict_rate": None,
            "per_case": {},
        }
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[_record_key(record)].append(record)
    per_case: dict[str, Any] = {}
    complete_count = 0
    all_pass_count = 0
    identical_functional_count = 0
    identical_audited_count = 0
    for case_id in sorted(cases):
        functional: list[str] = []
        audited: list[str] = []
        duplicate_repetitions: list[int] = []
        for repetition in range(1, repetitions + 1):
            values = grouped.get((case_id, repetition), [])
            if len(values) != 1:
                functional.append("MISSING" if not values else "DUPLICATE")
                audited.append("MISSING" if not values else "DUPLICATE")
                if len(values) > 1:
                    duplicate_repetitions.append(repetition)
                continue
            record = values[0]
            functional.append(str(record.get("benchmark_result", "MISSING")))
            audited.append(
                "PASS"
                if _audited_pass(record)
                else "FAIL"
                if record.get("run_status") == "COMPLETED"
                and record.get("benchmark_result") == "FAIL"
                else "UNKNOWN"
            )
        complete = all(
            len(grouped.get((case_id, repetition), [])) == 1
            for repetition in range(1, repetitions + 1)
        )
        all_pass = complete and all(value == "PASS" for value in audited)
        identical_functional = complete and len(set(functional)) == 1
        identical_audited = complete and len(set(audited)) == 1
        complete_count += complete
        all_pass_count += all_pass
        identical_functional_count += identical_functional
        identical_audited_count += identical_audited
        per_case[case_id] = {
            "complete": complete,
            "functional_verdicts": functional,
            "audited_verdicts": audited,
            "duplicate_repetitions": duplicate_repetitions,
            "all_audited_pass": all_pass,
            "identical_functional_verdict": identical_functional,
            "identical_audited_verdict": identical_audited,
        }
    eligible = len(cases)
    return {
        "planned_repetitions": repetitions,
        "eligible_cases": eligible,
        "complete_case_rate": _ratio(complete_count, eligible),
        "all_audited_pass_rate": _ratio(all_pass_count, eligible),
        "identical_functional_verdict_rate": _ratio(
            identical_functional_count, eligible
        ),
        "identical_audited_verdict_rate": _ratio(
            identical_audited_count, eligible
        ),
        "per_case": per_case,
    }


def build_conversation_summary(
    records: Iterable[Mapping[str, Any]],
    *,
    planned_runs: int | None = None,
    planned_cases: Sequence[Any] | None = None,
    repetitions: int = 1,
    near_miss_policy: str = "perceptual",
    bootstrap_replicates: int = 2_000,
    bootstrap_seed: int = 20_260_726,
) -> dict[str, Any]:
    """Build the durable v2 report with explicit estimands and denominators."""

    items = [dict(item) for item in records]
    cases = _planned_cases(items, planned_cases)
    exact_keys = _planned_keys(cases, repetitions) if planned_cases is not None else None
    planned = (
        planned_runs
        if planned_runs is not None
        else len(exact_keys) if exact_keys is not None else len(items)
    )
    integrity = _integrity(
        items,
        planned=planned,
        exact_planned_keys=exact_keys,
    )
    statuses = Counter(str(item.get("run_status")) for item in items)
    status_counts = {status: statuses[status] for status in RUN_STATUSES}
    unexpected_statuses = {
        status: count
        for status, count in statuses.items()
        if status not in RUN_STATUSES
    }
    functional_passed = sum(_functional_pass(item) for item in items)
    functional_failed = sum(
        item.get("benchmark_result") == "FAIL" for item in items
    )
    functional_not_scored = sum(
        item.get("benchmark_result") == "NOT_SCORED" for item in items
    )
    audited_passed = sum(_audited_pass(item) for item in items)
    audited_failed = sum(
        item.get("run_status") == "COMPLETED"
        and item.get("benchmark_result") == "FAIL"
        and _path(item, "artifact_status", "complete") is True
        for item in items
    )
    completed = status_counts["COMPLETED"]
    durations = [
        float(item["duration_seconds"])
        for item in items
        if isinstance(item.get("duration_seconds"), (int, float))
    ]
    roots = Counter(
        str(_path(item, "primary_failure", "root_cause_code", default=""))
        for item in items
        if item.get("primary_failure")
    )

    def metadata_key(name: str) -> Callable[[Mapping[str, Any]], str]:
        return lambda case: str(
            _path(case, "metadata", name, default="<none>")
        )

    results = {
        "functional": {
            "pass": functional_passed,
            "fail": functional_failed,
            "not_scored": functional_not_scored,
            "observed_rate_naive": wilson_interval(
                functional_passed, functional_passed + functional_failed
            ),
            "recorded_intention_to_treat_rate_naive": wilson_interval(
                functional_passed, len(items)
            ),
            "planned_intention_to_treat_rate_naive": wilson_interval(
                functional_passed, planned
            ),
        },
        "audited": {
            "pass": audited_passed,
            "fail": audited_failed,
            "unknown": len(items) - audited_passed - audited_failed,
            "completed_only_rate_naive": wilson_interval(
                audited_passed, completed
            ),
            "recorded_intention_to_treat_rate_naive": wilson_interval(
                audited_passed, len(items)
            ),
            "planned_intention_to_treat_rate_naive": wilson_interval(
                audited_passed, planned
            ),
        },
    }
    return {
        "schema_version": "robopref.conversation-summary.v2",
        "integrity": integrity,
        "run_statuses": {
            "counts": status_counts,
            "unexpected_counts": unexpected_statuses,
        },
        "results": results,
        "checks": _check_metrics(items),
        "validator": _validator_metrics(
            items,
            cases,
            repetitions=repetitions,
            near_miss_policy=near_miss_policy,
        ),
        "headlines": _headlines(
            items,
            near_miss_policy=near_miss_policy,
        ),
        "bootstrap": _bootstrap_interval(
            items,
            cases,
            integrity=integrity,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        ),
        "by_suite": _group_summary(
            items,
            cases,
            repetitions=repetitions,
            key=lambda case: str(case.get("suite", "<none>")),
        ),
        "by_profile": _group_summary(
            items,
            cases,
            repetitions=repetitions,
            key=lambda case: str(case.get("profile", "<none>")),
        ),
        "by_family": _group_summary(
            items,
            cases,
            repetitions=repetitions,
            key=metadata_key("family"),
        ),
        "by_outcome": _group_summary(
            items,
            cases,
            repetitions=repetitions,
            key=metadata_key("outcome"),
        ),
        "primary_root_causes": dict(sorted(roots.items())),
        "repetition_reliability": _repetition_reliability(
            items,
            cases,
            repetitions=repetitions,
        ),
        "duration_seconds": {
            "count": len(durations),
            "total": sum(durations),
            "mean": statistics.fmean(durations) if durations else None,
            "median": statistics.median(durations) if durations else None,
            "maximum": max(durations) if durations else None,
        },
        "interpretation_notes": [
            "Functional PASS means behavior matched the oracle, including correctly rejecting a failure packet.",
            "Audited PASS additionally requires COMPLETED status and complete durable artifacts.",
            "Planned intention-to-treat rates keep terminations and unrecorded planned slots in the denominator.",
            "Wilson intervals are naive descriptive intervals; dependency-aware uncertainty is reported by the cluster bootstrap.",
            "Recovery uses paired static counterfactual endpoints and does not prove a continuous physical trajectory.",
            "The safety suite measures abort/latch/clearance after an oracle-injected UNSAFE signal; it does not measure hazard detection.",
        ],
    }


__all__ = [
    "build_conversation_summary",
    "clustered_bootstrap_interval",
    "wilson_interval",
]
