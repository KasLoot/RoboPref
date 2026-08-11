"""Predeclared, episode-level analysis helpers for RoboPref campaigns.

The functions in this module intentionally return JSON-serializable records.
Missing or invalid attempts are never silently dropped: an estimand with
unresolved assigned episodes is reported as not estimable, together with its
best/worst-case bounds.  SciPy is used when available for the exact binomial
test; the stdlib fallback is mathematically equivalent.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
import csv
import hashlib
import json
import math
from pathlib import Path
import random
from statistics import NormalDist, fmean
from typing import Any


VALID_PASS = "VALID_PASS"
VALID_SYSTEM_FAILURE = "VALID_SYSTEM_FAILURE"
INFRA_INTERRUPTED = "INFRA_INTERRUPTED"
INVALID_HARNESS = "INVALID_HARNESS"
ABORTED_SAFETY = "ABORTED_SAFETY"
NOT_RUN = "NOT_RUN"
KNOWN_STATUSES = frozenset(
    {
        VALID_PASS,
        VALID_SYSTEM_FAILURE,
        INFRA_INTERRUPTED,
        INVALID_HARNESS,
        ABORTED_SAFETY,
        NOT_RUN,
    }
)
RESOLVED_EXPERIMENTAL_STATUSES = frozenset(
    {VALID_PASS, VALID_SYSTEM_FAILURE, ABORTED_SAFETY}
)

FAILURE_LAYERS = (
    "perception/SAM",
    "embedding/retrieval",
    "HRI",
    "planning",
    "execution",
    "monitoring",
    "validation",
    "coordination/state management",
    "simulator/robot",
    "model service/infrastructure",
    "recorder/oracle/harness",
    "unassigned",
)


class AnalysisError(ValueError):
    """Raised when an analysis input violates the predeclared unit of analysis."""


def not_estimable(reason: str, **details: Any) -> dict[str, Any]:
    """Return a uniform explicit marker instead of NaN or a fabricated value."""

    if not reason.strip():
        raise AnalysisError("not-estimable reason must be non-empty")
    return {"estimable": False, "estimate": None, "reason": reason, **details}


def wilson_interval(
    successes: int, total: int, *, confidence: float = 0.95
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    The empty-sample interval is the uninformative ``[0, 1]``; callers should
    additionally mark the associated estimand as not estimable.
    """

    if (
        isinstance(successes, bool)
        or isinstance(total, bool)
        or not isinstance(successes, int)
        or not isinstance(total, int)
        or total < 0
        or successes < 0
        or successes > total
    ):
        raise AnalysisError("successes and total must satisfy 0 <= successes <= total")
    if not 0 < confidence < 1:
        raise AnalysisError("confidence must lie strictly between zero and one")
    if total == 0:
        return (0.0, 1.0)
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total + z * z / (4 * total * total)
        )
        / denominator
    )
    return (max(0.0, centre - radius), min(1.0, centre + radius))


def _status(row: Mapping[str, Any]) -> str:
    value = row.get("run_status", row.get("status", NOT_RUN))
    status = str(value or NOT_RUN)
    if status not in KNOWN_STATUSES:
        raise AnalysisError(f"unknown run status: {status}")
    return status


def _boolean(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, float) and value in {0.0, 1.0}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "pass", "passed", "yes", "1"}:
            return True
        if normalized in {"false", "fail", "failed", "no", "0"}:
            return False
    raise AnalysisError(f"binary outcome is not Boolean: {value!r}")


_OVERALL_OUTCOMES = frozenset(
    {"contract_success", "overall_success", "success", "oracle_pass"}
)


def outcome_value(row: Mapping[str, Any], outcome: str) -> bool | None:
    """Read a binary episode outcome without converting missing data to failure."""

    status = _status(row)
    artifact_audit = row.get("artifact_audit_passed")
    if artifact_audit is False or (
        isinstance(artifact_audit, str)
        and artifact_audit.strip().casefold() == "false"
    ):
        return None
    if row.get("analysis_input_error") or row.get("result_identity_conflicts"):
        return None
    if status not in RESOLVED_EXPERIMENTAL_STATUSES:
        return None
    if outcome in _OVERALL_OUTCOMES:
        expected = status == VALID_PASS
        expected_verdict = "PASS" if expected else "FAIL"
        verdict = row.get("oracle_verdict")
        if status in {VALID_PASS, VALID_SYSTEM_FAILURE} and verdict not in {
            None,
            "",
            expected_verdict,
        }:
            return None
        explicit_values: list[Any] = []
        if outcome in row and row[outcome] not in (None, ""):
            explicit_values.append(row[outcome])
        outcomes = row.get("outcomes")
        if isinstance(outcomes, Mapping) and outcome in outcomes:
            explicit_values.append(outcomes[outcome])
        try:
            if any(_boolean(value) != expected for value in explicit_values):
                return None
        except AnalysisError:
            return None
        return expected
    if outcome in row and row[outcome] not in (None, ""):
        return _boolean(row[outcome])
    outcomes = row.get("outcomes")
    if isinstance(outcomes, Mapping) and outcome in outcomes:
        return _boolean(outcomes[outcome])
    # A safety abort is an observed failure for episode-level safety/contract
    # outcomes, but it cannot supply unrelated metrics such as preference recall.
    if status == ABORTED_SAFETY and outcome in {
        "safe_completion",
        "physical_goal_completion",
        "contract_success",
    }:
        return False
    return None


def select_canonical_attempts(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Select the latest preserved attempt for each randomized schedule entry."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    anonymous: list[dict[str, Any]] = []
    for index, source in enumerate(rows):
        row = dict(source)
        schedule_id = row.get("schedule_id")
        if schedule_id in (None, ""):
            row.setdefault("_analysis_row", index)
            anonymous.append(row)
        else:
            grouped[str(schedule_id)].append(row)
    selected = anonymous
    for schedule_id, attempts in grouped.items():
        numbers: list[int] = []
        for row in attempts:
            raw = row.get("attempt_number", 1)
            try:
                number = int(raw)
            except (TypeError, ValueError) as error:
                raise AnalysisError(
                    f"invalid attempt number for {schedule_id}: {raw!r}"
                ) from error
            numbers.append(number)
        if len(numbers) != len(set(numbers)):
            raise AnalysisError(f"duplicate attempt number for {schedule_id}")
        selected.append(attempts[numbers.index(max(numbers))])
    return tuple(
        sorted(
            selected,
            key=lambda row: (
                int(row.get("schedule_position", 10**18)),
                str(row.get("schedule_id", row.get("_analysis_row", ""))),
            ),
        )
    )


def intention_to_treat(
    rows: Iterable[Mapping[str, Any]],
    *,
    outcome: str = "contract_success",
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Episode-level ITT summary over every randomized schedule entry.

    ``INVALID_HARNESS``, unresolved infrastructure interruptions, and not-run
    assignments remain in the assigned denominator.  If any assigned outcome
    is unknown, the point estimand is explicitly not estimable and conservative
    lower/upper success bounds are returned.
    """

    canonical = select_canonical_attempts(rows)
    assigned = len(canonical)
    successes = 0
    failures = 0
    missing = 0
    missing_by_status: Counter[str] = Counter()
    for row in canonical:
        value = outcome_value(row, outcome)
        if value is True:
            successes += 1
        elif value is False:
            failures += 1
        else:
            missing += 1
            missing_by_status[_status(row)] += 1
    resolved = successes + failures
    observed_interval = wilson_interval(successes, resolved, confidence=confidence)
    common = {
        "outcome": outcome,
        "experimental_unit": "episode",
        "assigned": assigned,
        "resolved": resolved,
        "successes": successes,
        "failures": failures,
        "missing": missing,
        "missing_by_status": dict(sorted(missing_by_status.items())),
        "observed_rate": successes / resolved if resolved else None,
        "observed_wilson_interval": list(observed_interval),
        "confidence": confidence,
    }
    if assigned == 0:
        return not_estimable("no randomized episodes were supplied", **common)
    lower_bound = successes / assigned
    upper_bound = (successes + missing) / assigned
    if missing:
        return not_estimable(
            "assigned episode outcomes remain unresolved",
            success_rate_bounds=[lower_bound, upper_bound],
            **common,
        )
    interval = wilson_interval(successes, assigned, confidence=confidence)
    return {
        "estimable": True,
        "estimate": successes / assigned,
        "wilson_interval": list(interval),
        "success_rate_bounds": [lower_bound, upper_bound],
        **common,
    }


itt_summary = intention_to_treat


def grouped_intention_to_treat(
    rows: Iterable[Mapping[str, Any]],
    *,
    group_keys: Sequence[str] = ("profile_id",),
    outcome: str = "contract_success",
) -> list[dict[str, Any]]:
    canonical = select_canonical_attempts(rows)
    groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in canonical:
        key = tuple(str(row.get(item, "unassigned")) for item in group_keys)
        groups[key].append(row)
    results: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        label = dict(zip(group_keys, key, strict=True))
        results.append({**label, **intention_to_treat(group, outcome=outcome)})
    return results


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise AnalysisError("a percentile requires at least one value")
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight)


def bootstrap_interval(
    values: Sequence[float],
    *,
    statistic: Callable[[Sequence[float]], float] = fmean,
    confidence: float = 0.95,
    iterations: int = 5000,
    seed: int = 0,
) -> tuple[float, float]:
    """Deterministic percentile bootstrap interval over independent episodes."""

    clean = [float(value) for value in values]
    if not clean or not all(math.isfinite(value) for value in clean):
        raise AnalysisError("bootstrap values must be a non-empty finite sequence")
    if not 0 < confidence < 1:
        raise AnalysisError("confidence must lie strictly between zero and one")
    if isinstance(iterations, bool) or iterations < 100:
        raise AnalysisError("bootstrap iterations must be at least 100")
    generator = random.Random(seed)
    estimates = sorted(
        float(statistic(generator.choices(clean, k=len(clean))))
        for _ in range(iterations)
    )
    tail = (1 - confidence) / 2
    return (_percentile(estimates, tail), _percentile(estimates, 1 - tail))


def _exact_sign_p(positive: int, negative: int) -> float:
    discordant = positive + negative
    if discordant == 0:
        return 1.0
    try:
        from scipy.stats import binomtest  # type: ignore[import-not-found]

        return float(binomtest(positive, discordant, 0.5, alternative="two-sided").pvalue)
    except ImportError:
        edge = min(positive, negative)
        probability = sum(math.comb(discordant, k) for k in range(edge + 1)) / (
            2**discordant
        )
        return min(1.0, 2 * probability)


def _default_match_key(row: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        str(row.get(key, ""))
        for key in (
            "block_id",
            "crn_key",
            "matrix_id",
            "scenario_id",
            "scenario_variant_id",
            "repetition",
            "seed",
            "model_backbone",
            "executor",
            "split",
        )
    )


def paired_binary_comparison(
    rows: Iterable[Mapping[str, Any]],
    profile_a: str,
    profile_b: str,
    *,
    outcome: str = "contract_success",
    match_keys: Sequence[str] | None = None,
    confidence: float = 0.95,
    bootstrap_iterations: int = 5000,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    """Matched risk difference and exact paired test using CRN blocks."""

    if profile_a == profile_b:
        raise AnalysisError("paired profiles must differ")
    canonical = select_canonical_attempts(rows)
    matched: dict[tuple[str, ...], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in canonical:
        profile = str(row.get("profile_id", ""))
        if profile not in {profile_a, profile_b}:
            continue
        key = (
            tuple(str(row.get(item, "")) for item in match_keys)
            if match_keys is not None
            else _default_match_key(row)
        )
        if profile in matched[key]:
            raise AnalysisError(f"duplicate profile {profile} in matched block {key}")
        matched[key][profile] = row

    differences: list[float] = []
    positive = 0
    negative = 0
    tied_success = 0
    tied_failure = 0
    incomplete = 0
    missing_outcome = 0
    for pair in matched.values():
        if set(pair) != {profile_a, profile_b}:
            incomplete += 1
            continue
        value_a = outcome_value(pair[profile_a], outcome)
        value_b = outcome_value(pair[profile_b], outcome)
        if value_a is None or value_b is None:
            missing_outcome += 1
            continue
        difference = float(value_a) - float(value_b)
        differences.append(difference)
        if difference > 0:
            positive += 1
        elif difference < 0:
            negative += 1
        elif value_a:
            tied_success += 1
        else:
            tied_failure += 1
    common = {
        "profile_a": profile_a,
        "profile_b": profile_b,
        "outcome": outcome,
        "experimental_unit": "matched episode pair",
        "candidate_blocks": len(matched),
        "complete_pairs": len(differences),
        "incomplete_pairs": incomplete,
        "missing_outcome_pairs": missing_outcome,
        "a_only_success": positive,
        "b_only_success": negative,
        "both_success": tied_success,
        "both_failure": tied_failure,
    }
    if not differences:
        return not_estimable("no complete matched episode pairs", **common)
    interval = bootstrap_interval(
        differences,
        confidence=confidence,
        iterations=bootstrap_iterations,
        seed=bootstrap_seed,
    )
    # If assigned partners are missing, report the complete-pair descriptive
    # estimate but do not call it the predeclared matched estimand.
    estimate = fmean(differences)
    if incomplete or missing_outcome:
        return not_estimable(
            "matched assignments remain incomplete or unresolved",
            complete_pair_estimate=estimate,
            complete_pair_bootstrap_interval=list(interval),
            mcnemar_exact_p=_exact_sign_p(positive, negative),
            **common,
        )
    return {
        "estimable": True,
        "estimate": estimate,
        "estimand": f"Pr({profile_a} success) - Pr({profile_b} success)",
        "bootstrap_interval": list(interval),
        "confidence": confidence,
        "mcnemar_exact_p": _exact_sign_p(positive, negative),
        "matched_odds_ratio_corrected": (positive + 0.5) / (negative + 0.5),
        **common,
    }


paired_comparison = paired_binary_comparison


def bootstrap_power(
    pilot_differences: Sequence[float],
    sample_sizes: Sequence[int],
    *,
    simulations: int = 2000,
    inner_bootstrap_iterations: int = 500,
    alpha: float = 0.05,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Empirical paired-effect power curve using nested percentile bootstrap.

    This is a pilot planning aid, not a substitute for the preregistered mixed
    model.  It resamples matched episode differences under the empirical pilot
    alternative and counts intervals that exclude zero.
    """

    sizes: list[int] = []
    for raw_size in sample_sizes:
        if isinstance(raw_size, bool) or not isinstance(raw_size, int) or raw_size < 2:
            raise AnalysisError("planned sample sizes must be integers >= 2")
        sizes.append(raw_size)
    if (
        isinstance(simulations, bool)
        or not isinstance(simulations, int)
        or simulations < 100
        or isinstance(inner_bootstrap_iterations, bool)
        or not isinstance(inner_bootstrap_iterations, int)
        or inner_bootstrap_iterations < 100
    ):
        raise AnalysisError("power and inner bootstrap iterations must be integers >= 100")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise AnalysisError("bootstrap power seed must be an integer")
    values = [float(value) for value in pilot_differences]
    if len(values) < 2 or not all(math.isfinite(value) for value in values):
        return [
            not_estimable(
                "at least two finite pilot matched differences are required",
                sample_size=size,
            )
            for size in sizes
        ]
    if not 0 < alpha < 1:
        raise AnalysisError("alpha must lie strictly between zero and one")
    results: list[dict[str, Any]] = []
    for size in sizes:
        # Each planned size owns a derived stream so requesting or reordering
        # other sizes cannot change its estimate.
        stream_material = json.dumps(
            ["bootstrap-power", seed, size], separators=(",", ":")
        ).encode("utf-8")
        generator = random.Random(
            int.from_bytes(hashlib.sha256(stream_material).digest()[:16], "big")
        )
        detected = 0
        for simulation in range(simulations):
            sample = generator.choices(values, k=size)
            interval = bootstrap_interval(
                sample,
                confidence=1 - alpha,
                iterations=inner_bootstrap_iterations,
                seed=generator.randrange(0, 2**63),
            )
            detected += int(interval[0] > 0 or interval[1] < 0)
        interval = wilson_interval(detected, simulations)
        results.append(
            {
                "estimable": True,
                "sample_size": size,
                "power": detected / simulations,
                "monte_carlo_wilson_interval": list(interval),
                "simulations": simulations,
                "alpha": alpha,
                "method": "nested percentile bootstrap of matched episode differences",
            }
        )
    return results


def select_powered_sample_size(
    power_curve: Sequence[Mapping[str, Any]], *, target_power: float = 0.9
) -> int | None:
    if not 0 < target_power <= 1:
        raise AnalysisError("target power must lie in (0, 1]")
    eligible = [
        int(item["sample_size"])
        for item in power_curve
        if item.get("estimable") and float(item.get("power", 0)) >= target_power
    ]
    return min(eligible) if eligible else None


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Return step-down Holm adjusted p-values in original order."""

    clean = [float(value) for value in p_values]
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in clean):
        raise AnalysisError("p-values must be finite and lie in [0, 1]")
    count = len(clean)
    ordered = sorted(enumerate(clean), key=lambda item: item[1])
    adjusted = [0.0] * count
    running = 0.0
    for rank, (original, value) in enumerate(ordered):
        running = max(running, (count - rank) * value)
        adjusted[original] = min(1.0, running)
    return adjusted


def holm_correction(
    hypotheses: Mapping[str, float], *, alpha: float = 0.05
) -> dict[str, dict[str, Any]]:
    if not 0 < alpha < 1:
        raise AnalysisError("alpha must lie strictly between zero and one")
    labels = list(hypotheses)
    raw = [float(hypotheses[label]) for label in labels]
    adjusted = holm_adjust(raw)
    return {
        label: {
            "raw_p": raw[index],
            "holm_adjusted_p": adjusted[index],
            "reject": adjusted[index] <= alpha,
            "alpha": alpha,
        }
        for index, label in enumerate(labels)
    }


def failure_taxonomy(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Count preserved outcomes and post-run diagnosis labels separately."""

    canonical = select_canonical_attempts(rows)
    statuses: Counter[str] = Counter()
    layers: Counter[str] = Counter()
    diagnoses: Counter[str] = Counter()
    unresolved = 0
    for row in canonical:
        status = _status(row)
        statuses[status] += 1
        if status == VALID_PASS:
            continue
        layer = str(
            row.get("failure_layer", row.get("diagnosis_failure_layer", "unassigned"))
            or "unassigned"
        )
        if layer not in FAILURE_LAYERS:
            layer = "unassigned"
        layers[layer] += 1
        diagnosis = str(row.get("primary_diagnosis", "unassigned") or "unassigned")
        diagnoses[diagnosis] += 1
        if status in {INFRA_INTERRUPTED, INVALID_HARNESS, NOT_RUN}:
            unresolved += 1
    return {
        "experimental_unit": "episode",
        "episodes": len(canonical),
        "status_counts": {status: statuses[status] for status in sorted(KNOWN_STATUSES)},
        "failure_layer_counts": {
            layer: layers[layer] for layer in FAILURE_LAYERS if layers[layer]
        },
        "primary_diagnosis_counts": dict(sorted(diagnoses.items())),
        "unresolved_or_invalid": unresolved,
        "note": "oracle outcomes and post-run diagnoses are distinct fields",
    }


def attempt_history(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Report every preserved attempt separately from episode-level estimands."""

    materialized = [dict(row) for row in rows]

    def number(row: Mapping[str, Any]) -> int:
        try:
            return int(row.get("attempt_number", 0))
        except (TypeError, ValueError) as error:
            raise AnalysisError("attempt history contains an invalid attempt number") from error

    attempts = [row for row in materialized if number(row) > 0]
    statuses = Counter(_status(row) for row in attempts)
    return {
        "scheduled_episodes": len(select_canonical_attempts(materialized)),
        "preserved_attempts": len(attempts),
        "retry_attempts": sum(number(row) > 1 for row in attempts),
        "status_counts": {status: statuses[status] for status in sorted(KNOWN_STATUSES)},
        "artifact_audit_failures": sum(
            row.get("artifact_audit_passed") is False for row in attempts
        ),
    }


def load_campaign_rows(campaign_root: str | Path) -> tuple[dict[str, Any], ...]:
    """Regenerate analysis rows from RUN_STATE and immutable result artifacts."""

    root = Path(campaign_root).resolve()
    try:
        from .campaign import CampaignRegistry

        state = CampaignRegistry(root).snapshot()
    except Exception as error:
        raise AnalysisError(f"cannot validate RUN_STATE.json: {error}") from error
    rows: list[dict[str, Any]] = []
    for schedule_id, entry in state.get("entries", {}).items():
        assignment = dict(entry["assignment"])
        attempts = entry.get("attempts", [])
        if not attempts:
            rows.append(
                {
                    **assignment,
                    "schedule_id": schedule_id,
                    "attempt_number": 0,
                    "run_status": NOT_RUN,
                    "oracle_verdict": None,
                }
            )
            continue
        for attempt in attempts:
            row = {
                **assignment,
                **attempt,
                "schedule_id": schedule_id,
                "run_status": attempt["status"],
            }
            artifact = (root / attempt["artifact_path"]).resolve()
            try:
                artifact.relative_to(root)
            except ValueError as error:
                raise AnalysisError("registry artifact path escapes campaign root") from error
            result_path = artifact / "results.json"
            if result_path.is_file():
                try:
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                except (OSError, UnicodeError, json.JSONDecodeError) as error:
                    row["result_load_error"] = str(error)
                    row["analysis_input_error"] = "results.json could not be parsed"
                else:
                    if isinstance(result, dict):
                        # Registry assignment and attempt identity remain
                        # authoritative; result artifacts may add outcomes only.
                        protected = set(assignment) | set(attempt) | {
                            "schedule_id",
                            "attempt_number",
                            "run_status",
                            "status",
                            "artifact_path",
                        }
                        conflicts: dict[str, dict[str, Any]] = {}
                        for key, value in result.items():
                            if key in protected:
                                authoritative_key = "status" if key == "run_status" else key
                                authoritative = row.get(authoritative_key)
                                if key == "run_status":
                                    authoritative = row["run_status"]
                                if authoritative is not None and value != authoritative:
                                    conflicts[key] = {
                                        "registry": authoritative,
                                        "result": value,
                                    }
                            else:
                                row[key] = value
                        if conflicts:
                            row["result_identity_conflicts"] = conflicts
                    else:
                        row["analysis_input_error"] = "results.json is not an object"
            elif attempt["status"] in {
                VALID_PASS,
                VALID_SYSTEM_FAILURE,
                ABORTED_SAFETY,
            }:
                row["analysis_input_error"] = "resolved attempt lacks results.json"
            diagnosis_path = artifact / "diagnosis.json"
            if diagnosis_path.is_file():
                try:
                    diagnosis = json.loads(
                        diagnosis_path.read_text(encoding="utf-8")
                    )
                except (OSError, UnicodeError, json.JSONDecodeError) as error:
                    row["diagnosis_load_error"] = str(error)
                else:
                    if isinstance(diagnosis, dict):
                        layer = diagnosis.get("failure_layer_code")
                        if layer is None:
                            layer = diagnosis.get("likely_failure_layer")
                            if isinstance(layer, str):
                                layer = layer.split(" (", 1)[0]
                        primary = diagnosis.get("primary_diagnosis")
                        if isinstance(layer, str):
                            row["diagnosis_failure_layer"] = layer
                        if isinstance(primary, str):
                            row["primary_diagnosis"] = primary.split(" (", 1)[0]
            rows.append(row)
    return tuple(rows)


def analyze_campaign(
    campaign_root: str | Path,
    *,
    outcome: str = "contract_success",
    reference_profile: str = "T5",
    check_live_source: bool = True,
) -> dict[str, Any]:
    root = Path(campaign_root).resolve()
    rows = [dict(row) for row in load_campaign_rows(root)]
    try:
        from .campaign import CampaignRegistry, verify_frozen_campaign

        manifest = verify_frozen_campaign(root, check_source=check_live_source)
        registry = CampaignRegistry(root)
        state = registry.snapshot()
        completion = registry.completion()
    except Exception as error:
        raise AnalysisError(f"cannot validate frozen campaign: {error}") from error
    frozen_metadata = manifest.get("metadata")
    if not isinstance(frozen_metadata, Mapping):
        raise AnalysisError("frozen manifest metadata is malformed")
    analysis_spec = frozen_metadata.get("analysis_spec", {})
    if not isinstance(analysis_spec, Mapping):
        raise AnalysisError("frozen metadata lacks a valid analysis_spec")
    frozen_outcomes = analysis_spec.get("confirmatory_outcomes", ())
    frozen_reference = analysis_spec.get("reference_profile")
    predeclared_request = (
        isinstance(frozen_outcomes, list)
        and outcome in frozen_outcomes
        and reference_profile == frozen_reference
    )
    live_attempt_audits: list[dict[str, Any]] = []
    for row in rows:
        attempt_number = row.get("attempt_number")
        if not isinstance(attempt_number, int) or attempt_number < 1:
            continue
        artifact_path = row.get("artifact_path")
        if not isinstance(artifact_path, str):
            row["analysis_input_error"] = "registry row lacks an artifact path"
            live_attempt_audits.append(
                {
                    "schedule_id": row.get("schedule_id"),
                    "attempt_number": attempt_number,
                    "passed": False,
                    "reason": row["analysis_input_error"],
                }
            )
            continue
        try:
            from .audit import audit_attempt_artifacts

            artifact_report = audit_attempt_artifacts(
                root / artifact_path,
                verify_video=True,
                require_final_state=_status(row)
                in {VALID_PASS, VALID_SYSTEM_FAILURE},
            )
        except Exception as error:
            artifact_report = {
                "passed": False,
                "errors": [f"audit raised {type(error).__name__}: {error}"],
            }
        status = _status(row)
        registry_flag_required = status in {VALID_PASS, VALID_SYSTEM_FAILURE}
        audit_passed = artifact_report.get("passed") is True and (
            not registry_flag_required or row.get("artifact_audit_passed") is True
        )
        if not audit_passed:
            row["analysis_input_error"] = (
                "current independent artifact audit failed; outcomes are unavailable"
            )
        report_bytes = json.dumps(
            artifact_report,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        live_attempt_audits.append(
            {
                "schedule_id": row.get("schedule_id"),
                "attempt_number": attempt_number,
                "registry_audit_passed": row.get("artifact_audit_passed") is True,
                "passed": audit_passed,
                "audit_report_sha256": hashlib.sha256(report_bytes).hexdigest(),
                "errors": list(map(str, artifact_report.get("errors", ()))),
            }
        )
    canonical = select_canonical_attempts(rows)
    pairs: dict[str, dict[str, Any]] = {}
    by_matrix: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in canonical:
        by_matrix[str(row.get("matrix_id", "unassigned"))].append(row)
    frozen_families = analysis_spec.get("families", [])
    if not isinstance(frozen_families, list):
        raise AnalysisError("frozen analysis families are malformed")
    for family in frozen_families:
        if not isinstance(family, Mapping):
            raise AnalysisError("frozen analysis family is malformed")
        matrix_id = str(family.get("matrix_id", ""))
        matrix_rows = by_matrix.get(matrix_id, [])
        family_reference = str(family.get("reference_profile", ""))
        comparisons = family.get("comparison_profiles", [])
        if not isinstance(comparisons, list):
            raise AnalysisError("frozen comparison profile family is malformed")
        matrix_labels: list[str] = []
        for profile in comparisons:
            label = f"{matrix_id}:{family_reference}_vs_{profile}"
            pairs[label] = paired_binary_comparison(
                matrix_rows,
                family_reference,
                str(profile),
                outcome=outcome,
            )
            matrix_labels.append(label)
        p_values = {
            label: float(pairs[label]["mcnemar_exact_p"])
            for label in matrix_labels
            if pairs[label].get("estimable")
            and pairs[label].get("mcnemar_exact_p") is not None
        }
        if len(p_values) == len(matrix_labels):
            holm = holm_correction(p_values) if p_values else {}
            for label, correction in holm.items():
                pairs[label]["multiplicity"] = {
                    **correction,
                    "family": matrix_id,
                    "family_size": len(matrix_labels),
                    "family_complete": True,
                }
        else:
            for label in matrix_labels:
                pairs[label]["multiplicity"] = {
                    "raw_p": pairs[label].get("mcnemar_exact_p"),
                    "holm_adjusted_p": None,
                    "reject": None,
                    "alpha": 0.05,
                    "family": matrix_id,
                    "family_size": len(matrix_labels),
                    "estimable_comparisons": len(p_values),
                    "family_complete": False,
                    "reason": "at least one frozen comparison is not estimable",
                }
    mode = str(manifest.get("mode", "unknown"))
    unresolved_outcome = sum(
        outcome_value(row, outcome) is None for row in canonical
    )
    unaudited = sum(report["passed"] is not True for report in live_attempt_audits)
    authorized = bool(state.get("execution_authorized"))
    if mode == "locked":
        if not predeclared_request:
            analysis_tier = "locked_post_hoc_exploratory"
        elif not check_live_source:
            analysis_tier = "locked_source_unchecked_not_confirmatory"
        elif not authorized:
            analysis_tier = "locked_unauthorized_not_confirmatory"
        elif not completion["complete"] or unresolved_outcome or unaudited:
            analysis_tier = "locked_incomplete_not_confirmatory"
        else:
            analysis_tier = "confirmatory_locked"
    else:
        analysis_tier = "pilot_nonconfirmatory"
    return {
        "schema_version": 1,
        "campaign_id": manifest.get("campaign_id"),
        "protocol_sha256": manifest.get("protocol_sha256"),
        "mode": mode,
        "analysis_tier": analysis_tier,
        "analysis_gate": {
            "execution_authorized": authorized,
            "registry_complete": completion["complete"],
            "unresolved_outcomes": unresolved_outcome,
            "resolved_attempts_without_passing_audit": unaudited,
            "live_source_checked": check_live_source,
            "request_matches_frozen_analysis_spec": predeclared_request,
        },
        "analysis_spec": dict(analysis_spec),
        "outcome": outcome,
        "reference_profile": reference_profile,
        "episode_count": len(canonical),
        "by_profile": grouped_intention_to_treat(
            canonical, group_keys=("matrix_id", "profile_id"), outcome=outcome
        ),
        "paired_comparisons": pairs,
        "failure_taxonomy": failure_taxonomy(canonical),
        "attempt_history": attempt_history(rows),
        "live_attempt_audits": live_attempt_audits,
        "multiplicity_family": "reference-profile pairwise comparisons within matrix",
        "multiplicity_method": "Holm",
    }


def render_analysis_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Campaign analysis",
        "",
        f"- Campaign: `{report.get('campaign_id', 'unknown')}`",
        f"- Protocol: `{report.get('protocol_sha256', 'unknown')}`",
        f"- Tier: `{report.get('analysis_tier', 'unknown')}`",
        f"- Experimental unit: episode",
        f"- Outcome: `{report.get('outcome', 'unknown')}`",
        "",
        "## ITT summaries",
        "",
    ]
    for item in report.get("by_profile", []):
        estimate = item.get("estimate")
        rendered = "NOT ESTIMABLE" if estimate is None else f"{float(estimate):.4f}"
        lines.append(
            f"- `{item.get('profile_id')}`: {rendered}; assigned={item.get('assigned')}, "
            f"missing={item.get('missing')}"
        )
    lines.extend(["", "## Machine-readable report", "", "```json"])
    lines.append(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    lines.extend(["```", ""])
    return "\n".join(lines)


def write_analysis(
    campaign_root: str | Path,
    report: Mapping[str, Any],
    *,
    stem: str = "analysis_results",
) -> tuple[Path, Path]:
    if not stem or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for character in stem):
        raise AnalysisError("analysis output stem must be lowercase and path-safe")
    output = Path(campaign_root).resolve() / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / f"{stem}.json"
    markdown_path = output / f"{stem}.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_analysis_markdown(report), encoding="utf-8")
    return json_path, markdown_path


__all__ = [
    "ABORTED_SAFETY",
    "AnalysisError",
    "FAILURE_LAYERS",
    "INFRA_INTERRUPTED",
    "INVALID_HARNESS",
    "KNOWN_STATUSES",
    "NOT_RUN",
    "VALID_PASS",
    "VALID_SYSTEM_FAILURE",
    "analyze_campaign",
    "attempt_history",
    "bootstrap_interval",
    "bootstrap_power",
    "failure_taxonomy",
    "grouped_intention_to_treat",
    "holm_adjust",
    "holm_correction",
    "intention_to_treat",
    "itt_summary",
    "load_campaign_rows",
    "not_estimable",
    "outcome_value",
    "paired_binary_comparison",
    "paired_comparison",
    "render_analysis_markdown",
    "select_canonical_attempts",
    "select_powered_sample_size",
    "wilson_interval",
    "write_analysis",
]
