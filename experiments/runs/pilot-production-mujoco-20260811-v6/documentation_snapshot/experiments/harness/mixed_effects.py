"""Deterministic confirmatory binary mixed-effects analysis for RoboPref.

This module deliberately implements one narrow, auditable model rather than a
general formula language.  It fits a Bernoulli-logit generalized linear mixed
model with treatment-coded fixed effects and nested random intercepts using
deterministic nested Gauss-Hermite marginal-likelihood integration.  Only
NumPy and SciPy are required.

The public functions return JSON-serializable dictionaries.  Scientific fit
failures are represented as explicit ``estimable=False`` records; they never
produce a p-value or silently fall back to an ordinary logistic regression.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import platform
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import scipy
from scipy.optimize import linprog, minimize, root_scalar
from scipy.special import expit, logsumexp

from experiments.harness.analysis import AnalysisError, outcome_value, wilson_interval
from experiments.harness.recording import durable_write


SCHEMA_VERSION = "robopref.binary-mixed-effects.v1"
POWER_SCHEMA_VERSION = "robopref.binary-mixed-effects-power.v1"
DEFAULT_REPETITION_CANDIDATES = (10, 15, 20, 25)


class MixedEffectsError(AnalysisError):
    """Raised when the declared model or raw input is malformed."""


@dataclass(frozen=True)
class _Block:
    """One independent top-level random-effect integration block."""

    template: str
    observation_indices: np.ndarray
    membership: np.ndarray
    instance_levels: tuple[str, ...]


@dataclass(frozen=True)
class _PreparedData:
    y: np.ndarray
    x: np.ndarray
    fixed_names: tuple[str, ...]
    rows: tuple[dict[str, Any], ...]
    system_levels: tuple[str, ...]
    factor_levels: dict[str, tuple[str, ...]]
    factor_references: dict[str, str]
    blocks: tuple[_Block, ...]
    input_sha256: str
    overlap_counts: dict[str, int]


@dataclass(frozen=True)
class _CoreFit:
    theta: np.ndarray
    covariance: np.ndarray
    objective: float
    blocks: tuple[tuple[np.ndarray, np.ndarray, int, float], ...]
    optimizer: Any
    hessian: np.ndarray
    validation_objective: float


def _json_scalar(value: Any, *, field: str) -> str:
    if value is None or isinstance(value, (dict, list, tuple, set)):
        raise MixedEffectsError(f"{field} must be a non-null scalar label")
    if isinstance(value, float) and not math.isfinite(value):
        raise MixedEffectsError(f"{field} must be finite")
    rendered = str(value)
    if not rendered:
        raise MixedEffectsError(f"{field} must be non-empty")
    return rendered


def _binary_without_status(row: Mapping[str, Any], outcome: str) -> bool | None:
    if outcome not in row or row[outcome] in (None, ""):
        return None
    value = row[outcome]
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool) and value in {0, 1}:
        return bool(value)
    if isinstance(value, float) and value in {0.0, 1.0}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "pass", "passed", "yes", "1"}:
            return True
        if normalized in {"false", "fail", "failed", "no", "0"}:
            return False
    raise MixedEffectsError(f"{outcome} is not binary: {value!r}")


def _read_outcome(row: Mapping[str, Any], outcome: str) -> bool | None:
    if "run_status" in row or "status" in row:
        return outcome_value(row, outcome)
    return _binary_without_status(row, outcome)


def _not_estimable(reason: str, **details: Any) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "estimable": False,
        "estimate": None,
        "reason": reason,
        "inference": {
            "p_values": None,
            "confidence_intervals": None,
            "policy": "No inferential values are emitted for an invalid fit.",
        },
        **details,
    }


def _model_specification(
    *,
    outcome: str,
    system_field: str,
    reference_system: str,
    scenario_factors: Sequence[str],
    factor_references: Mapping[str, str],
    template_field: str,
    instance_field: str | None,
    confidence: float,
    likelihood_quadrature_order: int,
    likelihood_validation_order: int,
    marginal_quadrature_order: int,
    singular_sd_threshold: float,
) -> dict[str, Any]:
    fixed_terms = [
        f'treatment({system_field}, reference={json.dumps(reference_system)})'
    ]
    fixed_terms.extend(
        f'treatment({factor}, reference={json.dumps(factor_references[factor])})'
        for factor in scenario_factors
    )
    random_terms = [f"(1 | {template_field})"]
    if instance_field is not None:
        random_terms.append(f"(1 | {template_field}:{instance_field})")
    formula = f"binary({outcome}) ~ 1 + " + " + ".join(fixed_terms + random_terms)
    return {
        "formula": formula,
        "family": "Bernoulli",
        "link": "logit",
        "experimental_unit": "episode",
        "fixed_effects": [system_field, *scenario_factors],
        "fixed_effect_contrasts": "treatment/reference coding; no implicit interactions",
        "reference_levels": {
            system_field: reference_system,
            **{factor: factor_references[factor] for factor in scenario_factors},
        },
        "random_effects": random_terms,
        "random_effect_distribution": "independent zero-mean Gaussian intercepts",
        "likelihood": {
            "method": (
                "nested non-adaptive Gauss-Hermite integration of the marginal "
                "Bernoulli likelihood"
            ),
            "optimizer_order_per_random_effect": likelihood_quadrature_order,
            "validation_order_per_random_effect": likelihood_validation_order,
            "normal_expectation_transform": (
                "b = sqrt(2) * SD * Hermite node; normalized weights = weight / sqrt(pi)"
            ),
        },
        "confidence_level": confidence,
        "marginal_means": {
            "method": "empirical standardization over observed scenario-factor rows",
            "random_effect_integration": (
                "Gauss-Hermite integration over a new template and nested instance"
            ),
            "quadrature_order": marginal_quadrature_order,
            "interval_method": (
                "two-sided normal delta method using the full observed-Hessian covariance"
            ),
        },
        "singular_fit_definition": (
            f"any random-intercept SD <= {singular_sd_threshold:g}, a variance "
            "parameter at an optimizer bound, or a non-positive observed Hessian"
        ),
    }


def _prepare_data(
    rows: Iterable[Mapping[str, Any]],
    *,
    outcome: str,
    system_field: str,
    reference_system: str,
    scenario_factors: Sequence[str],
    factor_references: Mapping[str, Any],
    template_field: str,
    instance_field: str | None,
    minimum_random_groups: int,
) -> tuple[_PreparedData | None, dict[str, Any] | None]:
    if len(set(scenario_factors)) != len(tuple(scenario_factors)):
        raise MixedEffectsError("scenario_factors must not contain duplicates")
    reserved = {system_field, template_field}
    if instance_field is not None:
        reserved.add(instance_field)
    if any(factor in reserved for factor in scenario_factors):
        raise MixedEffectsError("fixed and random grouping fields must be distinct")
    if set(factor_references) != set(scenario_factors):
        raise MixedEffectsError(
            "factor_references must provide exactly one explicit reference "
            "for every scenario factor"
        )
    normalized_reference = _json_scalar(reference_system, field="reference_system")
    normalized_factor_refs = {
        factor: _json_scalar(factor_references[factor], field=f"reference for {factor}")
        for factor in scenario_factors
    }

    normalized: list[dict[str, Any]] = []
    missing_outcome = 0
    for source in rows:
        row = dict(source)
        value = _read_outcome(row, outcome)
        if value is None:
            missing_outcome += 1
            continue
        record: dict[str, Any] = {
            "outcome": int(value),
            "system": _json_scalar(row.get(system_field), field=system_field),
            "template": _json_scalar(row.get(template_field), field=template_field),
            "factors": {
                factor: _json_scalar(row.get(factor), field=factor)
                for factor in scenario_factors
            },
        }
        if instance_field is not None:
            record["instance"] = _json_scalar(
                row.get(instance_field), field=instance_field
            )
        normalized.append(record)
    if missing_outcome:
        return None, _not_estimable(
            "assigned episodes have unresolved or invalid binary outcomes",
            input_summary={
                "episodes_supplied": len(normalized) + missing_outcome,
                "episodes_resolved": len(normalized),
                "episodes_missing": missing_outcome,
            },
        )
    if not normalized:
        return None, _not_estimable(
            "no resolved episodes were supplied",
            input_summary={
                "episodes_supplied": 0,
                "episodes_resolved": 0,
                "episodes_missing": 0,
            },
        )

    canonical = tuple(
        sorted(
            normalized,
            key=lambda record: json.dumps(
                record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ),
        )
    )
    raw_payload = json.dumps(
        canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    input_sha256 = hashlib.sha256(raw_payload).hexdigest()

    system_levels = tuple(sorted({record["system"] for record in canonical}))
    if normalized_reference not in system_levels:
        raise MixedEffectsError("reference_system is absent from the analysis rows")
    if len(system_levels) < 2:
        return None, _not_estimable(
            "at least two system conditions are required",
            input_sha256=input_sha256,
            input_summary={"episodes_supplied": len(canonical)},
        )
    ordered_systems = (normalized_reference,) + tuple(
        level for level in system_levels if level != normalized_reference
    )

    factor_levels: dict[str, tuple[str, ...]] = {}
    for factor in scenario_factors:
        levels = tuple(sorted({record["factors"][factor] for record in canonical}))
        reference = normalized_factor_refs[factor]
        if reference not in levels:
            raise MixedEffectsError(f"reference level {reference!r} is absent for {factor}")
        if len(levels) < 2:
            return None, _not_estimable(
                "a declared scenario fixed effect has fewer than two observed levels",
                input_sha256=input_sha256,
                diagnostics={"factor": factor, "observed_levels": list(levels)},
            )
        factor_levels[factor] = (reference,) + tuple(
            level for level in levels if level != reference
        )

    fixed_names = ["Intercept"]
    fixed_names.extend(
        f"{system_field}[{level}]" for level in ordered_systems[1:]
    )
    for factor in scenario_factors:
        fixed_names.extend(
            f"{factor}[{level}]" for level in factor_levels[factor][1:]
        )
    x = np.ones((len(canonical), len(fixed_names)), dtype=np.float64)
    column = 1
    for level in ordered_systems[1:]:
        x[:, column] = [record["system"] == level for record in canonical]
        column += 1
    for factor in scenario_factors:
        for level in factor_levels[factor][1:]:
            x[:, column] = [record["factors"][factor] == level for record in canonical]
            column += 1
    y = np.asarray([record["outcome"] for record in canonical], dtype=np.float64)

    rank = int(np.linalg.matrix_rank(x))
    if rank != x.shape[1]:
        return None, _not_estimable(
            "fixed-effect design matrix is rank deficient",
            input_sha256=input_sha256,
            diagnostics={
                "fixed_columns": fixed_names,
                "design_rank": rank,
                "design_columns": int(x.shape[1]),
            },
        )
    design_condition = float(np.linalg.cond(x))
    if not math.isfinite(design_condition) or design_condition >= 1e8:
        return None, _not_estimable(
            "fixed-effect design matrix is numerically ill-conditioned",
            input_sha256=input_sha256,
            diagnostics={
                "fixed_columns": fixed_names,
                "design_condition_number": design_condition,
                "maximum_allowed_condition_number": 1e8,
            },
        )
    if len(canonical) < x.shape[1] + 10:
        return None, _not_estimable(
            "too few episodes for the declared fixed-effect model",
            input_sha256=input_sha256,
            diagnostics={"episodes": len(canonical), "fixed_parameters": int(x.shape[1])},
        )

    by_template: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(canonical):
        by_template[record["template"]].append(index)
    if len(by_template) < minimum_random_groups:
        return None, _not_estimable(
            "insufficient scenario-template groups for a random intercept",
            input_sha256=input_sha256,
            diagnostics={
                "template_groups": len(by_template),
                "minimum_random_groups": minimum_random_groups,
            },
        )

    blocks: list[_Block] = []
    nested_group_members: dict[tuple[str, str], list[int]] = defaultdict(list)
    for template in sorted(by_template):
        indices = np.asarray(by_template[template], dtype=np.int64)
        if instance_field is None:
            membership = np.ones((len(indices), 1), dtype=np.float64)
            instances: tuple[str, ...] = ()
        else:
            instances = tuple(
                sorted({canonical[index]["instance"] for index in indices})
            )
            if len(instances) < 2:
                return None, _not_estimable(
                    "template and nested-instance variances are not separately identifiable",
                    input_sha256=input_sha256,
                    diagnostics={"template": template, "nested_instance_groups": len(instances)},
                )
            instance_index = {level: position for position, level in enumerate(instances)}
            membership = np.zeros((len(indices), 1 + len(instances)), dtype=np.float64)
            membership[:, 0] = 1.0
            for local, global_index in enumerate(indices):
                instance = canonical[int(global_index)]["instance"]
                membership[local, 1 + instance_index[instance]] = 1.0
                nested_group_members[(template, instance)].append(int(global_index))
        blocks.append(
            _Block(
                template=template,
                observation_indices=indices,
                membership=membership,
                instance_levels=instances,
            )
        )

    if instance_field is not None:
        if len(nested_group_members) < minimum_random_groups:
            return None, _not_estimable(
                "insufficient generated-instance groups for a random intercept",
                input_sha256=input_sha256,
                diagnostics={
                    "nested_instance_groups": len(nested_group_members),
                    "minimum_random_groups": minimum_random_groups,
                },
            )
        singleton = [key for key, members in nested_group_members.items() if len(members) < 2]
        if singleton:
            return None, _not_estimable(
                "generated-instance groups must contain at least two matched system episodes",
                input_sha256=input_sha256,
                diagnostics={"singleton_nested_groups": len(singleton)},
            )

    overlap_counts: dict[str, int] = {}
    if instance_field is None:
        units = {(record["template"],): set() for record in canonical}
        for record in canonical:
            units[(record["template"],)].add(record["system"])
    else:
        units: dict[tuple[str, ...], set[str]] = defaultdict(set)
        for record in canonical:
            units[(record["template"], record["instance"])].add(record["system"])
    for level in ordered_systems[1:]:
        overlap = sum(
            normalized_reference in systems and level in systems
            for systems in units.values()
        )
        overlap_counts[level] = overlap
        if overlap < minimum_random_groups:
            return None, _not_estimable(
                "insufficient matched random-effect groups for a system contrast",
                input_sha256=input_sha256,
                diagnostics={
                    "comparison_system": level,
                    "matched_groups": overlap,
                    "minimum_random_groups": minimum_random_groups,
                },
            )

    return (
        _PreparedData(
            y=y,
            x=x,
            fixed_names=tuple(fixed_names),
            rows=canonical,
            system_levels=ordered_systems,
            factor_levels=factor_levels,
            factor_references=normalized_factor_refs,
            blocks=tuple(blocks),
            input_sha256=input_sha256,
            overlap_counts=overlap_counts,
        ),
        None,
    )


def _separation_diagnostic(y: np.ndarray, x: np.ndarray) -> dict[str, Any]:
    """Detect complete or quasi-complete fixed-effect separation by LP."""

    signed = (2.0 * y - 1.0)[:, None] * x
    bounds = [(None, None)] * x.shape[1]
    complete = linprog(
        np.zeros(x.shape[1]),
        A_ub=-signed,
        b_ub=-np.ones(len(y)),
        bounds=bounds,
        method="highs",
    )
    if complete.success:
        return {
            "detected": True,
            "kind": "complete",
            "method": "linear-program feasibility of signed margins >= 1",
            "solver_status": int(complete.status),
        }
    quasi_a = np.vstack((-signed, -np.sum(signed, axis=0, keepdims=True)))
    quasi_b = np.concatenate((np.zeros(len(y)), np.asarray([-1.0])))
    quasi = linprog(
        np.zeros(x.shape[1]),
        A_ub=quasi_a,
        b_ub=quasi_b,
        bounds=bounds,
        method="highs",
    )
    return {
        "detected": bool(quasi.success),
        "kind": "quasi-complete" if quasi.success else None,
        "method": (
            "linear-program feasibility of nonnegative signed margins with one strict margin"
        ),
        "solver_status": int(quasi.status),
    }


def _joint_log_density(
    b: np.ndarray,
    offset: np.ndarray,
    y: np.ndarray,
    membership: np.ndarray,
    precision: np.ndarray,
) -> float:
    eta = offset + membership @ b
    conditional = np.sum(y * eta - np.logaddexp(0.0, eta))
    return float(conditional - 0.5 * np.dot(precision * b, b))


def _solve_random_mode(
    *,
    offset: np.ndarray,
    y: np.ndarray,
    membership: np.ndarray,
    precision: np.ndarray,
    tolerance: float,
    maximum_iterations: int,
) -> tuple[np.ndarray, np.ndarray, int, float] | None:
    b = np.zeros(membership.shape[1], dtype=np.float64)
    for iteration in range(1, maximum_iterations + 1):
        eta = offset + membership @ b
        probabilities = expit(eta)
        weights = probabilities * (1.0 - probabilities)
        gradient = membership.T @ (y - probabilities) - precision * b
        hessian = (membership.T * weights) @ membership + np.diag(precision)
        if not np.all(np.isfinite(hessian)) or not np.all(np.isfinite(gradient)):
            return None
        maximum_gradient = float(np.max(np.abs(gradient)))
        if maximum_gradient <= tolerance:
            sign, _ = np.linalg.slogdet(hessian)
            return (b, hessian, iteration, maximum_gradient) if sign > 0 else None
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            return None
        current = _joint_log_density(b, offset, y, membership, precision)
        directional = float(np.dot(gradient, step))
        scale = 1.0
        accepted = False
        while scale >= 2.0**-24:
            candidate = b + scale * step
            value = _joint_log_density(
                candidate, offset, y, membership, precision
            )
            if value >= current + 1e-4 * scale * directional:
                b = candidate
                accepted = True
                break
            scale *= 0.5
        if not accepted:
            # At machine precision the Armijo increment can round away even
            # though the score is already scientifically negligible.
            if maximum_gradient <= 100.0 * tolerance:
                sign, _ = np.linalg.slogdet(hessian)
                return (b, hessian, iteration, maximum_gradient) if sign > 0 else None
            return None
    return None


def _conditional_modes(
    prepared: _PreparedData,
    theta: np.ndarray,
    *,
    instance_field: str | None,
    inner_tolerance: float,
    inner_maximum_iterations: int,
) -> tuple[tuple[np.ndarray, np.ndarray, int, float], ...] | None:
    """Compute posterior modes solely for QQ/residual diagnostics.

    These modes do not contribute to the fitted likelihood, optimizer, Hessian,
    covariance, p-values, or marginal estimates.  The inferential likelihood is
    exclusively the nested Gauss-Hermite objective below.
    """

    fixed_count = prepared.x.shape[1]
    variance_count = 1 + int(instance_field is not None)
    if len(theta) != fixed_count + variance_count or not np.all(np.isfinite(theta)):
        return None
    beta = theta[:fixed_count]
    standard_deviations = np.exp(theta[fixed_count:])
    solved: list[tuple[np.ndarray, np.ndarray, int, float]] = []
    for block in prepared.blocks:
        local_y = prepared.y[block.observation_indices]
        offset = prepared.x[block.observation_indices] @ beta
        if instance_field is None:
            precision = np.asarray([standard_deviations[0] ** -2])
        else:
            precision = np.concatenate(
                (
                    np.asarray([standard_deviations[0] ** -2]),
                    np.full(len(block.instance_levels), standard_deviations[1] ** -2),
                )
            )
        solution = _solve_random_mode(
            offset=offset,
            y=local_y,
            membership=block.membership,
            precision=precision,
            tolerance=inner_tolerance,
            maximum_iterations=inner_maximum_iterations,
        )
        if solution is None:
            return None
        solved.append(solution)
    return tuple(solved)


def _quadrature_objective_factory(
    prepared: _PreparedData,
    *,
    instance_field: str | None,
    quadrature_order: int,
):
    """Build the exact declared nested Gaussian-quadrature likelihood."""

    fixed_count = prepared.x.shape[1]
    variance_count = 1 + int(instance_field is not None)
    nodes, weights = np.polynomial.hermite.hermgauss(quadrature_order)
    normalized_log_weights = np.log(weights) - 0.5 * math.log(math.pi)

    def evaluate(theta: np.ndarray) -> float:
        if len(theta) != fixed_count + variance_count or not np.all(np.isfinite(theta)):
            return 1e100
        beta = theta[:fixed_count]
        standard_deviations = np.exp(theta[fixed_count:])
        template_nodes = math.sqrt(2.0) * standard_deviations[0] * nodes
        instance_nodes = (
            math.sqrt(2.0) * standard_deviations[1] * nodes
            if instance_field is not None
            else None
        )
        total_log_likelihood = 0.0
        for block in prepared.blocks:
            local_y = prepared.y[block.observation_indices]
            offset = prepared.x[block.observation_indices] @ beta
            if instance_field is None:
                eta = offset[:, None] + template_nodes[None, :]
                conditional = np.sum(
                    local_y[:, None] * eta - np.logaddexp(0.0, eta), axis=0
                )
                block_log_likelihood = float(
                    logsumexp(normalized_log_weights + conditional)
                )
            else:
                assert instance_nodes is not None
                template_conditional = np.zeros(quadrature_order, dtype=np.float64)
                for instance_index in range(len(block.instance_levels)):
                    selected = block.membership[:, instance_index + 1] == 1.0
                    instance_y = local_y[selected]
                    instance_offset = offset[selected]
                    eta = (
                        instance_offset[:, None, None]
                        + template_nodes[None, :, None]
                        + instance_nodes[None, None, :]
                    )
                    conditional = np.sum(
                        instance_y[:, None, None] * eta
                        - np.logaddexp(0.0, eta),
                        axis=0,
                    )
                    template_conditional += logsumexp(
                        conditional + normalized_log_weights[None, :], axis=1
                    )
                block_log_likelihood = float(
                    logsumexp(normalized_log_weights + template_conditional)
                )
            if not math.isfinite(block_log_likelihood):
                return 1e100
            total_log_likelihood += block_log_likelihood
        return -total_log_likelihood

    return evaluate


def _initial_fixed_effects(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    rate = min(1.0 - 1e-4, max(1e-4, float(np.mean(y))))
    initial = np.zeros(x.shape[1], dtype=np.float64)
    initial[0] = math.log(rate / (1.0 - rate))

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = x @ beta
        value = float(np.sum(np.logaddexp(0.0, eta) - y * eta))
        gradient = x.T @ (expit(eta) - y)
        return value, gradient

    result = minimize(
        objective,
        initial,
        method="BFGS",
        jac=True,
        options={"gtol": 1e-8, "maxiter": 300},
    )
    if result.success and np.all(np.isfinite(result.x)):
        return np.asarray(result.x, dtype=np.float64)
    return initial


def _numerical_hessian(function, point: np.ndarray, *, relative_step: float) -> np.ndarray:
    count = len(point)
    steps = relative_step * np.maximum(1.0, np.abs(point))
    hessian = np.empty((count, count), dtype=np.float64)
    centre = float(function(point))
    for row in range(count):
        plus = point.copy()
        minus = point.copy()
        plus[row] += steps[row]
        minus[row] -= steps[row]
        hessian[row, row] = (
            float(function(plus)) - 2.0 * centre + float(function(minus))
        ) / (steps[row] ** 2)
        for column in range(row):
            plus_plus = point.copy()
            plus_minus = point.copy()
            minus_plus = point.copy()
            minus_minus = point.copy()
            plus_plus[[row, column]] += [steps[row], steps[column]]
            plus_minus[row] += steps[row]
            plus_minus[column] -= steps[column]
            minus_plus[row] -= steps[row]
            minus_plus[column] += steps[column]
            minus_minus[[row, column]] -= [steps[row], steps[column]]
            value = (
                float(function(plus_plus))
                - float(function(plus_minus))
                - float(function(minus_plus))
                + float(function(minus_minus))
            ) / (4.0 * steps[row] * steps[column])
            hessian[row, column] = value
            hessian[column, row] = value
    return 0.5 * (hessian + hessian.T)


def _fit_core(
    prepared: _PreparedData,
    *,
    instance_field: str | None,
    optimizer_maximum_iterations: int,
    optimizer_gradient_tolerance: float,
    optimizer_function_tolerance: float,
    inner_tolerance: float,
    inner_maximum_iterations: int,
    hessian_relative_step: float,
    sd_lower_bound: float,
    sd_upper_bound: float,
    likelihood_quadrature_order: int,
    likelihood_validation_order: int,
) -> tuple[_CoreFit | None, dict[str, Any]]:
    objective = _quadrature_objective_factory(
        prepared,
        instance_field=instance_field,
        quadrature_order=likelihood_quadrature_order,
    )
    fixed_initial = _initial_fixed_effects(prepared.y, prepared.x)
    variance_count = 1 + int(instance_field is not None)
    theta_initial = np.concatenate(
        (fixed_initial, np.full(variance_count, math.log(0.5)))
    )
    fixed_count = prepared.x.shape[1]
    bounds = [(None, None)] * fixed_count + [
        (math.log(sd_lower_bound), math.log(sd_upper_bound))
    ] * variance_count
    result = minimize(
        objective,
        theta_initial,
        method="L-BFGS-B",
        jac="3-point",
        bounds=bounds,
        options={
            "maxiter": optimizer_maximum_iterations,
            "gtol": optimizer_gradient_tolerance,
            "ftol": optimizer_function_tolerance,
            "maxls": 40,
            "finite_diff_rel_step": 1e-5,
        },
    )
    gradient_maximum = (
        float(np.max(np.abs(result.jac)))
        if result.jac is not None and np.all(np.isfinite(result.jac))
        else math.inf
    )
    convergence = {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "iterations": int(result.nit),
        "function_evaluations": int(result.nfev),
        "maximum_absolute_gradient": gradient_maximum,
        "required_maximum_absolute_gradient": optimizer_gradient_tolerance * 10.0,
    }
    if (
        not result.success
        or not np.all(np.isfinite(result.x))
        or not math.isfinite(float(result.fun))
        or gradient_maximum > optimizer_gradient_tolerance * 10.0
    ):
        return None, convergence
    evaluated = float(objective(np.asarray(result.x)))
    if not math.isfinite(evaluated):
        convergence["success"] = False
        convergence["message"] = "marginal likelihood failed at the reported optimum"
        return None, convergence
    try:
        hessian = _numerical_hessian(
            objective, np.asarray(result.x), relative_step=hessian_relative_step
        )
        eigenvalues = np.linalg.eigvalsh(hessian)
        covariance = np.linalg.inv(hessian)
    except (ValueError, np.linalg.LinAlgError):
        convergence["success"] = False
        convergence["message"] = "observed Hessian could not be inverted"
        return None, convergence
    convergence["observed_hessian_minimum_eigenvalue"] = float(np.min(eigenvalues))
    convergence["observed_hessian_condition_number"] = float(np.linalg.cond(hessian))
    if (
        not np.all(np.isfinite(covariance))
        or float(np.min(eigenvalues)) <= 1e-7
        or float(np.linalg.cond(hessian)) >= 1e10
    ):
        convergence["success"] = False
        convergence["message"] = "observed Hessian is non-positive or ill-conditioned"
        return None, convergence
    validation_objective_function = _quadrature_objective_factory(
        prepared,
        instance_field=instance_field,
        quadrature_order=likelihood_validation_order,
    )
    validation_objective = float(validation_objective_function(np.asarray(result.x)))
    quadrature_difference = abs(validation_objective - evaluated)
    convergence["quadrature_validation_absolute_log_likelihood_difference"] = (
        quadrature_difference
    )
    convergence["quadrature_validation_difference_per_episode"] = (
        quadrature_difference / len(prepared.y)
    )
    if (
        not math.isfinite(validation_objective)
        or quadrature_difference / len(prepared.y) > 1e-6
    ):
        convergence["success"] = False
        convergence["message"] = (
            "marginal likelihood is not stable at the validation quadrature order"
        )
        return None, convergence

    # Conditional modes are diagnostic summaries only; they are not used in
    # the quadrature likelihood or inferential covariance.
    mode_evaluation = _conditional_modes(
        prepared,
        np.asarray(result.x),
        instance_field=instance_field,
        inner_tolerance=inner_tolerance,
        inner_maximum_iterations=inner_maximum_iterations,
    )
    if mode_evaluation is None:
        convergence["success"] = False
        convergence["message"] = "conditional random-effect modes failed"
        return None, convergence
    return (
        _CoreFit(
            theta=np.asarray(result.x, dtype=np.float64),
            covariance=covariance,
            objective=evaluated,
            blocks=mode_evaluation,
            optimizer=result,
            hessian=hessian,
            validation_objective=validation_objective,
        ),
        convergence,
    )


def _normal_interval(estimate: float, standard_error: float, confidence: float) -> list[float]:
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    return [estimate - z * standard_error, estimate + z * standard_error]


def _two_sided_normal_p(z_value: float) -> float:
    return min(1.0, 2.0 * NormalDist().cdf(-abs(z_value)))


def _marginal_probability(
    fixed_design: np.ndarray,
    theta: np.ndarray,
    *,
    fixed_count: int,
    quadrature_order: int,
) -> float:
    beta = theta[:fixed_count]
    variance = float(np.sum(np.exp(2.0 * theta[fixed_count:])))
    nodes, weights = np.polynomial.hermite.hermgauss(quadrature_order)
    eta = fixed_design @ beta
    probabilities = expit(
        eta[:, None] + math.sqrt(2.0 * variance) * nodes[None, :]
    )
    integrated = probabilities @ weights / math.sqrt(math.pi)
    return float(np.mean(integrated))


def _counterfactual_design(
    prepared: _PreparedData,
    system: str,
) -> np.ndarray:
    design = prepared.x.copy()
    first_factor_column = len(prepared.system_levels)
    design[:, 1:first_factor_column] = 0.0
    if system != prepared.system_levels[0]:
        column = 1 + prepared.system_levels[1:].index(system)
        design[:, column] = 1.0
    return design


def _numerical_gradient(function, point: np.ndarray, *, relative_step: float = 1e-5) -> np.ndarray:
    result = np.empty(len(point), dtype=np.float64)
    for index in range(len(point)):
        step = relative_step * max(1.0, abs(float(point[index])))
        plus = point.copy()
        minus = point.copy()
        plus[index] += step
        minus[index] -= step
        result[index] = (float(function(plus)) - float(function(minus))) / (2.0 * step)
    return result


def _delta_standard_error(gradient: np.ndarray, covariance: np.ndarray) -> float:
    variance = float(gradient @ covariance @ gradient)
    if variance < -1e-10 or not math.isfinite(variance):
        raise MixedEffectsError("delta-method variance is invalid")
    return math.sqrt(max(0.0, variance))


def _diagnostic_plot_data(
    prepared: _PreparedData,
    fit: _CoreFit,
    *,
    fixed_count: int,
    instance_field: str | None,
) -> tuple[dict[str, Any], list[float], list[float]]:
    beta = fit.theta[:fixed_count]
    fitted = np.empty(len(prepared.y), dtype=np.float64)
    template_modes: list[tuple[str, float]] = []
    instance_modes: list[tuple[str, float]] = []
    for block, solution in zip(prepared.blocks, fit.blocks, strict=True):
        b = solution[0]
        local_eta = (
            prepared.x[block.observation_indices] @ beta + block.membership @ b
        )
        fitted[block.observation_indices] = expit(local_eta)
        template_modes.append((block.template, float(b[0])))
        if instance_field is not None:
            instance_modes.extend(
                (f"{block.template}:{level}", float(b[index + 1]))
                for index, level in enumerate(block.instance_levels)
            )
    denominator = np.sqrt(np.maximum(1e-12, fitted * (1.0 - fitted)))
    residuals = (prepared.y - fitted) / denominator
    ordered = np.argsort(fitted, kind="stable")
    bins = []
    for number, indices in enumerate(np.array_split(ordered, min(10, len(ordered))), start=1):
        if len(indices):
            bins.append(
                {
                    "bin": number,
                    "episodes": int(len(indices)),
                    "mean_fitted_probability": float(np.mean(fitted[indices])),
                    "mean_pearson_residual": float(np.mean(residuals[indices])),
                    "observed_success_rate": float(np.mean(prepared.y[indices])),
                }
            )

    def qq(values: list[tuple[str, float]], sd: float) -> list[dict[str, Any]]:
        ordered_values = sorted(values, key=lambda item: (item[1], item[0]))
        count = len(ordered_values)
        return [
            {
                "level": level,
                "standardized_conditional_mode": value / sd,
                "theoretical_normal_quantile": NormalDist().inv_cdf(
                    (rank - 0.375) / (count + 0.25)
                ),
            }
            for rank, (level, value) in enumerate(ordered_values, start=1)
        ]

    standard_deviations = np.exp(fit.theta[fixed_count:])
    plots: dict[str, Any] = {
        "residual_vs_fitted_and_calibration": {
            "rendering": (
                "plot mean_pearson_residual and observed_success_rate against "
                "mean_fitted_probability"
            ),
            "binning": (
                "stable sort by fitted probability, then at most 10 equal-count "
                "numpy.array_split bins"
            ),
            "bins": bins,
        },
        "template_random_intercept_qq": {
            "rendering": "plot standardized_conditional_mode against theoretical_normal_quantile",
            "quantile_rule": "Normal inverse CDF at (rank - 0.375) / (groups + 0.25)",
            "points": qq(template_modes, float(standard_deviations[0])),
        },
    }
    if instance_field is not None:
        plots["nested_instance_random_intercept_qq"] = {
            "rendering": "plot standardized_conditional_mode against theoretical_normal_quantile",
            "quantile_rule": "Normal inverse CDF at (rank - 0.375) / (groups + 0.25)",
            "points": qq(instance_modes, float(standard_deviations[1])),
        }
    return plots, fitted.tolist(), residuals.tolist()


def fit_binary_mixed_effects(
    rows: Iterable[Mapping[str, Any]],
    *,
    outcome: str = "contract_success",
    system_field: str = "profile_id",
    reference_system: str,
    scenario_factors: Sequence[str] = (),
    factor_references: Mapping[str, Any] | None = None,
    template_field: str = "scenario_template",
    instance_field: str | None = "generated_instance",
    confidence: float = 0.95,
    minimum_random_groups: int = 5,
    singular_sd_threshold: float = 0.02,
    likelihood_quadrature_order: int = 20,
    likelihood_validation_order: int = 30,
    marginal_quadrature_order: int = 30,
    optimizer_maximum_iterations: int = 500,
    optimizer_gradient_tolerance: float = 1e-4,
    optimizer_function_tolerance: float = 1e-10,
    inner_tolerance: float = 1e-8,
    inner_maximum_iterations: int = 100,
    hessian_relative_step: float = 2e-4,
    sd_lower_bound: float = 1e-4,
    sd_upper_bound: float = 5.0,
) -> dict[str, Any]:
    """Fit the predeclared binary GLMM and return a complete audit record.

    ``scenario_factors`` are categorical main effects.  Every corresponding
    reference level must be supplied in ``factor_references``; this prevents a
    changed data ordering from silently changing the estimand.
    """

    factors = tuple(scenario_factors)
    references = dict(factor_references or {})
    if not outcome or not system_field or not template_field:
        raise MixedEffectsError("outcome and model field names must be non-empty")
    if not 0.0 < confidence < 1.0:
        raise MixedEffectsError("confidence must lie strictly between zero and one")
    if (
        isinstance(minimum_random_groups, bool)
        or not isinstance(minimum_random_groups, int)
        or minimum_random_groups < 3
    ):
        raise MixedEffectsError("minimum_random_groups must be an integer >= 3")
    if not 0.0 < singular_sd_threshold < sd_upper_bound:
        raise MixedEffectsError(
            "singular_sd_threshold must lie between zero and the upper SD bound"
        )
    for label, order in (
        ("likelihood_quadrature_order", likelihood_quadrature_order),
        ("likelihood_validation_order", likelihood_validation_order),
        ("marginal_quadrature_order", marginal_quadrature_order),
    ):
        if isinstance(order, bool) or not isinstance(order, int) or order < 10:
            raise MixedEffectsError(f"{label} must be an integer >= 10")
    if likelihood_validation_order <= likelihood_quadrature_order:
        raise MixedEffectsError(
            "likelihood_validation_order must exceed likelihood_quadrature_order"
        )
    if (
        isinstance(optimizer_maximum_iterations, bool)
        or not isinstance(optimizer_maximum_iterations, int)
        or isinstance(inner_maximum_iterations, bool)
        or not isinstance(inner_maximum_iterations, int)
        or optimizer_maximum_iterations < 1
        or inner_maximum_iterations < 1
    ):
        raise MixedEffectsError("optimizer iteration limits must be positive")
    if (
        optimizer_gradient_tolerance <= 0
        or optimizer_function_tolerance <= 0
        or inner_tolerance <= 0
        or hessian_relative_step <= 0
    ):
        raise MixedEffectsError("optimizer and finite-difference tolerances must be positive")
    if not 0.0 < sd_lower_bound < sd_upper_bound:
        raise MixedEffectsError("random-effect SD bounds are invalid")
    if set(references) != set(factors):
        raise MixedEffectsError(
            "factor_references must provide exactly one explicit reference "
            "for every scenario factor"
        )

    specification = _model_specification(
        outcome=outcome,
        system_field=system_field,
        reference_system=str(reference_system),
        scenario_factors=factors,
        factor_references={key: str(value) for key, value in references.items()},
        template_field=template_field,
        instance_field=instance_field,
        confidence=confidence,
        likelihood_quadrature_order=likelihood_quadrature_order,
        likelihood_validation_order=likelihood_validation_order,
        marginal_quadrature_order=marginal_quadrature_order,
        singular_sd_threshold=singular_sd_threshold,
    )
    prepared, early = _prepare_data(
        rows,
        outcome=outcome,
        system_field=system_field,
        reference_system=reference_system,
        scenario_factors=factors,
        factor_references=references,
        template_field=template_field,
        instance_field=instance_field,
        minimum_random_groups=minimum_random_groups,
    )
    if early is not None:
        early.setdefault("model_specification", specification)
        return early
    assert prepared is not None
    specification["fixed_design_columns"] = list(prepared.fixed_names)
    specification["categorical_levels_reference_first"] = {
        system_field: list(prepared.system_levels),
        **{
            factor: list(prepared.factor_levels[factor])
            for factor in factors
        },
    }
    specification["coding_rule"] = (
        "Intercept is one; each non-reference level owns one 0/1 column; "
        "the reference level is the all-zero vector for that factor."
    )
    separation = _separation_diagnostic(prepared.y, prepared.x)
    common = {
        "input_sha256": prepared.input_sha256,
        "model_specification": specification,
        "software": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "input_summary": {
            "episodes": len(prepared.y),
            "successes": int(np.sum(prepared.y)),
            "failures": int(len(prepared.y) - np.sum(prepared.y)),
            "system_counts": dict(
                sorted(Counter(record["system"] for record in prepared.rows).items())
            ),
            "template_groups": len(prepared.blocks),
            "nested_instance_groups": int(
                sum(len(block.instance_levels) for block in prepared.blocks)
            )
            if instance_field is not None
            else None,
            "matched_group_counts_by_contrast": prepared.overlap_counts,
        },
    }
    if separation["detected"]:
        return _not_estimable(
            "complete or quasi-complete fixed-effect separation was detected",
            diagnostics={"separation": separation},
            **common,
        )

    fit, convergence = _fit_core(
        prepared,
        instance_field=instance_field,
        optimizer_maximum_iterations=optimizer_maximum_iterations,
        optimizer_gradient_tolerance=optimizer_gradient_tolerance,
        optimizer_function_tolerance=optimizer_function_tolerance,
        inner_tolerance=inner_tolerance,
        inner_maximum_iterations=inner_maximum_iterations,
        hessian_relative_step=hessian_relative_step,
        sd_lower_bound=sd_lower_bound,
        sd_upper_bound=sd_upper_bound,
        likelihood_quadrature_order=likelihood_quadrature_order,
        likelihood_validation_order=likelihood_validation_order,
    )
    optimizer_record = {
        "outer": "scipy.optimize.minimize(method='L-BFGS-B', jac='3-point')",
        "objective": "nested Gauss-Hermite marginal Bernoulli negative log likelihood",
        "likelihood_quadrature_order_per_random_effect": likelihood_quadrature_order,
        "likelihood_validation_order_per_random_effect": likelihood_validation_order,
        "outer_maximum_iterations": optimizer_maximum_iterations,
        "outer_gradient_tolerance": optimizer_gradient_tolerance,
        "outer_function_tolerance": optimizer_function_tolerance,
        "finite_difference_relative_step": 1e-5,
        "random_mode_solver": "damped Newton with Armijo backtracking",
        "random_mode_gradient_tolerance": inner_tolerance,
        "random_mode_maximum_iterations": inner_maximum_iterations,
        "observed_hessian": (
            "central finite differences of the nested-quadrature marginal "
            "negative log likelihood"
        ),
        "observed_hessian_relative_step": hessian_relative_step,
        "random_effect_sd_bounds": [sd_lower_bound, sd_upper_bound],
        "convergence": convergence,
    }
    if fit is None:
        return _not_estimable(
            "mixed-effects optimizer or observed Hessian did not converge validly",
            optimizer=optimizer_record,
            diagnostics={"separation": separation},
            **common,
        )

    fixed_count = prepared.x.shape[1]
    sd_values = np.exp(fit.theta[fixed_count:])
    random_names = [template_field]
    if instance_field is not None:
        random_names.append(f"{template_field}:{instance_field}")
    random_effects = [
        {
            "term": name,
            "standard_deviation": float(sd),
            "variance": float(sd * sd),
            "singular_threshold": singular_sd_threshold,
        }
        for name, sd in zip(random_names, sd_values, strict=True)
    ]
    near_lower = any(sd <= singular_sd_threshold for sd in sd_values)
    at_bound = any(
        sd <= sd_lower_bound * 1.01 or sd >= sd_upper_bound / 1.01
        for sd in sd_values
    )
    if near_lower or at_bound:
        return _not_estimable(
            "random-effect variance is singular or at an optimizer bound",
            optimizer=optimizer_record,
            random_effects=random_effects,
            diagnostics={
                "separation": separation,
                "singular": near_lower,
                "variance_parameter_at_bound": at_bound,
            },
            **common,
        )

    standard_errors = np.sqrt(np.diag(fit.covariance))
    z_critical = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    fixed_effects = []
    for index, name in enumerate(prepared.fixed_names):
        estimate = float(fit.theta[index])
        standard_error = float(standard_errors[index])
        z_value = estimate / standard_error
        fixed_effects.append(
            {
                "term": name,
                "estimate_log_odds": estimate,
                "standard_error": standard_error,
                "confidence_interval_log_odds": _normal_interval(
                    estimate, standard_error, confidence
                ),
                "wald_z": z_value,
                "wald_p_two_sided": _two_sided_normal_p(z_value),
                "inference_label": (
                    "asymptotic Wald inference under the declared "
                    "nested-quadrature GLMM"
                ),
            }
        )

    designs = {
        system: _counterfactual_design(prepared, system)
        for system in prepared.system_levels
    }
    marginal: dict[str, dict[str, Any]] = {}
    marginal_functions = {}
    for system, design in designs.items():
        function = lambda theta, matrix=design: _marginal_probability(
            matrix,
            theta,
            fixed_count=fixed_count,
            quadrature_order=marginal_quadrature_order,
        )
        marginal_functions[system] = function
        estimate = function(fit.theta)
        gradient = _numerical_gradient(function, fit.theta)
        standard_error = _delta_standard_error(gradient, fit.covariance)
        interval = _normal_interval(estimate, standard_error, confidence)
        marginal[system] = {
            "system": system,
            "probability": estimate,
            "standard_error": standard_error,
            "confidence_interval": [max(0.0, interval[0]), min(1.0, interval[1])],
        }

    reference = prepared.system_levels[0]
    contrasts = []
    for system_index, system in enumerate(prepared.system_levels[1:], start=1):
        reference_function = marginal_functions[reference]
        system_function = marginal_functions[system]

        def difference_function(theta):
            return system_function(theta) - reference_function(theta)

        difference = difference_function(fit.theta)
        difference_gradient = _numerical_gradient(difference_function, fit.theta)
        difference_se = _delta_standard_error(difference_gradient, fit.covariance)
        difference_interval = _normal_interval(difference, difference_se, confidence)

        def log_risk_ratio_function(theta):
            return math.log(system_function(theta) / reference_function(theta))

        log_rr = log_risk_ratio_function(fit.theta)
        log_rr_gradient = _numerical_gradient(log_risk_ratio_function, fit.theta)
        log_rr_se = _delta_standard_error(log_rr_gradient, fit.covariance)
        log_rr_interval = _normal_interval(log_rr, log_rr_se, confidence)

        coefficient = float(fit.theta[system_index])
        coefficient_se = float(standard_errors[system_index])
        coefficient_interval = _normal_interval(coefficient, coefficient_se, confidence)
        contrasts.append(
            {
                "comparison": f"{system} - {reference}",
                "system": system,
                "reference_system": reference,
                "marginal_absolute_difference": difference,
                "marginal_absolute_difference_standard_error": difference_se,
                "marginal_absolute_difference_confidence_interval": difference_interval,
                "marginal_relative_risk": math.exp(log_rr),
                "marginal_relative_risk_confidence_interval": [
                    math.exp(log_rr_interval[0]),
                    math.exp(log_rr_interval[1]),
                ],
                "conditional_odds_ratio": math.exp(coefficient),
                "conditional_odds_ratio_confidence_interval": [
                    math.exp(coefficient_interval[0]),
                    math.exp(coefficient_interval[1]),
                ],
                "wald_z": coefficient / coefficient_se,
                "wald_p_two_sided": _two_sided_normal_p(coefficient / coefficient_se),
                "multiplicity_adjustment": "none; apply the preregistered Holm family externally",
            }
        )

    plots, fitted_probabilities, residuals = _diagnostic_plot_data(
        prepared,
        fit,
        fixed_count=fixed_count,
        instance_field=instance_field,
    )
    condition = float(np.linalg.cond(prepared.x))
    diagnostics = {
        "separation": separation,
        "fixed_design_rank": int(np.linalg.matrix_rank(prepared.x)),
        "fixed_design_columns": int(prepared.x.shape[1]),
        "fixed_design_condition_number": condition,
        "pearson_residual_summary": {
            "mean": float(np.mean(residuals)),
            "standard_deviation": float(np.std(residuals, ddof=1)),
            "maximum_absolute": float(np.max(np.abs(residuals))),
        },
        "conditional_fitted_probability_range": [
            float(np.min(fitted_probabilities)),
            float(np.max(fitted_probabilities)),
        ],
        "diagnostic_plot_data": plots,
        "singular": False,
        "variance_parameter_at_bound": False,
    }

    per_template_raw = []
    for template in sorted({record["template"] for record in prepared.rows}):
        template_rows = [
            record for record in prepared.rows if record["template"] == template
        ]
        systems = []
        for system in prepared.system_levels:
            outcomes = [
                record["outcome"]
                for record in template_rows
                if record["system"] == system
            ]
            systems.append(
                {
                    "system": system,
                    "episodes": len(outcomes),
                    "successes": int(sum(outcomes)),
                    "raw_success_rate": float(np.mean(outcomes)) if outcomes else None,
                }
            )
        per_template_raw.append(
            {
                "scenario_template": template,
                "scenario_factor_levels": {
                    factor: sorted(
                        {record["factors"][factor] for record in template_rows}
                    )
                    for factor in factors
                },
                "systems": systems,
            }
        )

    parameter_names = [*prepared.fixed_names, *[f"log_sd[{name}]" for name in random_names]]
    return {
        "schema_version": SCHEMA_VERSION,
        "estimable": True,
        "estimate": contrasts[0]["marginal_absolute_difference"]
        if len(contrasts) == 1
        else None,
        **common,
        "optimizer": optimizer_record,
        "fit": {
            "marginal_log_likelihood": -fit.objective,
            "validation_order_marginal_log_likelihood": -fit.validation_objective,
            "parameter_count": len(fit.theta),
            "aic": 2.0 * (len(fit.theta) + fit.objective),
            "parameter_names": parameter_names,
            "parameter_vector": fit.theta.tolist(),
            "parameter_covariance": fit.covariance.tolist(),
        },
        "fixed_effects": fixed_effects,
        "random_effects": random_effects,
        "estimated_marginal_means": [marginal[system] for system in prepared.system_levels],
        "system_contrasts": contrasts,
        "per_template_raw_outcomes": per_template_raw,
        "diagnostics": diagnostics,
        "raw_regeneration": {
            "canonical_input_sha256": prepared.input_sha256,
            "ordering": "canonical JSON of used fields, lexicographically sorted",
            "deterministic": True,
            "command_contract": (
                "rerun fit_binary_mixed_effects with the identical raw rows "
                "and declared keyword arguments"
            ),
        },
        "methodological_limits": [
            (
                "The likelihood uses finite-order nested Gauss-Hermite quadrature; "
                "the separately reported higher-order evaluation is its numerical "
                "stability check."
            ),
            (
                "Wald and delta-method intervals are asymptotic; small-sample "
                "bootstrap calibration is not claimed."
            ),
            (
                "Generated instances are modeled as nested within scenario template; "
                "crossed prompt, participant, or backbone effects require a separately "
                "implemented model."
            ),
            (
                "Multiplicity adjustment is intentionally external because "
                "hypothesis-family membership is protocol-level metadata."
            ),
        ],
    }


def _derive_seed(seed: int, *parts: Any) -> int:
    material = json.dumps(
        ["robopref-mixed-power", seed, *parts],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:16], "big")


def _planned_patterns(
    prepared: _PreparedData,
    *,
    reference_system: str,
    comparison_system: str,
    scenario_factors: Sequence[str],
) -> tuple[dict[str, Any], ...]:
    by_pattern: dict[tuple[Any, ...], set[str]] = defaultdict(set)
    factors_by_pattern: dict[tuple[Any, ...], dict[str, str]] = {}
    for record in prepared.rows:
        if record["system"] not in {reference_system, comparison_system}:
            continue
        key = (
            record["template"],
            *(record["factors"][factor] for factor in scenario_factors),
        )
        by_pattern[key].add(record["system"])
        factors_by_pattern[key] = dict(record["factors"])
    expected = {reference_system, comparison_system}
    incomplete = [key for key, systems in by_pattern.items() if systems != expected]
    if incomplete:
        raise MixedEffectsError(
            "pilot rows do not define a fully matched two-system planned design"
        )
    return tuple(
        {
            "pattern_id": f"pattern-{index:04d}",
            "template": key[0],
            "factors": factors_by_pattern[key],
        }
        for index, key in enumerate(sorted(by_pattern), start=1)
    )


def simulate_binary_mixed_effects_power(
    pilot_rows: Iterable[Mapping[str, Any]],
    *,
    comparison_system: str,
    reference_system: str,
    minimum_absolute_difference: float,
    outcome: str = "contract_success",
    system_field: str = "profile_id",
    scenario_factors: Sequence[str] = (),
    factor_references: Mapping[str, Any] | None = None,
    template_field: str = "scenario_template",
    instance_field: str = "generated_instance",
    repetition_candidates: Sequence[int] = DEFAULT_REPETITION_CANDIDATES,
    simulations: int = 1000,
    alpha: float = 0.05,
    target_power: float = 0.9,
    seed: int = 0,
    fit_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Simulate power for a prespecified two-system GLMM contrast.

    The signed ``minimum_absolute_difference`` is imposed on the marginal
    success probability of ``comparison_system - reference_system``.  Pilot
    estimates provide nuisance fixed effects and random-effect variances only.
    Non-estimable simulated fits count as non-detections.
    """

    if isinstance(simulations, bool) or not isinstance(simulations, int) or simulations < 2:
        raise MixedEffectsError("simulations must be an integer >= 2")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise MixedEffectsError("seed must be an integer")
    if not 0.0 < alpha < 1.0 or not 0.0 < target_power <= 1.0:
        raise MixedEffectsError("alpha and target_power are outside their valid ranges")
    if not -0.5 <= minimum_absolute_difference <= 0.5 or minimum_absolute_difference == 0:
        raise MixedEffectsError(
            "minimum_absolute_difference must be signed, nonzero, and within [-0.5, 0.5]"
        )
    candidates = tuple(repetition_candidates)
    if not candidates or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 2
        for value in candidates
    ):
        raise MixedEffectsError("repetition candidates must be integers >= 2")
    if len(set(candidates)) != len(candidates):
        raise MixedEffectsError("repetition candidates must be unique")
    publication_candidate_set = candidates == DEFAULT_REPETITION_CANDIDATES
    if simulations >= 1000 and not publication_candidate_set:
        raise MixedEffectsError(
            "publication power runs must use repetition candidates 10, 15, 20, 25 "
            "in that order"
        )
    factors = tuple(scenario_factors)
    references = dict(factor_references or {})
    rows = tuple(dict(row) for row in pilot_rows)
    options = dict(fit_options or {})
    forbidden = {
        "rows",
        "outcome",
        "system_field",
        "reference_system",
        "scenario_factors",
        "factor_references",
        "template_field",
        "instance_field",
    }.intersection(options)
    if forbidden:
        raise MixedEffectsError(f"fit_options cannot replace model identity: {sorted(forbidden)}")
    pilot = fit_binary_mixed_effects(
        rows,
        outcome=outcome,
        system_field=system_field,
        reference_system=reference_system,
        scenario_factors=factors,
        factor_references=references,
        template_field=template_field,
        instance_field=instance_field,
        **options,
    )
    common = {
        "schema_version": POWER_SCHEMA_VERSION,
        "estimable": False,
        "estimate": None,
        "comparison": f"{comparison_system} - {reference_system}",
        "minimum_absolute_difference": minimum_absolute_difference,
        "alpha": alpha,
        "target_power": target_power,
        "simulations": simulations,
        "seed": seed,
        "repetition_unit": "matched generated instances per scenario-template pattern",
        "candidate_repetitions": list(candidates),
    }
    if not pilot.get("estimable"):
        return {
            **common,
            "reason": "pilot nuisance model is not estimable",
            "pilot_fit": pilot,
            "power_curve": [],
            "selected_repetitions": None,
        }
    prepared, early = _prepare_data(
        rows,
        outcome=outcome,
        system_field=system_field,
        reference_system=reference_system,
        scenario_factors=factors,
        factor_references=references,
        template_field=template_field,
        instance_field=instance_field,
        minimum_random_groups=int(options.get("minimum_random_groups", 5)),
    )
    if early is not None or prepared is None:
        raise MixedEffectsError("pilot preprocessing changed after an estimable fit")
    if comparison_system not in prepared.system_levels:
        raise MixedEffectsError("comparison_system is absent from pilot rows")
    if set(prepared.system_levels) != {reference_system, comparison_system}:
        raise MixedEffectsError("power simulation requires pilot rows for exactly two systems")
    patterns = _planned_patterns(
        prepared,
        reference_system=reference_system,
        comparison_system=comparison_system,
        scenario_factors=factors,
    )
    parameter_names = pilot["fit"]["parameter_names"]
    theta = np.asarray(pilot["fit"]["parameter_vector"], dtype=np.float64)
    fixed_count = len(pilot["fixed_effects"])
    comparison_parameter = f"{system_field}[{comparison_system}]"
    comparison_index = parameter_names.index(comparison_parameter)

    # Build one fixed-design row per planned pattern and condition using the
    # exact treatment coding from the fitted pilot.
    synthetic_base = []
    for pattern in patterns:
        for system in (reference_system, comparison_system):
            synthetic_base.append(
                {
                    outcome: 0,
                    system_field: system,
                    template_field: pattern["template"],
                    instance_field: "calibration-instance",
                    **pattern["factors"],
                }
            )
    design_prepared, design_error = _prepare_data(
        synthetic_base * 5,
        outcome=outcome,
        system_field=system_field,
        reference_system=reference_system,
        scenario_factors=factors,
        factor_references=references,
        template_field=template_field,
        instance_field=None,
        minimum_random_groups=3,
    )
    if design_error is not None or design_prepared is None:
        # The synthetic duplication above only supplies coding; reaching this
        # branch means the planned factor matrix is structurally invalid.
        return {
            **common,
            "reason": "planned fixed-effect design is not identifiable",
            "pilot_fit": pilot,
            "power_curve": [],
            "selected_repetitions": None,
        }
    reference_design = _counterfactual_design(design_prepared, reference_system)
    comparison_design = _counterfactual_design(design_prepared, comparison_system)

    def imposed_difference(coefficient: float) -> float:
        candidate_theta = theta.copy()
        candidate_theta[comparison_index] = coefficient
        comparison_probability = _marginal_probability(
            comparison_design,
            candidate_theta,
            fixed_count=fixed_count,
            quadrature_order=30,
        )
        reference_probability = _marginal_probability(
            reference_design,
            candidate_theta,
            fixed_count=fixed_count,
            quadrature_order=30,
        )
        return comparison_probability - reference_probability

    lower_value = imposed_difference(-15.0)
    upper_value = imposed_difference(15.0)
    if not lower_value < minimum_absolute_difference < upper_value:
        return {
            **common,
            "reason": (
                "the requested marginal minimum effect is unattainable under "
                "the pilot nuisance model"
            ),
            "attainable_difference_range": [lower_value, upper_value],
            "pilot_fit": pilot,
            "power_curve": [],
            "selected_repetitions": None,
        }
    calibrated = root_scalar(
        lambda coefficient: imposed_difference(coefficient) - minimum_absolute_difference,
        bracket=(-15.0, 15.0),
        method="brentq",
        xtol=1e-12,
    )
    if not calibrated.converged:
        return {
            **common,
            "reason": "minimum-effect calibration did not converge",
            "pilot_fit": pilot,
            "power_curve": [],
            "selected_repetitions": None,
        }
    alternative_theta = theta.copy()
    alternative_theta[comparison_index] = float(calibrated.root)
    random_sds = np.exp(alternative_theta[fixed_count:])

    # Create fixed vectors directly from declared level positions.
    system_columns = {level: index + 1 for index, level in enumerate(prepared.system_levels[1:])}
    factor_columns: dict[tuple[str, str], int] = {}
    column = len(prepared.system_levels)
    for factor in factors:
        for level in prepared.factor_levels[factor][1:]:
            factor_columns[(factor, level)] = column
            column += 1

    def vector(system: str, factor_values: Mapping[str, str]) -> np.ndarray:
        result = np.zeros(fixed_count, dtype=np.float64)
        result[0] = 1.0
        if system != reference_system:
            result[system_columns[system]] = 1.0
        for factor, value in factor_values.items():
            if value != prepared.factor_references[factor]:
                result[factor_columns[(factor, value)]] = 1.0
        return result

    curve = []
    direction = 1.0 if minimum_absolute_difference > 0 else -1.0
    for repetitions in candidates:
        generator = np.random.Generator(
            np.random.PCG64(_derive_seed(seed, repetitions))
        )
        detected = 0
        nonestimable = 0
        nonestimable_reasons: Counter[str] = Counter()
        wrong_direction = 0
        for simulation in range(simulations):
            templates = tuple(sorted({pattern["template"] for pattern in patterns}))
            template_effects = {
                template: float(generator.normal(0.0, random_sds[0]))
                for template in templates
            }
            instance_effects = {
                (pattern["pattern_id"], repetition): float(
                    generator.normal(0.0, random_sds[1])
                )
                for pattern in patterns
                for repetition in range(repetitions)
            }
            simulated_rows = []
            for pattern in patterns:
                for repetition in range(repetitions):
                    for system in (reference_system, comparison_system):
                        eta = float(
                            vector(system, pattern["factors"]) @ alternative_theta[:fixed_count]
                            + template_effects[pattern["template"]]
                            + instance_effects[(pattern["pattern_id"], repetition)]
                        )
                        success = bool(generator.random() < expit(eta))
                        simulated_rows.append(
                            {
                                outcome: success,
                                system_field: system,
                                template_field: pattern["template"],
                                instance_field: (
                                    f"{pattern['pattern_id']}-power-{repetition:04d}"
                                ),
                                **pattern["factors"],
                            }
                        )
            fitted = fit_binary_mixed_effects(
                simulated_rows,
                outcome=outcome,
                system_field=system_field,
                reference_system=reference_system,
                scenario_factors=factors,
                factor_references=references,
                template_field=template_field,
                instance_field=instance_field,
                **options,
            )
            if not fitted.get("estimable"):
                nonestimable += 1
                nonestimable_reasons[str(fitted.get("reason", "unspecified"))] += 1
                continue
            contrast = next(
                item
                for item in fitted["system_contrasts"]
                if item["system"] == comparison_system
            )
            if direction * float(contrast["marginal_absolute_difference"]) <= 0:
                wrong_direction += 1
                continue
            detected += int(float(contrast["wald_p_two_sided"]) < alpha)
        power = detected / simulations
        mc_interval = wilson_interval(detected, simulations)
        publication_ready = simulations >= 1000 and publication_candidate_set
        curve.append(
            {
                "repetitions": repetitions,
                "planned_episodes": len(patterns) * 2 * repetitions,
                "power": power,
                "monte_carlo_wilson_interval": list(mc_interval),
                "detections": detected,
                "nonestimable_fits_counted_as_nondetections": nonestimable,
                "nonestimable_reason_counts": dict(sorted(nonestimable_reasons.items())),
                "wrong_direction_fits_counted_as_nondetections": wrong_direction,
                "simulations": simulations,
                "publication_ready_monte_carlo_size": publication_ready,
                "meets_target_point_estimate": power >= target_power,
                "meets_target_conservatively": (
                    publication_ready and mc_interval[0] >= target_power
                ),
            }
        )
    eligible = [
        item["repetitions"]
        for item in curve
        if item["meets_target_conservatively"]
    ]
    return {
        **common,
        "estimable": True,
        "estimate": None,
        "pilot_input_sha256": pilot["input_sha256"],
        "pilot_nuisance_fit": {
            "parameter_names": parameter_names,
            "parameter_vector": theta.tolist(),
            "random_effects": pilot["random_effects"],
        },
        "alternative": {
            "definition": (
                "signed marginal probability difference imposed at the minimum "
                "effect worth detecting"
            ),
            "calibrated_comparison_log_odds_coefficient": float(calibrated.root),
            "achieved_marginal_absolute_difference": imposed_difference(float(calibrated.root)),
            "nuisance_parameters_source": "estimable excluded pilot GLMM",
        },
        "rng": {
            "generator": "numpy.random.PCG64",
            "stream_derivation": "SHA-256(seed, repetition candidate); candidate-order invariant",
        },
        "test": {
            "statistic": "two-sided asymptotic Wald z for the declared system coefficient",
            "direction_required": True,
            "nonestimable_policy": "count as non-detection",
            "multiplicity": "alpha must already reflect the preregistered hypothesis family",
        },
        "planned_design_patterns": len(patterns),
        "power_curve": curve,
        "selected_repetitions": min(eligible) if eligible else None,
        "selection_rule": (
            "smallest candidate whose 95% Monte Carlo Wilson lower bound is at least "
            f"{target_power:g}, with at least 1000 simulations"
        ),
        "raw_regeneration": {
            "pilot_input_sha256": pilot["input_sha256"],
            "deterministic": True,
            "candidate_order_invariant": True,
        },
        "methodological_limits": [
            (
                "Pilot point estimates determine nuisance parameters and may be "
                "imprecise; sensitivity curves over plausible nuisance values remain necessary."
            ),
            (
                "The simulator addresses one two-system binary contrast and does not "
                "implement count, time, human, or multi-backbone power."
            ),
            (
                "A curve with fewer than 1000 simulations is explicitly preliminary "
                "and cannot select locked repetitions."
            ),
            (
                "The caller must supply a multiplicity-adjusted alpha when the contrast "
                "belongs to a larger confirmatory family."
            ),
        ],
    }


__all__ = [
    "DEFAULT_REPETITION_CANDIDATES",
    "MixedEffectsError",
    "fit_binary_mixed_effects",
    "main",
    "simulate_binary_mixed_effects_power",
]


_FIT_CONFIG_KEYS = frozenset(
    {
        "outcome",
        "system_field",
        "reference_system",
        "scenario_factors",
        "factor_references",
        "template_field",
        "instance_field",
        "confidence",
        "minimum_random_groups",
        "singular_sd_threshold",
        "likelihood_quadrature_order",
        "likelihood_validation_order",
        "marginal_quadrature_order",
        "optimizer_maximum_iterations",
        "optimizer_gradient_tolerance",
        "optimizer_function_tolerance",
        "inner_tolerance",
        "inner_maximum_iterations",
        "hessian_relative_step",
        "sd_lower_bound",
        "sd_upper_bound",
    }
)
_POWER_CONFIG_KEYS = frozenset(
    {
        "comparison_system",
        "reference_system",
        "minimum_absolute_difference",
        "outcome",
        "system_field",
        "scenario_factors",
        "factor_references",
        "template_field",
        "instance_field",
        "repetition_candidates",
        "simulations",
        "alpha",
        "target_power",
        "seed",
        "fit_options",
    }
)


def _read_cli_inputs(
    rows_path: Path, config_path: Path, *, allowed_config_keys: frozenset[str]
) -> tuple[list[dict[str, Any]], dict[str, Any], str, str]:
    try:
        rows_bytes = rows_path.read_bytes()
        config_bytes = config_path.read_bytes()
        rows_value = json.loads(rows_bytes)
        config_value = json.loads(config_bytes)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise MixedEffectsError(f"cannot read CLI input JSON: {error}") from error
    if not isinstance(rows_value, list) or not all(
        isinstance(row, dict) for row in rows_value
    ):
        raise MixedEffectsError("--rows JSON must be a list of row objects")
    if not isinstance(config_value, dict):
        raise MixedEffectsError("--config JSON must be an object")
    unknown = set(config_value) - allowed_config_keys
    if unknown:
        raise MixedEffectsError(f"unknown config keys: {sorted(unknown)}")
    return (
        [dict(row) for row in rows_value],
        dict(config_value),
        hashlib.sha256(rows_bytes).hexdigest(),
        hashlib.sha256(config_bytes).hexdigest(),
    )


def _write_cli_artifact(
    output: Path,
    *,
    analysis_kind: str,
    report: Mapping[str, Any],
    rows_file_sha256: str,
    config_file_sha256: str,
) -> tuple[Path, str]:
    checksum_path = Path(f"{output}.sha256")
    if output.exists() or checksum_path.exists():
        raise MixedEffectsError("refusing to overwrite an existing analysis artifact")
    implementation_path = Path(__file__).resolve()
    envelope = {
        "artifact_schema_version": "robopref.mixed-effects-artifact.v1",
        "analysis_kind": analysis_kind,
        "rows_file_sha256": rows_file_sha256,
        "config_file_sha256": config_file_sha256,
        "implementation_source_sha256": hashlib.sha256(
            implementation_path.read_bytes()
        ).hexdigest(),
        "report": dict(report),
    }
    payload = (
        json.dumps(envelope, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        durable_write(output, payload, exclusive=True)
        durable_write(
            checksum_path,
            f"{digest}  {output.name}\n".encode("utf-8"),
            exclusive=True,
        )
    except FileExistsError as error:
        raise MixedEffectsError(
            "refusing to overwrite an existing analysis artifact"
        ) from error
    return checksum_path, digest


def _cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the deterministic RoboPref binary mixed-effects analysis"
    )
    subparsers = parser.add_subparsers(dest="analysis_kind", required=True)
    for command in ("fit", "power"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--rows", type=Path, required=True)
        subparser.add_argument("--config", type=Path, required=True)
        subparser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run a fit or power job from immutable JSON inputs."""

    arguments = _cli_parser().parse_args(argv)
    allowed = _FIT_CONFIG_KEYS if arguments.analysis_kind == "fit" else _POWER_CONFIG_KEYS
    rows, config, rows_hash, config_hash = _read_cli_inputs(
        arguments.rows, arguments.config, allowed_config_keys=allowed
    )
    if arguments.analysis_kind == "fit":
        report = fit_binary_mixed_effects(rows, **config)
    else:
        report = simulate_binary_mixed_effects_power(rows, **config)
    checksum_path, digest = _write_cli_artifact(
        arguments.output,
        analysis_kind=arguments.analysis_kind,
        report=report,
        rows_file_sha256=rows_hash,
        config_file_sha256=config_hash,
    )
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "output_sha256": digest,
                "checksum": str(checksum_path),
                "estimable": report.get("estimable") is True,
                "selected_repetitions": report.get("selected_repetitions"),
            },
            sort_keys=True,
        )
    )
    return 0 if report.get("estimable") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
