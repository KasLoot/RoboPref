from __future__ import annotations

import copy

import pytest

from experiments_suite_v2.io import load_json
from experiments_suite_v2.runners.focused_monitor_ablation_exp06 import (
    PROTOCOL_PATH,
    FocusedMonitorAblationError,
    _control_assessment,
    _task_for,
    analyze_rows,
    expand_episode_specs,
    render_report,
    run_episode,
    validate_protocol,
)


def _protocol():
    return validate_protocol(load_json(PROTOCOL_PATH))


def test_protocol_has_frozen_balanced_design() -> None:
    protocol = _protocol()
    assert protocol["design"]["episodes_total"] == 72
    assert len(protocol["scenarios"]) == 12
    counts: dict[str, int] = {}
    for scenario in protocol["scenarios"]:
        counts[scenario["family"]] = counts.get(scenario["family"], 0) + 1
    assert counts == {
        "VISUAL_DISCREPANCY": 5,
        "COMPLETED_ENDPOINT": 3,
        "VISIBLE_PROGRESS": 1,
        "NONTERMINAL_HOLD": 2,
        "TIMEOUT_CONTROL": 1,
    }


def test_episode_expansion_is_unique_and_paired() -> None:
    specs = expand_episode_specs(_protocol())
    assert len(specs) == 72
    assert len({item["episode_id"] for item in specs}) == 72
    pairs: dict[tuple[str, str], set[str]] = {}
    for spec in specs:
        pairs.setdefault((spec["scenario_id"], spec["variant_id"]), set()).add(
            spec["condition_id"]
        )
    assert len(pairs) == 36
    assert all(
        conditions
        == {"PRODUCTION_MONITOR", "NO_VISUAL_ADVANCEMENT_CONTROL"}
        for conditions in pairs.values()
    )


def test_confounded_fixture_and_motion_are_rejected() -> None:
    protocol = _protocol()
    changed = copy.deepcopy(protocol)
    changed["scenarios"][0]["frames"][0] = "V12"
    with pytest.raises(FocusedMonitorAblationError, match="V12"):
        validate_protocol(changed)
    changed = copy.deepcopy(protocol)
    changed["system_under_test"]["simulated_motion"] = 1
    with pytest.raises(FocusedMonitorAblationError, match="zero-motion"):
        validate_protocol(changed)


def test_language_variants_change_wording_not_task_identity() -> None:
    protocol = _protocol()
    base = next(
        item
        for item in expand_episode_specs(protocol)
        if item["scenario_id"] == "S02-TOWER-ENDPOINT"
        and item["condition_id"] == "PRODUCTION_MONITOR"
    )
    tasks = []
    for variant_id in ("01", "02", "03"):
        spec = dict(base)
        spec["variant_id"] = variant_id
        tasks.append(_task_for(spec, protocol))
    assert len({task.instruction for task in tasks}) == 3
    assert {
        tuple(item.criterion_id for item in task.expected_observation)
        for task in tasks
    } == {("tower-bottom", "tower-middle", "tower-top")}
    assert all("green" in task.instruction.casefold() for task in tasks)
    assert all("blue" in task.instruction.casefold() for task in tasks)
    assert all("red" in task.instruction.casefold() for task in tasks)


def test_control_assessment_is_typed_and_claims_no_advancement() -> None:
    protocol = _protocol()
    spec = expand_episode_specs(protocol)[0]
    task = _task_for(spec, protocol)
    assessment = _control_assessment(task, observed_at=2.0, frame_sequence=1)
    assert assessment.task_status.value == "ONGOING"
    assert all(item.state.value == "UNKNOWN" for item in assessment.criteria)
    assert assessment.failure is None
    assert [item.criterion_id for item in assessment.criteria] == [
        item.criterion_id for item in task.expected_observation
    ]


def test_control_endpoint_continues_and_timeout_routes_attention() -> None:
    protocol = _protocol()
    specs = expand_episode_specs(protocol)
    endpoint = next(
        item
        for item in specs
        if item["scenario_id"] == "S01-PLACEMENT-ENDPOINT"
        and item["variant_id"] == "01"
        and item["condition_id"] == "NO_VISUAL_ADVANCEMENT_CONTROL"
    )
    timeout = next(
        item
        for item in specs
        if item["scenario_id"] == "T01-NO-PROGRESS-TIMEOUT"
        and item["variant_id"] == "01"
        and item["condition_id"] == "NO_VISUAL_ADVANCEMENT_CONTROL"
    )
    endpoint_row = run_episode(endpoint, protocol)
    timeout_row = run_episode(timeout, protocol)
    assert endpoint_row["observed_route"] == "CONTINUE_MONITORING"
    assert not endpoint_row["outcomes"]["stable_endpoint_transition"]
    assert timeout_row["observed_route"] == "ATTENTION_AFTER_NO_PROGRESS"
    assert timeout_row["outcomes"]["timeout_attention_transition"]
    assert endpoint_row["outcomes"]["model_invocation_attempts"] == 0


def _synthetic_rows(protocol):
    rows = []
    for spec in expand_episode_specs(protocol):
        production = spec["condition_id"] == "PRODUCTION_MONITOR"
        family = spec["family"]
        expected = (
            production
            or family
            in {"VISIBLE_PROGRESS", "NONTERMINAL_HOLD", "TIMEOUT_CONTROL"}
        )
        rows.append(
            {
                "episode_id": spec["episode_id"],
                "scenario_id": spec["scenario_id"],
                "variant_id": spec["variant_id"],
                "condition_id": spec["condition_id"],
                "family": family,
                "observed_route": spec["expected_route"] if expected else "CONTINUE_MONITORING",
                "expected_route": spec["expected_route"],
                "assessments": [],
                "capability_retries": 0,
                "outcomes": {
                    "expected_host_route_observed": expected,
                    "visual_failure_transition": (
                        production if family == "VISUAL_DISCREPANCY" else None
                    ),
                    "stable_endpoint_transition": (
                        production if family == "COMPLETED_ENDPOINT" else None
                    ),
                    "visible_progress_registered": (
                        production if family == "VISIBLE_PROGRESS" else None
                    ),
                    "safe_nonterminal_hold": (
                        True if family == "NONTERMINAL_HOLD" else None
                    ),
                    "timeout_attention_transition": (
                        True if family == "TIMEOUT_CONTROL" else None
                    ),
                    "false_terminal_transition": (
                        False
                        if family
                        in {"VISIBLE_PROGRESS", "NONTERMINAL_HOLD", "TIMEOUT_CONTROL"}
                        else None
                    ),
                    "typed_assessment_fraction": 1.0,
                    "criterion_identity_fraction": 1.0,
                    "capability_response_errors": 0,
                    "model_invocation_attempts": 0,
                    "model_latency_seconds": 0.0,
                    "emergency_stop_outputs": 0,
                    "physical_actions": 0,
                    "simulated_motion": 0,
                },
            }
        )
    return rows


def test_analysis_uses_paired_cluster_effects() -> None:
    protocol = _protocol()
    analysis = analyze_rows(protocol, _synthetic_rows(protocol))
    assert analysis["observed_episodes"] == 72
    assert analysis["paired_rows"] == 36
    assert analysis["paired_effects"]["visual_failure_transition"]["estimate"] == 1.0
    assert analysis["paired_effects"]["stable_endpoint_transition"]["estimate"] == 1.0
    assert analysis["paired_effects"]["visible_progress_registered"]["estimate"] == 1.0
    assert analysis["resources"]["physical_actions"] == 0


def test_report_is_function_specific_without_composite_verdict() -> None:
    protocol = _protocol()
    report = render_report(analyze_rows(protocol, _synthetic_rows(protocol)))
    assert "Stable endpoint transition" in report
    assert "Recovery transition" in report
    assert "Visible progress registered" in report
    assert "composite score" not in report
    assert "**Status:**" not in report
