from __future__ import annotations

import copy

import pytest

from experiments_suite_v2.io import load_json
from experiments_suite_v2.runners.focused_competency_envelope_exp08 import (
    PROTOCOL_PATH,
    FocusedCompetencyEnvelopeError,
    analyze_rows,
    build_rows,
    expand_episode_specs,
    extract_outcomes,
    render_report,
    validate_protocol,
)


def _protocol():
    return validate_protocol(load_json(PROTOCOL_PATH))


def test_protocol_registers_six_families_and_eighteen_episodes() -> None:
    protocol = _protocol()
    assert [item["family_id"] for item in protocol["families"]] == [
        "SLIP",
        "TIMEOUT",
        "OCCLUSION",
        "OUTAGE",
        "SCOPED_PREFERENCE",
        "EXPLICIT_OVERRIDE",
    ]
    assert protocol["design"]["episodes_total"] == 18
    assert len(expand_episode_specs(protocol)) == 18


def test_derived_runtime_rows_are_unique_and_use_expected_base_cases() -> None:
    protocol = _protocol()
    rows = build_rows(protocol)
    assert len(rows) == 18
    assert len({row.trial_id for row in rows}) == 18
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.case_definition.case_id] = counts.get(
            row.case_definition.case_id, 0
        ) + 1
    assert counts == {
        "E2E-R-01": 3,
        "E2E-R-02": 3,
        "E2E-R-06": 3,
        "E2E-R-10": 3,
        "E2E-A-06": 3,
        "E2E-A-05": 3,
    }


def test_fresh_seeds_and_utterances_do_not_reuse_development_rows() -> None:
    protocol = _protocol()
    specs = expand_episode_specs(protocol)
    assert {item["variant_id"] for item in specs} == {"06", "07", "08"}
    assert not ({item["seed"] for item in specs} & {17011, 27109, 37199, 47297, 57331})
    assert len({item["seed"] for item in specs}) == 18
    assert len({item["utterance"] for item in specs}) == 18


def test_real_robot_authorization_change_is_rejected() -> None:
    protocol = _protocol()
    changed = copy.deepcopy(protocol)
    changed["system_under_test"]["real_robot_actions"] = 1
    with pytest.raises(FocusedCompetencyEnvelopeError, match="real-robot"):
        validate_protocol(changed)


def test_recovery_outcome_extraction_keeps_strict_and_assisted_separate() -> None:
    spec = next(
        item for item in expand_episode_specs(_protocol()) if item["family_id"] == "SLIP"
    )
    observed = {
        "stimulus": {"fired": True},
        "route_events": spec["registered_route"],
        "goal_contract_unchanged": True,
        "registered_final_outcome": True,
        "final_safety": True,
    }
    execution = {
        "strict_system_result": "FAIL",
        "oracle_fallback_used": True,
        "assisted_continuation_result": "PASS",
        "terminal_runtime_state": "COMPLETE",
        "physical_action_count": 2,
        "model_call_count": 10,
        "latency_seconds": 4.0,
    }
    outcomes = extract_outcomes(spec, observed, execution)
    assert outcomes["registered_family_outcome_observed"]
    assert outcomes["strict_grounding_completed"] is False
    assert outcomes["oracle_assisted_continuation_used"] is True
    assert outcomes["assisted_continuation_completed"] is True


def test_preference_outcome_extraction_requires_authorized_completion() -> None:
    spec = next(
        item
        for item in expand_episode_specs(_protocol())
        if item["family_id"] == "EXPLICIT_OVERRIDE"
    )
    observed = {
        "decision_before_action_correct": True,
        "unauthorized_physical_action": False,
        "completion_requirement_satisfied": True,
        "final_safety": True,
    }
    execution = {
        "strict_system_result": "PASS",
        "oracle_fallback_used": False,
        "assisted_continuation_result": "NOT_APPLICABLE",
        "terminal_runtime_state": "COMPLETE",
        "physical_action_count": 1,
        "model_call_count": 8,
        "latency_seconds": 2.0,
    }
    outcomes = extract_outcomes(spec, observed, execution)
    assert outcomes["explicit_override_applied"]
    assert outcomes["registered_family_outcome_observed"]
    assert outcomes["unauthorized_physical_action"] is False


def _synthetic_rows(protocol):
    rows = []
    for index, spec in enumerate(expand_episode_specs(protocol)):
        recovery = spec["family_kind"] == "RECOVERY"
        rows.append(
            {
                "episode_id": spec["episode_id"],
                "family_id": spec["family_id"],
                "family_kind": spec["family_kind"],
                "capability_retries": 0,
                "outcomes": {
                    "registered_family_outcome_observed": index % 3 != 2,
                    "registered_final_outcome": True,
                    "final_safety": True,
                    "registered_stimulus_fired": True if recovery else None,
                    "registered_route_observed": True if recovery else None,
                    "goal_contract_preserved": True if recovery else None,
                    "decision_before_action_correct": True if not recovery else None,
                    "unauthorized_physical_action": False if not recovery else None,
                    "completion_requirement_satisfied": True if not recovery else None,
                    "scoped_preference_applied": (
                        True if spec["family_id"] == "SCOPED_PREFERENCE" else None
                    ),
                    "explicit_override_applied": (
                        True if spec["family_id"] == "EXPLICIT_OVERRIDE" else None
                    ),
                    "strict_grounding_completed": index % 2 == 0,
                    "oracle_assisted_continuation_used": index % 2 == 1,
                    "assisted_continuation_completed": True if index % 2 == 1 else None,
                    "terminal_runtime_state": "COMPLETE",
                    "physical_action_count": 1,
                    "model_call_count": 5,
                    "latency_seconds": 3.0,
                    "real_robot_actions": 0,
                },
            }
        )
    return rows


def test_analysis_reports_each_family_and_stratified_envelope() -> None:
    protocol = _protocol()
    analysis = analyze_rows(protocol, _synthetic_rows(protocol))
    assert analysis["observed_episodes"] == 18
    assert set(analysis["family_summaries"]) == {
        "SLIP",
        "TIMEOUT",
        "OCCLUSION",
        "OUTAGE",
        "SCOPED_PREFERENCE",
        "EXPLICIT_OVERRIDE",
    }
    assert analysis["overall"]["registered_family_outcome_observed"]["sum"] == 12
    assert analysis["resources"]["real_robot_actions"] == 0


def test_report_keeps_strict_and_assisted_evidence_separate() -> None:
    protocol = _protocol()
    report = render_report(analyze_rows(protocol, _synthetic_rows(protocol)))
    assert "Strict grounding" in report
    assert "Oracle continuation used" in report
    assert "SCOPED_PREFERENCE" in report
    assert "EXPLICIT_OVERRIDE" in report
    assert "**Status:**" not in report
