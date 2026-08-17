from __future__ import annotations

import copy

import pytest

from experiments_suite_v2.io import load_json
from experiments_suite_v2.runners.focused_simulated_end_to_end_exp09 import (
    PROTOCOL_PATH,
    FocusedSimulatedEndToEndError,
    analyze_rows,
    build_rows,
    expand_episode_specs,
    extract_outcomes,
    render_report,
    validate_protocol,
)


def _protocol():
    return validate_protocol(load_json(PROTOCOL_PATH))


def test_protocol_registers_four_contract_types_and_twelve_episodes() -> None:
    protocol = _protocol()
    assert [item["contract_type_id"] for item in protocol["contract_types"]] == [
        "OMITTED_BOARD",
        "OMITTED_CUBE",
        "EXPLICIT_OVERRIDE",
        "USER_SCOPED",
    ]
    assert protocol["design"]["episodes_total"] == 12
    assert len(expand_episode_specs(protocol)) == 12


def test_derived_runtime_rows_are_unique_and_use_expected_base_cases() -> None:
    rows = build_rows(_protocol())
    assert len(rows) == 12
    assert len({row.trial_id for row in rows}) == 12
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.case_definition.case_id] = counts.get(
            row.case_definition.case_id, 0
        ) + 1
    assert counts == {
        "E2E-A-01": 3,
        "E2E-A-02": 3,
        "E2E-A-05": 3,
        "E2E-A-06": 3,
    }


def test_fresh_pairs_do_not_reuse_development_or_exp08_variants() -> None:
    specs = expand_episode_specs(_protocol())
    assert {item["variant_id"] for item in specs} == {"09", "10", "11"}
    assert not (
        {item["seed"] for item in specs}
        & {17011, 27109, 37199, 47297, 57331}
    )
    assert len({item["seed"] for item in specs}) == 12
    assert len({item["utterance"] for item in specs}) == 12


def test_real_robot_authorization_change_is_rejected() -> None:
    protocol = _protocol()
    changed = copy.deepcopy(protocol)
    changed["system_under_test"]["real_robot_actions"] = 1
    with pytest.raises(FocusedSimulatedEndToEndError, match="real-robot"):
        validate_protocol(changed)


def _strict_fixture(spec):
    source = spec["expected_action"]["source"]
    target = spec["expected_action"]["target"]
    transcript = (
        "call_memory_agent "
        + " ".join(spec["expected_memory_record_ids"])
        + " request_goal_preview confirm_goal_execution"
    )
    context = {
        "goal": {
            "goal": f"Move the {source} to the {target}.",
            "final_expected_observation": [f"the {source} is on the {target}"],
            "nominal_tasks": [f"place the {source} on the {target}"],
        },
        "executor_outcomes": [
            {
                "strict_system_result": "PASS",
                "oracle_fallback_used": False,
                "downstream_execution_result": "PASS",
                "grounding_attempts": [
                    {"role": "source", "query": source},
                    {"role": "target", "query": target},
                ],
            }
        ],
    }
    observed = {
        "completion_requirement_satisfied": True,
        "final_safety": True,
    }
    execution = {
        "physical_action_count": 1,
        "model_call_count": 7,
        "latency_seconds": 3.0,
        "terminal_runtime_state": "COMPLETE",
        "timed_out": False,
    }
    return observed, execution, context, [transcript]


def test_strict_outcome_extraction_traces_every_registered_stage() -> None:
    spec = expand_episode_specs(_protocol())[0]
    observed, execution, context, transcripts = _strict_fixture(spec)
    outcomes, action_trace = extract_outcomes(
        spec, observed, execution, context, transcripts
    )
    assert outcomes["preference_resolution_observed"]
    assert outcomes["first_compiled_action_semantically_correct"]
    assert outcomes["strict_grounding_completed"]
    assert not outcomes["oracle_assisted_continuation_used"]
    assert outcomes["strict_end_to_end_chain_observed"]
    assert action_trace["memory_lookup_before_preview"]


def test_assisted_outcome_remains_separate_from_strict_grounding() -> None:
    spec = next(
        item
        for item in expand_episode_specs(_protocol())
        if item["contract_type_id"] == "EXPLICIT_OVERRIDE"
    )
    observed, execution, context, transcripts = _strict_fixture(spec)
    context["executor_outcomes"][0].update(
        {
            "strict_system_result": "FAIL_GROUNDING",
            "oracle_fallback_used": True,
            "downstream_execution_result": "PASS",
        }
    )
    outcomes, _ = extract_outcomes(spec, observed, execution, context, transcripts)
    assert outcomes["strict_grounding_completed"] is False
    assert outcomes["oracle_assisted_continuation_used"] is True
    assert outcomes["assisted_continuation_completed"] is True
    assert outcomes["assisted_end_to_end_chain_observed"] is True
    assert outcomes["strict_end_to_end_chain_observed"] is False


def _synthetic_rows(protocol):
    rows = []
    for index, spec in enumerate(expand_episode_specs(protocol)):
        assisted = index % 3 == 2
        outcomes = {
            "memory_lookup_observed": True,
            "expected_memory_records_observed": True,
            "goal_preview_observed": True,
            "goal_contract_correct": True,
            "confirmation_observed": True,
            "preference_resolution_observed": True,
            "first_compiled_action_semantically_correct": True,
            "wrong_target_action_query_observed": False,
            "expected_action_count_observed": True,
            "strict_grounding_completed": not assisted,
            "oracle_assisted_continuation_used": assisted,
            "assisted_continuation_completed": True if assisted else None,
            "physical_goal_satisfied": True,
            "validator_finalized_complete": True,
            "final_safety": True,
            "strict_end_to_end_chain_observed": not assisted,
            "assisted_end_to_end_chain_observed": assisted,
            "terminal_runtime_state": "COMPLETE",
            "physical_action_count": 1,
            "model_call_count": 6,
            "latency_seconds": 2.0,
            "timed_out": False,
            "real_robot_actions": 0,
        }
        rows.append(
            {
                "episode_id": spec["episode_id"],
                "contract_type_id": spec["contract_type_id"],
                "capability_retries": 0,
                "outcomes": outcomes,
            }
        )
    return rows


def test_analysis_reports_each_contract_type_and_direct_stage() -> None:
    protocol = _protocol()
    analysis = analyze_rows(protocol, _synthetic_rows(protocol))
    assert analysis["observed_episodes"] == 12
    assert set(analysis["contract_type_summaries"]) == {
        "OMITTED_BOARD",
        "OMITTED_CUBE",
        "EXPLICIT_OVERRIDE",
        "USER_SCOPED",
    }
    assert analysis["overall"]["strict_end_to_end_chain_observed"]["sum"] == 8
    assert analysis["overall"]["assisted_end_to_end_chain_observed"]["sum"] == 4
    assert analysis["resources"]["real_robot_actions"] == 0


def test_report_keeps_memory_and_execution_boundaries_explicit() -> None:
    protocol = _protocol()
    report = render_report(analyze_rows(protocol, _synthetic_rows(protocol)))
    assert "Strict grounding" in report
    assert "Oracle continuation used" in report
    assert "does not estimate production Memory retrieval accuracy" in report
    assert "supporting VLA-shaped execution surrogate" in report
    assert "**Status:**" not in report
