from __future__ import annotations

import copy

import pytest

from experiments_suite_v2.io import load_json
from experiments_suite_v2.runners.focused_validator_ablation_exp07 import (
    PROTOCOL_PATH,
    FocusedValidatorAblationError,
    _contract_for,
    _control_assessment,
    _criteria_for,
    analyze_rows,
    expand_episode_specs,
    render_report,
    run_episode,
    validate_protocol,
)


def _protocol():
    return validate_protocol(load_json(PROTOCOL_PATH))


def test_protocol_has_registered_incomplete_and_complete_clusters() -> None:
    protocol = _protocol()
    counts: dict[str, int] = {}
    for scenario in protocol["scenarios"]:
        counts[scenario["family"]] = counts.get(scenario["family"], 0) + 1
    assert counts == {
        "INCOMPLETE_FINAL_STATE": 7,
        "COMPLETE_FINAL_STATE": 5,
    }
    assert protocol["design"]["episodes_total"] == 72


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
        value == {"PRODUCTION_VALIDATOR", "STOP_AFTER_ACTIONS_CONTROL"}
        for value in pairs.values()
    )


def test_confounded_fixtures_and_motion_are_rejected() -> None:
    protocol = _protocol()
    changed = copy.deepcopy(protocol)
    changed["scenarios"][0]["frame"] = "V05"
    with pytest.raises(FocusedValidatorAblationError, match="confounded"):
        validate_protocol(changed)
    changed = copy.deepcopy(protocol)
    changed["system_under_test"]["physical_actions"] = 1
    with pytest.raises(FocusedValidatorAblationError, match="zero-motion"):
        validate_protocol(changed)


def test_language_variants_change_wording_not_criterion_count() -> None:
    protocol = _protocol()
    base = next(
        item
        for item in expand_episode_specs(protocol)
        if item["scenario_id"] == "C01-COMPLETE-TOWER"
    )
    criteria = []
    for variant_id in ("01", "02", "03"):
        spec = dict(base)
        spec["variant_id"] = variant_id
        criteria.append(_criteria_for(spec, protocol))
    assert all(len(items) == 3 for items in criteria)
    assert len({items[0] for items in criteria}) == 3
    assert all("green" in " ".join(items).casefold() for items in criteria)
    assert all("blue" in " ".join(items).casefold() for items in criteria)
    assert all("red" in " ".join(items).casefold() for items in criteria)


def test_stop_after_actions_control_is_typed_complete() -> None:
    protocol = _protocol()
    spec = expand_episode_specs(protocol)[0]
    contract = _contract_for(spec, protocol)
    assessment = _control_assessment(
        contract, publication_id=f"{spec['episode_id']}:publication"
    )
    assert [item.criterion_id for item in assessment.criteria] == [
        item.criterion_id for item in contract.detailed_criteria
    ]
    assert all(item.state.value == "MET" for item in assessment.criteria)


def test_control_false_completes_incomplete_and_finalizes_complete() -> None:
    protocol = _protocol()
    specs = expand_episode_specs(protocol)
    incomplete = next(
        item
        for item in specs
        if item["scenario_id"] == "I01-WRONG-TOWER-ORDER"
        and item["variant_id"] == "01"
        and item["condition_id"] == "STOP_AFTER_ACTIONS_CONTROL"
    )
    complete = next(
        item
        for item in specs
        if item["scenario_id"] == "C03-COMPLETE-PLACEMENT"
        and item["variant_id"] == "01"
        and item["condition_id"] == "STOP_AFTER_ACTIONS_CONTROL"
    )
    incomplete_row = run_episode(incomplete, protocol)
    complete_row = run_episode(complete, protocol)
    assert incomplete_row["observed_route"] == "FINALIZE_COMPLETE"
    assert incomplete_row["outcomes"]["false_completion"]
    assert complete_row["outcomes"]["complete_state_finalized"]
    assert incomplete_row["outcomes"]["model_invocation_attempts"] == 0


def _synthetic_rows(protocol):
    rows = []
    for spec in expand_episode_specs(protocol):
        production = spec["condition_id"] == "PRODUCTION_VALIDATOR"
        incomplete = spec["family"] == "INCOMPLETE_FINAL_STATE"
        if production:
            route = spec["expected_route"]
        else:
            route = "FINALIZE_COMPLETE"
        rows.append(
            {
                "episode_id": spec["episode_id"],
                "scenario_id": spec["scenario_id"],
                "variant_id": spec["variant_id"],
                "condition_id": spec["condition_id"],
                "family": spec["family"],
                "observed_route": route,
                "raw_assessments": [],
                "capability_retries": 0,
                "outcomes": {
                    "expected_finalization_route_observed": route
                    == spec["expected_route"],
                    "false_completion": route == "FINALIZE_COMPLETE"
                    if incomplete
                    else None,
                    "false_completion_prevented": route != "FINALIZE_COMPLETE"
                    if incomplete
                    else None,
                    "correction_requested": route == "REQUEST_CORRECTION"
                    if incomplete
                    else None,
                    "complete_state_finalized": route == "FINALIZE_COMPLETE"
                    if not incomplete
                    else None,
                    "false_rejection_of_complete_state": route
                    != "FINALIZE_COMPLETE"
                    if not incomplete
                    else None,
                    "more_evidence_requested": False,
                    "criterion_state_vector_exact": production or not incomplete,
                    "criterion_identity_exact": True,
                    "typed_assessment_available": True,
                    "capability_response_errors": 0,
                    "model_invocation_attempts": 0,
                    "model_latency_seconds": 0.0,
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
    assert analysis["paired_effects"]["false_completion"]["estimate"] == -1.0
    assert (
        analysis["paired_effects"]["false_completion_prevented"]["estimate"]
        == 1.0
    )
    assert analysis["paired_effects"]["complete_state_finalized"]["estimate"] == 0.0
    assert analysis["resources"]["physical_actions"] == 0


def test_report_separates_safety_benefit_and_completion_cost() -> None:
    protocol = _protocol()
    report = render_report(analyze_rows(protocol, _synthetic_rows(protocol)))
    assert "False completion, incomplete states" in report
    assert "Complete state finalized" in report
    assert "False rejection of complete state" in report
    assert "**Status:**" not in report
