from __future__ import annotations

import unittest

from experiments_suite_v2.io import load_json, sha256_file
from experiments_suite_v2.registry import SUITE_ROOT
from experiments_suite_v2.runners.cal_x05_centre_stack_supplement import (
    CENTRE_RESULT_SCHEMA,
    CalX05SupplementError,
    aggregate_centre_supplement,
    aggregate_tower_supplement,
    all_supplement_actions,
    build_centre_supplement_rows,
    build_tower_supplement_rows,
    supplement_definition,
    supplement_execution_boundary,
    supplement_registry_hashes,
    supplement_resource_estimate,
    validate_supplement_prerequisites,
    validate_supplement_registry,
    validate_centre_attempt_contract,
    validate_tower_attempt_contract,
)


def _attempt(row, *, scope, action_ids, number=0, predecessor_verdict=None):
    attempt = {
        "row_id": row.row_id,
        "trial_id": row.row_id,
        "attempt_id": f"{row.row_id}.A{number}",
        "attempt_number": number,
        "retry_of": None if number == 0 else f"{row.row_id}.A0",
        "supersedes_attempt": (
            None if number == 0 else f"{row.row_id}.A{number - 1}"
        ),
        "predecessor": None,
        "artifact_profile": "FULL_PHYSICAL",
        "attempt_allocated_before_reset": True,
        "initial_state_id": "V00",
        "reset_verified_before_first_inference": True,
        "reset_count_before_first_action": 1,
        "inter_action_reset_count": 0,
        "attempt_scope": scope,
        "partial_action_reuse": False,
        "response_reuse": False,
        "trajectory_reuse": False,
        "preallocated_action_ids": list(action_ids),
    }
    if number:
        attempt["predecessor"] = {
            "trial_id": row.row_id,
            "attempt_id": f"{row.row_id}.A{number - 1}",
            "attempt_number": number - 1,
            "status": "FINALIZED",
            "analytical_verdict": predecessor_verdict or "INVALID_RUN",
        }
    return attempt


def _centre_records(error_mm=(0.0, 0.0, 0.0)):
    return [
        {
            "row_id": row.row_id,
            "valid_terminal": True,
            "action_id": row.action.action_id,
            "program_contract_pass": True,
            "clean_v00_reset_verified": True,
            "attempt": _attempt(
                row,
                scope="SINGLE_ACTION_CENTRE",
                action_ids=[row.action.action_id],
            ),
            "scoring": {"final_placement_error_mm": list(error_mm)},
        }
        for row in build_centre_supplement_rows()
    ]


def _tower_records(*, common_shift_x_m=0.0, upper_extra_x_m=0.0):
    records = []
    for row in build_tower_supplement_rows():
        x, y = row.intended_axis_xy_m
        centres = {}
        speeds = {}
        for layer, cube_id in enumerate(row.order_bottom_to_top):
            extra = upper_extra_x_m if layer == 2 and row.order_index == 0 else 0.0
            centres[cube_id] = [
                x + common_shift_x_m + extra,
                y,
                0.037 + 0.05 * layer,
            ]
            speeds[cube_id] = 0.001
        records.append(
            {
                "row_id": row.row_id,
                "valid_terminal": True,
                "clean_v00_reset_verified": True,
                "whole_tower_attempt": True,
                "inter_action_reset_count": 0,
                "partial_action_reuse": False,
                "response_reuse": False,
                "trajectory_reuse": False,
                "attempt": _attempt(
                    row,
                    scope="ENTIRE_THREE_ACTION_TOWER",
                    action_ids=[action.action_id for action in row.actions],
                ),
                "board_id": row.board_id,
                "final_order_bottom_to_top": list(row.order_bottom_to_top),
                "final_board_id_by_cube": {
                    cube_id: row.board_id for cube_id in row.order_bottom_to_top
                },
                "final_support_chain": [
                    {
                        "support_object_id": row.board_id,
                        "supported_cube_id": row.order_bottom_to_top[0],
                        "contact_or_supported": True,
                    },
                    {
                        "support_object_id": row.order_bottom_to_top[0],
                        "supported_cube_id": row.order_bottom_to_top[1],
                        "contact_or_supported": True,
                    },
                    {
                        "support_object_id": row.order_bottom_to_top[1],
                        "supported_cube_id": row.order_bottom_to_top[2],
                        "contact_or_supported": True,
                    },
                ],
                "actions": [
                    {
                        "action_id": action.action_id,
                        "terminal_state": "SETTLED",
                        "source_cube_id": action.source_cube_id,
                        "source_selector": action.source_selector,
                        "source_anchor": action.source_anchor,
                        "target_object_id": action.target_object_id,
                        "target_selector": action.target_selector,
                        "target_anchor": action.target_anchor,
                        "program_contract_pass": True,
                    }
                    for action in row.actions
                ],
                "final_cube_centres_world_m": centres,
                "final_cube_linear_speed_mps": speeds,
            }
        )
    return records


class CalX05CentreStackSupplementTests(unittest.TestCase):
    def test_exact_registry_counts_hashes_order_and_action_contracts(self) -> None:
        centre = build_centre_supplement_rows()
        towers = build_tower_supplement_rows()
        actions = all_supplement_actions()
        self.assertEqual(len(centre), 60)
        self.assertEqual(len(towers), 10)
        self.assertEqual(len(actions), 90)
        self.assertEqual(len({row.row_id for row in (*centre, *towers)}), 70)
        self.assertEqual(len({action.action_id for action in actions}), 90)
        self.assertEqual(
            supplement_registry_hashes(),
            {
                "centre_rows_canonical_sha256": (
                    "dfd97d680f7e3c930809073880ecde51a25611c60f63ed4fa53cdbc480061cc0"
                ),
                "tower_rows_canonical_sha256": (
                    "3afce5b52c3ae9f3834362ad79165f723fcb79c59f15b9d7f2ae70bf0cb6f5ac"
                ),
                "all_actions_canonical_sha256": (
                    "24be83d1e37dc6c5e2bbe8d45fc3e8f6e816fa957127dddc9cf1e191c12d09bb"
                ),
                "prompt_tool_contract_canonical_sha256": (
                    "899e02e143ed884d0b8320a2716499abd5e4c70daece7c83a681037811257b5d"
                ),
                "execution_boundaries_canonical_sha256": (
                    "ebc119752ebadd1c6ca1ea9e15665870b5c2ebcae594a64c47bd8df064506b7d"
                ),
            },
        )
        self.assertTrue(all(row.source_core_row_id.endswith("CENTER-R%02d" % row.repeat_index) for row in centre))
        self.assertEqual(
            [row.tower_id for row in towers],
            [
                "V04-WHITE-GBR",
                "V05-CYAN-YPO",
                "V05-CYAN-YPO",
                "V04-WHITE-GBR",
                "V04-WHITE-GBR",
                "V05-CYAN-YPO",
                "V05-CYAN-YPO",
                "V04-WHITE-GBR",
                "V04-WHITE-GBR",
                "V05-CYAN-YPO",
            ],
        )
        for row in towers:
            self.assertEqual(
                [action.action_id for action in row.actions],
                [f"{row.row_id}-A00", f"{row.row_id}-A01", f"{row.row_id}-A02"],
            )
            self.assertEqual(row.actions[0].target_anchor, "surface_center")
            self.assertEqual(row.actions[1].target_anchor, "top_center")
            self.assertEqual(row.actions[2].target_anchor, "top_center")

    def test_registry_validation_and_definition_are_side_effect_free(self) -> None:
        validation = validate_supplement_registry(SUITE_ROOT)
        self.assertEqual(validation["status"], "PASS")
        self.assertFalse(validation["run_or_attempt_allocated"])
        self.assertFalse(validation["simulator_started"])
        self.assertEqual(validation["external_calls"], 0)
        self.assertEqual(validation["motion_commands"], 0)
        self.assertEqual(
            validation["runner_sha256"],
            sha256_file(SUITE_ROOT / validation["runner_path"]),
        )
        manifest = load_json(SUITE_ROOT / "protocol" / "protocol_manifest.json")
        runner_pins = [
            item
            for item in manifest["candidate_freeze_inputs"]
            if item["path"] == validation["runner_path"]
        ]
        self.assertEqual(len(runner_pins), 1)
        self.assertEqual(runner_pins[0]["sha256"], validation["runner_sha256"])
        definition = supplement_definition()
        self.assertEqual(definition["status"], "DEFINED_FROZEN_NOT_ALLOCATED")
        self.assertEqual(definition["counts"]["supplement_attempt_rows"], 70)
        self.assertEqual(definition["counts"]["supplement_physical_actions"], 90)
        self.assertEqual(
            definition["counts"]["nominal_sam_source_target_detections"], 180
        )
        self.assertFalse(definition["allocation_authorized"])
        self.assertFalse(definition["can_replace_or_rescue_core"])

    def test_upstream_gate_requires_canonical_q_and_complete_p(self) -> None:
        evidence = {
            "cal_x03": {
                "status": "PASS",
                "canonical_same_frame_rows": 520,
                "artifact_audit_status": "PASS",
                "protocol_designated_current": True,
                "development_or_diagnostic": False,
            },
            "phase_q_cal_x04": {
                "status": "PASS",
                "aim_id": "CAL-X-04",
                "purpose": "PRECEDING_ISOLATED_QUALIFICATION",
                "valid_terminal_rows": 300,
                "aggregate_gate": "PASS",
                "artifact_audit_status": "PASS",
                "suite_run_id": "phase-q",
            },
            "phase_p": {
                "status": "PASS",
                "valid_terminal_rows": 600,
                "paired_cal_x04_rows": 300,
                "paired_cal_x05_rows": 300,
                "cal_x04_aggregate_gate": "PASS",
                "cal_x05_aggregate_gate": "PASS",
                "artifact_audit_status": "PASS",
                "analysis_embargo_released": True,
                "complete_fresh_schedule": True,
                "suite_run_id": "phase-p",
            },
        }
        result = validate_supplement_prerequisites(
            evidence, supplement_suite_run_id="supplement"
        )
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["all_suite_run_ids_distinct"])

        evidence["cal_x03"]["development_or_diagnostic"] = True
        with self.assertRaisesRegex(CalX05SupplementError, "canonical CAL-X-03"):
            validate_supplement_prerequisites(
                evidence, supplement_suite_run_id="supplement"
            )
        evidence["cal_x03"]["development_or_diagnostic"] = False
        evidence["phase_p"]["cal_x05_aggregate_gate"] = "FAIL"
        with self.assertRaisesRegex(CalX05SupplementError, "600-row Phase-P"):
            validate_supplement_prerequisites(
                evidence, supplement_suite_run_id="supplement"
            )
        evidence["phase_p"]["cal_x05_aggregate_gate"] = "PASS"
        evidence["phase_q_cal_x04"]["suite_run_id"] = ""
        with self.assertRaisesRegex(CalX05SupplementError, "non-empty"):
            validate_supplement_prerequisites(
                evidence, supplement_suite_run_id="supplement"
            )
        evidence["phase_q_cal_x04"]["suite_run_id"] = "phase-q"
        evidence["phase_p"]["suite_run_id"] = "phase-q"
        with self.assertRaisesRegex(CalX05SupplementError, "600-row Phase-P"):
            validate_supplement_prerequisites(
                evidence, supplement_suite_run_id="supplement"
            )

    def test_tower_attempt_contract_requires_whole_clean_v00_retry(self) -> None:
        row = build_tower_supplement_rows()[0]
        attempt = _attempt(
            row,
            scope="ENTIRE_THREE_ACTION_TOWER",
            action_ids=[action.action_id for action in row.actions],
            number=1,
        )
        self.assertEqual(validate_tower_attempt_contract(row, attempt)["status"], "PASS")
        attempt["partial_action_reuse"] = True
        with self.assertRaisesRegex(CalX05SupplementError, "whole-row"):
            validate_tower_attempt_contract(row, attempt)

    def test_centre_and_tower_retry_lineage_rejects_wrong_or_capability_predecessor(self) -> None:
        centre = build_centre_supplement_rows()[0]
        centre_attempt = _attempt(
            centre,
            scope="SINGLE_ACTION_CENTRE",
            action_ids=[centre.action.action_id],
            number=2,
            predecessor_verdict="INVALID_SETUP",
        )
        self.assertEqual(
            validate_centre_attempt_contract(centre, centre_attempt)["status"],
            "PASS",
        )
        centre_attempt["supersedes_attempt"] = f"{centre.row_id}.A0"
        with self.assertRaisesRegex(CalX05SupplementError, "exact root"):
            validate_centre_attempt_contract(centre, centre_attempt)

        tower = build_tower_supplement_rows()[0]
        tower_attempt = _attempt(
            tower,
            scope="ENTIRE_THREE_ACTION_TOWER",
            action_ids=[action.action_id for action in tower.actions],
            number=1,
            predecessor_verdict="CAPABILITY_FAIL",
        )
        with self.assertRaisesRegex(CalX05SupplementError, "CAPABILITY_FAIL"):
            validate_tower_attempt_contract(tower, tower_attempt)

    def test_aggregates_fail_closed_when_attempt_contract_is_missing(self) -> None:
        centre_records = _centre_records()
        centre_records[0].pop("attempt")
        centre_gate = aggregate_centre_supplement(centre_records)
        self.assertEqual(centre_gate["status"], "FAIL")
        self.assertFalse(centre_gate["checks"]["attempt_contract_60_of_60"])

        valid_centre = aggregate_centre_supplement(_centre_records())
        tower_records = _tower_records()
        tower_records[0].pop("attempt")
        tower_gate = aggregate_tower_supplement(
            tower_records, centre_gate=valid_centre
        )
        self.assertEqual(tower_gate["status"], "FAIL")
        self.assertFalse(tower_gate["checks"]["attempt_contract_10_of_10"])

    def test_execution_boundary_routes_only_action_safe_symbolic_fields(self) -> None:
        row = build_tower_supplement_rows()[0]
        action = row.actions[1]
        with self.assertRaisesRegex(TypeError, "SupplementAction"):
            supplement_execution_boundary(row)
        boundary = supplement_execution_boundary(action)
        self.assertEqual(
            set(boundary["compiler_input"]),
            {"action_id", "instruction", "expected_observation"},
        )
        self.assertEqual(
            [set(item) for item in boundary["sam_inputs"]],
            [
                {"action_id", "role", "selector", "anchor"},
                {"action_id", "role", "selector", "anchor"},
            ],
        )
        self.assertEqual(
            boundary["controller_symbolic_input"]["runtime_anchor_source"],
            "SAM_CAMERA_RGBD_GROUNDING_ONLY",
        )
        program = boundary["controller_symbolic_input"]["program"]
        self.assertEqual(
            supplement_execution_boundary(action, compiled_program=program),
            boundary,
        )
        unsafe_program = dict(program)
        unsafe_program["hidden_target_world_m"] = [0.55, -0.16, 0.012]
        with self.assertRaisesRegex(CalX05SupplementError, "unsafe"):
            supplement_execution_boundary(action, compiled_program=unsafe_program)
        changed_program = dict(program)
        changed_program["target"] = dict(program["target"], query="white board")
        with self.assertRaisesRegex(CalX05SupplementError, "differs"):
            supplement_execution_boundary(action, compiled_program=changed_program)

    def test_centre_gate_is_exact_and_precedes_tower(self) -> None:
        centre = aggregate_centre_supplement(_centre_records())
        self.assertEqual(centre["schema_version"], CENTRE_RESULT_SCHEMA)
        self.assertEqual(centre["status"], "PASS")
        self.assertEqual(centre["valid_terminal_rows"], 60)
        self.assertEqual(set(centre["metrics"]["by_cube_id"]), {
            "red_cube", "green_cube", "blue_cube", "yellow_cube", "purple_cube", "orange_cube"
        })
        self.assertEqual(set(centre["metrics"]["by_board_id"]), {"white_board", "cyan_board"})

        failed = aggregate_centre_supplement(_centre_records((6.0, 0.0, 0.0)))
        self.assertEqual(failed["status"], "FAIL")
        with self.assertRaisesRegex(CalX05SupplementError, "centre PASS"):
            aggregate_tower_supplement(_tower_records(), centre_gate=failed)

    def test_tower_gate_scores_all_20_pairs_and_axis_is_diagnostic(self) -> None:
        centre = aggregate_centre_supplement(_centre_records())
        # A common 8 mm offset keeps adjacent layers aligned.  It is reported
        # against the intended tower axis but does not invent an extra formal
        # threshold beyond the registered 10 mm layer-to-layer gate.
        tower = aggregate_tower_supplement(
            _tower_records(common_shift_x_m=0.008), centre_gate=centre
        )
        self.assertEqual(tower["status"], "PASS")
        self.assertEqual(tower["complete_towers"], 10)
        self.assertEqual(tower["adjacent_comparisons"], 20)
        self.assertEqual(tower["passing_adjacent_comparisons"], 20)
        self.assertEqual(
            len(tower["intended_common_axis_displacements_diagnostic"]), 30
        )
        self.assertTrue(
            all(
                item["formal_gate"] is False
                for item in tower["intended_common_axis_displacements_diagnostic"]
            )
        )

        failed = aggregate_tower_supplement(
            _tower_records(upper_extra_x_m=0.011), centre_gate=centre
        )
        self.assertEqual(failed["status"], "FAIL")
        self.assertEqual(failed["passing_adjacent_comparisons"], 19)
        self.assertFalse(
            failed["checks"]["adjacent_displacements_within_10_mm_20_of_20"]
        )

        wrong_support = _tower_records()
        wrong_support[0]["final_support_chain"][2]["contact_or_supported"] = False
        topology_failed = aggregate_tower_supplement(
            wrong_support, centre_gate=centre
        )
        self.assertEqual(topology_failed["status"], "FAIL")
        self.assertFalse(topology_failed["failed_rows"][0]["support_contract"])

    def test_resource_math_is_additive_and_explicit(self) -> None:
        estimate = supplement_resource_estimate()
        self.assertEqual(estimate["centre"]["action_equivalents"], 60)
        self.assertEqual(estimate["tower"]["action_equivalents"], 30)
        self.assertEqual(estimate["total"]["action_equivalents"], 90)
        self.assertAlmostEqual(estimate["total"]["nominal_wall_hours"], 1.4646654687499905)
        self.assertAlmostEqual(estimate["total"]["nominal_storage_gb_decimal"], 1.96085664)
        self.assertFalse(estimate["invalid_retry_overhead_included"])
        self.assertFalse(estimate["service_latency_overhead_included"])


if __name__ == "__main__":
    unittest.main()
