from __future__ import annotations

import unittest

from experiments_suite_v2.io import load_json
from experiments_suite_v2.runners.focused_longitudinal_exp05 import (
    PROTOCOL_PATH,
    analyze_rows,
    expand_interaction_specs,
    score_interaction,
    validate_protocol,
)


def _state(records, digest):
    return {
        "sha256": digest,
        "record_count": len(records),
        "records": list(records),
        "embedding_shape": [len(records), 768],
        "consistent": True,
    }


def _score(spec, *, before, after, payloads=(), runtime_calls=(), response=""):
    return score_interaction(
        spec,
        before=before,
        after=after,
        payloads=payloads,
        retrieval_traces=(),
        runtime_calls=runtime_calls,
        response=response,
        capability_error=None,
        latency_seconds=1.0,
        model_calls={},
    )


class FocusedLongitudinalExp05Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = load_json(PROTOCOL_PATH)
        self.specs = expand_interaction_specs(self.protocol)

    def _spec(self, step_id):
        return next(
            item
            for item in self.specs
            if item["lifecycle_id"] == "L01" and item["step_id"] == step_id
        )

    def test_protocol_expands_eight_persistent_thirty_step_lifecycles(self) -> None:
        check = validate_protocol(self.protocol)
        self.assertEqual(check["lifecycle_clusters"], 8)
        self.assertEqual(check["interactions_per_cluster"], 30)
        self.assertEqual(check["interactions"], 240)
        self.assertEqual(check["target_board_counts"], {"cyan": 4, "white": 4})
        self.assertFalse(self.protocol["design"]["reset_within_lifecycle"])
        self.assertEqual(len({item["interaction_id"] for item in self.specs}), 240)
        for lifecycle in {item["lifecycle_id"] for item in self.specs}:
            indices = [
                item["step_index"]
                for item in self.specs
                if item["lifecycle_id"] == lifecycle
            ]
            self.assertEqual(indices, list(range(1, 31)))

    def test_authorized_creation_is_measured_from_store_delta(self) -> None:
        spec = self._spec("CREATE_MAIN")
        before = _state([], "before")
        after = _state([{"id": "main", "text": spec["expected_text"]}], "after")
        outcomes = _score(
            spec,
            before=before,
            after=after,
            payloads=({"status": "REMEMBERED"},),
        )
        self.assertTrue(outcomes["authorized_mutation_effect"])
        self.assertEqual(outcomes["mutation_status"], "REMEMBERED")
        self.assertFalse(outcomes["memory_store_unchanged"])

    def test_user_and_task_isolation_require_only_target_return(self) -> None:
        for step_id in ("RETRIEVE_USER_ISOLATION", "RETRIEVE_TASK_ISOLATION"):
            spec = self._spec(step_id)
            records = [
                {"id": "main", "text": spec["expected_text"]},
                {"id": "other", "text": spec["other_user_text"]},
                {"id": "secondary", "text": spec["secondary_text"]},
            ]
            state = _state(records, "same")
            payload = {
                "status": "FOUND",
                "retrieved_memory": [records[0]],
            }
            outcomes = _score(
                spec, before=state, after=state, payloads=(payload,)
            )
            with self.subTest(step_id=step_id):
                self.assertTrue(outcomes["main_preference_retrieved"])
                self.assertTrue(outcomes["out_of_scope_records_excluded"])

    def test_scoped_hri_goal_requires_concrete_cube_and_board(self) -> None:
        spec = self._spec("HRI_USE_MAIN_INITIAL")
        state = _state([{"id": "main", "text": spec["expected_text"]}], "same")
        outcomes = _score(
            spec,
            before=state,
            after=state,
            runtime_calls=(
                {
                    "method": "request_goal_preview",
                    "arguments": {
                        "clarified_goal": (
                            f"Place the {spec['expected_cube']} cube on the "
                            f"{spec['target_board']} board."
                        )
                    },
                },
            ),
        )
        self.assertTrue(outcomes["correct_goal_staged"])
        self.assertFalse(outcomes["unresolved_reference_preview"])

    def test_missing_task_scope_is_measured_as_abstention_and_clarification(self) -> None:
        spec = self._spec("HRI_AMBIGUOUS_WITHOUT_TASK")
        state = _state(
            [
                {"id": "main", "text": spec["initial_text"]},
                {"id": "secondary", "text": spec["secondary_text"]},
            ],
            "same",
        )
        outcomes = _score(
            spec,
            before=state,
            after=state,
            response="Which task should I use to choose the usual cube?",
        )
        self.assertTrue(outcomes["ambiguity_abstained"])
        self.assertTrue(outcomes["clarification_observed"])

    def test_explicit_update_measures_replacement_and_identity_preservation(self) -> None:
        spec = self._spec("UPDATE_MAIN")
        before = _state([{"id": "main", "text": spec["initial_text"]}], "before")
        after = _state([{"id": "main", "text": spec["updated_text"]}], "after")
        outcomes = _score(
            spec,
            before=before,
            after=after,
            payloads=({"status": "UPDATED"},),
        )
        self.assertTrue(outcomes["authorized_mutation_effect"])
        self.assertTrue(outcomes["main_identity_preserved"])
        self.assertTrue(outcomes["updated_value_active"])
        self.assertTrue(outcomes["old_value_absent"])

    def test_revocation_requires_store_removal_and_empty_followup(self) -> None:
        revoke = self._spec("REVOKE_SECONDARY")
        before = _state(
            [
                {"id": "main", "text": revoke["updated_text"]},
                {"id": "secondary", "text": revoke["secondary_text"]},
            ],
            "before",
        )
        after = _state([{"id": "main", "text": revoke["updated_text"]}], "after")
        outcomes = _score(
            revoke,
            before=before,
            after=after,
            payloads=({"status": "FORGOTTEN"},),
        )
        self.assertTrue(outcomes["authorized_mutation_effect"])
        self.assertTrue(outcomes["secondary_preference_removed"])

        verify = self._spec("RETRIEVE_REVOKED_SECONDARY")
        verified = _score(
            verify,
            before=after,
            after=after,
            payloads=({"status": "EMPTY", "retrieved_memory": []},),
        )
        self.assertTrue(verified["secondary_preference_removed"])
        self.assertTrue(verified["out_of_scope_records_excluded"])

    def test_analysis_uses_lifecycle_as_paired_unit(self) -> None:
        rows = []
        for spec in self.specs:
            step_id = spec["step_id"]
            mutation = spec["kind"] == "MEMORY_MUTATION"
            main_retrieval = step_id in {
                "RETRIEVE_MAIN_INITIAL",
                "RETRIEVE_USER_ISOLATION",
                "RETRIEVE_TASK_ISOLATION",
                "RETRIEVE_MAIN_UPDATED",
                "RETRIEVE_RETENTION",
                "RETRIEVE_FINAL_MAIN",
            }
            scoped_hri = step_id in {
                "HRI_USE_MAIN_INITIAL",
                "HRI_USE_MAIN_UPDATED",
                "HRI_USE_RETENTION",
            }
            rows.append(
                {
                    "interaction_id": spec["interaction_id"],
                    "context_id": spec["lifecycle_id"],
                    "lifecycle_id": spec["lifecycle_id"],
                    "step_index": spec["step_index"],
                    "step_id": step_id,
                    "kind": spec["kind"],
                    "outcomes": {
                        "authorized_mutation_effect": True if mutation else None,
                        "main_identity_preserved": True if step_id == "UPDATE_MAIN" else None,
                        "main_preference_retrieved": True if main_retrieval else None,
                        "out_of_scope_records_excluded": (
                            True
                            if main_retrieval or step_id == "RETRIEVE_REVOKED_SECONDARY"
                            else None
                        ),
                        "correct_goal_staged": True if scoped_hri else None,
                        "ambiguity_abstained": (
                            True if step_id == "HRI_AMBIGUOUS_WITHOUT_TASK" else None
                        ),
                        "clarification_observed": (
                            True if step_id == "HRI_AMBIGUOUS_WITHOUT_TASK" else None
                        ),
                        "updated_value_active": True if spec["step_index"] >= 9 else None,
                        "old_value_absent": True if spec["step_index"] >= 9 else None,
                        "secondary_preference_removed": (
                            True
                            if step_id in {"REVOKE_SECONDARY", "RETRIEVE_REVOKED_SECONDARY"}
                            else None
                        ),
                        "memory_store_unchanged": not mutation,
                        "store_consistent": True,
                        "store_record_count": min(spec["step_index"], 17),
                        "retrieval_candidate_count": 1 if "RETRIEVE" in step_id else None,
                        "unresolved_reference_preview": False,
                        "unauthorized_control_calls": 0,
                        "capability_error": None,
                        "latency_seconds": 1.0,
                        "model_calls": {},
                    },
                }
            )
        analysis = analyze_rows(self.protocol, rows)
        self.assertEqual(analysis["recorded_observations"], 240)
        effect = analysis["paired_effects"][
            "correct_goal_staged__HRI_USE_RETENTION_MINUS_HRI_USE_MAIN_INITIAL"
        ]
        self.assertEqual(effect["cluster_mean_difference"], 0.0)
        self.assertEqual(effect["lifecycle_clusters"], 8)
        self.assertEqual(
            analysis["group_summary"]["DISTRACTOR_MUTATIONS"]
            ["authorized_mutation_effect"]["sum"],
            112.0,
        )

    def test_outcomes_contain_no_pass_fail_or_composite_flag(self) -> None:
        spec = self._spec("RETRIEVE_MAIN_INITIAL")
        state = _state([], "same")
        outcomes = _score(
            spec,
            before=state,
            after=state,
            payloads=({"status": "EMPTY", "retrieved_memory": []},),
        )
        forbidden = {"pass", "passed", "fail", "failed", "success", "score"}
        self.assertFalse(forbidden & {key.casefold() for key in outcomes})


if __name__ == "__main__":
    unittest.main()
