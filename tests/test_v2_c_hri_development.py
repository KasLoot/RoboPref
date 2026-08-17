from __future__ import annotations

import unittest

from experiments_suite_v2.io import load_json, sha256_file
from experiments_suite_v2.runners.c_hri_development import (
    FIXTURE_ROOT,
    PROTOCOL_PATH,
    _VISUAL_CONTEXT_TO_STATE,
    evaluate_response,
    fixture_context,
)
from experiments_suite_v2.runners.component import expand_component_trials


class CHRIDevelopmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.trials = tuple(
            trial
            for trial in expand_component_trials(
                software_config_commit="c-hri-development-test"
            )
            if trial.case_definition.aim_id.startswith("C-HRI-")
        )

    def _trial(self, aim: str, label: str):
        return next(
            trial
            for trial in self.trials
            if trial.case_definition.aim_id == aim
            and trial.trial_tuple.extra_frozen_factors["context_label"] == label
        )

    def test_scope_is_exact_six_aims_and_240_rows(self) -> None:
        self.assertEqual(len(self.trials), 240)
        aims = []
        for trial in self.trials:
            if trial.case_definition.aim_id not in aims:
                aims.append(trial.case_definition.aim_id)
        self.assertEqual(
            aims,
            [
                "C-HRI-ID",
                "C-HRI-VIS",
                "C-HRI-AMB",
                "C-HRI-TOOL",
                "C-HRI-PREF",
                "C-HRI-STATE",
            ],
        )
        for aim in aims:
            self.assertEqual(
                sum(item.case_definition.aim_id == aim for item in self.trials),
                40,
            )

    def test_every_registered_context_has_a_nonempty_rubric(self) -> None:
        for trial in self.trials:
            result = evaluate_response(
                trial,
                {"content": "No supported response."},
                [],
            )
            self.assertTrue(result["checks"], trial.trial_id)
            self.assertEqual(
                result["passed"], all(result["checks"].values())
            )

    def test_identity_role_requires_claims_not_the_expected_label(self) -> None:
        trial = self._trial("C-HRI-ID", "role")
        passed = evaluate_response(
            trial,
            {
                "content": (
                    "I am the HRI Agent, PrefMem's user-facing component. I "
                    "clarify your intended outcome. I do not physically move "
                    "objects; the runtime and execution adapter do that."
                )
            },
            [],
        )
        self.assertTrue(passed["passed"])
        failed = evaluate_response(
            trial,
            {"content": "role"},
            [],
        )
        self.assertFalse(failed["passed"])

    def test_tool_route_scores_exact_runtime_call_and_arguments(self) -> None:
        trial = self._trial("C-HRI-TOOL", "execution_request")
        good = evaluate_response(
            trial,
            {"content": "The execution request is recorded and planning begins."},
            [
                {
                    "method": "confirm_goal",
                    "arguments": {
                        "goal_id": "goal-fixture-01",
                        "revision": 1,
                        "confirmed": True,
                    },
                }
            ],
        )
        self.assertTrue(good["passed"])
        bad = evaluate_response(
            trial,
            {"content": "The execution request is recorded."},
            [
                {
                    "method": "confirm_goal",
                    "arguments": {
                        "goal_id": "invented",
                        "revision": 1,
                        "confirmed": True,
                    },
                }
            ],
        )
        self.assertFalse(bad["passed"])
        self.assertFalse(bad["checks"]["exact_confirmation"])

    def test_state_report_cannot_use_control_tool_or_claim_completion(self) -> None:
        trial = self._trial("C-HRI-STATE", "final_evidence_unknown")
        good = evaluate_response(
            trial,
            {
                "content": (
                    "Final validation needs evidence: the contact is unknown. "
                    "Move only the camera to obtain another view, then resume."
                )
            },
            [],
        )
        self.assertTrue(good["passed"])
        bad = evaluate_response(
            trial,
            {"content": "The task is complete."},
            [],
        )
        self.assertFalse(bad["passed"])
        self.assertFalse(bad["checks"]["no_false_completion"])

    def test_fixture_runtime_context_is_non_scoring_and_no_motion(self) -> None:
        trial = self._trial("C-HRI-TOOL", "plan_preview")
        context = fixture_context(trial)
        fixture = context["experiment_fixture"]
        self.assertEqual(fixture["context_label"], "plan_preview")
        self.assertEqual(
            fixture["scenario"]["registered_route"], "request_goal_preview"
        )
        self.assertEqual(context["state"], "IDLE")

    def test_visual_fixture_manifests_resolve_and_hash_match(self) -> None:
        for context_id, state_id in _VISUAL_CONTEXT_TO_STATE.items():
            for variant_id in ("01", "02", "03", "04", "05"):
                manifest_path = (
                    FIXTURE_ROOT
                    / f"ctx-{context_id}"
                    / f"var-{variant_id}"
                    / "frames.json"
                )
                manifest = load_json(manifest_path)
                self.assertEqual(manifest["registered_state_or_trace"], state_id)
                image_path = (manifest_path.parent / manifest["image_path"]).resolve()
                self.assertTrue(image_path.is_file())
                self.assertEqual(
                    sha256_file(image_path), manifest["image_sha256"]
                )

    def test_protocol_is_development_only_and_contains_no_results(self) -> None:
        protocol = load_json(PROTOCOL_PATH)
        self.assertEqual(
            protocol["status"], "DEVELOPMENT_FROZEN_BEFORE_LIVE_CALLS"
        )
        self.assertFalse(protocol["confirmatory"])
        self.assertEqual(protocol["scope"]["total_rows"], 240)
        self.assertEqual(protocol["scope"]["cal_x_retry_or_fallback"], "PROHIBITED_BY_USER")
        self.assertEqual(protocol["live_calls_performed_when_frozen"], 0)
        self.assertFalse(protocol["results_included_when_frozen"])


if __name__ == "__main__":
    unittest.main()
