from __future__ import annotations

import unittest

from prefmem.contracts import (
    BroadValidationResult,
    CriterionState,
    GoalContract,
    ValidationAssessment,
    ValidationChecklistDraft,
    ValidationContract,
    ValidationReport,
    ValidationStatus,
    freeze_validation_contract,
)


def _goal() -> GoalContract:
    return GoalContract(
        goal_id="goal-1",
        revision=2,
        goal="Put the sorting scene in its requested final state.",
        final_expected_observation=(
            "All red blocks are in the red tray.",
            "All blue blocks are in the blue tray.",
            "No blocks remain on the table.",
        ),
        constraints=("Do not move either tray.",),
    )


def _draft() -> ValidationChecklistDraft:
    # Deliberately out of order: broad_index is the model's mapping key and the
    # host restores the frozen goal order.
    return ValidationChecklistDraft.from_model_output(
        {
            "broad_items": [
                {
                    "broad_index": 2,
                    "detailed_criteria": [
                        "The visible work surface contains no loose blocks."
                    ],
                },
                {
                    "broad_index": 0,
                    "detailed_criteria": [
                        "Every visible red block is inside the red tray boundary.",
                        "No visible red block is outside the red tray.",
                    ],
                },
                {
                    "broad_index": 1,
                    "detailed_criteria": [
                        "Every visible blue block is inside the blue tray boundary.",
                        "No visible blue block is outside the blue tray.",
                    ],
                },
            ]
        }
    )


def _contract() -> ValidationContract:
    return freeze_validation_contract(
        _goal(),
        _draft(),
        validation_id="validation-1",
    )


def _assessment(
    states: tuple[CriterionState | str, ...],
) -> ValidationAssessment:
    contract = _contract()
    return ValidationAssessment.from_model_output(
        contract,
        {
            "criteria": [
                {
                    "id": criterion.criterion_id,
                    "state": state,
                    "evidence": f"Frame evidence covers check {index}.",
                }
                for index, (criterion, state) in enumerate(
                    zip(contract.detailed_criteria, states, strict=True),
                    start=1,
                )
            ],
            "observation": "The full sorting area is visible in the frame.",
        },
        publication_id="pub-final-1",
        observed_at=42.5,
        frame_sequence=17,
    )


class ValidationChecklistDraftTests(unittest.TestCase):
    def test_model_draft_is_strict_and_rejects_duplicate_mappings(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown fields: confidence"):
            ValidationChecklistDraft.from_model_output(
                {"broad_items": [], "confidence": 0.9}
            )
        with self.assertRaisesRegex(ValueError, "duplicate broad_index"):
            ValidationChecklistDraft.from_model_output(
                {
                    "broad_items": [
                        {"broad_index": 0, "detailed_criteria": ["First check."]},
                        {"broad_index": 0, "detailed_criteria": ["Second check."]},
                    ]
                }
            )
        with self.assertRaisesRegex(ValueError, "1 to 5"):
            ValidationChecklistDraft.from_model_output(
                {
                    "broad_items": [
                        {"broad_index": 0, "detailed_criteria": []}
                    ]
                }
            )
        with self.assertRaisesRegex(ValueError, "must not contain duplicates"):
            ValidationChecklistDraft.from_model_output(
                {
                    "broad_items": [
                        {
                            "broad_index": 0,
                            "detailed_criteria": ["Visible check.", "visible CHECK."],
                        }
                    ]
                }
            )

    def test_freeze_uses_exact_goal_labels_order_and_deterministic_ids(self) -> None:
        contract = _contract()
        self.assertEqual(
            tuple(item.label for item in contract.broad_items),
            _goal().final_expected_observation,
        )
        self.assertEqual(
            [item.broad_index for item in contract.broad_items],
            [0, 1, 2],
        )
        self.assertEqual(contract.broad_items[0].broad_id, "validation-1:b1")
        self.assertEqual(
            contract.broad_items[0].detailed_criteria[1].criterion_id,
            "validation-1:b1:c2",
        )
        self.assertEqual(
            ValidationContract.from_dict(contract.to_dict()),
            contract,
        )

    def test_freeze_requires_exact_broad_index_coverage(self) -> None:
        incomplete = ValidationChecklistDraft.from_model_output(
            {
                "broad_items": [
                    {"broad_index": 0, "detailed_criteria": ["One check."]},
                    {"broad_index": 2, "detailed_criteria": ["Three check."]},
                ]
            }
        )
        with self.assertRaisesRegex(ValueError, "exactly once"):
            freeze_validation_contract(
                _goal(),
                incomplete,
                validation_id="validation-2",
            )

    def test_contract_rejects_tampered_goal_labels_and_ids(self) -> None:
        payload = _contract().to_dict()
        payload["broad_items"][0]["label"] = "A different requirement."
        with self.assertRaisesRegex(ValueError, "exactly match"):
            ValidationContract.from_dict(payload)

        payload = _contract().to_dict()
        payload["broad_items"][0]["detailed_criteria"][0]["id"] = "model-id"
        with self.assertRaisesRegex(ValueError, "deterministic"):
            ValidationContract.from_dict(payload)


class ValidationAssessmentTests(unittest.TestCase):
    def test_assessment_has_trusted_envelope_and_model_only_view(self) -> None:
        assessment = _assessment(
            (
                CriterionState.MET,
                CriterionState.MET,
                CriterionState.MET,
                CriterionState.MET,
                CriterionState.MET,
            )
        )
        self.assertEqual(assessment.validation_id, "validation-1")
        self.assertEqual(assessment.goal_id, "goal-1")
        self.assertEqual(assessment.goal_revision, 2)
        self.assertEqual(assessment.publication_id, "pub-final-1")
        self.assertEqual(assessment.observed_at, 42.5)
        self.assertEqual(assessment.frame_sequence, 17)
        self.assertEqual(
            set(assessment.to_model_dict()),
            {"criteria", "observation"},
        )
        self.assertEqual(
            assessment.to_dict()["criteria"][0]["evidence"],
            "Frame evidence covers check 1.",
        )

    def test_assessment_requires_exact_ids_and_order(self) -> None:
        contract = _contract()
        valid_criteria = [
            {
                "id": item.criterion_id,
                "state": "MET",
                "evidence": "The expected visual state is present.",
            }
            for item in contract.detailed_criteria
        ]
        valid_criteria[0], valid_criteria[1] = valid_criteria[1], valid_criteria[0]
        with self.assertRaisesRegex(ValueError, "exact frozen IDs.*contract order"):
            ValidationAssessment.from_model_output(
                contract,
                {
                    "criteria": valid_criteria,
                    "observation": "The scene is visible.",
                },
                publication_id="pub-1",
                observed_at=1.0,
                frame_sequence=1,
            )

        valid_criteria[0] = dict(valid_criteria[1])
        with self.assertRaisesRegex(ValueError, "must be unique"):
            ValidationAssessment.from_model_output(
                contract,
                {
                    "criteria": valid_criteria,
                    "observation": "The scene is visible.",
                },
                publication_id="pub-1",
                observed_at=1.0,
                frame_sequence=1,
            )

    def test_assessment_rejects_unknown_fields_and_multisentence_evidence(self) -> None:
        contract = _contract()
        criteria = [
            {
                "id": item.criterion_id,
                "state": "UNKNOWN",
                "evidence": "The view is blocked. A second view is needed.",
            }
            for item in contract.detailed_criteria
        ]
        with self.assertRaisesRegex(ValueError, "must be one sentence"):
            ValidationAssessment.from_model_output(
                contract,
                {"criteria": criteria, "observation": "The scene is visible."},
                publication_id="pub-1",
                observed_at=1.0,
                frame_sequence=1,
            )
        criteria[0]["evidence"] = "The view is blocked."
        criteria[0]["confidence"] = 0.5
        with self.assertRaisesRegex(ValueError, "unknown fields: confidence"):
            ValidationAssessment.from_model_output(
                contract,
                {"criteria": criteria, "observation": "The scene is visible."},
                publication_id="pub-1",
                observed_at=1.0,
                frame_sequence=1,
            )


class ValidationReportTests(unittest.TestCase):
    def test_complete_is_derived_when_all_detailed_checks_are_met(self) -> None:
        report = ValidationReport.from_assessment(
            _contract(),
            _assessment(("MET", "MET", "MET", "MET", "MET")),
        )
        self.assertIs(report.status, ValidationStatus.COMPLETE)
        self.assertEqual(
            [item.state for item in report.checklist],
            [CriterionState.MET, CriterionState.MET, CriterionState.MET],
        )
        self.assertEqual(report.evidence_requests, ())
        self.assertEqual(
            report.summary,
            "Task complete — all 3 required outcomes were visually verified.",
        )
        self.assertNotIn("detailed_criteria", report.to_dict())

    def test_not_met_precedes_unknown_and_drives_incomplete(self) -> None:
        report = ValidationReport.from_assessment(
            _contract(),
            _assessment(("MET", "UNKNOWN", "UNKNOWN", "NOT_MET", "MET")),
        )
        self.assertIs(report.status, ValidationStatus.INCOMPLETE)
        self.assertEqual(
            [item.state for item in report.checklist],
            [CriterionState.UNKNOWN, CriterionState.NOT_MET, CriterionState.MET],
        )
        self.assertEqual(
            report.checklist[1].evidence,
            "Frame evidence covers check 4.",
        )
        self.assertEqual(
            report.evidence_requests,
            (
                "Provide a clear camera view to verify: All red blocks are in "
                "the red tray.",
            ),
        )
        self.assertEqual(
            report.summary,
            "Task incomplete — 1 of 3 required outcomes were visually verified.",
        )

    def test_unknown_without_not_met_drives_needs_evidence(self) -> None:
        report = ValidationReport.from_assessment(
            _contract(),
            _assessment(("MET", "MET", "UNKNOWN", "MET", "MET")),
        )
        self.assertIs(report.status, ValidationStatus.NEEDS_EVIDENCE)
        self.assertEqual(
            report.summary,
            "Final validation needs more evidence — 1 of 3 required outcomes "
            "could not be verified.",
        )
        self.assertEqual(len(report.evidence_requests), 1)

    def test_report_constructor_rejects_status_or_request_disagreement(self) -> None:
        checklist = (
            BroadValidationResult(
                label="The outcome is visible.",
                state="UNKNOWN",
                evidence="The target is occluded.",
            ),
        )
        with self.assertRaisesRegex(ValueError, "conflicts"):
            ValidationReport(
                validation_id="validation-1",
                status="COMPLETE",
                checklist=checklist,
                evidence_requests=(
                    "Provide a clear camera view to verify: The outcome is visible.",
                ),
                observation="The scene is partly hidden.",
            )
        with self.assertRaisesRegex(ValueError, "exactly match"):
            ValidationReport(
                validation_id="validation-1",
                status="NEEDS_EVIDENCE",
                checklist=checklist,
                evidence_requests=("Try again.",),
                observation="The scene is partly hidden.",
            )


if __name__ == "__main__":
    unittest.main()
