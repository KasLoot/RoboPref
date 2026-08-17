from __future__ import annotations

import json
import unittest

from experiments_suite_v2.schemas import (
    AnalyticalClassification,
    ArtifactProfile,
    AttemptIdentity,
    CaseDefinition,
    FormalVerdict,
    InvalidityEvidence,
    ResultEnvelope,
    SchemaError,
    TrialTuple,
    canonical_json_bytes,
)


def make_trial(**overrides: object) -> TrialTuple:
    values: dict[str, object] = {
        "suite_version": "PrefMem-Experiment-Suite-v2",
        "case_id": "C-HRI-ID",
        "aim_id": "C-HRI-ID",
        "prefix": "C-HRI",
        "condition_id": "PRODUCTION",
        "scene_state_id": "NOT_APPLICABLE",
        "reset_manifest_hash": "NOT_APPLICABLE",
        "scene_config_hash": "NOT_APPLICABLE",
        "robot_asset_id": "NOT_APPLICABLE",
        "language_context_id": "01",
        "exact_utterance": "Which component may change my goal?",
        "memory_store_snapshot_hash": "NOT_APPLICABLE",
        "model_config_hash": "model-hash",
        "prompt_tool_schema_hash": "prompt-hash",
        "artifact_profile": ArtifactProfile.MODEL_COMPONENT,
        "random_seed": 11,
        "software_config_commit": "commit",
        "variant_id": "01",
    }
    values.update(overrides)
    return TrialTuple(**values)


def make_case(**overrides: object) -> CaseDefinition:
    values: dict[str, object] = {
        "suite_version": "PrefMem-Experiment-Suite-v2",
        "scene_version": "NOT_APPLICABLE",
        "robot_asset_id": "NOT_APPLICABLE",
        "case_id": "C-HRI-ID",
        "aim_id": "C-HRI-ID",
        "prefix": "C-HRI",
        "title": "HRI identity boundary",
        "target_boundary": "HRI",
        "live_components": ("HRI",),
        "input_spec": {"utterance": "frozen"},
        "typed_state_fixture": "NOT_APPLICABLE",
        "frame_fixture": "NOT_APPLICABLE",
        "memory_fixture": "NOT_APPLICABLE",
        "model_build_prompt_hashes": {"prompt": "abc"},
        "deterministic_settings": {"temperature": 0},
        "seeds": (11,),
        "permitted_variations": {"variant": "01"},
        "primary_oracle": "claim_contract",
        "must_pass": ("truthful_authority",),
        "should_measure": ("latency_ms",),
        "invalid_if": ("endpoint_disconnect",),
        "artifact_profile": ArtifactProfile.MODEL_COMPONENT,
    }
    values.update(overrides)
    return CaseDefinition(**values)


class V2SchemaTests(unittest.TestCase):
    def test_trial_tuple_hash_is_canonical_and_every_factor_changes_identity(self) -> None:
        trial = make_trial()
        same = TrialTuple.from_dict(trial.to_dict())
        changed = make_trial(exact_utterance="A different frozen utterance")
        self.assertEqual(same.trial_tuple_sha256, trial.trial_tuple_sha256)
        self.assertNotEqual(changed.trial_tuple_sha256, trial.trial_tuple_sha256)
        self.assertEqual(trial.trial_id, "C-HRI-ID__PRODUCTION__ctx-01__var-01__seed-11")

    def test_result_classifications_enforce_formal_verdict_and_denominator(self) -> None:
        trial = make_trial()
        passed = ResultEnvelope(
            attempt_id=f"{trial.trial_id}.A0",
            trial_tuple_sha256=trial.trial_tuple_sha256,
            formal_verdict=FormalVerdict.PASS,
            analytical_classification=AnalyticalClassification.PASS,
            primary_oracle="claim_contract",
            must_pass={"truthful_authority": True},
            failed_predicates=(),
        )
        self.assertTrue(passed.enters_capability_denominator)

        capability_fail = ResultEnvelope(
            attempt_id=f"{trial.trial_id}.A0",
            trial_tuple_sha256=trial.trial_tuple_sha256,
            formal_verdict=FormalVerdict.FAIL,
            analytical_classification=AnalyticalClassification.CAPABILITY_FAIL,
            primary_oracle="claim_contract",
            must_pass={"truthful_authority": False},
            failed_predicates=("truthful_authority",),
        )
        self.assertTrue(capability_fail.enters_capability_denominator)

        invalid = ResultEnvelope(
            attempt_id=f"{trial.trial_id}.A0",
            trial_tuple_sha256=trial.trial_tuple_sha256,
            formal_verdict=FormalVerdict.FAIL,
            analytical_classification=AnalyticalClassification.INVALID_RUN,
            primary_oracle="claim_contract",
            must_pass={"truthful_authority": False},
            failed_predicates=("truthful_authority",),
            invalidity=InvalidityEvidence(
                reason_code="ENDPOINT_DISCONNECT",
                explanation="SSH forwarding ended before a usable response.",
                evidence_paths=("terminal.log",),
                repair="Restore the exact service and rerun the same tuple.",
            ),
        )
        self.assertFalse(invalid.enters_capability_denominator)

    def test_inconsistent_results_are_rejected(self) -> None:
        trial = make_trial()
        cases = (
            (FormalVerdict.PASS, AnalyticalClassification.CAPABILITY_FAIL, {"p": False}, ("p",)),
            (FormalVerdict.FAIL, AnalyticalClassification.PASS, {"p": True}, ()),
            (FormalVerdict.FAIL, AnalyticalClassification.CAPABILITY_FAIL, {"p": True}, ()),
        )
        for formal, analytical, must_pass, failed in cases:
            with self.subTest(formal=formal, analytical=analytical):
                with self.assertRaises(SchemaError):
                    ResultEnvelope(
                        attempt_id=f"{trial.trial_id}.A0",
                        trial_tuple_sha256=trial.trial_tuple_sha256,
                        formal_verdict=formal,
                        analytical_classification=analytical,
                        primary_oracle="oracle",
                        must_pass=must_pass,
                        failed_predicates=failed,
                    )

    def test_retry_identity_requires_complete_lineage(self) -> None:
        trial = make_trial()
        with self.assertRaisesRegex(SchemaError, "A1"):
            AttemptIdentity(trial.trial_id, trial.trial_tuple_sha256, 1)
        identity = AttemptIdentity(
            trial.trial_id,
            trial.trial_tuple_sha256,
            1,
            retry_of=f"{trial.trial_id}.A0",
            supersedes_attempt=f"{trial.trial_id}.A0",
        )
        self.assertTrue(identity.attempt_id.endswith(".A1"))

    def test_paired_case_requires_pair_id_policy(self) -> None:
        with self.assertRaisesRegex(SchemaError, "pair_id_policy"):
            make_case(paired_condition_ids=("CONTROL", "TREATMENT"))

    def test_serialized_derived_denominator_fields_cannot_be_tampered(self) -> None:
        trial = make_trial()
        result = ResultEnvelope(
            attempt_id=f"{trial.trial_id}.A0",
            trial_tuple_sha256=trial.trial_tuple_sha256,
            formal_verdict=FormalVerdict.PASS,
            analytical_classification=AnalyticalClassification.PASS,
            primary_oracle="oracle",
            must_pass={"p": True},
            failed_predicates=(),
        ).to_dict()
        result["enters_capability_denominator"] = False
        with self.assertRaisesRegex(SchemaError, "denominator"):
            ResultEnvelope.from_dict(result)

    def test_multiple_failed_predicates_survive_canonical_key_sorting(self) -> None:
        trial = make_trial()
        result = ResultEnvelope(
            attempt_id=f"{trial.trial_id}.A0",
            trial_tuple_sha256=trial.trial_tuple_sha256,
            formal_verdict=FormalVerdict.FAIL,
            analytical_classification=AnalyticalClassification.INVALID_SETUP,
            primary_oracle="profile_aware_full_physical_artifact_audit",
            must_pass={
                "physical_execution_completed": False,
                "full_physical_artifact_audit": False,
            },
            failed_predicates=(
                "physical_execution_completed",
                "full_physical_artifact_audit",
            ),
            invalidity=InvalidityEvidence(
                reason_code="HARNESS_FAILURE",
                explanation="The harness could not produce complete evidence.",
                evidence_paths=("terminal.log",),
                repair="Repair the harness and allocate an exact retry.",
            ),
        )
        canonical_payload = json.loads(
            canonical_json_bytes(result.to_dict()).decode("utf-8")
        )
        self.assertEqual(
            tuple(canonical_payload["must_pass"]),
            ("full_physical_artifact_audit", "physical_execution_completed"),
        )
        restored = ResultEnvelope.from_dict(canonical_payload)
        self.assertEqual(restored.failed_predicates, result.failed_predicates)

    def test_failed_predicates_still_reject_duplicates_or_wrong_keys(self) -> None:
        trial = make_trial()
        for failed in (("a", "a"), ("a", "c"), ("a",)):
            with self.subTest(failed=failed), self.assertRaisesRegex(
                SchemaError, "exactly match"
            ):
                ResultEnvelope(
                    attempt_id=f"{trial.trial_id}.A0",
                    trial_tuple_sha256=trial.trial_tuple_sha256,
                    formal_verdict=FormalVerdict.FAIL,
                    analytical_classification=AnalyticalClassification.CAPABILITY_FAIL,
                    primary_oracle="oracle",
                    must_pass={"a": False, "b": False},
                    failed_predicates=failed,
                )


if __name__ == "__main__":
    unittest.main()
