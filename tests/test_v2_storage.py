from __future__ import annotations

from contextlib import chdir
from pathlib import Path
import tempfile
import unittest

from experiments_suite_v2.artifacts import verify_artifact_manifest, verify_run_checksums
from experiments_suite_v2.audit import audit_attempt, audit_run
from experiments_suite_v2.io import append_jsonl, atomic_write_json, atomic_write_text, load_json
from experiments_suite_v2.schemas import (
    AnalyticalClassification,
    ArtifactProfile,
    CaseDefinition,
    FormalVerdict,
    InvalidityEvidence,
    ResultEnvelope,
    TrialTuple,
    AttemptIdentity,
)
from experiments_suite_v2.storage import RunStore, StorageError


def make_trial(*, case_id: str = "C-HRI-ID") -> TrialTuple:
    return TrialTuple(
        suite_version="PrefMem-Experiment-Suite-v2",
        case_id=case_id,
        aim_id=case_id,
        prefix="C-HRI",
        condition_id="PRODUCTION",
        scene_state_id="NOT_APPLICABLE",
        reset_manifest_hash="NOT_APPLICABLE",
        scene_config_hash="NOT_APPLICABLE",
        robot_asset_id="NOT_APPLICABLE",
        language_context_id="01",
        exact_utterance="Which component may change my goal?",
        memory_store_snapshot_hash="NOT_APPLICABLE",
        model_config_hash="model-hash",
        prompt_tool_schema_hash="prompt-hash",
        artifact_profile=ArtifactProfile.MODEL_COMPONENT,
        random_seed=11,
        software_config_commit="commit",
        variant_id="01",
    )


def make_case(*, case_id: str = "C-HRI-ID") -> CaseDefinition:
    return CaseDefinition(
        suite_version="PrefMem-Experiment-Suite-v2",
        scene_version="NOT_APPLICABLE",
        robot_asset_id="NOT_APPLICABLE",
        case_id=case_id,
        aim_id=case_id,
        prefix="C-HRI",
        title="HRI identity boundary",
        target_boundary="DETERMINISTIC_HARNESS",
        live_components=("DETERMINISTIC_HARNESS",),
        input_spec={"utterance": "frozen"},
        typed_state_fixture="NOT_APPLICABLE",
        frame_fixture="NOT_APPLICABLE",
        memory_fixture="NOT_APPLICABLE",
        model_build_prompt_hashes={"fixture": "hash"},
        deterministic_settings={"seed": 11},
        seeds=(11,),
        permitted_variations={"variant": "01"},
        primary_oracle="claim_contract",
        must_pass=("truthful_authority",),
        should_measure=("latency_ms",),
        invalid_if=("endpoint_disconnect",),
        artifact_profile=ArtifactProfile.MODEL_COMPONENT,
    )


def fill_model_component_artifacts(attempt: Path) -> None:
    atomic_write_json(attempt / "environment.json", {"python": "test"}, overwrite=False)
    atomic_write_json(attempt / "setup.json", {"frozen": True}, overwrite=False)
    atomic_write_json(attempt / "inputs.json", {"inputs": ["fixture"]}, overwrite=False)
    atomic_write_text(attempt / "process.md", "# Frozen process\n", overwrite=False)
    atomic_write_text(attempt / "terminal.log", "fixture output\n", overwrite=False)
    atomic_write_text(attempt / "results.md", "# Result\n", overwrite=False)


def pass_result(trial: TrialTuple, attempt_id: str) -> ResultEnvelope:
    return ResultEnvelope(
        attempt_id=attempt_id,
        trial_tuple_sha256=trial.trial_tuple_sha256,
        formal_verdict=FormalVerdict.PASS,
        analytical_classification=AnalyticalClassification.PASS,
        primary_oracle="claim_contract",
        must_pass={"truthful_authority": True},
        failed_predicates=(),
        retry_lineage={"retry_of": None, "supersedes_attempt": None},
    )


class V2StorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.suite_root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_attempt_finalization_manifest_audit_and_run_checksums(self) -> None:
        store = RunStore(self.suite_root)
        run = store.create_run("dev-run", source_commit="abc", dirty_state={"dirty": False})
        trial = make_trial()
        attempt = store.allocate_attempt(
            "dev-run", trial, make_case(),
            learned_model_live=False, memory_live=False, physical_actions=False,
        )
        self.assertEqual(attempt.parent.name, "C-HRI")
        self.assertTrue(attempt.name.endswith("__A0"))
        store.start_attempt(attempt)
        fill_model_component_artifacts(attempt)
        identity = load_json(attempt / "attempt_manifest.json")["identity"]
        store.write_result(attempt, pass_result(trial, identity["attempt_id"]))
        store.finalize_attempt(attempt)

        self.assertEqual(verify_artifact_manifest(attempt), [])
        self.assertTrue(audit_attempt(attempt).passed)
        with self.assertRaisesRegex(StorageError, "immutable"):
            store.append_event(attempt, "LATE_EVENT", {})

        checksum_path = store.close_run("dev-run")
        self.assertEqual(checksum_path, run / "checksums.sha256")
        self.assertEqual(verify_run_checksums(run), [])
        self.assertTrue(audit_run(run, require_closed=True).passed)
        run_manifest = load_json(run / "run_manifest.json")
        sentinel = run_manifest["services"]["frozen_identities_and_probe_evidence"]["services"][0]["probe"]["sentinel"]
        self.assertEqual(sentinel, "SERVICE_OK")

    def test_relative_attempt_directory_finalizes_and_verifies(self) -> None:
        # Reproduce operator invocation from a working directory where the
        # RunStore root and returned attempt path are both relative.
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            with chdir(parent):
                store = RunStore(Path("suite"))
                store.create_run(
                    "relative-run", source_commit="abc", dirty_state={}
                )
                trial = make_trial()
                attempt = store.allocate_attempt(
                    "relative-run",
                    trial,
                    make_case(),
                    learned_model_live=False,
                    memory_live=False,
                    physical_actions=False,
                )
                self.assertFalse(attempt.is_absolute())
                store.start_attempt(attempt)
                fill_model_component_artifacts(attempt)
                identity = load_json(attempt / "attempt_manifest.json")["identity"]
                store.write_result(
                    attempt, pass_result(trial, identity["attempt_id"])
                )
                store.finalize_attempt(attempt)
                self.assertEqual(verify_artifact_manifest(attempt), [])
                checksum_path = store.close_run("relative-run")
                self.assertTrue(checksum_path.is_absolute())
                self.assertEqual(
                    verify_run_checksums(
                        Path("suite/results/shared-campaigns/relative-run")
                    ),
                    [],
                )

    def test_invalid_attempt_gets_exact_retry_but_capability_failure_does_not(self) -> None:
        store = RunStore(self.suite_root)
        store.create_run("retry-run", source_commit="abc", dirty_state={})
        trial = make_trial()
        attempt = store.allocate_attempt(
            "retry-run", trial, make_case(),
            learned_model_live=False, memory_live=False, physical_actions=False,
        )
        fill_model_component_artifacts(attempt)
        identity = load_json(attempt / "attempt_manifest.json")["identity"]
        invalid = ResultEnvelope(
            attempt_id=identity["attempt_id"],
            trial_tuple_sha256=trial.trial_tuple_sha256,
            formal_verdict=FormalVerdict.FAIL,
            analytical_classification=AnalyticalClassification.INVALID_RUN,
            primary_oracle="claim_contract",
            must_pass={"truthful_authority": False},
            failed_predicates=("truthful_authority",),
            invalidity=InvalidityEvidence(
                reason_code="ENDPOINT_DISCONNECT",
                explanation="Connection ended before a usable response.",
                evidence_paths=("terminal.log",),
                repair="Restore the exact endpoint.",
            ),
            retry_lineage={"retry_of": None, "supersedes_attempt": None},
        )
        store.write_result(attempt, invalid)
        store.finalize_attempt(attempt)
        retry = store.allocate_retry(attempt)
        retry_identity = load_json(retry / "attempt_manifest.json")["identity"]
        self.assertEqual(retry_identity["attempt_number"], 1)
        self.assertEqual(retry_identity["retry_of"], identity["attempt_id"])
        self.assertEqual(retry_identity["supersedes_attempt"], identity["attempt_id"])
        self.assertEqual(
            load_json(retry / "attempt_manifest.json")["trial_tuple"],
            load_json(attempt / "attempt_manifest.json")["trial_tuple"],
        )
        ledger = load_json(
            self.suite_root
            / "results/shared-campaigns/retry-run/invalid_retry_ledger.json"
        )
        self.assertEqual(ledger["entries"][0]["retry_attempt_id"], retry_identity["attempt_id"])

        second_trial = make_trial(case_id="C-HRI-STATE")
        second = store.allocate_attempt(
            "retry-run", second_trial, make_case(case_id="C-HRI-STATE"),
            learned_model_live=False, memory_live=False, physical_actions=False,
        )
        fill_model_component_artifacts(second)
        second_id = load_json(second / "attempt_manifest.json")["identity"]["attempt_id"]
        capability_fail = ResultEnvelope(
            attempt_id=second_id,
            trial_tuple_sha256=second_trial.trial_tuple_sha256,
            formal_verdict=FormalVerdict.FAIL,
            analytical_classification=AnalyticalClassification.CAPABILITY_FAIL,
            primary_oracle="claim_contract",
            must_pass={"truthful_authority": False},
            failed_predicates=("truthful_authority",),
            retry_lineage={"retry_of": None, "supersedes_attempt": None},
        )
        store.write_result(second, capability_fail)
        store.finalize_attempt(second)
        with self.assertRaisesRegex(StorageError, "cannot be replaced"):
            store.allocate_retry(second)

    def test_manifest_detects_post_finalization_tampering(self) -> None:
        store = RunStore(self.suite_root)
        store.create_run("tamper-run", source_commit="abc", dirty_state={})
        trial = make_trial()
        attempt = store.allocate_attempt(
            "tamper-run", trial, make_case(),
            learned_model_live=False, memory_live=False, physical_actions=False,
        )
        fill_model_component_artifacts(attempt)
        identity = load_json(attempt / "attempt_manifest.json")["identity"]
        store.write_result(attempt, pass_result(trial, identity["attempt_id"]))
        store.finalize_attempt(attempt)
        (attempt / "terminal.log").write_text("tampered\n", encoding="utf-8")
        issues = verify_artifact_manifest(attempt)
        self.assertTrue(any(issue.code in {"HASH_MISMATCH", "SIZE_MISMATCH"} for issue in issues))

    def test_confirmatory_run_cannot_be_allocated_before_protocol_freeze(self) -> None:
        with self.assertRaisesRegex(StorageError, "not authorized"):
            RunStore(self.suite_root).create_run(
                "confirmatory", source_commit="abc", dirty_state={}, run_kind="CONFIRMATORY",
            )

    def test_recovery_completes_only_interrupted_finalization(self) -> None:
        from unittest.mock import patch

        store = RunStore(self.suite_root)
        store.create_run("recovery-run", source_commit="abc", dirty_state={})
        trial = make_trial()
        attempt = store.allocate_attempt(
            "recovery-run", trial, make_case(),
            learned_model_live=False, memory_live=False, physical_actions=False,
        )
        fill_model_component_artifacts(attempt)
        identity = load_json(attempt / "attempt_manifest.json")["identity"]
        store.write_result(attempt, pass_result(trial, identity["attempt_id"]))
        with patch("experiments_suite_v2.storage.build_artifact_manifest", side_effect=RuntimeError("simulated crash")):
            with self.assertRaisesRegex(RuntimeError, "simulated crash"):
                store.finalize_attempt(attempt)
        self.assertEqual(load_json(attempt / "attempt_manifest.json")["status"], "FINALIZED")
        self.assertFalse((attempt / "artifact_manifest.json").exists())
        self.assertEqual(store.discover_recovery_candidates("recovery-run")[0]["action"], "RECOVER_FINALIZATION")
        store.recover_finalization(attempt)
        self.assertTrue(audit_attempt(attempt).passed)

    def test_batch_profile_validates_row_level_trial_units(self) -> None:
        store = RunStore(self.suite_root)
        store.create_run("batch-run", source_commit="abc", dirty_state={})
        container_trial = make_trial().to_dict()
        container_trial.update({
            "case_id": "AB-MEM-Q",
            "aim_id": "AB-MEM-Q",
            "prefix": "AB-MEM-Q",
            "condition_id": "GRID-BATCH",
            "artifact_profile": "BATCH_EVAL",
            "language_context_id": "BATCH",
            "variant_id": "BATCH",
        })
        container = TrialTuple.from_dict(container_trial)
        case = make_case().to_dict()
        case.update({
            "case_id": "AB-MEM-Q",
            "aim_id": "AB-MEM-Q",
            "prefix": "AB-MEM-Q",
            "title": "Memory retrieval grid",
            "artifact_profile": "BATCH_EVAL",
        })
        batch_case = CaseDefinition.from_dict(case)
        attempt = store.allocate_attempt(
            "batch-run", container, batch_case,
            learned_model_live=False, memory_live=False, physical_actions=False,
        )
        atomic_write_json(attempt / "environment.json", {"python": "test"}, overwrite=False)
        append_jsonl(attempt / "retrieval_trace.jsonl", {"row_id": "row-1", "trace": "fixture"})

        row_trial_data = container.to_dict()
        row_trial_data.update({
            "condition_id": "q1-k1-c1-n5",
            "language_context_id": "need-001",
            "variant_id": "cell-001",
        })
        row_trial = TrialTuple.from_dict(row_trial_data)
        row_identity = AttemptIdentity(row_trial.trial_id, row_trial.trial_tuple_sha256, 0)
        append_jsonl(attempt / "input_rows.jsonl", {
            "row_id": "row-1",
            "trial_tuple": row_trial.to_dict(),
            "identity": row_identity.to_dict(),
            "retrieval_input": {"queries": ["frozen"]},
        })
        row_result = ResultEnvelope(
            attempt_id=row_identity.attempt_id,
            trial_tuple_sha256=row_trial.trial_tuple_sha256,
            formal_verdict=FormalVerdict.PASS,
            analytical_classification=AnalyticalClassification.PASS,
            primary_oracle="required_memory_recall_at_cap",
            must_pass={"required_memory_recalled": True},
            failed_predicates=(),
            retry_lineage={"retry_of": None, "supersedes_attempt": None},
        )
        append_jsonl(attempt / "result_rows.jsonl", {"row_id": "row-1", "result": row_result.to_dict()})
        store.finalize_attempt(attempt)
        manifest = load_json(attempt / "attempt_manifest.json")
        self.assertTrue(manifest["result_summary"]["batch_bundle"])
        self.assertEqual(manifest["result_summary"]["row_count"], 1)
        self.assertEqual(manifest["result_summary"]["outcome_counts"]["PASS"], 1)


if __name__ == "__main__":
    unittest.main()
