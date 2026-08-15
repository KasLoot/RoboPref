from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from experiments_suite_v2.io import atomic_write_bytes, atomic_write_json, append_jsonl, load_json
from experiments_suite_v2.runners.bundles import update_bundle_attempt
from experiments_suite_v2.runners import cal_x03_five_tile_winner as diagnostic
from prefmem.execution.sam import SamDetection


def _detection(score: float | None, area: int, object_id: int = 0) -> SamDetection:
    mask = np.zeros((8, 8), dtype=bool)
    mask.reshape(-1)[: min(area, 64)] = True
    return SamDetection(
        object_id=object_id,
        box_xyxy=(0, 0, 8, 8),
        mask_area=area,
        mask=mask,
        score=score,
    )


class FiveTileWinnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = diagnostic._input_rows()

    def test_frozen_schedule_is_exact_row_major_36_by_5_and_retry_ids_are_fresh(self) -> None:
        a0 = diagnostic._call_contracts(
            self.rows, attempt_id="CAL-X03-FIVE-TILE-WINNER-v1-A0"
        )
        a1 = diagnostic._call_contracts(
            self.rows, attempt_id="CAL-X03-FIVE-TILE-WINNER-v1-A1"
        )
        self.assertEqual(len(a0), 180)
        self.assertEqual(len({item["request_id"] for item in a0}), 180)
        self.assertFalse(
            {item["request_id"] for item in a0}
            & {item["request_id"] for item in a1}
        )
        self.assertEqual(
            diagnostic.canonical_sha256(a0),
            load_json(diagnostic.PROTOCOL_PATH)["dispatch_schedule"][
                "full_request_contract_schedule_canonical_sha256"
            ],
        )
        for row_ordinal in range(36):
            block = a0[row_ordinal * 5 : row_ordinal * 5 + 5]
            self.assertEqual(
                [item["tile_id"] for item in block],
                ["T00", "T01", "T10", "T11", "TC"],
            )
            self.assertEqual(
                [item["dispatch_ordinal"] for item in block],
                list(range(row_ordinal * 5, row_ordinal * 5 + 5)),
            )

    def test_real_manifest_final_transition_preserves_180_raw_call_unit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            final_request_id = "FTWIN-A1-MSDEV-035--TC"
            atomic_write_json(
                root / "attempt_manifest.json",
                {
                    "status": "AWAITING_PRODUCTION_POST_EVIDENCE",
                    "completed_rows": 180,
                    "completed_raw_calls": 180,
                    "completed_condition_rows": 36,
                    "last_completed_row_id": final_request_id,
                },
            )
            atomic_write_json(
                root / "request_index.json",
                {
                    "records": [
                        {
                            "request_id": (
                                final_request_id
                                if index == 179
                                else f"FTWIN-A1-FIXTURE-{index:03d}"
                            )
                        }
                        for index in range(180)
                    ]
                },
            )
            manifest = diagnostic._transition_valid_attempt_finalized(
                root,
                completed_at="2026-08-13T01:00:00Z",
                recommendation="NO_FIVE_TILE_WINNER_CANDIDATE_TERMINAL",
            )
            self.assertEqual(manifest["status"], "FINALIZED")
            self.assertEqual(manifest["completed_rows"], 180)
            self.assertEqual(manifest["completed_raw_calls"], 180)
            self.assertEqual(manifest["completed_condition_rows"], 36)
            self.assertEqual(manifest["last_completed_row_id"], final_request_id)

    def test_winner_zero_one_null_margin_boundary_and_area_tie_break(self) -> None:
        forwarded, decision = diagnostic.select_response_only_winner([])
        self.assertEqual(forwarded, ())
        self.assertEqual(decision["disposition"], "NO_COMPONENTS")
        sole = _detection(None, 2)
        forwarded, decision = diagnostic.select_response_only_winner([sole])
        self.assertEqual(forwarded, (sole,))
        self.assertEqual(decision["disposition"], "SOLE_COMPONENT_FORWARDED")
        forwarded, decision = diagnostic.select_response_only_winner(
            [_detection(None, 9), _detection(0.8, 2)]
        )
        self.assertEqual(forwarded, ())
        self.assertEqual(decision["disposition"], "AMBIGUOUS_UNRANKED")
        forwarded, decision = diagnostic.select_response_only_winner(
            [_detection(0.749999, 5), _detection(0.70, 4)]
        )
        self.assertEqual(forwarded, ())
        self.assertLess(decision["top_two_score_gap"], 0.05)
        forwarded, decision = diagnostic.select_response_only_winner(
            [_detection(0.75, 5, 1), _detection(0.70, 4, 2)]
        )
        self.assertEqual(len(forwarded), 1)
        self.assertEqual(forwarded[0].object_id, 1)
        self.assertGreaterEqual(decision["top_two_score_gap"], 0.05)
        forwarded, decision = diagnostic.select_response_only_winner(
            [_detection(0.8, 3, 1), _detection(0.8, 7, 2)]
        )
        self.assertEqual(decision["ranked_components"][0]["object_id"], 2)
        self.assertEqual(forwarded, ())
        self.assertEqual(decision["disposition"], "AMBIGUOUS_SCORE_MARGIN")

    def test_visible_payload_derivation_never_loads_scorer_oracles_and_dummy_is_invariant(self) -> None:
        row = dict(self.rows[0])
        contract = diagnostic._call_contracts(
            self.rows, attempt_id="CAL-X03-FIVE-TILE-WINNER-v1-A0"
        )[4]
        dummy = dict(row)
        dummy["scorer_oracle_masks"] = {
            name: {"shape": [768, 768], "fill": False}
            for name in row["scorer_oracle_masks"]
        }
        with mock.patch.object(
            diagnostic.multiscale,
            "_load_observation_and_masks",
            side_effect=AssertionError("oracle loaded before five receipts"),
        ):
            _crop_a, payload_a, _decoded_a = diagnostic._derive_payload_bytes(
                contract, row
            )
            _crop_b, payload_b, _decoded_b = diagnostic._derive_payload_bytes(
                contract, dummy
            )
        self.assertEqual(hashlib.sha256(payload_a).digest(), hashlib.sha256(payload_b).digest())
        dummy_rows = [dummy, *self.rows[1:]]
        self.assertEqual(
            diagnostic._call_contracts(self.rows, attempt_id="X-A0"),
            diagnostic._call_contracts(dummy_rows, attempt_id="X-A0"),
        )

    def test_http_classification_count_zero_malformed_4xx_and_5xx(self) -> None:
        row = self.rows[0]
        contract = diagnostic._call_contracts(self.rows, attempt_id="X-A0")[0]
        contract = {**contract, "payload_sha256": "0" * 64}
        valid = json.dumps(
            {
                "prompt": contract["prompt"],
                "threshold": 0.5,
                "image": {"width": 512, "height": 512},
                "count": 0,
                "detections": [],
            }
        ).encode()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            detections, response = diagnostic._normalise_response(
                root, contract, valid, 200, persist_masks=True
            )
            self.assertEqual(detections, ())
            self.assertFalse(response["capability_failure"])
            _detections, malformed = diagnostic._normalise_response(
                root, contract, b"not-json", 200, persist_masks=True
            )
            self.assertTrue(malformed["capability_failure"])
            with self.assertRaisesRegex(diagnostic.AttemptInvalidity, "HTTP 400"):
                diagnostic._normalise_response(root, contract, b"{}", 400, persist_masks=True)
            with self.assertRaisesRegex(diagnostic.AttemptInvalidity, "HTTP 503"):
                diagnostic._normalise_response(root, contract, b"{}", 503, persist_masks=True)

    def test_strict_offline_receipt_rejects_receipt_selected_raw_path(self) -> None:
        row = self.rows[0]
        contract = diagnostic._call_contracts(self.rows, attempt_id="X-A0")[0]
        contract = {
            **contract,
            "payload_sha256": "a" * 64,
            "payload_size_bytes": 123,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atomic_write_json(
                root / "inputs.json",
                {"production_pre_evidence": {"derived_fields_sha256": "b" * 64}},
            )
            ordinal = int(contract["dispatch_ordinal"])
            stem = f"{ordinal:03d}-{contract['request_id']}"
            intent_path = root / "post_intents" / f"{stem}.json"
            intent = {
                "schema_version": f"{diagnostic.SCHEMA}.post-intent",
                "request_id": contract["request_id"],
                "dispatch_ordinal": ordinal,
                "method": "POST",
                "url": contract["endpoint"],
                "payload_sha256": contract["payload_sha256"],
                "payload_size_bytes": contract["payload_size_bytes"],
                "filename": contract["filename"],
                "mime_type": contract["mime_type"],
                "prompt_utf8_sha256": contract["prompt_utf8_sha256"],
                "threshold_form": "0.5",
                "production_pre_derived_fields_sha256": "b" * 64,
                "recorded_before_invocation": True,
                "recorded_at": "2026-08-13T00:00:00Z",
            }
            atomic_write_json(intent_path, intent)
            raw = b"{}"
            expected_raw = diagnostic._response_root(root, contract) / "http_response.bin"
            atomic_write_bytes(expected_raw, raw)
            receipt_path = root / "post_receipts" / f"{stem}.json"
            receipt = {
                "schema_version": f"{diagnostic.SCHEMA}.post-receipt",
                "request_id": contract["request_id"],
                "dispatch_ordinal": ordinal,
                "status_code": 200,
                "headers": {},
                "latency_seconds": 0.1,
                "raw_response_path": "responses/attacker-selected.bin",
                "raw_response_sha256": hashlib.sha256(raw).hexdigest(),
                "post_intent_path": intent_path.relative_to(root).as_posix(),
                "post_intent_sha256": diagnostic.sha256_file(intent_path),
                "recorded_immediately_after_return": True,
                "recorded_at": "2026-08-13T00:00:01Z",
            }
            atomic_write_json(receipt_path, receipt)
            with self.assertRaisesRegex(diagnostic.DiagnosticError, "semantics or linkage"):
                diagnostic._reparse_response(root, contract)

    def test_counterfactual_is_a_required_eligibility_gate_and_a2_profile_is_exact(self) -> None:
        rows = []
        control_collisions_left = 4
        for source in self.rows:
            row_id = str(source["diagnostic_row_id"])
            a2_intended = row_id not in {"MSDEV-016", "MSDEV-018", "MSDEV-032", "MSDEV-034"}
            a2_collision = source["cohort"] == "STABLE_CONTROL" and control_collisions_left > 0
            if a2_collision:
                control_collisions_left -= 1
            rows.append(
                {
                    "row_id": row_id,
                    "diagnostic_row_id": row_id,
                    "unique_image_prompt_cell_id": source["unique_image_prompt_cell_id"],
                    "repeat_index": source["repeat_index"],
                    "cohort": source["cohort"],
                    "production_selected_safe_intended_grounding": True,
                    "sam_anchor_error_mm": [0.0, 0.0, 0.0],
                    "sam_radial_xy_error_mm": 0.0,
                    "usable_response_count": 5,
                    "capability_failure_count": 0,
                    "winner_ambiguous": False,
                    "forwarded_candidate_count": 1,
                    "forwarded_any_candidate_cross_object_collision": False,
                    "forwarded_selected_candidate_cross_object_collision": False,
                    "raw_merged_any_candidate_cross_object_collision": a2_collision,
                    "a2_corner_response_reproduction_count": 4,
                    "a2_four_tile_reproduction": {
                        "pass": True,
                        "actual_profile": {
                            "production_grounding_succeeded": a2_intended,
                            "production_selected_intended_iou_pass": a2_intended,
                            "any_candidate_cross_object_collision": a2_collision,
                            "selected_candidate_cross_object_collision": False,
                            "ambiguous_selection_failure": False,
                        },
                    },
                }
            )
        stability = {"pass": True}
        absent = diagnostic.aggregate_rows(rows, stability=stability)
        self.assertFalse(absent["five_tile_winner_candidate_eligible"])
        self.assertFalse(
            diagnostic.aggregate_rows(
                rows, stability=stability, oracle_counterfactual={"pass": False}
            )["five_tile_winner_candidate_eligible"]
        )
        passed = diagnostic.aggregate_rows(
            rows, stability=stability, oracle_counterfactual={"pass": True}
        )
        self.assertTrue(passed["a2_four_tile_exact_reproduction_pass"])
        self.assertTrue(passed["five_tile_winner_candidate_eligible"])
        self.assertEqual(
            passed["recommendation"], "FIVE_TILE_WINNER_DEVELOPMENT_CANDIDATE_ONLY"
        )

    def test_freeze_drift_fails_closed_and_sealed_manifest_must_be_finalized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fake = Path(directory) / "freeze.json"
            atomic_write_json(
                fake,
                {
                    "schema_version": f"{diagnostic.SCHEMA}.implementation-freeze",
                    "status": "FROZEN_NOT_ALLOCATED",
                    "files": [],
                    "files_canonical_sha256": diagnostic.canonical_sha256([]),
                },
            )
            with mock.patch.object(diagnostic, "FREEZE_PATH", fake):
                with self.assertRaisesRegex(diagnostic.DiagnosticError, "contract differs"):
                    diagnostic._verify_implementation_freeze()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atomic_write_json(root / "attempt_manifest.json", {"status": "RUNNING"})
            with mock.patch.object(diagnostic, "verify_row_bundle", return_value=[]):
                issues = diagnostic.verify_attempt_semantics(root, sealed=True)
            self.assertIn("sealed attempt manifest status is not FINALIZED", issues)

    def test_terminal_failure_preserves_no_resume_and_full_fresh_retry_rule(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atomic_write_json(root / "attempt_manifest.json", {"status": "LIVE_DISPATCH_RUNNING"})
            with mock.patch.object(diagnostic, "update_bundle_attempt"):
                diagnostic._terminal_invalid(
                    root,
                    classification="INVALID_SETUP",
                    reason="injected scorer crash",
                    confirmed_calls=180,
                )
            failure = load_json(root / "failure.json")
            self.assertFalse(failure["same_attempt_resume"])
            self.assertEqual(failure["confirmed_physical_post_attempts"], 180)
            self.assertIn("call 1/180", failure["retry"])

    def test_zero_network_abandoned_intent_terminalizer_counts_one_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contracts = diagnostic._call_contracts(self.rows, attempt_id="X-A0")
            request_id = contracts[0]["request_id"]
            intent = root / "post_intents" / f"000-{request_id}.json"
            atomic_write_json(intent, {"request_id": request_id})
            atomic_write_json(
                root / "attempt_manifest.json",
                {
                    "status": "LIVE_DISPATCH_RUNNING",
                    "attempt_id": "CAL-X03-FIVE-TILE-WINNER-v1-A0",
                    "attempt_number": 0,
                },
            )
            atomic_write_json(root / "request_index.json", {"records": contracts})
            for row in self.rows:
                append_jsonl(root / "input_rows.jsonl", row)
            with mock.patch.object(diagnostic, "_validate_allocation_lock"):
                failure = diagnostic.classify_abandoned_partial(root)
            self.assertEqual(failure["classification"], "INVALID_RUN")
            self.assertEqual(failure["confirmed_physical_post_attempts"], 0)
            self.assertEqual(failure["unknown_physical_post_outcomes"], 1)
            self.assertFalse(failure["same_attempt_resume"])
            self.assertIn("call 1/180", failure["retry"])
            a1 = diagnostic._call_contracts(
                self.rows, attempt_id="CAL-X03-FIVE-TILE-WINNER-v1-A1"
            )
            a0_ids = {item["request_id"] for item in contracts}
            a1_ids = {item["request_id"] for item in a1}
            self.assertEqual(len(a1), 180)
            self.assertFalse(a0_ids & a1_ids)
            # Build a real allocation-time snapshot closure without consulting
            # current sources during invalid sealing.
            inputs = {
                "runner": {"sha256": "historical-runner"},
                "protocol": {"sha256": "historical-protocol"},
                "implementation_freeze": {"sha256": "historical-freeze"},
            }
            manifest = load_json(root / "attempt_manifest.json")
            manifest["frozen"] = {
                "inputs_sha256": diagnostic.canonical_sha256(inputs),
                "input_rows_sha256": diagnostic.sha256_file(root / "input_rows.jsonl"),
                "row_count": len(self.rows),
            }
            atomic_write_json(root / "attempt_manifest.json", manifest, overwrite=True)
            atomic_write_json(root / "inputs.json", inputs)
            protocol_snapshot = root / "protocol_snapshot" / diagnostic.PROTOCOL_PATH.name
            freeze_snapshot = root / "protocol_snapshot" / diagnostic.FREEZE_PATH.name
            atomic_write_bytes(protocol_snapshot, b"historical protocol")
            inputs["protocol"]["sha256"] = diagnostic.sha256_file(protocol_snapshot)
            allocated_files = [
                {
                    "path": "runners/cal_x03_five_tile_winner.py",
                    "sha256": "historical-runner",
                }
            ]
            freeze_value = {
                "files": allocated_files,
                "files_canonical_sha256": diagnostic.canonical_sha256(allocated_files),
            }
            atomic_write_json(freeze_snapshot, freeze_value)
            inputs["implementation_freeze"]["sha256"] = diagnostic.sha256_file(freeze_snapshot)
            atomic_write_json(root / "inputs.json", inputs, overwrite=True)
            manifest = load_json(root / "attempt_manifest.json")
            manifest["frozen"]["inputs_sha256"] = diagnostic.canonical_sha256(inputs)
            atomic_write_json(root / "attempt_manifest.json", manifest, overwrite=True)
            diagnostic.seal_invalid_attempt(root)
            self.assertEqual(diagnostic.verify_invalid_attempt_semantics(root, sealed=True), [])
            with (
                mock.patch.object(
                    diagnostic, "canonical_sha256", wraps=diagnostic.canonical_sha256
                ),
            ):
                lineage = diagnostic._validate_retry(
                    attempt_id="CAL-X03-FIVE-TILE-WINNER-v1-A1",
                    attempt_number=1,
                    retry_of="CAL-X03-FIVE-TILE-WINNER-v1-A0",
                    supersedes_attempt="CAL-X03-FIVE-TILE-WINNER-v1-A0",
                    predecessor=root,
                    rows=self.rows,
                    contracts=a1,
                )
            self.assertEqual(lineage["shared_request_id_count"], 0)
            self.assertEqual(lineage["response_artifacts_reused"], 0)

    def test_finalized_before_seal_crash_recovers_offline_on_second_finalize(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            post_path = root / "fresh-post-source.json"
            atomic_write_json(post_path, {"fresh": True})
            atomic_write_json(root / "production_pre_evidence.json", {})
            append_jsonl(root / "input_rows.jsonl", {"row_id": "fixture"})
            atomic_write_json(
                root / "request_index.json",
                {
                    "records": [
                        {"request_id": f"FTWIN-A0-FIXTURE-{index:03d}"}
                        for index in range(180)
                    ]
                },
            )
            atomic_write_json(
                root / "attempt_manifest.json",
                {
                    "status": "AWAITING_PRODUCTION_POST_EVIDENCE",
                    "completed_raw_calls": 180,
                    "attempt_id": "CAL-X03-FIVE-TILE-WINNER-v1-A0",
                },
            )
            stability = {"pass": True}
            candidate = {
                "five_tile_winner_candidate_eligible": False,
                "recommendation": "NO_FIVE_TILE_WINNER_CANDIDATE_TERMINAL",
            }
            atomic_write_json(root / "duplicate_stability.json", stability)
            atomic_write_json(root / "summary_candidate.json", candidate)
            atomic_write_json(root / "oracle_counterfactual.json", {"pass": True})
            identity = {key: f"stable-{index}" for index, key in enumerate(diagnostic.PRODUCTION_IMMUTABLE_FIELDS)}
            pre = {**identity, "derived_fields_sha256": "a" * 64}
            post = {**identity, "derived_fields_sha256": "b" * 64}
            results = [{"row_id": "MSDEV-034"}]
            real_finalize_bundle = diagnostic.finalize_bundle_files
            real_immutable_json = diagnostic._immutable_json
            seal_calls = 0
            partial_summary_crashed = False
            manifest_transition_crashed = False

            def update_manifest(path, **kwargs):
                nonlocal manifest_transition_crashed
                if (
                    kwargs.get("status") == "FINALIZED"
                    and not manifest_transition_crashed
                ):
                    manifest_transition_crashed = True
                    raise SystemExit("injected hard crash after results before FINALIZED")
                value = load_json(Path(path) / "attempt_manifest.json")
                value.update(kwargs.get("extra", {}))
                if "status" in kwargs:
                    value["status"] = kwargs["status"]
                if "completed_rows" in kwargs:
                    value["completed_rows"] = kwargs["completed_rows"]
                atomic_write_json(Path(path) / "attempt_manifest.json", value, overwrite=True)

            def flaky_seal(path, *, schema_version):
                nonlocal seal_calls
                seal_calls += 1
                if seal_calls == 1:
                    raise RuntimeError("injected crash after FINALIZED")
                return real_finalize_bundle(path, schema_version=schema_version)

            def crash_after_summary(path, value):
                nonlocal partial_summary_crashed
                if Path(path).name == "results.json" and not partial_summary_crashed:
                    partial_summary_crashed = True
                    raise SystemExit("injected hard crash after immutable summary")
                return real_immutable_json(path, value)

            with (
                mock.patch.object(diagnostic, "validate_precommit"),
                mock.patch.object(diagnostic, "_validate_allocation_lock"),
                mock.patch.object(diagnostic, "validate_production_evidence", return_value=pre),
                mock.patch.object(diagnostic, "_load_production_evidence", return_value=(post, "c" * 64)),
                mock.patch.object(diagnostic, "_require_fresh_post_evidence"),
                mock.patch.object(diagnostic, "_offline_rederive", return_value=([], results)),
                mock.patch.object(diagnostic, "duplicate_stability", return_value=stability),
                mock.patch.object(diagnostic, "oracle_counterfactual_proof", return_value={"pass": True}),
                mock.patch.object(diagnostic, "aggregate_rows", return_value=candidate),
                mock.patch.object(diagnostic, "verify_attempt_semantics", return_value=[]),
                mock.patch.object(diagnostic, "update_bundle_attempt", side_effect=update_manifest),
                mock.patch.object(diagnostic, "finalize_bundle_files", side_effect=flaky_seal),
                mock.patch.object(diagnostic, "_immutable_json", side_effect=crash_after_summary),
            ):
                with self.assertRaisesRegex(SystemExit, "hard crash"):
                    diagnostic.finalize_run(root, post_path)
                self.assertEqual(
                    load_json(root / "attempt_manifest.json")["status"],
                    "AWAITING_PRODUCTION_POST_EVIDENCE",
                )
                first_completed_at = load_json(root / "summary.json")["completed_at"]
                self.assertEqual(
                    first_completed_at,
                    load_json(root / "finalization_precommit.json")["completed_at"],
                )
                self.assertFalse((root / "failure.json").exists())
                with self.assertRaisesRegex(SystemExit, "before FINALIZED"):
                    diagnostic.finalize_run(root, post_path)
                self.assertTrue((root / "results.json").is_file())
                self.assertEqual(
                    load_json(root / "attempt_manifest.json")["status"],
                    "AWAITING_PRODUCTION_POST_EVIDENCE",
                )
                self.assertEqual(
                    load_json(root / "summary.json")["completed_at"], first_completed_at
                )
                self.assertFalse((root / "failure.json").exists())
                with self.assertRaisesRegex(
                    diagnostic.DiagnosticError, "rerun offline finalize"
                ):
                    diagnostic.finalize_run(root, post_path)
                self.assertEqual(
                    load_json(root / "summary.json")["completed_at"], first_completed_at
                )
                self.assertEqual(
                    load_json(root / "attempt_manifest.json")["status"], "FINALIZED"
                )
                self.assertFalse((root / "failure.json").exists())
                summary = diagnostic.finalize_run(root, post_path)
            self.assertEqual(summary["recommendation"], candidate["recommendation"])
            self.assertEqual(seal_calls, 2)
            self.assertTrue((root / "artifact_manifest.json").is_file())
            self.assertTrue((root / "checksums.sha256").is_file())
            self.assertEqual(
                load_json(root / "attempt_manifest.json")["status"], "FINALIZED"
            )


if __name__ == "__main__":
    unittest.main()
