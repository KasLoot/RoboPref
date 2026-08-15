from __future__ import annotations

import base64
import json
import tempfile
from types import MappingProxyType, SimpleNamespace
import unittest
from pathlib import Path
from unittest import mock

import cv2
import httpx
import numpy as np

from experiments_suite_v2.runners import cal_x03_multiscale_diagnostic as diagnostic
from prefmem.execution.sam import SamDetection


def _detection(mask: np.ndarray, *, score: float = 0.9, object_id: int = 0) -> SamDetection:
    rows, columns = np.nonzero(mask)
    return SamDetection(
        object_id=object_id,
        box_xyxy=(
            int(columns.min()),
            int(rows.min()),
            int(columns.max()) + 1,
            int(rows.max()) + 1,
        ),
        mask_area=int(mask.sum()),
        mask=mask,
        score=score,
    )


class _Response:
    def __init__(self, status_code: int, content: bytes) -> None:
        self.status_code = status_code
        self.content = content


class _ScriptedClient:
    def __init__(self, posts=(), *, health: dict[str, object] | None = None) -> None:
        self.actions = list(posts)
        self.health = health
        self.posted: list[dict[str, object]] = []
        self.health_gets = 0

    def post(self, url: str, *, data, files):
        self.posted.append({"url": url, "data": data, "files": files})
        action = self.actions.pop(0)
        if isinstance(action, Exception):
            raise action
        return action

    def get(self, url: str):
        self.health_gets += 1
        return _Response(200, json.dumps(self.health).encode("utf-8"))


class CalX03MultiscaleContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = diagnostic._input_rows()
        cls.contracts = diagnostic._call_contracts(cls.rows)

    def _prepared_contract(self, output: Path) -> tuple[dict[str, object], bytes]:
        contract = dict(self.contracts[0])
        payload = b"exact-precommitted-jpeg-bytes"
        path = output / str(contract["payload_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        contract["payload_sha256"] = diagnostic.sha256_file(path)
        contract["payload_size_bytes"] = len(payload)
        return contract, payload

    @staticmethod
    def _sam_body(
        contract: dict[str, object], detections: list[dict[str, object]] | None = None
    ) -> bytes:
        values = [] if detections is None else detections
        return json.dumps(
            {
                "image": {
                    "width": contract["expected_width"],
                    "height": contract["expected_height"],
                },
                "count": len(values),
                "detections": values,
            }
        ).encode("utf-8")

    def test_frozen_split_and_call_budget(self) -> None:
        self.assertEqual(len(self.rows), 36)
        self.assertEqual(sum(row["cohort"] == "RESIDUAL" for row in self.rows), 18)
        self.assertEqual(
            sum(
                row["cohort"] == "STABLE_CONTROL"
                and row["immutable_transition"] == "STABLE_SUCCESS"
                for row in self.rows
            ),
            18,
        )
        self.assertEqual(len({row["source_row_id"] for row in self.rows}), 36)
        self.assertEqual(len(self.contracts), 180)
        self.assertEqual(len({item["request_id"] for item in self.contracts}), 180)
        self.assertTrue(
            all(item["request_id"].startswith("MSCALL-A2-") for item in self.contracts)
        )
        a1_ids = {
            item["request_id"]
            for item in diagnostic.load_json(
                diagnostic.INVALID_A1_OUTPUT / "partial_response_index.json"
            )["responses"]
        }
        self.assertFalse(a1_ids & {item["request_id"] for item in self.contracts})
        self.assertEqual(sum(item["condition"] == "FULL_768" for item in self.contracts), 36)
        self.assertEqual(
            sum(item["condition"] == "TILED_512_O256" for item in self.contracts),
            144,
        )
        for row in self.rows:
            selected = [
                item
                for item in self.contracts
                if item["diagnostic_row_id"] == row["diagnostic_row_id"]
            ]
            self.assertEqual(
                [item["call_label"] for item in selected],
                ["FULL", "T00", "T01", "T10", "T11"],
            )

    def test_tile_inverse_map_area_box_and_seam_provenance(self) -> None:
        contract = next(item for item in self.contracts if item["tile_id"] == "T11")
        local = np.zeros((512, 512), dtype=bool)
        local[0:10, 0:20] = True
        mapped = diagnostic.map_tile_detections(contract, (_detection(local),))
        self.assertEqual(len(mapped), 1)
        candidate = mapped[0]
        self.assertEqual(candidate.local_mask_area, 200)
        self.assertEqual(candidate.global_mask_area, 200)
        self.assertEqual(candidate.global_box_xyxy, (256, 256, 276, 266))
        self.assertTrue(candidate.seam["truncated_at_internal_seam"])
        self.assertEqual(
            candidate.seam["edge_classification"]["left"], "INTERNAL_TILE_SEAM"
        )
        self.assertEqual(
            candidate.seam["edge_classification"]["top"], "INTERNAL_TILE_SEAM"
        )
        self.assertEqual(
            candidate.seam["edge_classification"]["right"], "GLOBAL_IMAGE_BOUNDARY"
        )

    def test_null_mask_is_capability_failure_with_no_box_fallback(self) -> None:
        contract = next(item for item in self.contracts if item["tile_id"] == "T00")
        missing = SamDetection(
            object_id=0,
            box_xyxy=(10, 10, 20, 20),
            mask_area=100,
            mask=None,
            score=0.9,
        )
        with self.assertRaises(diagnostic.MaskCapabilityFailure):
            diagnostic.map_tile_detections(contract, (missing,))

    def test_exact_bytes_raw_receipt_cache_and_parse_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "A1"
            contract, payload = self._prepared_contract(output)
            client = _ScriptedClient(
                [_Response(200, self._sam_body(contract))]
            )
            detections, response = diagnostic._dispatch_or_load(
                output,
                contract,
                client,
                base_url="http://127.0.0.1:9000",
            )
            self.assertEqual(detections, ())
            self.assertTrue(response["usable_response"])
            self.assertEqual(client.posted[0]["files"]["image"][1], payload)
            response_root = output / "responses" / str(contract["request_id"])
            receipt = diagnostic.load_json(response_root / "http_response_000.json")
            self.assertTrue(receipt["raw_http_body_persisted_before_schema_parse"])
            self.assertEqual(
                base64.b64decode(receipt["raw_body_base64"]),
                (response_root / "http_response_000.bin").read_bytes(),
            )

            # A normalized cache hit performs no call.
            diagnostic._dispatch_or_load(
                output, contract, client, base_url="http://127.0.0.1:9000"
            )
            self.assertEqual(len(client.posted), 1)

            # A crash after the raw receipt but before normalization reparses
            # the durable receipt and still performs no call.
            (response_root / "response.json").unlink()
            _, recovered = diagnostic._dispatch_or_load(
                output, contract, client, base_url="http://127.0.0.1:9000"
            )
            self.assertTrue(recovered["recovered_from_durable_raw_http_body"])
            self.assertEqual(len(client.posted), 1)

    def test_transport_and_http_status_attempts_retry_without_response_reuse(self) -> None:
        for first_action, expected_reason in (
            (
                httpx.ConnectError(
                    "fixture disconnect",
                    request=httpx.Request("POST", "http://127.0.0.1:9000/detect"),
                ),
                "SAM_TRANSPORT_FAILURE",
            ),
            (_Response(503, b'{"error":"unavailable"}'), "SAM_HTTP_STATUS_INVALID_RUN"),
        ):
            with self.subTest(reason=expected_reason), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary) / "A1"
                contract, _payload = self._prepared_contract(output)
                client = _ScriptedClient(
                    [first_action, _Response(200, self._sam_body(contract))]
                )
                with self.assertRaises(diagnostic.DiagnosticServiceInterruption):
                    diagnostic._dispatch_or_load(
                        output,
                        contract,
                        client,
                        base_url="http://127.0.0.1:9000",
                    )
                root = output / "responses" / str(contract["request_id"])
                invalid = diagnostic.load_json(
                    root / "request_attempt_000_invalid.json"
                )
                self.assertEqual(invalid["reason_code"], expected_reason)
                self.assertFalse((root / "response.json").exists())
                _, response = diagnostic._dispatch_or_load(
                    output,
                    contract,
                    client,
                    base_url="http://127.0.0.1:9000",
                )
                self.assertEqual(response["attempt_number"], 1)
                self.assertEqual(len(client.posted), 2)
                self.assertEqual(
                    diagnostic.load_json(root / "request_attempt_001.json")[
                        "prior_response_reuse"
                    ],
                    False,
                )

    def test_malformed_null_and_all_false_masks_are_2xx_capability_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            cases: list[tuple[str, bytes, str]] = []
            template = dict(self.contracts[0])
            cases.append(
                (
                    "malformed",
                    b"not-json",
                    "MALFORMED_RESPONSE_CAPABILITY_FAILURE",
                )
            )
            null_detection = {
                "object_id": 0,
                "box_xyxy": [1, 1, 2, 2],
                "mask_area": 1,
                "mask_png_base64": None,
                "score": 0.9,
            }
            cases.append(
                (
                    "null",
                    self._sam_body(template, [null_detection]),
                    "MISSING_OR_NULL_MASK_CAPABILITY_FAILURE",
                )
            )
            black = np.zeros(
                (int(template["expected_height"]), int(template["expected_width"])),
                dtype=np.uint8,
            )
            ok, encoded = cv2.imencode(".png", black)
            self.assertTrue(ok)
            all_false_detection = {
                **null_detection,
                "mask_png_base64": base64.b64encode(encoded.tobytes()).decode("ascii"),
            }
            cases.append(
                (
                    "all_false",
                    self._sam_body(template, [all_false_detection]),
                    "ALL_FALSE_MASK_CAPABILITY_FAILURE",
                )
            )
            for index, (name, body, expected) in enumerate(cases):
                with self.subTest(case=name):
                    output = base / name
                    contract, _payload = self._prepared_contract(output)
                    contract["request_id"] = f"fixture-{index}"
                    contract["response_path"] = (
                        f"responses/{contract['request_id']}/response.json"
                    )
                    client = _ScriptedClient([_Response(200, body)])
                    detections, response = diagnostic._dispatch_or_load(
                        output,
                        contract,
                        client,
                        base_url="http://127.0.0.1:9000",
                    )
                    self.assertTrue(response["usable_response"])
                    self.assertTrue(response["capability_failure"])
                    self.assertEqual(response["capability_failure_type"], expected)
                    if name != "null":
                        self.assertEqual(detections, ())

    def test_attempt_local_health_identity_gate_blocks_mismatch_without_detect(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "A1"
            output.mkdir(parents=True)
            (output / "attempt_manifest.json").write_text(
                json.dumps(
                    {
                        "status": "RUNNING",
                        "completed_rows": 0,
                        "attempt_id": diagnostic.ATTEMPT_ID,
                    }
                ),
                encoding="utf-8",
            )
            inputs = diagnostic._inputs(self.rows)
            client = _ScriptedClient(
                health={
                    "status": "ok",
                    "model": "wrong-model",
                    "checkpoint": inputs["detector"]["checkpoint"],
                }
            )
            with self.assertRaises(diagnostic.DiagnosticHealthGateFailure):
                diagnostic._sam_health_gate(
                    output,
                    inputs,
                    client,
                    completed_raw_calls_before_gate=0,
                )
            self.assertEqual(client.health_gets, 1)
            self.assertEqual(client.posted, [])
            manifest = diagnostic.load_json(output / "attempt_manifest.json")
            self.assertEqual(manifest["status"], "INVALID_SETUP")
            result = diagnostic.load_json(
                output / "service_health_gates/gate_000/result.json"
            )
            self.assertEqual(result["experimental_calls_dispatched_by_gate"], 0)

    def test_successful_grounding_telemetry_is_json_serializable(self) -> None:
        value = self.rows[0]
        observation, masks = diagnostic._load_observation_and_masks(value)
        mask = observation.oracle_mask
        detection = _detection(mask, object_id=0)
        telemetry = MappingProxyType(
            {"sam_detection": {"object_id": 0}, "fixture": "success"}
        )
        scoring = {
            "mask_iou": 1.0,
            "false_positive_area_pixels": 0,
            "centroid_error_uv_pixels": [0.0, 0.0],
            "sam_anchor_error_mm": [0.0, 0.0, 0.0],
            "incremental_sam_anchor_error_mm": [0.0, 0.0, 0.0],
            "sam_radial_xy_error_mm": 0.0,
            "incremental_sam_radial_xy_error_mm": 0.0,
        }
        response = {
            "request_id": "fixture-success",
            "raw_http_response_path": "fixture.bin",
            "raw_http_response_sha256": "0" * 64,
            "usable_response": True,
            "capability_failure": False,
        }
        with mock.patch.object(
            diagnostic,
            "evaluate_cal_x03_row",
            return_value=SimpleNamespace(scoring=scoring, telemetry=telemetry),
        ):
            result = diagnostic._condition_result(
                value,
                observation,
                masks,
                "TILED_512_O256",
                (detection,),
                (response,),
            )
        self.assertIs(type(result["production_grounding_telemetry"]), dict)
        diagnostic.canonical_json_bytes(result)

    def test_merge_thresholds_transitivity_and_numeric_ordering(self) -> None:
        base = self.contracts[1]
        masks = []
        left = np.zeros((512, 512), dtype=bool)
        left[100:120, 100:120] = True
        middle = np.zeros((512, 512), dtype=bool)
        middle[100:120, 110:130] = True  # IoU exactly 1/3, containment .5: no edge.
        half = np.zeros((512, 512), dtype=bool)
        half[100:120, 100:110] = True  # contained in left -> containment 1.0.
        masks.extend((left, middle, half))
        mapped = []
        for index, mask in enumerate(masks):
            item = dict(base)
            item["request_id"] = f"req-{10-index}"
            item["tile_ordinal"] = index
            item["tile_id"] = f"T{index}"
            mapped.extend(diagnostic.map_tile_detections(item, (_detection(mask),)))
        merged, provenance = diagnostic.merge_mapped_candidates(mapped)
        self.assertEqual(len(merged), 2)
        self.assertEqual(provenance["component_count"], 2)
        first_members = provenance["components"][0]["member_keys"]
        self.assertEqual([item[0] for item in first_members], [0, 2])
        self.assertEqual(provenance["components"][0]["component_sort_key"][0], 0)

    def test_precommit_blocks_dispatch_on_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "A1"
            allocation = diagnostic.allocate(output)
            self.assertEqual(allocation["live_response_count"], 0)
            committed = diagnostic.precommit_payloads(output)
            self.assertEqual(committed["request_payload_count"], 180)
            self.assertEqual(committed["live_response_count"], 0)
            first = next(output.glob("request_payloads/**/*.jpg"))
            first.write_bytes(first.read_bytes() + b"tamper")
            with self.assertRaises(diagnostic.DiagnosticError):
                diagnostic.validate_precommit(output)

    def test_dummy_oracle_cannot_change_payload_or_merge(self) -> None:
        row = self.rows[0]
        observation, masks = diagnostic._load_observation_and_masks(row)
        contracts = [
            item
            for item in self.contracts
            if item["diagnostic_row_id"] == row["diagnostic_row_id"]
        ]
        before = [
            diagnostic._jpeg_bytes(
                diagnostic._crop_for_contract(observation.frame.rgb, contract)
            )
            for contract in contracts
        ]
        for mask in masks.values():
            mask[:] = False
        after = [
            diagnostic._jpeg_bytes(
                diagnostic._crop_for_contract(observation.frame.rgb, contract)
            )
            for contract in contracts
        ]
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
