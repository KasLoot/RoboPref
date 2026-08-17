from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import httpx

from experiments_suite_v2.services import ServiceGateError, ServiceHealthGate, TUNNELS
from experiments_suite_v2.registry import SUITE_ROOT


class V2ServiceGateTests(unittest.TestCase):
    def _gate(self) -> ServiceHealthGate:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/models"):
                model = (
                    "/workspace/models/embeddinggemma-300m"
                    if request.url.port == 8080
                    else "/workspace/models/gemma-4-26B-A4B-it"
                )
                return httpx.Response(200, json={"object": "list", "data": [{"id": model}]})
            if request.url.path.endswith("/chat/completions"):
                return httpx.Response(
                    200,
                    json={
                        "choices": [{"message": {"content": "SERVICE_OK"}}],
                        "model": "/workspace/models/gemma-4-26B-A4B-it",
                        "system_fingerprint": "vllm-0.27.1-f39dd7e9",
                    },
                )
            if request.url.path.endswith("/embeddings"):
                return httpx.Response(
                    200,
                    json={
                        "data": [{"embedding": [0.01] * 768}],
                        "model": "/workspace/models/embeddinggemma-300m",
                    },
                )
            if request.url.path.endswith("/health"):
                return httpx.Response(
                    200,
                    json={
                        "status": "ok",
                        "model": "sam3.1",
                        "checkpoint": "/workspace/models/sam3.1/sam3.1_multiplex.pt",
                    },
                )
            if request.url.path.endswith("/detect"):
                return httpx.Response(200, json={"count": 1})
            return httpx.Response(404, json={"error": "not found"})

        return ServiceHealthGate(client=httpx.Client(transport=httpx.MockTransport(handler)))

    def test_all_frozen_identity_probes_pass_with_valid_responses(self) -> None:
        gate = self._gate()
        self.addCleanup(gate.close)
        evidence = gate.probe_all()
        self.assertEqual(len(evidence), 3)
        self.assertTrue(all(item["status"] == "PASS" for item in evidence))
        sam = next(item for item in evidence if item["kind"] == "SAM_SEGMENTATION")
        self.assertEqual(
            sam["evidence"]["detect"]["image_path"],
            "camera_previews/motion_revision_candidate/sam_top_down.png",
        )
        self.assertEqual(
            sam["evidence"]["detect"]["image_sha256"],
            "dbff8a97eca5b9744ce9058d5e51ba1aa7084e8300b23d9dcb065f191e60aeeb",
        )

    def test_sam_probe_rejects_stale_nonapproved_image_before_request(self) -> None:
        gate = self._gate()
        self.addCleanup(gate.close)
        with self.assertRaises(ServiceGateError) as caught:
            gate.probe(
                "sam3.1-grounding-v1",
                sam_image_path=SUITE_ROOT / "camera_previews" / "sam_top_down.png",
            )
        self.assertEqual(caught.exception.stage, "probe_image")

    def test_model_identity_mismatch_fails_closed(self) -> None:
        client = httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    json={"object": "list", "data": [{"id": "wrong-model"}]},
                )
            )
        )
        gate = ServiceHealthGate(client=client)
        self.addCleanup(gate.close)
        with self.assertRaises(ServiceGateError) as caught:
            gate.probe("gemma-hri-planner-monitor-validator-v1")
        self.assertEqual(caught.exception.stage, "models")

    def test_tunnel_commands_match_only_user_authorized_forwards(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            identity = Path(temporary) / "key"
            identity.write_text("fixture", encoding="utf-8")
            commands = {
                service_id: spec.command(identity)
                for service_id, spec in TUNNELS.items()
            }
        self.assertIn("8000:127.0.0.1:8000", commands["gemma-hri-planner-monitor-validator-v1"])
        self.assertIn("9000:127.0.0.1:9000", commands["sam3.1-grounding-v1"])
        self.assertIn("8080:127.0.0.1:8080", commands["embeddinggemma-retriever-v1"])
        for command in commands.values():
            self.assertEqual(command[0:2], ("ssh", "-N"))
            self.assertIn("ExitOnForwardFailure=yes", command)


if __name__ == "__main__":
    unittest.main()
