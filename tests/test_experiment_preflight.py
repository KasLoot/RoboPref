from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import tempfile
from typing import Any
import unittest

import httpx

from experiments.harness.preflight import (
    FailureClass,
    PreflightPolicy,
    PreflightRunner,
    ProbePhase,
    ServiceConfig,
    ServiceKind,
    classify_attempt_service_failure,
    write_preflight_report,
    write_recovery_bundle,
)
from experiments.harness.campaign import (
    CampaignRegistry,
    RunStatus,
    ScheduleCell,
    build_blocked_schedule,
)


class _FakeServices:
    def __init__(self) -> None:
        self.counts: dict[tuple[str, str], int] = defaultdict(int)
        self.model_health_statuses: list[int] = []
        self.chat_content = "OK"
        self.embedding: list[float] = [0.1, -0.2, 0.3]
        self.sam_count = 1
        self.disconnect = False

    def handle(self, request: httpx.Request) -> httpx.Response:
        key = (request.method, request.url.path)
        self.counts[key] += 1
        if self.disconnect:
            raise httpx.ConnectError("controlled disconnected tunnel", request=request)
        if key == ("GET", "/v1/models"):
            index = self.counts[key] - 1
            status = (
                self.model_health_statuses[index]
                if index < len(self.model_health_statuses)
                else 200
            )
            payload: Any = (
                {"data": [{"id": "gemma-test"}, {"id": "embedding-test"}]}
                if status == 200
                else {"detail": "warming"}
            )
            return httpx.Response(status, json=payload)
        if key == ("GET", "/health"):
            return httpx.Response(
                200, json={"status": "ok", "model": "sam3.1-test"}
            )
        if key == ("POST", "/v1/chat/completions"):
            parsed = json.loads(request.content)
            if not parsed["messages"][0]["content"].startswith("Preflight only"):
                raise AssertionError("chat smoke payload changed")
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": self.chat_content}}]},
            )
        if key == ("POST", "/v1/embeddings"):
            parsed = json.loads(request.content)
            if parsed["input"] != ["RoboPref preflight"]:
                raise AssertionError("embedding smoke payload changed")
            return httpx.Response(
                200, json={"data": [{"embedding": self.embedding}]}
            )
        if key == ("POST", "/detect"):
            request_bytes = request.content
            for required in (b'name="prompt"', b"red square", b"calibration.jpg"):
                if required not in request_bytes:
                    raise AssertionError(f"SAM multipart smoke lacks {required!r}")
            detections = [
                {
                    "object_id": 1,
                    "box_xyxy": [24, 24, 72, 72],
                    "mask_area": 2304,
                }
            ][: self.sam_count]
            return httpx.Response(
                200,
                json={
                    "image": {"width": 96, "height": 96},
                    "count": self.sam_count,
                    "detections": detections,
                },
            )
        return httpx.Response(404, json={"detail": "not found"})

    def client_factory(self, timeout: float) -> httpx.Client:
        return httpx.Client(
            transport=httpx.MockTransport(self.handle),
            timeout=timeout,
            trust_env=False,
        )


def _configs(url: str = "http://127.0.0.1:1") -> tuple[ServiceConfig, ...]:
    return (
        ServiceConfig(ServiceKind.GEMMA, url, "Gemma", "gemma-test"),
        ServiceConfig(
            ServiceKind.EMBEDDING, url, "EmbeddingGemma", "embedding-test"
        ),
        ServiceConfig(ServiceKind.SAM, url, "SAM 3.1"),
    )


class ExperimentPreflightTests(unittest.TestCase):
    def test_two_health_successes_then_nonmutating_smoke(self) -> None:
        fake = _FakeServices()
        report = PreflightRunner(
            policy=PreflightPolicy(retry_backoff_seconds=0),
            client_factory=fake.client_factory,
        ).run(_configs())
        self.assertTrue(report.passed)
        self.assertTrue(all(service.health_successes == 2 for service in report.services))
        self.assertTrue(all(service.smoke_passed for service in report.services))
        self.assertEqual(fake.counts[("GET", "/v1/models")], 4)
        self.assertEqual(fake.counts[("GET", "/health")], 2)
        self.assertEqual(fake.counts[("POST", "/v1/chat/completions")], 1)
        self.assertEqual(fake.counts[("POST", "/v1/embeddings")], 1)
        self.assertEqual(fake.counts[("POST", "/detect")], 1)
        for service in report.services:
            self.assertIs(service.evidence[-1].phase, ProbePhase.SMOKE)
            self.assertTrue(all(item.path.startswith("/") for item in service.evidence))
            self.assertTrue(
                all("127.0.0.1" not in item.path for item in service.evidence)
            )
        with tempfile.TemporaryDirectory() as directory:
            path = write_preflight_report(Path(directory) / "preflight.json", report)
            persisted = json.loads(path.read_text())
            self.assertTrue(persisted["passed"])
            self.assertEqual(len(persisted["services"]), 3)

    def test_health_retry_capped_at_three_requires_consecutive_successes(self) -> None:
        fake = _FakeServices()
        fake.model_health_statuses = [503, 200, 200]
        report = PreflightRunner(
            policy=PreflightPolicy(maximum_attempts=3, retry_backoff_seconds=0),
            client_factory=fake.client_factory,
        ).run((ServiceConfig(ServiceKind.GEMMA, "http://127.0.0.1:1", "Gemma", "gemma-test"),))
        self.assertTrue(report.passed)
        service = report.services[0]
        health = [item for item in service.evidence if item.phase is ProbePhase.HEALTH]
        self.assertEqual(len(health), 3)
        self.assertEqual([item.ok for item in health], [False, True, True])
        self.assertIs(health[0].failure_class, FailureClass.INFRASTRUCTURE)

    def test_semantic_failure_not_retried_or_called_infrastructure(self) -> None:
        fake = _FakeServices()
        fake.chat_content = "I am ready"
        report = PreflightRunner(
            policy=PreflightPolicy(retry_backoff_seconds=0),
            client_factory=fake.client_factory,
        ).run((ServiceConfig(ServiceKind.GEMMA, "http://127.0.0.1:1", "Gemma", "gemma-test"),))
        self.assertFalse(report.passed)
        service = report.services[0]
        self.assertIs(service.failure_class, FailureClass.SYSTEM)
        self.assertIs(classify_attempt_service_failure(report), FailureClass.SYSTEM)
        self.assertEqual(fake.counts[("POST", "/v1/chat/completions")], 1)

    def test_sam_zero_detection_is_valid_service_smoke(self) -> None:
        fake = _FakeServices()
        fake.sam_count = 0
        report = PreflightRunner(
            policy=PreflightPolicy(retry_backoff_seconds=0),
            client_factory=fake.client_factory,
        ).run((ServiceConfig(ServiceKind.SAM, "http://127.0.0.1:1", "SAM"),))
        self.assertTrue(report.passed)
        self.assertTrue(report.services[0].smoke_passed)
        self.assertIn("0 detection(s)", report.services[0].evidence[-1].detail)

    def test_disconnection_is_infrastructure_with_finite_attempts(self) -> None:
        fake = _FakeServices()
        fake.disconnect = True
        report = PreflightRunner(
            policy=PreflightPolicy(
                timeout_seconds=0.2, maximum_attempts=2, retry_backoff_seconds=0
            ),
            client_factory=fake.client_factory,
        ).run(
            (
                ServiceConfig(
                    ServiceKind.GEMMA,
                    "http://127.0.0.1:1",
                    "disconnected-Gemma",
                ),
            )
        )
        self.assertFalse(report.passed)
        service = report.services[0]
        self.assertIs(service.failure_class, FailureClass.INFRASTRUCTURE)
        self.assertEqual(len(service.evidence), 2)
        self.assertIs(
            classify_attempt_service_failure(report), FailureClass.INFRASTRUCTURE
        )

    def test_credentials_nonloopback_and_excess_retries_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "credentials"):
            ServiceConfig(
                ServiceKind.GEMMA,
                "http://user:pass@127.0.0.1:8000",
                "bad",
            )
        with self.assertRaisesRegex(ValueError, "loopback"):
            ServiceConfig(ServiceKind.GEMMA, "http://example.com:8000", "bad")
        with self.assertRaisesRegex(ValueError, "maximum_attempts"):
            PreflightPolicy(maximum_attempts=4)

    def test_recovery_bundle_is_raw_hashed_and_registry_authorizable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "campaign"
            schedule = build_blocked_schedule(
                [ScheduleCell("matrix", "scenario", "pilot-v", ("T5",), 1)],
                master_seed=7,
                split="pilot",
            )
            registry = CampaignRegistry.create(
                root,
                campaign_id="recovery-pilot",
                mode="pilot",
                protocol_sha256="a" * 64,
                schedule=schedule,
            )
            lease = registry.begin_attempt()
            lease.attempt_path.mkdir(parents=True)
            (lease.attempt_path / "infra.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "classification": "INFRA_INTERRUPTED",
                        "service_label": "gemma-recovery-test",
                        "endpoint_label": "loopback-test",
                        "failure_kind": "controlled_disconnect",
                        "transport_error_type": "ConnectError",
                        "safe_stop_confirmed": True,
                        "health_observations": [{"ok": False}],
                    }
                ),
                encoding="utf-8",
            )
            registry.seal_attempt(
                lease.schedule_id,
                1,
                RunStatus.INFRA_INTERRUPTED,
                reason="controlled outage",
                artifact_audit_passed=False,
                independently_evidenced_infrastructure=True,
                infrastructure_evidence="infra.json",
            )
            fake = _FakeServices()
            preflight = PreflightRunner(
                policy=PreflightPolicy(retry_backoff_seconds=0),
                client_factory=fake.client_factory,
            )
            report = write_recovery_bundle(
                root,
                lease.schedule_id,
                ServiceConfig(
                    ServiceKind.GEMMA,
                    "http://127.0.0.1:1/v1",
                    "gemma-recovery-test",
                    "gemma-test",
                ),
                runner=preflight,
            )
            relative = report.relative_to(root).as_posix()
            authorization = registry.authorize_infrastructure_retry(
                lease.schedule_id, relative
            )
            self.assertEqual(authorization["path"], relative)
            self.assertEqual(registry.next_entry().schedule_id, lease.schedule_id)
            raw = sorted(report.parent.glob("*-health-*.json"))
            self.assertEqual(len(raw), 2)


if __name__ == "__main__":
    unittest.main()
