from __future__ import annotations

import json
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from experiments.harness.campaign import (
    ApprovalError,
    CampaignRegistry,
    FreezeError,
    RegistryError,
    RunStatus,
    ScheduleCell,
    build_blocked_schedule,
    freeze_protocol,
    parse_locked_approval,
    schedule_sha256,
    verify_frozen_campaign,
)


class ScheduleTests(unittest.TestCase):
    def test_blocked_schedule_is_deterministic_sequential_and_crn_matched(self) -> None:
        cells = [
            ScheduleCell("matrix", "scenario-b", "variant-b", ("T5", "B1", "B2"), 2),
            ScheduleCell("matrix", "scenario-a", "variant-a", ("B2", "T5", "B1"), 2),
        ]
        first = build_blocked_schedule(cells, master_seed=12345, split="pilot")
        second = build_blocked_schedule(
            list(reversed(cells)), master_seed=12345, split="pilot"
        )
        different = build_blocked_schedule(cells, master_seed=54321, split="pilot")

        self.assertEqual(first, second)
        self.assertEqual(
            [entry.schedule_position for entry in first], list(range(1, len(first) + 1))
        )
        self.assertNotEqual(schedule_sha256(first), schedule_sha256(different))
        by_block: dict[str, list] = {}
        for entry in first:
            by_block.setdefault(entry.block_id, []).append(entry)
        self.assertEqual(len(by_block), 4)
        for block in by_block.values():
            self.assertEqual(len({entry.seed for entry in block}), 1)
            self.assertEqual(len({entry.crn_key for entry in block}), 1)
            self.assertEqual({entry.profile_id for entry in block}, {"T5", "B1", "B2"})


class RegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "campaign"
        self.schedule = build_blocked_schedule(
            [ScheduleCell("matrix", "scenario", "pilot-v", ("T5",), 1)],
            master_seed=7,
            split="pilot",
        )

    def registry(self) -> CampaignRegistry:
        return CampaignRegistry.create(
            self.root,
            campaign_id="pilot-one",
            mode="pilot",
            protocol_sha256="a" * 64,
            schedule=self.schedule,
        )

    @staticmethod
    def create_attempt_path(lease) -> None:
        lease.attempt_path.mkdir(parents=True)

    def authorize_recovery(
        self, registry: CampaignRegistry, schedule_id: str, attempt_number: int
    ) -> None:
        recovery = self.root / "environment" / "recovery" / (
            f"{schedule_id}-after-{attempt_number}.json"
        )
        recovery.parent.mkdir(parents=True, exist_ok=True)
        latest_attempt = registry.snapshot()["entries"][schedule_id]["attempts"][-1]
        interrupted_at = datetime.fromisoformat(
            latest_attempt["ended_utc"].replace("Z", "+00:00")
        )
        observations = []
        for sequence in (1, 2, 3):
            observed = (
                interrupted_at + timedelta(microseconds=sequence)
            ).isoformat().replace("+00:00", "Z")
            raw = recovery.parent / f"raw-{schedule_id}-{attempt_number}-{sequence}.json"
            observation = {
                "sequence": sequence,
                "observed_utc": observed,
                "passed": True,
                "service_id": "fixture-service",
                "check_id": f"check-{sequence}",
            }
            if sequence == 3:
                observation["non_mutating"] = True
            payload = json.dumps(observation).encode()
            raw.write_bytes(payload)
            observations.append(
                {
                    **observation,
                    "evidence_path": raw.relative_to(self.root).as_posix(),
                    "evidence_sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
        recovery.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "schedule_id": schedule_id,
                    "after_attempt_number": attempt_number,
                    "interruption_evidence_sha256": latest_attempt[
                        "infrastructure_evidence"
                    ]["sha256"],
                    "service_restored": True,
                    "health_checks": observations[:2],
                    "smoke_test": {**observations[2], "non_mutating": True},
                }
            ),
            encoding="utf-8",
        )
        registry.authorize_infrastructure_retry(
            schedule_id, recovery.relative_to(self.root).as_posix()
        )

    def write_infrastructure_evidence(self, registry: CampaignRegistry, lease) -> None:
        attempt = registry.snapshot()["entries"][lease.schedule_id]["attempts"][-1]
        started = datetime.fromisoformat(attempt["started_utc"].replace("Z", "+00:00"))
        observed = (started + timedelta(microseconds=1)).isoformat().replace(
            "+00:00", "Z"
        )
        raw_value = {
            "observed_utc": observed,
            "passed": False,
            "service_id": "fixture-service",
            "outage_id": f"outage-{lease.attempt_number}",
            "check_id": f"failed-check-{lease.attempt_number}",
        }
        raw_payload = json.dumps(raw_value).encode()
        (lease.attempt_path / "raw_outage.json").write_bytes(raw_payload)
        report = {
            "schema_version": 1,
            "schedule_id": lease.schedule_id,
            "attempt_number": lease.attempt_number,
            "classification": "external_infrastructure",
            "service_id": raw_value["service_id"],
            "outage_id": raw_value["outage_id"],
            "observed_utc": observed,
            "health_observation": {
                **raw_value,
                "evidence_path": "raw_outage.json",
                "evidence_sha256": hashlib.sha256(raw_payload).hexdigest(),
            },
        }
        (lease.attempt_path / "health_evidence.json").write_text(
            json.dumps(report), encoding="utf-8"
        )

    def test_valid_system_failure_is_final_and_cannot_be_retried(self) -> None:
        registry = self.registry()
        lease = registry.begin_attempt()
        self.create_attempt_path(lease)
        with patch(
            "experiments.harness.audit.audit_attempt_artifacts",
            return_value={"passed": True, "errors": []},
        ):
            registry.seal_attempt(
                lease.schedule_id,
                lease.attempt_number,
                RunStatus.VALID_SYSTEM_FAILURE,
                oracle_verdict="FAIL",
                reason="planner emitted an invalid plan",
                artifact_audit_passed=True,
            )

        self.assertIsNone(registry.next_entry())
        with self.assertRaisesRegex(RegistryError, "retry forbidden"):
            registry.begin_attempt(lease.schedule_id)
        snapshot = registry.snapshot()
        attempts = snapshot["entries"][lease.schedule_id]["attempts"]
        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0]["status"], "VALID_SYSTEM_FAILURE")
        self.assertTrue((self.root / "run_index.csv").is_file())

    def test_only_infrastructure_retries_and_third_interruption_quarantines(self) -> None:
        registry = self.registry()
        schedule_id = self.schedule[0].schedule_id
        for attempt_number in range(1, 4):
            lease = registry.begin_attempt(schedule_id)
            self.assertEqual(lease.attempt_number, attempt_number)
            self.create_attempt_path(lease)
            self.write_infrastructure_evidence(registry, lease)
            registry.seal_attempt(
                schedule_id,
                attempt_number,
                RunStatus.INFRA_INTERRUPTED,
                reason="independently observed service outage",
                artifact_audit_passed=False,
                independently_evidenced_infrastructure=True,
                infrastructure_evidence="health_evidence.json",
            )
            if attempt_number < 3:
                with self.assertRaisesRegex(RegistryError, "recovery"):
                    registry.next_entry()
                self.authorize_recovery(registry, schedule_id, attempt_number)
        state = registry.snapshot()["entries"][schedule_id]
        self.assertEqual(len(state["attempts"]), 3)
        self.assertEqual(state["disposition"], "QUARANTINED_INFRA_RETRY_EXHAUSTED")
        self.assertFalse(registry.completion()["complete"])
        self.assertTrue(
            registry.completion()["execution_closed_with_documented_unresolved"]
        )
        with self.assertRaisesRegex(RegistryError, "retry limit"):
            registry.begin_attempt(schedule_id)

    def test_open_attempt_blocks_parallel_start_and_is_reconciled(self) -> None:
        registry = self.registry()
        lease = registry.begin_attempt()
        self.create_attempt_path(lease)
        with self.assertRaisesRegex(RegistryError, "sequential policy"):
            registry.begin_attempt()
        self.assertEqual(
            registry.reconcile_open_attempts(reason="runner process disappeared"), 1
        )
        self.assertEqual(
            registry.snapshot()["entries"][lease.schedule_id]["status"],
            RunStatus.INVALID_HARNESS.value,
        )
        self.assertFalse(registry.completion()["complete"])

    def test_artifact_failure_cannot_be_labelled_valid(self) -> None:
        registry = self.registry()
        lease = registry.begin_attempt()
        self.create_attempt_path(lease)
        with self.assertRaisesRegex(RegistryError, "passing artifact audit"):
            registry.seal_attempt(
                lease.schedule_id,
                lease.attempt_number,
                RunStatus.VALID_PASS,
                oracle_verdict="PASS",
                artifact_audit_passed=False,
            )
        registry.seal_attempt(
            lease.schedule_id,
            lease.attempt_number,
            RunStatus.INVALID_HARNESS,
            reason="video checksum mismatch",
            artifact_audit_passed=False,
        )
        self.assertFalse(registry.completion()["complete"])
        self.assertEqual(registry.completion()["invalid_harness"], 1)

    def test_open_attempt_is_retryable_only_with_evidenced_infrastructure(self) -> None:
        registry = self.registry()
        lease = registry.begin_attempt()
        self.create_attempt_path(lease)
        with self.assertRaisesRegex(RegistryError, "independent external evidence"):
            registry.reconcile_open_attempts(
                reason="unspecified crash", status=RunStatus.INFRA_INTERRUPTED
            )
        self.write_infrastructure_evidence(registry, lease)
        self.assertEqual(
            registry.reconcile_open_attempts(
                reason="health journal recorded external service loss",
                status=RunStatus.INFRA_INTERRUPTED,
                independently_evidenced_infrastructure=True,
                infrastructure_evidence="health_evidence.json",
            ),
            1,
        )
        self.authorize_recovery(registry, lease.schedule_id, 1)
        self.assertEqual(registry.next_entry().schedule_id, lease.schedule_id)

    def test_explicit_schedule_id_cannot_jump_frozen_order(self) -> None:
        schedule = build_blocked_schedule(
            [
                ScheduleCell("matrix", "scenario-a", "pilot-a", ("T5",), 1),
                ScheduleCell("matrix", "scenario-b", "pilot-b", ("T5",), 1),
            ],
            master_seed=3,
            split="pilot",
        )
        registry = CampaignRegistry.create(
            self.root,
            campaign_id="pilot-one",
            mode="pilot",
            protocol_sha256="a" * 64,
            schedule=schedule,
        )
        with self.assertRaisesRegex(RegistryError, "frozen order"):
            registry.begin_attempt(schedule[1].schedule_id)
        self.assertEqual(registry.begin_attempt(schedule[0].schedule_id).schedule_id, schedule[0].schedule_id)

    def test_locked_registry_refuses_until_exact_bound_approval(self) -> None:
        locked_schedule = build_blocked_schedule(
            [ScheduleCell("matrix", "scenario", "locked-v", ("T5", "B1"), 1)],
            master_seed=7,
            split="locked_test",
        )
        repository = Path(__file__).resolve().parents[1]
        engine_source = repository / "experiments" / "harness" / "campaign.py"
        manifest = freeze_protocol(
            self.root,
            campaign_id="locked-one",
            mode="locked",
            protocol_source="# Locked registry test\n",
            schedule=locked_schedule,
            repo_root=repository,
            source_roots=[repository / "experiments" / "harness" / "campaign.py"],
            frozen_paths=[repository / "experiments" / "config" / "profiles.json"],
            metadata={
                "execution_engine": {
                    "engine_id": "locked-test-registry",
                    "module": "experiments.harness.campaign",
                    "class_name": "CampaignRegistry",
                    "source_path": "experiments/harness/campaign.py",
                    "source_sha256": hashlib.sha256(engine_source.read_bytes()).hexdigest(),
                },
                "production_engine_config": {"fixture": True},
                "video_config": {"fixture": True},
                "model_service_mapping": {"fixture": "fixture"},
                "reported_model_ids": {"fixture": "fixture"},
                "service_endpoints": {"fixture": "http://127.0.0.1:1"},
                "controls": {"fixture": True},
                "hashes": {"fixture": "0" * 64},
                "health_checks": {"passed": True},
                "tunnel_epochs": {"fixture": "fixture"},
                "environment": {"fixture": "fixture"},
                "known_deviations": [],
            },
        )
        registry = CampaignRegistry(self.root)
        with self.assertRaises(ApprovalError):
            registry.next_entry()
        with self.assertRaises(ApprovalError):
            registry.authorize_locked(
                f"APPROVE LOCKED CAMPAIGN wrong {manifest['protocol_sha256']}"
            )
        registry.authorize_locked(
            f"APPROVE LOCKED CAMPAIGN locked-one {manifest['protocol_sha256']}"
        )
        self.assertEqual(registry.next_entry(), locked_schedule[0])

    def test_approval_parser_is_exact(self) -> None:
        valid = f"APPROVE LOCKED CAMPAIGN campaign-v1 {'c' * 64}"
        self.assertEqual(parse_locked_approval(valid).campaign_id, "campaign-v1")
        for invalid in (" " + valid, valid + "\n", valid + " please", valid.upper()):
            with self.subTest(invalid=invalid), self.assertRaises(ApprovalError):
                parse_locked_approval(invalid)


class FreezeTests(unittest.TestCase):
    def _repository(self, root: Path) -> Path:
        repo = root / "repository"
        (repo / "src").mkdir(parents=True)
        (repo / "experiments" / "config").mkdir(parents=True)
        (repo / "src" / "system.py").write_text("VALUE = 1\n", encoding="utf-8")
        (repo / "experiments" / "config" / "frozen.json").write_text(
            '{"threshold": 2}\n', encoding="utf-8"
        )
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        subprocess.run(
            ["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "add", "."],
            cwd=repo,
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.invalid",
                "commit",
                "-qm",
                "fixture",
            ],
            cwd=repo,
            check=True,
        )
        return repo

    def test_freeze_is_canonical_idempotent_and_detects_source_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo = self._repository(Path(temporary))
            campaign = repo / "experiments" / "runs" / "pilot-freeze"
            schedule = build_blocked_schedule(
                [ScheduleCell("matrix", "scenario", "pilot-v", ("T5",), 1)],
                master_seed=9,
                split="pilot",
            )
            arguments = dict(
                campaign_id="pilot-freeze",
                mode="pilot",
                protocol_source="# Frozen protocol\n",
                schedule=schedule,
                repo_root=repo,
                source_roots=[repo / "src"],
                frozen_paths=[repo / "experiments" / "config" / "frozen.json"],
            )
            first = freeze_protocol(campaign, **arguments)
            second = freeze_protocol(campaign, **arguments)
            self.assertEqual(first, second)
            self.assertEqual(
                json.loads((campaign / "CAMPAIGN_MANIFEST.json").read_text())[
                    "protocol_sha256"
                ],
                first["protocol_sha256"],
            )
            verify_frozen_campaign(campaign, check_source=True)
            (repo / "src" / "new_module.py").write_text("NEW = True\n", encoding="utf-8")
            with self.assertRaisesRegex(FreezeError, "membership changed"):
                verify_frozen_campaign(campaign, check_source=True)
            (repo / "src" / "new_module.py").unlink()
            (repo / "src" / "system.py").write_text("VALUE = 2\n", encoding="utf-8")
            with self.assertRaisesRegex(FreezeError, "source changed after freeze"):
                verify_frozen_campaign(campaign, check_source=True)
            with self.assertRaises(FreezeError):
                freeze_protocol(campaign, **{**arguments, "protocol_source": "changed"})

    def test_freeze_refuses_missing_crosswalk_target_without_partial_campaign(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo = self._repository(Path(temporary))
            crosswalk = repo / "experiments" / "REQUIREMENTS_CROSSWALK.md"
            crosswalk.write_text(
                "# Crosswalk\n\n[missing](harness/not-present.py)\n",
                encoding="utf-8",
            )
            campaign = repo / "experiments" / "runs" / "pilot-missing-doc"
            schedule = build_blocked_schedule(
                [ScheduleCell("matrix", "scenario", "pilot-v", ("T5",), 1)],
                master_seed=9,
                split="pilot",
            )
            with self.assertRaisesRegex(
                FreezeError, "missing or external documentation target"
            ):
                freeze_protocol(
                    campaign,
                    campaign_id="pilot-missing-doc",
                    mode="pilot",
                    protocol_source="# Frozen protocol\n",
                    schedule=schedule,
                    repo_root=repo,
                    source_roots=[repo / "src"],
                    frozen_paths=[
                        repo / "experiments" / "config" / "frozen.json"
                    ],
                )
            self.assertFalse(campaign.exists())


if __name__ == "__main__":
    unittest.main()
