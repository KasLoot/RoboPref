from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
import unittest
import json

from experiments.harness.audit import (
    AuditError,
    audit_campaign,
    audit_crosslinks,
    audit_diagnosis_evidence,
    verify_checksum_manifest,
    write_campaign_checksums,
    write_checksum_manifest,
)
from experiments.harness.campaign import (
    ScheduleCell,
    build_blocked_schedule,
    freeze_protocol,
)


class ChecksumAuditTests(unittest.TestCase):
    def test_complete_checksum_manifest_detects_tamper_and_uncovered_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "nested").mkdir()
            (root / "a.txt").write_text("alpha", encoding="utf-8")
            (root / "nested" / "b.txt").write_text("beta", encoding="utf-8")
            write_checksum_manifest(root)
            self.assertTrue(verify_checksum_manifest(root)["passed"])
            (root / "a.txt").write_text("tampered", encoding="utf-8")
            report = verify_checksum_manifest(root)
            self.assertFalse(report["passed"])
            self.assertIn("checksum mismatch: a.txt", report["errors"])
            (root / "later.txt").write_text("uncovered", encoding="utf-8")
            report = verify_checksum_manifest(root)
            self.assertIn("artifact missing from checksums: later.txt", report["errors"])
            with self.assertRaisesRegex(AuditError, "refusing to overwrite"):
                write_checksum_manifest(root)

    def test_checksum_manifest_rejects_parent_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "root"
            root.mkdir()
            outside = root.parent / "outside.txt"
            outside.write_text("outside", encoding="utf-8")
            (root / "checksums.sha256").write_text(
                f"{'0' * 64}  ../outside.txt\n", encoding="utf-8"
            )
            report = verify_checksum_manifest(root, require_complete=False)
            self.assertFalse(report["passed"])
            self.assertIn("unsafe checksum target", report["errors"][0])


class CrosslinkAuditTests(unittest.TestCase):
    def test_local_files_explicit_event_anchors_and_video_fragments(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "experiment_record.md").write_text(
                '# Record\n\n<a id="event-e000123"></a>\n## Event E000123\n',
                encoding="utf-8",
            )
            (root / "video.mp4").write_bytes(b"not-decoded-by-link-audit")
            (root / "diagnosis.md").write_text(
                "[event](./experiment_record.md#event-e000123) "
                "[video](./video.mp4#t=2.5)\n",
                encoding="utf-8",
            )
            self.assertTrue(audit_crosslinks(root)["passed"])
            (root / "diagnosis.md").write_text(
                "[event](./experiment_record.md#event-e999999)\n", encoding="utf-8"
            )
            report = audit_crosslinks(root)
            self.assertFalse(report["passed"])
            self.assertIn("missing Markdown anchor", report["errors"][0])

    def test_parent_relative_link_may_stay_inside_root_but_not_escape_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "root"
            (root / "docs").mkdir(parents=True)
            (root / "target.md").write_text("# Target\n", encoding="utf-8")
            source = root / "docs" / "source.md"
            source.write_text("[target](../target.md#target)\n", encoding="utf-8")
            self.assertTrue(audit_crosslinks(root)["passed"])
            outside = root.parent / "outside.md"
            outside.write_text("# Outside\n", encoding="utf-8")
            source.write_text("[outside](../../outside.md)\n", encoding="utf-8")
            self.assertFalse(audit_crosslinks(root)["passed"])
            source.write_text("[private](file:///etc/passwd)\n", encoding="utf-8")
            self.assertFalse(audit_crosslinks(root)["passed"])

    def test_links_inside_code_are_not_artifact_crosslinks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "report.md").write_text(
                "```json\n{\"model_output\": \"[not a link](missing.md)\"}\n```\n"
                "Inline `[also not](absent.md)` text.\n",
                encoding="utf-8",
            )
            self.assertTrue(audit_crosslinks(root)["passed"])

    def test_video_evidence_requires_a_media_time_fragment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "video.mp4").write_bytes(b"fixture")
            (root / "diagnosis.md").write_text(
                "[video](./video.mp4#definitely-not-a-time)\n", encoding="utf-8"
            )
            self.assertFalse(audit_crosslinks(root)["passed"])

    def test_each_factual_diagnosis_section_needs_its_own_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "experiment_record.md").write_text(
                '# Record\n\n<a id="event-e000001"></a>\n## Event E000001\n',
                encoding="utf-8",
            )
            (root / "video.mp4").write_bytes(b"fixture")
            diagnosis = root / "diagnosis.md"
            diagnosis.write_text(
                "# Diagnosis\n\n"
                "## Run status\n\nVALID_SYSTEM_FAILURE\n\n"
                "## Behavioral summary\n\n"
                "Observed failure ([event](./experiment_record.md#event-e000001)).\n\n"
                "## Likely failure layer\n\nplanning\n",
                encoding="utf-8",
            )
            report = audit_diagnosis_evidence(root)
            self.assertFalse(report["passed"])
            self.assertTrue(
                any("Likely failure layer" in error for error in report["errors"])
            )
            diagnosis.write_text(
                diagnosis.read_text(encoding="utf-8").replace(
                    "planning\n",
                    "planning ([video](./video.mp4#t=0.0)).\n",
                ),
                encoding="utf-8",
            )
            self.assertTrue(audit_diagnosis_evidence(root)["passed"])


class CampaignAuditTests(unittest.TestCase):
    def _freeze_fixture(self, temporary: str) -> Path:
        repo = Path(temporary) / "repo"
        (repo / "src").mkdir(parents=True)
        (repo / "experiments" / "config").mkdir(parents=True)
        (repo / "experiments" / "harness").mkdir(parents=True)
        (repo / "src" / "app.py").write_text("X = 1\n", encoding="utf-8")
        (repo / "experiments" / "harness" / "campaign.py").write_text(
            "# frozen source target\n", encoding="utf-8"
        )
        (repo / "ROOT_NOTES.md").write_text(
            "# Root notes\n\n[not part of the crosswalk](missing-nested.md)\n",
            encoding="utf-8",
        )
        (repo / "experiments" / "REQUIREMENTS_CROSSWALK.md").write_text(
            "# Requirements crosswalk\n\n"
            "[harness](harness/campaign.py)\n"
            "[root notes](../ROOT_NOTES.md)\n"
            "[external](https://example.invalid/reference)\n"
            "Inline `[not a link](missing-inline.md)`.\n",
            encoding="utf-8",
        )
        frozen = repo / "experiments" / "config" / "frozen.json"
        frozen.write_text("{}\n", encoding="utf-8")
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        subprocess.run(
            ["git", "-c", "user.name=Test", "-c", "user.email=t@example.invalid", "add", "."],
            cwd=repo,
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=Test",
                "-c",
                "user.email=t@example.invalid",
                "commit",
                "-qm",
                "fixture",
            ],
            cwd=repo,
            check=True,
        )
        schedule = build_blocked_schedule(
            [ScheduleCell("matrix", "scenario", "pilot-v", ("T5",), 1)],
            master_seed=11,
            split="pilot",
        )
        campaign = repo / "experiments" / "runs" / "pilot-audit"
        freeze_protocol(
            campaign,
            campaign_id="pilot-audit",
            mode="pilot",
            protocol_source="# Protocol\n",
            schedule=schedule,
            repo_root=repo,
            source_roots=[repo / "src"],
            frozen_paths=[frozen],
        )
        return campaign

    def test_freeze_rewrites_and_binds_self_contained_crosswalk_targets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            campaign = self._freeze_fixture(temporary)
            manifest = json.loads(
                (campaign / "CAMPAIGN_MANIFEST.json").read_text(encoding="utf-8")
            )
            snapshot = manifest["documentation_snapshot"]
            rendered = (campaign / "REQUIREMENTS_CROSSWALK.md").read_text(
                encoding="utf-8"
            )
            self.assertIn(
                "documentation_snapshot/experiments/harness/campaign.py", rendered
            )
            self.assertIn("documentation_snapshot/ROOT_NOTES.md.source", rendered)
            self.assertNotIn("](../ROOT_NOTES.md)", rendered)
            canonical = campaign / snapshot["canonical_snapshot_path"]
            self.assertEqual(
                canonical.read_text(encoding="utf-8"),
                (
                    Path(temporary)
                    / "repo"
                    / "experiments"
                    / "REQUIREMENTS_CROSSWALK.md"
                ).read_text(encoding="utf-8"),
            )
            report = audit_crosslinks(campaign)
            self.assertTrue(report["passed"], report["errors"])

            target = campaign / "documentation_snapshot" / "ROOT_NOTES.md.source"
            target.write_text("tampered\n", encoding="utf-8")
            report = audit_campaign(
                campaign, verify_video=False, require_complete=False
            )
            self.assertFalse(report["passed"])
            self.assertTrue(
                any(
                    "documentation snapshot file changed" in error
                    for error in report["errors"]
                ),
                report["errors"],
            )

    def test_incomplete_frozen_campaign_is_auditable_but_not_complete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            campaign = self._freeze_fixture(temporary)
            inspection = audit_campaign(
                campaign, verify_video=False, require_complete=False
            )
            self.assertTrue(inspection["passed"], inspection["errors"])
            self.assertFalse(inspection["registry"]["completion"]["complete"])
            final = audit_campaign(campaign, verify_video=False, require_complete=True)
            self.assertFalse(final["passed"])
            self.assertIn(
                "campaign has runnable, not-run, open, or unresolved schedule entries",
                final["errors"],
            )

    def test_campaign_checksum_manifest_is_independently_verified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            campaign = self._freeze_fixture(temporary)
            write_campaign_checksums(campaign)
            report = audit_campaign(campaign, verify_video=False, require_complete=False)
            self.assertTrue(report["campaign_checksums"]["passed"])
            (campaign / "PROTOCOL.md").write_text("tampered", encoding="utf-8")
            report = audit_campaign(campaign, verify_video=False, require_complete=False)
            self.assertFalse(report["passed"])
            self.assertTrue(any("protocol" in item.casefold() for item in report["errors"]))

    def test_stale_run_index_cannot_be_blessed_by_new_checksums(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            campaign = self._freeze_fixture(temporary)
            (campaign / "run_index.csv").write_text(
                "schedule_id,forged\nS999999,true\n", encoding="utf-8"
            )
            write_campaign_checksums(campaign)
            report = audit_campaign(campaign, verify_video=False, require_complete=False)
            self.assertFalse(report["passed"])
            self.assertIn(
                "run_index.csv differs from authoritative RUN_STATE.json",
                report["registry"]["errors"],
            )

    def test_malformed_registry_is_reported_without_crashing_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            campaign = self._freeze_fixture(temporary)
            state_path = campaign / "RUN_STATE.json"
            state = json.loads(state_path.read_text(encoding="utf-8"))
            entry = next(iter(state["entries"].values()))
            entry["attempts"] = [
                {
                    "attempt_number": 1,
                    "attempt_id": "malformed",
                    "status": "NOT_RUN",
                }
            ]
            state_path.write_text(json.dumps(state), encoding="utf-8")
            report = audit_campaign(
                campaign, verify_video=False, require_complete=False
            )
            self.assertFalse(report["passed"])
            self.assertTrue(
                any("registry" in error.casefold() for error in report["errors"])
            )


if __name__ == "__main__":
    unittest.main()
