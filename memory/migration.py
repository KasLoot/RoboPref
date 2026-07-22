from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from memory.repositories import HistoryRepository, MemoryRepositoryError


@dataclass(slots=True)
class MigrationReport:
    source: str
    user_id: str
    legacy_records: int = 0
    history_candidates: list[dict[str, Any]] = field(default_factory=list)
    durable_review_candidates: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    applied_history_episode_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LegacyMemoryMigrator:
    """Conservative schema-v1 migration that never promotes inferred preferences.

    Legacy files remain read-only. Unconsented candidate records can be imported as
    episodes describing what happened. Records that appear to contain durable consent
    are only surfaced for semantic/user review; this tool deliberately does not turn a
    legacy model label into authority for a new active preference.
    """

    DURABLE_ORIGINS = {
        "dedicated_memory_confirmation",
        "explicit_future_statement",
        "explicit_memory_request",
    }
    DURABLE_MARKERS = {"explicit_future", "explicit", "durable"}

    def inspect(self, source: str | Path, *, user_id: str = "default") -> MigrationReport:
        source_path = Path(source).expanduser().resolve()
        try:
            data = json.loads(source_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise MemoryRepositoryError(
                f"Could not inspect legacy memory {source_path}: {error}"
            ) from error
        if not isinstance(data, dict) or not isinstance(data.get("preferences"), list):
            raise MemoryRepositoryError("Legacy preference file must contain a preferences list.")
        report = MigrationReport(
            source=str(source_path),
            user_id=user_id,
            legacy_records=len(data["preferences"]),
        )
        for record in data["preferences"]:
            if not isinstance(record, dict):
                report.warnings.append("Skipped a non-object legacy preference record.")
                continue
            if str(record.get("user_id", user_id)) != user_id:
                report.warnings.append(
                    f"Skipped cross-user legacy record {record.get('id', '<missing>')}."
                )
                continue
            candidate = self._history_candidate(record, user_id, source_path)
            report.history_candidates.append(candidate)
            durable_evidence = [
                evidence
                for evidence in record.get("evidence", [])
                if isinstance(evidence, dict) and self._looks_durable(evidence)
            ]
            if durable_evidence:
                report.durable_review_candidates.append(
                    {
                        "legacy_preference_id": record.get("id"),
                        "record": record,
                        "traceable_evidence": durable_evidence,
                        "disposition": (
                            "REVIEW_WITH_MEMORY_VLM_AND_USER; not imported automatically"
                        ),
                    }
                )
        return report

    def apply_history(
        self,
        report: MigrationReport,
        history_repository: HistoryRepository,
    ) -> MigrationReport:
        for episode in report.history_candidates:
            stored = history_repository.append(episode)
            report.applied_history_episode_ids.append(str(stored["episode_id"]))
        return report

    @classmethod
    def _looks_durable(cls, evidence: dict[str, Any]) -> bool:
        return (
            str(evidence.get("origin", "")).lower() in cls.DURABLE_ORIGINS
            or str(evidence.get("scope_marker", "")).lower() in cls.DURABLE_MARKERS
        ) and bool(str(evidence.get("quote", "")).strip())

    @staticmethod
    def _history_candidate(
        record: dict[str, Any], user_id: str, source_path: Path
    ) -> dict[str, Any]:
        legacy_id = str(record.get("id", "unknown"))
        stable = hashlib.sha256(
            f"{source_path}|{legacy_id}|{user_id}".encode("utf-8")
        ).hexdigest()[:24]
        value = record.get("value")
        return {
            "episode_id": f"legacy-{stable}",
            "user_id": user_id,
            "summary": (
                "Legacy interaction evidence recorded a task choice; it was not "
                "authorized as a durable preference."
            ),
            "user_request": None,
            "resolved_task": {
                "task_type": record.get("task_type"),
                "legacy_key": record.get("key"),
                "context": record.get("context", {}),
            },
            "user_choices": {str(record.get("key", "choice")): value},
            "actions": [],
            "execution": [],
            "validation": [],
            "terminal_outcome": "UNKNOWN_LEGACY",
            "user_visible_result": "Imported as history only; outcome was not reasserted.",
            "memory_events": [],
            "tags": ["legacy_migration", "history_only", "unconsented_candidate"],
            "legacy_provenance": {
                "source_file": str(source_path),
                "legacy_preference_id": record.get("id"),
                "legacy_status": record.get("status"),
                "evidence": record.get("evidence", []),
            },
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect legacy PrefMem schema-v1 memory without promoting candidates."
    )
    parser.add_argument("--legacy", required=True, help="Legacy preferences JSON file.")
    parser.add_argument("--user-id", default="default")
    parser.add_argument("--history-store", help="New history-v2 JSON path.")
    parser.add_argument(
        "--apply-history",
        action="store_true",
        help="Append history-only migration candidates; default is dry-run.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    migrator = LegacyMemoryMigrator()
    report = migrator.inspect(args.legacy, user_id=args.user_id)
    if args.apply_history:
        if not args.history_store:
            raise SystemExit("--apply-history requires --history-store")
        migrator.apply_history(report, HistoryRepository(args.history_store))
    print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
