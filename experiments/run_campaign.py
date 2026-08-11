"""CLI for approval, resume, and append-only campaign registry transitions."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import asdict, fields
import ipaddress
import json
from pathlib import Path
from typing import Sequence
from urllib.parse import urlsplit

try:
    from experiments.harness.campaign import (
        CampaignRegistry,
        RunStatus,
        verify_frozen_campaign,
    )
except ModuleNotFoundError:  # Support direct script execution.
    from harness.campaign import (  # type: ignore[no-redef]
        CampaignRegistry,
        RunStatus,
        verify_frozen_campaign,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Control a frozen campaign registry")
    parser.add_argument("campaign", type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status")
    commands.add_parser("next")
    authorize = commands.add_parser("authorize")
    authorize.add_argument("--approval", required=True)
    begin = commands.add_parser("begin")
    begin.add_argument("--schedule-id")
    seal = commands.add_parser("seal")
    seal.add_argument("schedule_id")
    seal.add_argument("attempt_number", type=int)
    seal.add_argument("status", choices=[status.value for status in RunStatus if status is not RunStatus.NOT_RUN])
    seal.add_argument("--oracle-verdict")
    seal.add_argument("--reason")
    audit_group = seal.add_mutually_exclusive_group()
    audit_group.add_argument("--artifact-audit-passed", action="store_true")
    audit_group.add_argument("--artifact-audit-failed", action="store_true")
    seal.add_argument(
        "--independently-evidenced-infrastructure",
        action="store_true",
        help="required when sealing INFRA_INTERRUPTED; asserts external evidence exists",
    )
    seal.add_argument(
        "--infrastructure-evidence",
        help="relative path under the attempt directory to contemporaneous health evidence",
    )
    reconcile = commands.add_parser("reconcile")
    reconcile.add_argument("--reason", required=True)
    reconcile.add_argument(
        "--status",
        choices=(
            RunStatus.INVALID_HARNESS.value,
            RunStatus.INFRA_INTERRUPTED.value,
            RunStatus.ABORTED_SAFETY.value,
        ),
        default=RunStatus.INVALID_HARNESS.value,
    )
    reconcile.add_argument(
        "--independently-evidenced-infrastructure",
        action="store_true",
        help="required to make an open attempt infrastructure-retryable",
    )
    reconcile.add_argument(
        "--infrastructure-evidence",
        help="relative path under the attempt directory to contemporaneous health evidence",
    )
    retry = commands.add_parser(
        "authorize-retry",
        help="bind a restoration/two-health-check/smoke report to one infra retry",
    )
    retry.add_argument("schedule_id")
    retry.add_argument("recovery_evidence_path")
    commands.add_parser("index")
    execute = commands.add_parser(
        "execute", help="run one or all sequential attempts through the bound engine"
    )
    execute.add_argument(
        "--engine",
        required=True,
        choices=(
            "scripted-pass",
            "scripted-failure",
            "mujoco-pass",
            "mujoco-failure",
            "service-live",
            "production-mujoco",
        ),
    )
    target = execute.add_mutually_exclusive_group()
    target.add_argument("--schedule-id")
    target.add_argument("--all", action="store_true")
    execute.add_argument("--setup-metadata", type=Path)
    execute.add_argument(
        "--no-source-check",
        action="store_true",
        help="pilot diagnosis only; records a deviation and remains forbidden for locked runs",
    )
    return parser


def _canonical(value: object) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise SystemExit(f"frozen execution configuration is invalid: {error}") from error


def _frozen_dataclass(
    metadata: Mapping[str, object], key: str, config_type: type
) -> object:
    """Construct one complete config from manifest metadata, with no defaults."""

    raw = metadata.get(key)
    expected = {item.name for item in fields(config_type)}
    if not isinstance(raw, dict) or set(raw) != expected:
        missing = sorted(expected - set(raw or ())) if isinstance(raw, dict) else sorted(expected)
        extra = sorted(set(raw or ()) - expected) if isinstance(raw, dict) else []
        raise SystemExit(
            f"manifest metadata.{key} must contain every exact config field "
            f"(missing={missing}, extra={extra})"
        )
    try:
        config = config_type(**raw)
    except (TypeError, ValueError) as error:
        raise SystemExit(f"manifest metadata.{key} is invalid: {error}") from error
    if _canonical(asdict(config)) != _canonical(raw):
        raise SystemExit(f"manifest metadata.{key} is non-canonical")
    return config


def _validate_loopback_service_endpoints(metadata: Mapping[str, object]) -> None:
    endpoints = metadata.get("service_endpoints")
    if not isinstance(endpoints, dict) or not endpoints:
        raise SystemExit("manifest metadata.service_endpoints must be a non-empty object")
    for label, value in endpoints.items():
        if not isinstance(label, str) or not isinstance(value, str):
            raise SystemExit("frozen service endpoint labels and URLs must be strings")
        parsed = urlsplit(value)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise SystemExit(f"frozen service endpoint {label!r} is not a safe HTTP URL")
        if parsed.hostname.casefold() == "localhost":
            continue
        try:
            address = ipaddress.ip_address(parsed.hostname)
        except ValueError as error:
            raise SystemExit(
                f"frozen service endpoint {label!r} must use a loopback host"
            ) from error
        if not address.is_loopback:
            raise SystemExit(
                f"frozen service endpoint {label!r} must use a loopback host"
            )


def _print(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, default=str))


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    registry = CampaignRegistry(arguments.campaign)
    if registry.mode.value == "locked" and arguments.command in {
        "begin",
        "seal",
        "reconcile",
    }:
        raise SystemExit(
            "manual locked attempt transitions are disabled; use `execute --engine "
            "production-mujoco` so source, instance config, video, and artifacts remain bound"
        )
    if arguments.command == "status":
        _print(registry.completion())
    elif arguments.command == "next":
        entry = registry.next_entry()
        _print(None if entry is None else entry.to_dict())
    elif arguments.command == "authorize":
        approval = registry.authorize_locked(arguments.approval)
        _print(
            {
                "authorized": True,
                "mode": registry.mode.value,
                "approval": None if approval is None else asdict(approval),
            }
        )
    elif arguments.command == "begin":
        lease = registry.begin_attempt(arguments.schedule_id)
        _print(
            {
                **asdict(lease),
                "assignment": lease.assignment.to_dict(),
                "attempt_path": str(lease.attempt_path),
                "next_step": (
                    "create AttemptRecorder at this exact schedule/attempt, then seal once"
                ),
            }
        )
    elif arguments.command == "seal":
        artifact_audit: bool | None = None
        if arguments.artifact_audit_passed:
            artifact_audit = True
        elif arguments.artifact_audit_failed:
            artifact_audit = False
        registry.seal_attempt(
            arguments.schedule_id,
            arguments.attempt_number,
            arguments.status,
            oracle_verdict=arguments.oracle_verdict,
            reason=arguments.reason,
            artifact_audit_passed=artifact_audit,
            independently_evidenced_infrastructure=(
                arguments.independently_evidenced_infrastructure
            ),
            infrastructure_evidence=arguments.infrastructure_evidence,
        )
        _print(registry.completion())
    elif arguments.command == "reconcile":
        _print(
            {
                "reconciled": registry.reconcile_open_attempts(
                    reason=arguments.reason,
                    status=arguments.status,
                    independently_evidenced_infrastructure=(
                        arguments.independently_evidenced_infrastructure
                    ),
                    infrastructure_evidence=arguments.infrastructure_evidence,
                ),
                "disposition": arguments.status,
                "completion": registry.completion(),
            }
        )
    elif arguments.command == "authorize-retry":
        _print(
            {
                "retry_authorization": registry.authorize_infrastructure_retry(
                    arguments.schedule_id, arguments.recovery_evidence_path
                ),
                "completion": registry.completion(),
            }
        )
    elif arguments.command == "index":
        _print({"run_index": str(registry.export_run_index())})
    elif arguments.command == "execute":
        production = arguments.engine == "production-mujoco"
        if registry.mode.value != "pilot" and not production:
            raise SystemExit(
                "the bundled scripted/MuJoCo engines are instrumentation pilots only; "
                "locked execution requires --engine production-mujoco"
            )
        if registry.mode.value != "pilot" and arguments.no_source_check:
            raise SystemExit("locked execution cannot disable live-source verification")
        if registry.mode.value != "pilot" and arguments.setup_metadata:
            raise SystemExit(
                "locked setup metadata is read only from CAMPAIGN_MANIFEST.json"
            )
        try:
            from experiments.harness.engines import (
                LiveModelServicePilotEngine,
                MujocoRecordingPilotEngine,
                ScriptedEvidencePilotEngine,
            )
            from experiments.harness.production_engine import (
                ProductionEngineConfig,
                ProductionMujocoEngine,
            )
            from experiments.harness.runner import CampaignRunner
            from experiments.harness.video import VideoConfig
        except ModuleNotFoundError:
            from harness.engines import (  # type: ignore[no-redef]
                LiveModelServicePilotEngine,
                MujocoRecordingPilotEngine,
                ScriptedEvidencePilotEngine,
            )
            from harness.production_engine import (  # type: ignore[no-redef]
                ProductionEngineConfig,
                ProductionMujocoEngine,
            )
            from harness.runner import CampaignRunner  # type: ignore[no-redef]
            from harness.video import VideoConfig  # type: ignore[no-redef]

        manifest = verify_frozen_campaign(
            arguments.campaign, check_source=not arguments.no_source_check
        )
        manifest_metadata = manifest.get("metadata")
        if not isinstance(manifest_metadata, dict):
            raise SystemExit("campaign manifest metadata is malformed")

        metadata: dict[str, object]
        video_config: VideoConfig | None = None
        if production:
            if arguments.setup_metadata:
                raise SystemExit(
                    "production-mujoco setup metadata is read only from "
                    "CAMPAIGN_MANIFEST.json"
                )
            metadata = dict(manifest_metadata)
            _validate_loopback_service_endpoints(manifest_metadata)
            production_config = _frozen_dataclass(
                manifest_metadata, "production_engine_config", ProductionEngineConfig
            )
            frozen_video = _frozen_dataclass(
                manifest_metadata, "video_config", VideoConfig
            )
            assert isinstance(production_config, ProductionEngineConfig)
            assert isinstance(frozen_video, VideoConfig)
            engine = ProductionMujocoEngine(config=production_config)
            # Locked CampaignRunner deliberately resolves this from the manifest
            # itself.  Pilot production runs receive the same frozen object.
            video_config = frozen_video if registry.mode.value == "pilot" else None
        else:
            metadata = dict(manifest_metadata)
        if arguments.setup_metadata and not production:
            loaded = json.loads(arguments.setup_metadata.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise SystemExit("--setup-metadata must contain a JSON object")
            if _canonical(loaded) != _canonical(manifest_metadata):
                raise SystemExit(
                    "--setup-metadata differs from CAMPAIGN_MANIFEST.json; "
                    "freeze it first instead of overriding an attempt"
                )
        if production:
            pass
        elif arguments.engine == "service-live":
            engine = LiveModelServicePilotEngine()
        elif arguments.engine.startswith("scripted"):
            engine = ScriptedEvidencePilotEngine(
                controlled_failure=arguments.engine.endswith("failure")
            )
        else:
            engine = MujocoRecordingPilotEngine(
                controlled_failure=arguments.engine.endswith("failure")
            )
        reports = []
        stopped_for_infrastructure = False
        runner = None
        try:
            runner = CampaignRunner(
                arguments.campaign,
                engine=engine,
                setup_metadata=(
                    None
                    if production and registry.mode.value != "pilot"
                    else metadata
                ),
                video_config=video_config,
                verify_source=not arguments.no_source_check,
            )
            if arguments.all:
                while (report := runner.run_next()) is not None:
                    reports.append(asdict(report))
                    if report.run_status.value == "INFRA_INTERRUPTED":
                        # Restoration, two health checks, and smoke inference must
                        # occur before a fresh attempt.  Never burn the remaining
                        # retry allowance in a tight loop against a failed service.
                        stopped_for_infrastructure = True
                        break
            else:
                report = (
                    runner.run_schedule_id(arguments.schedule_id)
                    if arguments.schedule_id
                    else runner.run_next()
                )
                if report is not None:
                    reports.append(asdict(report))
        finally:
            if runner is not None:
                runner.close()
            else:
                close = getattr(engine, "close", None)
                if callable(close):
                    close()
        _print(
            {
                "engine": arguments.engine,
                "campaign_mode": registry.mode.value,
                "publication_valid": production,
                "pilot_only": not production,
                "attempts": reports,
                "stopped_for_infrastructure": stopped_for_infrastructure,
                "completion": registry.completion(),
            }
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
