"""Independent integrity and cross-link audit for experiment artifacts."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any
from urllib.parse import unquote, urlsplit


_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\((?P<target>[^)]+)\)")
_EXPLICIT_ANCHOR = re.compile(
    r"<(?:a|span)\s+(?:[^>]*?\s)?id=[\"'](?P<id>[^\"']+)[\"'][^>]*>",
    flags=re.IGNORECASE,
)
_HEADING = re.compile(r"^#{1,6}\s+(?P<title>.+?)\s*#*\s*$", flags=re.MULTILINE)
_MEDIA_TIME = re.compile(r"^t=(?P<seconds>(?:0|[1-9]\d*)(?:\.\d+)?)$")
_CAMPAIGN_CHECKSUM_EXCLUSIONS = frozenset(
    {
        ".run_state.lock",
        "artifact_audit/campaign_checksums.sha256",
        "artifact_audit/audit.json",
        "artifact_audit/audit.sha256",
    }
)


class AuditError(ValueError):
    """Raised for an unsafe or malformed audit request."""


def _sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_target(root: Path, relative: str) -> tuple[Path | None, str | None]:
    if not relative or "\x00" in relative or "\n" in relative or "\r" in relative:
        return None, "empty or control-character path"
    candidate_path = Path(relative)
    if candidate_path.is_absolute() or ".." in candidate_path.parts:
        return None, "absolute or parent-traversing path"
    candidate = root / candidate_path
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root.resolve())
    except (OSError, ValueError):
        return None, "missing target or target outside audit root"
    if candidate.is_symlink() or not candidate.is_file():
        return None, "target is not a regular non-symlink file"
    return candidate, None


def verify_checksum_manifest(
    root: str | Path,
    checksum_file: str | Path = "checksums.sha256",
    *,
    require_complete: bool = True,
    excluded_paths: Iterable[str] = (),
) -> dict[str, Any]:
    """Verify GNU-style SHA-256 entries without trusting manifest paths."""

    base = Path(root).resolve()
    manifest = Path(checksum_file)
    if not manifest.is_absolute():
        manifest = base / manifest
    errors: list[str] = []
    verified = 0
    covered: set[str] = set()
    if not manifest.is_file() or manifest.is_symlink():
        return {
            "passed": False,
            "verified": 0,
            "covered": [],
            "errors": [f"missing or unsafe checksum manifest: {manifest.name}"],
        }
    try:
        manifest_relative = manifest.resolve().relative_to(base).as_posix()
    except ValueError as error:
        raise AuditError("checksum manifest must live under the audited root") from error
    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        return {
            "passed": False,
            "verified": 0,
            "covered": [],
            "errors": [f"cannot read checksum manifest: {error}"],
        }
    for line_number, line in enumerate(lines, start=1):
        if not line:
            errors.append(f"blank checksum line {line_number}")
            continue
        digest, separator, relative = line.partition("  ")
        if not separator or _DIGEST.fullmatch(digest) is None:
            errors.append(f"invalid checksum syntax on line {line_number}")
            continue
        if relative in covered:
            errors.append(f"duplicate checksum target: {relative}")
            continue
        target, unsafe = _safe_target(base, relative)
        if unsafe:
            errors.append(f"unsafe checksum target {relative!r}: {unsafe}")
            continue
        covered.add(relative)
        assert target is not None
        actual = _sha256(target)
        if actual != digest:
            errors.append(f"checksum mismatch: {relative}")
        else:
            verified += 1

    excluded = set(excluded_paths) | {manifest_relative}
    if require_complete:
        expected: set[str] = set()
        for path in base.rglob("*"):
            relative = path.relative_to(base).as_posix()
            if path.is_symlink():
                errors.append(f"symlink artifact is forbidden: {relative}")
            elif path.is_file() and relative not in excluded:
                expected.add(relative)
        for missing in sorted(expected - covered):
            errors.append(f"artifact missing from checksums: {missing}")
        for extra in sorted(covered - expected):
            if extra not in excluded:
                errors.append(f"checksum covers an unexpected artifact: {extra}")
    return {
        "passed": not errors,
        "verified": verified,
        "covered": sorted(covered),
        "errors": errors,
    }


def write_checksum_manifest(
    root: str | Path,
    checksum_file: str | Path = "checksums.sha256",
    *,
    excluded_paths: Iterable[str] = (),
) -> Path:
    """Write a sorted checksum manifest, refusing symlink artifacts."""

    base = Path(root).resolve()
    manifest = Path(checksum_file)
    if not manifest.is_absolute():
        manifest = base / manifest
    try:
        manifest_relative = manifest.resolve(strict=False).relative_to(base).as_posix()
    except ValueError as error:
        raise AuditError("checksum manifest must live under the audited root") from error
    excluded = set(excluded_paths) | {manifest_relative}
    if manifest.exists():
        existing = verify_checksum_manifest(
            base,
            manifest,
            excluded_paths=excluded_paths,
        )
        if existing.get("passed") is not True:
            raise AuditError(
                "refusing to overwrite a failing immutable checksum manifest: "
                + "; ".join(map(str, existing.get("errors", ())))
            )
        return manifest
    entries: list[str] = []
    for path in sorted(base.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(base).as_posix()
        if path.is_symlink():
            raise AuditError(f"refusing to checksum symlink artifact: {relative}")
        if not path.is_file() or relative in excluded:
            continue
        if "\n" in relative or "\r" in relative:
            raise AuditError("artifact path contains a newline")
        entries.append(f"{_sha256(path)}  {relative}")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(entries) + "\n")
    return manifest


def write_campaign_checksums(campaign_root: str | Path) -> Path:
    return write_checksum_manifest(
        campaign_root,
        "artifact_audit/campaign_checksums.sha256",
        excluded_paths=_CAMPAIGN_CHECKSUM_EXCLUSIONS,
    )


def _slug(value: str) -> str:
    # Close enough to GitHub's documented heading IDs for protocol-authored
    # ASCII headings; explicit event anchors are parsed separately.
    text = re.sub(r"<[^>]+>", "", value).strip().casefold()
    text = re.sub(r"[`*_~]", "", text)
    text = re.sub(r"[^\w\- ]", "", text)
    return re.sub(r"\s+", "-", text)


def _markdown_prose(source: str) -> str:
    """Remove fenced and inline code before interpreting Markdown semantics."""

    kept: list[str] = []
    fence_character: str | None = None
    fence_length = 0
    for line in source.splitlines(keepends=True):
        marker = re.match(r"^[ \t]*(?P<run>`{3,}|~{3,})", line)
        if fence_character is not None:
            if (
                marker is not None
                and marker.group("run")[0] == fence_character
                and len(marker.group("run")) >= fence_length
            ):
                fence_character = None
                fence_length = 0
            kept.append("\n" if line.endswith("\n") else "")
            continue
        if marker is not None:
            fence_character = marker.group("run")[0]
            fence_length = len(marker.group("run"))
            kept.append("\n" if line.endswith("\n") else "")
            continue
        kept.append(re.sub(r"`+[^`\n]*`+", "", line))
    return "".join(kept)


def markdown_anchors(path: str | Path) -> frozenset[str]:
    source = _markdown_prose(Path(path).read_text(encoding="utf-8"))
    anchors = {match.group("id") for match in _EXPLICIT_ANCHOR.finditer(source)}
    seen: Counter[str] = Counter()
    for match in _HEADING.finditer(source):
        base = _slug(match.group("title"))
        suffix = seen[base]
        seen[base] += 1
        anchors.add(base if suffix == 0 else f"{base}-{suffix}")
    return frozenset(anchors)


def _link_destination(raw: str) -> str:
    value = raw.strip()
    if value.startswith("<") and ">" in value:
        return value[1 : value.index(">")]
    # Markdown permits an optional title after whitespace. Paths containing
    # spaces must be percent-encoded or enclosed in angle brackets.
    return value.split(maxsplit=1)[0]


def audit_crosslinks(
    root: str | Path,
    *,
    markdown_files: Sequence[str | Path] | None = None,
) -> dict[str, Any]:
    """Validate all local Markdown targets and Markdown fragments."""

    base = Path(root).resolve()
    if markdown_files is None:
        paths = sorted(base.rglob("*.md"))
    else:
        paths = [
            path if Path(path).is_absolute() else base / Path(path)
            for path in markdown_files
        ]
    errors: list[str] = []
    warnings: list[str] = []
    checked = 0
    anchor_cache: dict[Path, frozenset[str]] = {}
    for source in paths:
        source = Path(source)
        try:
            relative_source = source.resolve(strict=True).relative_to(base).as_posix()
            text = source.read_text(encoding="utf-8")
        except (OSError, UnicodeError, ValueError) as error:
            errors.append(f"cannot read Markdown source {source}: {error}")
            continue
        text = _markdown_prose(text)
        for match in _MARKDOWN_LINK.finditer(text):
            destination = _link_destination(match.group("target"))
            parsed = urlsplit(destination)
            if parsed.scheme in {"http", "https", "mailto"} and not parsed.netloc.startswith(
                "."
            ):
                # External availability is intentionally outside a reproducible
                # offline artifact audit.
                continue
            if parsed.scheme or parsed.netloc:
                errors.append(
                    f"{relative_source}: unsafe or unsupported link scheme {destination!r}"
                )
                continue
            checked += 1
            decoded_path = unquote(parsed.path)
            fragment = unquote(parsed.fragment)
            if not decoded_path:
                target = source
            else:
                candidate = Path(decoded_path)
                if candidate.is_absolute():
                    errors.append(
                        f"{relative_source}: non-portable or escaping link {destination!r}"
                    )
                    continue
                target = source.parent / candidate
            try:
                resolved = target.resolve(strict=True)
                resolved.relative_to(base)
            except (OSError, ValueError):
                errors.append(f"{relative_source}: missing link target {destination!r}")
                continue
            if target.is_symlink():
                errors.append(f"{relative_source}: link target is a symlink {destination!r}")
                continue
            if fragment and resolved.suffix.casefold() in {".md", ".markdown"}:
                if resolved not in anchor_cache:
                    try:
                        anchor_cache[resolved] = markdown_anchors(resolved)
                    except (OSError, UnicodeError) as error:
                        errors.append(f"cannot parse anchors in {resolved}: {error}")
                        continue
                if fragment not in anchor_cache[resolved]:
                    errors.append(
                        f"{relative_source}: missing Markdown anchor #{fragment} "
                        f"in {resolved.relative_to(base).as_posix()}"
                    )
            elif fragment and resolved.suffix.casefold() not in {
                ".mp4",
                ".webm",
                ".mov",
            }:
                warnings.append(
                    f"{relative_source}: fragment on non-Markdown target was not interpreted: "
                    f"{destination!r}"
                )
            elif fragment and resolved.suffix.casefold() in {".mp4", ".webm", ".mov"}:
                match = _MEDIA_TIME.fullmatch(fragment)
                if match is None or not math.isfinite(float(match.group("seconds"))):
                    errors.append(
                        f"{relative_source}: invalid media-time fragment {destination!r}"
                    )
    return {
        "passed": not errors,
        "markdown_files": len(paths),
        "links_checked": checked,
        "errors": errors,
        "warnings": warnings,
    }


def audit_diagnosis_evidence(root: str | Path) -> dict[str, Any]:
    """Require every factual diagnosis section to cite raw evidence directly."""

    base = Path(root).resolve()
    errors: list[str] = []
    checked = 0
    evidence_kinds: Counter[str] = Counter()
    for path in sorted(base.rglob("diagnosis.md")):
        checked += 1
        relative = path.relative_to(base).as_posix()
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            errors.append(f"cannot read {relative}: {error}")
            continue
        prose = _markdown_prose(text)

        def evidence_in(fragment_text: str) -> set[str]:
            found: set[str] = set()
            for match in _MARKDOWN_LINK.finditer(fragment_text):
                destination = _link_destination(match.group("target"))
                parsed = urlsplit(destination)
                if parsed.scheme or parsed.netloc:
                    continue
                decoded_target = unquote(parsed.path)
                target_path = decoded_target.casefold()
                link_fragment = unquote(parsed.fragment).casefold()
                candidate = path if not decoded_target else path.parent / decoded_target
                try:
                    resolved = candidate.resolve(strict=True)
                    resolved.relative_to(base)
                except (OSError, ValueError):
                    continue
                if candidate.is_symlink() or not resolved.is_file():
                    continue
                if target_path.endswith(
                    "experiment_record.md"
                ) and link_fragment.startswith("event-e"):
                    found.add("event")
                elif "/frames/" in "/" + target_path:
                    found.add("request_frame")
                elif "/model_calls/" in "/" + target_path:
                    found.add("model_call")
                elif (
                    target_path.endswith(("video.mp4", "robot_camera.mp4"))
                    and _MEDIA_TIME.fullmatch(link_fragment) is not None
                ):
                    found.add("video_time")
            return found

        evidence = evidence_in(prose)
        if not evidence:
            errors.append(
                f"{relative}: diagnosis has no event, frame, model-call, or video evidence link"
            )
        sections = re.split(r"(?m)^##\s+(.+?)\s*$", prose)[1:]
        for label, body in zip(sections[0::2], sections[1::2]):
            normalized = label.strip().casefold()
            if normalized in {"run status", "oracle verdict"}:
                continue
            if not evidence_in(body):
                errors.append(
                    f"{relative}: factual diagnosis section lacks direct evidence: "
                    f"{label.strip()}"
                )
        evidence_kinds.update(evidence)
    return {
        "passed": not errors,
        "diagnoses_checked": checked,
        "evidence_kind_counts": dict(sorted(evidence_kinds.items())),
        "errors": errors,
    }


def audit_attempt_artifacts(
    attempt_dir: str | Path,
    *,
    verify_video: bool = True,
    require_final_state: bool = False,
) -> dict[str, Any]:
    """Run both the recorder's semantic audit and an independent link audit."""

    root = Path(attempt_dir).resolve()
    try:
        from .recording import audit_attempt as recorder_audit

        structural = recorder_audit(
            root,
            verify_video=verify_video,
            require_final_state=require_final_state,
        )
    except Exception as error:  # Audit must preserve and report partial evidence.
        structural = {
            "passed": False,
            "errors": [f"recorder audit raised {type(error).__name__}: {error}"],
            "warnings": [],
        }
    checksums = verify_checksum_manifest(root)
    crosslinks = audit_crosslinks(root)
    diagnosis_evidence = audit_diagnosis_evidence(root)
    errors = [
        *map(str, structural.get("errors", [])),
        *map(str, checksums.get("errors", [])),
        *map(str, crosslinks.get("errors", [])),
        *map(str, diagnosis_evidence.get("errors", [])),
    ]
    warnings = [
        *map(str, structural.get("warnings", [])),
        *map(str, crosslinks.get("warnings", [])),
    ]
    return {
        "passed": not errors,
        "attempt_dir": root.name,
        "structural": structural,
        "checksums": checksums,
        "crosslinks": crosslinks,
        "diagnosis_evidence": diagnosis_evidence,
        "errors": errors,
        "warnings": warnings,
    }


def _required_campaign_paths(root: Path) -> tuple[list[str], list[str]]:
    files = [
        "CAMPAIGN_MANIFEST.json",
        "PROTOCOL.md",
        "PROTOCOL.sha256",
        "REQUIREMENTS_CROSSWALK.md",
        "RUN_STATE.json",
        "run_schedule.csv",
        "run_index.csv",
        "EXCLUSIONS.md",
        "CHANGELOG.md",
    ]
    directories = ["environment", "analysis", "artifact_audit", "episodes"]
    errors: list[str] = []
    checked: list[str] = []
    for relative in files:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            errors.append(f"missing or unsafe campaign artifact: {relative}")
        elif path.stat().st_size == 0:
            errors.append(f"empty campaign artifact: {relative}")
        else:
            checked.append(relative)
    for relative in directories:
        path = root / relative
        if not path.is_dir() or path.is_symlink():
            errors.append(f"missing or unsafe campaign directory: {relative}/")
        else:
            checked.append(relative + "/")
    return errors, checked


def _registry_audit(root: Path) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    try:
        from .campaign import CampaignRegistry, RunStatus, run_index_csv_bytes

        registry = CampaignRegistry(root)
        state = registry.snapshot()
        completion = registry.completion()
    except Exception as error:
        return {
            "passed": False,
            "errors": [f"registry validation failed: {type(error).__name__}: {error}"],
            "warnings": [],
        }
    try:
        manifest = json.loads(
            (root / "CAMPAIGN_MANIFEST.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        errors.append(f"cannot cross-check registry against manifest: {error}")
        manifest = {}
    identity_pairs = {
        "campaign_id": (state.get("campaign_id"), manifest.get("campaign_id")),
        "mode": (state.get("mode"), manifest.get("mode")),
        "protocol_sha256": (
            state.get("protocol_sha256"),
            manifest.get("protocol_sha256"),
        ),
        "schedule_sha256": (
            state.get("schedule_sha256"),
            manifest.get("schedule_sha256"),
        ),
        "schedule_size": (state.get("schedule_size"), manifest.get("schedule_size")),
        "reuse_policy": (state.get("reuse_policy"), manifest.get("reuse_policy")),
    }
    for label, (registry_value, manifest_value) in identity_pairs.items():
        if registry_value != manifest_value:
            errors.append(f"manifest/registry {label} mismatch")
    index_path = root / "run_index.csv"
    if index_path.is_file():
        try:
            actual_index = index_path.read_bytes()
            expected_index = run_index_csv_bytes(state)
        except OSError as error:
            errors.append(f"cannot cross-check run_index.csv: {error}")
        else:
            if actual_index != expected_index:
                errors.append("run_index.csv differs from authoritative RUN_STATE.json")
    expected_attempt_paths: set[str] = set()
    expected_exclusions: set[tuple[str, int, str]] = set()
    for schedule_id, entry in state["entries"].items():
        for attempt in entry["attempts"]:
            expected = attempt["artifact_path"]
            expected_attempt_paths.add(expected)
            if attempt["status"] in {
                RunStatus.INFRA_INTERRUPTED.value,
                RunStatus.INVALID_HARNESS.value,
                RunStatus.ABORTED_SAFETY.value,
            }:
                expected_exclusions.add(
                    (schedule_id, int(attempt["attempt_number"]), attempt["status"])
                )
            path = root / expected
            if not path.is_dir() or path.is_symlink():
                errors.append(
                    f"registry attempt has no preserved directory: {schedule_id} "
                    f"attempt {attempt['attempt_number']}"
                )
                continue
            evidence = attempt.get("infrastructure_evidence")
            if evidence is not None:
                evidence_path = path / str(evidence.get("path", ""))
                try:
                    resolved_evidence = evidence_path.resolve(strict=True)
                    resolved_evidence.relative_to(path.resolve())
                except (OSError, ValueError):
                    errors.append(f"missing/unsafe infrastructure evidence for {expected}")
                else:
                    if (
                        evidence_path.is_symlink()
                        or not resolved_evidence.is_file()
                        or _sha256(resolved_evidence) != evidence.get("sha256")
                    ):
                        errors.append(
                            f"infrastructure evidence hash mismatch for {expected}"
                        )
            recovery = attempt.get("recovery_authorization")
            if recovery is not None:
                recovery_path = root / str(recovery.get("path", ""))
                try:
                    resolved_recovery = recovery_path.resolve(strict=True)
                    resolved_recovery.relative_to(root)
                except (OSError, ValueError):
                    errors.append(f"missing/unsafe consumed recovery evidence for {expected}")
                else:
                    if (
                        recovery_path.is_symlink()
                        or not resolved_recovery.is_file()
                        or _sha256(resolved_recovery) != recovery.get("sha256")
                    ):
                        errors.append(
                            f"consumed recovery evidence hash mismatch for {expected}"
                        )
            result_path = path / "results.json"
            if result_path.is_file():
                try:
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                except (OSError, UnicodeError, json.JSONDecodeError) as error:
                    errors.append(f"invalid result for {expected}: {error}")
                else:
                    observed = result.get("run_status") if isinstance(result, dict) else None
                    if observed != attempt["status"]:
                        errors.append(
                            f"registry/result status mismatch for {expected}: "
                            f"{attempt['status']} != {observed}"
                        )
                    observed_verdict = (
                        result.get("oracle_verdict") if isinstance(result, dict) else None
                    )
                    if observed_verdict != attempt.get("oracle_verdict"):
                        errors.append(f"registry/result oracle mismatch for {expected}")
                    if attempt["status"] in {
                        RunStatus.VALID_PASS.value,
                        RunStatus.VALID_SYSTEM_FAILURE.value,
                    }:
                        expected_success = (
                            attempt["status"] == RunStatus.VALID_PASS.value
                        )
                        if (
                            isinstance(result, dict)
                            and result.get("contract_success") is not None
                            and result.get("contract_success") is not expected_success
                        ):
                            errors.append(
                                f"registry/result contract-success contradiction for {expected}"
                            )
                    expected_result_identity = {
                        "attempt_id": attempt["attempt_id"],
                        "scenario_id": entry["assignment"]["scenario_id"],
                        "profile_id": entry["assignment"]["profile_id"],
                    }
                    if isinstance(result, dict):
                        for label, expected_value in expected_result_identity.items():
                            if result.get(label) != expected_value:
                                errors.append(
                                    f"registry/result {label} mismatch for {expected}"
                                )
            elif attempt["status"] in {
                RunStatus.VALID_PASS.value,
                RunStatus.VALID_SYSTEM_FAILURE.value,
            }:
                errors.append(f"valid outcome lacks results.json: {expected}")
            setup_path = path / "setup.json"
            if setup_path.is_file():
                try:
                    setup = json.loads(setup_path.read_text(encoding="utf-8"))
                except (OSError, UnicodeError, json.JSONDecodeError) as error:
                    errors.append(f"invalid setup for {expected}: {error}")
                else:
                    expected_setup_identity = {
                        "campaign_id": state["campaign_id"],
                        "schedule_id": schedule_id,
                        "attempt_id": attempt["attempt_id"],
                        "attempt_number": attempt["attempt_number"],
                    }
                    for label, expected_value in expected_setup_identity.items():
                        if not isinstance(setup, dict) or setup.get(label) != expected_value:
                            errors.append(
                                f"registry/setup {label} mismatch for {expected}"
                            )
                    if isinstance(setup, dict) and setup.get("assignment") != entry[
                        "assignment"
                    ]:
                        errors.append(f"registry/setup assignment mismatch for {expected}")
        if entry["status"] == RunStatus.INFRA_INTERRUPTED.value:
            latest = entry["attempts"][-1]
            if not latest.get("reason"):
                errors.append(f"infrastructure interruption lacks reason: {schedule_id}")
            if len(entry["attempts"]) == state["max_infrastructure_attempts"]:
                if entry["disposition"] != "QUARANTINED_INFRA_RETRY_EXHAUSTED":
                    errors.append(f"exhausted infrastructure entry is not quarantined: {schedule_id}")
        pending_recovery = entry.get("retry_authorization")
        if pending_recovery is not None:
            recovery_path = root / str(pending_recovery.get("path", ""))
            try:
                resolved_recovery = recovery_path.resolve(strict=True)
                resolved_recovery.relative_to(root)
            except (OSError, ValueError):
                errors.append(f"missing/unsafe pending recovery evidence: {schedule_id}")
            else:
                if (
                    recovery_path.is_symlink()
                    or not resolved_recovery.is_file()
                    or _sha256(resolved_recovery) != pending_recovery.get("sha256")
                ):
                    errors.append(
                        f"pending recovery evidence hash mismatch: {schedule_id}"
                    )

    actual_attempt_paths: set[str] = set()
    episodes = root / "episodes"
    if episodes.is_dir():
        for path in episodes.glob("*/attempt_*"):
            if path.is_dir():
                actual_attempt_paths.add(path.relative_to(root).as_posix())
    for unexpected in sorted(actual_attempt_paths - expected_attempt_paths):
        errors.append(f"attempt directory is absent from registry: {unexpected}")
    exclusions_path = root / "EXCLUSIONS.md"
    if exclusions_path.is_file():
        try:
            exclusion_text = exclusions_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            errors.append(f"cannot parse EXCLUSIONS.md: {error}")
        else:
            observed_exclusions = {
                (schedule, int(attempt), status)
                for schedule, attempt, status in re.findall(
                    r"(?m)^- EXCLUSION schedule_id=([^\s]+) attempt=(\d+) "
                    r"status=([^\s]+) ",
                    exclusion_text,
                )
            }
            for missing in sorted(expected_exclusions - observed_exclusions):
                errors.append(
                    "EXCLUSIONS.md lacks registry disposition: "
                    f"{missing[0]} attempt {missing[1]} {missing[2]}"
                )
            for unexpected in sorted(observed_exclusions - expected_exclusions):
                errors.append(
                    "EXCLUSIONS.md contains an unregistered disposition: "
                    f"{unexpected[0]} attempt {unexpected[1]} {unexpected[2]}"
                )
    return {
        "passed": not errors,
        "completion": completion,
        "registered_attempts": len(expected_attempt_paths),
        "preserved_attempt_directories": len(actual_attempt_paths),
        "errors": errors,
        "warnings": warnings,
    }


def audit_campaign(
    campaign_root: str | Path,
    *,
    verify_video: bool = True,
    require_complete: bool = True,
    check_live_source: bool = False,
) -> dict[str, Any]:
    """Audit a campaign without changing it or hiding partial attempts."""

    root = Path(campaign_root).resolve()
    started = datetime.now(timezone.utc)
    errors, checked = _required_campaign_paths(root)
    warnings: list[str] = []

    manifest_check: dict[str, Any]
    validated_manifest: Mapping[str, Any] | None = None
    try:
        from .campaign import verify_frozen_campaign

        validated_manifest = verify_frozen_campaign(
            root, check_source=check_live_source
        )
        manifest_check = {
            "passed": True,
            "campaign_id": validated_manifest["campaign_id"],
            "protocol_sha256": validated_manifest["protocol_sha256"],
            "manifest_sha256": validated_manifest["manifest_sha256"],
            "live_source_checked": check_live_source,
        }
        provenance_files = {
            item.get("path"): item.get("sha256")
            for item in validated_manifest.get("provenance", {}).get(
                "source_files", []
            )
            if isinstance(item, Mapping)
        }
        documentation_snapshot = validated_manifest.get("documentation_snapshot")
        if documentation_snapshot is None:
            # Legacy freezes copied source bytes directly. New freezes bind the
            # exact source and rewritten, self-contained entry independently.
            expected_crosswalk = provenance_files.get(
                "experiments/REQUIREMENTS_CROSSWALK.md"
            ) or provenance_files.get("experiments/CLAIM_EXPERIMENT_CROSSWALK.md")
            crosswalk_path = root / "REQUIREMENTS_CROSSWALK.md"
            if expected_crosswalk is None:
                message = "source provenance does not bind a requirements crosswalk"
                if validated_manifest.get("mode") == "locked":
                    errors.append(message)
                else:
                    warnings.append(message)
            elif crosswalk_path.is_file() and _sha256(
                crosswalk_path
            ) != expected_crosswalk:
                errors.append(
                    "campaign REQUIREMENTS_CROSSWALK.md differs from its frozen source"
                )
        elif documentation_snapshot.get("source_entry_path") is None:
            message = "documentation snapshot has no canonical requirements source"
            if validated_manifest.get("mode") == "locked":
                errors.append(message)
            else:
                warnings.append(message)
    except Exception as error:
        manifest_check = {
            "passed": False,
            "live_source_checked": check_live_source,
            "errors": [f"{type(error).__name__}: {error}"],
        }
        errors.extend(manifest_check["errors"])

    registry = _registry_audit(root)
    errors.extend(registry.get("errors", []))
    warnings.extend(registry.get("warnings", []))
    if require_complete and not registry.get("completion", {}).get("complete", False):
        errors.append("campaign has runnable, not-run, open, or unresolved schedule entries")
    campaign_mode = (
        validated_manifest.get("mode") if validated_manifest is not None else None
    )
    # Re-read only the mode when manifest validation failed before producing a
    # manifest object; the error remains independently visible.
    if not campaign_mode:
        try:
            campaign_mode = json.loads(
                (root / "CAMPAIGN_MANIFEST.json").read_text(encoding="utf-8")
            ).get("mode")
        except Exception:
            campaign_mode = None
    if campaign_mode == "locked" and require_complete and not verify_video:
        errors.append("locked final audit may not disable video verification")
    if campaign_mode == "locked" and require_complete and not check_live_source:
        errors.append("locked final audit must verify live source against the freeze")

    analysis_check: dict[str, Any] = {
        "passed": not require_complete,
        "required_for_final_audit": require_complete,
        "errors": [],
    }
    if require_complete and registry.get("completion", {}).get("complete", False):
        analysis_errors: list[str] = []
        json_path = root / "analysis" / "analysis_results.json"
        markdown_path = root / "analysis" / "analysis_results.md"
        if not json_path.is_file() or json_path.is_symlink():
            analysis_errors.append("missing analysis/analysis_results.json")
        if not markdown_path.is_file() or markdown_path.is_symlink():
            analysis_errors.append("missing analysis/analysis_results.md")
        if not analysis_errors:
            try:
                from .analysis import analyze_campaign, render_analysis_markdown

                saved = json.loads(json_path.read_text(encoding="utf-8"))
                regenerated = analyze_campaign(
                    root, check_live_source=check_live_source
                )
                if saved != regenerated:
                    analysis_errors.append(
                        "saved analysis differs from immutable-attempt regeneration"
                    )
                if markdown_path.read_text(encoding="utf-8") != render_analysis_markdown(
                    regenerated
                ):
                    analysis_errors.append(
                        "analysis Markdown differs from machine-readable rendering"
                    )
                required_tier = (
                    "confirmatory_locked"
                    if campaign_mode == "locked"
                    else "pilot_nonconfirmatory"
                )
                if regenerated.get("analysis_tier") != required_tier:
                    analysis_errors.append(
                        f"final analysis tier is not {required_tier}"
                    )
            except Exception as error:
                analysis_errors.append(
                    f"analysis regeneration failed: {type(error).__name__}: {error}"
                )
        analysis_check = {
            "passed": not analysis_errors,
            "required_for_final_audit": True,
            "errors": analysis_errors,
        }
        errors.extend(analysis_errors)

    attempt_reports: list[dict[str, Any]] = []
    # Do not reparse an invalid registry after `_registry_audit` has reported
    # it.  Only a schema-validated snapshot is safe to drive filesystem reads.
    try:
        from .campaign import CampaignRegistry

        audit_state: Mapping[str, Any] = CampaignRegistry(root).snapshot()
    except Exception:
        audit_state = {}
    entries = audit_state.get("entries", {})
    if isinstance(entries, Mapping):
        for entry in entries.values():
            if not isinstance(entry, Mapping):
                continue
            assignment = entry.get("assignment", {})
            attempts = entry.get("attempts", ())
            if not isinstance(assignment, Mapping) or not isinstance(attempts, list):
                continue
            schedule_id = assignment.get("schedule_id")
            for attempt in attempts:
                if not isinstance(attempt, Mapping):
                    continue
                artifact_path = attempt.get("artifact_path")
                attempt_number = attempt.get("attempt_number")
                if (
                    not isinstance(schedule_id, str)
                    or not isinstance(artifact_path, str)
                    or not isinstance(attempt_number, int)
                ):
                    continue
                path = root / artifact_path
                if not path.is_dir():
                    continue
                report = audit_attempt_artifacts(
                    path,
                    verify_video=verify_video,
                    require_final_state=attempt.get("status")
                    in {"VALID_PASS", "VALID_SYSTEM_FAILURE"},
                )
                report["schedule_id"] = schedule_id
                report["attempt_number"] = attempt_number
                attempt_reports.append(report)
                registry_audit_value = attempt.get("artifact_audit_passed")
                if registry_audit_value is not None and bool(
                    registry_audit_value
                ) != bool(report["passed"]):
                    errors.append(
                        f"{schedule_id}/"
                        f"attempt_{attempt_number}: registry artifact-audit "
                        f"flag differs from independent audit"
                    )
                for message in report["errors"]:
                    errors.append(
                        f"{schedule_id}/"
                        f"attempt_{attempt_number}: {message}"
                    )
                warnings.extend(report["warnings"])

    crosslinks = audit_crosslinks(root)
    errors.extend(crosslinks["errors"])
    warnings.extend(crosslinks["warnings"])

    campaign_checksum_path = root / "artifact_audit" / "campaign_checksums.sha256"
    if campaign_checksum_path.is_file():
        campaign_checksums = verify_checksum_manifest(
            root,
            "artifact_audit/campaign_checksums.sha256",
            excluded_paths=_CAMPAIGN_CHECKSUM_EXCLUSIONS,
        )
        errors.extend(campaign_checksums["errors"])
    else:
        campaign_checksums = {
            "passed": False,
            "verified": 0,
            "errors": [
                "campaign checksum manifest has not been generated; run final audit with "
                "--write-checksums"
            ],
        }
        if require_complete:
            errors.extend(campaign_checksums["errors"])
        else:
            warnings.extend(campaign_checksums["errors"])

    ended = datetime.now(timezone.utc)
    return {
        "schema_version": 1,
        "passed": not errors,
        "campaign_root": root.name,
        "started_utc": started.isoformat().replace("+00:00", "Z"),
        "ended_utc": ended.isoformat().replace("+00:00", "Z"),
        "required_artifacts_checked": checked,
        "manifest": manifest_check,
        "registry": registry,
        "attempts": attempt_reports,
        "attempt_count": len(attempt_reports),
        "attempts_passed": sum(report["passed"] for report in attempt_reports),
        "video_verified": verify_video,
        "live_source_checked": check_live_source,
        "crosslinks": crosslinks,
        "campaign_checksums": campaign_checksums,
        "analysis": analysis_check,
        "errors": errors,
        "warnings": warnings,
    }


def write_audit_report(campaign_root: str | Path, report: Mapping[str, Any]) -> Path:
    output = Path(campaign_root).resolve() / "artifact_audit" / "audit.json"
    digest_path = output.with_name("audit.sha256")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    if output.exists() or digest_path.exists():
        if not output.is_file() or not digest_path.is_file():
            raise AuditError("immutable final audit report/digest pair is incomplete")
        expected = f"{_sha256(output)}  audit.json\n"
        if digest_path.read_text(encoding="ascii") != expected:
            raise AuditError("immutable final audit report digest mismatch")
        if output.read_bytes() != payload:
            raise AuditError(
                "refusing to overwrite an existing final audit; use a new campaign "
                "or explicitly version a diagnostic report"
            )
        return output
    with output.open("xb") as handle:
        handle.write(payload)
    with digest_path.open("x", encoding="ascii") as handle:
        handle.write(f"{_sha256(output)}  audit.json\n")
    return output


__all__ = [
    "AuditError",
    "audit_attempt_artifacts",
    "audit_campaign",
    "audit_crosslinks",
    "audit_diagnosis_evidence",
    "markdown_anchors",
    "verify_checksum_manifest",
    "write_audit_report",
    "write_campaign_checksums",
    "write_checksum_manifest",
]
