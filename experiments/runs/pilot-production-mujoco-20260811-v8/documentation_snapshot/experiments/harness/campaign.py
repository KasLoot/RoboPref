"""Deterministic campaign scheduling, protocol freezing, and run registry.

The registry is deliberately small and dependency free.  ``RUN_STATE.json`` is
an atomic snapshot of an append-only attempt history: an attempt is reserved
once, may be sealed once, and is never removed or rewritten after it is sealed.
Only an infrastructure interruption permits another attempt for the same
scheduled episode.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from enum import StrEnum
import csv
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import posixpath
import random
import re
import subprocess
import tempfile
from typing import Any, Iterator
from urllib.parse import quote, unquote, urlsplit


SCHEMA_VERSION = 1
MAX_INFRASTRUCTURE_ATTEMPTS = 3


class CampaignError(RuntimeError):
    """Base class for campaign-control failures."""


class ScheduleError(CampaignError, ValueError):
    """Raised when a schedule is incomplete or ambiguous."""


class RegistryError(CampaignError):
    """Raised when an append-only registry transition is invalid."""


class FreezeError(CampaignError):
    """Raised when frozen provenance is missing or has changed."""


class ApprovalError(CampaignError):
    """Raised when locked execution lacks the exact approval token."""


class CampaignMode(StrEnum):
    PILOT = "pilot"
    LOCKED = "locked"


class RunStatus(StrEnum):
    VALID_PASS = "VALID_PASS"
    VALID_SYSTEM_FAILURE = "VALID_SYSTEM_FAILURE"
    INFRA_INTERRUPTED = "INFRA_INTERRUPTED"
    INVALID_HARNESS = "INVALID_HARNESS"
    ABORTED_SAFETY = "ABORTED_SAFETY"
    NOT_RUN = "NOT_RUN"


class ReusePolicy(StrEnum):
    """Whether an episode may serve more than one predeclared contrast."""

    FRESH_ATTEMPTS_ONLY = "fresh_attempts_only"
    PREREGISTERED_SHARED_REFERENCE = "preregistered_shared_reference"


SEALED_STATUSES = frozenset(set(RunStatus) - {RunStatus.NOT_RUN})
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_APPROVAL = re.compile(
    r"APPROVE LOCKED CAMPAIGN "
    r"(?P<campaign_id>[A-Za-z0-9][A-Za-z0-9._-]{0,127}) "
    r"(?P<protocol_sha256>[0-9a-f]{64})"
)
_MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\((?P<target>[^)]+)\)")
_MARKDOWN_SUFFIXES = frozenset({".md", ".markdown"})
_DOCUMENTATION_SNAPSHOT_ROOT = "documentation_snapshot"

DEFAULT_ANALYSIS_SPEC: Mapping[str, Any] = {
    "schema_version": 1,
    "confirmatory_outcomes": ["contract_success"],
    "reference_profile": "T5",
    "experimental_unit": "episode",
    "pairing_key": "crn_key",
    "multiplicity_method": "Holm within matrix",
    "families": [],
}


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def canonical_json_bytes(value: Any) -> bytes:
    """Return the only JSON encoding used for hashes and registry snapshots."""

    try:
        rendered = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise CampaignError(f"value is not canonical JSON: {error}") from error
    return (rendered + "\n").encode("utf-8")


def _normalize_campaign_metadata(
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return canonical frozen metadata with an explicit analysis request.

    The default is part of the harness contract, not an analysis-time default:
    it is copied into and covered by the campaign manifest hash at freeze time.
    Callers may replace it only before the first freeze.
    """

    value = dict(metadata or {})
    supplied = value.get("analysis_spec", {})
    if not isinstance(supplied, Mapping):
        raise FreezeError("metadata.analysis_spec must be an object")
    unknown = set(supplied) - set(DEFAULT_ANALYSIS_SPEC)
    if unknown:
        raise FreezeError(
            f"unsupported analysis_spec keys: {sorted(unknown)}"
        )
    analysis_spec = {**DEFAULT_ANALYSIS_SPEC, **dict(supplied)}
    outcomes = analysis_spec.get("confirmatory_outcomes")
    if (
        not isinstance(outcomes, (list, tuple))
        or not outcomes
        or any(
            not isinstance(outcome, str) or _SAFE_ID.fullmatch(outcome) is None
            for outcome in outcomes
        )
        or len(set(outcomes)) != len(outcomes)
    ):
        raise FreezeError(
            "analysis_spec.confirmatory_outcomes must be a non-empty unique array "
            "of safe identifiers"
        )
    reference = analysis_spec.get("reference_profile")
    if not isinstance(reference, str) or _SAFE_ID.fullmatch(reference) is None:
        raise FreezeError("analysis_spec.reference_profile must be a safe identifier")
    for key in (
        "schema_version",
        "experimental_unit",
        "pairing_key",
        "multiplicity_method",
    ):
        if analysis_spec.get(key) != DEFAULT_ANALYSIS_SPEC[key]:
            raise FreezeError(
                f"unsupported analysis_spec.{key}: {analysis_spec.get(key)!r}"
            )
    families = analysis_spec.get("families")
    if not isinstance(families, list):
        raise FreezeError("analysis_spec.families must be an array")
    normalized_families: list[dict[str, Any]] = []
    for index, family in enumerate(families):
        if not isinstance(family, dict) or set(family) != {
            "matrix_id",
            "reference_profile",
            "comparison_profiles",
        }:
            raise FreezeError(f"malformed analysis_spec.families[{index}]")
        comparisons = family.get("comparison_profiles")
        if (
            not isinstance(family.get("matrix_id"), str)
            or _SAFE_ID.fullmatch(family["matrix_id"]) is None
            or not isinstance(family.get("reference_profile"), str)
            or _SAFE_ID.fullmatch(family["reference_profile"]) is None
            or not isinstance(comparisons, list)
            or any(
                not isinstance(profile, str) or _SAFE_ID.fullmatch(profile) is None
                for profile in comparisons
            )
            or comparisons != sorted(set(comparisons))
        ):
            raise FreezeError(f"unsafe analysis_spec.families[{index}]")
        normalized_families.append(dict(family))
    if normalized_families != sorted(
        normalized_families, key=lambda family: family["matrix_id"]
    ) or len({family["matrix_id"] for family in normalized_families}) != len(
        normalized_families
    ):
        raise FreezeError("analysis_spec families must be unique and matrix-sorted")
    analysis_spec["confirmatory_outcomes"] = list(outcomes)
    analysis_spec["families"] = normalized_families
    value["analysis_spec"] = analysis_spec
    # Validate JSON serializability and detach nested caller-owned containers.
    return json.loads(canonical_json_bytes(value))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _require_id(value: str, label: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise ScheduleError(f"{label} must match {_SAFE_ID.pattern!r}")
    return value


def _derived_seed(master_seed: int, *parts: object) -> int:
    if isinstance(master_seed, bool) or not isinstance(master_seed, int):
        raise ScheduleError("master seed must be an integer")
    payload = canonical_json_bytes([master_seed, *map(str, parts)])
    # Stay in the portable positive 63-bit range accepted by common runtimes.
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def _relative_roots(repo_root: Path, roots: Sequence[Path]) -> list[str]:
    repository = repo_root.resolve()
    labels: list[str] = []
    for requested in roots:
        path = requested if requested.is_absolute() else repository / requested
        try:
            resolved = path.resolve(strict=True)
            relative = resolved.relative_to(repository).as_posix()
        except (OSError, ValueError) as error:
            raise FreezeError(
                f"provenance root is unavailable or outside repository: {path}"
            ) from error
        labels.append(relative)
    return sorted(set(labels))


def _safe_frozen_file(root: Path, relative: Any, label: str) -> Path:
    """Resolve one manifest-controlled path without allowing an external read."""

    if not isinstance(relative, str) or not relative:
        raise FreezeError(f"{label} must be a non-empty relative path")
    requested = Path(relative)
    if requested.is_absolute() or ".." in requested.parts:
        raise FreezeError(f"unsafe {label}: {relative!r}")
    candidate = root / requested
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root.resolve())
    except (OSError, ValueError) as error:
        raise FreezeError(f"missing or external {label}: {relative!r}") from error
    if candidate.is_symlink() or not resolved.is_file():
        raise FreezeError(f"{label} is not a regular non-symlink file: {relative!r}")
    return resolved


def _validate_locked_execution_engine(
    metadata: Mapping[str, Any], repository: Path
) -> dict[str, str]:
    value = metadata.get("execution_engine")
    required = {
        "engine_id",
        "module",
        "class_name",
        "source_path",
        "source_sha256",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise FreezeError(
            "locked metadata.execution_engine must contain exactly "
            "engine_id, module, class_name, source_path, and source_sha256"
        )
    engine_id = value.get("engine_id")
    module = value.get("module")
    class_name = value.get("class_name")
    source_path = value.get("source_path")
    source_digest = value.get("source_sha256")
    if not isinstance(engine_id, str) or _SAFE_ID.fullmatch(engine_id) is None:
        raise FreezeError("execution_engine.engine_id must be a safe identifier")
    if not isinstance(module, str) or re.fullmatch(
        r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", module
    ) is None:
        raise FreezeError("execution_engine.module must be a dotted Python module")
    if not isinstance(class_name, str) or not class_name.isidentifier():
        raise FreezeError("execution_engine.class_name must be a Python identifier")
    source = _safe_frozen_file(
        repository, source_path, "execution_engine.source_path"
    )
    if _SHA256.fullmatch(str(source_digest)) is None:
        raise FreezeError("execution_engine.source_sha256 must be a SHA-256 digest")
    if sha256_file(source) != source_digest:
        raise FreezeError("execution engine source hash does not match live source")
    return {key: str(value[key]) for key in sorted(required)}


def _validate_locked_runtime_metadata(
    metadata: dict[str, Any], *, populate_hashes: bool
) -> None:
    """Bind the concrete engine/video/service configuration into the freeze.

    The manifest hash already covers metadata, but explicit component digests
    make setup/audit comparisons unambiguous and keep a class-level engine hash
    from being mistaken for an instantiated-system freeze.
    """

    mapping_keys = (
        "production_engine_config",
        "video_config",
        "model_service_mapping",
        "reported_model_ids",
        "service_endpoints",
        "controls",
        "hashes",
        "health_checks",
        "tunnel_epochs",
        "environment",
    )
    for key in mapping_keys:
        value = metadata.get(key)
        if not isinstance(value, dict) or not value:
            raise FreezeError(f"locked metadata.{key} must be a non-empty object")
        # Reject NaN/non-JSON client objects and detach caller-owned mappings.
        metadata[key] = json.loads(canonical_json_bytes(value))
    deviations = metadata.get("known_deviations")
    if (
        not isinstance(deviations, list)
        or any(not isinstance(item, str) or not item.strip() for item in deviations)
    ):
        raise FreezeError(
            "locked metadata.known_deviations must be an array of non-empty strings"
        )

    for key in ("production_engine_config", "video_config"):
        digest_key = f"{key}_sha256"
        expected = sha256_bytes(canonical_json_bytes(metadata[key]))
        if populate_hashes:
            supplied = metadata.get(digest_key)
            if supplied is not None and supplied != expected:
                raise FreezeError(f"locked metadata.{digest_key} is incorrect")
            metadata[digest_key] = expected
        elif metadata.get(digest_key) != expected:
            raise FreezeError(f"locked metadata.{digest_key} does not verify")


def _validate_provenance_files(value: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise FreezeError(f"provenance {label} must be an array")
    validated: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict) or set(item) != {"path", "size_bytes", "sha256"}:
            raise FreezeError(f"malformed provenance {label}[{index}]")
        relative = item.get("path")
        size = item.get("size_bytes")
        digest = item.get("sha256")
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or _SHA256.fullmatch(str(digest)) is None
        ):
            raise FreezeError(f"unsafe provenance {label}[{index}]")
        validated.append({"path": relative, "size_bytes": size, "sha256": digest})
    if [item["path"] for item in validated] != sorted(
        {item["path"] for item in validated}
    ):
        raise FreezeError(f"provenance {label} paths must be unique and sorted")
    return validated


@dataclass(frozen=True, slots=True)
class ScheduleCell:
    """One matched block definition before repetitions are expanded."""

    matrix_id: str
    scenario_id: str
    scenario_variant_id: str
    profiles: tuple[str, ...]
    repetitions: int
    model_backbones: tuple[str, ...] = ("primary",)
    executor: str = "mujoco"
    reuse_group: str | None = None

    def __post_init__(self) -> None:
        _require_id(self.matrix_id, "matrix_id")
        _require_id(self.scenario_id, "scenario_id")
        _require_id(self.scenario_variant_id, "scenario_variant_id")
        _require_id(self.executor, "executor")
        if not self.profiles or len(self.profiles) != len(set(self.profiles)):
            raise ScheduleError("profiles must be non-empty and unique")
        if not self.model_backbones or len(self.model_backbones) != len(
            set(self.model_backbones)
        ):
            raise ScheduleError("model_backbones must be non-empty and unique")
        for profile in self.profiles:
            _require_id(profile, "profile_id")
        for model in self.model_backbones:
            _require_id(model, "model_backbone")
        if (
            isinstance(self.repetitions, bool)
            or not isinstance(self.repetitions, int)
            or self.repetitions < 1
        ):
            raise ScheduleError("repetitions must be a positive integer")
        if self.reuse_group is not None:
            _require_id(self.reuse_group, "reuse_group")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ScheduleCell":
        expected = {
            "matrix_id",
            "scenario_id",
            "scenario_variant_id",
            "profiles",
            "repetitions",
            "model_backbones",
            "executor",
            "reuse_group",
        }
        unknown = set(value) - expected
        required = {"matrix_id", "scenario_id", "profiles", "repetitions"}
        missing = required - set(value)
        if unknown:
            raise ScheduleError(f"unknown schedule-cell keys: {sorted(unknown)}")
        if missing:
            raise ScheduleError(f"missing schedule-cell keys: {sorted(missing)}")
        if not isinstance(value["profiles"], (list, tuple)):
            raise ScheduleError("schedule-cell profiles must be an array")
        models = value.get("model_backbones", ("primary",))
        if not isinstance(models, (list, tuple)):
            raise ScheduleError("schedule-cell model_backbones must be an array")
        repetitions = value["repetitions"]
        if isinstance(repetitions, bool) or not isinstance(repetitions, int):
            raise ScheduleError("schedule-cell repetitions must be an integer")
        for key in ("matrix_id", "scenario_id"):
            if not isinstance(value[key], str):
                raise ScheduleError(f"schedule-cell {key} must be a string")
        if "scenario_variant_id" in value and not isinstance(
            value["scenario_variant_id"], str
        ):
            raise ScheduleError("schedule-cell scenario_variant_id must be a string")
        return cls(
            matrix_id=value["matrix_id"],
            scenario_id=value["scenario_id"],
            scenario_variant_id=(
                value.get("scenario_variant_id", value["scenario_id"])
            ),
            profiles=tuple(value["profiles"]),
            repetitions=repetitions,
            model_backbones=tuple(models),
            executor=str(value.get("executor", "mujoco")),
            reuse_group=(
                None if value.get("reuse_group") is None else str(value["reuse_group"])
            ),
        )


SCHEDULE_FIELDS = (
    "schedule_id",
    "schedule_position",
    "block_id",
    "block_position",
    "matrix_id",
    "scenario_id",
    "scenario_variant_id",
    "profile_id",
    "model_backbone",
    "repetition",
    "seed",
    "split",
    "executor",
    "crn_key",
    "reuse_group",
)

RUN_INDEX_FIELDS = SCHEDULE_FIELDS + (
    "attempt_number",
    "attempt_id",
    "run_status",
    "oracle_verdict",
    "started_utc",
    "ended_utc",
    "reason",
    "artifact_path",
    "artifact_audit_passed",
    "disposition",
)


@dataclass(frozen=True, slots=True)
class ScheduleEntry:
    schedule_id: str
    schedule_position: int
    block_id: str
    block_position: int
    matrix_id: str
    scenario_id: str
    scenario_variant_id: str
    profile_id: str
    model_backbone: str
    repetition: int
    seed: int
    split: str
    executor: str
    crn_key: str
    reuse_group: str | None = None

    def __post_init__(self) -> None:
        for label in (
            "schedule_id",
            "block_id",
            "matrix_id",
            "scenario_id",
            "scenario_variant_id",
            "profile_id",
            "model_backbone",
            "split",
            "executor",
        ):
            _require_id(str(getattr(self, label)), label)
        if self.schedule_position < 1 or self.block_position < 1 or self.repetition < 1:
            raise ScheduleError("positions and repetition must be positive")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ScheduleError("seed must be a non-negative integer")
        if not self.crn_key:
            raise ScheduleError("crn_key must be non-empty")
        if self.reuse_group is not None:
            _require_id(self.reuse_group, "reuse_group")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ScheduleEntry":
        missing = set(SCHEDULE_FIELDS) - set(value)
        unknown = set(value) - set(SCHEDULE_FIELDS)
        if missing or unknown:
            raise ScheduleError(
                f"schedule entry missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        return cls(
            schedule_id=str(value["schedule_id"]),
            schedule_position=int(value["schedule_position"]),
            block_id=str(value["block_id"]),
            block_position=int(value["block_position"]),
            matrix_id=str(value["matrix_id"]),
            scenario_id=str(value["scenario_id"]),
            scenario_variant_id=str(value["scenario_variant_id"]),
            profile_id=str(value["profile_id"]),
            model_backbone=str(value["model_backbone"]),
            repetition=int(value["repetition"]),
            seed=int(value["seed"]),
            split=str(value["split"]),
            executor=str(value["executor"]),
            crn_key=str(value["crn_key"]),
            reuse_group=(
                None
                if value.get("reuse_group") in (None, "")
                else str(value["reuse_group"])
            ),
        )


def build_blocked_schedule(
    cells: Iterable[ScheduleCell | Mapping[str, Any]],
    *,
    master_seed: int,
    split: str,
    reuse_policy: ReusePolicy | str = ReusePolicy.FRESH_ATTEMPTS_ONLY,
) -> tuple[ScheduleEntry, ...]:
    """Expand cells into deterministic, matched, sequential schedule entries.

    Blocks are shuffled deterministically and profiles are shuffled within each
    block.  Every profile in a block receives the same generated-instance seed
    and CRN key.  Entries are returned in their sole permitted execution order;
    this API never introduces parallel lanes.
    """

    _require_id(split, "split")
    policy = ReusePolicy(reuse_policy)
    if policy is ReusePolicy.PREREGISTERED_SHARED_REFERENCE:
        raise ScheduleError(
            "shared-reference outcome reuse is not implemented; freeze fresh attempts only"
        )
    normalized = [
        cell if isinstance(cell, ScheduleCell) else ScheduleCell.from_mapping(cell)
        for cell in cells
    ]
    if not normalized:
        raise ScheduleError("at least one schedule cell is required")

    blocks: list[tuple[str, ScheduleCell, int, str, int, str, list[str]]] = []
    seen_block_assignments: set[tuple[str, str, str, int, str]] = set()
    for cell in sorted(
        normalized,
        key=lambda item: (
            item.matrix_id,
            item.scenario_id,
            item.scenario_variant_id,
            item.executor,
        ),
    ):
        for model in sorted(cell.model_backbones):
            for repetition in range(1, cell.repetitions + 1):
                assignment = (
                    cell.matrix_id,
                    cell.scenario_variant_id,
                    model,
                    repetition,
                    cell.executor,
                )
                if assignment in seen_block_assignments:
                    raise ScheduleError(f"duplicate schedule block: {assignment}")
                seen_block_assignments.add(assignment)
                crn_key = ":".join(map(str, assignment))
                seed = _derived_seed(master_seed, "episode", crn_key)
                block_hash = sha256_bytes(canonical_json_bytes(assignment))[:12]
                block_id = f"B-{block_hash}"
                profiles = sorted(cell.profiles)
                random.Random(_derived_seed(master_seed, "profiles", crn_key)).shuffle(
                    profiles
                )
                blocks.append(
                    (block_id, cell, repetition, model, seed, crn_key, profiles)
                )

    # Block ordering is independent of caller ordering and profile enumeration.
    random.Random(_derived_seed(master_seed, "block-order", split)).shuffle(blocks)
    entries: list[ScheduleEntry] = []
    for block_id, cell, repetition, model, seed, crn_key, profiles in blocks:
        for block_position, profile in enumerate(profiles, start=1):
            entries.append(
                ScheduleEntry(
                    schedule_id="PENDING",
                    schedule_position=len(entries) + 1,
                    block_id=block_id,
                    block_position=block_position,
                    matrix_id=cell.matrix_id,
                    scenario_id=cell.scenario_id,
                    scenario_variant_id=cell.scenario_variant_id,
                    profile_id=profile,
                    model_backbone=model,
                    repetition=repetition,
                    seed=seed,
                    split=split,
                    executor=cell.executor,
                    crn_key=crn_key,
                    reuse_group=cell.reuse_group,
                )
            )
    width = max(6, len(str(len(entries))))
    return tuple(
        replace(entry, schedule_id=f"S{position:0{width}d}")
        for position, entry in enumerate(entries, start=1)
    )


def generate_schedule(
    scenario_ids: Sequence[str],
    profile_ids: Sequence[str],
    *,
    repetitions: int,
    master_seed: int,
    split: str,
    matrix_id: str = "custom",
    executor: str = "mujoco",
) -> tuple[ScheduleEntry, ...]:
    """Convenience schedule builder for a rectangular matrix."""

    cells = [
        ScheduleCell(
            matrix_id=matrix_id,
            scenario_id=scenario,
            scenario_variant_id=scenario,
            profiles=tuple(profile_ids),
            repetitions=repetitions,
            executor=executor,
        )
        for scenario in scenario_ids
    ]
    return build_blocked_schedule(cells, master_seed=master_seed, split=split)


def default_protocol_schedule(
    *,
    split: str,
    matched_repetitions: int,
    master_seed: int,
    sensitivity_repetitions: int = 10,
    sensitivity_backbones: Sequence[str] = ("primary", "alternative", "small"),
) -> tuple[ScheduleEntry, ...]:
    """Construct the design-document matrices from the validated catalogue.

    Reference episodes are intentionally fresh in each matrix.  This costs more
    runs, but avoids outcome-dependent reuse across confirmatory comparisons.
    """

    from .scenarios import ABLATION_MARKERS, load_scenarios

    catalogue = load_scenarios()
    cells: list[ScheduleCell] = []

    def add(
        matrix_id: str,
        scenarios: Iterable[Any],
        profiles: Sequence[str],
        repetitions: int,
        models: Sequence[str] = ("primary",),
    ) -> None:
        for scenario in scenarios:
            selected = tuple(
                profile for profile in profiles if profile in scenario.applicable_profiles
            )
            if len(selected) != len(profiles):
                missing = sorted(set(profiles) - set(selected))
                raise ScheduleError(
                    f"{scenario.scenario_id} is not applicable to {missing} in {matrix_id}"
                )
            cells.append(
                ScheduleCell(
                    matrix_id=matrix_id,
                    scenario_id=scenario.scenario_id,
                    scenario_variant_id=scenario.split_variants[split].variant_id,
                    profiles=selected,
                    repetitions=repetitions,
                    model_backbones=tuple(models),
                    executor=scenario.setup.executor,
                )
            )

    add(
        "main_system",
        catalogue.scenarios,
        ("T5", "B1", "B2", "B4"),
        matched_repetitions,
    )
    add(
        "topology24",
        catalogue.subset("topology24"),
        ("T5", "T4", "T3", "T2", "T1"),
        matched_repetitions,
    )
    for marker in sorted(ABLATION_MARKERS):
        profile = marker.removeprefix("ablation:")
        add(
            "ablation-" + profile,
            catalogue.subset(marker),
            ("T5", profile),
            matched_repetitions,
        )
    add(
        "model_sensitivity",
        catalogue.subset("sensitivity20"),
        ("T5", "B2"),
        sensitivity_repetitions,
        sensitivity_backbones,
    )
    return build_blocked_schedule(cells, master_seed=master_seed, split=split)


def schedule_csv_bytes(schedule: Sequence[ScheduleEntry]) -> bytes:
    validate_schedule(schedule)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=SCHEDULE_FIELDS, lineterminator="\n")
    writer.writeheader()
    for entry in schedule:
        row = entry.to_dict()
        row["reuse_group"] = row["reuse_group"] or ""
        writer.writerow(row)
    return buffer.getvalue().encode("utf-8")


def run_index_csv_bytes(state: Mapping[str, Any]) -> bytes:
    """Render the exact denormalized index without mutating campaign state."""

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=RUN_INDEX_FIELDS, lineterminator="\n")
    writer.writeheader()
    ordered = sorted(
        state["entries"].values(),
        key=lambda item: item["assignment"]["schedule_position"],
    )
    for entry in ordered:
        assignment = dict(entry["assignment"])
        assignment["reuse_group"] = assignment.get("reuse_group") or ""
        attempts = entry["attempts"] or [
            {
                "attempt_number": 0,
                "attempt_id": "",
                "status": RunStatus.NOT_RUN.value,
                "oracle_verdict": None,
                "started_utc": None,
                "ended_utc": None,
                "reason": None,
                "artifact_path": "",
                "artifact_audit_passed": None,
            }
        ]
        for attempt in attempts:
            writer.writerow(
                {
                    **assignment,
                    "attempt_number": attempt["attempt_number"],
                    "attempt_id": attempt["attempt_id"],
                    "run_status": attempt["status"],
                    "oracle_verdict": attempt.get("oracle_verdict") or "",
                    "started_utc": attempt.get("started_utc") or "",
                    "ended_utc": attempt.get("ended_utc") or "",
                    "reason": attempt.get("reason") or "",
                    "artifact_path": attempt.get("artifact_path") or "",
                    "artifact_audit_passed": (
                        ""
                        if attempt.get("artifact_audit_passed") is None
                        else str(bool(attempt["artifact_audit_passed"])).lower()
                    ),
                    "disposition": entry.get("disposition") or "",
                }
            )
    return buffer.getvalue().encode("utf-8")


def schedule_sha256(schedule: Sequence[ScheduleEntry]) -> str:
    return sha256_bytes(schedule_csv_bytes(schedule))


def validate_schedule(schedule: Sequence[ScheduleEntry]) -> None:
    if not schedule:
        raise ScheduleError("schedule must not be empty")
    ids = [entry.schedule_id for entry in schedule]
    positions = [entry.schedule_position for entry in schedule]
    if len(ids) != len(set(ids)):
        raise ScheduleError("schedule IDs are not unique")
    if positions != list(range(1, len(schedule) + 1)):
        raise ScheduleError("schedule positions are not contiguous and ordered")
    by_block: dict[str, list[ScheduleEntry]] = {}
    crn_to_block: dict[str, str] = {}
    for entry in schedule:
        by_block.setdefault(entry.block_id, []).append(entry)
        previous_block = crn_to_block.setdefault(entry.crn_key, entry.block_id)
        if previous_block != entry.block_id:
            raise ScheduleError(
                f"CRN key {entry.crn_key} is shared by distinct blocks"
            )
    for block_id, entries in by_block.items():
        if [entry.block_position for entry in entries] != list(
            range(1, len(entries) + 1)
        ):
            raise ScheduleError(f"block {block_id} positions are not contiguous")
        schedule_positions = [entry.schedule_position for entry in entries]
        if schedule_positions != list(
            range(schedule_positions[0], schedule_positions[0] + len(entries))
        ):
            raise ScheduleError(f"block {block_id} is interleaved with another block")
        crn_values = {(entry.seed, entry.crn_key) for entry in entries}
        if len(crn_values) != 1:
            raise ScheduleError(f"block {block_id} violates common random numbers")
        if len({entry.profile_id for entry in entries}) != len(entries):
            raise ScheduleError(f"block {block_id} repeats a profile")
        assignment_dimensions = {
            (
                entry.matrix_id,
                entry.scenario_id,
                entry.scenario_variant_id,
                entry.model_backbone,
                entry.repetition,
                entry.seed,
                entry.crn_key,
                entry.executor,
                entry.split,
                entry.reuse_group,
            )
            for entry in entries
        }
        if len(assignment_dimensions) != 1:
            raise ScheduleError(
                f"block {block_id} mixes non-profile assignment dimensions"
            )


def write_schedule(path: str | Path, schedule: Sequence[ScheduleEntry]) -> Path:
    output = Path(path)
    _exclusive_write(output, schedule_csv_bytes(schedule))
    return output


def read_schedule(path: str | Path) -> tuple[ScheduleEntry, ...]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != SCHEDULE_FIELDS:
            raise ScheduleError("schedule CSV header differs from the frozen schema")
        schedule = tuple(ScheduleEntry.from_dict(row) for row in reader)
    validate_schedule(schedule)
    return schedule


@dataclass(frozen=True, slots=True)
class Approval:
    campaign_id: str
    protocol_sha256: str


def parse_locked_approval(text: str) -> Approval:
    """Parse the exact approval sentence; trimming or extra prose is rejected."""

    if not isinstance(text, str):
        raise ApprovalError("approval must be text")
    match = _APPROVAL.fullmatch(text)
    if match is None:
        raise ApprovalError(
            "expected exactly: APPROVE LOCKED CAMPAIGN <campaign_id> <protocol_sha256>"
        )
    return Approval(**match.groupdict())


def require_locked_approval(
    text: str, *, campaign_id: str, protocol_sha256: str
) -> Approval:
    approval = parse_locked_approval(text)
    if approval.campaign_id != campaign_id:
        raise ApprovalError("approval campaign ID does not match the frozen campaign")
    if approval.protocol_sha256 != protocol_sha256:
        raise ApprovalError("approval protocol hash does not match PROTOCOL.sha256")
    return approval


@dataclass(frozen=True, slots=True)
class AttemptLease:
    campaign_id: str
    schedule_id: str
    attempt_number: int
    attempt_id: str
    attempt_path: Path
    assignment: ScheduleEntry


def _atomic_json(path: Path, value: Any, *, exclusive: bool = False) -> None:
    data = canonical_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        _exclusive_write(path, data)
        return
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _exclusive_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class CampaignRegistry:
    """Atomic, process-safe snapshot of immutable scheduled attempts."""

    def __init__(self, campaign_root: str | Path) -> None:
        self.root = Path(campaign_root).resolve()
        self.path = self.root / "RUN_STATE.json"
        self.lock_path = self.root / ".run_state.lock"
        if not self.path.is_file():
            raise RegistryError(f"missing run registry: {self.path}")
        self._validate(self._read())

    @classmethod
    def create(
        cls,
        campaign_root: str | Path,
        *,
        campaign_id: str,
        mode: CampaignMode | str,
        protocol_sha256: str,
        schedule: Sequence[ScheduleEntry],
        reuse_policy: ReusePolicy | str = ReusePolicy.FRESH_ATTEMPTS_ONLY,
        max_infrastructure_attempts: int = MAX_INFRASTRUCTURE_ATTEMPTS,
    ) -> "CampaignRegistry":
        root = Path(campaign_root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        _require_id(campaign_id, "campaign_id")
        actual_mode = CampaignMode(mode)
        actual_policy = ReusePolicy(reuse_policy)
        if actual_policy is ReusePolicy.PREREGISTERED_SHARED_REFERENCE:
            raise RegistryError(
                "shared-reference outcome reuse is unsupported; use fresh_attempts_only"
            )
        if _SHA256.fullmatch(protocol_sha256) is None:
            raise RegistryError("protocol_sha256 must be a lowercase SHA-256 digest")
        if max_infrastructure_attempts != MAX_INFRASTRUCTURE_ATTEMPTS:
            raise RegistryError("the protocol permits exactly three infrastructure attempts")
        validate_schedule(schedule)
        if actual_mode is CampaignMode.PILOT and any(
            entry.split != "pilot" for entry in schedule
        ):
            raise RegistryError("pilot registries may contain only pilot schedule entries")
        if actual_mode is CampaignMode.LOCKED and any(
            entry.split not in {"locked", "locked_test"} for entry in schedule
        ):
            raise RegistryError("locked registries may contain only locked schedule entries")
        now = utc_now()
        state = {
            "schema_version": SCHEMA_VERSION,
            "campaign_id": campaign_id,
            "mode": actual_mode.value,
            "protocol_sha256": protocol_sha256,
            "schedule_sha256": schedule_sha256(schedule),
            "schedule_size": len(schedule),
            "reuse_policy": actual_policy.value,
            "max_infrastructure_attempts": max_infrastructure_attempts,
            "execution_authorized": actual_mode is CampaignMode.PILOT,
            "approval": None,
            "created_utc": now,
            "updated_utc": now,
            "registry_revision": 0,
            "entries": {
                entry.schedule_id: {
                    "assignment": entry.to_dict(),
                    "status": RunStatus.NOT_RUN.value,
                    "attempts": [],
                    "disposition": None,
                    "retry_authorization": None,
                }
                for entry in schedule
            },
            "events": [
                {
                    "event": "REGISTRY_CREATED",
                    "utc": now,
                    "schedule_sha256": schedule_sha256(schedule),
                }
            ],
        }
        path = root / "RUN_STATE.json"
        if path.exists():
            existing = cls(root)
            snapshot = existing.snapshot()
            immutable = (
                snapshot["campaign_id"],
                snapshot["mode"],
                snapshot["protocol_sha256"],
                snapshot["schedule_sha256"],
                snapshot["reuse_policy"],
            )
            requested = (
                campaign_id,
                actual_mode.value,
                protocol_sha256,
                schedule_sha256(schedule),
                actual_policy.value,
            )
            if immutable != requested:
                raise RegistryError("refusing to reuse registry with different frozen inputs")
            return existing
        _atomic_json(path, state, exclusive=True)
        registry = cls(root)
        registry.export_run_index()
        return registry

    def _read(self) -> dict[str, Any]:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise RegistryError(f"cannot read registry: {error}") from error
        if not isinstance(value, dict):
            raise RegistryError("RUN_STATE.json must contain an object")
        return value

    @contextmanager
    def _edit(self) -> Iterator[dict[str, Any]]:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            state = self._read()
            self._validate(state)
            original_revision = state["registry_revision"]
            try:
                yield state
            except Exception:
                raise
            else:
                state["registry_revision"] = original_revision + 1
                state["updated_utc"] = utc_now()
                self._validate(state)
                _atomic_json(self.path, state)
                self.export_run_index()
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _validate(state: Mapping[str, Any]) -> None:
        required = {
            "schema_version",
            "campaign_id",
            "mode",
            "protocol_sha256",
            "schedule_sha256",
            "schedule_size",
            "reuse_policy",
            "max_infrastructure_attempts",
            "execution_authorized",
            "approval",
            "created_utc",
            "updated_utc",
            "registry_revision",
            "entries",
            "events",
        }
        if set(state) != required:
            raise RegistryError(
                f"registry keys missing={sorted(required - set(state))}, "
                f"unknown={sorted(set(state) - required)}"
            )
        if state["schema_version"] != SCHEMA_VERSION:
            raise RegistryError("unsupported registry schema")
        mode = CampaignMode(state["mode"])
        ReusePolicy(state["reuse_policy"])
        if state["max_infrastructure_attempts"] != MAX_INFRASTRUCTURE_ATTEMPTS:
            raise RegistryError("infrastructure-attempt limit changed")
        if mode is CampaignMode.PILOT:
            if not state["execution_authorized"] or state["approval"] is not None:
                raise RegistryError("pilot authorization state is malformed")
        else:
            if bool(state["execution_authorized"]) != (state["approval"] is not None):
                raise RegistryError("locked approval/authorization state is inconsistent")
            if state["approval"] is not None:
                approval = state["approval"]
                expected_approval_keys = {
                    "campaign_id",
                    "protocol_sha256",
                    "approved_utc",
                    "text_sha256",
                }
                if not isinstance(approval, dict) or set(approval) != expected_approval_keys:
                    raise RegistryError("locked approval record is malformed")
                if (
                    approval["campaign_id"] != state["campaign_id"]
                    or approval["protocol_sha256"] != state["protocol_sha256"]
                    or _SHA256.fullmatch(str(approval["text_sha256"])) is None
                ):
                    raise RegistryError("locked approval is not bound to this freeze")
        entries = state["entries"]
        if not isinstance(entries, dict) or len(entries) != state["schedule_size"]:
            raise RegistryError("registry entry count differs from frozen schedule")
        schedule: list[ScheduleEntry] = []
        for schedule_id, item in entries.items():
            if not isinstance(item, dict) or set(item) != {
                "assignment",
                "status",
                "attempts",
                "disposition",
                "retry_authorization",
            }:
                raise RegistryError(f"malformed entry {schedule_id}")
            assignment = ScheduleEntry.from_dict(item["assignment"])
            if assignment.schedule_id != schedule_id:
                raise RegistryError(f"entry key does not match assignment: {schedule_id}")
            schedule.append(assignment)
            RunStatus(item["status"])
            attempts = item["attempts"]
            if not isinstance(attempts, list) or len(attempts) > MAX_INFRASTRUCTURE_ATTEMPTS:
                raise RegistryError(f"invalid attempt history for {schedule_id}")
            for index, attempt in enumerate(attempts, start=1):
                expected_attempt_keys = {
                    "attempt_number",
                    "attempt_id",
                    "status",
                    "started_utc",
                    "ended_utc",
                    "oracle_verdict",
                    "reason",
                    "artifact_path",
                    "artifact_audit_passed",
                    "infrastructure_evidence",
                    "recovery_authorization",
                }
                if not isinstance(attempt, dict) or set(attempt) != expected_attempt_keys:
                    raise RegistryError(f"malformed attempt {schedule_id}/{index}")
                if attempt.get("attempt_number") != index:
                    raise RegistryError(f"non-contiguous attempt numbers for {schedule_id}")
                attempt_status = RunStatus(attempt.get("status"))
                if attempt["attempt_id"] != f"{schedule_id}-A{index}":
                    raise RegistryError(f"attempt ID mismatch for {schedule_id}/{index}")
                expected_path = f"episodes/{schedule_id}/attempt_{index}"
                if attempt["artifact_path"] != expected_path:
                    raise RegistryError(f"attempt path mismatch for {schedule_id}/{index}")
                if attempt_status is RunStatus.NOT_RUN:
                    if attempt["ended_utc"] is not None:
                        raise RegistryError(f"open attempt has an end time: {schedule_id}/{index}")
                elif not attempt["ended_utc"]:
                    raise RegistryError(f"sealed attempt lacks an end time: {schedule_id}/{index}")
                audit_flag = attempt.get("artifact_audit_passed")
                if audit_flag is not None and not isinstance(audit_flag, bool):
                    raise RegistryError(f"invalid artifact-audit flag: {schedule_id}/{index}")
                if attempt_status in {
                    RunStatus.VALID_PASS,
                    RunStatus.VALID_SYSTEM_FAILURE,
                }:
                    expected_verdict = (
                        "PASS"
                        if attempt_status is RunStatus.VALID_PASS
                        else "FAIL"
                    )
                    if attempt.get("oracle_verdict") != expected_verdict:
                        raise RegistryError(
                            f"valid status/oracle contradiction: {schedule_id}/{index}"
                        )
                    if attempt.get("artifact_audit_passed") is not True:
                        raise RegistryError(f"valid outcome lacks passing audit: {schedule_id}/{index}")
                if attempt_status in {
                    RunStatus.INFRA_INTERRUPTED,
                    RunStatus.INVALID_HARNESS,
                    RunStatus.ABORTED_SAFETY,
                } and not attempt.get("reason"):
                    raise RegistryError(f"non-valid outcome lacks reason: {schedule_id}/{index}")
                infrastructure_evidence = attempt.get("infrastructure_evidence")
                if attempt_status is RunStatus.INFRA_INTERRUPTED:
                    evidence_path = (
                        Path(infrastructure_evidence.get("path", ""))
                        if isinstance(infrastructure_evidence, dict)
                        else Path()
                    )
                    if (
                        not isinstance(infrastructure_evidence, dict)
                        or set(infrastructure_evidence) != {"path", "sha256"}
                        or not isinstance(infrastructure_evidence.get("path"), str)
                        or not infrastructure_evidence.get("path")
                        or evidence_path.is_absolute()
                        or ".." in evidence_path.parts
                        or _SHA256.fullmatch(
                            str(infrastructure_evidence.get("sha256"))
                        )
                        is None
                    ):
                        raise RegistryError(
                            f"infrastructure interruption lacks bound evidence: "
                            f"{schedule_id}/{index}"
                        )
                elif infrastructure_evidence is not None:
                    raise RegistryError(
                        f"non-infrastructure attempt carries infrastructure evidence: "
                        f"{schedule_id}/{index}"
                    )
                recovery_authorization = attempt.get("recovery_authorization")
                if index == 1:
                    if recovery_authorization is not None:
                        raise RegistryError(
                            f"first attempt carries a retry authorization: {schedule_id}"
                        )
                else:
                    previous = attempts[index - 2]
                    if previous.get("status") != RunStatus.INFRA_INTERRUPTED.value:
                        raise RegistryError(
                            f"retry does not follow infrastructure interruption: "
                            f"{schedule_id}/{index}"
                        )
                    self_authorization = recovery_authorization
                    if (
                        not isinstance(self_authorization, dict)
                        or set(self_authorization)
                        != {
                            "path",
                            "sha256",
                            "authorized_utc",
                            "after_attempt_number",
                        }
                        or self_authorization.get("after_attempt_number") != index - 1
                        or not isinstance(self_authorization.get("path"), str)
                        or Path(self_authorization.get("path", "")).is_absolute()
                        or ".." in Path(self_authorization.get("path", "")).parts
                        or not isinstance(
                            self_authorization.get("authorized_utc"), str
                        )
                        or _SHA256.fullmatch(
                            str(self_authorization.get("sha256"))
                        )
                        is None
                    ):
                        raise RegistryError(
                            f"retry lacks a valid consumed recovery authorization: "
                            f"{schedule_id}/{index}"
                        )
                if index < len(attempts) and attempt.get("status") == RunStatus.NOT_RUN:
                    raise RegistryError(f"unsealed attempt precedes another for {schedule_id}")
                if index < len(attempts) and attempt_status is not RunStatus.INFRA_INTERRUPTED:
                    raise RegistryError(
                        f"only an infrastructure interruption may precede a retry: "
                        f"{schedule_id}/{index}"
                    )
            if attempts and item["status"] != attempts[-1]["status"]:
                raise RegistryError(f"entry status differs from latest attempt: {schedule_id}")
            if not attempts and item["status"] != RunStatus.NOT_RUN:
                raise RegistryError(f"status without attempt history: {schedule_id}")
            pending_authorization = item.get("retry_authorization")
            if pending_authorization is not None:
                if (
                    not attempts
                    or attempts[-1]["status"] != RunStatus.INFRA_INTERRUPTED.value
                    or len(attempts) >= MAX_INFRASTRUCTURE_ATTEMPTS
                    or not isinstance(pending_authorization, dict)
                    or set(pending_authorization)
                    != {
                        "path",
                        "sha256",
                        "authorized_utc",
                        "after_attempt_number",
                    }
                    or pending_authorization.get("after_attempt_number") != len(attempts)
                    or not isinstance(pending_authorization.get("path"), str)
                    or Path(pending_authorization.get("path", "")).is_absolute()
                    or ".." in Path(pending_authorization.get("path", "")).parts
                    or not isinstance(
                        pending_authorization.get("authorized_utc"), str
                    )
                    or _SHA256.fullmatch(
                        str(pending_authorization.get("sha256"))
                    )
                    is None
                ):
                    raise RegistryError(
                        f"invalid pending recovery authorization: {schedule_id}"
                    )
        schedule.sort(key=lambda entry: entry.schedule_position)
        validate_schedule(schedule)
        if schedule_sha256(schedule) != state["schedule_sha256"]:
            raise RegistryError("embedded schedule differs from its frozen hash")

    def snapshot(self) -> dict[str, Any]:
        state = self._read()
        self._validate(state)
        # JSON round-trip prevents callers from mutating the loaded snapshot.
        return json.loads(json.dumps(state))

    @property
    def campaign_id(self) -> str:
        return str(self.snapshot()["campaign_id"])

    @property
    def mode(self) -> CampaignMode:
        return CampaignMode(self.snapshot()["mode"])

    @property
    def protocol_sha256(self) -> str:
        return str(self.snapshot()["protocol_sha256"])

    def authorize_locked(self, approval_text: str) -> Approval | None:
        """Authorize this immutable locked version, or no-op for a pilot."""

        if self.mode is CampaignMode.LOCKED:
            verify_frozen_campaign(self.root, check_source=True)
        with self._edit() as state:
            if CampaignMode(state["mode"]) is CampaignMode.PILOT:
                return None
            approval = require_locked_approval(
                approval_text,
                campaign_id=state["campaign_id"],
                protocol_sha256=state["protocol_sha256"],
            )
            if state["approval"] is not None:
                if state["approval"]["text_sha256"] != sha256_bytes(
                    approval_text.encode("utf-8")
                ):
                    raise ApprovalError("locked campaign was authorized by a different token")
                return approval
            now = utc_now()
            state["execution_authorized"] = True
            state["approval"] = {
                "campaign_id": approval.campaign_id,
                "protocol_sha256": approval.protocol_sha256,
                "approved_utc": now,
                "text_sha256": sha256_bytes(approval_text.encode("utf-8")),
            }
            state["events"].append({"event": "LOCKED_CAMPAIGN_AUTHORIZED", "utc": now})
            return approval

    def _assert_authorized(self, state: Mapping[str, Any]) -> None:
        if CampaignMode(state["mode"]) is CampaignMode.LOCKED and not state[
            "execution_authorized"
        ]:
            raise ApprovalError(
                "locked execution refused; supply the exact approval sentence first"
            )

    @staticmethod
    def _open_attempts(state: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
        found: list[tuple[str, Mapping[str, Any]]] = []
        for schedule_id, entry in state["entries"].items():
            if entry["attempts"] and entry["attempts"][-1]["status"] == RunStatus.NOT_RUN:
                found.append((schedule_id, entry["attempts"][-1]))
        return found

    def next_entry(self) -> ScheduleEntry | None:
        state = self.snapshot()
        self._assert_authorized(state)
        if self._open_attempts(state):
            raise RegistryError(
                "an attempt remains open; seal or reconcile it before resuming"
            )
        ordered = sorted(
            state["entries"].values(),
            key=lambda item: item["assignment"]["schedule_position"],
        )
        for item in ordered:
            status = RunStatus(item["status"])
            if status is RunStatus.NOT_RUN:
                return ScheduleEntry.from_dict(item["assignment"])
            if status is RunStatus.INFRA_INTERRUPTED and len(item["attempts"]) < state[
                "max_infrastructure_attempts"
            ]:
                if item.get("retry_authorization") is None:
                    raise RegistryError(
                        "infrastructure retry is blocked until a recovery report with "
                        "two successful health checks and a smoke test is authorized"
                    )
                return ScheduleEntry.from_dict(item["assignment"])
        return None

    def _validate_recovery_report(
        self,
        relative_path: str,
        *,
        schedule_id: str,
        after_attempt_number: int,
        after_utc: str,
        interruption_sha256: str,
        interruption_service_id: str,
        expected_sha256: str | None = None,
    ) -> str:
        relative = Path(relative_path)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or len(relative.parts) < 3
            or relative.parts[:2] != ("environment", "recovery")
        ):
            raise RegistryError(
                "recovery evidence must be a safe path under environment/recovery/"
            )
        candidate = self.root / relative
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to((self.root / "environment" / "recovery").resolve())
        except (OSError, ValueError) as error:
            raise RegistryError("recovery evidence is missing or outside campaign") from error
        if candidate.is_symlink() or not resolved.is_file():
            raise RegistryError("recovery evidence must be a regular non-symlink file")
        try:
            payload = resolved.read_bytes()
            evidence = json.loads(payload.decode("utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise RegistryError(f"cannot parse recovery evidence: {error}") from error
        digest = sha256_bytes(payload)
        if expected_sha256 is not None and digest != expected_sha256:
            raise RegistryError("recovery evidence changed after authorization")
        if not isinstance(evidence, dict):
            raise RegistryError("recovery evidence must contain a JSON object")

        def observation(item: Any, *, smoke: bool = False) -> tuple[datetime, str, str, str]:
            required = {
                "sequence",
                "observed_utc",
                "passed",
                "evidence_path",
                "evidence_sha256",
                "service_id",
                "check_id",
            }
            if smoke:
                required.add("non_mutating")
            if not isinstance(item, dict) or not required <= set(item):
                raise RegistryError("health/smoke observations lack raw-evidence fields")
            if (
                isinstance(item.get("sequence"), bool)
                or not isinstance(item.get("sequence"), int)
                or item.get("passed") is not True
                or (smoke and item.get("non_mutating") is not True)
                or _SHA256.fullmatch(str(item.get("evidence_sha256"))) is None
                or not isinstance(item.get("service_id"), str)
                or _SAFE_ID.fullmatch(item["service_id"]) is None
                or not isinstance(item.get("check_id"), str)
                or _SAFE_ID.fullmatch(item["check_id"]) is None
            ):
                raise RegistryError("health/smoke observation is not a passing evidence record")
            raw_relative = Path(str(item.get("evidence_path", "")))
            if (
                raw_relative.is_absolute()
                or ".." in raw_relative.parts
                or len(raw_relative.parts) < 3
                or raw_relative.parts[:2] != ("environment", "recovery")
            ):
                raise RegistryError("raw recovery evidence has an unsafe path")
            raw_candidate = self.root / raw_relative
            try:
                raw_resolved = raw_candidate.resolve(strict=True)
                raw_resolved.relative_to(
                    (self.root / "environment" / "recovery").resolve()
                )
            except (OSError, ValueError) as error:
                raise RegistryError("raw recovery evidence is missing") from error
            if (
                raw_candidate.is_symlink()
                or not raw_resolved.is_file()
                or sha256_file(raw_resolved) != item["evidence_sha256"]
            ):
                raise RegistryError("raw recovery evidence hash mismatch")
            try:
                raw_value = json.loads(raw_resolved.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                raise RegistryError("raw recovery evidence must be JSON") from error
            if not isinstance(raw_value, dict) or any(
                raw_value.get(key) != item.get(key)
                for key in (
                    "sequence",
                    "observed_utc",
                    "passed",
                    "service_id",
                    "check_id",
                )
            ) or (smoke and raw_value.get("non_mutating") is not True):
                raise RegistryError(
                    "raw recovery evidence contradicts the summarized observation"
                )
            observed = item.get("observed_utc")
            if not isinstance(observed, str):
                raise RegistryError("recovery observation lacks a UTC timestamp")
            try:
                parsed = datetime.fromisoformat(observed.replace("Z", "+00:00"))
            except ValueError as error:
                raise RegistryError("recovery observation timestamp is invalid") from error
            if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
                raise RegistryError("recovery observation timestamp must be UTC")
            return (
                parsed,
                raw_relative.as_posix(),
                item["check_id"],
                item["service_id"],
            )

        health_checks = evidence.get("health_checks")
        smoke_test = evidence.get("smoke_test")
        if (
            evidence.get("schema_version") != 1
            or evidence.get("schedule_id") != schedule_id
            or evidence.get("after_attempt_number") != after_attempt_number
            or evidence.get("interruption_evidence_sha256")
            != interruption_sha256
            or evidence.get("service_restored") is not True
            or not isinstance(health_checks, list)
            or len(health_checks) < 2
        ):
            raise RegistryError(
                "recovery report does not bind the interrupted attempt and restoration"
            )
        try:
            interrupted_at = datetime.fromisoformat(after_utc.replace("Z", "+00:00"))
        except (AttributeError, ValueError) as error:
            raise RegistryError("interrupted attempt end time is invalid") from error
        penultimate, latest = health_checks[-2:]
        first_time, first_path, first_id, first_service = observation(penultimate)
        second_time, second_path, second_id, second_service = observation(latest)
        smoke_time, smoke_path, smoke_id, smoke_service = observation(
            smoke_test, smoke=True
        )
        current_time = datetime.now(timezone.utc)
        if (
            latest["sequence"] != penultimate["sequence"] + 1
            or len({first_path, second_path, smoke_path}) != 3
            or len({first_id, second_id, smoke_id}) != 3
            or len({first_service, second_service, smoke_service}) != 1
            or first_service != interruption_service_id
            or not interrupted_at < first_time < second_time <= smoke_time <= current_time
        ):
            raise RegistryError(
                "recovery report needs two distinct consecutive checks followed by smoke"
            )
        return digest

    def _validate_infrastructure_evidence(
        self,
        relative_path: str,
        *,
        schedule_id: str,
        attempt_number: int,
        attempt_root: Path,
        started_utc: str,
    ) -> tuple[str, str]:
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise RegistryError(
                "infrastructure evidence must be a safe relative attempt path"
            )
        candidate = attempt_root / relative
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(attempt_root.resolve())
        except (OSError, ValueError) as error:
            raise RegistryError(
                "infrastructure evidence is missing or outside the attempt"
            ) from error
        if candidate.is_symlink() or not resolved.is_file():
            raise RegistryError(
                "infrastructure evidence must be a regular non-symlink file"
            )
        try:
            payload = resolved.read_bytes()
            evidence = json.loads(payload.decode("utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise RegistryError(
                "infrastructure evidence must be a readable JSON report"
            ) from error
        # CampaignRunner emits this typed, redacted transport/health contract.
        # Its exact attempt-directory location plus this report hash binds it
        # even though it intentionally does not duplicate campaign IDs.
        if (
            isinstance(evidence, dict)
            and evidence.get("classification") == RunStatus.INFRA_INTERRUPTED.value
        ):
            observations = evidence.get("health_observations")
            service = evidence.get("service_label")
            if (
                evidence.get("schema_version") != 1
                or not isinstance(service, str)
                or not service.strip()
                or not isinstance(evidence.get("endpoint_label"), str)
                or not evidence["endpoint_label"].strip()
                or not isinstance(evidence.get("failure_kind"), str)
                or not evidence["failure_kind"].strip()
                or not isinstance(evidence.get("transport_error_type"), str)
                or not evidence["transport_error_type"].strip()
                or evidence.get("safe_stop_confirmed") is not True
                or not isinstance(observations, list)
                or not observations
                or any(
                    not isinstance(observation, dict)
                    or observation.get("ok") is not False
                    for observation in observations
                )
            ):
                raise RegistryError("typed infrastructure evidence is malformed")
            try:
                started = datetime.fromisoformat(started_utc.replace("Z", "+00:00"))
                written = datetime.fromtimestamp(resolved.stat().st_mtime, timezone.utc)
            except (ValueError, OSError) as error:
                raise RegistryError("typed infrastructure evidence time is invalid") from error
            if not started <= written <= datetime.now(timezone.utc):
                raise RegistryError("typed infrastructure evidence is not contemporaneous")
            return sha256_bytes(payload), service
        observation = evidence.get("health_observation") if isinstance(evidence, dict) else None
        required_observation = {
            "observed_utc",
            "passed",
            "service_id",
            "outage_id",
            "check_id",
            "evidence_path",
            "evidence_sha256",
        }
        if (
            not isinstance(evidence, dict)
            or evidence.get("schema_version") != 1
            or evidence.get("schedule_id") != schedule_id
            or evidence.get("attempt_number") != attempt_number
            or evidence.get("classification") != "external_infrastructure"
            or not isinstance(observation, dict)
            or not required_observation <= set(observation)
            or observation.get("passed") is not False
            or evidence.get("service_id") != observation.get("service_id")
            or evidence.get("outage_id") != observation.get("outage_id")
            or evidence.get("observed_utc") != observation.get("observed_utc")
            or not isinstance(observation.get("service_id"), str)
            or _SAFE_ID.fullmatch(observation["service_id"]) is None
            or not isinstance(observation.get("outage_id"), str)
            or _SAFE_ID.fullmatch(observation["outage_id"]) is None
            or not isinstance(observation.get("check_id"), str)
            or _SAFE_ID.fullmatch(observation["check_id"]) is None
            or _SHA256.fullmatch(str(observation.get("evidence_sha256"))) is None
        ):
            raise RegistryError(
                "infrastructure report must bind this attempt to a typed failing "
                "external-service health observation"
            )
        raw_relative = Path(str(observation.get("evidence_path", "")))
        if raw_relative.is_absolute() or ".." in raw_relative.parts or not raw_relative.parts:
            raise RegistryError("raw interruption evidence path is unsafe")
        raw_candidate = attempt_root / raw_relative
        try:
            raw_resolved = raw_candidate.resolve(strict=True)
            raw_resolved.relative_to(attempt_root.resolve())
        except (OSError, ValueError) as error:
            raise RegistryError("raw interruption evidence is missing") from error
        if (
            raw_candidate.is_symlink()
            or not raw_resolved.is_file()
            or sha256_file(raw_resolved) != observation["evidence_sha256"]
        ):
            raise RegistryError("raw interruption evidence hash mismatch")
        try:
            raw_value = json.loads(raw_resolved.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise RegistryError("raw interruption evidence must be JSON") from error
        if not isinstance(raw_value, dict) or any(
            raw_value.get(key) != observation.get(key)
            for key in (
                "observed_utc",
                "passed",
                "service_id",
                "outage_id",
                "check_id",
            )
        ):
            raise RegistryError(
                "raw interruption evidence contradicts the summarized outage"
            )
        try:
            started = datetime.fromisoformat(started_utc.replace("Z", "+00:00"))
            observed = datetime.fromisoformat(
                str(observation["observed_utc"]).replace("Z", "+00:00")
            )
        except ValueError as error:
            raise RegistryError("interruption evidence timestamp is invalid") from error
        if not started <= observed <= datetime.now(timezone.utc):
            raise RegistryError(
                "interruption evidence timestamp lies outside the live attempt"
            )
        return sha256_bytes(payload), str(observation["service_id"])

    def authorize_infrastructure_retry(
        self, schedule_id: str, recovery_evidence_path: str
    ) -> dict[str, Any]:
        """Bind a passed restoration/health/smoke report to one fresh retry.

        Recovery evidence lives outside the already-sealed attempt directory so
        authorizing a retry cannot mutate that attempt's checksum set.
        """

        if self.mode is CampaignMode.LOCKED:
            verify_frozen_campaign(self.root, check_source=True)
        with self._edit() as state:
            self._assert_authorized(state)
            if schedule_id not in state["entries"]:
                raise RegistryError(f"unknown schedule ID: {schedule_id}")
            if self._open_attempts(state):
                raise RegistryError("cannot authorize a retry while an attempt is open")
            entry = state["entries"][schedule_id]
            attempts = entry["attempts"]
            if (
                not attempts
                or entry["status"] != RunStatus.INFRA_INTERRUPTED.value
                or len(attempts) >= state["max_infrastructure_attempts"]
            ):
                raise RegistryError(
                    "recovery authorization requires a retryable infrastructure interruption"
                )
            if entry.get("retry_authorization") is not None:
                raise RegistryError("this infrastructure retry is already authorized")
            latest_number = len(attempts)
            interruption_binding = attempts[-1]["infrastructure_evidence"]
            interruption_digest, interruption_service = (
                self._validate_infrastructure_evidence(
                    interruption_binding["path"],
                    schedule_id=schedule_id,
                    attempt_number=latest_number,
                    attempt_root=(self.root / attempts[-1]["artifact_path"]),
                    started_utc=attempts[-1]["started_utc"],
                )
            )
            if interruption_digest != interruption_binding["sha256"]:
                raise RegistryError("interruption evidence changed after attempt seal")
            evidence_digest = self._validate_recovery_report(
                recovery_evidence_path,
                schedule_id=schedule_id,
                after_attempt_number=latest_number,
                after_utc=attempts[-1]["ended_utc"],
                interruption_sha256=interruption_digest,
                interruption_service_id=interruption_service,
            )
            binding = {
                "path": Path(recovery_evidence_path).as_posix(),
                "sha256": evidence_digest,
                "authorized_utc": utc_now(),
                "after_attempt_number": latest_number,
            }
            entry["retry_authorization"] = binding
            state["events"].append(
                {
                    "event": "INFRASTRUCTURE_RETRY_AUTHORIZED",
                    "utc": binding["authorized_utc"],
                    "schedule_id": schedule_id,
                    "after_attempt_number": latest_number,
                    "recovery_evidence": {
                        "path": binding["path"],
                        "sha256": binding["sha256"],
                    },
                }
            )
            return json.loads(json.dumps(binding))

    def begin_attempt(self, schedule_id: str | None = None) -> AttemptLease:
        """Reserve the next fresh attempt without creating its recorder directory."""

        if self.mode is CampaignMode.LOCKED:
            verify_frozen_campaign(self.root, check_source=True)
        with self._edit() as state:
            self._assert_authorized(state)
            open_attempts = self._open_attempts(state)
            if open_attempts:
                raise RegistryError(
                    "sequential policy forbids starting while another attempt is open"
                )
            candidates = sorted(
                state["entries"].items(),
                key=lambda pair: pair[1]["assignment"]["schedule_position"],
            )
            runnable: list[str] = []
            for key, item in candidates:
                if item["status"] == RunStatus.NOT_RUN:
                    runnable.append(key)
                elif (
                    item["status"] == RunStatus.INFRA_INTERRUPTED
                    and len(item["attempts"])
                    < state["max_infrastructure_attempts"]
                ):
                    if item.get("retry_authorization") is None:
                        raise RegistryError(
                            f"frozen order is blocked at {key}: authorize recovery "
                            "evidence before retrying"
                        )
                    runnable.append(key)
            if schedule_id is None:
                if not runnable:
                    raise RegistryError("no runnable schedule entry remains")
                schedule_id = runnable[0]
            elif schedule_id not in state["entries"]:
                raise RegistryError(f"unknown schedule ID: {schedule_id}")
            elif schedule_id not in runnable:
                requested = state["entries"][schedule_id]
                requested_status = RunStatus(requested["status"])
                if (
                    requested_status is RunStatus.INFRA_INTERRUPTED
                    and len(requested["attempts"])
                    >= state["max_infrastructure_attempts"]
                ):
                    raise RegistryError(
                        f"{schedule_id} exhausted its infrastructure retry limit"
                    )
                raise RegistryError(
                    f"{schedule_id} is sealed as {requested_status.value}; retry forbidden"
                )
            elif schedule_id != runnable[0]:
                raise RegistryError(
                    f"frozen order requires {runnable[0]} next; refusing {schedule_id}"
                )
            entry = state["entries"][schedule_id]
            status = RunStatus(entry["status"])
            if status is RunStatus.NOT_RUN and entry["attempts"]:
                raise RegistryError("unsealed attempt must be reconciled, not reused")
            if status is not RunStatus.NOT_RUN and status is not RunStatus.INFRA_INTERRUPTED:
                raise RegistryError(f"{schedule_id} is sealed as {status.value}; retry forbidden")
            if len(entry["attempts"]) >= state["max_infrastructure_attempts"]:
                raise RegistryError(f"{schedule_id} exhausted its infrastructure retry limit")
            pending_recovery = entry.get("retry_authorization")
            if status is RunStatus.INFRA_INTERRUPTED:
                if not isinstance(pending_recovery, dict):
                    raise RegistryError(
                        "infrastructure retry lacks a recovery authorization"
                    )
                prior_attempt = entry["attempts"][-1]
                interruption_binding = prior_attempt["infrastructure_evidence"]
                interruption_digest, interruption_service = (
                    self._validate_infrastructure_evidence(
                        interruption_binding["path"],
                        schedule_id=schedule_id,
                        attempt_number=len(entry["attempts"]),
                        attempt_root=self.root / prior_attempt["artifact_path"],
                        started_utc=prior_attempt["started_utc"],
                    )
                )
                if interruption_digest != interruption_binding["sha256"]:
                    raise RegistryError(
                        "interruption evidence changed after attempt seal"
                    )
                self._validate_recovery_report(
                    pending_recovery["path"],
                    schedule_id=schedule_id,
                    after_attempt_number=len(entry["attempts"]),
                    after_utc=entry["attempts"][-1]["ended_utc"],
                    interruption_sha256=interruption_digest,
                    interruption_service_id=interruption_service,
                    expected_sha256=pending_recovery["sha256"],
                )
            attempt_number = len(entry["attempts"]) + 1
            expected = (
                self.root
                / "episodes"
                / schedule_id
                / f"attempt_{attempt_number}"
            )
            if expected.exists():
                raise RegistryError(f"refusing to reuse existing attempt path: {expected}")
            now = utc_now()
            attempt = {
                "attempt_number": attempt_number,
                "attempt_id": f"{schedule_id}-A{attempt_number}",
                "status": RunStatus.NOT_RUN.value,
                "started_utc": now,
                "ended_utc": None,
                "oracle_verdict": None,
                "reason": None,
                "artifact_path": expected.relative_to(self.root).as_posix(),
                "artifact_audit_passed": None,
                "infrastructure_evidence": None,
                "recovery_authorization": entry.get("retry_authorization"),
            }
            entry["retry_authorization"] = None
            entry["attempts"].append(attempt)
            entry["status"] = RunStatus.NOT_RUN.value
            entry["disposition"] = None
            state["events"].append(
                {
                    "event": "ATTEMPT_RESERVED",
                    "utc": now,
                    "schedule_id": schedule_id,
                    "attempt_number": attempt_number,
                }
            )
            assignment = ScheduleEntry.from_dict(entry["assignment"])
            return AttemptLease(
                campaign_id=state["campaign_id"],
                schedule_id=schedule_id,
                attempt_number=attempt_number,
                attempt_id=attempt["attempt_id"],
                attempt_path=expected,
                assignment=assignment,
            )

    def seal_attempt(
        self,
        schedule_id: str,
        attempt_number: int,
        status: RunStatus | str,
        *,
        oracle_verdict: str | None = None,
        reason: str | None = None,
        artifact_audit_passed: bool | None = None,
        independently_evidenced_infrastructure: bool = False,
        infrastructure_evidence: str | None = None,
    ) -> None:
        actual_status = RunStatus(status)
        if actual_status not in SEALED_STATUSES:
            raise RegistryError("an attempt may only be sealed with a terminal status")
        if actual_status in {
            RunStatus.INFRA_INTERRUPTED,
            RunStatus.INVALID_HARNESS,
            RunStatus.ABORTED_SAFETY,
        } and (reason is None or not reason.strip()):
            raise RegistryError(f"{actual_status.value} requires a documented reason")
        if (
            actual_status is RunStatus.INFRA_INTERRUPTED
            and not independently_evidenced_infrastructure
        ):
            raise RegistryError(
                "INFRA_INTERRUPTED requires independent external evidence; otherwise "
                "seal INVALID_HARNESS or a valid studied-system outcome"
            )
        if actual_status is RunStatus.INFRA_INTERRUPTED and not infrastructure_evidence:
            raise RegistryError(
                "INFRA_INTERRUPTED requires a relative evidence-file path under the "
                "attempt directory"
            )
        if (
            actual_status is not RunStatus.INFRA_INTERRUPTED
            and (independently_evidenced_infrastructure or infrastructure_evidence)
        ):
            raise RegistryError(
                "the infrastructure-evidence assertion is valid only for "
                "INFRA_INTERRUPTED"
            )
        if actual_status in {
            RunStatus.VALID_PASS,
            RunStatus.VALID_SYSTEM_FAILURE,
        } and artifact_audit_passed is not True:
            raise RegistryError(
                "a passing artifact audit is required for a valid experimental outcome"
            )
        if actual_status in {
            RunStatus.VALID_PASS,
            RunStatus.VALID_SYSTEM_FAILURE,
        }:
            try:
                from .audit import audit_attempt_artifacts

                independent_audit = audit_attempt_artifacts(
                    self.root
                    / "episodes"
                    / schedule_id
                    / f"attempt_{attempt_number}",
                    verify_video=True,
                    require_final_state=True,
                )
            except Exception as error:
                raise RegistryError(
                    f"independent artifact audit could not run: {error}"
                ) from error
            if independent_audit.get("passed") is not True:
                detail = "; ".join(map(str, independent_audit.get("errors", ())))
                raise RegistryError(
                    "valid outcome refused because the independent artifact audit "
                    f"failed{': ' + detail if detail else ''}"
                )
        with self._edit() as state:
            if schedule_id not in state["entries"]:
                raise RegistryError(f"unknown schedule ID: {schedule_id}")
            entry = state["entries"][schedule_id]
            if not entry["attempts"] or attempt_number != len(entry["attempts"]):
                raise RegistryError("only the current contiguous attempt may be sealed")
            attempt = entry["attempts"][-1]
            if attempt["status"] != RunStatus.NOT_RUN:
                raise RegistryError("sealed attempts are immutable")
            if actual_status in {
                RunStatus.VALID_PASS,
                RunStatus.VALID_SYSTEM_FAILURE,
            }:
                expected_verdict = (
                    "PASS" if actual_status is RunStatus.VALID_PASS else "FAIL"
                )
                if oracle_verdict != expected_verdict:
                    raise RegistryError(
                        f"{actual_status.value} requires oracle verdict "
                        f"{expected_verdict}"
                    )
            evidence_binding: dict[str, str] | None = None
            if actual_status is RunStatus.INFRA_INTERRUPTED:
                relative_evidence = Path(str(infrastructure_evidence))
                attempt_root = (self.root / attempt["artifact_path"]).resolve()
                evidence_digest, _ = self._validate_infrastructure_evidence(
                    relative_evidence.as_posix(),
                    schedule_id=schedule_id,
                    attempt_number=attempt_number,
                    attempt_root=attempt_root,
                    started_utc=attempt["started_utc"],
                )
                evidence_binding = {
                    "path": relative_evidence.as_posix(),
                    "sha256": evidence_digest,
                }
            now = utc_now()
            attempt["status"] = actual_status.value
            attempt["ended_utc"] = now
            attempt["oracle_verdict"] = oracle_verdict
            attempt["reason"] = reason
            attempt["artifact_audit_passed"] = artifact_audit_passed
            attempt["infrastructure_evidence"] = evidence_binding
            entry["status"] = actual_status.value
            if (
                actual_status is RunStatus.INFRA_INTERRUPTED
                and len(entry["attempts"]) >= state["max_infrastructure_attempts"]
            ):
                entry["disposition"] = "QUARANTINED_INFRA_RETRY_EXHAUSTED"
            state["events"].append(
                {
                    "event": "ATTEMPT_SEALED",
                    "utc": now,
                    "schedule_id": schedule_id,
                    "attempt_number": attempt_number,
                    "status": actual_status.value,
                    "independently_evidenced_infrastructure": (
                        independently_evidenced_infrastructure
                    ),
                }
            )
        if actual_status in {
            RunStatus.INFRA_INTERRUPTED,
            RunStatus.INVALID_HARNESS,
            RunStatus.ABORTED_SAFETY,
        }:
            self._append_exclusion(
                utc=now,
                schedule_id=schedule_id,
                attempt_number=attempt_number,
                status=actual_status,
                reason=str(reason),
            )

    finalize_attempt = seal_attempt

    def reconcile_open_attempts(
        self,
        *,
        reason: str,
        status: RunStatus | str = RunStatus.INVALID_HARNESS,
        independently_evidenced_infrastructure: bool = False,
        infrastructure_evidence: str | None = None,
    ) -> int:
        """Seal crash-left reservations without laundering failures as retries."""

        if not reason.strip():
            raise RegistryError("reconciliation reason must be non-empty")
        actual_status = RunStatus(status)
        if actual_status not in {
            RunStatus.INFRA_INTERRUPTED,
            RunStatus.INVALID_HARNESS,
            RunStatus.ABORTED_SAFETY,
        }:
            raise RegistryError("open attempts require an infra, harness, or safety disposition")
        if (
            actual_status is RunStatus.INFRA_INTERRUPTED
            and not independently_evidenced_infrastructure
        ):
            raise RegistryError(
                "INFRA_INTERRUPTED reconciliation requires independent external evidence"
            )
        if actual_status is RunStatus.INFRA_INTERRUPTED and not infrastructure_evidence:
            raise RegistryError(
                "INFRA_INTERRUPTED reconciliation requires a relative evidence-file path"
            )
        if actual_status is not RunStatus.INFRA_INTERRUPTED and (
            independently_evidenced_infrastructure or infrastructure_evidence
        ):
            raise RegistryError(
                "infrastructure evidence may accompany only INFRA_INTERRUPTED"
            )
        reconciled = 0
        exclusions: list[tuple[str, str, int]] = []
        with self._edit() as state:
            for schedule_id, entry in state["entries"].items():
                if not entry["attempts"]:
                    continue
                attempt = entry["attempts"][-1]
                if attempt["status"] != RunStatus.NOT_RUN:
                    continue
                evidence_binding: dict[str, str] | None = None
                if actual_status is RunStatus.INFRA_INTERRUPTED:
                    relative_evidence = Path(str(infrastructure_evidence))
                    attempt_root = (self.root / attempt["artifact_path"]).resolve()
                    evidence_digest, _ = self._validate_infrastructure_evidence(
                        relative_evidence.as_posix(),
                        schedule_id=schedule_id,
                        attempt_number=attempt["attempt_number"],
                        attempt_root=attempt_root,
                        started_utc=attempt["started_utc"],
                    )
                    evidence_binding = {
                        "path": relative_evidence.as_posix(),
                        "sha256": evidence_digest,
                    }
                now = utc_now()
                attempt["status"] = actual_status.value
                attempt["ended_utc"] = now
                attempt["reason"] = reason
                attempt["artifact_audit_passed"] = False
                attempt["infrastructure_evidence"] = evidence_binding
                entry["status"] = actual_status.value
                if (
                    actual_status is RunStatus.INFRA_INTERRUPTED
                    and len(entry["attempts"]) >= state["max_infrastructure_attempts"]
                ):
                    entry["disposition"] = "QUARANTINED_INFRA_RETRY_EXHAUSTED"
                state["events"].append(
                    {
                        "event": "OPEN_ATTEMPT_RECONCILED",
                        "utc": now,
                        "schedule_id": schedule_id,
                        "attempt_number": attempt["attempt_number"],
                        "status": actual_status.value,
                    }
                )
                exclusions.append((now, schedule_id, attempt["attempt_number"]))
                reconciled += 1
        for observed_utc, observed_schedule, observed_attempt in exclusions:
            self._append_exclusion(
                utc=observed_utc,
                schedule_id=observed_schedule,
                attempt_number=observed_attempt,
                status=actual_status,
                reason=reason,
            )
        return reconciled

    def _append_exclusion(
        self,
        *,
        utc: str,
        schedule_id: str,
        attempt_number: int,
        status: RunStatus,
        reason: str,
    ) -> None:
        if status not in {
            RunStatus.INFRA_INTERRUPTED,
            RunStatus.INVALID_HARNESS,
            RunStatus.ABORTED_SAFETY,
        }:
            return
        path = self.root / "EXCLUSIONS.md"
        if not path.exists():
            _exclusive_write(
                path,
                b"# Exclusions\n\nAppend-only excluded/invalid attempt dispositions follow.\n",
            )
        if not path.is_file() or path.is_symlink():
            raise RegistryError("EXCLUSIONS.md is missing or unsafe")
        safe_reason = reason.replace("\r", " ").replace("\n", " ").replace("|", "\\|")
        line = (
            f"- EXCLUSION schedule_id={schedule_id} attempt={attempt_number} "
            f"status={status.value} utc={utc} reason={safe_reason}\n"
        )
        with path.open("ab") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.write(line.encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        _fsync_directory(path.parent)

    def status_counts(self) -> dict[str, int]:
        state = self.snapshot()
        counts = Counter(item["status"] for item in state["entries"].values())
        return {status.value: counts[status.value] for status in RunStatus}

    def export_run_index(self) -> Path:
        """Regenerate the denormalized CSV index from authoritative RUN_STATE."""

        state = self.snapshot()
        output = self.root / "run_index.csv"
        descriptor, temporary_name = tempfile.mkstemp(
            dir=output.parent, prefix=".run_index.", suffix=".tmp"
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
                handle.write(run_index_csv_bytes(state).decode("utf-8"))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, output)
            _fsync_directory(output.parent)
        finally:
            if temporary.exists():
                temporary.unlink()
        return output

    def completion(self) -> dict[str, Any]:
        state = self.snapshot()
        retryable = 0
        recovery_blocked = 0
        quarantined = 0
        invalid_harness = 0
        open_count = len(self._open_attempts(state))
        for item in state["entries"].values():
            if item["status"] == RunStatus.NOT_RUN:
                if not item["attempts"]:
                    retryable += 1
            elif item["status"] == RunStatus.INFRA_INTERRUPTED:
                if len(item["attempts"]) < state["max_infrastructure_attempts"]:
                    retryable += 1
                    if item.get("retry_authorization") is None:
                        recovery_blocked += 1
                else:
                    quarantined += 1
            elif item["status"] == RunStatus.INVALID_HARNESS:
                invalid_harness += 1
        execution_closed = retryable == 0 and open_count == 0 and invalid_harness == 0
        return {
            "complete": execution_closed and quarantined == 0,
            "execution_closed_with_documented_unresolved": (
                execution_closed and quarantined > 0
            ),
            "retryable_or_not_run": retryable,
            "infrastructure_retries_awaiting_recovery": recovery_blocked,
            "open_attempts": open_count,
            "quarantined_infrastructure": quarantined,
            "invalid_harness": invalid_harness,
            "status_counts": self.status_counts(),
        }


def _git_text(repo_root: Path, *arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.rstrip("\n")


def _git_stream_sha256(repo_root: Path, *arguments: str) -> str | None:
    try:
        process = subprocess.Popen(
            ["git", *arguments],
            cwd=repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return None
    digest = hashlib.sha256()
    assert process.stdout is not None
    with process.stdout:
        while chunk := process.stdout.read(1024 * 1024):
            digest.update(chunk)
    if process.wait() != 0:
        return None
    return digest.hexdigest()


_SOURCE_EXCLUDED_PARTS = frozenset(
    {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "runs"}
)


def _manifest_files(repo_root: Path, roots: Sequence[Path]) -> list[dict[str, Any]]:
    resolved_root = repo_root.resolve()
    files: dict[str, Path] = {}
    for requested in roots:
        path = requested if requested.is_absolute() else repo_root / requested
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(resolved_root)
        except (OSError, ValueError) as error:
            raise FreezeError(f"provenance path is unavailable or outside repository: {path}") from error
        candidates = [resolved] if resolved.is_file() else resolved.rglob("*")
        for candidate in candidates:
            relative = candidate.relative_to(resolved_root)
            if set(relative.parts) & _SOURCE_EXCLUDED_PARTS:
                continue
            if candidate.is_symlink():
                raise FreezeError(f"refusing to freeze symlink: {relative}")
            if candidate.is_file():
                files[relative.as_posix()] = candidate
    return [
        {
            "path": relative,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for relative, path in sorted(files.items())
    ]


def collect_provenance(
    repo_root: str | Path,
    *,
    source_roots: Sequence[str | Path] | None = None,
    frozen_paths: Sequence[str | Path] = (),
) -> dict[str, Any]:
    """Capture commit, dirty patch, and complete content manifests."""

    root = Path(repo_root).resolve()
    defaults = [
        path
        for path in (
            "src",
            "simulation",
            "experiments",
            "tests",
            "PROJECT_CONTEXT.md",
            "README.md",
            "EXPERIMENT_SUITE_DESIGN.md",
            "pyproject.toml",
            "uv.lock",
        )
        if (root / path).exists()
    ]
    selected_roots = [Path(path) for path in (source_roots or defaults)]
    selected_frozen = [Path(path) for path in frozen_paths]
    source_files = _manifest_files(root, selected_roots)
    frozen_files = _manifest_files(root, selected_frozen)
    status = _git_text(root, "status", "--porcelain=v1", "--untracked-files=all")
    return {
        "repository_root_label": root.name,
        "git_commit": _git_text(root, "rev-parse", "HEAD"),
        "git_describe": _git_text(root, "describe", "--always", "--dirty", "--tags"),
        "dirty": bool(status),
        "git_status_sha256": (
            sha256_bytes((status or "").encode("utf-8")) if status is not None else None
        ),
        "dirty_patch_sha256": _git_stream_sha256(root, "diff", "--binary", "HEAD", "--", "."),
        "source_roots": _relative_roots(root, selected_roots),
        "source_files": source_files,
        "source_manifest_sha256": sha256_bytes(canonical_json_bytes(source_files)),
        "frozen_input_roots": _relative_roots(root, selected_frozen),
        "frozen_inputs": frozen_files,
        "frozen_inputs_sha256": sha256_bytes(canonical_json_bytes(frozen_files)),
    }


def _read_protocol(value: str | Path) -> bytes:
    if isinstance(value, Path):
        return value.read_bytes()
    candidate = Path(value)
    if "\n" not in value and candidate.is_file():
        return candidate.read_bytes()
    return value.encode("utf-8")


def _markdown_prose_mask(source: str) -> str:
    """Mask code while preserving offsets used to rewrite Markdown links."""

    def masked_line(line: str) -> str:
        return "".join(
            "\n" if char == "\n" else "\r" if char == "\r" else " "
            for char in line
        )

    masked: list[str] = []
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
            masked.append(masked_line(line))
            continue
        if marker is not None:
            fence_character = marker.group("run")[0]
            fence_length = len(marker.group("run"))
            masked.append(masked_line(line))
            continue
        line_mask = list(line)
        for inline in re.finditer(r"`+[^`\n]*`+", line):
            for index in range(inline.start(), inline.end()):
                if line_mask[index] not in {"\n", "\r"}:
                    line_mask[index] = " "
        masked.append("".join(line_mask))
    return "".join(masked)


def _markdown_destination(raw: str) -> tuple[str, int, int]:
    """Return a Markdown link destination and its offsets in the raw target."""

    start = len(raw) - len(raw.lstrip())
    value = raw[start:]
    if value.startswith("<"):
        end = value.find(">", 1)
        if end < 0:
            raise FreezeError("unterminated angle-bracket Markdown destination")
        return value[1:end], start + 1, start + end
    match = re.match(r"[^\s]+", value)
    if match is None:
        raise FreezeError("empty Markdown destination")
    return match.group(0), start + match.start(), start + match.end()


def _source_link_path(entry_path: str, destination: str) -> str | None:
    """Resolve a local documentation link to a safe repository-relative path."""

    parsed = urlsplit(destination)
    if parsed.scheme in {"http", "https", "mailto"} and not parsed.netloc.startswith(
        "."
    ):
        return None
    if parsed.scheme or parsed.netloc:
        raise FreezeError(
            f"unsupported documentation link scheme in {entry_path}: {destination!r}"
        )
    decoded = unquote(parsed.path)
    if not decoded:
        return None
    if Path(decoded).is_absolute() or "\x00" in decoded:
        raise FreezeError(
            f"unsafe documentation link in {entry_path}: {destination!r}"
        )
    normalized = posixpath.normpath(
        posixpath.join(posixpath.dirname(entry_path), decoded)
    )
    if normalized in {"", ".", ".."} or normalized.startswith("../"):
        raise FreezeError(
            f"documentation link escapes the repository in {entry_path}: "
            f"{destination!r}"
        )
    return normalized


def _snapshot_path_for_source(source_path: str) -> str:
    suffix = Path(source_path).suffix.casefold()
    snapshot = f"{_DOCUMENTATION_SNAPSHOT_ROOT}/{source_path}"
    # Exact Markdown source bytes remain available but are deliberately stored
    # as inert source blobs.  Only the rewritten campaign entry is interpreted
    # as Markdown, so links outside the crosswalk are neither silently rebased
    # nor mistaken for campaign-artifact links.
    if suffix in _MARKDOWN_SUFFIXES:
        snapshot += ".source"
    return snapshot


def _render_documentation_entry(
    source: str,
    *,
    source_entry_path: str,
    targets: Mapping[str, str],
) -> str:
    """Rewrite only local link destinations to immutable snapshot files."""

    masked = _markdown_prose_mask(source)
    rendered: list[str] = []
    cursor = 0
    for match in _MARKDOWN_LINK.finditer(masked):
        raw_start, raw_end = match.span("target")
        raw = source[raw_start:raw_end]
        destination, destination_start, destination_end = _markdown_destination(raw)
        source_path = _source_link_path(source_entry_path, destination)
        if source_path is None:
            continue
        snapshot_path = targets.get(source_path)
        if snapshot_path is None:
            raise FreezeError(
                f"documentation snapshot lacks local target {source_path!r}"
            )
        parsed = urlsplit(destination)
        rewritten = quote(snapshot_path, safe="/._-")
        if parsed.query:
            rewritten += f"?{parsed.query}"
        if parsed.fragment:
            rewritten += f"#{parsed.fragment}"
        absolute_start = raw_start + destination_start
        absolute_end = raw_start + destination_end
        rendered.extend((source[cursor:absolute_start], rewritten))
        cursor = absolute_end
    rendered.append(source[cursor:])
    return "".join(rendered)


def _documentation_snapshot_plan(
    repository: Path,
) -> tuple[bytes, dict[str, Any], list[tuple[str, bytes]]]:
    """Build a content-bound, campaign-relative requirements snapshot plan."""

    candidates = (
        "experiments/REQUIREMENTS_CROSSWALK.md",
        "experiments/CLAIM_EXPERIMENT_CROSSWALK.md",
    )
    source_entry_path = next(
        (relative for relative in candidates if (repository / relative).is_file()),
        None,
    )
    if source_entry_path is None:
        rendered = (
            b"# Requirements crosswalk\n\n"
            b"Global crosswalk unavailable at freeze time.\n"
        )
        specification: dict[str, Any] = {
            "schema_version": 1,
            "source_entry_path": None,
            "source_entry_sha256": None,
            "canonical_snapshot_path": None,
            "rendered_entry_path": "REQUIREMENTS_CROSSWALK.md",
            "rendered_entry_sha256": sha256_bytes(rendered),
            "files": [],
            "files_manifest_sha256": sha256_bytes(canonical_json_bytes([])),
        }
        return rendered, specification, []

    entry = repository / source_entry_path
    if entry.is_symlink():
        raise FreezeError("refusing to snapshot a symlink requirements crosswalk")
    try:
        source_bytes = entry.read_bytes()
        source_text = source_bytes.decode("utf-8")
    except (OSError, UnicodeError) as error:
        raise FreezeError(f"cannot read requirements crosswalk: {error}") from error

    target_paths: set[str] = set()
    masked = _markdown_prose_mask(source_text)
    for match in _MARKDOWN_LINK.finditer(masked):
        raw_start, raw_end = match.span("target")
        destination, _, _ = _markdown_destination(source_text[raw_start:raw_end])
        source_path = _source_link_path(source_entry_path, destination)
        if source_path is not None:
            target_paths.add(source_path)

    source_paths = sorted(target_paths | {source_entry_path})
    snapshot_paths = {
        source_path: _snapshot_path_for_source(source_path)
        for source_path in source_paths
    }
    if len(set(snapshot_paths.values())) != len(snapshot_paths):
        raise FreezeError("documentation snapshot paths collide")

    files: list[dict[str, Any]] = []
    payloads: list[tuple[str, bytes]] = []
    root = repository.resolve()
    for source_path in source_paths:
        candidate = repository / source_path
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(root)
        except (OSError, ValueError) as error:
            raise FreezeError(
                f"missing or external documentation target: {source_path}"
            ) from error
        if candidate.is_symlink() or not resolved.is_file():
            raise FreezeError(
                f"documentation target is not a regular non-symlink file: {source_path}"
            )
        data = resolved.read_bytes()
        snapshot_path = snapshot_paths[source_path]
        files.append(
            {
                "source_path": source_path,
                "snapshot_path": snapshot_path,
                "size_bytes": len(data),
                "sha256": sha256_bytes(data),
            }
        )
        payloads.append((snapshot_path, data))

    rendered = _render_documentation_entry(
        source_text,
        source_entry_path=source_entry_path,
        targets=snapshot_paths,
    ).encode("utf-8")
    specification = {
        "schema_version": 1,
        "source_entry_path": source_entry_path,
        "source_entry_sha256": sha256_bytes(source_bytes),
        "canonical_snapshot_path": snapshot_paths[source_entry_path],
        "rendered_entry_path": "REQUIREMENTS_CROSSWALK.md",
        "rendered_entry_sha256": sha256_bytes(rendered),
        "files": files,
        "files_manifest_sha256": sha256_bytes(canonical_json_bytes(files)),
    }
    return rendered, specification, payloads


def _verify_documentation_snapshot(
    campaign_root: Path,
    value: Any,
) -> list[dict[str, Any]]:
    """Verify immutable documentation bytes and deterministic link rewriting."""

    required = {
        "schema_version",
        "source_entry_path",
        "source_entry_sha256",
        "canonical_snapshot_path",
        "rendered_entry_path",
        "rendered_entry_sha256",
        "files",
        "files_manifest_sha256",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise FreezeError("malformed documentation_snapshot manifest")
    if value.get("schema_version") != 1:
        raise FreezeError("unsupported documentation_snapshot schema")
    if value.get("rendered_entry_path") != "REQUIREMENTS_CROSSWALK.md":
        raise FreezeError("documentation snapshot has an invalid rendered entry path")
    rendered_digest = value.get("rendered_entry_sha256")
    files_digest = value.get("files_manifest_sha256")
    if _SHA256.fullmatch(str(rendered_digest)) is None or _SHA256.fullmatch(
        str(files_digest)
    ) is None:
        raise FreezeError("documentation snapshot has an invalid digest")

    raw_files = value.get("files")
    if not isinstance(raw_files, list):
        raise FreezeError("documentation snapshot files must be an array")
    files: list[dict[str, Any]] = []
    seen_snapshots: set[str] = set()
    for index, item in enumerate(raw_files):
        if not isinstance(item, dict) or set(item) != {
            "source_path",
            "snapshot_path",
            "size_bytes",
            "sha256",
        }:
            raise FreezeError(f"malformed documentation snapshot file {index}")
        source_path = item.get("source_path")
        snapshot_path = item.get("snapshot_path")
        size = item.get("size_bytes")
        digest = item.get("sha256")
        if (
            not isinstance(source_path, str)
            or not source_path
            or Path(source_path).is_absolute()
            or ".." in Path(source_path).parts
            or not isinstance(snapshot_path, str)
            or not snapshot_path.startswith(f"{_DOCUMENTATION_SNAPSHOT_ROOT}/")
            or Path(snapshot_path).is_absolute()
            or ".." in Path(snapshot_path).parts
            or snapshot_path in seen_snapshots
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or _SHA256.fullmatch(str(digest)) is None
        ):
            raise FreezeError(f"unsafe documentation snapshot file {index}")
        seen_snapshots.add(snapshot_path)
        normalized = {
            "source_path": source_path,
            "snapshot_path": snapshot_path,
            "size_bytes": size,
            "sha256": digest,
        }
        files.append(normalized)
    if [item["source_path"] for item in files] != sorted(
        {item["source_path"] for item in files}
    ):
        raise FreezeError("documentation snapshot source paths must be unique and sorted")
    if sha256_bytes(canonical_json_bytes(files)) != files_digest:
        raise FreezeError("documentation snapshot file manifest digest mismatch")

    rendered = _safe_frozen_file(
        campaign_root,
        value["rendered_entry_path"],
        "rendered documentation entry",
    )
    if sha256_file(rendered) != rendered_digest:
        raise FreezeError("rendered requirements crosswalk hash mismatch")
    for item in files:
        snapshot = _safe_frozen_file(
            campaign_root,
            item["snapshot_path"],
            "documentation snapshot file",
        )
        if snapshot.stat().st_size != item["size_bytes"] or sha256_file(
            snapshot
        ) != item["sha256"]:
            raise FreezeError(
                f"documentation snapshot file changed: {item['snapshot_path']}"
            )

    source_entry_path = value.get("source_entry_path")
    if source_entry_path is None:
        if (
            value.get("source_entry_sha256") is not None
            or value.get("canonical_snapshot_path") is not None
            or files
        ):
            raise FreezeError("placeholder documentation snapshot is contradictory")
        return files
    if (
        not isinstance(source_entry_path, str)
        or source_entry_path
        not in {
            "experiments/REQUIREMENTS_CROSSWALK.md",
            "experiments/CLAIM_EXPERIMENT_CROSSWALK.md",
        }
        or _SHA256.fullmatch(str(value.get("source_entry_sha256"))) is None
    ):
        raise FreezeError("documentation snapshot source entry is invalid")
    by_source = {item["source_path"]: item for item in files}
    source_item = by_source.get(source_entry_path)
    if (
        source_item is None
        or value.get("canonical_snapshot_path") != source_item["snapshot_path"]
        or value.get("source_entry_sha256") != source_item["sha256"]
    ):
        raise FreezeError("canonical documentation source binding is invalid")
    canonical = _safe_frozen_file(
        campaign_root,
        source_item["snapshot_path"],
        "canonical documentation source",
    )
    try:
        source_text = canonical.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise FreezeError(f"cannot decode canonical documentation source: {error}") from error
    masked = _markdown_prose_mask(source_text)
    expected_targets: set[str] = set()
    for match in _MARKDOWN_LINK.finditer(masked):
        raw_start, raw_end = match.span("target")
        destination, _, _ = _markdown_destination(source_text[raw_start:raw_end])
        source_path = _source_link_path(source_entry_path, destination)
        if source_path is not None:
            expected_targets.add(source_path)
    if set(by_source) != expected_targets | {source_entry_path}:
        raise FreezeError("documentation snapshot target membership mismatch")
    regenerated = _render_documentation_entry(
        source_text,
        source_entry_path=source_entry_path,
        targets={
            source_path: item["snapshot_path"]
            for source_path, item in by_source.items()
        },
    ).encode("utf-8")
    if rendered.read_bytes() != regenerated:
        raise FreezeError("rendered requirements crosswalk is not canonical")
    return files


def _manifest_payload_hash(manifest: Mapping[str, Any]) -> str:
    payload = dict(manifest)
    payload.pop("manifest_sha256", None)
    return sha256_bytes(canonical_json_bytes(payload))


def freeze_protocol(
    campaign_root: str | Path,
    *,
    campaign_id: str,
    mode: CampaignMode | str,
    protocol_source: str | Path,
    schedule: Sequence[ScheduleEntry],
    repo_root: str | Path,
    frozen_paths: Sequence[str | Path],
    source_roots: Sequence[str | Path] | None = None,
    metadata: Mapping[str, Any] | None = None,
    reuse_policy: ReusePolicy | str = ReusePolicy.FRESH_ATTEMPTS_ONLY,
) -> dict[str, Any]:
    """Create an immutable protocol snapshot and full provenance manifest.

    Repeating the operation with an already-valid identical campaign is an
    idempotent read.  Any mismatch is a hard refusal; callers must create a new
    campaign ID/version instead of overwriting the existing freeze.
    """

    root = Path(campaign_root).resolve()
    repo = Path(repo_root).resolve()
    _require_id(campaign_id, "campaign_id")
    actual_mode = CampaignMode(mode)
    actual_policy = ReusePolicy(reuse_policy)
    if actual_policy is ReusePolicy.PREREGISTERED_SHARED_REFERENCE:
        raise FreezeError(
            "shared-reference outcome reuse is unsupported; use fresh_attempts_only"
        )
    validate_schedule(schedule)
    frozen_metadata = _normalize_campaign_metadata(metadata)
    reference_profile = frozen_metadata["analysis_spec"]["reference_profile"]
    profiles_by_matrix: dict[str, set[str]] = {}
    for entry in schedule:
        profiles_by_matrix.setdefault(entry.matrix_id, set()).add(entry.profile_id)
    derived_families = [
        {
            "matrix_id": matrix_id,
            "reference_profile": reference_profile,
            "comparison_profiles": sorted(profiles - {reference_profile}),
        }
        for matrix_id, profiles in sorted(profiles_by_matrix.items())
    ]
    supplied_families = frozen_metadata["analysis_spec"].get("families", [])
    if supplied_families and supplied_families != derived_families:
        raise FreezeError("analysis_spec families differ from the frozen schedule")
    frozen_metadata["analysis_spec"]["families"] = derived_families
    if actual_mode is CampaignMode.LOCKED:
        for family in derived_families:
            profiles = profiles_by_matrix[family["matrix_id"]]
            if reference_profile not in profiles or not family["comparison_profiles"]:
                raise FreezeError(
                    f"locked analysis family {family['matrix_id']} must schedule the "
                    "reference and at least one comparison profile"
                )
    frozen_engine: dict[str, str] | None = None
    if actual_mode is CampaignMode.LOCKED:
        frozen_engine = _validate_locked_execution_engine(frozen_metadata, repo)
        _validate_locked_runtime_metadata(frozen_metadata, populate_hashes=True)
    protocol_bytes = _read_protocol(protocol_source)
    protocol_hash = sha256_bytes(protocol_bytes)
    requested_schedule_hash = schedule_sha256(schedule)

    manifest_path = root / "CAMPAIGN_MANIFEST.json"
    if manifest_path.exists():
        existing = verify_frozen_campaign(root, check_source=True, repo_root=repo)
        expected = (
            campaign_id,
            actual_mode.value,
            protocol_hash,
            requested_schedule_hash,
            actual_policy.value,
        )
        observed = (
            existing["campaign_id"],
            existing["mode"],
            existing["protocol_sha256"],
            existing["schedule_sha256"],
            existing["reuse_policy"],
        )
        if expected != observed:
            raise FreezeError("refusing to overwrite a different frozen campaign")
        defaults = [
            path
            for path in (
                "src",
                "simulation",
                "experiments",
                "tests",
                "PROJECT_CONTEXT.md",
                "README.md",
                "EXPERIMENT_SUITE_DESIGN.md",
                "pyproject.toml",
                "uv.lock",
            )
            if (repo / path).exists()
        ]
        requested_sources = _manifest_files(
            repo, [Path(path) for path in (source_roots or defaults)]
        )
        requested_inputs = _manifest_files(
            repo, [Path(path) for path in frozen_paths]
        )
        if requested_sources != existing.get("provenance", {}).get("source_files"):
            raise FreezeError("refusing to change the frozen source-root manifest")
        if requested_inputs != existing.get("provenance", {}).get("frozen_inputs"):
            raise FreezeError("refusing to change the frozen input manifest")
        if frozen_metadata != existing.get("metadata"):
            raise FreezeError("refusing to change frozen campaign metadata")
        return existing
    if root.exists() and any(path.is_file() for path in root.rglob("*")):
        raise FreezeError("campaign directory contains files but no valid frozen manifest")

    (
        rendered_crosswalk,
        documentation_snapshot,
        documentation_payloads,
    ) = _documentation_snapshot_plan(repo)
    root.mkdir(parents=True, exist_ok=True)
    if actual_mode is CampaignMode.PILOT and any(
        entry.split != "pilot" for entry in schedule
    ):
        raise FreezeError("pilot freeze contains a non-pilot schedule entry")
    if actual_mode is CampaignMode.LOCKED and any(
        entry.split not in {"locked", "locked_test"} for entry in schedule
    ):
        raise FreezeError("locked freeze contains a non-locked schedule entry")
    for directory in ("environment", "analysis", "artifact_audit", "episodes"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    provenance = collect_provenance(
        repo, source_roots=source_roots, frozen_paths=frozen_paths
    )
    if frozen_engine is not None and frozen_engine["source_path"] not in {
        item["path"] for item in provenance["source_files"]
    }:
        raise FreezeError(
            "locked execution engine source must be included in source provenance"
        )
    _exclusive_write(root / "PROTOCOL.md", protocol_bytes)
    _exclusive_write(
        root / "PROTOCOL.sha256",
        f"{protocol_hash}  PROTOCOL.md\n".encode("ascii"),
    )
    write_schedule(root / "run_schedule.csv", schedule)
    _exclusive_write(
        root / "run_index.csv",
        (
            "schedule_id,attempt_number,run_status,oracle_verdict,artifact_path,"
            "artifact_audit_passed\n"
        ).encode("utf-8"),
    )
    _exclusive_write(
        root / "EXCLUSIONS.md",
        b"# Exclusions\n\nAppend-only excluded/invalid attempt dispositions follow.\n",
    )
    _exclusive_write(
        root / "CHANGELOG.md",
        (
            "# Campaign changelog\n\n"
            f"- {utc_now()}: protocol version frozen.\n"
        ).encode("utf-8"),
    )
    _exclusive_write(root / "REQUIREMENTS_CROSSWALK.md", rendered_crosswalk)
    for snapshot_path, payload in documentation_payloads:
        _exclusive_write(root / snapshot_path, payload)
    _atomic_json(root / "environment" / "provenance.json", provenance, exclusive=True)
    _exclusive_write(
        root / "analysis" / "README.md",
        b"# Analysis\n\nGenerated analysis belongs here; raw attempts remain immutable.\n",
    )
    _exclusive_write(
        root / "artifact_audit" / "README.md",
        b"# Artifact audit\n\nIndependent audit outputs belong here.\n",
    )

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": campaign_id,
        "mode": actual_mode.value,
        "frozen_utc": utc_now(),
        "protocol_path": "PROTOCOL.md",
        "protocol_sha256": protocol_hash,
        "schedule_path": "run_schedule.csv",
        "schedule_sha256": requested_schedule_hash,
        "schedule_size": len(schedule),
        "sequential_execution": True,
        "common_random_numbers": True,
        "reuse_policy": actual_policy.value,
        "max_infrastructure_attempts": MAX_INFRASTRUCTURE_ATTEMPTS,
        "locked_approval_required": actual_mode is CampaignMode.LOCKED,
        "provenance": provenance,
        "documentation_snapshot": documentation_snapshot,
        "metadata": frozen_metadata,
    }
    manifest["manifest_sha256"] = _manifest_payload_hash(manifest)
    _atomic_json(manifest_path, manifest, exclusive=True)
    CampaignRegistry.create(
        root,
        campaign_id=campaign_id,
        mode=actual_mode,
        protocol_sha256=protocol_hash,
        schedule=schedule,
        reuse_policy=actual_policy,
    )
    return verify_frozen_campaign(root, check_source=True, repo_root=repo)


def verify_frozen_campaign(
    campaign_root: str | Path,
    *,
    check_source: bool = True,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Verify protocol, manifest, schedule, and optionally live source hashes."""

    root = Path(campaign_root).resolve()
    try:
        manifest = json.loads(
            (root / "CAMPAIGN_MANIFEST.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FreezeError(f"cannot read campaign manifest: {error}") from error
    if not isinstance(manifest, dict):
        raise FreezeError("campaign manifest must contain an object")
    if manifest.get("manifest_sha256") != _manifest_payload_hash(manifest):
        raise FreezeError("campaign manifest hash mismatch")
    try:
        CampaignMode(manifest.get("mode"))
        ReusePolicy(manifest.get("reuse_policy"))
    except (TypeError, ValueError) as error:
        raise FreezeError("campaign manifest has an invalid mode or reuse policy") from error
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or not isinstance(manifest.get("campaign_id"), str)
        or _SAFE_ID.fullmatch(manifest["campaign_id"]) is None
        or _SHA256.fullmatch(str(manifest.get("protocol_sha256"))) is None
        or _SHA256.fullmatch(str(manifest.get("schedule_sha256"))) is None
        or isinstance(manifest.get("schedule_size"), bool)
        or not isinstance(manifest.get("schedule_size"), int)
        or manifest["schedule_size"] < 1
        or manifest.get("max_infrastructure_attempts")
        != MAX_INFRASTRUCTURE_ATTEMPTS
        or not isinstance(manifest.get("metadata"), dict)
        or not isinstance(manifest.get("provenance"), dict)
    ):
        raise FreezeError("campaign manifest schema is invalid")
    if manifest.get("protocol_path") != "PROTOCOL.md":
        raise FreezeError("campaign protocol_path must be PROTOCOL.md")
    if manifest.get("schedule_path") != "run_schedule.csv":
        raise FreezeError("campaign schedule_path must be run_schedule.csv")
    protocol = _safe_frozen_file(root, manifest["protocol_path"], "protocol_path")
    if sha256_file(protocol) != manifest.get("protocol_sha256"):
        raise FreezeError("frozen protocol hash mismatch")
    expected_line = f"{manifest['protocol_sha256']}  PROTOCOL.md\n"
    try:
        actual_line = (root / "PROTOCOL.sha256").read_text(encoding="ascii")
    except (OSError, UnicodeError) as error:
        raise FreezeError(f"cannot read PROTOCOL.sha256: {error}") from error
    if actual_line != expected_line:
        raise FreezeError("PROTOCOL.sha256 content mismatch")
    schedule_path = _safe_frozen_file(
        root, manifest["schedule_path"], "schedule_path"
    )
    try:
        schedule = read_schedule(schedule_path)
    except (OSError, UnicodeError, csv.Error, CampaignError, ValueError) as error:
        raise FreezeError(f"cannot validate frozen schedule: {error}") from error
    if schedule_sha256(schedule) != manifest.get("schedule_sha256"):
        raise FreezeError("frozen schedule hash mismatch")
    if len(schedule) != manifest.get("schedule_size"):
        raise FreezeError("frozen schedule size mismatch")
    if manifest.get("sequential_execution") is not True:
        raise FreezeError("frozen campaign must require sequential execution")
    if manifest.get("common_random_numbers") is not True:
        raise FreezeError("frozen campaign must require common random numbers")
    expected_approval_gate = manifest["mode"] == CampaignMode.LOCKED.value
    if manifest.get("locked_approval_required") is not expected_approval_gate:
        raise FreezeError("locked approval-gate metadata contradicts campaign mode")
    if manifest["reuse_policy"] != ReusePolicy.FRESH_ATTEMPTS_ONLY.value:
        raise FreezeError("unsupported frozen outcome-reuse policy")
    if _normalize_campaign_metadata(manifest["metadata"]) != manifest["metadata"]:
        raise FreezeError("campaign metadata is not in canonical frozen form")

    documentation_files: list[dict[str, Any]] | None = None
    if "documentation_snapshot" in manifest:
        documentation_files = _verify_documentation_snapshot(
            root, manifest["documentation_snapshot"]
        )

    provenance = manifest["provenance"]
    source_files = _validate_provenance_files(
        provenance.get("source_files"), "source_files"
    )
    frozen_inputs = _validate_provenance_files(
        provenance.get("frozen_inputs"), "frozen_inputs"
    )
    if provenance.get("source_manifest_sha256") != sha256_bytes(
        canonical_json_bytes(source_files)
    ):
        raise FreezeError("source provenance manifest digest mismatch")
    if provenance.get("frozen_inputs_sha256") != sha256_bytes(
        canonical_json_bytes(frozen_inputs)
    ):
        raise FreezeError("frozen-input provenance manifest digest mismatch")
    provenance_path = _safe_frozen_file(
        root, "environment/provenance.json", "provenance artifact"
    )
    try:
        direct_provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FreezeError(f"cannot parse provenance artifact: {error}") from error
    if direct_provenance != provenance:
        raise FreezeError("manifest provenance differs from environment/provenance.json")

    try:
        registry = CampaignRegistry(root)
        state = registry.snapshot()
    except RegistryError as error:
        raise FreezeError(f"frozen campaign registry is invalid or missing: {error}") from error
    registry_identity = (
        state["campaign_id"],
        state["mode"],
        state["protocol_sha256"],
        state["schedule_sha256"],
        state["schedule_size"],
        state["reuse_policy"],
    )
    manifest_identity = (
        manifest.get("campaign_id"),
        manifest.get("mode"),
        manifest.get("protocol_sha256"),
        manifest.get("schedule_sha256"),
        manifest.get("schedule_size"),
        manifest.get("reuse_policy"),
    )
    if registry_identity != manifest_identity:
        raise FreezeError("RUN_STATE identity differs from the frozen manifest")

    if check_source:
        repository = (
            Path(repo_root).resolve() if repo_root is not None else Path.cwd().resolve()
        )
        # Campaign roots normally live at <repo>/experiments/runs/<id>.  Find
        # the named repository ancestor without persisting an absolute path.
        label = provenance.get("repository_root_label")
        if not isinstance(label, str) or _SAFE_ID.fullmatch(label) is None:
            raise FreezeError("invalid repository_root_label in frozen provenance")
        if repo_root is None:
            for candidate in (root, *root.parents):
                if candidate.name == label and (candidate / ".git").exists():
                    repository = candidate
                    break
        if repository.name != label:
            raise FreezeError("repository root does not match frozen provenance label")
        if manifest["mode"] == CampaignMode.LOCKED.value:
            frozen_engine = _validate_locked_execution_engine(
                manifest["metadata"], repository
            )
            _validate_locked_runtime_metadata(
                manifest["metadata"], populate_hashes=False
            )
            if frozen_engine["source_path"] not in {
                item["path"] for item in source_files
            }:
                raise FreezeError(
                    "locked execution engine source is absent from source provenance"
                )
        frozen_commit = provenance.get("git_commit")
        current_commit = _git_text(repository, "rev-parse", "HEAD")
        if frozen_commit is not None and current_commit != frozen_commit:
            raise FreezeError("repository commit changed after freeze")
        for item in source_files:
            path = _safe_frozen_file(
                repository, item["path"], "provenance source path"
            )
            if path.stat().st_size != item["size_bytes"] or sha256_file(path) != item["sha256"]:
                raise FreezeError(f"source changed after freeze: {item['path']}")
        for item in frozen_inputs:
            path = _safe_frozen_file(
                repository, item["path"], "provenance frozen-input path"
            )
            if path.stat().st_size != item["size_bytes"] or sha256_file(path) != item["sha256"]:
                raise FreezeError(f"frozen input changed after freeze: {item['path']}")
        for item in documentation_files or ():
            path = _safe_frozen_file(
                repository, item["source_path"], "documentation source path"
            )
            if path.stat().st_size != item["size_bytes"] or sha256_file(
                path
            ) != item["sha256"]:
                raise FreezeError(
                    f"documentation source changed after freeze: {item['source_path']}"
                )
        raw_source_roots = provenance.get("source_roots")
        raw_frozen_roots = provenance.get("frozen_input_roots")
        if not isinstance(raw_source_roots, list) or not isinstance(
            raw_frozen_roots, list
        ):
            raise FreezeError("provenance source/input roots must be arrays")
        source_roots = [Path(path) for path in raw_source_roots]
        frozen_roots = [Path(path) for path in raw_frozen_roots]
        if not source_roots or _manifest_files(repository, source_roots) != source_files:
            raise FreezeError("source manifest membership changed after freeze")
        if _manifest_files(repository, frozen_roots) != frozen_inputs:
            raise FreezeError("frozen-input manifest membership changed after freeze")
    return manifest


__all__ = [
    "Approval",
    "ApprovalError",
    "AttemptLease",
    "CampaignError",
    "CampaignMode",
    "CampaignRegistry",
    "FreezeError",
    "MAX_INFRASTRUCTURE_ATTEMPTS",
    "RegistryError",
    "ReusePolicy",
    "RUN_INDEX_FIELDS",
    "RunStatus",
    "SCHEDULE_FIELDS",
    "ScheduleCell",
    "ScheduleEntry",
    "ScheduleError",
    "build_blocked_schedule",
    "canonical_json_bytes",
    "collect_provenance",
    "default_protocol_schedule",
    "freeze_protocol",
    "generate_schedule",
    "parse_locked_approval",
    "read_schedule",
    "require_locked_approval",
    "run_index_csv_bytes",
    "schedule_sha256",
    "sha256_file",
    "validate_schedule",
    "verify_frozen_campaign",
    "write_schedule",
]
