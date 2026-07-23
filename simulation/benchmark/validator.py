from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from .catalog import FAMILY_DEFINITIONS, build_catalog, build_control_catalog
from .generator import SCHEMA_VERSION, _scenario_manifest
from .protocols import build_memory_protocols, memory_protocol_fixtures


OPAQUE_SCENARIO_ID = re.compile(r"^ep-[0-9a-f]{24}$")
REQUIRED_MANIFEST_FIELDS = {
    "schema_version",
    "scenario_id",
    "scene",
    "target",
    "expected_outcome",
    "frames",
    "ground_truth",
    "benchmark_expectations",
    "generation",
    "frame_sha256",
}


@dataclass(frozen=True, slots=True)
class ValidationReport:
    root: Path
    valid: bool
    scenario_count: int
    errors: tuple[str, ...]
    warnings: tuple[str, ...] = ()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("root value is not an object")
    return value


def _frame_names(value: Any) -> set[str]:
    if isinstance(value, dict):
        values = value.values()
    elif isinstance(value, list):
        values = value
    else:
        return set()
    return {str(item) for item in values}


def validate_benchmark(root: str | Path) -> ValidationReport:
    root = Path(root).expanduser().resolve()
    errors: list[str] = []
    warnings: list[str] = []
    if not root.is_dir():
        return ValidationReport(
            root=root,
            valid=False,
            scenario_count=0,
            errors=(f"Benchmark root does not exist: {root}",),
        )

    manifest_paths = sorted(root.rglob("manifest.json"))
    if not manifest_paths:
        return ValidationReport(
            root=root,
            valid=False,
            scenario_count=0,
            errors=("No episode manifests were found.",),
        )

    seen_ids: set[str] = set()
    initial_digests: dict[tuple[str, str, int, str], set[str]] = defaultdict(set)
    final_digests: dict[
        tuple[str, str, int, str, str | None], dict[str, str]
    ] = defaultdict(dict)
    manifests: dict[str, dict[str, Any]] = {}
    canonical_cache: dict[
        tuple[str, int, str | None], dict[tuple[str, str, str], Any]
    ] = {}

    for manifest_path in manifest_paths:
        episode_directory = manifest_path.parent
        label = episode_directory.relative_to(root).as_posix()
        try:
            manifest = _load_json(manifest_path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            errors.append(f"{label}: invalid manifest: {error}")
            continue

        missing = sorted(REQUIRED_MANIFEST_FIELDS - manifest.keys())
        if missing:
            errors.append(f"{label}: missing manifest fields: {', '.join(missing)}")
            continue
        if manifest.get("schema_version") != SCHEMA_VERSION:
            errors.append(f"{label}: unsupported schema_version")

        scenario_id = str(manifest.get("scenario_id", ""))
        if scenario_id != episode_directory.name:
            errors.append(f"{label}: scenario_id does not match its directory")
        if not OPAQUE_SCENARIO_ID.fullmatch(scenario_id):
            errors.append(f"{label}: scenario_id is not opaque/canonical")
        if scenario_id in seen_ids:
            errors.append(f"{label}: duplicate scenario_id {scenario_id}")
        seen_ids.add(scenario_id)
        manifests[scenario_id] = manifest

        names = {path.name for path in episode_directory.iterdir()}
        if names != {"1.png", "2.png", "manifest.json"}:
            errors.append(
                f"{label}: episode packet must contain exactly 1.png, 2.png, manifest.json"
            )
        if _frame_names(manifest.get("frames")) != {"1.png", "2.png"}:
            errors.append(f"{label}: frames must reference exactly 1.png and 2.png")

        for frame_name in ("1.png", "2.png"):
            frame_path = episode_directory / frame_name
            if not frame_path.is_file():
                errors.append(f"{label}: missing {frame_name}")
                continue
            try:
                with Image.open(frame_path) as image:
                    image.verify()
                    if image.format != "PNG":
                        errors.append(f"{label}: {frame_name} is not PNG")
            except (OSError, UnidentifiedImageError) as error:
                errors.append(f"{label}: invalid {frame_name}: {error}")
        frame_hashes = manifest.get("frame_sha256")
        if not isinstance(frame_hashes, dict) or set(frame_hashes) != {
            "1.png",
            "2.png",
        }:
            errors.append(f"{label}: frame_sha256 must cover both numbered frames")
        else:
            for frame_name in ("1.png", "2.png"):
                frame_path = episode_directory / frame_name
                if frame_path.is_file():
                    actual_digest = hashlib.sha256(frame_path.read_bytes()).hexdigest()
                    if frame_hashes.get(frame_name) != actual_digest:
                        errors.append(
                            f"{label}: {frame_name} does not match its recorded digest"
                        )

        scene = manifest.get("scene")
        target = manifest.get("target")
        ground_truth = manifest.get("ground_truth")
        if not isinstance(scene, dict):
            errors.append(f"{label}: scene must be an object")
            continue
        if not isinstance(target, dict):
            errors.append(f"{label}: target must be an object")
            continue
        if not isinstance(ground_truth, dict):
            errors.append(f"{label}: ground_truth must be an object")
            continue

        family = str(scene.get("family", ""))
        control_kind = (
            str(scene["control_kind"])
            if scene.get("control_kind") is not None
            else None
        )
        definition = FAMILY_DEFINITIONS.get(family)
        if definition is None:
            errors.append(f"{label}: unknown family {family!r}")
            continue
        variant = str(scene.get("variant", ""))
        target_id = str(target.get("target_id", ""))
        outcome = str(manifest.get("expected_outcome", ""))
        if control_kind not in {None, "already_satisfied"}:
            errors.append(f"{label}: unknown control kind {control_kind!r}")
        if control_kind is None and variant not in definition.scene_variants:
            errors.append(f"{label}: unknown scene variant {variant!r}")
        if (
            control_kind == "already_satisfied"
            and variant != "control_already_satisfied"
        ):
            errors.append(f"{label}: invalid already-satisfied scene variant")
        if target_id not in definition.targets:
            errors.append(f"{label}: unknown target {target_id!r}")
        if outcome not in definition.outcomes:
            errors.append(f"{label}: unknown outcome {outcome!r}")

        generation = manifest.get("generation")
        backend = (
            str(generation.get("backend", ""))
            if isinstance(generation, dict)
            else ""
        )
        if backend not in {"synthetic", "mujoco"}:
            errors.append(f"{label}: unsupported or missing generation backend")
        try:
            scene_seed = int(scene["seed"])
        except (KeyError, TypeError, ValueError):
            scene_seed = 0
        cache_key = (family, scene_seed, control_kind)
        if definition is not None and backend in {"synthetic", "mujoco"}:
            candidates = canonical_cache.get(cache_key)
            if candidates is None:
                candidate_catalog = (
                    build_control_catalog(
                        families=[family],
                        seeds=[scene_seed],
                    )
                    if control_kind == "already_satisfied"
                    else build_catalog(
                        families=[family],
                        seeds=[scene_seed],
                    )
                )
                candidates = {
                    (
                        scenario.scene_variant,
                        scenario.target_id,
                        scenario.outcome,
                    ): scenario
                    for scenario in candidate_catalog
                }
                canonical_cache[cache_key] = candidates
            canonical_spec = candidates.get((variant, target_id, outcome))
            if canonical_spec is None:
                errors.append(
                    f"{label}: manifest does not identify a declared scenario cell"
                )
            else:
                canonical = _scenario_manifest(canonical_spec, backend=backend)
                if scenario_id != canonical_spec.scenario_id:
                    errors.append(
                        f"{label}: scenario_id does not match its semantic scenario cell"
                    )
                for field in (
                    "scene",
                    "target",
                    "expected_outcome",
                    "frames",
                    "ground_truth",
                    "benchmark_expectations",
                    "generation",
                ):
                    if manifest.get(field) != canonical.get(field):
                        errors.append(
                            f"{label}: {field} differs from the deterministic catalog"
                        )

        goals = {str(item) for item in target.get("goal_predicates", [])}
        actual = {str(item) for item in ground_truth.get("actual_predicates", [])}
        if not goals:
            errors.append(f"{label}: target has no goal predicates")
        if outcome in {"success", "unknown"} and not goals <= actual:
            errors.append(f"{label}: {outcome} endpoint does not satisfy its goal")
        if outcome in {"wrong_complete", "partial", "near_miss"} and goals <= actual:
            errors.append(f"{label}: failed endpoint unexpectedly satisfies its goal")
        if outcome == "partial" and not (goals & actual):
            errors.append(f"{label}: partial endpoint satisfies no requested predicate")

        try:
            group_key = (
                family,
                variant,
                int(scene["seed"]),
                target_id if control_kind else "",
            )
            initial_path = episode_directory / "1.png"
            if initial_path.is_file():
                initial_digests[group_key].add(
                    hashlib.sha256(initial_path.read_bytes()).hexdigest()
                )
            final_path = episode_directory / "2.png"
            if final_path.is_file():
                final_digests[
                    (
                        family,
                        variant,
                        int(scene["seed"]),
                        target_id,
                        control_kind,
                    )
                ][outcome] = hashlib.sha256(final_path.read_bytes()).hexdigest()
        except (KeyError, TypeError, ValueError):
            errors.append(f"{label}: scene seed must be an integer")

    for group_key, digests in initial_digests.items():
        if len(digests) != 1:
            errors.append(
                f"{group_key}: initial pixels differ across target/outcome siblings"
            )
    for group_key, by_outcome in final_digests.items():
        if (
            "success" in by_outcome
            and "near_miss" in by_outcome
            and by_outcome["success"] == by_outcome["near_miss"]
        ):
            errors.append(
                f"{group_key}: success and near-miss final observations are "
                "pixel-identical"
            )

    index_path = root / "index.json"
    declared_scenarios: list[Any] | None = None
    if index_path.is_file():
        try:
            index = _load_json(index_path)
            indexed = {str(value) for value in index.get("scenario_ids", [])}
            if indexed != seen_ids:
                errors.append("index.json scenario_ids do not match episode manifests")
            if index.get("scenario_count") != len(seen_ids):
                errors.append("index.json scenario_count is incorrect")
            indexed_backend = str(index.get("backend", ""))
            if indexed_backend not in {"synthetic", "mujoco"}:
                errors.append("index.json backend is invalid")
            for scenario_id, manifest in manifests.items():
                generation = manifest.get("generation")
                manifest_backend = (
                    str(generation.get("backend", ""))
                    if isinstance(generation, dict)
                    else ""
                )
                if manifest_backend != indexed_backend:
                    errors.append(
                        f"{scenario_id}: generation backend differs from index.json"
                    )
            indexed_families = index.get("families")
            indexed_seeds = index.get("seeds")
            if not isinstance(indexed_families, list) or not isinstance(
                indexed_seeds, list
            ):
                errors.append("index.json must declare generated families and seeds")
            else:
                try:
                    declared_families = [
                        str(value) for value in indexed_families
                    ]
                    declared_seeds = [int(value) for value in indexed_seeds]
                    declared_scenarios = list(
                        build_catalog(
                            families=declared_families,
                            seeds=declared_seeds,
                        )
                    )
                    declared_controls = (
                        list(
                            build_control_catalog(
                                families=declared_families,
                                seeds=declared_seeds,
                            )
                        )
                        if index.get("include_controls") is True
                        else []
                    )
                    declared_scenarios.extend(declared_controls)
                    expected_ids = {
                        scenario.scenario_id
                        for scenario in declared_scenarios
                    }
                    if indexed != expected_ids:
                        errors.append(
                            "index.json does not contain the complete declared "
                            "family/seed scenario matrix"
                        )
                    expected_control_ids = (
                        {
                            scenario.scenario_id
                            for scenario in declared_controls
                        }
                        if index.get("include_controls") is True
                        else set()
                    )
                    if {
                        str(value) for value in index.get("control_ids", [])
                    } != expected_control_ids:
                        errors.append("index.json control_ids are incorrect")
                except (TypeError, ValueError) as error:
                    errors.append(
                        f"index.json family/seed declaration is invalid: {error}"
                    )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            errors.append(f"index.json is invalid: {error}")
    else:
        warnings.append("index.json is missing; matrix completeness was not checked")

    protocols_path = root / "protocols.json"
    if protocols_path.is_file():
        try:
            protocols = _load_json(protocols_path)
            if declared_scenarios is None:
                errors.append(
                    "protocols.json cannot be bound because index.json has no "
                    "valid family/seed declaration"
                )
            else:
                expected_protocols = {
                    "schema_version": "robopref.memory-protocols.v1",
                    "fixtures": memory_protocol_fixtures(),
                    "protocols": build_memory_protocols(
                        declared_scenarios
                    ),
                }
                if protocols != expected_protocols:
                    errors.append(
                        "protocols.json differs from the declared scenario "
                        "matrix or canonical fixture bundle"
                    )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            errors.append(f"protocols.json is invalid: {error}")

    return ValidationReport(
        root=root,
        valid=not errors,
        scenario_count=len(manifests),
        errors=tuple(errors),
        warnings=tuple(warnings),
    )
