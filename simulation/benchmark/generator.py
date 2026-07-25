from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .canonical import catalog_values_equal, stable_float
from .catalog import FAMILY_DEFINITIONS, build_catalog, build_control_catalog
from .render import render_pair


SCHEMA_VERSION = "robopref.counterfactual.v1"
OPAQUE_EPISODE_PREFIX = "ep-"
OPAQUE_EPISODE_DIRECTORY = re.compile(r"^ep-[0-9a-f]{24}$")


@dataclass(frozen=True, slots=True)
class GenerationReport:
    output_root: Path
    backend: str
    scenario_count: int
    generated: int
    skipped: int
    scenario_ids: tuple[str, ...]


def _plain(value: Any) -> Any:
    if type(value) is float:
        return stable_float(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _plain(to_dict())
    if is_dataclass(value):
        return {key: _plain(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _canonical_predicate(value: Any) -> str:
    canonical = getattr(value, "canonical", None)
    if callable(canonical):
        return str(canonical())
    if canonical is not None:
        return str(canonical)
    return str(value)


def _predicate_strings(values: Iterable[Any]) -> list[str]:
    return [_canonical_predicate(value) for value in values]


def _scenario_manifest(spec: Any, *, backend: str) -> dict[str, Any]:
    goals = _predicate_strings(spec.goal_predicates)
    actual = _predicate_strings(spec.final_predicates)
    actual_set = set(actual)
    outcome = str(spec.outcome)
    initially_satisfied = (
        getattr(spec, "control_kind", None) == "already_satisfied"
    )

    if outcome == "success":
        validator_outcome: str | None = "SUCCESS"
        task_complete: bool | None = True
        validation_called = True
        recovery = "NONE"
    elif outcome == "unknown":
        validator_outcome = "UNKNOWN"
        task_complete = False
        validation_called = True
        recovery = "REOBSERVE"
    elif outcome == "unsafe":
        validator_outcome = None
        task_complete = None
        validation_called = False
        recovery = "ABORT_SAFETY"
    else:
        validator_outcome = "PARTIAL" if outcome == "partial" else "FAILURE"
        task_complete = False
        validation_called = True
        recovery = "REPLAN"

    observability = {
        "status": str(getattr(spec, "observation_status", "observable")),
        "occluded_object_ids": list(getattr(spec, "occluded_object_ids", ())),
    }
    safety_events = (
        ["safety_constraint_violated"] if outcome == "unsafe" else []
    )
    initial_objects = _plain(spec.initial_objects)
    final_objects = _plain(spec.final_objects)

    return {
        "schema_version": SCHEMA_VERSION,
        "scenario_id": spec.scenario_id,
        "scene": {
            "family": spec.family,
            "variant": spec.scene_variant,
            "seed": spec.seed,
            "counterfactual_group_id": spec.counterfactual_group_id,
            "control_kind": getattr(spec, "control_kind", None),
            "initial_objects": initial_objects,
        },
        "target": {
            "target_id": spec.target_id,
            "instruction": spec.instruction,
            "description": str(getattr(spec, "target_description", "")),
            "goal_predicates": goals,
        },
        "expected_outcome": outcome,
        "frames": {
            "initial": "1.png",
            "final": "2.png",
        },
        "ground_truth": {
            "initial_objects": initial_objects,
            "final_objects": final_objects,
            "actual_predicates": actual,
            "predicate_measurements": _plain(spec.final_predicates),
            "goal_checks": [
                {"predicate": predicate, "satisfied": predicate in actual_set}
                for predicate in goals
            ],
            "task_goal_satisfied": set(goals) <= actual_set,
            "physical_success": bool(
                getattr(spec, "expected_success", set(goals) <= actual_set)
            ),
            "failure_mode": getattr(spec, "failure_mode", None),
            "outcome_description": str(
                getattr(spec, "outcome_description", "")
            ),
            "evidence": dict(getattr(spec, "evidence", ())),
            "evidence_provenance": {
                "spatial_predicates": "derived_from_terminal_object_state",
                "safety_and_contact_values": "scripted_counterfactual_fixture",
                "measured_physics_telemetry": False,
            },
            "observability": observability,
            "safety_events": safety_events,
        },
        "benchmark_expectations": {
            "hri": {
                "response_mode": "REPORT",
                "accepted_response_modes": (
                    ["REPORT", "MEMORY_CONFIRM"]
                    if outcome == "success"
                    else ["REPORT"]
                ),
                "terminal_outcome": (
                    "SUCCESS"
                    if outcome == "success"
                    else "PARTIAL"
                    if outcome == "partial"
                    else "UNKNOWN"
                    if outcome == "unknown"
                    else "ABORTED_SAFETY"
                    if outcome == "unsafe"
                    else "FAILED"
                ),
                "preference_store_delta_without_consent": 0,
            },
            "planner": {
                "must_preserve_confirmed_intent": True,
                "expected_status": (
                    "ALREADY_SATISFIED" if initially_satisfied else "READY"
                ),
                "goal_predicates": goals,
            },
            "execution": {
                "expected_status": (
                    "OBSERVATION_ONLY"
                    if initially_satisfied
                    else "UNSAFE"
                    if outcome == "unsafe"
                    else "OBSERVED_RECORDED_ATTEMPT"
                ),
                **(
                    {"expected_vla_dispatches": 0}
                    if initially_satisfied
                    else {}
                ),
            },
            "validator": {
                "called": validation_called,
                "outcome": validator_outcome,
                "task_complete": task_complete,
            },
            "recovery": recovery,
            "history": {
                "expected_terminal_record_delta": 1,
            },
            "preference": {
                "expected_record_delta_without_explicit_consent": 0,
            },
        },
        "generation": {
            "backend": backend,
            "transition_kind": "privileged_endpoint_snapshot",
            "robot_trajectory_executed": False,
            "physics_telemetry_measured": False,
            "renderer_truth_is_not_model_context": True,
            "archived_robot_asset_reference": (
                "_old_1/simulation/assets/robots/arx_l5/scene.xml"
                if backend == "mujoco"
                else None
            ),
        },
    }


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_if_changed(path: Path, content: bytes) -> None:
    if path.is_file() and path.read_bytes() == content:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _directory_is_complete(
    directory: Path, expected_manifest_without_hashes: dict[str, Any]
) -> bool:
    expected_names = {"1.png", "2.png", "manifest.json"}
    if not directory.is_dir():
        return False
    if {path.name for path in directory.iterdir()} != expected_names:
        return False
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(manifest, dict):
        return False
    frame_hashes = manifest.pop("frame_sha256", None)
    if not catalog_values_equal(
        manifest, expected_manifest_without_hashes
    ) or not isinstance(
        frame_hashes, dict
    ):
        return False
    for frame_name in ("1.png", "2.png"):
        frame_path = directory / frame_name
        if not frame_path.is_file():
            return False
        digest = hashlib.sha256(frame_path.read_bytes()).hexdigest()
        if frame_hashes.get(frame_name) != digest:
            return False
    return set(frame_hashes) == {"1.png", "2.png"}


def _assert_safe_episode_directory(directory: Path, episodes_root: Path) -> None:
    resolved_root = episodes_root.resolve()
    resolved = directory.resolve()
    if resolved.parent != resolved_root:
        raise ValueError(f"Episode directory escapes output root: {directory}")
    if not directory.name.startswith(OPAQUE_EPISODE_PREFIX):
        raise ValueError(f"Refusing to replace non-episode directory: {directory}")


def _render_scenario(spec: Any, directory: Path, *, backend: str) -> None:
    if backend == "synthetic":
        render_pair(spec, directory / "1.png", directory / "2.png")
        return
    if backend == "mujoco":
        from .mujoco_render import render_pair_mujoco

        render_pair_mujoco(spec, directory / "1.png", directory / "2.png")
        return
    raise ValueError(f"Unsupported benchmark backend: {backend!r}")


def _normalise_families(families: Sequence[str] | None) -> tuple[str, ...]:
    selected = tuple(FAMILY_DEFINITIONS) if families is None else tuple(families)
    if not selected:
        raise ValueError("At least one benchmark family is required.")
    unknown = sorted(set(selected) - set(FAMILY_DEFINITIONS))
    if unknown:
        raise ValueError(f"Unknown benchmark families: {', '.join(unknown)}")
    if len(set(selected)) != len(selected):
        raise ValueError("Benchmark family names must be unique.")
    return selected


def _normalise_seeds(seeds: Sequence[int]) -> tuple[int, ...]:
    raw = tuple(seeds)
    if any(not isinstance(seed, int) or isinstance(seed, bool) for seed in raw):
        raise TypeError("Scene seeds must contain only integers (not booleans).")
    selected = raw
    if not selected:
        raise ValueError("At least one scene seed is required.")
    if len(set(selected)) != len(selected):
        raise ValueError("Scene seeds must be unique.")
    if any(seed < 0 for seed in selected):
        raise ValueError("Scene seeds must be non-negative.")
    return selected


def _prepare_existing_root(
    output_root: Path,
    episodes_root: Path,
    *,
    desired_ids: set[str],
    backend: str,
    overwrite: bool,
) -> None:
    """Reject incompatible roots, or prune only known generated packets."""

    index_path = output_root / "index.json"
    existing_index: dict[str, Any] | None = None
    if index_path.is_file():
        try:
            value = json.loads(index_path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                existing_index = value
        except (OSError, json.JSONDecodeError):
            existing_index = None
        if existing_index is None and not overwrite:
            raise ValueError(
                f"Existing benchmark index is invalid: {index_path}. "
                "Use a new output root or --overwrite."
            )

    existing_episode_directories = {
        path.name: path
        for path in episodes_root.iterdir()
        if path.is_dir() and OPAQUE_EPISODE_DIRECTORY.fullmatch(path.name)
    }
    existing_ids = set(existing_episode_directories)
    indexed_ids = (
        {str(value) for value in existing_index.get("scenario_ids", [])}
        if existing_index is not None
        else set()
    )
    indexed_backend = (
        str(existing_index.get("backend", ""))
        if existing_index is not None
        else ""
    )
    incompatible = bool(existing_ids or existing_index) and (
        existing_ids != desired_ids
        or indexed_ids != desired_ids
        or indexed_backend != backend
    )
    if incompatible and not overwrite:
        raise ValueError(
            "The output root contains a different benchmark matrix or backend. "
            "Use a new output root, or pass --overwrite to replace its generated "
            f"episode packets: {output_root}"
        )
    if overwrite:
        for scenario_id in sorted(existing_ids - desired_ids):
            directory = existing_episode_directories[scenario_id]
            _assert_safe_episode_directory(directory, episodes_root)
            shutil.rmtree(directory)


def generate_benchmark(
    output_root: str | Path,
    *,
    families: Sequence[str] | None = None,
    seeds: Sequence[int] = (1,),
    backend: str = "synthetic",
    overwrite: bool = False,
    include_controls: bool = False,
) -> GenerationReport:
    """Generate a deterministic counterfactual endpoint benchmark.

    Every sibling in a ``(family, scene_variant, seed)`` group receives the exact
    same initial image.  Only the target and outcome change.  Simulator/oracle
    fields live in ``manifest.json`` and must never be included in model prompts.
    """

    output_root = Path(output_root).expanduser().resolve()
    selected_families = _normalise_families(families)
    selected_seeds = _normalise_seeds(seeds)
    backend = backend.strip().lower()
    if backend not in {"synthetic", "mujoco"}:
        raise ValueError("backend must be 'synthetic' or 'mujoco'.")

    core_catalog = build_catalog(
        families=selected_families,
        seeds=selected_seeds,
    )
    control_catalog = (
        build_control_catalog(
            families=selected_families,
            seeds=selected_seeds,
        )
        if include_controls
        else ()
    )
    catalog = (*core_catalog, *control_catalog)
    episodes_root = output_root / "episodes"
    episodes_root.mkdir(parents=True, exist_ok=True)
    _prepare_existing_root(
        output_root,
        episodes_root,
        desired_ids={spec.scenario_id for spec in catalog},
        backend=backend,
        overwrite=overwrite,
    )

    generated = 0
    skipped = 0
    initial_images: dict[str, bytes] = {}
    for spec in catalog:
        initial_group_key = (
            f"{spec.family}|{spec.scene_variant}|{spec.seed}|{spec.target_id}"
            if getattr(spec, "control_kind", None)
            else f"{spec.family}|{spec.scene_variant}|{spec.seed}"
        )
        episode_directory = episodes_root / spec.scenario_id
        manifest = _scenario_manifest(spec, backend=backend)
        if not overwrite and _directory_is_complete(episode_directory, manifest):
            initial = (episode_directory / "1.png").read_bytes()
            previous = initial_images.setdefault(initial_group_key, initial)
            if initial != previous:
                raise RuntimeError(
                    "Existing counterfactual siblings have different initial pixels: "
                    f"{initial_group_key}"
                )
            skipped += 1
            continue

        if episode_directory.exists() and not overwrite:
            raise FileExistsError(
                "Existing episode does not match the requested deterministic packet; "
                f"validate it or regenerate with overwrite=True: {episode_directory}"
            )
        if episode_directory.exists():
            _assert_safe_episode_directory(episode_directory, episodes_root)
            shutil.rmtree(episode_directory)

        temporary_directory = Path(
            tempfile.mkdtemp(prefix=f".{spec.scenario_id}.", dir=episodes_root)
        )
        try:
            _render_scenario(spec, temporary_directory, backend=backend)
            initial_path = temporary_directory / "1.png"
            initial = initial_path.read_bytes()
            canonical_initial = initial_images.setdefault(
                initial_group_key, initial
            )
            if initial != canonical_initial:
                initial_path.write_bytes(canonical_initial)
            manifest["frame_sha256"] = {
                frame_name: hashlib.sha256(
                    (temporary_directory / frame_name).read_bytes()
                ).hexdigest()
                for frame_name in ("1.png", "2.png")
            }
            manifest_bytes = _json_bytes(manifest)
            (temporary_directory / "manifest.json").write_bytes(manifest_bytes)
            if {path.name for path in temporary_directory.iterdir()} != {
                "1.png",
                "2.png",
                "manifest.json",
            }:
                raise RuntimeError(
                    f"Renderer produced an invalid episode packet: {temporary_directory}"
                )
            os.replace(temporary_directory, episode_directory)
        except Exception:
            shutil.rmtree(temporary_directory, ignore_errors=True)
            raise
        generated += 1

    index = {
        "schema_version": SCHEMA_VERSION,
        "backend": backend,
        "families": list(selected_families),
        "seeds": list(selected_seeds),
        "scenario_count": len(catalog),
        "scenario_ids": [spec.scenario_id for spec in catalog],
        "include_controls": include_controls,
        "control_ids": [
            spec.scenario_id
            for spec in control_catalog
        ],
        "episode_root": "episodes",
        "oracle_warning": (
            "manifest.json contains benchmark truth. Pass only numbered images and "
            "the human query to PrefMem."
        ),
        "catalog_digest": hashlib.sha256(
            "\n".join(spec.scenario_id for spec in catalog).encode("ascii")
        ).hexdigest(),
    }
    _write_if_changed(output_root / "index.json", _json_bytes(index))

    return GenerationReport(
        output_root=output_root,
        backend=backend,
        scenario_count=len(catalog),
        generated=generated,
        skipped=skipped,
        scenario_ids=tuple(spec.scenario_id for spec in catalog),
    )
