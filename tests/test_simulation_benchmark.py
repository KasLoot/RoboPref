from __future__ import annotations

import hashlib
import itertools
import json
import os
import re
import shutil
import tempfile
import unittest
from collections import defaultdict
from dataclasses import asdict, is_dataclass, replace
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

from PIL import Image

from agents.configs import PrefMemConfig
from agents.hri import HRIOrchestrator
from dataset.benchmark import BenchmarkEpisode
from dataset.episode import DatasetEpisode
from simulation.benchmark import (
    BenchmarkEpisodeExecutor,
    FAMILY_DEFINITIONS,
    build_catalog,
    build_control_catalog,
    generate_benchmark,
    score_agent_result,
    validate_benchmark,
)
from simulation.benchmark.catalog import _yaw_quaternion
from simulation.benchmark.evaluators import evaluate_goal_predicates
from tests.fakes import RecordingMemoryAgent, ScriptedJsonModel


FAMILIES = ("block_stack", "category_sort", "place_setting")
COMMON_OUTCOMES = ("success", "wrong_complete", "partial", "near_miss")
OPAQUE_ID = re.compile(
    r"^(?:(?:ep|scenario)-|scn_)?[0-9a-f]{16,64}$"
)
REPO_ROOT = Path(__file__).resolve().parents[1]


def _plain(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _episode_directories(root: Path) -> list[Path]:
    return sorted(path.parent for path in root.rglob("manifest.json"))


def _tree_digest(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _manifest(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise AssertionError(f"{path} did not contain a JSON object")
    return value


def _satisfied_goal_keys(scenario: Any) -> set[tuple[str, tuple[str, ...]]]:
    final_by_key = {
        predicate.key: predicate.value for predicate in scenario.final_predicates
    }
    return {
        predicate.key
        for predicate in scenario.goal_predicates
        if final_by_key.get(predicate.key) == predicate.value
    }


def _frame_names(frames: Any) -> set[str]:
    if isinstance(frames, dict):
        values = frames.values()
    elif isinstance(frames, list):
        values = frames
    else:
        raise AssertionError("manifest.frames must be an object or list")
    return {str(value) for value in values}


class BenchmarkCatalogTests(unittest.TestCase):
    def test_yaw_quaternion_is_stable_across_one_ulp_libm_difference(self) -> None:
        with (
            patch(
                "simulation.benchmark.catalog.math.sin",
                return_value=0.7071067811865475,
            ),
            patch(
                "simulation.benchmark.catalog.math.cos",
                return_value=0.7071067811865475,
            ),
        ):
            linux_result = _yaw_quaternion(1.0)
        with (
            patch(
                "simulation.benchmark.catalog.math.sin",
                return_value=0.7071067811865476,
            ),
            patch(
                "simulation.benchmark.catalog.math.cos",
                return_value=0.7071067811865476,
            ),
        ):
            windows_result = _yaw_quaternion(1.0)

        self.assertEqual(linux_result, windows_result)
        self.assertEqual(
            linux_result,
            (0.0, 0.0, 0.707106781187, 0.707106781187),
        )

    def test_declared_family_domains_form_the_complete_scenario_matrix(self) -> None:
        seeds = (3, 17)
        catalog = build_catalog(families=list(FAMILIES), seeds=list(seeds))

        self.assertEqual(
            len({scenario.scenario_id for scenario in catalog}),
            len(catalog),
            "scenario IDs must be unique",
        )
        for family in FAMILIES:
            definition = FAMILY_DEFINITIONS[family]
            self.assertTrue(definition.scene_variants)
            self.assertTrue(definition.targets)
            self.assertTrue(definition.outcomes)
            self.assertTrue(set(COMMON_OUTCOMES) <= set(definition.outcomes))

            expected = set(
                itertools.product(
                    definition.scene_variants,
                    definition.targets,
                    definition.outcomes,
                    seeds,
                )
            )
            actual = {
                (
                    scenario.scene_variant,
                    scenario.target_id,
                    scenario.outcome,
                    scenario.seed,
                )
                for scenario in catalog
                if scenario.family == family
            }
            self.assertEqual(actual, expected, f"incomplete matrix for {family}")
            self.assertEqual(
                sum(scenario.family == family for scenario in catalog),
                len(expected),
                f"duplicate matrix cells for {family}",
            )

    def test_catalog_is_deterministic_and_scenario_ids_are_opaque(self) -> None:
        arguments = {"families": ["block_stack"], "seeds": [1, 9]}
        first = build_catalog(**arguments)
        second = build_catalog(**arguments)

        self.assertEqual([_plain(item) for item in first], [_plain(item) for item in second])
        for scenario in first:
            self.assertRegex(scenario.scenario_id, OPAQUE_ID)
            semantic_tokens = (
                scenario.family,
                scenario.scene_variant,
                scenario.target_id,
                scenario.outcome,
            )
            identifier = scenario.scenario_id.casefold()
            for token in semantic_tokens:
                normalized = re.sub(r"[^a-z0-9]+", "", str(token).casefold())
                if len(normalized) >= 4:
                    self.assertNotIn(
                        normalized,
                        re.sub(r"[^a-z0-9]+", "", identifier),
                        "opaque ID contains a semantic benchmark label",
                    )

    def test_outcome_predicates_are_conditioned_on_each_target(self) -> None:
        catalog = build_catalog(families=list(FAMILIES), seeds=[23])
        groups: dict[tuple[str, str, int], list[Any]] = defaultdict(list)
        for scenario in catalog:
            groups[(scenario.family, scenario.scene_variant, scenario.seed)].append(scenario)

        for group_key, scenarios in groups.items():
            success_by_target = {
                scenario.target_id: scenario
                for scenario in scenarios
                if scenario.outcome == "success"
            }
            self.assertEqual(
                set(success_by_target),
                set(FAMILY_DEFINITIONS[group_key[0]].targets),
            )

            success_final_states: set[tuple[Any, ...]] = set()
            for target_id, success in success_by_target.items():
                goals = {predicate.key for predicate in success.goal_predicates}
                satisfied = _satisfied_goal_keys(success)
                self.assertTrue(goals, f"{group_key}/{target_id} has no goal predicates")
                self.assertEqual(
                    satisfied,
                    goals,
                    f"success does not satisfy its target for {group_key}/{target_id}",
                )
                success_final_states.add(tuple(success.final_objects))
            self.assertEqual(
                len(success_final_states),
                len(success_by_target),
                f"success rendering ignores target in {group_key}",
            )

            for scenario in scenarios:
                goals = {predicate.key for predicate in scenario.goal_predicates}
                satisfied = _satisfied_goal_keys(scenario)
                self.assertEqual(scenario.goals_satisfied, scenario.expected_success)
                if scenario.outcome == "success":
                    self.assertTrue(scenario.expected_success)
                    self.assertEqual(scenario.expected_validator_label, "SUCCESS")
                    continue
                if scenario.outcome == "partial":
                    self.assertTrue(
                        0 < len(satisfied) < len(goals),
                        f"partial must satisfy a strict non-empty goal subset: {scenario}",
                    )
                elif scenario.outcome == "wrong_complete":
                    self.assertNotEqual(
                        satisfied,
                        goals,
                        f"wrong_complete satisfies requested target: {scenario}",
                    )
                    alternate_targets = [
                        other
                        for target_id, other in success_by_target.items()
                        if target_id != scenario.target_id
                    ]
                    self.assertTrue(
                        any(
                            scenario.final_objects == other.final_objects
                            for other in alternate_targets
                        ),
                        f"wrong_complete is not a complete alternate target: {scenario}",
                    )
                elif scenario.outcome == "near_miss":
                    self.assertNotEqual(
                        satisfied,
                        goals,
                        f"near_miss satisfies requested target: {scenario}",
                    )
                    self.assertFalse(
                        any(
                            scenario.final_objects == other.final_objects
                            for other in success_by_target.values()
                        ),
                        f"near_miss accidentally satisfies a declared target: {scenario}",
                    )
                elif scenario.outcome == "unknown":
                    success = success_by_target[scenario.target_id]
                    self.assertEqual(scenario.final_objects, success.final_objects)
                    self.assertTrue(scenario.expected_success)
                    self.assertEqual(scenario.expected_validator_label, "UNKNOWN")
                    self.assertEqual(scenario.observation_status, "occluded")
                elif scenario.outcome == "unsafe":
                    success = success_by_target[scenario.target_id]
                    self.assertEqual(
                        scenario.final_objects,
                        success.final_objects,
                        "unsafe must be distinguishable by execution evidence, not endpoint geometry",
                    )
                    self.assertFalse(scenario.expected_success)
                    unsafe_goals = [
                        predicate
                        for predicate in scenario.goal_predicates
                        if predicate.name == "safe_execution"
                    ]
                    self.assertEqual(len(unsafe_goals), 1)
                    self.assertFalse(unsafe_goals[0].observable)

    def test_instruction_does_not_reveal_expected_outcome(self) -> None:
        for scenario in build_catalog(families=list(FAMILIES), seeds=[31]):
            instruction = scenario.instruction.casefold().replace("_", " ")
            self.assertNotIn(scenario.scenario_id.casefold(), instruction)
            for label in FAMILY_DEFINITIONS[scenario.family].outcomes:
                self.assertNotIn(
                    label.casefold().replace("_", " "),
                    instruction,
                    f"instruction leaks benchmark outcome {label!r}",
                )

    def test_expected_labels_match_live_prefmem_contracts(self) -> None:
        scenarios = build_catalog(families=["block_stack"], seeds=[37])
        labels = {
            scenario.outcome: scenario.expected_validator_label
            for scenario in scenarios
        }
        self.assertEqual(labels["success"], "SUCCESS")
        self.assertEqual(labels["wrong_complete"], "FAILURE")
        self.assertEqual(labels["partial"], "PARTIAL")
        self.assertEqual(labels["near_miss"], "FAILURE")
        self.assertEqual(labels["unknown"], "UNKNOWN")
        self.assertEqual(labels["unsafe"], "NOT_RUN")

    def test_already_satisfied_controls_are_observation_only(self) -> None:
        controls = build_control_catalog(
            families=list(FAMILIES),
            seeds=[41],
        )
        self.assertEqual(len(controls), len(FAMILIES) * 2)
        for control in controls:
            self.assertEqual(control.control_kind, "already_satisfied")
            self.assertEqual(control.outcome, "success")
            self.assertEqual(control.initial_objects, control.final_objects)
            self.assertTrue(control.goals_satisfied)

    def test_spatial_oracle_is_recomputed_from_terminal_geometry(self) -> None:
        success = next(
            scenario
            for scenario in build_catalog(families=["block_stack"], seeds=[43])
            if scenario.outcome == "success"
        )
        top = max(success.final_objects, key=lambda item: item.position_m[2])
        displaced = replace(
            top,
            position_m=(
                top.position_m[0] + 0.08,
                top.position_m[1],
                top.position_m[2],
            ),
        )
        tampered_objects = tuple(
            displaced if item.object_id == top.object_id else item
            for item in success.final_objects
        )

        measured = evaluate_goal_predicates(
            success.family,
            success.goal_predicates,
            tampered_objects,
            success.evidence,
            observable=True,
        )

        self.assertFalse(all(predicate.value for predicate in measured))
        self.assertTrue(
            any(
                predicate.name in {"supported_by", "vertically_aligned"}
                and not predicate.value
                for predicate in measured
            )
        )


class BenchmarkGenerationTests(unittest.TestCase):
    def _generate(self, root: Path, *, families: list[str] | None = None) -> None:
        generate_benchmark(
            root,
            families=families or ["block_stack"],
            seeds=[7],
            backend="synthetic",
            overwrite=False,
        )

    def test_generated_episodes_are_numbered_and_dataset_episode_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._generate(root)

            episodes = _episode_directories(root)
            self.assertTrue(episodes)
            for episode_dir in episodes:
                self.assertEqual(
                    {path.name for path in episode_dir.iterdir()},
                    {"1.png", "2.png", "manifest.json"},
                )
                episode = DatasetEpisode.from_path(episode_dir, root)
                self.assertEqual(episode.initial_frame.name, "1.png")
                self.assertEqual(episode.final_frame.name, "2.png")
                for frame in (episode.initial_frame, episode.final_frame):
                    with Image.open(frame) as image:
                        image.verify()
                        self.assertEqual(image.format, "PNG")

    def test_manifest_structure_matches_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._generate(root)
            catalog = {
                scenario.scenario_id: scenario
                for scenario in build_catalog(families=["block_stack"], seeds=[7])
            }

            self.assertEqual(
                {path.name for path in _episode_directories(root)},
                set(catalog),
            )
            required = {
                "schema_version",
                "scenario_id",
                "scene",
                "target",
                "expected_outcome",
                "frames",
                "ground_truth",
                "benchmark_expectations",
            }
            for episode_dir in _episode_directories(root):
                manifest = _manifest(episode_dir / "manifest.json")
                self.assertTrue(required <= manifest.keys())
                self.assertEqual(manifest["scenario_id"], episode_dir.name)
                self.assertIn(manifest["scenario_id"], catalog)
                self.assertEqual(
                    manifest["expected_outcome"],
                    catalog[manifest["scenario_id"]].outcome,
                )
                self.assertEqual(_frame_names(manifest["frames"]), {"1.png", "2.png"})
                self.assertIsInstance(manifest["scene"], dict)
                self.assertIsInstance(manifest["target"], dict)
                self.assertIsInstance(manifest["ground_truth"], dict)
                self.assertIsInstance(manifest["benchmark_expectations"], dict)

    def test_generation_is_reproducible_across_roots(self) -> None:
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first_root = Path(first_dir)
            second_root = Path(second_dir)
            self._generate(first_root, families=["block_stack", "category_sort"])
            self._generate(second_root, families=["block_stack", "category_sort"])

            self.assertEqual(_tree_digest(first_root), _tree_digest(second_root))

    def test_generation_without_overwrite_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._generate(root)
            files = [path for path in root.rglob("*") if path.is_file()]
            self.assertTrue(files)
            fixed_time = 1_600_000_000
            for path in files:
                os.utime(path, (fixed_time, fixed_time))
            before_digest = _tree_digest(root)
            before_mtimes = {
                path.relative_to(root).as_posix(): path.stat().st_mtime_ns for path in files
            }

            self._generate(root)

            after_files = [path for path in root.rglob("*") if path.is_file()]
            after_mtimes = {
                path.relative_to(root).as_posix(): path.stat().st_mtime_ns
                for path in after_files
            }
            self.assertEqual(_tree_digest(root), before_digest)
            self.assertEqual(after_mtimes, before_mtimes)

    def test_generation_reuses_packet_with_negligible_float_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._generate(root)
            episode = _episode_directories(root)[0]
            manifest_path = episode / "manifest.json"
            manifest = _manifest(manifest_path)
            orientation = manifest["scene"]["initial_objects"][0][
                "orientation_xyzw"
            ]
            orientation[0] = float(orientation[0]) + 5e-13
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            report = generate_benchmark(
                root,
                families=["block_stack"],
                seeds=[7],
                backend="synthetic",
                overwrite=False,
            )

            self.assertEqual(report.generated, 0)
            self.assertEqual(report.skipped, report.scenario_count)

    def test_incompatible_root_is_rejected_before_existing_packets_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._generate(root, families=["block_stack"])
            before = _tree_digest(root)

            with self.assertRaisesRegex(ValueError, "different benchmark"):
                generate_benchmark(
                    root,
                    families=["category_sort"],
                    seeds=[7],
                    backend="synthetic",
                    overwrite=False,
                )

            self.assertEqual(_tree_digest(root), before)
            self.assertTrue(validate_benchmark(root).valid)

    def test_overwrite_prunes_stale_generated_packets_for_a_new_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._generate(root, families=["block_stack"])
            old_ids = {path.name for path in _episode_directories(root)}

            generate_benchmark(
                root,
                families=["category_sort"],
                seeds=[7],
                backend="synthetic",
                overwrite=True,
            )

            new_ids = {path.name for path in _episode_directories(root)}
            self.assertFalse(old_ids & new_ids)
            report = validate_benchmark(root)
            self.assertTrue(report.valid, report.errors)

    def test_generation_can_include_already_satisfied_control_packets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = generate_benchmark(
                root,
                families=["block_stack"],
                seeds=[7],
                backend="synthetic",
                include_controls=True,
            )
            self.assertEqual(report.scenario_count, 26)
            control_manifests = [
                (path.parent, _manifest(path))
                for path in root.rglob("manifest.json")
                if _manifest(path)["scene"]["control_kind"]
                == "already_satisfied"
            ]
            self.assertEqual(len(control_manifests), 2)
            for episode, manifest in control_manifests:
                self.assertEqual(
                    (episode / "1.png").read_bytes(),
                    (episode / "2.png").read_bytes(),
                )
                self.assertEqual(
                    manifest["benchmark_expectations"]["planner"]["expected_status"],
                    "ALREADY_SATISFIED",
                )
                self.assertEqual(
                    manifest["benchmark_expectations"]["execution"],
                    {
                        "expected_status": "OBSERVATION_ONLY",
                        "expected_vla_dispatches": 0,
                    },
                )
            validation = validate_benchmark(root)
            self.assertTrue(validation.valid, validation.errors)

    def test_initial_frame_cannot_leak_target_or_outcome(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._generate(root)
            initial_digests: dict[tuple[str, str, int], set[str]] = defaultdict(set)

            for episode_dir in _episode_directories(root):
                manifest = _manifest(episode_dir / "manifest.json")
                scene = manifest["scene"]
                key = (
                    str(scene["family"]),
                    str(scene["variant"]),
                    int(scene["seed"]),
                )
                initial_digests[key].add(
                    hashlib.sha256((episode_dir / "1.png").read_bytes()).hexdigest()
                )
                with Image.open(episode_dir / "1.png") as image:
                    textual_metadata = getattr(image, "text", {})
                    metadata_text = json.dumps(
                        image.info, default=str, sort_keys=True
                    ).casefold()
                    forbidden_metadata = {
                        token
                        for token in (
                            "label",
                            "outcome",
                            "target",
                            "scenario",
                            "instruction",
                        )
                        if token in metadata_text
                    }
                self.assertFalse(
                    textual_metadata,
                    f"initial PNG has textual metadata: {textual_metadata}",
                )
                self.assertFalse(
                    forbidden_metadata,
                    f"initial PNG metadata leaks labels: {forbidden_metadata}",
                )

            self.assertTrue(initial_digests)
            for key, digests in initial_digests.items():
                self.assertEqual(
                    len(digests),
                    1,
                    f"initial pixels vary with target/outcome for base scene {key}",
                )

    def test_model_context_excludes_oracle_labels_and_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._generate(root)
            contexts_by_initial_digest: dict[str, set[str]] = defaultdict(set)

            for episode_dir in _episode_directories(root):
                packet = BenchmarkEpisode.from_path(episode_dir, root)
                context = packet.model_context()
                self.assertEqual(
                    set(context),
                    {"observation_id", "available_modalities"},
                )
                self.assertRegex(
                    context["observation_id"],
                    r"^obs-[0-9a-f]{24}$",
                )
                initial_digest = hashlib.sha256(
                    (episode_dir / "1.png").read_bytes()
                ).hexdigest()
                contexts_by_initial_digest[initial_digest].add(
                    context["observation_id"]
                )
                serialized = json.dumps(context, sort_keys=True).casefold()
                manifest = packet.manifest
                forbidden = (
                    str(manifest["expected_outcome"]),
                    str(manifest["target"]["target_id"]),
                    str(manifest["target"]["instruction"]),
                    "ground_truth",
                    "benchmark_expectations",
                    str(episode_dir),
                )
                for value in forbidden:
                    normalized = value.casefold()
                    if normalized:
                        self.assertNotIn(
                            normalized,
                            serialized,
                            f"model context leaks oracle value {value!r}",
                        )
            self.assertTrue(contexts_by_initial_digest)
            self.assertTrue(
                any(
                    sum(
                        hashlib.sha256(
                            (episode_dir / "1.png").read_bytes()
                        ).hexdigest()
                        == digest
                        for episode_dir in _episode_directories(root)
                    )
                    > 1
                    for digest in contexts_by_initial_digest
                ),
                "test fixture has no counterfactual siblings",
            )
            self.assertTrue(
                all(len(observation_ids) == 1 for observation_ids in contexts_by_initial_digest.values()),
                "counterfactual siblings expose distinct observation IDs",
            )

    def test_unknown_endpoint_is_fully_blocked_by_a_physical_panel(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._generate(root)
            unknown_directory = next(
                path.parent
                for path in root.rglob("manifest.json")
                if _manifest(path)["expected_outcome"] == "unknown"
            )
            with Image.open(unknown_directory / "2.png") as image:
                centre = image.convert("RGB").crop((100, 190, 540, 420))
                self.assertEqual(
                    len(centre.getcolors(maxcolors=centre.width * centre.height) or []),
                    1,
                    "unknown final view leaves task pixels visible through the panel",
                )


class BenchmarkValidationTests(unittest.TestCase):
    def _valid_root(self, root: Path) -> list[Path]:
        generate_benchmark(
            root,
            families=["block_stack"],
            seeds=[5],
            backend="synthetic",
            overwrite=False,
        )
        episodes = _episode_directories(root)
        self.assertTrue(episodes)
        report = validate_benchmark(root)
        self.assertTrue(report.valid, report.errors)
        self.assertFalse(report.errors)
        return episodes

    def test_validator_accepts_a_fresh_benchmark(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self._valid_root(Path(directory))

    def test_validator_accepts_negligible_quaternion_float_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episodes = self._valid_root(root)
            manifest_path = episodes[0] / "manifest.json"
            manifest = _manifest(manifest_path)
            orientation = manifest["scene"]["initial_objects"][0][
                "orientation_xyzw"
            ]
            orientation[0] = float(orientation[0]) + 5e-13
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            report = validate_benchmark(root)

            self.assertTrue(report.valid, report.errors)
            self.assertFalse(report.errors)

    def test_validator_rejects_material_quaternion_float_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episodes = self._valid_root(root)
            manifest_path = episodes[0] / "manifest.json"
            manifest = _manifest(manifest_path)
            orientation = manifest["scene"]["initial_objects"][0][
                "orientation_xyzw"
            ]
            orientation[0] = float(orientation[0]) + 1e-4
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            report = validate_benchmark(root)

            self.assertFalse(report.valid)
            self.assertTrue(
                any(
                    "scene differs from the deterministic catalog" in error
                    for error in report.errors
                ),
                report.errors,
            )

    def test_validator_rejects_quaternion_numeric_type_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episodes = self._valid_root(root)
            manifest_path = episodes[0] / "manifest.json"
            manifest = _manifest(manifest_path)
            orientation = manifest["scene"]["initial_objects"][0][
                "orientation_xyzw"
            ]
            orientation[0] = str(orientation[0])
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            report = validate_benchmark(root)

            self.assertFalse(report.valid)
            self.assertTrue(
                any(
                    "scene differs from the deterministic catalog" in error
                    for error in report.errors
                ),
                report.errors,
            )

    def test_validator_rejects_integer_to_float_type_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episodes = self._valid_root(root)
            manifest_path = episodes[0] / "manifest.json"
            manifest = _manifest(manifest_path)
            manifest["scene"]["seed"] = float(manifest["scene"]["seed"])
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            report = validate_benchmark(root)

            self.assertFalse(report.valid)
            self.assertTrue(
                any(
                    "scene differs from the deterministic catalog" in error
                    for error in report.errors
                ),
                report.errors,
            )

    def test_validator_rejects_a_missing_numbered_frame(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episodes = self._valid_root(root)
            (episodes[0] / "2.png").unlink()

            report = validate_benchmark(root)

            self.assertFalse(report.valid)
            self.assertTrue(report.errors)

    def test_validator_rejects_manifest_directory_identity_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episodes = self._valid_root(root)
            manifest_path = episodes[0] / "manifest.json"
            manifest = _manifest(manifest_path)
            manifest["scenario_id"] = "scenario-" + "0" * 32
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            report = validate_benchmark(root)

            self.assertFalse(report.valid)
            self.assertTrue(report.errors)

    def test_validator_rejects_a_missing_required_manifest_field(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episodes = self._valid_root(root)
            manifest_path = episodes[0] / "manifest.json"
            manifest = _manifest(manifest_path)
            del manifest["ground_truth"]
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            report = validate_benchmark(root)

            self.assertFalse(report.valid)
            self.assertTrue(report.errors)

    def test_validator_rejects_unknown_outcome_and_bad_frame_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episodes = self._valid_root(root)
            manifest_path = episodes[0] / "manifest.json"
            manifest = _manifest(manifest_path)
            manifest["expected_outcome"] = "looks_good"
            manifest["frames"] = ["initial.png", "final.png"]
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            report = validate_benchmark(root)

            self.assertFalse(report.valid)
            self.assertTrue(report.errors)

    def test_validator_rejects_semantically_relabelled_oracle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episodes = self._valid_root(root)
            success_path = next(
                path / "manifest.json"
                for path in episodes
                if _manifest(path / "manifest.json")["expected_outcome"] == "success"
            )
            manifest = _manifest(success_path)
            manifest["expected_outcome"] = "unknown"
            manifest["benchmark_expectations"]["validator"]["outcome"] = "UNKNOWN"
            manifest["benchmark_expectations"]["validator"]["task_complete"] = None
            success_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            report = validate_benchmark(root)

            self.assertFalse(report.valid)
            self.assertTrue(
                any("deterministic catalog" in error or "semantic scenario" in error
                    for error in report.errors),
                report.errors,
            )

    def test_validator_detects_a_missing_matrix_cell_even_if_index_is_rewritten(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episodes = self._valid_root(root)
            removed_id = episodes[0].name
            shutil.rmtree(episodes[0])
            index_path = root / "index.json"
            index = _manifest(index_path)
            index["scenario_ids"] = [
                value for value in index["scenario_ids"] if value != removed_id
            ]
            index["scenario_count"] = len(index["scenario_ids"])
            index_path.write_text(
                json.dumps(index, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            report = validate_benchmark(root)

            self.assertFalse(report.valid)
            self.assertTrue(report.errors)


class BenchmarkExecutorTests(unittest.TestCase):
    def test_unsafe_packet_stops_at_execution_without_leaking_oracle_label(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generate_benchmark(
                root,
                families=["block_stack"],
                seeds=[13],
                backend="synthetic",
            )
            unsafe_directory = next(
                path.parent
                for path in root.rglob("manifest.json")
                if _manifest(path)["expected_outcome"] == "unsafe"
            )
            packet = BenchmarkEpisode.from_path(unsafe_directory, root)
            executor = BenchmarkEpisodeExecutor(packet)

            result = executor.execute_subtask(
                {"task_instruction": "Move the first block."},
                index=1,
                attempt=1,
            )

            self.assertEqual(result["status"], "UNSAFE")
            self.assertFalse(result["physical_execution_claimed"])
            serialized = json.dumps(result, sort_keys=True).casefold()
            self.assertNotIn(packet.manifest["target"]["target_id"], serialized)
            self.assertNotIn("expected_outcome", serialized)

    def test_hri_and_memory_models_receive_no_benchmark_filesystem_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            private_root = Path(directory) / "rgb_success_private_label"
            generate_benchmark(
                private_root,
                families=["block_stack"],
                seeds=[17],
                backend="synthetic",
            )
            episode_directory = _episode_directories(private_root)[0]
            packet = BenchmarkEpisode.from_path(episode_directory, REPO_ROOT)
            executor = BenchmarkEpisodeExecutor(packet)
            model = ScriptedJsonModel(
                {
                    "resolve_hri_turn": {
                        "mode": "ASK",
                        "user_message": "Which bottom-to-top order should I use?",
                        "task_contract": None,
                        "pending_question": {
                            "kind": "TASK_CLARIFICATION",
                            "payload": {},
                        },
                        "memory_action": {"action": "NONE"},
                        "report": None,
                        "trace": {
                            "grounding": [],
                            "memory_refs": [],
                            "history_refs": [],
                            "assumptions": [],
                        },
                    }
                }
            )
            memory = RecordingMemoryAgent()
            config = PrefMemConfig(
                workspace_root=str(REPO_ROOT),
                dataset_path=str(episode_directory),
                history_store_path=str(Path(directory) / "history.json"),
                history_outbox_path=str(Path(directory) / "history_outbox.json"),
                preference_store_path=str(Path(directory) / "preferences.json"),
                user_id="path-sanitization-user",
            )
            orchestrator = HRIOrchestrator(
                config,
                hri_model=model,
                memory_agent=memory,
                planner_agent=Mock(),
                validator_agent=Mock(),
                executor=executor,
            )

            orchestrator.handle_user_message("Stack the blocks.")

            expected_context = packet.model_context()
            hri_scene = model.calls_for("resolve_hri_turn")[0]["payload"]["scene"]
            self.assertEqual(hri_scene, expected_context)
            self.assertEqual(memory.context_queries[0].scene, expected_context)
            serialized = json.dumps(
                {
                    "hri_scene": hri_scene,
                    "memory_scene": memory.context_queries[0].scene,
                },
                sort_keys=True,
            )
            self.assertNotIn("rgb_success_private_label", serialized)
            self.assertNotIn(str(episode_directory), serialized)


class BenchmarkScorerTests(unittest.TestCase):
    @staticmethod
    def _oracle_aligned_semantics(
        manifest: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        intent = str(manifest["target"]["instruction"])
        resolved_task = {
            "confirmed_intent": intent,
            "parameters": {
                "target_id": str(manifest["target"]["target_id"]),
            },
        }
        validation_spec = {
            "spec_id": "spec-oracle-aligned",
            "confirmed_intent": intent,
            "goal_conditions": [
                {
                    "id": f"goal-{index}",
                    "description": "Structured benchmark goal.",
                    "predicate": predicate,
                    "arguments": [],
                }
                for index, predicate in enumerate(
                    manifest["target"]["goal_predicates"],
                    start=1,
                )
            ],
        }
        return resolved_task, validation_spec

    def test_scorer_consumes_native_hri_result_shape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generate_benchmark(
                root,
                families=["block_stack"],
                seeds=[19],
                backend="synthetic",
            )
            manifest = next(
                _manifest(path)
                for path in root.rglob("manifest.json")
                if _manifest(path)["expected_outcome"] == "success"
            )
            resolved_task, validation_spec = self._oracle_aligned_semantics(
                manifest
            )
            runtime_result = {
                "hri": {"mode": "REPORT", "report": {"outcome": "SUCCESS"}},
                "memory": None,
                "resolved_task": resolved_task,
                "task": {
                    "outcome": "SUCCESS",
                    "next_action": "NONE",
                    "attempts": [
                        {
                            "plan": {
                                "planning_status": "READY",
                                "validation_spec": validation_spec,
                            },
                            "execution": {
                                "status": "OBSERVED_RECORDED_ATTEMPT"
                            },
                            "validation": {
                                "outcome": "SUCCESS",
                                "task_complete": True,
                            },
                            "validation_assurance": {"next_action": "NONE"},
                        }
                    ],
                },
            }

            report = score_agent_result(
                manifest,
                runtime_result,
                allow_partial=True,
            )
            strict_report = score_agent_result(manifest, runtime_result)

            self.assertTrue(report.passed, report.checks)
            self.assertGreater(report.total, 0)
            self.assertEqual(report.skipped, 1)  # history delta needs store snapshots
            self.assertFalse(strict_report.passed)

    def test_scorer_requires_target_and_goal_oracle_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generate_benchmark(
                root,
                families=["block_stack"],
                seeds=[31],
                backend="synthetic",
            )
            manifest = next(
                _manifest(path)
                for path in root.rglob("manifest.json")
                if _manifest(path)["expected_outcome"] == "success"
            )

            report = score_agent_result(
                manifest,
                {
                    "task": {
                        "attempts": [
                            {"plan": {"planning_status": "READY"}}
                        ]
                    }
                },
                allow_partial=True,
            )

            for name in (
                "hri_resolved_target",
                "planner_confirmed_intent",
                "planner_goal_predicates",
            ):
                check = next(
                    item for item in report.checks if item["name"] == name
                )
                self.assertTrue(check["evaluated"])
                self.assertFalse(check["passed"])
            self.assertFalse(report.passed)

    def test_scorer_rejects_wrong_hri_target_despite_self_consistent_result(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generate_benchmark(
                root,
                families=["block_stack"],
                seeds=[37],
                backend="synthetic",
            )
            manifest = next(
                _manifest(path)
                for path in root.rglob("manifest.json")
                if _manifest(path)["target"]["target_id"]
                == "rgb_bottom_to_top"
                and _manifest(path)["expected_outcome"] == "success"
            )
            wrong_intent = (
                "Stack the blocks in blue, green, red order from bottom to top."
            )
            _, validation_spec = self._oracle_aligned_semantics(manifest)
            validation_spec["confirmed_intent"] = wrong_intent
            result = {
                "resolved_task": {
                    "confirmed_intent": wrong_intent,
                    "parameters": {
                        "color_positions": {
                            "bottom": "blue",
                            "middle": "green",
                            "top": "red",
                        }
                    },
                },
                "task": {
                    "attempts": [
                        {
                            "plan": {
                                "planning_status": "READY",
                                "validation_spec": validation_spec,
                            }
                        }
                    ]
                },
            }

            report = score_agent_result(
                manifest,
                result,
                allow_partial=True,
            )

            target_check = next(
                item
                for item in report.checks
                if item["name"] == "hri_resolved_target"
            )
            self.assertEqual(
                target_check["actual"],
                "bgr_bottom_to_top",
            )
            self.assertFalse(target_check["passed"])

    def test_scorer_canonicalises_semantic_predicate_arguments(self) -> None:
        intent = (
            "Stack the red block at the bottom and the blue block on top."
        )
        manifest = {
            "scenario_id": "semantic-alias-case",
            "scene": {
                "family": "block_stack",
                "initial_objects": [
                    {
                        "object_id": "obj-red",
                        "object_type": "cube",
                        "attributes": {"colour": "red"},
                    },
                    {
                        "object_id": "obj-green",
                        "object_type": "cube",
                        "attributes": {"colour": "green"},
                    },
                    {
                        "object_id": "obj-blue",
                        "object_type": "cube",
                        "attributes": {"colour": "blue"},
                    },
                ],
            },
            "target": {
                "target_id": "rgb_bottom_to_top",
                "instruction": intent,
                "goal_predicates": [
                    "SUPPORTED_BY(obj-green,obj-red)=true",
                    "SUPPORTED_BY(obj-blue,obj-green)=true",
                ],
            },
            "benchmark_expectations": {},
        }
        result = {
            "resolved_task": {
                "confirmed_intent": (
                    "Put red at the bottom, green in the middle, "
                    "and blue at the top."
                ),
                "parameters": {},
            },
            "task": {
                "attempts": [
                    {
                        "plan": {
                            "validation_spec": {
                                "confirmed_intent": (
                                    "Put red at the bottom, green in the middle, "
                                    "and blue at the top."
                                ),
                                "goal_conditions": [
                                    {
                                        "id": "middle-on-bottom",
                                        "predicate": "supported-by",
                                        "arguments": [
                                            "the green block",
                                            "red cube",
                                        ],
                                    },
                                    {
                                        "id": "top-on-middle",
                                        "predicate": "SUPPORTED_BY",
                                        "arguments": ["blue", "green"],
                                    },
                                ],
                            }
                        }
                    }
                ]
            },
        }

        report = score_agent_result(manifest, result)

        self.assertTrue(report.passed, report.checks)
        predicate_check = next(
            item
            for item in report.checks
            if item["name"] == "planner_goal_predicates"
        )
        self.assertTrue(predicate_check["passed"])

    def test_scorer_treats_set_valued_goal_arguments_as_unordered(self) -> None:
        intent = (
            "Place printed items on the left and electronic devices on the right."
        )
        manifest = {
            "scenario_id": "unordered-category-completeness",
            "scene": {
                "family": "category_sort",
                "initial_objects": [
                    {
                        "object_id": "obj-001",
                        "object_type": "tablet",
                        "attributes": {
                            "role": "tablet",
                            "semantic_category": "electronic",
                        },
                    },
                    {
                        "object_id": "obj-002",
                        "object_type": "book",
                        "attributes": {
                            "role": "book",
                            "semantic_category": "printed",
                        },
                    },
                ],
            },
            "target": {
                "target_id": "printed_left",
                "instruction": intent,
                "goal_predicates": [
                    "ALL_ITEMS_ASSIGNED(obj-001,obj-002)=true",
                ],
            },
            "benchmark_expectations": {},
        }
        result = {
            "resolved_task": {
                "confirmed_intent": intent,
                "parameters": {
                    "printed": "left",
                    "electronic": "right",
                },
            },
            "task": {
                "attempts": [
                    {
                        "plan": {
                            "validation_spec": {
                                "confirmed_intent": intent,
                                "goal_conditions": [
                                    {
                                        "id": "all-assigned",
                                        "predicate": "ALL_ITEMS_ASSIGNED",
                                        # Semantic item order need not reproduce
                                        # the oracle's hidden object-ID order.
                                        "arguments": ["book", "tablet"],
                                    }
                                ],
                            }
                        }
                    }
                ]
            },
        }

        report = score_agent_result(manifest, result)

        self.assertTrue(report.passed, report.checks)

    def test_scorer_rejects_description_only_and_extra_goal_conditions(
        self,
    ) -> None:
        manifest = {
            "scenario_id": "strict-goal-schema",
            "scene": {"family": "place_setting", "initial_objects": []},
            "target": {
                "target_id": "right_handed",
                "instruction": "Create a right-handed place setting.",
                "goal_predicates": ["AT_ANCHOR(plate,place-centre)=true"],
            },
            "benchmark_expectations": {},
        }
        result = {
            "resolved_task": {
                "confirmed_intent": "Create a right-handed place setting.",
                "parameters": {},
            },
            "task": {
                "attempts": [
                    {
                        "plan": {
                            "validation_spec": {
                                "confirmed_intent": (
                                    "Create a right-handed place setting."
                                ),
                                "goal_conditions": [
                                    {
                                        "id": "description-only",
                                        "description": "The plate is centred.",
                                    },
                                    {
                                        "id": "extra",
                                        "predicate": "DECORATED",
                                        "arguments": ["plate"],
                                    },
                                ],
                            }
                        }
                    }
                ]
            },
        }

        report = score_agent_result(
            manifest,
            result,
            allow_partial=True,
        )

        predicate_check = next(
            item
            for item in report.checks
            if item["name"] == "planner_goal_predicates"
        )
        self.assertFalse(predicate_check["passed"])
        self.assertIn("issues", predicate_check["details"])
        self.assertFalse(report.passed)

    def test_non_benchmark_manifest_keeps_legacy_partial_scoring(self) -> None:
        report = score_agent_result(
            {
                "scenario_id": "legacy",
                "benchmark_expectations": {
                    "planner": {"expected_status": "READY"}
                },
            },
            {
                "task": {
                    "attempts": [
                        {"plan": {"planning_status": "READY"}}
                    ]
                }
            },
            allow_partial=True,
        )

        self.assertTrue(report.passed, report.checks)
        self.assertEqual(
            [check["name"] for check in report.checks],
            ["planner_status"],
        )

    def test_scorer_detects_validator_call_after_unsafe_execution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generate_benchmark(
                root,
                families=["block_stack"],
                seeds=[29],
                backend="synthetic",
            )
            manifest = next(
                _manifest(path)
                for path in root.rglob("manifest.json")
                if _manifest(path)["expected_outcome"] == "unsafe"
            )
            invalid_result = {
                "hri": {
                    "mode": "REPORT",
                    "report": {"outcome": "ABORTED_SAFETY"},
                },
                "memory": None,
                "task": {
                    "outcome": "ABORTED_SAFETY",
                    "next_action": "ABORT_SAFETY",
                    "attempts": [
                        {
                            "execution": {"status": "UNSAFE"},
                            "execution_assurance": {
                                "next_action": "ABORT_SAFETY"
                            },
                            "validation": {
                                "outcome": "UNSAFE",
                                "task_complete": False,
                            },
                        }
                    ],
                },
            }

            report = score_agent_result(manifest, invalid_result)

            self.assertFalse(report.passed)
            called_check = next(
                check
                for check in report.checks
                if check["name"] == "validator_called"
            )
            self.assertFalse(called_check["passed"])


if __name__ == "__main__":
    unittest.main()
