from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from agents.configs import normalize_model_base_url


CONVERSATION_SCHEMA_VERSION = "robopref.conversation-evaluation.v1"
CASE_SCHEMA_VERSION = "robopref.conversation-cases.v1"

SUITES = ("endpoint", "dialogue", "memory", "recovery", "safety")
NEAR_MISS_POLICIES = ("perceptual", "strict")
RUN_STATUSES = (
    "COMPLETED",
    "AGENT_TERMINATED",
    "INFRASTRUCTURE_ERROR",
    "BENCHMARK_ERROR",
    "TIMEOUT",
    "ARTIFACT_ERROR",
)
BENCHMARK_RESULTS = ("PASS", "FAIL", "NOT_SCORED")
CHECK_STATUSES = ("PASS", "FAIL", "MISSING", "NOT_APPLICABLE")
CHECK_CRITICALITIES = ("HARD", "DIAGNOSTIC")


def stable_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class EpisodeDescriptor:
    path: Path
    scenario_id: str
    family: str
    scene_variant: str
    seed: int
    counterfactual_group_id: str
    target_id: str
    outcome: str
    control_kind: str | None
    instruction: str
    initial_frame_sha256: str
    final_frame_sha256: str
    manifest: dict[str, Any] = field(repr=False, compare=False)

    @classmethod
    def from_manifest(
        cls,
        path: str | Path,
        manifest: Mapping[str, Any],
    ) -> EpisodeDescriptor:
        episode_path = Path(path).resolve()
        scene = manifest.get("scene")
        target = manifest.get("target")
        frames = manifest.get("frames")
        hashes = manifest.get("frame_sha256")
        if not isinstance(scene, Mapping):
            raise ValueError("Manifest scene must be an object.")
        if not isinstance(target, Mapping):
            raise ValueError("Manifest target must be an object.")
        if not isinstance(frames, Mapping) or not isinstance(hashes, Mapping):
            raise ValueError("Manifest frames and frame_sha256 must be objects.")
        initial_name = str(frames.get("initial", ""))
        final_name = str(frames.get("final", ""))
        initial_hash = str(hashes.get(initial_name, "")).strip()
        final_hash = str(hashes.get(final_name, "")).strip()
        scenario_id = str(manifest.get("scenario_id", "")).strip()
        if not scenario_id or scenario_id != episode_path.name:
            raise ValueError("Manifest scenario_id must match its episode directory.")
        if not initial_hash or not final_hash:
            raise ValueError("Manifest frame hashes are incomplete.")
        return cls(
            path=episode_path,
            scenario_id=scenario_id,
            family=str(scene.get("family", "")).strip(),
            scene_variant=str(scene.get("variant", "")).strip(),
            seed=int(scene.get("seed")),
            counterfactual_group_id=str(
                scene.get("counterfactual_group_id", "")
            ).strip(),
            target_id=str(target.get("target_id", "")).strip(),
            outcome=str(manifest.get("expected_outcome", "")).strip(),
            control_kind=(
                str(scene["control_kind"]).strip()
                if scene.get("control_kind") is not None
                else None
            ),
            instruction=str(target.get("instruction", "")).strip(),
            initial_frame_sha256=initial_hash,
            final_frame_sha256=final_hash,
            manifest=copy.deepcopy(dict(manifest)),
        )

    def public_metadata(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "family": self.family,
            "scene_variant": self.scene_variant,
            "seed": self.seed,
            "counterfactual_group_id": self.counterfactual_group_id,
            "target_id": self.target_id,
            "outcome": self.outcome,
            "control_kind": self.control_kind,
            "initial_frame_sha256": self.initial_frame_sha256,
            "final_frame_sha256": self.final_frame_sha256,
        }


@dataclass(frozen=True, slots=True)
class PreferenceFixture:
    owner_user_id: str
    statement: str
    task_type_hint: str
    applicability: dict[str, Any]
    structured_value: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(asdict(self))


@dataclass(frozen=True, slots=True)
class CommandSpec:
    command_id: str
    episode_id: str
    query: str
    target_id: str | None
    query_kind: str
    response_policy: str = "target"
    memory_consent_policy: str = "decline"
    expected_initial_modes: tuple[str, ...] = ("EXECUTE",)
    score_endpoint: bool = True
    packet_outcome: str | None = None
    expected_history_delta: int | None = 1
    expected_preference_delta: int | None = 0
    expected_preference_use: bool | None = None
    expected_history_use: bool | None = None
    expected_terminal_outcome: str | None = None
    recovery_episode_id: str | None = None
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.command_id.strip():
            raise ValueError("Command IDs must be non-empty.")
        if not self.episode_id.strip():
            raise ValueError("Command specs require an episode ID.")
        if not self.query.strip():
            raise ValueError("Command specs require a user query.")
        if not self.expected_initial_modes:
            raise ValueError("Command specs require at least one allowed initial mode.")
        if self.memory_consent_policy not in {
            "commit",
            "decline",
            "defer",
            "keep-open",
        }:
            raise ValueError(
                f"Unsupported memory consent policy: {self.memory_consent_policy}"
            )

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(asdict(self))


@dataclass(frozen=True, slots=True)
class ConversationCase:
    case_id: str
    suite: str
    profile: str
    user_id: str
    commands: tuple[CommandSpec, ...]
    cluster_id: str
    fixture: PreferenceFixture | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.suite not in SUITES:
            raise ValueError(f"Unsupported conversation suite: {self.suite}")
        if not self.case_id.strip() or not self.user_id.strip():
            raise ValueError("Conversation case and user IDs must be non-empty.")
        if not self.commands:
            raise ValueError("Conversation cases require at least one command.")
        command_ids = [item.command_id for item in self.commands]
        if len(command_ids) != len(set(command_ids)):
            raise ValueError("Command IDs must be unique within a conversation case.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CASE_SCHEMA_VERSION,
            "case_id": self.case_id,
            "suite": self.suite,
            "profile": self.profile,
            "user_id": self.user_id,
            "cluster_id": self.cluster_id,
            "fixture": self.fixture.to_dict() if self.fixture else None,
            "commands": [item.to_dict() for item in self.commands],
            "metadata": copy.deepcopy(self.metadata),
        }

    @property
    def digest(self) -> str:
        return stable_digest(self.to_dict())


@dataclass(frozen=True, slots=True)
class CheckResult:
    name: str
    stage: str
    status: str
    criticality: str
    expected: Any = None
    actual: Any = None
    details: dict[str, Any] = field(default_factory=dict)
    command_id: str | None = None
    attempt: int | None = None
    root_cause_code: str | None = None

    def __post_init__(self) -> None:
        if self.status not in CHECK_STATUSES:
            raise ValueError(f"Unsupported check status: {self.status}")
        if self.criticality not in CHECK_CRITICALITIES:
            raise ValueError(
                f"Unsupported check criticality: {self.criticality}"
            )

    @property
    def passed(self) -> bool | None:
        if self.status == "PASS":
            return True
        if self.status in {"FAIL", "MISSING"}:
            return False
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "stage": self.stage,
            "status": self.status,
            "criticality": self.criticality,
            "expected": copy.deepcopy(self.expected),
            "actual": copy.deepcopy(self.actual),
            "details": copy.deepcopy(self.details),
            "command_id": self.command_id,
            "attempt": self.attempt,
            "root_cause_code": self.root_cause_code,
        }


@dataclass(slots=True)
class ConversationEvaluationConfig:
    benchmark_root: Path
    output_dir: Path
    suites: tuple[str, ...] = ("endpoint",)
    repetitions: int = 1
    condition: str = "full"
    families: tuple[str, ...] = ()
    scene_variants: tuple[str, ...] = ()
    target_ids: tuple[str, ...] = ()
    outcomes: tuple[str, ...] = ()
    seeds: tuple[int, ...] = ()
    scenario_ids: tuple[str, ...] = ()
    include_controls: bool = True
    max_cases: int | None = None
    shuffle_seed: int = 0
    near_miss_policy: str = "perceptual"
    max_user_turns: int = 6
    resume: bool = True
    fail_fast: bool = False
    show_progress: bool = True
    memory_mode: str = "full"
    model: str | None = None
    model_provider: str | None = None
    model_base_url: str | None = None
    ollama_host: str | None = None
    temperature: float | None = None
    model_seed: int | None = None
    timeout_seconds: float | None = None
    resize_images: bool = False
    display_all: bool = False
    max_replans: int = 1
    max_reobservations: int = 1
    bootstrap_replicates: int = 2_000
    bootstrap_seed: int = 20_260_726

    def __post_init__(self) -> None:
        self.benchmark_root = Path(self.benchmark_root).expanduser().resolve()
        self.output_dir = Path(self.output_dir).expanduser().resolve()
        if self.repetitions <= 0:
            raise ValueError("Repetitions must be positive.")
        if not self.suites:
            raise ValueError("Select at least one conversation suite.")
        unknown_suites = set(self.suites) - set(SUITES)
        if unknown_suites:
            raise ValueError(f"Unsupported suites: {sorted(unknown_suites)}")
        if self.near_miss_policy not in NEAR_MISS_POLICIES:
            raise ValueError(
                f"Unsupported near-miss policy: {self.near_miss_policy}"
            )
        if self.max_cases is not None and self.max_cases <= 0:
            raise ValueError("max_cases must be positive.")
        if self.max_user_turns <= 0:
            raise ValueError("max_user_turns must be positive.")
        if self.max_replans < 0 or self.max_reobservations < 0:
            raise ValueError("Recovery budgets cannot be negative.")
        if self.bootstrap_replicates <= 0:
            raise ValueError("bootstrap_replicates must be positive.")
        if self.model_seed is not None and self.model_seed < 0:
            raise ValueError("model_seed cannot be negative.")
        if self.model_provider is not None:
            self.model_provider = str(self.model_provider).strip().casefold()
            if self.model_provider not in {"ollama", "vllm"}:
                raise ValueError("model_provider must be 'ollama' or 'vllm'.")
        if self.model_base_url is not None:
            if self.model_provider != "vllm":
                raise ValueError(
                    "model_base_url requires model_provider='vllm'."
                )
            self.model_base_url = normalize_model_base_url(
                self.model_base_url
            )
        if self.ollama_host is not None and self.model_provider == "vllm":
            raise ValueError(
                "ollama_host cannot be used with model_provider='vllm'."
            )
        if self.temperature is not None and self.temperature < 0:
            raise ValueError("temperature cannot be negative.")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")

    def semantic_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CONVERSATION_SCHEMA_VERSION,
            "benchmark_root": str(self.benchmark_root),
            "suites": list(self.suites),
            "repetitions": self.repetitions,
            "condition": self.condition,
            "filters": {
                "families": list(self.families),
                "scene_variants": list(self.scene_variants),
                "target_ids": list(self.target_ids),
                "outcomes": list(self.outcomes),
                "seeds": list(self.seeds),
                "scenario_ids": list(self.scenario_ids),
                "include_controls": self.include_controls,
                "max_cases": self.max_cases,
            },
            "shuffle_seed": self.shuffle_seed,
            "near_miss_policy": self.near_miss_policy,
            "max_user_turns": self.max_user_turns,
            "memory_mode": self.memory_mode,
            "model": {
                "provider": self.model_provider,
                "name": self.model,
                "base_url": self.model_base_url,
                "ollama_host": self.ollama_host,
                "temperature": self.temperature,
                "model_seed": self.model_seed,
                "timeout_seconds": self.timeout_seconds,
            },
            "resize_images": self.resize_images,
            "max_replans": self.max_replans,
            "max_reobservations": self.max_reobservations,
        }


@dataclass(frozen=True, slots=True)
class ConversationRunContext:
    case: ConversationCase
    repetition: int
    run_directory: Path
    config: Any
    model_seed: int | None
    episodes: Mapping[str, EpisodeDescriptor]


@dataclass(frozen=True, slots=True)
class ConversationBatchReport:
    output_dir: Path
    results_path: Path
    summary_path: Path
    selected_cases: int
    planned_runs: int
    executed_runs: int
    resumed_runs: int
    passed_runs: int
    failed_runs: int
    agent_terminated_runs: int
    infrastructure_error_runs: int
    benchmark_error_runs: int
    timeout_runs: int
    artifact_error_runs: int

    def to_dict(self) -> dict[str, Any]:
        return {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(self).items()
        }


__all__ = [
    "BENCHMARK_RESULTS",
    "CASE_SCHEMA_VERSION",
    "CHECK_CRITICALITIES",
    "CHECK_STATUSES",
    "CONVERSATION_SCHEMA_VERSION",
    "NEAR_MISS_POLICIES",
    "RUN_STATUSES",
    "SUITES",
    "CheckResult",
    "CommandSpec",
    "ConversationBatchReport",
    "ConversationCase",
    "ConversationEvaluationConfig",
    "ConversationRunContext",
    "EpisodeDescriptor",
    "PreferenceFixture",
    "stable_digest",
]
