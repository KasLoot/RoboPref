"""Configuration for PrefMem model endpoints, prompts, memory, and control.

Defaults match the documented local SSH tunnel for the agent VLM (port 8000)
and the local EmbeddingGemma vLLM server (port 8080).  Environment variables
allow experiments to change endpoints without editing source.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from prefmem.agents.contracts import ReplanPolicy
from prefmem.agents.model_client import ModelEndpoint


AGENTS_ROOT = Path(__file__).resolve().parent
PROMPT_ROOT = AGENTS_ROOT / "prompt"

DEFAULT_AGENT_MODEL = "/workspace/models/gemma-4-26B-A4B-it"
DEFAULT_AGENT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_EMBEDDING_BASE_URL = "http://127.0.0.1:8080/v1"
DEFAULT_EMBEDDING_DIMENSIONS = 768


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value.strip() if value and value.strip() else default


def _positive(value: float | int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


@dataclass(frozen=True, slots=True)
class EmbeddingConfig:
    """Connection and batching settings for the EmbeddingGemma service."""

    base_url: str = field(
        default_factory=lambda: _env(
            "PREFMEM_EMBEDDING_BASE_URL",
            DEFAULT_EMBEDDING_BASE_URL,
        )
    )
    # None deliberately enables strict single-model discovery at /v1/models.
    model: str | None = field(
        default_factory=lambda: (
            os.environ.get("PREFMEM_EMBEDDING_MODEL", "").strip() or None
        )
    )
    api_key: str = field(
        default_factory=lambda: _env("PREFMEM_EMBEDDING_API_KEY", "EMPTY")
    )
    dimensions: int | None = DEFAULT_EMBEDDING_DIMENSIONS
    timeout_seconds: float = 30.0
    max_batch_size: int = 64

    def __post_init__(self) -> None:
        if not self.base_url.strip():
            raise ValueError("embedding base_url must not be empty")
        _positive(self.timeout_seconds, "embedding timeout_seconds")
        _positive(self.max_batch_size, "embedding max_batch_size")
        if self.dimensions not in {None, 128, 256, 512, 768}:
            raise ValueError(
                "embedding dimensions must be one of 128, 256, 512, 768, or None"
            )


@dataclass(frozen=True, slots=True)
class MemoryConfig:
    """Persistent Markdown-memory and semantic-retrieval settings."""

    root: Path = Path(".prefmem/memory")
    retrieval_limit: int = 5
    semantic_candidate_limit: int = 20
    similarity_threshold: float = 0.35
    compact_min_episodes: int = 3
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)

    def __post_init__(self) -> None:
        _positive(self.retrieval_limit, "memory retrieval_limit")
        _positive(
            self.semantic_candidate_limit,
            "memory semantic_candidate_limit",
        )
        _positive(self.compact_min_episodes, "memory compact_min_episodes")
        if not -1.0 <= self.similarity_threshold <= 1.0:
            raise ValueError("memory similarity_threshold must be between -1 and 1")


@dataclass(frozen=True, slots=True)
class ControllerConfig:
    """Finite budgets for every potentially cyclic controller transition."""

    replan_policy: ReplanPolicy = ReplanPolicy.ON_DEVIATION
    max_memory_retrievals_per_turn: int = 1
    max_contract_repairs: int = 1
    max_replans: int = 2
    max_reobservations: int = 2
    max_validation_attempts: int = 3
    max_monitor_cycles_per_subtask: int = 120
    max_dispatches_per_episode: int = 32
    monitor_window_size: int = 3
    success_confirmations: int = 2
    monitor_interval_seconds: float = 1.0
    minimum_planner_confidence: float = 0.8
    minimum_monitor_confidence: float = 0.8
    minimum_validator_confidence: float = 0.8

    def __post_init__(self) -> None:
        for name in (
            "max_memory_retrievals_per_turn",
            "max_contract_repairs",
            "max_replans",
            "max_reobservations",
            "max_validation_attempts",
            "max_monitor_cycles_per_subtask",
            "max_dispatches_per_episode",
            "monitor_window_size",
            "success_confirmations",
        ):
            value = getattr(self, name)
            # A zero replan/reobserve/repair budget is useful for ablations.
            if name in {
                "max_contract_repairs",
                "max_replans",
                "max_reobservations",
            }:
                if value < 0:
                    raise ValueError(f"{name} must be zero or greater")
            else:
                _positive(value, name)
        _positive(self.monitor_interval_seconds, "monitor_interval_seconds")
        for name in (
            "minimum_planner_confidence",
            "minimum_monitor_confidence",
            "minimum_validator_confidence",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class PrefMemConfig:
    """Top-level settings used by the deterministic application assembly."""

    username: str = "default"
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)

    def __post_init__(self) -> None:
        if not self.username.strip():
            raise ValueError("username must not be empty")


class VLLMConfig:
    """Backward-compatible agent-model configuration."""

    def __init__(self) -> None:
        self.model = _env("PREFMEM_AGENT_MODEL", DEFAULT_AGENT_MODEL)
        self.model_base_url = _env(
            "PREFMEM_AGENT_BASE_URL",
            DEFAULT_AGENT_BASE_URL,
        )
        self.api_key = _env("PREFMEM_AGENT_API_KEY", "EMPTY")


class OllamaConfig:
    """Backward-compatible cloud Ollama configuration."""

    def __init__(self) -> None:
        self.model = _env("PREFMEM_OLLAMA_MODEL", "gemma4:31b-cloud")
        self.model_base_url = _env(
            "PREFMEM_OLLAMA_BASE_URL",
            "https://ollama.com/api",
        )
        self.api_key = os.environ.get("OLLAMA_API_KEY", "")


class Global_Model_Config:
    """Compatibility wrapper retained for the current HRI/Planner classes."""

    def __init__(self, model_config: str = "vllm") -> None:
        if model_config == "vllm":
            selected = VLLMConfig()
        elif model_config == "ollama":
            selected = OllamaConfig()
        else:
            raise ValueError(f"Unsupported model configuration: {model_config}")
        self.model_config = selected


class _ReasoningAgentConfig(Global_Model_Config):
    prompt_path: Path

    def __init__(self, model_config: str = "vllm") -> None:
        super().__init__(model_config)
        self.model = self.model_config.model
        self.model_base_url = self.model_config.model_base_url
        self.api_key = self.model_config.api_key
        self.system_prompt_path = self.prompt_path
        self.system_prompt = self.system_prompt_path.read_text(encoding="utf-8")
        self.endpoint = ModelEndpoint(
            model=self.model,
            base_url=self.model_base_url,
            api_key=self.api_key or "EMPTY",
        )


class HRI_Config(_ReasoningAgentConfig):
    prompt_path = PROMPT_ROOT / "hri" / "hri-prompt-v3.md"


class Planner_Config(_ReasoningAgentConfig):
    prompt_path = PROMPT_ROOT / "planner" / "planner-prompt-v2.md"


class Memory_Config(_ReasoningAgentConfig):
    prompt_path = PROMPT_ROOT / "memory" / "prompt-v2.md"


class Monitor_Config(_ReasoningAgentConfig):
    prompt_path = PROMPT_ROOT / "monitor" / "prompt-v1.md"


class Validator_Config(_ReasoningAgentConfig):
    prompt_path = PROMPT_ROOT / "validator" / "prompt-v2.md"


# Modern spellings are aliases so old imports and new application assembly use one
# source of truth.
HRIConfig = HRI_Config
PlannerConfig = Planner_Config
MemoryAgentConfig = Memory_Config
MonitorConfig = Monitor_Config
ValidatorConfig = Validator_Config


__all__ = [
    "AGENTS_ROOT",
    "ControllerConfig",
    "DEFAULT_AGENT_BASE_URL",
    "DEFAULT_AGENT_MODEL",
    "DEFAULT_EMBEDDING_BASE_URL",
    "DEFAULT_EMBEDDING_DIMENSIONS",
    "EmbeddingConfig",
    "Global_Model_Config",
    "HRIConfig",
    "HRI_Config",
    "MemoryAgentConfig",
    "MemoryConfig",
    "Memory_Config",
    "MonitorConfig",
    "Monitor_Config",
    "OllamaConfig",
    "PROMPT_ROOT",
    "PlannerConfig",
    "Planner_Config",
    "PrefMemConfig",
    "VLLMConfig",
    "ValidatorConfig",
    "Validator_Config",
]
