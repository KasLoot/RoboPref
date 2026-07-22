from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]


@dataclass(slots=True)
class AgentModelConfig:
    model: str = "gemma4:31b-cloud"
    temperature: float = 0.0
    system_prompt_path: str = ""
    host: str | None = None
    timeout_seconds: float = 120.0

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("Agent model name cannot be empty.")
        if self.temperature < 0:
            raise ValueError("Agent temperature cannot be negative.")
        if self.timeout_seconds <= 0:
            raise ValueError("Agent timeout must be positive.")


@dataclass(slots=True)
class VisionConfig:
    resize_images: bool = False
    image_width: int = 640
    image_height: int = 480
    image_jpeg_quality: int = 85

    def __post_init__(self) -> None:
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("Vision dimensions must be positive.")
        if not 1 <= self.image_jpeg_quality <= 95:
            raise ValueError("Vision JPEG quality must be between 1 and 95.")


@dataclass(slots=True)
class PrefMemConfig:
    workspace_root: str = str(WORKSPACE_ROOT)
    dataset_path: str = str(WORKSPACE_ROOT / "dataset" / "v3")
    history_store_path: str = str(WORKSPACE_ROOT / "memory" / "history.json")
    history_outbox_path: str | None = None
    preference_store_path: str = str(WORKSPACE_ROOT / "memory" / "preferences.json")
    user_id: str = "default"
    max_replans: int = 1
    max_reobservations: int = 1
    recent_history_limit: int = 5
    history_semantic_scan_limit: int = 50
    semantic_history_limit: int = 3
    semantic_preference_limit: int = 5
    semantic_preference_threshold: float = 0.75
    preference_proposal_min_episodes: int = 2
    preference_proposal_confidence: float = 0.75
    compact_preferences_after_write: bool = True
    minimum_planner_confidence: float = 0.5
    minimum_validator_confidence: float = 0.5
    vision: VisionConfig = field(default_factory=VisionConfig)
    hri: AgentModelConfig = field(
        default_factory=lambda: AgentModelConfig(
            system_prompt_path=str(WORKSPACE_ROOT / "prompt" / "hri" / "prompt-v2.md")
        )
    )
    memory: AgentModelConfig = field(
        default_factory=lambda: AgentModelConfig(
            system_prompt_path=str(WORKSPACE_ROOT / "prompt" / "memory" / "prompt-v2.md")
        )
    )
    planner: AgentModelConfig = field(
        default_factory=lambda: AgentModelConfig(
            system_prompt_path=str(WORKSPACE_ROOT / "prompt" / "planner" / "prompt-v2.md")
        )
    )
    validator: AgentModelConfig = field(
        default_factory=lambda: AgentModelConfig(
            system_prompt_path=str(WORKSPACE_ROOT / "prompt" / "validator" / "prompt-v2.md")
        )
    )

    def __post_init__(self) -> None:
        if self.max_replans < 0 or self.max_reobservations < 0:
            raise ValueError("Recovery budgets cannot be negative.")
        for name in (
            "recent_history_limit",
            "history_semantic_scan_limit",
            "semantic_history_limit",
            "semantic_preference_limit",
            "preference_proposal_min_episodes",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive.")
        for name in (
            "semantic_preference_threshold",
            "preference_proposal_confidence",
            "minimum_planner_confidence",
            "minimum_validator_confidence",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1.")


# Compatibility names for callers that configure individual agents.
HRI_Agent_Config = PrefMemConfig
Memory_Agent_Config = AgentModelConfig
Planner_Agent_Config = AgentModelConfig
Validator_Agent_Config = AgentModelConfig
