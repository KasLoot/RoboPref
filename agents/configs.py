from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]


def normalize_model_base_url(value: str) -> str:
    """Validate an HTTP(S) API root without persisting embedded credentials."""

    normalized = str(value).strip().rstrip("/")
    if not normalized:
        raise ValueError("Agent model base URL cannot be empty.")
    if any(character.isspace() for character in normalized):
        raise ValueError("Agent model base URL cannot contain whitespace.")
    try:
        parsed = urlsplit(normalized)
    except ValueError:
        raise ValueError(
            "Agent model base URL must contain a valid host and port."
        ) from None
    try:
        hostname = parsed.hostname
        _ = parsed.port
    except ValueError:
        raise ValueError(
            "Agent model base URL must contain a valid host and port."
        ) from None
    if hostname is None:
        raise ValueError(
            "Agent model base URL must contain a valid host and port."
        )
    if parsed.scheme.casefold() not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Agent model base URL must be an absolute HTTP(S) URL.")
    if (
        parsed.username is not None
        or parsed.password is not None
        or "?" in normalized
        or "#" in normalized
    ):
        raise ValueError(
            "Agent model base URL must not contain credentials, query parameters, or a fragment."
        )
    if not parsed.path.endswith("/v1"):
        raise ValueError(
            "Agent model base URL must be the API root ending in '/v1'."
        )
    return normalized


@dataclass(slots=True)
class AgentModelConfig:
    model: str = "gemma4:31b-cloud"
    temperature: float = 0.0
    system_prompt_path: str = ""
    host: str | None = None
    timeout_seconds: float = 120.0
    seed: int | None = None
    provider: str = "ollama"
    base_url: str | None = None

    def __post_init__(self) -> None:
        self.provider = str(self.provider).strip().casefold()
        if self.provider not in {"ollama", "vllm"}:
            raise ValueError("Agent model provider must be 'ollama' or 'vllm'.")
        if not self.model.strip():
            raise ValueError("Agent model name cannot be empty.")
        if self.base_url is not None:
            if self.provider != "vllm":
                raise ValueError(
                    "Agent model base URL requires provider='vllm'."
                )
            self.base_url = normalize_model_base_url(self.base_url)
        if self.host is not None and self.provider != "ollama":
            raise ValueError("Agent Ollama host requires provider='ollama'.")
        if self.temperature < 0:
            raise ValueError("Agent temperature cannot be negative.")
        if self.timeout_seconds <= 0:
            raise ValueError("Agent timeout must be positive.")
        if self.seed is not None and (
            not isinstance(self.seed, int)
            or isinstance(self.seed, bool)
            or self.seed < 0
        ):
            raise ValueError("Agent seed must be a non-negative integer or null.")


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
