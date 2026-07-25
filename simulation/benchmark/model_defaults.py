"""Model defaults used only by the batch evaluation entry points."""

from __future__ import annotations

from agents.configs import PrefMemConfig


DEFAULT_EVALUATION_PROVIDER = "vllm"
DEFAULT_EVALUATION_MODEL = "/workspace/models/gemma-4-26B-A4B-it"


def apply_evaluation_model_defaults(config: PrefMemConfig) -> PrefMemConfig:
    """Configure every model-backed evaluation agent for local vLLM."""

    for model_config in (
        config.hri,
        config.memory,
        config.planner,
        config.validator,
    ):
        model_config.provider = DEFAULT_EVALUATION_PROVIDER
        model_config.model = DEFAULT_EVALUATION_MODEL
    return config


def default_evaluation_prefmem_config() -> PrefMemConfig:
    """Return a fresh PrefMem config with the evaluation backend defaults."""

    return apply_evaluation_model_defaults(PrefMemConfig())


__all__ = [
    "DEFAULT_EVALUATION_MODEL",
    "DEFAULT_EVALUATION_PROVIDER",
    "apply_evaluation_model_defaults",
    "default_evaluation_prefmem_config",
]
