"""Counterfactual task/outcome benchmark for PrefMem."""

from .ablations import MEMORY_MODES, MemoryAblationAgent, apply_memory_mode
from .catalog import FAMILY_DEFINITIONS, build_catalog, build_control_catalog
from .evaluation import (
    EvaluationConfig,
    EvaluationReport,
    run_cold_memory_evaluation,
)
from .evaluation_runtime import (
    build_evaluation_orchestrator,
    evaluation_orchestrator_factory,
)
from .executor import BenchmarkEpisodeExecutor
from .generator import GenerationReport, generate_benchmark
from .models import FamilyDefinition, ObjectState, PredicateSpec, ScenarioSpec
from .protocol_evaluation import (
    ProtocolBatchResult,
    ProtocolEvaluationError,
    evaluate_memory_protocols,
    load_protocol_bundle,
)
from .protocol_runner import ProtocolReport, ProtocolStepReport, run_memory_protocol
from .scorer import ScoreReport, score_agent_result
from .validator import ValidationReport, validate_benchmark

__all__ = [
    "FAMILY_DEFINITIONS",
    "MEMORY_MODES",
    "BenchmarkEpisodeExecutor",
    "EvaluationConfig",
    "EvaluationReport",
    "FamilyDefinition",
    "GenerationReport",
    "MemoryAblationAgent",
    "ObjectState",
    "PredicateSpec",
    "ProtocolBatchResult",
    "ProtocolEvaluationError",
    "ProtocolReport",
    "ProtocolStepReport",
    "ScenarioSpec",
    "ScoreReport",
    "ValidationReport",
    "apply_memory_mode",
    "build_catalog",
    "build_control_catalog",
    "build_evaluation_orchestrator",
    "evaluate_memory_protocols",
    "evaluation_orchestrator_factory",
    "generate_benchmark",
    "load_protocol_bundle",
    "run_cold_memory_evaluation",
    "score_agent_result",
    "run_memory_protocol",
    "validate_benchmark",
]
