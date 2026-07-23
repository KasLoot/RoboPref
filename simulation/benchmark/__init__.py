"""Counterfactual task/outcome benchmark for PrefMem."""

from .catalog import FAMILY_DEFINITIONS, build_catalog, build_control_catalog
from .executor import BenchmarkEpisodeExecutor
from .generator import GenerationReport, generate_benchmark
from .models import FamilyDefinition, ObjectState, PredicateSpec, ScenarioSpec
from .protocol_runner import ProtocolReport, ProtocolStepReport, run_memory_protocol
from .scorer import ScoreReport, score_agent_result
from .validator import ValidationReport, validate_benchmark

__all__ = [
    "FAMILY_DEFINITIONS",
    "BenchmarkEpisodeExecutor",
    "FamilyDefinition",
    "GenerationReport",
    "ObjectState",
    "PredicateSpec",
    "ProtocolReport",
    "ProtocolStepReport",
    "ScenarioSpec",
    "ScoreReport",
    "ValidationReport",
    "build_catalog",
    "build_control_catalog",
    "generate_benchmark",
    "score_agent_result",
    "run_memory_protocol",
    "validate_benchmark",
]
