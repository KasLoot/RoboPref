"""Counterfactual endpoint and conversation benchmark for PrefMem."""

from .ablations import MEMORY_MODES, MemoryAblationAgent, apply_memory_mode
from .catalog import FAMILY_DEFINITIONS, build_catalog, build_control_catalog
from .conversation_cases import (
    build_conversation_cases,
    case_matrix,
    load_episode_catalog,
)
from .conversation_evaluation import (
    ConversationEvaluationError,
    evaluate_conversations,
)
from .conversation_metrics import build_conversation_summary
from .conversation_models import (
    CheckResult,
    CommandSpec,
    ConversationBatchReport,
    ConversationCase,
    ConversationEvaluationConfig,
    EpisodeDescriptor,
)
from .conversation_runner import run_conversation_case
from .conversation_scoring import score_conversation_run
from .executor import BenchmarkEpisodeExecutor, CounterfactualSequenceExecutor
from .generator import GenerationReport, generate_benchmark
from .models import FamilyDefinition, ObjectState, PredicateSpec, ScenarioSpec
from .semantic import (
    CanonicalPredicate,
    PredicateDiff,
    canonical_goal_predicates,
    infer_target_id,
    planner_predicate_diff,
)
from .validator import ValidationReport, validate_benchmark

__all__ = [
    "FAMILY_DEFINITIONS",
    "MEMORY_MODES",
    "BenchmarkEpisodeExecutor",
    "CanonicalPredicate",
    "CheckResult",
    "CommandSpec",
    "ConversationBatchReport",
    "ConversationCase",
    "ConversationEvaluationConfig",
    "ConversationEvaluationError",
    "CounterfactualSequenceExecutor",
    "EpisodeDescriptor",
    "FamilyDefinition",
    "GenerationReport",
    "MemoryAblationAgent",
    "ObjectState",
    "PredicateDiff",
    "PredicateSpec",
    "ScenarioSpec",
    "ValidationReport",
    "apply_memory_mode",
    "build_catalog",
    "build_control_catalog",
    "build_conversation_cases",
    "build_conversation_summary",
    "canonical_goal_predicates",
    "case_matrix",
    "evaluate_conversations",
    "generate_benchmark",
    "infer_target_id",
    "load_episode_catalog",
    "planner_predicate_diff",
    "run_conversation_case",
    "score_conversation_run",
    "validate_benchmark",
]
