"""Executable mechanism and baseline adapters for frozen experiment profiles."""

from __future__ import annotations

from dataclasses import dataclass
import threading
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence

from .profiles import ExperimentProfile


_MODE_VALUES = {
    "memory": frozenset(
        {
            "persistent_consent_store",
            "merged_interaction_memory",
            "actor_shared_context",
            "phase_moded_unified",
            "removed_empty_isolated_store",
            "absent",
            "oracle",
        }
    ),
    "hri": frozenset(
        {
            "clarify_and_propose",
            "merged_interaction_memory",
            "actor_shared_context",
            "phase_moded_unified",
            "raw_request_to_planner",
            "absent",
            "oracle",
        }
    ),
    "planning": frozenset(
        {
            "receding_horizon",
            "actor_receding_horizon",
            "phase_moded_unified",
            "frozen_complete_queue",
            "one_shot_complete_plan",
            "text_feedback_replanning",
            "hidden_state_oracle",
        }
    ),
    "monitor": frozenset(
        {
            "visual_step_assessment",
            "phase_moded_verifier",
            "critic_step_mode",
            "phase_moded_unified",
            "synthetic_executor_settled",
            "absent",
            "text_feedback_only",
            "hidden_state_oracle",
        }
    ),
    "validator": frozenset(
        {
            "independent_frozen_checklist",
            "phase_moded_verifier",
            "critic_final_mode",
            "phase_moded_unified",
            "planner_self_declare_complete",
            "absent",
            "hidden_state_oracle",
        }
    ),
    "confirmation": frozenset(
        {
            "exact_goal_revision",
            "minimal_host_confirmation",
            "auto_confirm_valid_preview",
            "none",
            "oracle",
        }
    ),
    "evidence": frozenset(
        {
            "temporally_repeated",
            "single_frame_zero_stability",
            "single_text_feedback",
            "none",
            "oracle",
        }
    ),
    "dynamic_guard": frozenset(
        {"live_membership", "membership_frozen_at_confirmation", "none", "oracle"}
    ),
    "host_fence": frozenset(
        {"trusted_identity", "accept_model_or_stale_identity"}
    ),
    "personalization": frozenset({"enabled", "disabled", "oracle"}),
}


class VariantConfigurationError(RuntimeError):
    """Raised when a profile cannot resolve to safe executable adapters."""


class VariantExecutionError(RuntimeError):
    """Raised when an adapter receives an invalid runtime payload."""


def validate_profile_modes(profile: ExperimentProfile, *, executor: str) -> None:
    """Reject unknown or unsafe modes before any construction side effect."""

    for mechanism, allowed in _MODE_VALUES.items():
        mode = profile.mechanism_modes.get(mechanism)
        if mode not in allowed:
            raise VariantConfigurationError(
                f"profile {profile.profile_id} has unsupported {mechanism} mode {mode!r}"
            )
    if (
        profile.mechanism_modes["confirmation"] == "auto_confirm_valid_preview"
        and executor not in {"synthetic_event", "mujoco"}
    ):
        raise VariantConfigurationError("automatic confirmation is simulation-only")
    if (
        profile.mechanism_modes["host_fence"] == "accept_model_or_stale_identity"
        and executor != "synthetic_event"
    ):
        raise VariantConfigurationError("untrusted host fence is synthetic-only")


class RoleInterface(Protocol):
    logical_role: str
    model_instance_id: str

    def invoke_phase(self, phase: str, payload: Any, **kwargs: Any) -> Any: ...


@dataclass(frozen=True, slots=True)
class BypassedRole:
    """Explicit non-runtime boundary for a role removed by a profile."""

    logical_role: str
    reason: str
    model_instance_id: None = None
    available: bool = False

    def invoke_phase(self, phase: str, payload: Any, **kwargs: Any) -> Any:
        del phase, payload, kwargs
        raise VariantExecutionError(
            f"logical role {self.logical_role} is bypassed: {self.reason}"
        )


def _active(components: Mapping[str, object], role: str) -> RoleInterface:
    component = components[role]
    if isinstance(component, BypassedRole):
        raise VariantConfigurationError(
            f"profile mode requires active {role}, but it is bypassed"
        )
    invoke = getattr(component, "invoke_phase", None)
    if not callable(invoke):
        raise VariantConfigurationError(
            f"component {role} does not implement invoke_phase()"
        )
    return component  # type: ignore[return-value]


def _payload_mapping(response: Any, *, context: str) -> Mapping[str, Any]:
    if isinstance(response, Mapping):
        return response
    to_dict = getattr(response, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, Mapping):
            return payload
    raise VariantExecutionError(f"{context} must return a mapping")


@dataclass(frozen=True, slots=True)
class MemoryMutationResult:
    applied: bool
    reason: str
    response: Any = None


class RoutedMemoryAdapter:
    mode = "routed"

    def __init__(self, role: RoleInterface) -> None:
        self.role = role

    def retrieve(self, query: str) -> Any:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("memory query must be non-empty")
        return self.role.invoke_phase("MEMORY_RETRIEVE", {"query": query.strip()})

    def mutate(
        self, operation: str, payload: Mapping[str, Any], *, authorized: bool
    ) -> MemoryMutationResult:
        if operation not in {"create", "update", "forget"}:
            raise ValueError("unsupported memory mutation")
        if not authorized:
            return MemoryMutationResult(False, "future_facing_consent_absent")
        response = self.role.invoke_phase(
            "MEMORY_MUTATE", {"operation": operation, "payload": dict(payload)}
        )
        return MemoryMutationResult(True, "authorized", response)


class DisabledMemoryAdapter:
    mode = "disabled_empty_isolated_store"

    def retrieve(self, query: str) -> tuple[()]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("memory query must be non-empty")
        return ()

    def mutate(
        self, operation: str, payload: Mapping[str, Any], *, authorized: bool
    ) -> MemoryMutationResult:
        del operation, payload, authorized
        return MemoryMutationResult(False, "memory_unavailable")


class OracleMemoryAdapter:
    mode = "hidden_state_oracle"

    def retrieve(self, query: str, *, hidden_state: Mapping[str, Any]) -> Any:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("memory query must be non-empty")
        return tuple(hidden_state.get("oracle_preferences", ()))

    def mutate(
        self, operation: str, payload: Mapping[str, Any], *, authorized: bool
    ) -> MemoryMutationResult:
        del operation, payload
        return MemoryMutationResult(authorized, "oracle_consent_policy")


class RoutedInteractionAdapter:
    mode = "hri"

    def __init__(self, role: RoleInterface) -> None:
        self.role = role

    def preview(self, request: str, scene: Any) -> Any:
        return self.role.invoke_phase("PREVIEW", {"request": request, "scene": scene})


class DirectPlannerInteractionAdapter:
    mode = "no_hri_direct_preview"

    def __init__(self, planner: RoleInterface) -> None:
        self.planner = planner

    def preview(self, request: str, scene: Any) -> Any:
        return self.planner.invoke_phase(
            "DIRECT_PREVIEW", {"raw_user_request": request, "scene": scene}
        )


class OracleInteractionAdapter:
    mode = "hidden_state_oracle"

    def preview(self, request: str, scene: Any) -> Any:
        del request
        if not isinstance(scene, Mapping) or "oracle_preview" not in scene:
            raise VariantExecutionError("oracle interaction needs scene.oracle_preview")
        return scene["oracle_preview"]


class RecedingHorizonPlanningAdapter:
    mode = "receding_horizon"

    def __init__(self, planner: RoleInterface) -> None:
        self.planner = planner

    def next_action(self, *, goal: Any, observation: Any, history: Sequence[Any]) -> Any:
        return self.planner.invoke_phase(
            "PLAN",
            {"goal": goal, "observation": observation, "history": list(history)},
        )


class FrozenOpenLoopPlanningAdapter:
    """Plan once, freeze the queue, and ignore all later observations."""

    mode = "frozen_open_loop_queue"

    def __init__(self, planner: RoleInterface) -> None:
        self.planner = planner
        self._lock = threading.RLock()
        self._queue: tuple[Any, ...] | None = None
        self._cursor = 0

    @property
    def frozen_queue(self) -> tuple[Any, ...] | None:
        with self._lock:
            return self._queue

    def freeze(self, actions: Sequence[Any]) -> tuple[Any, ...]:
        if isinstance(actions, (str, bytes)) or not actions:
            raise VariantExecutionError("open-loop queue must contain at least one action")
        with self._lock:
            if self._queue is not None:
                raise VariantExecutionError("open-loop queue is already frozen")
            self._queue = tuple(actions)
            return self._queue

    def plan_once(self, *, goal: Any, scene: Any) -> tuple[Any, ...]:
        with self._lock:
            if self._queue is not None:
                raise VariantExecutionError("open-loop queue is already frozen")
            response = self.planner.invoke_phase(
                "PLAN_OPEN_LOOP", {"goal": goal, "scene": scene}
            )
            payload = _payload_mapping(response, context="open-loop Planner")
            actions = payload.get("actions")
            if not isinstance(actions, Sequence) or isinstance(actions, (str, bytes)):
                raise VariantExecutionError("open-loop Planner response needs actions[]")
            return self.freeze(actions)

    def next_action(self, *, observation: Any = None) -> Any | None:
        del observation
        with self._lock:
            if self._queue is None:
                raise VariantExecutionError("open-loop queue has not been planned")
            if self._cursor >= len(self._queue):
                return None
            action = self._queue[self._cursor]
            self._cursor += 1
            return action


class FeedbackPlanningAdapter:
    mode = "text_feedback_replanning"

    def __init__(self, planner: RoleInterface) -> None:
        self.planner = planner

    def next_action(
        self,
        *,
        instruction: str,
        observation: Any,
        previous_action: Any,
        textual_feedback: str,
        history: Sequence[Any],
    ) -> Any:
        return self.planner.invoke_phase(
            "FEEDBACK_PLAN",
            {
                "instruction": instruction,
                "observation": observation,
                "previous_action": previous_action,
                "textual_feedback": textual_feedback,
                "history": list(history),
            },
        )


class OraclePlanningAdapter:
    mode = "hidden_state_oracle"

    def __init__(self) -> None:
        self._cursor = 0
        self._lock = threading.Lock()

    def next_action(self, *, hidden_state: Mapping[str, Any]) -> Any | None:
        actions = hidden_state.get("oracle_actions")
        if not isinstance(actions, Sequence) or isinstance(actions, (str, bytes)):
            raise VariantExecutionError("hidden_state.oracle_actions must be a sequence")
        with self._lock:
            if self._cursor >= len(actions):
                return None
            action = actions[self._cursor]
            self._cursor += 1
            return action


@dataclass(frozen=True, slots=True)
class StepAssessment:
    status: str
    source: str
    detail: str


class VisualMonitorAdapter:
    mode = "visual_step_assessment"

    def __init__(self, monitor: RoleInterface) -> None:
        self.monitor = monitor

    def assess(self, published_task: Any, frame: Any) -> Any:
        return self.monitor.invoke_phase(
            "STEP_ASSESS", {"published_task": published_task, "frame": frame}
        )


class SyntheticSettledMonitorAdapter:
    mode = "synthetic_executor_settled"

    def assess(self, executor_event: Any) -> StepAssessment:
        state = (
            executor_event.get("state")
            if isinstance(executor_event, Mapping)
            else getattr(executor_event, "state", None)
        )
        state = getattr(state, "value", state)
        if state == "SETTLED":
            return StepAssessment("MET", self.mode, "authoritative simulated settled event")
        if state == "FAULT":
            return StepAssessment("SYSTEM_ERROR", self.mode, "executor fault is not evidence")
        return StepAssessment("UNKNOWN", self.mode, f"nonterminal executor state {state!r}")


class DisabledMonitorAdapter:
    mode = "disabled"

    def assess(self, value: Any) -> StepAssessment:
        del value
        return StepAssessment("UNOBSERVED", self.mode, "profile receives no step assessment")


class TextFeedbackMonitorAdapter:
    mode = "text_feedback_only"

    def assess(self, feedback: str) -> StepAssessment:
        normalized = feedback.strip().lower()
        if normalized in {"success", "met", "settled"}:
            status = "MET"
        elif normalized in {"failure", "not_met", "fault"}:
            status = "NOT_MET"
        else:
            status = "UNKNOWN"
        return StepAssessment(status, self.mode, feedback)


class OracleMonitorAdapter:
    mode = "hidden_state_oracle"

    def assess(self, hidden_state: Mapping[str, Any]) -> StepAssessment:
        if "step_success" not in hidden_state:
            raise VariantExecutionError("oracle monitor needs hidden step_success")
        return StepAssessment(
            "MET" if hidden_state["step_success"] is True else "NOT_MET",
            self.mode,
            "hidden physical predicate",
        )


@dataclass(frozen=True, slots=True)
class ValidationDecision:
    complete: bool | None
    source: str
    response: Any = None


class IndependentValidatorAdapter:
    mode = "independent_frozen_checklist"

    def __init__(self, validator: RoleInterface) -> None:
        self.validator = validator

    def compile(self, goal_contract: Any, frame: Any) -> Any:
        return self.validator.invoke_phase(
            "COMPILE_VALIDATION", {"goal_contract": goal_contract, "frame": frame}
        )

    def assess(self, validation_contract: Any, frame: Any) -> ValidationDecision:
        response = self.validator.invoke_phase(
            "FINAL_ASSESS",
            {"validation_contract": validation_contract, "frame": frame},
        )
        payload = _payload_mapping(response, context="Validator")
        complete = payload.get("complete")
        if type(complete) is not bool:
            raise VariantExecutionError("Validator response needs boolean complete")
        return ValidationDecision(complete, self.mode, response)


class PlannerSelfCompletionAdapter:
    mode = "planner_self_declare_complete"

    def __init__(self, planner: RoleInterface) -> None:
        self.planner = planner

    def compile(self, goal_contract: Any, frame: Any) -> Mapping[str, Any]:
        return MappingProxyType({"goal_contract": goal_contract, "confirmation_frame": frame})

    def assess(self, validation_contract: Any, frame: Any) -> ValidationDecision:
        response = self.planner.invoke_phase(
            "DECLARE_COMPLETE",
            {"validation_contract": validation_contract, "fresh_final_frame": frame},
        )
        payload = _payload_mapping(response, context="Planner self-completion")
        complete = payload.get("complete")
        if type(complete) is not bool:
            raise VariantExecutionError("Planner self-completion needs boolean complete")
        return ValidationDecision(complete, self.mode, response)


class DisabledValidatorAdapter:
    mode = "disabled"

    def compile(self, goal_contract: Any, frame: Any) -> None:
        del goal_contract, frame
        return None

    def assess(self, validation_contract: Any, frame: Any) -> ValidationDecision:
        del validation_contract, frame
        return ValidationDecision(None, self.mode, None)


class OracleValidatorAdapter:
    mode = "hidden_state_oracle"

    def compile(self, goal_contract: Any, frame: Any) -> Any:
        del frame
        return goal_contract

    def assess(self, hidden_state: Mapping[str, Any], frame: Any = None) -> ValidationDecision:
        del frame
        complete = hidden_state.get("goal_completed")
        if type(complete) is not bool:
            raise VariantExecutionError("oracle validator needs hidden goal_completed")
        return ValidationDecision(complete, self.mode, hidden_state)


@dataclass(frozen=True, slots=True)
class ConfirmationPolicy:
    mode: str

    def authorize(
        self,
        *,
        preview_status: str,
        staged_goal_id: str,
        staged_revision: int,
        confirmed_goal_id: str | None = None,
        confirmed_revision: int | None = None,
        user_confirmed: bool = False,
        oracle_authorized: bool | None = None,
    ) -> bool:
        ready = preview_status in {"READY", "ALREADY_SATISFIED"}
        if self.mode == "exact_goal_revision":
            return (
                ready
                and user_confirmed is True
                and confirmed_goal_id == staged_goal_id
                and confirmed_revision == staged_revision
            )
        if self.mode == "auto_confirm_valid_preview":
            return ready
        if self.mode == "none":
            return ready
        if self.mode == "minimal_host_confirmation":
            return (
                ready
                and user_confirmed is True
                and confirmed_goal_id == staged_goal_id
                and confirmed_revision == staged_revision
            )
        if self.mode == "oracle":
            return ready and oracle_authorized is True
        raise VariantExecutionError(f"unknown confirmation mode: {self.mode}")


@dataclass(frozen=True, slots=True)
class EvidencePolicy:
    mode: str
    success_confirmations: int
    failure_confirmations: int
    success_stability_seconds: float
    enabled: bool = True

    def terminal(
        self,
        label: str,
        *,
        consecutive_count: int,
        stable_seconds: float,
        oracle_terminal: bool | None = None,
    ) -> bool:
        if self.mode == "oracle":
            return oracle_terminal is True
        if not self.enabled:
            return False
        if label == "MET":
            return (
                consecutive_count >= self.success_confirmations
                and stable_seconds >= self.success_stability_seconds
            )
        if label == "NOT_MET":
            return consecutive_count >= self.failure_confirmations
        return False


@dataclass(frozen=True, slots=True)
class GuardDecision:
    cancel_active_motion: bool
    effective_members: frozenset[str]
    new_members: frozenset[str]
    mode: str


@dataclass(frozen=True, slots=True)
class DynamicGuardPolicy:
    mode: str

    def evaluate(
        self,
        *,
        confirmed_members: frozenset[str],
        current_members: frozenset[str],
        motion_active: bool,
        oracle_cancel: bool | None = None,
    ) -> GuardDecision:
        new_members = current_members - confirmed_members
        if self.mode == "live_membership":
            return GuardDecision(
                bool(new_members and motion_active), current_members, new_members, self.mode
            )
        if self.mode == "membership_frozen_at_confirmation":
            return GuardDecision(False, confirmed_members, new_members, self.mode)
        if self.mode == "none":
            return GuardDecision(False, current_members, new_members, self.mode)
        if self.mode == "oracle":
            return GuardDecision(
                oracle_cancel is True, current_members, new_members, self.mode
            )
        raise VariantExecutionError(f"unknown dynamic guard mode: {self.mode}")


@dataclass(frozen=True, slots=True)
class HostFencePolicy:
    mode: str

    def accepts(
        self,
        report: Mapping[str, Any],
        *,
        active_publication_id: str,
        active_step_id: str,
        active_criterion_ids: frozenset[str],
    ) -> bool:
        if self.mode == "untrusted_identity":
            return True
        if self.mode != "trusted_identity":
            raise VariantExecutionError(f"unknown host-fence mode: {self.mode}")
        criterion_id = report.get("criterion_id")
        return (
            report.get("publication_id") == active_publication_id
            and report.get("step_id") == active_step_id
            and isinstance(criterion_id, str)
            and criterion_id in active_criterion_ids
        )


@dataclass(frozen=True, slots=True)
class VariantRuntime:
    profile_id: str
    executor: str
    memory: Any
    interaction: Any
    planning: Any
    monitor: Any
    validator: Any
    confirmation: ConfirmationPolicy
    evidence: EvidencePolicy
    dynamic_guard: DynamicGuardPolicy
    host_fence: HostFencePolicy
    personalization_enabled: bool
    components: Mapping[str, object]


def _confirmation(mode: str, executor: str) -> ConfirmationPolicy:
    if mode == "auto_confirm_valid_preview":
        if executor not in {"synthetic_event", "mujoco"}:
            raise VariantConfigurationError("automatic confirmation is simulation-only")
        return ConfirmationPolicy("auto_confirm_valid_preview")
    if mode in {"exact_goal_revision", "minimal_host_confirmation", "none"}:
        return ConfirmationPolicy(mode)
    if mode == "oracle":
        return ConfirmationPolicy("oracle")
    raise VariantConfigurationError(f"unsupported confirmation mode: {mode}")


def _evidence(mode: str) -> EvidencePolicy:
    if mode == "single_frame_zero_stability":
        return EvidencePolicy(mode, 1, 1, 0.0)
    if mode in {"none"}:
        return EvidencePolicy(mode, 0, 0, 0.0, enabled=False)
    if mode == "oracle":
        return EvidencePolicy(mode, 0, 0, 0.0)
    if mode == "single_text_feedback":
        return EvidencePolicy(mode, 1, 1, 0.0)
    return EvidencePolicy("temporally_repeated", 2, 2, 2.0)


def build_variant_runtime(
    profile: ExperimentProfile,
    *,
    executor: str,
    components: Mapping[str, object],
) -> VariantRuntime:
    """Resolve every profile mode to concrete, side-effect-free driver adapters."""

    validate_profile_modes(profile, executor=executor)
    modes = profile.mechanism_modes

    memory_mode = modes["memory"]
    if memory_mode in {"removed_empty_isolated_store", "absent"}:
        memory: Any = DisabledMemoryAdapter()
    elif memory_mode == "oracle":
        memory = OracleMemoryAdapter()
    elif memory_mode in {
        "persistent_consent_store",
        "merged_interaction_memory",
        "actor_shared_context",
        "phase_moded_unified",
    }:
        memory = RoutedMemoryAdapter(_active(components, "memory"))
    else:
        raise VariantConfigurationError(f"unsupported memory mode: {memory_mode}")

    hri_mode = modes["hri"]
    if hri_mode in {"raw_request_to_planner", "absent"}:
        interaction: Any = DirectPlannerInteractionAdapter(
            _active(components, "planner")
        )
    elif hri_mode == "oracle":
        interaction = OracleInteractionAdapter()
    elif hri_mode in {
        "clarify_and_propose",
        "merged_interaction_memory",
        "actor_shared_context",
        "phase_moded_unified",
    }:
        interaction = RoutedInteractionAdapter(_active(components, "hri"))
    else:
        raise VariantConfigurationError(f"unsupported HRI mode: {hri_mode}")

    planning_mode = modes["planning"]
    if planning_mode in {"frozen_complete_queue", "one_shot_complete_plan"}:
        planning: Any = FrozenOpenLoopPlanningAdapter(
            _active(components, "planner")
        )
    elif planning_mode == "text_feedback_replanning":
        planning = FeedbackPlanningAdapter(_active(components, "planner"))
    elif planning_mode == "hidden_state_oracle":
        planning = OraclePlanningAdapter()
    elif planning_mode in {
        "receding_horizon",
        "actor_receding_horizon",
        "phase_moded_unified",
    }:
        planning = RecedingHorizonPlanningAdapter(_active(components, "planner"))
    else:
        raise VariantConfigurationError(f"unsupported planning mode: {planning_mode}")

    monitor_mode = modes["monitor"]
    if monitor_mode == "synthetic_executor_settled":
        monitor: Any = SyntheticSettledMonitorAdapter()
    elif monitor_mode == "absent":
        monitor = DisabledMonitorAdapter()
    elif monitor_mode == "text_feedback_only":
        monitor = TextFeedbackMonitorAdapter()
    elif monitor_mode == "hidden_state_oracle":
        monitor = OracleMonitorAdapter()
    elif monitor_mode in {
        "visual_step_assessment",
        "phase_moded_verifier",
        "critic_step_mode",
        "phase_moded_unified",
    }:
        monitor = VisualMonitorAdapter(_active(components, "monitor"))
    else:
        raise VariantConfigurationError(f"unsupported monitor mode: {monitor_mode}")

    validator_mode = modes["validator"]
    if validator_mode == "planner_self_declare_complete":
        validator: Any = PlannerSelfCompletionAdapter(
            _active(components, "planner")
        )
    elif validator_mode == "absent":
        validator = DisabledValidatorAdapter()
    elif validator_mode == "hidden_state_oracle":
        validator = OracleValidatorAdapter()
    elif validator_mode in {
        "independent_frozen_checklist",
        "phase_moded_verifier",
        "critic_final_mode",
        "phase_moded_unified",
    }:
        validator = IndependentValidatorAdapter(_active(components, "validator"))
    else:
        raise VariantConfigurationError(f"unsupported validator mode: {validator_mode}")

    dynamic_mode = modes["dynamic_guard"]
    if dynamic_mode == "membership_frozen_at_confirmation":
        dynamic_guard = DynamicGuardPolicy(dynamic_mode)
    elif dynamic_mode in {"none", "oracle"}:
        dynamic_guard = DynamicGuardPolicy(dynamic_mode)
    elif dynamic_mode == "live_membership":
        dynamic_guard = DynamicGuardPolicy("live_membership")
    else:
        raise VariantConfigurationError(f"unsupported dynamic guard mode: {dynamic_mode}")

    fence_mode = modes["host_fence"]
    if fence_mode == "accept_model_or_stale_identity":
        if executor != "synthetic_event":
            raise VariantConfigurationError("untrusted host fence is synthetic-only")
        host_fence = HostFencePolicy("untrusted_identity")
    elif fence_mode == "trusted_identity":
        host_fence = HostFencePolicy("trusted_identity")
    else:
        raise VariantConfigurationError(f"unsupported host-fence mode: {fence_mode}")

    return VariantRuntime(
        profile_id=profile.profile_id,
        executor=executor,
        memory=memory,
        interaction=interaction,
        planning=planning,
        monitor=monitor,
        validator=validator,
        confirmation=_confirmation(modes["confirmation"], executor),
        evidence=_evidence(modes["evidence"]),
        dynamic_guard=dynamic_guard,
        host_fence=host_fence,
        personalization_enabled=modes["personalization"] not in {
            "disabled",
            "absent",
        },
        components=components,
    )
