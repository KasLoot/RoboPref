"""Deterministic event-driven controller for the PrefMem agent system."""

from __future__ import annotations

import re
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from prefmem.agents.contracts import (
    ConditionCheck,
    ConditionState,
    EpisodeRecord,
    ExecutionCommand,
    FailureKind,
    HRIDecisionKind,
    InteractionKind,
    MemoryActionKind,
    MemoryContext,
    MemoryStatus,
    MonitorAction,
    MonitorProgress,
    MonitorRequest,
    MonitorResult,
    MonitorSafetyStatus,
    ObservationQuality,
    PlanResult,
    PlanningRequest,
    PlanningStatus,
    ReplanPolicy,
    TaskContract,
    TaskPhase,
    TaskStatus,
    ValidationRequest,
    ValidationResult,
    ValidatorOutcome,
    ValidatorRecoverability,
)
from prefmem.agents.services import (
    HRIReasoner,
    MonitorReasoner,
    PlannerReasoner,
    ValidatorReasoner,
)
from prefmem.execution import CancellationReceipt, IdempotentVLAAdapter
from prefmem.memory.service import PrefMemMemoryService
from prefmem.observations import ObservationEnvelope, ObservationSource
from prefmem.recording import ExperimentRecorder, NullRecorder


@dataclass(frozen=True, slots=True)
class ControllerLimits:
    hri_routes_per_turn: int = 4
    max_monitor_checks: int = 30
    max_reobservations: int = 3
    max_replans: int = 2
    monitor_interval_seconds: float = 1.0
    monitor_window: int = 3
    success_confirmations: int = 2
    max_dispatches: int = 32

    def __post_init__(self) -> None:
        if self.hri_routes_per_turn < 1:
            raise ValueError("hri_routes_per_turn must be positive")
        if self.max_monitor_checks < 1:
            raise ValueError("max_monitor_checks must be positive")
        if self.max_reobservations < 0 or self.max_replans < 0:
            raise ValueError("recovery budgets cannot be negative")
        if self.monitor_interval_seconds < 0:
            raise ValueError("monitor_interval_seconds cannot be negative")
        if self.monitor_window < 1:
            raise ValueError("monitor_window must be positive")
        if self.success_confirmations < 1:
            raise ValueError("success_confirmations must be positive")
        if self.max_dispatches < 1:
            raise ValueError("max_dispatches must be positive")


@dataclass(slots=True)
class TurnResult:
    text: str
    phase: TaskPhase
    task_id: str | None = None
    plan: PlanResult | None = None
    validation: ValidationResult | None = None
    memory: MemoryContext | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class _ExecutionState:
    task_id: str
    task_contract: TaskContract
    planning_request: PlanningRequest
    plan: PlanResult
    completed_subtasks: list[str] = field(default_factory=list)
    dispatches: list[dict[str, Any]] = field(default_factory=list)
    monitor_results: list[MonitorResult] = field(default_factory=list)
    replans: int = 0
    boundary_replans: int = 0
    reobservations: int = 0


_REMEMBER_COMMAND = re.compile(r"^/remember\s+(.+)$", re.IGNORECASE | re.DOTALL)
_FORGET_COMMAND = re.compile(r"^/forget\s+(\S+)\s*$", re.IGNORECASE)
_EXPLICIT_NATURAL_MEMORY = re.compile(
    r"^(?:please\s+)?remember(?:\s+that)?\s+"
    r"(?P<statement>.+(?:\bprefer\b|\bpreference\b|\bdefault\b|\balways\b|"
    r"\bfor future\b).*)$",
    re.IGNORECASE | re.DOTALL,
)


class PrefMemController:
    """Own all state transitions, budgets, memory writes, and publication."""

    def __init__(
        self,
        *,
        username: str,
        observation_source: ObservationSource,
        hri: HRIReasoner,
        planner: PlannerReasoner,
        monitor: MonitorReasoner,
        validator: ValidatorReasoner,
        memory: PrefMemMemoryService,
        vla: IdempotentVLAAdapter | None = None,
        plan_only: bool = True,
        replan_policy: ReplanPolicy = ReplanPolicy.ON_DEVIATION,
        limits: ControllerLimits | None = None,
        recorder: ExperimentRecorder | None = None,
        wait: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.username = username
        self.observation_source = observation_source
        self.hri = hri
        self.planner = planner
        self.monitor = monitor
        self.validator = validator
        self.memory = memory
        self.vla = vla
        self.plan_only = plan_only
        self.replan_policy = replan_policy
        self.limits = limits or ControllerLimits()
        self.recorder = recorder or NullRecorder()
        self.wait = wait
        self.clock = clock
        self.phase = TaskPhase.IDLE
        self.conversation: list[dict[str, str]] = []
        self._pending_memory_consent: dict[str, Any] | None = None
        self._active_memory_consent: dict[str, Any] | None = None

    def _transition(
        self,
        phase: TaskPhase,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        previous = self.phase
        self.phase = phase
        self.recorder.record_event(
            "Controller transition",
            {
                "from": previous,
                "to": phase,
                "details": dict(details or {}),
            },
        )

    def _reply(
        self,
        text: str,
        *,
        phase: TaskPhase,
        **result_fields: Any,
    ) -> TurnResult:
        self._transition(phase)
        self.conversation.append({"role": "assistant", "content": text})
        self.recorder.record_assistant(text)
        return TurnResult(text=text, phase=phase, **result_fields)

    def _handle_memory_command(self, user_text: str) -> TurnResult | None:
        remember = _REMEMBER_COMMAND.match(user_text)
        if remember:
            statement = remember.group(1).strip()
            preference = self.memory.remember_preference(
                username=self.username,
                statement=statement,
                authorized=True,
                evidence=[{"user_utterance": user_text}],
            )
            return self._reply(
                (
                    f"I saved that future preference as `{preference.id}`: "
                    f"{preference.statement}"
                ),
                phase=TaskPhase.COMPLETE,
                metadata={"preference_id": preference.id},
            )

        forget = _FORGET_COMMAND.match(user_text)
        if forget:
            preference = self.memory.revoke_preference(
                username=self.username,
                preference_id=forget.group(1),
                authorized=True,
                evidence=[{"user_utterance": user_text}],
            )
            return self._reply(
                f"I revoked preference `{preference.id}`.",
                phase=TaskPhase.COMPLETE,
                metadata={"preference_id": preference.id},
            )

        if user_text.strip().lower() == "/preferences":
            preferences = self.memory.store.load_preferences(
                username=self.username
            )
            if not preferences:
                text = "You do not have any active saved preferences."
            else:
                lines = [
                    f"- `{record.id}`: {record.statement}"
                    for record in preferences
                ]
                text = "Your active saved preferences are:\n\n" + "\n".join(lines)
            return self._reply(text, phase=TaskPhase.COMPLETE)
        return None

    def _memory_action_authorized(
        self,
        *,
        user_text: str,
        decision: Any,
    ) -> bool:
        action = decision.memory_action.action
        lowered = user_text.casefold()
        if self._active_memory_consent is not None and lowered in {
            "yes",
            "yes.",
            "y",
            "confirm",
            "please do",
            "do that",
        }:
            proposal = self._active_memory_consent.get("proposal")
            if not isinstance(proposal, Mapping):
                return False
            expected_action = str(
                proposal.get("action", MemoryActionKind.REMEMBER.value)
            ).upper()
            expected_statement = proposal.get("statement")
            expected_scope = proposal.get("scope", {})
            expected_value = proposal.get("structured_value", {})
            expected_target = proposal.get("target_record_id")
            actual_statement = decision.memory_action.statement
            if (
                expected_action != action.value
                or not isinstance(expected_statement, str)
                or actual_statement is None
                or " ".join(expected_statement.split())
                != " ".join(actual_statement.split())
                or expected_scope != decision.memory_action.scope
                or expected_value != decision.memory_action.structured_value
            ):
                return False
            if expected_target is not None and decision.memory_refs_used != [
                expected_target
            ]:
                return False
            return True
        if action in {MemoryActionKind.REMEMBER, MemoryActionKind.UPDATE}:
            return bool(
                _EXPLICIT_NATURAL_MEMORY.match(user_text)
                or re.search(
                    r"\b(from now on|make (?:that|this) my default|"
                    r"save (?:that|this) (?:as )?(?:my )?preference|"
                    r"update my preference)\b",
                    lowered,
                )
            )
        if action == MemoryActionKind.FORGET:
            return bool(
                re.search(
                    r"\b(forget|delete|remove|stop remembering)\b",
                    lowered,
                )
            )
        return False

    def _commit_hri_memory_action(
        self,
        *,
        decision: Any,
        user_text: str,
    ) -> str | None:
        action = decision.memory_action.action
        if action == MemoryActionKind.NONE:
            return None
        authorized = self._memory_action_authorized(
            user_text=user_text,
            decision=decision,
        )
        if not authorized:
            message = (
                "The proposed durable memory change was not committed because "
                "the current user message did not explicitly authorize it."
            )
            self.recorder.record_event(
                "Preference mutation rejected",
                {
                    "action": decision.memory_action,
                    "user_text": user_text,
                    "reason": "missing explicit authorization",
                },
            )
            return message

        statement = decision.memory_action.statement
        assert statement is not None
        scope_payload = dict(decision.memory_action.scope)
        scope = str(scope_payload.pop("kind", "contextual"))
        evidence = [
            {
                "user_utterance": user_text,
                "hri_reason_code": decision.reason_code,
            }
        ]
        try:
            if action == MemoryActionKind.REMEMBER:
                record = self.memory.remember_preference(
                    username=self.username,
                    statement=statement,
                    authorized=True,
                    scope=scope,
                    applicability=scope_payload,
                    structured_value=decision.memory_action.structured_value,
                    evidence=evidence,
                )
                note = f"Saved preference `{record.id}`."
            else:
                referenced = [
                    record_id
                    for record_id in decision.memory_refs_used
                    if record_id.startswith("pref-")
                ]
                if len(referenced) != 1:
                    raise ValueError(
                        f"{action.value} requires exactly one retrieved preference ID"
                    )
                if action == MemoryActionKind.UPDATE:
                    record = self.memory.update_preference(
                        username=self.username,
                        preference_id=referenced[0],
                        statement=statement,
                        authorized=True,
                        scope=scope,
                        applicability=scope_payload,
                        structured_value=decision.memory_action.structured_value,
                        evidence=evidence,
                    )
                    note = (
                        f"Updated preference `{referenced[0]}` as `{record.id}`."
                    )
                else:
                    record = self.memory.revoke_preference(
                        username=self.username,
                        preference_id=referenced[0],
                        authorized=True,
                        evidence=evidence,
                    )
                    note = f"Revoked preference `{record.id}`."
        except Exception as exc:
            note = f"The memory change was not committed: {exc}"
            self.recorder.record_event(
                "Preference mutation failed",
                {
                    "action": decision.memory_action,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            return note

        self._pending_memory_consent = None
        return note

    def handle_user(self, user_text: str) -> TurnResult:
        user_text = user_text.strip()
        if not user_text:
            return TurnResult(
                text="Please enter a request.",
                phase=TaskPhase.WAITING_FOR_USER,
            )

        self._transition(TaskPhase.HRI)
        self.conversation.append({"role": "user", "content": user_text})
        self._active_memory_consent = self._pending_memory_consent
        self._pending_memory_consent = None

        if _REMEMBER_COMMAND.match(user_text) or _FORGET_COMMAND.match(
            user_text
        ) or user_text.strip().lower() == "/preferences":
            self.recorder.record_event("User", user_text)
        try:
            command_result = self._handle_memory_command(user_text)
        except Exception as exc:
            self.recorder.record_event(
                "Memory command failed",
                {"error": f"{type(exc).__name__}: {exc}"},
            )
            return self._reply(
                f"I could not apply that memory command: {exc}",
                phase=TaskPhase.FAILED,
            )
        if command_result is not None:
            return command_result

        try:
            initial = self.observation_source.capture(purpose="hri-input")
        except Exception as exc:
            self.recorder.record_event(
                "Observation failure",
                {"error": f"{type(exc).__name__}: {exc}"},
            )
            return self._reply(
                f"I could not read the camera frame: {exc}",
                phase=TaskPhase.FAILED,
            )

        self.recorder.record_user(user_text, frame_path=initial.frame_path)
        memory_context = MemoryContext(status=MemoryStatus.NOT_RETRIEVED)
        retrieval_signatures: set[str] = set()

        try:
            for _ in range(self.limits.hri_routes_per_turn):
                decision = self.hri.decide(
                    username=self.username,
                    user_query=user_text,
                    observation=initial.observation,
                    memory=memory_context,
                    conversation=self.conversation[-12:],
                    pending_memory_consent=self._active_memory_consent,
                    frame_path=initial.frame_path,
                )
                self.recorder.record_event("HRI route", decision)

                if decision.decision == HRIDecisionKind.RETRIEVE_MEMORY:
                    assert decision.memory_request is not None
                    signature = decision.memory_request.model_dump_json(
                        exclude={"request_id"}
                    )
                    if signature in retrieval_signatures:
                        memory_context = MemoryContext(
                            status=MemoryStatus.UNAVAILABLE,
                            request_id=decision.memory_request.request_id,
                            warnings=[
                                "HRI repeated an identical retrieval request; "
                                "the host suppressed the loop."
                            ],
                        )
                    else:
                        retrieval_signatures.add(signature)
                        try:
                            memory_context = self.memory.retrieve(
                                username=self.username,
                                request=decision.memory_request,
                            )
                        except Exception as exc:
                            memory_context = MemoryContext(
                                status=MemoryStatus.UNAVAILABLE,
                                request_id=decision.memory_request.request_id,
                                warnings=[
                                    f"Memory retrieval unavailable: "
                                    f"{type(exc).__name__}: {exc}"
                                ],
                            )
                            self.recorder.record_event(
                                "Memory retrieval unavailable",
                                {
                                    "request": decision.memory_request,
                                    "error": f"{type(exc).__name__}: {exc}",
                                },
                            )
                    continue

                supplied_memory_ids = {
                    str(item["record_id"])
                    for item in [
                        *memory_context.relevant_preferences,
                        *memory_context.relevant_history,
                        *memory_context.conflicts,
                    ]
                    if item.get("record_id") is not None
                }
                referenced_ids = set(decision.memory_refs_used)
                if not referenced_ids.issubset(supplied_memory_ids):
                    raise ValueError(
                        "HRI referenced persistent memory IDs that were not "
                        "supplied by the host"
                    )
                if decision.task_contract is not None and not set(
                    decision.task_contract.preference_refs
                ).issubset(supplied_memory_ids):
                    raise ValueError(
                        "task contract referenced preferences outside the "
                        "retrieved host context"
                    )

                memory_note = self._commit_hri_memory_action(
                    decision=decision,
                    user_text=user_text,
                )

                if decision.decision == HRIDecisionKind.ASK_USER:
                    assert decision.interaction is not None
                    assert decision.reply_to_user is not None
                    if decision.interaction.kind == InteractionKind.MEMORY_CONSENT:
                        proposed = decision.interaction.proposed_value
                        if not isinstance(proposed, Mapping):
                            raise ValueError(
                                "MEMORY_CONSENT requires a structured proposal"
                            )
                        required = {
                            "statement",
                            "scope",
                            "structured_value",
                        }
                        missing = sorted(required - set(proposed))
                        if missing:
                            raise ValueError(
                                "MEMORY_CONSENT proposal is missing "
                                f"{missing}"
                            )
                        self._pending_memory_consent = {
                            "pending_question": decision.reply_to_user,
                            "proposal": dict(proposed),
                        }
                    reply = decision.reply_to_user
                    if memory_note:
                        reply = f"{reply}\n\n{memory_note}"
                    return self._reply(
                        reply,
                        phase=TaskPhase.WAITING_FOR_USER,
                        memory=memory_context,
                    )

                if decision.decision == HRIDecisionKind.RESPOND:
                    assert decision.reply_to_user is not None
                    reply = decision.reply_to_user
                    if memory_note:
                        reply = f"{reply}\n\n{memory_note}"
                    return self._reply(
                        reply,
                        phase=TaskPhase.COMPLETE,
                        memory=memory_context,
                    )

                assert decision.task_contract is not None
                return self._run_task(
                    decision.task_contract,
                    initial=initial,
                    memory_context=memory_context,
                    memory_note=memory_note,
                )
        except Exception as exc:
            self.recorder.record_event(
                "Controller error",
                {"stage": self.phase, "error": f"{type(exc).__name__}: {exc}"},
            )
            return self._reply(
                f"I could not safely complete this turn: {exc}",
                phase=TaskPhase.FAILED,
                memory=memory_context,
            )

        return self._reply(
            "I could not resolve the request within the HRI routing budget.",
            phase=TaskPhase.FAILED,
            memory=memory_context,
        )

    @staticmethod
    def _plan_text(plan: PlanResult) -> str:
        if plan.planning_status == PlanningStatus.ALREADY_SATISFIED:
            return "The planner found that the requested final state may already hold."
        lines = [
            f"{index}. {subtask.task_instruction}"
            for index, subtask in enumerate(plan.subtasks, start=1)
        ]
        return "\n".join(lines)

    @staticmethod
    def _with_memory_note(text: str, note: str | None) -> str:
        return f"{text}\n\n{note}" if note else text

    def _present_controller_result(
        self,
        *,
        fallback: str,
        observation: ObservationEnvelope,
        memory: MemoryContext,
        payload: Mapping[str, Any],
    ) -> str:
        del observation, memory
        # Authoritative controller outcomes are rendered by host code.  Passing
        # them through free-form model prose would let a presentation model
        # contradict plan-only, safety, or final-validation state.
        self.recorder.record_event(
            "Deterministic controller presentation",
            {
                "controller_result": dict(payload),
                "user_visible_text": fallback,
            },
        )
        return fallback

    @staticmethod
    def _validation_text(
        validation: ValidationResult,
        *,
        confirmed_intent: str,
    ) -> str:
        if validation.outcome == ValidatorOutcome.SUCCESS:
            return (
                "Final validation verified task completion: "
                f"{confirmed_intent}"
            )
        if validation.outcome == ValidatorOutcome.UNSAFE:
            return (
                "The task is not complete. Final validation reported an unsafe "
                "condition, so execution is stopped."
            )
        if validation.outcome == ValidatorOutcome.FAILURE:
            return (
                "The task is not complete. Final validation found that the "
                "requested final state was not achieved."
            )
        return (
            "The task is not complete because final validation cannot verify "
            "the requested final state."
        )

    def _save_episode(
        self,
        *,
        state: _ExecutionState,
        summary: str,
        result: Mapping[str, Any] | str,
        validation: ValidationResult | None = None,
    ) -> None:
        episode = EpisodeRecord(
            id=f"episode-{uuid4().hex}",
            username=self.username,
            request=state.task_contract.confirmed_intent,
            resolved_task=state.task_contract.model_dump(mode="json"),
            actions=[
                subtask.model_dump(mode="json")
                for subtask in state.plan.subtasks
            ],
            execution={
                "completed_subtasks": state.completed_subtasks,
                "dispatches": state.dispatches,
                "monitor_results": [
                    item.model_dump(mode="json")
                    for item in state.monitor_results
                ],
                "replans": state.replans,
                "boundary_replans": state.boundary_replans,
                "reobservations": state.reobservations,
                "plan_only": self.plan_only,
            },
            validation=(
                validation.model_dump(mode="json")
                if validation is not None
                else {}
            ),
            result=dict(result) if isinstance(result, Mapping) else result,
            summary=summary,
            tags=[state.task_contract.task_type],
        )
        self.memory.save_episode(episode)

    def _run_task(
        self,
        task_contract: TaskContract,
        *,
        initial: ObservationEnvelope,
        memory_context: MemoryContext,
        memory_note: str | None = None,
    ) -> TurnResult:
        task_id = f"task-{uuid4().hex}"
        planning_request = PlanningRequest(
            plan_id=f"plan-{uuid4().hex}",
            plan_version=1,
            planning_mode=self.replan_policy,
            validation_spec_id=f"spec-{uuid4().hex}",
        )
        self._transition(TaskPhase.PLANNING, details={"task_id": task_id})
        plan = self.planner.plan(
            planning_request=planning_request,
            task_contract=task_contract,
            observation=initial.observation,
            frame_path=initial.frame_path,
        )
        plan.require_intent(task_contract)
        plan.require_request(planning_request)
        self.recorder.record_event("Plan accepted by host", plan)
        state = _ExecutionState(
            task_id=task_id,
            task_contract=task_contract,
            planning_request=planning_request,
            plan=plan,
        )

        if plan.planning_status not in {
            PlanningStatus.READY,
            PlanningStatus.ALREADY_SATISFIED,
        }:
            assert plan.failure is not None
            planner_text = (
                f"Planning stopped with {plan.planning_status.value} "
                f"({plan.failure.code}). No execution command was published."
            )
            text = self._present_controller_result(
                fallback=planner_text,
                observation=initial,
                memory=memory_context,
                payload={
                    "kind": "PLANNER_RESULT",
                    "plan": plan.model_dump(mode="json"),
                    "host_truth": "No execution command was published.",
                },
            )
            text = self._with_memory_note(text, memory_note)
            self._save_episode(
                state=state,
                summary=f"Planning stopped: {text}",
                result={"status": plan.planning_status, "message": text},
            )
            return self._reply(
                text,
                phase=TaskPhase.FAILED,
                task_id=task_id,
                plan=plan,
                memory=memory_context,
            )

        if self.plan_only and plan.planning_status == PlanningStatus.READY:
            fallback = (
                "I grounded the request in the current camera frame and prepared "
                "this plan:\n\n"
                f"{self._plan_text(plan)}\n\n"
                "Plan-only mode is active, so no VLA command was sent and I am "
                "not claiming that the scene changed."
            )
            text = self._present_controller_result(
                fallback=fallback,
                observation=initial,
                memory=memory_context,
                payload={
                    "kind": "PLAN_READY_NOT_EXECUTED",
                    "plan": plan.model_dump(mode="json"),
                    "host_truth": (
                        "Plan-only mode: no VLA command was published and no "
                        "physical completion may be claimed."
                    ),
                },
            )
            text = self._with_memory_note(text, memory_note)
            self._save_episode(
                state=state,
                summary="A grounded plan was produced but not executed.",
                result={"status": "PLANNED_NOT_EXECUTED"},
            )
            return self._reply(
                text,
                phase=TaskPhase.COMPLETE,
                task_id=task_id,
                plan=plan,
                memory=memory_context,
            )

        if self.vla is None and plan.planning_status == PlanningStatus.READY:
            fallback = (
                "The plan is ready, but no VLA execution adapter is configured. "
                "No command was sent."
            )
            text = self._present_controller_result(
                fallback=fallback,
                observation=initial,
                memory=memory_context,
                payload={
                    "kind": "EXECUTION_UNAVAILABLE",
                    "plan": plan.model_dump(mode="json"),
                    "host_truth": "No VLA command was published.",
                },
            )
            text = self._with_memory_note(text, memory_note)
            self._save_episode(
                state=state,
                summary=text,
                result={"status": "EXECUTION_UNAVAILABLE"},
            )
            return self._reply(
                text,
                phase=TaskPhase.FAILED,
                task_id=task_id,
                plan=plan,
                memory=memory_context,
            )

        assert plan.validation_spec is not None
        frozen_spec = plan.validation_spec

        if plan.planning_status == PlanningStatus.READY:
            execution_failure = self._execute_plan(
                state,
                initial=initial,
                memory_context=memory_context,
            )
            if execution_failure is not None:
                self._save_episode(
                    state=state,
                    summary=execution_failure,
                    result={"status": "EXECUTION_STOPPED"},
                )
                terminal_phase = (
                    TaskPhase.SAFETY_STOP
                    if self.phase == TaskPhase.SAFETY_STOP
                    else TaskPhase.FAILED
                )
                presented = self._present_controller_result(
                    fallback=execution_failure,
                    observation=initial,
                    memory=memory_context,
                    payload={
                        "kind": "EXECUTION_STOPPED",
                        "message": execution_failure,
                        "execution": {
                            "completed_subtasks": state.completed_subtasks,
                            "dispatches": state.dispatches,
                            "monitor_results": [
                                item.model_dump(mode="json")
                                for item in state.monitor_results
                            ],
                        },
                        "host_truth": "Final validation did not authorize completion.",
                    },
                )
                return self._reply(
                    self._with_memory_note(presented, memory_note),
                    phase=terminal_phase,
                    task_id=task_id,
                    plan=state.plan,
                    memory=memory_context,
                )

        while True:
            validation, terminal = self._validate_final(
                state,
                frozen_spec=frozen_spec,
            )
            if validation.outcome in {
                ValidatorOutcome.SUCCESS,
                ValidatorOutcome.UNSAFE,
            }:
                break
            if validation.recoverability not in {
                ValidatorRecoverability.AUTO_LOCAL,
                ValidatorRecoverability.REPLAN,
            }:
                break
            replanned = self._replan(
                state,
                observation=terminal,
                memory_context=memory_context,
                frozen_spec=frozen_spec,
                reason={
                    "trigger": "FINAL_VALIDATION",
                    "validation_result": validation.model_dump(mode="json"),
                    "completed_subtasks": state.completed_subtasks,
                },
            )
            if isinstance(replanned, str):
                validation = validation.model_copy(
                    update={
                        "user_message": (
                            f"{validation.user_message} {replanned}"
                        )
                    }
                )
                break
            if replanned.planning_status == PlanningStatus.ALREADY_SATISFIED:
                if state.reobservations >= self.limits.max_reobservations:
                    break
                state.reobservations += 1
                continue
            execution_failure = self._execute_plan(
                state,
                initial=terminal,
                memory_context=memory_context,
            )
            if execution_failure is not None:
                validation = validation.model_copy(
                    update={
                        "user_message": (
                            f"{validation.user_message} {execution_failure}"
                        )
                    }
                )
                break

        text = self._present_controller_result(
            fallback=self._validation_text(
                validation,
                confirmed_intent=state.task_contract.confirmed_intent,
            ),
            observation=terminal,
            memory=memory_context,
            payload={
                "kind": "FINAL_VALIDATION",
                "validation": validation.model_dump(mode="json"),
                "host_truth": (
                    "Completion is authorized only when outcome is SUCCESS and "
                    "task_complete is true."
                ),
            },
        )
        text = self._with_memory_note(text, memory_note)
        self._save_episode(
            state=state,
            summary=text,
            result={"status": validation.outcome},
            validation=validation,
        )
        phase = (
            TaskPhase.COMPLETE
            if validation.outcome == ValidatorOutcome.SUCCESS
            else (
                TaskPhase.SAFETY_STOP
                if validation.outcome == ValidatorOutcome.UNSAFE
                else TaskPhase.FAILED
            )
        )
        return self._reply(
            text,
            phase=phase,
            task_id=task_id,
            plan=state.plan,
            validation=validation,
            memory=memory_context,
            metadata={"terminal_observation_id": terminal.observation.observation_id},
        )

    def _execute_plan(
        self,
        state: _ExecutionState,
        *,
        initial: ObservationEnvelope,
        memory_context: MemoryContext,
    ) -> str | None:
        assert self.vla is not None
        assert state.plan.validation_spec is not None
        frozen_spec = state.plan.validation_spec
        pending = deque(state.plan.subtasks)
        latest = initial

        while pending:
            if len(state.dispatches) >= self.limits.max_dispatches:
                return "Execution stopped because the dispatch budget was exhausted."
            subtask = pending.popleft()
            dispatch_observation = self.observation_source.capture(
                purpose=f"dispatch-{subtask.subtask_id}"
            )
            latest = dispatch_observation
            dispatch_id = f"dispatch-{uuid4().hex}"
            command = ExecutionCommand(
                dispatch_id=dispatch_id,
                subtask_id=subtask.subtask_id,
                task_instruction=subtask.task_instruction,
                observation_id=dispatch_observation.observation.observation_id,
                current_frame=dispatch_observation.observation.image_block,
            )
            self._transition(
                TaskPhase.PUBLISHING,
                details={
                    "task_id": state.task_id,
                    "dispatch_id": dispatch_id,
                    "subtask_id": subtask.subtask_id,
                },
            )
            receipt = self.vla.publish(command)
            state.dispatches.append(
                {
                    "command": command.model_dump(
                        mode="json",
                        exclude={"current_frame"},
                    ),
                    "receipt": receipt.model_dump(mode="json"),
                }
            )

            try:
                result, latest = self._monitor_subtask(
                    state,
                    subtask=subtask,
                    dispatch_id=dispatch_id,
                    dispatch_observation=dispatch_observation,
                )
            except Exception as exc:
                cancellation = self._cancel_dispatch(
                    state,
                    dispatch_id=dispatch_id,
                    reason="live_monitor_error",
                )
                self.recorder.record_event(
                    "Live monitoring stopped",
                    {
                        "dispatch_id": dispatch_id,
                        "subtask_id": subtask.subtask_id,
                        "error": f"{type(exc).__name__}: {exc}",
                        "cancellation": (
                            cancellation.as_json()
                            if cancellation is not None
                            else None
                        ),
                    },
                )
                return (
                    f"Execution monitoring stopped at subtask "
                    f"`{subtask.subtask_id}`: {exc}. "
                    "I am not claiming task completion."
                )
            state.monitor_results.append(result)

            if result.task_status == TaskStatus.SUCCESS:
                state.completed_subtasks.append(subtask.subtask_id)
                if (
                    self.replan_policy == ReplanPolicy.EVERY_SUBTASK
                    and pending
                ):
                    replanned = self._replan(
                        state,
                        observation=latest,
                        memory_context=memory_context,
                        frozen_spec=frozen_spec,
                        reason={
                            "trigger": "EVERY_SUBTASK",
                            "completed_subtasks": state.completed_subtasks,
                            "remaining_subtasks": [
                                item.model_dump(mode="json") for item in pending
                            ],
                        },
                    )
                    if isinstance(replanned, str):
                        return replanned
                    pending = deque(replanned.subtasks)
                continue

            if result.recommended_action == MonitorAction.ABORT_SAFETY:
                self._cancel_dispatch(
                    state,
                    dispatch_id=dispatch_id,
                    reason="live_monitor_safety_abort",
                )
                self._transition(TaskPhase.SAFETY_STOP)
                return (
                    "Execution was stopped because the live monitor reported an "
                    "unsafe condition. I am not claiming task completion."
                )

            if result.recommended_action == MonitorAction.USER_ASSIST:
                self._cancel_dispatch(
                    state,
                    dispatch_id=dispatch_id,
                    reason="live_monitor_user_assist",
                )
                self._transition(TaskPhase.WAITING_FOR_USER)
                return (
                    f"Execution stopped at subtask `{subtask.subtask_id}` and "
                    "needs user assistance."
                )

            if result.recommended_action == MonitorAction.REPLAN:
                self._cancel_dispatch(
                    state,
                    dispatch_id=dispatch_id,
                    reason="live_monitor_replan",
                )
                replanned = self._replan(
                    state,
                    observation=latest,
                    memory_context=memory_context,
                    frozen_spec=frozen_spec,
                    reason={
                        "trigger": "MONITOR",
                        "failed_subtask": subtask.model_dump(mode="json"),
                        "monitor_result": result.model_dump(mode="json"),
                        "completed_subtasks": state.completed_subtasks,
                    },
                )
                if isinstance(replanned, str):
                    return replanned
                pending = deque(replanned.subtasks)
                continue

            self._cancel_dispatch(
                state,
                dispatch_id=dispatch_id,
                reason=(
                    "terminal_monitor_action:"
                    f"{result.recommended_action.value}"
                ),
            )
            return (
                f"Execution stopped at subtask `{subtask.subtask_id}` with "
                f"monitor action {result.recommended_action}."
            )
        return None

    def _cancel_dispatch(
        self,
        state: _ExecutionState,
        *,
        dispatch_id: str,
        reason: str,
    ) -> CancellationReceipt | None:
        """Best-effort stop request for a command that was already published."""

        assert self.vla is not None
        try:
            receipt = self.vla.cancel(dispatch_id, reason)
            cancellation_record: dict[str, Any] = receipt.as_json()
        except Exception as exc:
            receipt = None
            cancellation_record = {
                "dispatch_id": dispatch_id,
                "reason": reason,
                "supported": None,
                "accepted": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

        for dispatch in reversed(state.dispatches):
            command = dispatch.get("command", {})
            if command.get("dispatch_id") == dispatch_id:
                dispatch["cancellation"] = cancellation_record
                break
        self.recorder.record_event(
            "VLA cancellation boundary",
            cancellation_record,
        )
        return receipt

    def _monitor_subtask(
        self,
        state: _ExecutionState,
        *,
        subtask: Any,
        dispatch_id: str,
        dispatch_observation: ObservationEnvelope,
    ) -> tuple[MonitorResult, ObservationEnvelope]:
        self._transition(
            TaskPhase.MONITORING,
            details={"dispatch_id": dispatch_id},
        )
        started = self.clock()
        recent: deque[ObservationEnvelope] = deque(
            maxlen=self.limits.monitor_window
        )
        previous: MonitorResult | None = None
        success_confirmations = 0
        successful_observation_keys: set[str] = set()
        last_progress_at = 0.0

        def host_timeout_result(
            *,
            message: str,
            action: MonitorAction,
        ) -> MonitorResult:
            return MonitorResult(
                dispatch_id=dispatch_id,
                subtask_id=subtask.subtask_id,
                task_status=TaskStatus.FAILURE,
                progress=MonitorProgress.STALLED,
                observation_quality=ObservationQuality.STALE,
                safety_status=MonitorSafetyStatus.UNKNOWN,
                failure_kind=FailureKind.STALLED,
                recommended_action=action,
                condition_checks=[
                    ConditionCheck(
                        condition_id=condition.condition_id,
                        state=ConditionState.UNKNOWN,
                        evidence=message,
                        confidence=0.0,
                    )
                    for condition in subtask.expected_outcome.conditions
                ],
                confidence=1.0,
            )

        for check_index in range(self.limits.max_monitor_checks):
            elapsed = self.clock() - started
            if elapsed >= subtask.timeout_policy.timeout_seconds:
                action = (
                    MonitorAction.REPLAN
                    if subtask.timeout_policy.on_timeout == "REPLAN"
                    else MonitorAction.USER_ASSIST
                )
                latest = recent[-1] if recent else dispatch_observation
                return (
                    host_timeout_result(
                        message=(
                            f"Host timeout after {elapsed:.3f} seconds; "
                            "local completion is unverified."
                        ),
                        action=action,
                    ),
                    latest,
                )
            stalled_for = elapsed - last_progress_at
            if stalled_for >= subtask.timeout_policy.stall_seconds:
                stall_action = MonitorAction(
                    subtask.timeout_policy.on_stall
                )
                latest = recent[-1] if recent else dispatch_observation
                stalled = host_timeout_result(
                    message=(
                        f"Host stall threshold reached after {stalled_for:.3f} "
                        "seconds without monitor-reported progress."
                    ),
                    action=stall_action,
                )
                if stall_action != MonitorAction.REOBSERVE:
                    return stalled, latest
                state.reobservations += 1
                if state.reobservations > self.limits.max_reobservations:
                    return (
                        stalled.model_copy(
                            update={
                                "recommended_action": MonitorAction.REPLAN,
                            }
                        ),
                        latest,
                    )
                last_progress_at = elapsed
            if check_index and self.limits.monitor_interval_seconds:
                self.wait(self.limits.monitor_interval_seconds)

            current = self.observation_source.capture(
                purpose=f"monitor-{subtask.subtask_id}-{check_index + 1}"
            )
            recent.append(current)
            elapsed = self.clock() - started
            request = MonitorRequest(
                dispatch_id=dispatch_id,
                subtask_id=subtask.subtask_id,
                task_instruction=subtask.task_instruction,
                expected_outcome=subtask.expected_outcome,
                dispatch_observation=dispatch_observation.observation,
                recent_observations=[
                    envelope.observation for envelope in recent
                ],
                elapsed_seconds=elapsed,
                previous_result=(
                    previous.model_dump(mode="json")
                    if previous is not None
                    else None
                ),
            )
            paths = [
                path
                for path in [
                    dispatch_observation.frame_path,
                    *(item.frame_path for item in recent),
                ]
                if path is not None
            ]
            result = self.monitor.evaluate(request, frame_paths=paths)
            result.require_current_dispatch(
                dispatch_id=request.dispatch_id,
                subtask_id=request.subtask_id,
            )
            result.require_expected_outcome(request.expected_outcome)
            self.recorder.record_event("Monitor result accepted by host", result)
            if result.progress in {
                MonitorProgress.ADVANCING,
                MonitorProgress.VERIFYING,
            }:
                last_progress_at = elapsed

            if result.task_status == TaskStatus.SUCCESS:
                observation_key = (
                    current.observation.content_hash
                    or current.observation.observation_id
                )
                if observation_key in successful_observation_keys:
                    self.recorder.record_event(
                        "Duplicate success observation suppressed",
                        {
                            "dispatch_id": dispatch_id,
                            "subtask_id": subtask.subtask_id,
                            "observation_key": observation_key,
                        },
                    )
                    previous = result
                    continue
                successful_observation_keys.add(observation_key)
                success_confirmations += 1
                if success_confirmations >= self.limits.success_confirmations:
                    return result, current
                previous = result
                continue
            success_confirmations = 0
            if result.task_status == TaskStatus.FAILURE:
                if result.recommended_action == MonitorAction.REOBSERVE:
                    state.reobservations += 1
                    if state.reobservations <= self.limits.max_reobservations:
                        previous = result
                        continue
                return result, current
            if result.recommended_action in {
                MonitorAction.REPLAN,
                MonitorAction.USER_ASSIST,
                MonitorAction.ABORT_SAFETY,
            }:
                return result, current
            if result.recommended_action == MonitorAction.REOBSERVE:
                state.reobservations += 1
                if state.reobservations > self.limits.max_reobservations:
                    raise RuntimeError("live-monitor reobservation budget exhausted")
            previous = result

        latest = recent[-1] if recent else dispatch_observation
        return (
            host_timeout_result(
                message=(
                    "The finite live-monitor check budget was exhausted; "
                    "local completion is unverified."
                ),
                action=MonitorAction.REPLAN,
            ),
            latest,
        )

    def _replan(
        self,
        state: _ExecutionState,
        *,
        observation: ObservationEnvelope,
        memory_context: MemoryContext,
        frozen_spec: Any,
        reason: Mapping[str, Any],
    ) -> PlanResult | str:
        is_boundary_replan = reason.get("trigger") == "EVERY_SUBTASK"
        if not is_boundary_replan:
            if state.replans >= self.limits.max_replans:
                return (
                    "Execution stopped because the replanning budget was "
                    "exhausted."
                )
            state.replans += 1
        else:
            state.boundary_replans += 1
        self._transition(
            TaskPhase.REPLANNING,
            details={
                "replan": (
                    state.boundary_replans
                    if is_boundary_replan
                    else state.replans
                ),
                "replan_kind": (
                    "BOUNDARY" if is_boundary_replan else "RECOVERY"
                ),
                **dict(reason),
            },
        )
        planning_request = state.planning_request.model_copy(
            update={"plan_version": state.planning_request.plan_version + 1}
        )
        plan = self.planner.plan(
            planning_request=planning_request,
            task_contract=state.task_contract,
            observation=observation.observation,
            recovery_context=reason,
            frozen_validation_spec=frozen_spec,
            frame_path=observation.frame_path,
        )
        plan.require_intent(state.task_contract)
        plan.require_request(planning_request)
        if plan.validation_spec is not None:
            frozen_spec.require_exact_match(plan.validation_spec)
        if plan.planning_status not in {
            PlanningStatus.READY,
            PlanningStatus.ALREADY_SATISFIED,
        }:
            code = (
                plan.failure.code
                if plan.failure is not None
                else "NO_FAILURE_DETAIL"
            )
            return (
                "Recovery planning stopped with "
                f"{plan.planning_status.value} ({code})."
            )
        state.planning_request = planning_request
        state.plan = plan
        self.recorder.record_event("Recovery plan accepted by host", plan)
        return plan

    def _validate_final(
        self,
        state: _ExecutionState,
        *,
        frozen_spec: Any,
    ) -> tuple[ValidationResult, ObservationEnvelope]:
        attempts = 0
        terminal_observations: list[ObservationEnvelope] = []
        while True:
            self._transition(
                TaskPhase.VALIDATING,
                details={"attempt": attempts + 1},
            )
            terminal = self.observation_source.capture(
                purpose=f"validation-{attempts + 1}"
            )
            terminal_observations.append(terminal)
            request = ValidationRequest(
                validation_spec=frozen_spec,
                terminal_observations=[
                    item.observation for item in terminal_observations
                ],
                execution_evidence={
                    "completed_subtasks": state.completed_subtasks,
                    "dispatches": state.dispatches,
                    "monitor_results": [
                        item.model_dump(mode="json")
                        for item in state.monitor_results
                    ],
                },
            )
            result = self.validator.validate(
                request,
                frame_paths=[
                    item.frame_path
                    for item in terminal_observations
                    if item.frame_path is not None
                ],
            )
            result.require_request(request)
            self.recorder.record_event("Validation accepted by host", result)
            if (
                result.outcome == ValidatorOutcome.UNKNOWN
                and result.recoverability
                == ValidatorRecoverability.REOBSERVE
                and state.reobservations < self.limits.max_reobservations
            ):
                attempts += 1
                state.reobservations += 1
                continue
            return result, terminal
