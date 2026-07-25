from __future__ import annotations

import copy
import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agents.configs import HRI_Agent_Config, PrefMemConfig
from agents.contracts import ExecutionResult, PlanResult, ValidationResult
from agents.diagnostics import AgentOutputDisplay, DisplayingJsonModel
from agents.memory import MemoryAgent
from agents.model import JsonModel, OllamaJsonModel
from agents.planner import PlannerAgent
from agents.validator import ValidatorAgent
from agents.vision import prepare_vision_image
from agents.vla import RecordedEpisodeExecutor, VLAExecutor
from assurance.task_assurance import TaskAssurance, TaskAssuranceResult
from dataset.episode import DatasetEpisode, DatasetEpisodeError
from memory.models import AgentTurn, ConsentEvidence, MemoryContext, MemoryQuery, PendingQuestion
from memory.repositories import HistoryOutboxRepository, HistoryRepository, PreferenceRepository


HRI_MODES = {"ASK", "CONFIRM", "MEMORY_CONFIRM", "EXECUTE", "REPORT", "RESTART"}
MEMORY_ACTIONS = {
    "NONE",
    "COMMIT",
    "DELETE",
    "DECLINE",
    "DEFER",
    "CORRECT",
    "CANCEL",
}


def _agent_failure_evidence(
    stage: str,
    error: Exception,
    *,
    attempt: int,
    event: str,
) -> dict[str, Any]:
    name = type(error).__name__
    message = str(error)
    lowered = f"{name} {message}".casefold()
    if name in {"PlannerAgentError", "ValidatorAgentError", "HRIContractError"}:
        classification = "OUTPUT_CONTRACT"
    elif "timeout" in lowered or "timed out" in lowered:
        classification = "INFRASTRUCTURE_TIMEOUT"
    elif any(
        marker in lowered
        for marker in ("connection", "connecterror", "ollama", "http")
    ):
        classification = "MODEL_OR_CONNECTION"
    else:
        classification = "AGENT_RUNTIME"
    return {
        "event_id": (
            f"{stage.casefold()}-attempt-{attempt}-{event.casefold()}"
        ),
        "attempt": attempt,
        "event": event,
        "stage": stage,
        "error_type": name,
        "classification": classification,
        "message": message,
    }


class HRIContractError(ValueError):
    pass


class HRIOrchestrator:
    """The only user-facing agent and owner of all sub-agent orchestration."""

    def __init__(
        self,
        config: PrefMemConfig,
        *,
        hri_model: JsonModel | None = None,
        memory_agent: MemoryAgent | None = None,
        planner_agent: PlannerAgent | None = None,
        validator_agent: ValidatorAgent | None = None,
        executor: VLAExecutor | None = None,
        task_assurance: TaskAssurance | None = None,
        output_display: AgentOutputDisplay | None = None,
    ):
        self.config = config
        self.output_display = output_display or AgentOutputDisplay()
        self.workspace_root = Path(config.workspace_root).resolve()
        self.dataset_episode = DatasetEpisode.from_path(config.dataset_path, self.workspace_root)
        base_hri_model = hri_model or OllamaJsonModel(
            config.hri.model,
            config.hri.temperature,
            host=config.hri.host,
            timeout_seconds=config.hri.timeout_seconds,
            seed=config.hri.seed,
        )
        self.hri_model = DisplayingJsonModel(
            base_hri_model,
            self.output_display,
            "HRI Agent",
        )
        self.hri_prompt = Path(config.hri.system_prompt_path).read_text(encoding="utf-8")
        if memory_agent is None:
            memory_agent = MemoryAgent(
                config.memory,
                HistoryRepository(config.history_store_path),
                PreferenceRepository(config.preference_store_path),
                recent_history_limit=config.recent_history_limit,
                history_semantic_scan_limit=config.history_semantic_scan_limit,
                semantic_history_limit=config.semantic_history_limit,
                semantic_preference_limit=config.semantic_preference_limit,
                semantic_preference_threshold=config.semantic_preference_threshold,
                preference_proposal_min_episodes=config.preference_proposal_min_episodes,
                preference_proposal_confidence=config.preference_proposal_confidence,
                compact_after_write=config.compact_preferences_after_write,
                output_display=self.output_display,
            )
        self.memory_agent = memory_agent
        outbox_path = config.history_outbox_path or str(
            Path(config.history_store_path).with_name("history_outbox.json")
        )
        self.history_outbox = HistoryOutboxRepository(outbox_path)
        self.planner_agent = planner_agent or PlannerAgent(
            config.planner,
            vision=config.vision,
            output_display=self.output_display,
        )
        self.validator_agent = validator_agent or ValidatorAgent(
            config.validator,
            vision=config.vision,
            output_display=self.output_display,
        )
        self.executor = executor or RecordedEpisodeExecutor(self.dataset_episode)
        self.task_assurance = task_assurance or TaskAssurance(
            minimum_planner_confidence=config.minimum_planner_confidence,
            minimum_validator_confidence=config.minimum_validator_confidence,
        )
        self.session_id = uuid.uuid4().hex
        self.history_context = MemoryContext()
        self.pending_question: PendingQuestion | None = None
        self.last_task_result: dict[str, Any] | None = None
        self._command_active = False
        self._episode_id: str | None = None
        self._started_at: str | None = None
        self._initial_request = ""
        self._dialogue: list[AgentTurn] = []
        self._memory_events: list[dict[str, Any]] = []
        self._terminal_task_contract: dict[str, Any] | None = None
        self._terminal_task_result: dict[str, Any] | None = None
        self._terminal_hri_response: dict[str, Any] | None = None
        self._safety_latched = False
        self._safety_event: dict[str, Any] | None = None
        self._display_all = False

    @property
    def initial_frame_path(self) -> str:
        return str(self.dataset_episode.initial_frame)

    @property
    def final_frame_path(self) -> str:
        return str(self.dataset_episode.final_frame)

    def _model_scene_context(self) -> dict[str, Any]:
        provider = getattr(self.executor, "model_scene_context", None)
        if callable(provider):
            context = provider()
            if isinstance(context, dict):
                return copy.deepcopy(context)
        observation_id = "obs-" + hashlib.sha256(
            str(self.dataset_episode.directory).encode("utf-8")
        ).hexdigest()[:24]
        return {
            "observation_id": observation_id,
            "available_modalities": ["initial_rgb"],
        }

    @property
    def command_active(self) -> bool:
        return self._command_active

    def handle_user_message(self, user_message: str) -> dict[str, Any]:
        message = str(user_message).strip()
        if not message:
            raise HRIContractError("User message cannot be empty.")
        if not self._command_active and self._safety_latched:
            clearance = self._handle_safety_clearance(message)
            if clearance is not None:
                return clearance
        is_new_command = not self._command_active
        is_post_task_memory_reply = self._terminal_task_result is not None
        turn_id = f"turn-{uuid.uuid4().hex}"
        if is_new_command:
            self._begin_command(message)
        pending_before = copy.deepcopy(self.pending_question)
        dialogue_before = copy.deepcopy(self._dialogue)
        user_turn = AgentTurn(turn_id=turn_id, role="user", content=message)
        dialogue_for_model = [*self._dialogue, user_turn]

        image = prepare_vision_image(
            self.initial_frame_path,
            resize=self.config.vision.resize_images,
            width=self.config.vision.image_width,
            height=self.config.vision.image_height,
            jpeg_quality=self.config.vision.image_jpeg_quality,
        )
        hri_payload = {
            "event": (
                "NEW_COMMAND"
                if is_new_command
                else "POST_TASK_MEMORY_REPLY"
                if is_post_task_memory_reply
                else "USER_REPLY"
            ),
            "user_message": message,
            "dialogue": [turn.to_dict() for turn in dialogue_for_model],
            "memory_context": self.history_context.to_dict(),
            "pending_question": self.pending_question.to_dict()
            if self.pending_question
            else None,
            "scene": {
                **self._model_scene_context(),
            },
        }
        try:
            response = self._generate_hri_turn(
                hri_payload,
                images=[image] if is_new_command else [],
                pending_before=pending_before,
                post_task_memory_reply=is_post_task_memory_reply,
            )
        except Exception:
            if is_new_command:
                self._reset_command_state()
            raise
        self._dialogue.append(user_turn)
        self._dialogue.append(
            AgentTurn(
                turn_id=f"turn-{uuid.uuid4().hex}",
                role="assistant",
                content=response["user_message"],
            )
        )
        try:
            memory_result = self._handle_memory_action(response, message, turn_id)
            if memory_result is not None:
                self._display_agent_output(
                    "Memory Agent",
                    "preference action",
                    memory_result,
                )
            if isinstance(memory_result, dict) and memory_result.get("committed") is False:
                response["user_message"] = (
                    "I could not update your saved preference; nothing from this request was committed."
                )
                self._dialogue[-1].content = response["user_message"]
            self._set_pending_question(response)
        except Exception:
            self._dialogue = dialogue_before
            self.pending_question = pending_before
            if is_new_command:
                self._reset_command_state()
            raise

        if response["mode"] == "RESTART":
            if is_post_task_memory_reply:
                terminal_result = copy.deepcopy(self._terminal_task_result)
                if terminal_result is None:
                    raise HRIContractError("Restart lost the completed task result.")
                self._finish_episode(
                    copy.deepcopy(self._terminal_hri_response or response),
                    copy.deepcopy(self._terminal_task_contract),
                    terminal_result,
                )
            else:
                cancelled = {
                    "phase": "HRI",
                    "outcome": "CANCELLED",
                    "proceed": False,
                    "next_action": "NONE",
                    "message": "The previous request was cancelled by a new command.",
                    "failure": {"code": "INTERRUPTED_BY_NEW_COMMAND"},
                }
                self._finish_episode(response, None, cancelled)
            return self.handle_user_message(message)

        result: dict[str, Any] = {
            "hri": copy.deepcopy(response),
            "memory": memory_result,
            "task": None,
            "awaiting_user": response["mode"] != "EXECUTE" and response["mode"] != "REPORT",
        }
        if is_post_task_memory_reply:
            if response["mode"] == "REPORT":
                terminal_contract = copy.deepcopy(self._terminal_task_contract)
                terminal_result = copy.deepcopy(self._terminal_task_result)
                terminal_hri = copy.deepcopy(self._terminal_hri_response or response)
                if terminal_result is None:
                    raise HRIContractError("Post-task memory state lost its terminal result.")
                self._finish_episode(terminal_hri, terminal_contract, terminal_result)
                result["awaiting_user"] = False
            else:
                result["awaiting_user"] = True
            return result

        if response["mode"] == "EXECUTE":
            task_contract = response.get("task_contract")
            if not isinstance(task_contract, dict):
                raise HRIContractError("EXECUTE requires a task_contract object.")
            task_contract = self._normalize_task_contract(task_contract, turn_id)
            task_result = self._execute_task(task_contract)
            if str(task_result.get("outcome", "")).upper() == "ABORTED_SAFETY":
                self._safety_latched = True
                self._safety_event = {
                    "task_id": task_contract.get("task_id"),
                    "episode_id": self._episode_id,
                    "failure": copy.deepcopy(task_result.get("failure")),
                    "latched_at": self._now(),
                }
            result["task"] = task_result
            # Keep deterministic task provenance in the returned command
            # envelope even though the final user-facing HRI mode is REPORT or
            # MEMORY_CONFIRM (both correctly require task_contract=null).
            result["resolved_task"] = copy.deepcopy(task_contract)
            episode_source = self._build_episode_source(response, task_contract, task_result)
            proposal = self._maybe_propose_preference(episode_source, task_result)
            history_checkpointed = False
            if proposal is not None:
                history_checkpointed = self._checkpoint_terminal_episode(
                    episode_source
                )
                if not history_checkpointed:
                    proposal = None
            if proposal is not None:
                result["history_checkpointed"] = True
                prompt_id = f"prompt-{uuid.uuid4().hex}"
                payload = {
                    "preference_request": copy.deepcopy(proposal["preference_request"]),
                    "source_episode_ids": list(proposal["source_episode_ids"]),
                    "requested_action": "UPSERT",
                }
                self.pending_question = PendingQuestion(
                    kind="MEMORY_CONSENT",
                    payload=payload,
                    question=proposal["question"],
                    prompt_id=prompt_id,
                )
                final_message = f"{task_result['message']} {proposal['question']}".strip()
                decision_trace = (
                    copy.deepcopy(response.get("trace"))
                    if isinstance(response.get("trace"), dict)
                    else {}
                )
                boundary_response = {
                    "mode": "MEMORY_CONFIRM",
                    "user_message": final_message,
                    "task_contract": None,
                    "pending_question": self.pending_question.to_dict(),
                    "memory_action": {"action": "NONE"},
                    "report": None,
                    "trace": {
                        "grounding": list(decision_trace.get("grounding", [])),
                        "history_refs": list(
                            decision_trace.get("history_refs", [])
                        ),
                        "memory_refs": list(
                            decision_trace.get("memory_refs", [])
                        ),
                        "assumptions": list(
                            decision_trace.get("assumptions", [])
                        ),
                        # These IDs support the optional preference proposal;
                        # they are deliberately distinct from the history records
                        # the HRI cited while resolving the task.
                        "proposal_history_refs": list(
                            proposal["source_episode_ids"]
                        ),
                    },
                }
                self._dialogue.append(
                    AgentTurn(
                        turn_id=f"turn-{uuid.uuid4().hex}",
                        role="assistant",
                        content=final_message,
                    )
                )
                self._terminal_task_contract = copy.deepcopy(task_contract)
                self._terminal_task_result = copy.deepcopy(task_result)
                self._terminal_hri_response = copy.deepcopy(response)
                result["hri"] = boundary_response
                result["awaiting_user"] = True
            else:
                final_response = self._task_report_response(
                    task_result,
                    trace=response.get("trace"),
                )
                if final_response["user_message"] != response["user_message"]:
                    self._dialogue.append(
                        AgentTurn(
                            turn_id=f"turn-{uuid.uuid4().hex}",
                            role="assistant",
                            content=final_response["user_message"],
                        )
                    )
                result["hri"] = final_response
                result["awaiting_user"] = False
                self._finish_episode(final_response, task_contract, task_result)
        elif response["mode"] == "REPORT":
            reported_outcome = str(response.get("report", {}).get("outcome", "BLOCKED")).upper()
            if reported_outcome in {"SUCCESS", "SUCCESS_RECOVERED", "TASK_COMPLETE"}:
                raise HRIContractError("HRI cannot report task success without Planner and Validator assurance.")
            task_result = {
                "phase": "HRI",
                "outcome": str(response.get("report", {}).get("outcome", "BLOCKED")),
                "proceed": False,
                "next_action": str(response.get("report", {}).get("next_action", "NONE")),
                "message": response["user_message"],
                "failure": copy.deepcopy(response.get("report")),
            }
            self.last_task_result = task_result
            result["task"] = task_result
            result["awaiting_user"] = False
            self._finish_episode(response, None, task_result)
        return result

    def _generate_hri_turn(
        self,
        payload: dict[str, Any],
        *,
        images: list[bytes],
        pending_before: PendingQuestion | None,
        post_task_memory_reply: bool,
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        for response_attempt in range(2):
            retry_payload = copy.deepcopy(payload)
            if response_attempt:
                retry_payload["contract_retry"] = {
                    "error": str(last_error),
                    "instruction": (
                        "Preserve the intended user-facing meaning, but repair the schema error. "
                        "Return exactly one object matching the HRI contract."
                    ),
                    "valid_question_pairs": {
                        "ASK": "TASK_CLARIFICATION",
                        "CONFIRM": "TASK_CONFIRMATION",
                        "MEMORY_CONFIRM": "MEMORY_CONSENT",
                    },
                    "report_rule": (
                        "Ordinary REPORT and RESTART require a structured report. A resolved "
                        "POST_TASK_MEMORY_REPLY must use REPORT with a resolving memory action."
                    ),
                }
            try:
                raw = self.hri_model.generate(
                    purpose="resolve_hri_turn",
                    system_prompt=self.hri_prompt,
                    payload=retry_payload,
                    images=images,
                )
                response = self._validate_hri_output(raw)
                normalized_response = self._normalize_hri_transition_output(
                    response,
                    pending_before=pending_before,
                    post_task_memory_reply=post_task_memory_reply,
                )
                if normalized_response != response:
                    self._display_agent_output(
                        "HRI Contract",
                        f"normalized response attempt {response_attempt + 1}",
                        {
                            "before": response,
                            "after": normalized_response,
                        },
                    )
                response = normalized_response
                self._validate_hri_transition(
                    response,
                    pending_before=pending_before,
                    post_task_memory_reply=post_task_memory_reply,
                )
                return response
            except Exception as error:
                last_error = error
                self._display_agent_output(
                    "HRI Contract",
                    f"rejected response attempt {response_attempt + 1}",
                    {
                        "error_type": type(error).__name__,
                        "error": str(error),
                    },
                )
        raise HRIContractError(
            f"HRI model failed its output contract twice: {last_error}"
        ) from last_error

    def _begin_command(self, message: str) -> None:
        outbox_warning: str | None = None
        try:
            self._retry_history_outbox()
        except Exception as error:
            outbox_warning = f"history retry queue unavailable: {error}"
        query = MemoryQuery(
            user_id=self.config.user_id,
            request=message,
            scene={
                **self._model_scene_context(),
            },
            limit=max(self.config.semantic_history_limit, self.config.semantic_preference_limit),
        )
        context = self.memory_agent.get_memory_context(query)
        if outbox_warning:
            context.warnings.append(outbox_warning)
        self._display_agent_output(
            "Memory Agent",
            "retrieved context",
            context.to_dict(),
        )
        self._command_active = True
        self._episode_id = f"episode-{uuid.uuid4().hex}"
        self._started_at = self._now()
        self._initial_request = message
        self._dialogue = []
        self._memory_events = []
        self.pending_question = None
        self._terminal_task_contract = None
        self._terminal_task_result = None
        self._terminal_hri_response = None
        self.history_context = context

    def _handle_memory_action(
        self,
        response: dict[str, Any],
        user_message: str,
        turn_id: str,
    ) -> dict[str, Any] | None:
        memory_action = response.get("memory_action") or {"action": "NONE"}
        action = str(memory_action.get("action", "NONE")).upper()
        if action not in MEMORY_ACTIONS:
            raise HRIContractError(f"Unsupported HRI memory action: {action}")
        if action == "NONE":
            return None
        if action in {"DECLINE", "DEFER", "CORRECT"}:
            if self.pending_question is None or self.pending_question.kind != "MEMORY_CONSENT":
                raise HRIContractError(f"{action} requires a pending MEMORY_CONSENT question.")
            event = {
                "action": action,
                "prompt_id": self.pending_question.prompt_id,
                "quote": user_message,
            }
            self._memory_events.append(event)
            self.pending_question = None
            return event
        if action == "CANCEL":
            if self.pending_question is None:
                raise HRIContractError("CANCEL requires an active pending question.")
            event = {
                "action": "CANCEL",
                "question_kind": self.pending_question.kind,
                "prompt_id": self.pending_question.prompt_id,
                "quote": user_message,
            }
            self._memory_events.append(event)
            self.pending_question = None
            return event

        pending_request: dict[str, Any] | None = None
        consent_kind: str
        prompt_id: str | None = None
        displayed_question: str | None = None
        authorized_action = "DELETE" if action == "DELETE" else "UPSERT"
        if self.pending_question is not None:
            if self.pending_question.kind != "MEMORY_CONSENT":
                raise HRIContractError(
                    "A task-confirmation reply cannot authorize a preference mutation."
                )
            pending_request = copy.deepcopy(
                self.pending_question.payload.get("preference_request")
            )
            pending_action = str(
                self.pending_question.payload.get("requested_action", "UPSERT")
            ).upper()
            if pending_action != authorized_action:
                raise HRIContractError(
                    "The reply's memory action does not match the displayed memory proposal."
                )
            prompt_id = self.pending_question.prompt_id
            displayed_question = self.pending_question.question
            consent_kind = (
                "forget_confirmation" if action == "DELETE" else "memory_confirmation"
            )
        elif memory_action.get("explicit_consent") is True:
            pending_request = copy.deepcopy(memory_action.get("request"))
            self._verify_direct_memory_consent(
                user_message=user_message,
                requested_action=authorized_action,
                proposed_request=pending_request,
            )
            consent_kind = (
                "forget_confirmation" if action == "DELETE" else "explicit_future_language"
            )
        else:
            raise HRIContractError(
                "Preference mutation requires a pending memory question or explicit user consent."
            )
        if not isinstance(pending_request, dict):
            raise HRIContractError("Preference mutation requires a structured request.")
        pending_request.update(
            {
                "user_id": self.config.user_id,
                "episode_id": self._episode_id,
                "turn_id": turn_id,
                "instruction": str(
                    pending_request.get("instruction") or memory_action.get("instruction") or user_message
                ),
                "requested_action": "DELETE" if action == "DELETE" else "UPSERT",
            }
        )
        consent = ConsentEvidence(
            kind=consent_kind,  # type: ignore[arg-type]
            quote=user_message,
            turn_id=turn_id,
            episode_id=self._episode_id,
            prompt_id=prompt_id,
            authorized_action=authorized_action,
            displayed_question=displayed_question,
            proposal=copy.deepcopy(pending_request),
        )
        try:
            update = self.memory_agent.update_preference_memory(
                pending_request,
                consent=consent,
                transaction_id=f"txn-{turn_id}",
            )
        except Exception as error:
            update = {"error": str(error), "committed": False}
        else:
            update["committed"] = True
        self._memory_events.append(copy.deepcopy(update))
        self.pending_question = None
        return update

    def _set_pending_question(self, response: dict[str, Any]) -> None:
        raw = response.get("pending_question")
        if raw is None:
            if response["mode"] not in {"ASK", "CONFIRM", "MEMORY_CONFIRM"}:
                self.pending_question = None
            return
        if not isinstance(raw, dict):
            raise HRIContractError("pending_question must be an object or null.")
        kind = str(raw.get("kind", ""))
        if kind not in {"TASK_CLARIFICATION", "TASK_CONFIRMATION", "MEMORY_CONSENT"}:
            raise HRIContractError(f"Unsupported pending question kind: {kind}")
        payload = raw.get("payload", {})
        if not isinstance(payload, dict):
            raise HRIContractError("Pending question payload must be an object.")
        if kind == "MEMORY_CONSENT" and not isinstance(
            payload.get("preference_request"), dict
        ):
            raise HRIContractError("Memory consent questions require preference_request payload.")
        if kind == "MEMORY_CONSENT":
            requested_action = str(payload.get("requested_action", "UPSERT")).upper()
            if requested_action not in {"UPSERT", "DELETE"}:
                raise HRIContractError("Memory consent requested_action must be UPSERT or DELETE.")
            payload = copy.deepcopy(payload)
            payload["requested_action"] = requested_action
        self.pending_question = PendingQuestion(
            kind=kind,  # type: ignore[arg-type]
            payload=copy.deepcopy(payload),
            question=response["user_message"],
            prompt_id=str(raw.get("prompt_id") or f"prompt-{uuid.uuid4().hex}"),
        )

    def _execute_task(self, task_contract: dict[str, Any]) -> dict[str, Any]:
        attempts: list[dict[str, Any]] = []
        observation_path = self.initial_frame_path
        recovery_context: dict[str, Any] | None = None
        final_assurance: TaskAssuranceResult | None = None
        frozen_validation_spec = None
        for attempt in range(1, self.config.max_replans + 2):
            try:
                plan = self.planner_agent.plan(
                    task_contract,
                    observation_path,
                    recovery_context=recovery_context,
                )
            except Exception as error:
                failure_evidence = _agent_failure_evidence(
                    "PLANNING",
                    error,
                    attempt=attempt,
                    event="planner-call",
                )
                self._display_agent_output(
                    "Planner Agent",
                    f"attempt {attempt} error",
                    {"error_type": type(error).__name__, "error": str(error)},
                )
                final_assurance = self.task_assurance.runtime_failure(
                    stage="PLANNING",
                    code="PLANNER_UNAVAILABLE",
                    message="The planner is unavailable, so no action was dispatched.",
                    next_action="REPLAN",
                    observed=failure_evidence,
                )
                attempts.append(
                    {
                        "attempt": attempt,
                        "planner_error": str(error),
                        "failure": failure_evidence,
                    }
                )
                if attempt <= self.config.max_replans:
                    recovery_context = {
                        "planner_error": str(error),
                        "instruction": "Retry planning without changing the confirmed task.",
                    }
                    continue
                break
            self._display_agent_output(
                "Planner Agent",
                f"attempt {attempt} parsed output",
                {
                    "parsed": plan.to_dict(),
                    "raw_model_output": copy.deepcopy(plan.raw),
                },
            )
            plan_gate = self.task_assurance.assess_plan(plan)
            self._display_agent_output(
                "Task Assurance",
                f"planning gate attempt {attempt}",
                plan_gate.to_dict(),
            )
            attempt_record: dict[str, Any] = {
                "attempt": attempt,
                "plan": plan.to_dict(),
                "plan_assurance": plan_gate.to_dict(),
            }
            if plan.status.upper() in {"READY", "ALREADY_SATISFIED"}:
                if frozen_validation_spec is None:
                    frozen_validation_spec = plan.validation_spec
                elif plan.validation_spec != frozen_validation_spec:
                    final_assurance = self.task_assurance.runtime_failure(
                        stage="PLANNING",
                        code="VALIDATION_SPEC_DRIFT",
                        message="The recovery plan changed the approved validation goals.",
                        next_action="USER_ASSIST",
                        observed={
                            "expected": frozen_validation_spec.to_dict(),
                            "returned": plan.validation_spec.to_dict()
                            if plan.validation_spec
                            else None,
                        },
                    )
                    attempt_record["spec_drift"] = final_assurance.to_dict()
                    attempts.append(attempt_record)
                    break
            if plan.status.upper() == "ALREADY_SATISFIED":
                if frozen_validation_spec is None:
                    final_assurance = self.task_assurance.runtime_failure(
                        stage="PLANNING",
                        code="INCOMPLETE_ALREADY_SATISFIED_PLAN",
                        message="The claimed current state has no validation specification.",
                        next_action="REPLAN",
                    )
                    attempts.append(attempt_record)
                    break
                execution = ExecutionResult(
                    status="OBSERVATION_ONLY",
                    final_observation=observation_path,
                    evidence={"attempt": attempt, "source": "planner_already_satisfied_claim"},
                )
                attempt_record["execution"] = execution.to_model_dict()
                validation, validation_gate = self._validate_execution(
                    frozen_validation_spec,
                    execution,
                    attempt,
                    event="already-satisfied",
                )
                attempt_record["validation"] = validation.to_dict() if validation else None
                attempt_record["validation_assurance"] = validation_gate.to_dict()
                attempts.append(attempt_record)
                final_assurance = validation_gate
                break
            if not plan_gate.proceed:
                final_assurance = plan_gate
                attempts.append(attempt_record)
                break
            try:
                execution = self._dispatch_plan(plan, attempt)
            except Exception as error:
                failure_evidence = _agent_failure_evidence(
                    "EXECUTION",
                    error,
                    attempt=attempt,
                    event="vla-dispatch",
                )
                self._display_agent_output(
                    "VLA Agent",
                    f"attempt {attempt} error",
                    {"error_type": type(error).__name__, "error": str(error)},
                )
                final_assurance = self.task_assurance.runtime_failure(
                    stage="EXECUTION",
                    code="VLA_EXECUTOR_UNAVAILABLE",
                    message="The VLA executor failed before validation.",
                    next_action="REPLAN",
                    observed=failure_evidence,
                )
                attempt_record["execution_error"] = str(error)
                attempt_record["failure"] = failure_evidence
                attempts.append(attempt_record)
                if attempt <= self.config.max_replans:
                    recovery_context = {
                        "previous_plan": plan.to_dict(),
                        "execution_error": str(error),
                        "frozen_validation_spec": frozen_validation_spec.to_dict()
                        if frozen_validation_spec
                        else None,
                        "instruction": "Retry only if safe; preserve every frozen validation goal.",
                    }
                    continue
                break
            attempt_record["execution"] = execution.to_model_dict()
            self._display_agent_output(
                "VLA Agent",
                f"attempt {attempt} output",
                execution.to_dict(),
            )
            observation_path = execution.final_observation
            execution_gate = self.task_assurance.assess_execution(execution)
            if execution_gate is not None:
                self._display_agent_output(
                    "Task Assurance",
                    f"execution gate attempt {attempt}",
                    execution_gate.to_dict(),
                )
                attempt_record["execution_assurance"] = execution_gate.to_dict()
                attempts.append(attempt_record)
                final_assurance = execution_gate
                break
            if frozen_validation_spec is None:
                raise HRIContractError("Ready plan has no validation specification.")
            validation, validation_gate = self._validate_execution(
                frozen_validation_spec,
                execution,
                attempt,
                event="initial-observation",
            )
            reobservations: list[dict[str, Any]] = []
            for reobservation_index in range(1, self.config.max_reobservations + 1):
                if validation_gate.next_action != "REOBSERVE":
                    break
                observe = getattr(self.executor, "observe", None)
                refreshed_path = str(observe()) if callable(observe) else execution.final_observation
                refreshed = ExecutionResult(
                    status="OBSERVATION_ONLY",
                    final_observation=refreshed_path,
                    subtask_results=copy.deepcopy(execution.subtask_results),
                    evidence={
                        **copy.deepcopy(execution.evidence),
                        "reobservation": reobservation_index,
                    },
                )
                validation, validation_gate = self._validate_execution(
                    frozen_validation_spec,
                    refreshed,
                    attempt,
                    event=f"reobservation-{reobservation_index}",
                )
                reobservations.append(
                    {
                        "index": reobservation_index,
                        "observation": refreshed.to_dict(),
                        "validation": validation.to_dict() if validation else None,
                        "assurance": validation_gate.to_dict(),
                    }
                )
            if reobservations:
                attempt_record["reobservations"] = reobservations
            attempt_record["validation"] = validation.to_dict() if validation else None
            attempt_record["validation_assurance"] = validation_gate.to_dict()
            attempts.append(attempt_record)
            final_assurance = validation_gate
            if validation_gate.outcome in {"SUCCESS", "SUCCESS_RECOVERED"}:
                break
            if validation_gate.next_action not in {"REPLAN", "AUTO_LOCAL"}:
                break
            recovery_context = {
                "previous_plan": plan.to_dict(),
                "frozen_validation_spec": frozen_validation_spec.to_dict()
                if frozen_validation_spec
                else None,
                "validation": validation.to_dict() if validation else None,
                "assurance": validation_gate.to_dict(),
                "instruction": "Plan only corrective actions for unmet frozen goals.",
            }
        if final_assurance is None:
            final_assurance = self.task_assurance.runtime_failure(
                stage="RUNTIME",
                code="NO_TASK_RESULT",
                message="The task ended without an assurance result.",
                next_action="USER_ASSIST",
            )
        self.last_task_result = {
            **final_assurance.to_dict(),
            "attempts": attempts,
        }
        self._display_agent_output(
            "Task Assurance",
            "terminal task result",
            self.last_task_result,
        )
        return copy.deepcopy(self.last_task_result)

    def _dispatch_plan(self, plan: PlanResult, attempt: int) -> ExecutionResult:
        """Dispatch each VLA subtask from HRI when the adapter exposes that API."""
        execute_subtask = getattr(self.executor, "execute_subtask", None)
        observe = getattr(self.executor, "observe", None)
        if not callable(execute_subtask) or not callable(observe):
            return self.executor.execute(plan, attempt=attempt)
        results: list[dict[str, Any]] = []
        for index, subtask in enumerate(plan.subtasks, start=1):
            result = execute_subtask(copy.deepcopy(subtask), index=index, attempt=attempt)
            if not isinstance(result, dict):
                raise HRIContractError("VLA subtask execution must return an object.")
            result = copy.deepcopy(result)
            result.setdefault("index", index)
            results.append(result)
            status = str(result.get("status", "")).upper()
            if status in {"FAILED", "UNSAFE", "ABORTED", "CANCELLED"}:
                return ExecutionResult(
                    status=status,
                    final_observation=str(observe()),
                    subtask_results=results,
                    evidence={"attempt": attempt, "source": "sequential_vla"},
                    error=str(result.get("error") or f"Subtask {index} returned {status}."),
                )
        return ExecutionResult(
            status="COMPLETED"
            if not isinstance(self.executor, RecordedEpisodeExecutor)
            else "OBSERVED_RECORDED_ATTEMPT",
            final_observation=str(observe()),
            subtask_results=results,
            evidence={
                "attempt": attempt,
                "source": "recorded_dataset"
                if isinstance(self.executor, RecordedEpisodeExecutor)
                else "sequential_vla",
                "physical_execution_claimed": not isinstance(
                    self.executor, RecordedEpisodeExecutor
                ),
            },
        )

    def _validate_execution(
        self,
        validation_spec: Any,
        execution: ExecutionResult,
        attempt: int,
        *,
        event: str,
    ) -> tuple[ValidationResult | None, TaskAssuranceResult]:
        try:
            validation = self.validator_agent.validate(validation_spec, execution)
            self._display_agent_output(
                "Validator Agent",
                f"attempt {attempt} parsed output",
                {
                    "parsed": validation.to_dict(),
                    "raw_model_output": copy.deepcopy(validation.raw),
                },
            )
            gate = self.task_assurance.assess_validation(
                validation,
                validation_spec,
                attempt=attempt,
                max_replans=self.config.max_replans,
            )
            self._display_agent_output(
                "Task Assurance",
                f"validation gate attempt {attempt}",
                gate.to_dict(),
            )
            return validation, gate
        except Exception as error:
            failure_evidence = _agent_failure_evidence(
                "VALIDATION",
                error,
                attempt=attempt,
                event=event,
            )
            self._display_agent_output(
                "Validator Agent",
                f"attempt {attempt} error",
                {"error_type": type(error).__name__, "error": str(error)},
            )
            gate = self.task_assurance.runtime_failure(
                stage="VALIDATION",
                code="VALIDATOR_UNAVAILABLE",
                message="The final state could not be verified.",
                next_action="REOBSERVE",
                observed=failure_evidence,
            )
            self._display_agent_output(
                "Task Assurance",
                f"validation runtime gate attempt {attempt}",
                gate.to_dict(),
            )
            return None, gate

    def _finish_episode(
        self,
        hri_response: dict[str, Any],
        task_contract: dict[str, Any] | None,
        task_result: dict[str, Any],
    ) -> None:
        source = self._build_episode_source(hri_response, task_contract, task_result)
        try:
            update = self.memory_agent.update_history_memory(source)
            self._display_agent_output(
                "Memory Agent",
                "history update",
                update,
            )
            self.history_outbox.remove(str(source.get("episode_id", "")))
            raw_context = update.get("history_context")
            if isinstance(raw_context, dict):
                self.history_context = MemoryContext(
                    history_summary=str(raw_context.get("history_summary", "")),
                    relevant_history=copy.deepcopy(raw_context.get("relevant_history", [])),
                    relevant_preferences=copy.deepcopy(
                        raw_context.get(
                            "relevant_preferences",
                            self.history_context.relevant_preferences,
                        )
                    ),
                    history_version=int(raw_context.get("history_version", 0)),
                    preference_version=int(
                        raw_context.get(
                            "preference_version", self.history_context.preference_version
                        )
                    ),
                    warnings=list(map(str, raw_context.get("warnings", []))),
                )
            else:
                self.history_context.history_summary = update["history_summary"].get(
                    "text", ""
                )
                self.history_context.history_version = update["history_version"]
                stored = update.get("episode")
                if isinstance(stored, dict):
                    self.history_context.relevant_history = [
                        copy.deepcopy(stored),
                        *[
                            item
                            for item in self.history_context.relevant_history
                            if item.get("episode_id") != stored.get("episode_id")
                        ],
                    ][: self.config.recent_history_limit]
        except Exception as error:
            self._display_agent_output(
                "Memory Agent",
                "history update error",
                {"error_type": type(error).__name__, "error": str(error)},
            )
            self.history_context.warnings.append(f"history update failed: {error}")
            retry_source = self._history_retry_source(source)
            try:
                self.history_outbox.enqueue(
                    retry_source,
                    str(error),
                    replace_source=True,
                )
            except Exception as outbox_error:
                self.history_context.warnings.append(
                    f"history retry queue failed: {outbox_error}"
                )
        self._reset_command_state(keep_history_context=True)

    def _build_episode_source(
        self,
        hri_response: dict[str, Any],
        task_contract: dict[str, Any] | None,
        task_result: dict[str, Any],
    ) -> dict[str, Any]:
        attempts = task_result.get("attempts", [])
        actions: list[dict[str, Any]] = []
        executions: list[dict[str, Any]] = []
        validations: list[dict[str, Any]] = []
        for attempt in attempts if isinstance(attempts, list) else []:
            if not isinstance(attempt, dict):
                continue
            plan = attempt.get("plan") or {}
            actions.extend(copy.deepcopy(plan.get("subtasks", [])))
            if attempt.get("execution"):
                executions.append(copy.deepcopy(attempt["execution"]))
            if attempt.get("validation"):
                validations.append(copy.deepcopy(attempt["validation"]))
        return {
            "episode_id": self._episode_id,
            "user_id": self.config.user_id,
            "session_id": self.session_id,
            "started_at": self._started_at,
            "completed_at": self._now(),
            "user_request": self._initial_request,
            "dialogue": [turn.to_dict() for turn in self._dialogue],
            "resolved_task": copy.deepcopy(task_contract),
            "actions": actions,
            "execution": executions,
            "validation": validations,
            "user_choices": copy.deepcopy((task_contract or {}).get("parameters", {})),
            "user_visible_result": str(task_result.get("message", hri_response["user_message"])),
            "terminal_outcome": str(task_result.get("outcome", "UNKNOWN")),
            "assurance": {
                "phase": task_result.get("phase"),
                "outcome": task_result.get("outcome"),
                "next_action": task_result.get("next_action"),
                "message": task_result.get("message"),
                "failure": copy.deepcopy(task_result.get("failure")),
            },
            "memory_events": copy.deepcopy(self._memory_events),
            # These fields prove the sanitizer removes internal machinery.
            "trace": copy.deepcopy(hri_response.get("trace")),
            "planning_status": "internal",
            "task_complete": task_result.get("outcome") == "SUCCESS",
        }

    def _maybe_propose_preference(
        self,
        episode_source: dict[str, Any],
        task_result: dict[str, Any],
    ) -> dict[str, Any] | None:
        if str(task_result.get("outcome", "")).upper() not in {
            "SUCCESS",
            "SUCCESS_RECOVERED",
        }:
            return None
        resolved_task = episode_source.get("resolved_task") or {}
        if isinstance(resolved_task, dict) and resolved_task.get("preference_refs"):
            return None
        if any(event.get("committed") is True for event in self._memory_events):
            return None
        proposer = getattr(self.memory_agent, "propose_preference_question", None)
        if not callable(proposer):
            return None
        try:
            proposal = proposer(
                {
                    "user_id": self.config.user_id,
                    "current_episode": copy.deepcopy(episode_source),
                }
            )
            self._display_agent_output(
                "Memory Agent",
                "preference proposal",
                proposal,
            )
        except Exception as error:
            self._display_agent_output(
                "Memory Agent",
                "preference proposal error",
                {"error_type": type(error).__name__, "error": str(error)},
            )
            self._memory_events.append(
                {"action": "PROPOSAL_FAILED", "error": str(error)}
            )
            return None
        return copy.deepcopy(proposal) if isinstance(proposal, dict) else None

    @staticmethod
    def _task_report_response(
        task_result: dict[str, Any],
        *,
        trace: Any = None,
    ) -> dict[str, Any]:
        message = str(task_result.get("message") or "The task has ended.")
        safe_trace = (
            copy.deepcopy(trace)
            if isinstance(trace, dict)
            else {
                "grounding": [],
                "memory_refs": [],
                "history_refs": [],
                "assumptions": [],
            }
        )
        return {
            "mode": "REPORT",
            "user_message": message,
            "task_contract": None,
            "pending_question": None,
            "memory_action": {"action": "NONE"},
            "report": {
                "outcome": str(task_result.get("outcome", "UNKNOWN")),
                "next_action": str(task_result.get("next_action", "NONE")),
            },
            "trace": safe_trace,
        }

    def _reset_command_state(self, *, keep_history_context: bool = True) -> None:
        self._command_active = False
        self.pending_question = None
        self._episode_id = None
        self._started_at = None
        self._initial_request = ""
        self._dialogue = []
        self._memory_events = []
        self._terminal_task_contract = None
        self._terminal_task_result = None
        self._terminal_hri_response = None
        if not keep_history_context:
            self.history_context = MemoryContext()

    def _retry_history_outbox(self) -> None:
        for pending in self.history_outbox.list_pending(self.config.user_id):
            try:
                self.memory_agent.update_history_memory(copy.deepcopy(pending["source"]))
            except Exception:
                continue
            self.history_outbox.remove(str(pending["episode_id"]))

    def _checkpoint_terminal_episode(self, source: dict[str, Any]) -> bool:
        """Durably retain a completed task while a post-task memory reply is pending."""
        try:
            self.history_outbox.enqueue(
                self._history_retry_source(source),
                "Terminal task checkpoint awaiting a memory-consent reply.",
            )
            return True
        except Exception as error:
            self.history_context.warnings.append(
                f"terminal history checkpoint failed: {error}"
            )
            return False

    @staticmethod
    def _history_retry_source(source: dict[str, Any]) -> dict[str, Any]:
        forbidden = {
            "trace",
            "thinking",
            "reasoning",
            "chain_of_thought",
            "planning_status",
            "task_complete",
            "scene_inventory",
        }

        def clean(value: Any) -> Any:
            if isinstance(value, dict):
                return {
                    key: clean(item)
                    for key, item in value.items()
                    if str(key).lower() not in forbidden
                    and not str(key).lower().endswith("_reasoning")
                }
            if isinstance(value, list):
                return [clean(item) for item in value]
            return copy.deepcopy(value)

        return clean(source)

    def switch_dataset(self, selected_path: str) -> bool:
        selected = str(selected_path).strip()
        if not selected:
            return False
        episode = DatasetEpisode.from_path(selected, self.workspace_root)
        if hasattr(self.executor, "set_episode"):
            self.executor.set_episode(episode)  # type: ignore[attr-defined]
        self.dataset_episode = episode
        return True

    def get_response(self, *, display_all: bool = False) -> None:
        """Interactive CLI retained for the thesis prototype."""
        self._display_all = bool(display_all)
        self.output_display.enabled = self._display_all
        print("\n--- Start of Conversation ---\n")
        while True:
            try:
                user_message = input("User: ")
            except (EOFError, KeyboardInterrupt):
                print("\nSession ended.")
                return
            if not user_message.strip():
                continue
            try:
                result = self.handle_user_message(user_message)
            except Exception as error:
                print(f"[PrefMem Error] {error}")
                continue
            print("\nHRI:")
            print(result["hri"]["user_message"])
            if not self.command_active:
                current = self.dataset_episode.display_path(self.workspace_root)
                selected = input(f"Dataset for next task [{current}] (Enter to keep): ").strip()
                if selected:
                    try:
                        self.switch_dataset(selected)
                    except DatasetEpisodeError as error:
                        print(f"[Dataset] {error}; keeping {current}.")

    def _display_agent_output(
        self,
        agent: str,
        stage: str,
        output: Any,
    ) -> None:
        """Print structured diagnostics only when the CLI explicitly enables them."""
        self.output_display.emit(agent, stage, output)

    @staticmethod
    def _validate_hri_output(raw: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(raw, dict):
            raise HRIContractError("HRI model output must be an object.")
        mode = str(raw.get("mode", "")).upper()
        if mode not in HRI_MODES:
            raise HRIContractError(f"Unsupported HRI mode: {mode or '<missing>'}")
        message = str(raw.get("user_message", "")).strip()
        if not message:
            raise HRIContractError("HRI output requires a user_message.")
        result = copy.deepcopy(raw)
        result["mode"] = mode
        result["user_message"] = message
        pending_question = result.get("pending_question")
        if isinstance(pending_question, dict):
            kind = str(pending_question.get("kind", ""))
            if (
                kind in {"TASK_CLARIFICATION", "TASK_CONFIRMATION"}
                and pending_question.get("payload") is None
            ):
                # Some JSON-mode models use null for an empty optional object.
                # Task questions do not carry mutation authority, so normalizing
                # null to an empty object is lossless. MEMORY_CONSENT remains
                # fail-closed because its payload defines the proposed write.
                pending_question["payload"] = {}
        memory_action = result.get("memory_action")
        if memory_action is None:
            result["memory_action"] = {"action": "NONE"}
        elif not isinstance(memory_action, dict):
            raise HRIContractError("memory_action must be an object.")
        return result

    @staticmethod
    def _normalize_hri_transition_output(
        response: dict[str, Any],
        *,
        pending_before: PendingQuestion | None,
        post_task_memory_reply: bool,
    ) -> dict[str, Any]:
        """Repair only lossless, non-authorizing HRI schema variations."""
        result = copy.deepcopy(response)
        pending_raw = result.get("pending_question")

        # ASK and CONFIRM are both current-task-only questions. The question kind
        # carries the precise semantics, so it can safely canonicalize the mode.
        # Never canonicalize a task mode into or out of MEMORY_CONFIRM.
        if result["mode"] in {"ASK", "CONFIRM"} and isinstance(pending_raw, dict):
            task_mode = {
                "TASK_CLARIFICATION": "ASK",
                "TASK_CONFIRMATION": "CONFIRM",
            }.get(str(pending_raw.get("kind", "")))
            if task_mode is not None:
                result["mode"] = task_mode

        # The physical task is already terminal here. A null report from the model
        # is harmless only when it resolves the exact pending MEMORY_CONSENT prompt.
        # Synthesize a non-claiming envelope; the actual transaction still happens
        # later and can independently fail.
        action = str((result.get("memory_action") or {}).get("action", "NONE")).upper()
        memory_outcomes = {
            "COMMIT": "MEMORY_COMMIT_REQUESTED",
            "DELETE": "MEMORY_DELETE_REQUESTED",
            "DECLINE": "MEMORY_DECLINED",
            "DEFER": "MEMORY_DEFERRED",
        }
        if (
            post_task_memory_reply
            and result["mode"] == "REPORT"
            and result.get("report") is None
            and pending_before is not None
            and pending_before.kind == "MEMORY_CONSENT"
            and action in memory_outcomes
        ):
            result["report"] = {
                "outcome": memory_outcomes[action],
                "next_action": "NONE",
            }
        return result

    @staticmethod
    def _validate_hri_transition(
        response: dict[str, Any],
        *,
        pending_before: PendingQuestion | None,
        post_task_memory_reply: bool,
    ) -> None:
        mode = response["mode"]
        pending_raw = response.get("pending_question")
        required_question_kind = {
            "ASK": "TASK_CLARIFICATION",
            "CONFIRM": "TASK_CONFIRMATION",
            "MEMORY_CONFIRM": "MEMORY_CONSENT",
        }.get(mode)
        if required_question_kind is not None:
            if not isinstance(pending_raw, dict) or str(pending_raw.get("kind")) != required_question_kind:
                raise HRIContractError(
                    f"{mode} requires a {required_question_kind} pending question."
                )
        elif pending_raw is not None:
            raise HRIContractError(f"{mode} cannot install a pending question.")
        if mode == "EXECUTE" and not isinstance(response.get("task_contract"), dict):
            raise HRIContractError("EXECUTE requires a task_contract object.")
        if mode != "EXECUTE" and response.get("task_contract") is not None:
            raise HRIContractError(f"{mode} cannot dispatch a task_contract.")
        action = str((response.get("memory_action") or {}).get("action", "NONE")).upper()
        if mode in {"REPORT", "RESTART"} and not isinstance(response.get("report"), dict):
            raise HRIContractError(f"{mode} requires a structured report.")
        if action in {"COMMIT", "DELETE", "DECLINE", "DEFER", "CORRECT", "CANCEL"}:
            explicit = (response.get("memory_action") or {}).get("explicit_consent") is True
            if pending_before is None and not (explicit and action in {"COMMIT", "DELETE"}):
                raise HRIContractError(f"{action} has no memory-consent context.")
        if action in {"DECLINE", "DEFER"} and pending_raw is not None:
            raise HRIContractError(f"{action} must close the memory question, not replace it.")
        if action == "CORRECT" and (
            not isinstance(pending_raw, dict)
            or str(pending_raw.get("kind")) != "MEMORY_CONSENT"
        ):
            raise HRIContractError("CORRECT requires a corrected MEMORY_CONSENT question.")
        if action == "CANCEL" and pending_raw is not None:
            raise HRIContractError("CANCEL must close the pending question.")
        if mode == "RESTART" and (pending_before is None or action != "CANCEL"):
            raise HRIContractError("RESTART requires cancelling an active pending question.")
        if post_task_memory_reply and mode not in {"MEMORY_CONFIRM", "REPORT", "RESTART"}:
            raise HRIContractError(
                "A post-task memory reply may only continue the memory question or report its resolution."
            )
        if post_task_memory_reply:
            if pending_before is None or pending_before.kind != "MEMORY_CONSENT":
                raise HRIContractError(
                    "A post-task memory reply requires the existing MEMORY_CONSENT prompt."
                )
            if mode == "REPORT" and action not in {
                "COMMIT",
                "DELETE",
                "DECLINE",
                "DEFER",
            }:
                raise HRIContractError(
                    "A post-task REPORT must resolve memory with COMMIT, DELETE, DECLINE, or DEFER."
                )
            if mode == "MEMORY_CONFIRM" and action not in {"NONE", "CORRECT"}:
                raise HRIContractError(
                    "MEMORY_CONFIRM may only keep the question open or present a correction."
                )

    def _normalize_task_contract(
        self, raw: dict[str, Any], approval_turn_id: str
    ) -> dict[str, Any]:
        contract = copy.deepcopy(raw)
        confirmed_intent = str(contract.get("confirmed_intent", "")).strip()
        if not confirmed_intent:
            raise HRIContractError("Task contract requires confirmed_intent.")
        parameters = contract.get("parameters", {})
        objects = contract.get("objects", [])
        if not isinstance(parameters, dict):
            raise HRIContractError("Task contract parameters must be an object.")
        if not isinstance(objects, list):
            raise HRIContractError("Task contract objects must be a list.")
        refs = contract.get("preference_refs", contract.get("memory_refs", []))
        if not isinstance(refs, list):
            raise HRIContractError("Task contract preference_refs must be a list.")
        has_conflict = any(
            item.get("match", {}).get("relation") == "CONFLICT"
            and float(item.get("match", {}).get("confidence", 0.0))
            >= self.config.semantic_preference_threshold
            for item in self.history_context.relevant_preferences
        )
        allowed_refs = (
            set()
            if has_conflict
            else {
                str(item.get("id"))
                for item in self.history_context.relevant_preferences
                if item.get("match", {}).get("usable_without_confirmation") is True
            }
        )
        unknown_refs = set(map(str, refs)) - allowed_refs
        if unknown_refs:
            raise HRIContractError(
                f"Task contract cites unapproved or uncertain preference IDs: {sorted(unknown_refs)}"
            )
        contract.update(
            {
                "schema_version": 1,
                "task_id": str(contract.get("task_id") or f"task-{uuid.uuid4().hex}"),
                "episode_id": self._episode_id,
                "user_id": self.config.user_id,
                "confirmed_intent": confirmed_intent,
                "parameters": parameters,
                "objects": list(map(str, objects)),
                "preference_refs": list(map(str, refs)),
                "approval": {
                    "turn_id": approval_turn_id,
                    "source": "hri_resolved_user_turn",
                },
            }
        )
        return contract

    def _verify_direct_memory_consent(
        self,
        *,
        user_message: str,
        requested_action: str,
        proposed_request: Any,
    ) -> None:
        if not isinstance(proposed_request, dict):
            raise HRIContractError("Direct memory consent requires a structured proposal.")
        try:
            verdict = self.hri_model.generate(
                purpose="verify_direct_memory_consent",
                system_prompt=self.hri_prompt,
                payload={
                    "event": "DIRECT_MEMORY_CONSENT_CHECK",
                    "exact_user_message": user_message,
                    "requested_action": requested_action,
                    "proposed_request": copy.deepcopy(proposed_request),
                    "rule": (
                        "Authorize only if the exact user message semantically entails this durable "
                        "future-memory or forgetting operation. Current-task wording is insufficient."
                    ),
                },
            )
        except Exception as error:
            raise HRIContractError(
                "Direct durable-memory consent could not be verified; ask a dedicated question."
            ) from error
        try:
            confidence = float(verdict.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        if (
            verdict.get("entailed") is not True
            or str(verdict.get("requested_action", "")).upper() != requested_action
            or confidence < 0.9
        ):
            raise HRIContractError(
                "The current user message does not clearly authorize that durable memory change."
            )

    def _handle_safety_clearance(self, user_message: str) -> dict[str, Any] | None:
        """Keep physical dispatch latched until the user explicitly clears safety."""
        try:
            verdict = self.hri_model.generate(
                purpose="verify_safety_clearance",
                system_prompt=self.hri_prompt,
                payload={
                    "event": "SAFETY_CLEARANCE_CHECK",
                    "exact_user_message": user_message,
                    "latched_safety_event": copy.deepcopy(self._safety_event),
                    "rule": (
                        "Clear only on an explicit assertion that the hazard was checked and "
                        "it is safe to resume. A bare retry command is insufficient."
                    ),
                },
            )
        except Exception:
            verdict = {"cleared": False}
        if verdict.get("cleared") is not True:
            return {
                "hri": {
                    "mode": "REPORT",
                    "user_message": (
                        "Execution remains safety-stopped. Please confirm that the hazard has "
                        "been checked and it is safe before asking me to resume."
                    ),
                    "task_contract": None,
                    "pending_question": None,
                    "memory_action": {"action": "NONE"},
                    "report": {"outcome": "ABORTED_SAFETY", "next_action": "SAFETY_CLEARANCE"},
                    "trace": {"grounding": [], "memory_refs": [], "history_refs": [], "assumptions": []},
                },
                "memory": None,
                "task": None,
                "awaiting_user": True,
            }
        self._safety_latched = False
        self._safety_event = None
        if verdict.get("contains_task_request") is True:
            return None
        return {
            "hri": {
                "mode": "REPORT",
                "user_message": "The safety stop is cleared. I am ready for a new task.",
                "task_contract": None,
                "pending_question": None,
                "memory_action": {"action": "NONE"},
                "report": {"outcome": "SAFETY_CLEARED", "next_action": "NONE"},
                "trace": {"grounding": [], "memory_refs": [], "history_refs": [], "assumptions": []},
            },
            "memory": None,
            "task": None,
            "awaiting_user": False,
        }

    def _display_path(self, path: Path) -> str:
        try:
            return path.relative_to(self.workspace_root).as_posix()
        except ValueError:
            return str(path)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()


HRI_Agent = HRIOrchestrator

__all__ = ["HRIOrchestrator", "HRI_Agent", "HRI_Agent_Config", "HRIContractError"]
