"""Execute deterministic, multi-turn PrefMem benchmark conversations."""

from __future__ import annotations

import copy
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping

from dataset.benchmark import BenchmarkEpisode
from memory.models import ConsentEvidence

from .conversation_cases import OPPOSITE_TARGETS, TARGET_REPLIES
from .conversation_models import (
    CommandSpec,
    ConversationRunContext,
    PreferenceFixture,
    stable_digest,
)
from .conversation_runtime import default_conversation_orchestrator_factory
from .executor import CounterfactualSequenceExecutor
from .semantic import infer_target_id


ConversationOrchestratorFactory = Callable[[ConversationRunContext], Any]


class ConversationRunError(RuntimeError):
    pass


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}
def _sequence(value: Any) -> tuple[Any, ...] | list[Any]:
    if isinstance(value, (list, tuple)):
        return value
    return []




def _snapshot_memory(orchestrator: Any, user_id: str) -> dict[str, Any]:
    try:
        memory_agent = getattr(orchestrator, "memory_agent", None)
        history_repository = getattr(memory_agent, "history_repository", None)
        preference_repository = getattr(
            memory_agent, "preference_repository", None
        )
        history_reader = getattr(history_repository, "list_episodes", None)
        summary_reader = getattr(history_repository, "get_summary", None)
        preference_reader = getattr(
            preference_repository, "list_preferences", None
        )
        if (
            not callable(history_reader)
            or not callable(summary_reader)
            or not callable(preference_reader)
        ):
            return {
                "available": False,
                "error": "memory repositories are not inspectable",
            }
        histories = history_reader(user_id)
        history_summary = summary_reader(user_id)
        if not isinstance(history_summary, Mapping):
            return {
                "available": False,
                "error": "history summary is not inspectable",
            }
        try:
            preferences = preference_reader(
                user_id,
                include_inactive=True,
            )
        except TypeError:
            preferences = preference_reader(user_id)
        outbox = getattr(orchestrator, "history_outbox", None)
        outbox_reader = getattr(outbox, "list_pending", None)
        if not callable(outbox_reader):
            return {
                "available": False,
                "error": "history outbox is not inspectable",
            }
        outbox_records = list(outbox_reader(user_id))

        def history_id(item: Any) -> str:
            return str(_mapping(item).get("episode_id", "")).strip()

        def preference_id(item: Any) -> str:
            return str(_mapping(item).get("id", "")).strip()

        def outbox_id(item: Any) -> str:
            record = _mapping(item)
            source = _mapping(record.get("source"))
            return str(
                source.get("episode_id", record.get("episode_id", ""))
            ).strip()

        history_ids = sorted(filter(None, map(history_id, histories)))
        preference_ids = sorted(filter(None, map(preference_id, preferences)))
        outbox_ids = sorted(filter(None, map(outbox_id, outbox_records)))
        return {
            "available": True,
            "history_summary": copy.deepcopy(dict(history_summary)),
            "history_state_digest": stable_digest(
                {
                    "records": sorted(histories, key=history_id),
                    "summary": history_summary,
                }
            ),
            "preference_state_digest": stable_digest(
                sorted(preferences, key=preference_id)
            ),
            "outbox_state_digest": stable_digest(
                sorted(outbox_records, key=outbox_id)
            ),
            "history_ids": history_ids,
            "preference_ids": preference_ids,
            "outbox_episode_ids": outbox_ids,
            "history_count": len(history_ids),
            "preference_count": len(preference_ids),
            "outbox_count": len(outbox_ids),
            "history_version": getattr(history_repository, "version", None),
            "preference_version": getattr(
                preference_repository, "version", None
            ),
        }
    except Exception as error:
        return {
            "available": False,
            "error": f"{type(error).__name__}: {error}",
        }


def _memory_delta(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> dict[str, Any]:
    if before.get("available") is not True or after.get("available") is not True:
        return {
            "history_delta": None,
            "preference_delta": None,
            "outbox_delta": None,
            "history_state_changed": None,
            "preference_state_changed": None,
            "outbox_state_changed": None,
            "history_version_delta": None,
            "preference_version_delta": None,
        }
    def version_delta(key: str) -> int | None:
        before_value = before.get(key)
        after_value = after.get(key)
        return (
            after_value - before_value
            if type(before_value) is int and type(after_value) is int
            else None
        )
    return {
        "history_delta": int(after["history_count"]) - int(before["history_count"]),
        "preference_delta": int(after["preference_count"]) - int(before["preference_count"]),
        "outbox_delta": int(after["outbox_count"]) - int(before["outbox_count"]),
        "history_state_changed": before.get("history_state_digest") != after.get("history_state_digest"),
        "preference_state_changed": before.get("preference_state_digest") != after.get("preference_state_digest"),
        "outbox_state_changed": before.get("outbox_state_digest") != after.get("outbox_state_digest"),
        "history_version_delta": version_delta("history_version"),
        "preference_version_delta": version_delta("preference_version"),
    }


def _ensure_isolated(snapshot: Mapping[str, Any], *, user_id: str) -> None:
    if snapshot.get("available") is not True:
        raise ConversationRunError(
            f"Cannot verify cold memory for {user_id}: {snapshot.get('error')}"
        )
    if any(
        int(snapshot.get(key, 0))
        for key in ("history_count", "preference_count", "outbox_count")
    ):
        raise ConversationRunError(
            f"Conversation memory for {user_id} was not empty at run start."
        )
    summary = _mapping(snapshot.get("history_summary"))
    if str(summary.get("text", "")).strip() or _sequence(
        summary.get("source_episode_ids")
    ):
        raise ConversationRunError(
            "Conversation history summary for "
            f"{user_id} was not empty at run start."
        )


def _apply_fixture(orchestrator: Any, fixture: PreferenceFixture) -> dict[str, Any]:
    memory_agent = getattr(orchestrator, "memory_agent", None)
    repository = getattr(memory_agent, "preference_repository", None)
    apply_transaction = getattr(repository, "apply_transaction", None)
    if not callable(apply_transaction):
        raise ConversationRunError(
            "Preference fixture requires an inspectable PreferenceRepository."
        )
    transaction_id = f"fixture-{uuid.uuid4().hex}"
    proposal = {
        "instruction": fixture.statement,
        "preference": fixture.to_dict(),
    }
    consent = ConsentEvidence(
        kind="explicit_future_language",
        quote="Remember this preference for future tasks.",
        turn_id=f"turn-{transaction_id}",
        episode_id=f"episode-{transaction_id}",
        authorized_action="UPSERT",
        proposal=copy.deepcopy(proposal),
    )
    transaction = apply_transaction(
        user_id=fixture.owner_user_id,
        transaction_id=transaction_id,
        operations=[
            {
                "operation_id": f"op-{transaction_id}",
                "action": "ADD",
                "preference": {
                    "statement": fixture.statement,
                    "scope": fixture.task_type_hint,
                    "applicability": copy.deepcopy(fixture.applicability),
                    "structured_value": copy.deepcopy(
                        fixture.structured_value
                    ),
                },
                "evidence": [
                    {"episode_id": f"episode-{transaction_id}"}
                ],
            }
        ],
        authorization=consent,
        requested_action="UPSERT",
    )
    return copy.deepcopy(transaction)


def _delivery_for_turn(orchestrator: Any, user_id: str) -> dict[str, Any] | None:
    memory_agent = getattr(orchestrator, "memory_agent", None)
    reader = getattr(memory_agent, "delivery_for_current_turn", None)
    if not callable(reader):
        return None
    delivery = reader(
        attached_context=getattr(orchestrator, "history_context", None),
        user_id=user_id,
    )
    if delivery is None:
        return None
    to_dict = getattr(delivery, "to_dict", None)
    return copy.deepcopy(to_dict() if callable(to_dict) else delivery)


def _strict_proposed_target(
    manifest: Mapping[str, Any],
    pending_question: Mapping[str, Any],
) -> tuple[str | None, bool]:
    payload = _mapping(pending_question.get("payload"))
    proposal = _mapping(payload.get("proposed_task"))
    if not proposal:
        return None, False
    structured = {
        "confirmed_intent": "",
        "task_type": proposal.get("task_type"),
        "objects": copy.deepcopy(proposal.get("objects", [])),
        "parameters": copy.deepcopy(proposal.get("parameters", {})),
    }
    return infer_target_id(manifest, structured), True


def _memory_reply(policy: str) -> str | None:
    return {
        "commit": "Yes, remember that preference for future tasks.",
        "decline": "No, do not save that as a preference.",
        "defer": "Not now; ask me again later.",
        "keep-open": "Could you explain what you would remember?",
    }.get(policy)


def _task_reply(
    command: CommandSpec,
    manifest: Mapping[str, Any],
    pending_question: Mapping[str, Any],
) -> tuple[str | None, dict[str, Any]]:
    target_id = command.target_id
    if target_id is None or command.response_policy == "none":
        return None, {
            "kind": "task_resolution",
            "status": "NOT_APPLICABLE",
        }
    kind = str(pending_question.get("kind", ""))
    if kind == "TASK_CLARIFICATION":
        return TARGET_REPLIES[target_id], {
            "kind": "task_resolution",
            "status": "RESOLVED_EXPLICITLY",
        }
    if kind != "TASK_CONFIRMATION":
        return None, {
            "kind": "task_resolution",
            "status": "UNSUPPORTED_QUESTION",
            "pending_kind": kind,
        }
    proposed, present = _strict_proposed_target(
        manifest,
        pending_question,
    )
    evidence = {
        "kind": "structured_confirmation",
        "proposal_present": present,
        "proposed_target_id": proposed,
        "desired_target_id": target_id,
    }
    if proposed == target_id:
        return "Yes.", {**evidence, "status": "ACCEPTED_DESIRED"}
    if proposed == OPPOSITE_TARGETS.get(target_id):
        return "No, do the opposite.", {
            **evidence,
            "status": "REJECTED_OPPOSITE",
        }
    return f"No. {TARGET_REPLIES[target_id]}", {
        **evidence,
        "status": "SAFE_EXPLICIT_CORRECTION",
    }


def scripted_reply(
    command: CommandSpec,
    manifest: Mapping[str, Any],
    result: Mapping[str, Any],
) -> tuple[str | None, dict[str, Any]]:
    hri = _mapping(result.get("hri"))
    pending = _mapping(hri.get("pending_question"))
    if not pending:
        pending = _mapping(result.get("pending_question"))
    kind = str(pending.get("kind", ""))
    if kind == "MEMORY_CONSENT":
        reply = _memory_reply(command.memory_consent_policy)
        return reply, {
            "kind": "memory_consent",
            "status": command.memory_consent_policy.upper(),
            "prompt_id": pending.get("prompt_id"),
        }
    return _task_reply(command, manifest, pending)


def _configure_command(
    orchestrator: Any,
    context: ConversationRunContext,
    command: CommandSpec,
) -> None:
    primary = context.episodes[command.episode_id]
    switch = getattr(orchestrator, "switch_dataset", None)
    if not callable(switch) or switch(str(primary.path)) is not True:
        raise ConversationRunError(
            f"Could not switch to packet {command.episode_id}."
        )
    executor = getattr(orchestrator, "executor", None)
    configure = getattr(executor, "configure_recovery", None)
    if callable(configure):
        recovery = (
            BenchmarkEpisode.from_path(
                context.episodes[command.recovery_episode_id].path,
                context.config.benchmark_root,
            )
            if command.recovery_episode_id
            else None
        )
        configure(recovery)
    elif command.recovery_episode_id:
        raise ConversationRunError(
            "Recovery cases require CounterfactualSequenceExecutor."
        )


def _run_command(
    orchestrator: Any,
    context: ConversationRunContext,
    command: CommandSpec,
) -> dict[str, Any]:
    _configure_command(orchestrator, context, command)
    descriptor = context.episodes[command.episode_id]
    executor = getattr(orchestrator, "executor", None)
    initial_dispatch_count = getattr(executor, "dispatch_count", None)
    before = _snapshot_memory(orchestrator, context.case.user_id)
    message: str | None = command.query
    turns: list[dict[str, Any]] = []
    script_events: list[dict[str, Any]] = []
    task_result: dict[str, Any] | None = None
    resolved_task: dict[str, Any] | None = None
    for turn_index in range(1, context.config.max_user_turns + 1):
        if message is None:
            break
        memory_agent = getattr(orchestrator, "memory_agent", None)
        begin_turn = getattr(memory_agent, "begin_evaluation_turn", None)
        if callable(begin_turn):
            begin_turn()
        turn_before = _snapshot_memory(orchestrator, context.case.user_id)
        result = orchestrator.handle_user_message(message)
        if not isinstance(result, Mapping):
            raise ConversationRunError(
                "HRI handle_user_message must return an object."
            )
        result = copy.deepcopy(dict(result))
        turn_after = _snapshot_memory(orchestrator, context.case.user_id)
        turn_task = result.get("task")
        if isinstance(turn_task, Mapping):
            task_result = copy.deepcopy(dict(turn_task))
        turn_contract = result.get("resolved_task")
        if isinstance(turn_contract, Mapping):
            resolved_task = copy.deepcopy(dict(turn_contract))
        decision = _mapping(result.get("hri_decision"))
        if not decision:
            decision = _mapping(result.get("hri"))
        turns.append(
            {
                "turn": turn_index,
                "user_message": message,
                "decision_mode": str(decision.get("mode", "")).upper(),
                "result": result,
                "memory_before": turn_before,
                "memory_after": turn_after,
                "memory_delta": _memory_delta(turn_before, turn_after),
                "delivered_memory": _delivery_for_turn(
                    orchestrator,
                    context.case.user_id,
                ),
            }
        )
        if result.get("awaiting_user") is not True:
            message = None
            break
        message, event = scripted_reply(
            command,
            descriptor.manifest,
            result,
        )
        script_events.append(event)

    after = _snapshot_memory(orchestrator, context.case.user_id)
    final_dispatch_count = getattr(executor, "dispatch_count", None)
    dispatch_delta = (
        final_dispatch_count - initial_dispatch_count
        if isinstance(initial_dispatch_count, int)
        and isinstance(final_dispatch_count, int)
        else None
    )
    return {
        "command": command.to_dict(),
        "episode": descriptor.public_metadata(),
        "turns": turns,
        "script_events": script_events,
        "task": task_result,
        "resolved_task": resolved_task,
        "memory_before": before,
        "memory_after": after,
        "memory_delta": _memory_delta(before, after),
        "vla_dispatch_delta": dispatch_delta,
        "command_complete": not bool(
            getattr(orchestrator, "command_active", False)
        ),
        "turn_limit_reached": (
            len(turns) >= context.config.max_user_turns
            and bool(getattr(orchestrator, "command_active", False))
        ),
    }


def run_conversation_case(
    context: ConversationRunContext,
    *,
    orchestrator_factory: ConversationOrchestratorFactory = (
        default_conversation_orchestrator_factory
    ),
) -> dict[str, Any]:
    """Run one isolated conversation case without interpreting its oracle."""

    started = time.monotonic()
    context.run_directory.mkdir(parents=True, exist_ok=True)
    orchestrator = orchestrator_factory(context)
    initial = _snapshot_memory(orchestrator, context.case.user_id)
    _ensure_isolated(initial, user_id=context.case.user_id)
    fixture_result: dict[str, Any] | None = None
    if context.case.fixture is not None:
        owner_initial = _snapshot_memory(
            orchestrator,
            context.case.fixture.owner_user_id,
        )
        _ensure_isolated(
            owner_initial,
            user_id=context.case.fixture.owner_user_id,
        )
        fixture_result = _apply_fixture(orchestrator, context.case.fixture)

    commands: list[dict[str, Any]] = []
    for command in context.case.commands:
        record = _run_command(orchestrator, context, command)
        commands.append(record)
        if not record["command_complete"]:
            break
    artifact_reader = getattr(
        orchestrator, "evaluation_artifact_status", None
    )
    artifact_status = (
        copy.deepcopy(artifact_reader())
        if callable(artifact_reader)
        else {"complete": None, "reason": "not exposed by orchestrator"}
    )
    return {
        "schema_version": "robopref.conversation-run.v1",
        "case": context.case.to_dict(),
        "case_digest": context.case.digest,
        "repetition": context.repetition,
        "model_seed": context.model_seed,
        "run_directory": str(context.run_directory),
        "initial_memory": initial,
        "fixture_transaction": fixture_result,
        "commands": commands,
        "artifact_status": artifact_status,
        "duration_seconds": time.monotonic() - started,
    }


__all__ = [
    "ConversationOrchestratorFactory",
    "ConversationRunError",
    "run_conversation_case",
    "scripted_reply",
]

