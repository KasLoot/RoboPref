"""Scene-grounded, stateless Planner model boundary.

The Planner is deliberately not the owner of execution state.  A preview call
helps the HRI present a nominal strategy before confirmation.  After
confirmation, each cycle receives a fresh image and a complete, host-owned
request containing the frozen goal and terminal execution history.  Only the
first task returned by a cycle is eligible for publication by the controller.
"""

from __future__ import annotations

import json
import operator
import re
from collections.abc import Mapping, Sequence
from time import perf_counter
from typing import TYPE_CHECKING, Any

from colorama import init as colorama_init
from langchain.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from termcolor import colored
from typing_extensions import Annotated, TypedDict

from prefmem.agents.config import Planner_Config
from prefmem.agents.metrics import TurnMetrics
from prefmem.agents.vision import get_live_frame

if TYPE_CHECKING:
    from prefmem.contracts import (
        GoalProposal,
        PlannerCycleRequest,
        PlannerDecision,
    )


colorama_init()


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int


class CurrentFrameContext(TypedDict):
    current_frame: dict[str, object]


class PlannerOutputError(ValueError):
    """Raised when all Planner responses violate the typed JSON contract."""


class VLLMChatOpenAI(ChatOpenAI):
    """Preserve vLLM reasoning and request metrics on streamed chunks."""

    def _convert_chunk_to_generation_chunk(
        self,
        chunk,
        default_chunk_class,
        base_generation_info,
    ):
        generation = super()._convert_chunk_to_generation_chunk(
            chunk,
            default_chunk_class,
            base_generation_info,
        )
        choices = chunk.get("choices") or chunk.get("chunk", {}).get("choices") or []
        if generation is not None and choices:
            delta = choices[0].get("delta") or {}
            reasoning = delta.get("reasoning") or delta.get("reasoning_content")
            if reasoning:
                generation.message.additional_kwargs["reasoning"] = reasoning

        request_metrics = chunk.get("metrics") or chunk.get("chunk", {}).get("metrics")
        if generation is not None and request_metrics:
            generation.message.response_metadata["vllm_metrics"] = request_metrics
        return generation


_THINKING_BLOCK = re.compile(
    r"<(?P<tag>think|thinking|analysis|reasoning)>.*?</(?P=tag)>",
    flags=re.IGNORECASE | re.DOTALL,
)
_JSON_FENCE = re.compile(
    r"```(?:json)?\s*(?P<body>.*?)\s*```",
    flags=re.IGNORECASE | re.DOTALL,
)

_MAX_SCHEMA_CORRECTIONS = 2
_CONTRACT_CORRECTION_RULES = {
    "PlannerDecision": (
        "For ACT, candidate_tasks must contain 1 to 3 tasks; for every "
        "other decision it must be empty.",
        "Each candidate_tasks[i].expected_observation must contain 1 to 3 "
        "non-empty strings.",
        "Each task's expected_observation is task-local, not a copy of the "
        "goal-wide final checklist. Merge related observable conditions into "
        "at most 3 strings while preserving the complete relevant post-state.",
    ),
}


def _response_text(response: object) -> str:
    """Return visible text from a LangChain response or a raw test response."""

    content = getattr(response, "content", response)
    if isinstance(content, str):
        if not content.strip():
            raise PlannerOutputError("Planner returned an empty response")
        return content

    if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
                continue
            if not isinstance(block, Mapping):
                continue
            block_type = block.get("type")
            # Reasoning blocks are intentionally not interpreted as answer JSON.
            if block_type in {"reasoning", "thinking", "analysis"}:
                continue
            text = block.get("text")
            if not isinstance(text, str):
                text = block.get("content")
            if isinstance(text, str):
                parts.append(text)
        result = "".join(parts)
        if result.strip():
            return result

    raise PlannerOutputError("Planner response did not contain visible text")


def parse_json_object(response: object) -> dict[str, Any]:
    """Extract exactly one JSON object while tolerating Gemma wrappers.

    Gemma/vLLM can expose an empty or populated ``<think>`` block even when the
    application asks for JSON only.  Markdown JSON fences are also common.  We
    remove only those transport wrappers; the resulting JSON object is still
    decoded normally and its schema is validated strictly by the contract.
    """

    text = _THINKING_BLOCK.sub("", _response_text(response)).strip()
    fence_matches = list(_JSON_FENCE.finditer(text))
    if fence_matches:
        outside_fences = _JSON_FENCE.sub("", text).strip()
        if outside_fences:
            raise PlannerOutputError(
                "Planner response contains content outside its JSON fence"
            )
        candidates = [match.group("body") for match in fence_matches]
    else:
        candidates = [text]
    decoded: list[dict[str, Any]] = []

    for candidate in candidates:
        candidate = candidate.strip()
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            # Some model servers leave harmless turn markers or ghost text in
            # front of the answer.  Find the earliest complete JSON object,
            # but do not accept non-whitespace after it.
            decoder = json.JSONDecoder()
            index = candidate.find("{")
            if index < 0:
                continue
            try:
                value, end = decoder.raw_decode(candidate, index)
            except json.JSONDecodeError:
                continue
            if candidate[end:].strip():
                continue

        if isinstance(value, dict):
            decoded.append(value)

    if len(decoded) != 1:
        if not decoded:
            raise PlannerOutputError("Planner response must contain one JSON object")
        raise PlannerOutputError("Planner response contains multiple JSON objects")
    return decoded[0]


def _contract_class(name: str):
    """Resolve contracts lazily so this boundary remains easy to unit test."""

    from prefmem import contracts

    try:
        return getattr(contracts, name)
    except AttributeError as error:  # pragma: no cover - integration guard
        raise RuntimeError(f"prefmem.contracts does not expose {name}") from error


def _model_payload(value: object, *, field_name: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, Mapping):
            return dict(payload)
    raise TypeError(f"{field_name} must be a mapping or expose to_dict()")


def _request_text(payload: Mapping[str, Any]) -> str:
    """Serialize request data without relying on HRI conversation history."""

    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _human_message(current_frame: Mapping[str, Any], text: str) -> HumanMessage:
    if not isinstance(current_frame, Mapping):
        raise TypeError("current_frame must be a model-compatible image mapping")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Planner request text must be non-empty")
    # Gemma's required order is system turn, then image, then request text.
    return HumanMessage(
        content=[
            dict(current_frame),
            {"type": "text", "text": text},
        ]
    )


class Planner_Agent:
    """Planner VLM facade with typed preview and receding-horizon calls."""

    def __init__(
        self,
        model_config: str = "vllm",
        args=None,
        metrics: TurnMetrics | None = None,
        *,
        model_name: str | None = None,
        model_base_url: str | None = None,
        model: object | None = None,
    ) -> None:
        self.config = Planner_Config(
            model_config,
            model=model_name,
            model_base_url=model_base_url,
        )
        self.args = args
        self.metrics = metrics
        self.llm = model or VLLMChatOpenAI(
            model=self.config.model,
            api_key="EMPTY",
            base_url=self.config.model_base_url,
            max_tokens=2048,
            temperature=0,
            streaming=True,
            stream_usage=True,
            timeout=300,
        )
        self.system_prompt = self.config.system_prompt
        think = getattr(args, "think", ()) if args is not None else ()
        if isinstance(think, str):
            enabled_agents = (think,)
        elif isinstance(think, Sequence):
            enabled_agents = tuple(think)
        else:
            enabled_agents = ()
        self.thinking_enabled = "all" in enabled_agents or "Planner" in enabled_agents
        self.print_raw = bool(getattr(args, "print_raw", False))
        self.agent = self.build_agent()

    def build_agent(self):
        """Build the legacy streaming entry point used by the HRI shell."""

        builder = StateGraph(MessagesState, context_schema=CurrentFrameContext)
        builder.add_node("Planner Agent", self.llm_call)
        builder.add_edge(START, "Planner Agent")
        builder.add_edge("Planner Agent", END)
        return builder.compile()

    def _request_options(self) -> dict[str, str]:
        return {"reasoning_effort": "high"} if self.thinking_enabled else {}

    def _record_metrics(
        self,
        prompt_messages: list[AnyMessage],
        response: object,
        elapsed_seconds: float,
    ) -> None:
        if self.metrics is not None:
            self.metrics.record(
                agent="Planner Agent",
                prompt_messages=prompt_messages,
                response=response,
                elapsed_seconds=elapsed_seconds,
            )

    def _invoke_model(
        self,
        request_text: str,
        current_frame: Mapping[str, Any],
    ) -> object:
        prompt_messages: list[AnyMessage] = [
            SystemMessage(content=self.system_prompt),
            _human_message(current_frame, request_text),
        ]
        started = perf_counter()
        response = self.llm.invoke(prompt_messages, **self._request_options())
        self._record_metrics(
            prompt_messages,
            response,
            perf_counter() - started,
        )
        return response

    def _invoke_typed(
        self,
        *,
        request_payload: Mapping[str, Any],
        current_frame: Mapping[str, Any],
        contract_name: str,
    ):
        """Call and validate the model with bounded, targeted corrections."""

        contract = _contract_class(contract_name)
        request = _request_text(request_payload)
        last_error: Exception | None = None
        invalid_output = ""

        for attempt in range(_MAX_SCHEMA_CORRECTIONS + 1):
            if attempt == 0:
                model_request = request
            else:
                correction_rules = _CONTRACT_CORRECTION_RULES.get(
                    contract_name,
                    (),
                )
                model_request = _request_text(
                    {
                        "request": request_payload,
                        "schema_correction": {
                            "attempt": attempt,
                            "contract": contract_name,
                            "error": str(last_error),
                            "invalid_output": invalid_output,
                            "contract_rules": list(correction_rules),
                            "instruction": (
                                "Return one corrected JSON object only. Preserve "
                                "the requested goal and semantics, but fix every "
                                "invalid field shape and count. Do not preserve "
                                "an invalid array merely to keep clauses separate; "
                                "merge related clauses when needed to satisfy its "
                                "bound."
                            ),
                        },
                    }
                )

            response = self._invoke_model(model_request, current_frame)
            try:
                payload = parse_json_object(response)
                return contract.from_dict(payload)
            except (TypeError, ValueError) as error:
                last_error = error
                try:
                    invalid_output = _response_text(response)
                except PlannerOutputError:
                    invalid_output = "<no visible response>"

        raise PlannerOutputError(
            f"Planner returned invalid {contract_name} JSON after "
            f"{_MAX_SCHEMA_CORRECTIONS} schema corrections: "
            f"{last_error}"
        ) from last_error

    def preview(
        self,
        clarified_goal: str,
        current_frame: Mapping[str, Any],
        *,
        constraints: Sequence[str] = (),
        operator_guidance: str | None = None,
    ) -> GoalProposal:
        """Generate the nominal, non-executable proposal shown for confirmation."""

        if not isinstance(clarified_goal, str) or not clarified_goal.strip():
            raise ValueError("clarified_goal must be a non-empty string")
        if isinstance(constraints, (str, bytes)) or not isinstance(constraints, Sequence):
            raise TypeError("constraints must be a sequence of strings")
        clean_constraints: list[str] = []
        for index, constraint in enumerate(constraints):
            if not isinstance(constraint, str) or not constraint.strip():
                raise ValueError(f"constraints[{index}] must be a non-empty string")
            clean_constraints.append(constraint.strip())
        if operator_guidance is not None and (
            not isinstance(operator_guidance, str) or not operator_guidance.strip()
        ):
            raise ValueError("operator_guidance must be non-empty when provided")

        return self._invoke_typed(
            request_payload={
                "request_kind": "PREVIEW",
                "clarified_goal": clarified_goal.strip(),
                "constraints": clean_constraints,
                "operator_guidance": (
                    operator_guidance.strip() if operator_guidance is not None else None
                ),
            },
            current_frame=current_frame,
            contract_name="GoalProposal",
        )

    # A descriptive alias makes coordinator code read naturally.
    request_preview = preview

    def plan_cycle(
        self,
        request: PlannerCycleRequest,
        current_frame: Mapping[str, Any],
    ) -> PlannerDecision:
        """Plan from a fresh frame, frozen goal, trigger, and terminal history."""

        payload = _model_payload(request, field_name="request")
        return self._invoke_typed(
            request_payload={"request_kind": "PLAN_CYCLE", **payload},
            current_frame=current_frame,
            contract_name="PlannerDecision",
        )

    run_cycle = plan_cycle

    def llm_call(self, state: dict, runtime: Runtime[CurrentFrameContext]):
        """Legacy graph node; image remains before the final human text."""

        model_messages = list(state["messages"])
        for index in range(len(model_messages) - 1, -1, -1):
            message = model_messages[index]
            if not isinstance(message, HumanMessage):
                continue
            content = message.content
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False, default=str)
            model_messages[index] = _human_message(
                runtime.context["current_frame"],
                content,
            )
            break

        prompt_messages = [
            SystemMessage(content=getattr(self, "system_prompt", self.config.system_prompt))
        ] + model_messages
        started = perf_counter()
        answer = self.llm.invoke(prompt_messages, **self._request_options())
        self._record_metrics(prompt_messages, answer, perf_counter() - started)
        return {
            "messages": [answer],
            "llm_calls": state.get("llm_calls", 0) + 1,
        }

    def invoke_agent(self, messages, current_frame) -> dict:
        """Stream the legacy graph once and return its terminal state."""

        final_state = None
        for part in self.agent.stream(
            input={"messages": messages},
            config={"recursion_limit": 20},
            context={"current_frame": current_frame},
            stream_mode=["messages", "updates", "values"],
            version="v2",
        ):
            if part["type"] == "values":
                final_state = part["data"]
                continue
            if part["type"] != "updates":
                continue
            for node_name, update in part["data"].items():
                if node_name != "Planner Agent" or not isinstance(update, dict):
                    continue
                for message in update.get("messages", []):
                    if not isinstance(message, AIMessage):
                        continue
                    reasoning = message.additional_kwargs.get("reasoning")
                    if reasoning:
                        print(colored("Planner Agent Thinking:", "black", "on_white"))
                        print(reasoning)
                    if message.content:
                        print(colored("Planner Agent Response:", "black", "on_white"))
                        print(message.content)
                    if self.print_raw:
                        print(colored("Planner Agent Raw Response:", "black", "on_white"))
                        print(
                            json.dumps(
                                message.model_dump(mode="json"),
                                indent=2,
                                ensure_ascii=False,
                                default=str,
                            )
                        )

        if final_state is None:
            raise RuntimeError("Planner completed without producing final state.")
        return final_state

    def run(self, messages, current_frame=None):
        """Legacy HRI entry point.

        Passing a frame is authoritative: no second live snapshot is fetched.
        The fallback fetch remains only for old callers that provide no frame.
        """

        if current_frame is None:
            current_frame = get_live_frame()
        print(colored("\nPlanner Agent:", "white", "on_blue"))
        return self.invoke_agent(messages, current_frame=current_frame)
