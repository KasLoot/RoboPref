from langchain.tools import tool
from langchain.chat_models import init_chat_model
import os
from langchain.messages import AIMessage, AnyMessage
from typing_extensions import TypedDict, Annotated, Literal
from prefmem.agents.config import HRI_Config
from langchain.messages import ToolMessage

from langgraph.graph import StateGraph, START, END
import operator
from langchain.messages import SystemMessage, HumanMessage

from langchain_openai import ChatOpenAI

from langchain_core.tools import StructuredTool

from IPython.display import Image, display
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field, field_validator
from langgraph.checkpoint.memory import InMemorySaver  
from langgraph.runtime import Runtime
from prefmem.agents.vision import image_data_url, get_start_end_frames, get_live_frame
from prefmem.agents.planner import Planner_Agent
from prefmem.agents.memory import Memory_Agent
import json
import queue
import threading
import uuid

from time import perf_counter

from prefmem.agents.metrics import (
    TurnMetrics,
    VLLMTokenCounter,
)

from colorama import init
from termcolor import colored
init()





class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int


class CurrentFrameContext(TypedDict):
    current_frame: dict


class StrictToolInput(BaseModel):
    """Reject coercion and unknown fields at an HRI control boundary."""

    model_config = ConfigDict(strict=True, extra="forbid")


class GoalPreviewToolInput(StrictToolInput):
    """Planner-preview arguments with tolerant model-output normalization.

    The advertised schema remains an array of constraint strings, but Gemma
    can occasionally serialize a single or compound constraint as one JSON
    string.  Normalize that recoverable shape before StructuredTool validation
    so one malformed argument cannot terminate the interactive process.
    """

    clarified_goal: str = Field(
        description="Complete physical outcome the user wants."
    )
    constraints: list[str] = Field(
        default_factory=list,
        description="Explicit task constraints that must remain frozen.",
    )
    operator_guidance: str | None = Field(
        default=None,
        description="Optional natural-language planning guidance.",
    )

    @field_validator("constraints", mode="before")
    @classmethod
    def normalize_constraints(cls, value):
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return value


class ConfirmGoalToolInput(StrictToolInput):
    goal_id: str = Field(description="Exact staged goal identifier.")
    revision: int = Field(description="Exact staged goal revision.")
    confirmed: bool = Field(
        description="True only after explicit user confirmation."
    )


class MemoryToolInput(StrictToolInput):
    message: str = Field(
        description="A labeled RETRIEVE REQUEST or MUTATE REQUEST."
    )


class ReplanToolInput(StrictToolInput):
    operator_guidance: str | None = Field(
        default=None,
        description="Optional new observation or route guidance.",
    )


class EmptyToolInput(StrictToolInput):
    pass


class VLLMChatOpenAI(ChatOpenAI):
    def _convert_chunk_to_generation_chunk(
        self, chunk, default_chunk_class, base_generation_info
    ):
        generation = super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info
        )

        choices = (
            chunk.get("choices")
            or chunk.get("chunk", {}).get("choices")
            or []
        )
        if generation is not None and choices:
            delta = choices[0].get("delta") or {}
            reasoning = (
                delta.get("reasoning")
                or delta.get("reasoning_content")
            )
            if reasoning:
                generation.message.additional_kwargs["reasoning"] = reasoning

        request_metrics = (
            chunk.get("metrics")
            or chunk.get("chunk", {}).get("metrics")
        )
        if generation is not None and request_metrics:
            generation.message.response_metadata[
                "vllm_metrics"
            ] = request_metrics

        return generation


class HRI_Agent:
    _VALIDATION_REPORT_STATUSES = frozenset(
        {"COMPLETE", "INCOMPLETE", "NEEDS_EVIDENCE"}
    )
    _VALIDATION_STATUS_ALIASES = {
        "SUCCESS": "COMPLETE",
        "FAILURE": "INCOMPLETE",
        "PARTIAL": "INCOMPLETE",
        "UNKNOWN": "NEEDS_EVIDENCE",
    }
    _CHECKLIST_SYMBOLS = {
        "MET": "✓",
        "NOT_MET": "✗",
        "UNKNOWN": "?",
    }

    def __init__(
        self,
        model_config,
        args,
        *,
        model_name: str | None = None,
        model_base_url: str | None = None,
        model: object | None = None,
        planner_agent: object | None = None,
        memory_agent: object | None = None,
        metrics: TurnMetrics | None = None,
        model_instance_id: str = "hri",
        conversation_id: str | None = None,
    ):
        self.config = HRI_Config(
            model_config,
            model=model_name,
            model_base_url=model_base_url,
        )
        self.args = args
        if not isinstance(model_instance_id, str) or not model_instance_id.strip():
            raise ValueError("model_instance_id must be a non-empty string")
        self.model_instance_id = model_instance_id.strip()
        if conversation_id is not None and (
            not isinstance(conversation_id, str) or not conversation_id.strip()
        ):
            raise ValueError("conversation_id must be a non-empty string or None")
        self.conversation_id = (
            conversation_id.strip()
            if conversation_id is not None
            else f"hri-{uuid.uuid4().hex}"
        )

        self._hri_model = model or VLLMChatOpenAI(
            model=self.config.model,
            api_key="EMPTY",
            base_url=self.config.model_base_url,
            max_tokens=2048,  # Smaller while debugging
            temperature=0,
            streaming=True,
            stream_usage=True,
            timeout=300,
        )
        self.runtime = None
        self.call_sub_agent_tool = StructuredTool.from_function(
            func=self.call_sub_agent
        )
        self._install_tools([self.call_sub_agent_tool])
        self.hri_agent = self.build_agent()
        self.system_prompt = self.config.system_prompt
        think = getattr(self.args, "think", ())
        self.thinking_enabled = bool(
            think and (think == "all" or "HRI" in think)
        )
        self.completed_memory_mutations: list[dict] = []

        self.metrics = metrics or TurnMetrics(
            VLLMTokenCounter(
                base_url=self.config.model_base_url,
                model=self.config.model,
            )
            if model_config == "vllm"
            else None
        )

        self.planner_agent = planner_agent or Planner_Agent(
            model_config=model_config,
            args=args,
            metrics=self.metrics,
            model_name=self.config.model,
            model_base_url=self.config.model_base_url,
        )

        self.memory_agent = memory_agent or Memory_Agent(
            model_config=model_config,
            args=args,
            metrics=self.metrics,
            model_name=self.config.model,
            model_base_url=self.config.model_base_url,
            embedding_model_name=getattr(args, "embedding_model", None),
            embedding_model_base_url=getattr(
                args,
                "embedding_model_base_url",
                None,
            ),
        )

    def _install_tools(self, tools: list[StructuredTool]) -> None:
        """Install one coherent tool surface on the HRI model.

        Construction starts with the historical generic sub-agent tool so the
        HRI class remains usable on its own.  ``attach_runtime`` replaces that
        surface with the narrower production API owned by PrefMemRuntime.
        """

        self.TOOLS = tools
        self.TOOLS_BY_NAME = {tool.name: tool for tool in tools}
        self.hri_llm = self._hri_model.bind_tools(tools)

    def attach_runtime(self, runtime) -> None:
        """Attach the receding-horizon runtime and expose only safe controls.

        Planner cycles, task publication, monitoring, and terminal-history
        mutation are deliberately absent from the HRI tool schema.  They stay
        behind these four intent-level runtime calls.  The legacy
        ``call_sub_agent`` method remains available to Python callers and older
        tests, but it is no longer an LLM tool once production is attached.
        """

        required_methods = (
            "request_goal_preview",
            "confirm_goal",
            "resume_current_task",
            "request_replan",
            "context_json",
        )
        missing = [
            name for name in required_methods
            if not callable(getattr(runtime, name, None))
        ]
        if missing:
            raise TypeError(
                "runtime is missing required methods: " + ", ".join(missing)
            )
        if not hasattr(runtime, "shutdown_event"):
            raise TypeError("runtime must expose shutdown_event")

        self.runtime = runtime
        self.call_memory_agent_tool = StructuredTool.from_function(
            func=self.call_memory_agent,
            args_schema=MemoryToolInput,
        )
        self.request_goal_preview_tool = StructuredTool.from_function(
            func=self.request_goal_preview,
            args_schema=GoalPreviewToolInput,
        )
        self.confirm_goal_execution_tool = StructuredTool.from_function(
            func=self.confirm_goal_execution,
            args_schema=ConfirmGoalToolInput,
        )
        self.resume_current_task_tool = StructuredTool.from_function(
            func=self.resume_current_task,
            args_schema=EmptyToolInput,
        )
        self.request_execution_replan_tool = StructuredTool.from_function(
            func=self.request_execution_replan,
            args_schema=ReplanToolInput,
        )
        self._install_tools(
            [
                self.call_memory_agent_tool,
                self.request_goal_preview_tool,
                self.confirm_goal_execution_tool,
                self.resume_current_task_tool,
                self.request_execution_replan_tool,
            ]
        )

    def _invoke_runtime(self, method_name: str, **kwargs) -> str:
        runtime = getattr(self, "runtime", None)
        if runtime is None:
            return json.dumps(
                {
                    "status": "ERROR",
                    "error": "PrefMem runtime is not attached.",
                }
            )
        try:
            result = getattr(runtime, method_name)(**kwargs)
        except Exception as error:
            return json.dumps(
                {
                    "status": "ERROR",
                    "error": str(error),
                },
                ensure_ascii=False,
                default=str,
            )
        return json.dumps(result, ensure_ascii=False, default=str)

    def request_goal_preview(
        self,
        clarified_goal: str,
        constraints: list[str] | str | None = None,
        operator_guidance: str | None = None,
    ) -> str:
        """Preview a clarified high-level goal before asking for confirmation.

        Args:
            clarified_goal: Complete physical outcome the user wants.
            constraints: Explicit task constraints that must remain frozen.
            operator_guidance: Optional natural-language planning guidance.

        Returns:
            A JSON goal proposal, exact goal ID/revision, and confirmation flag.
        """

        normalized_constraints = (
            (constraints,)
            if isinstance(constraints, str)
            else tuple(constraints or ())
        )
        return self._invoke_runtime(
            "request_goal_preview",
            clarified_goal=clarified_goal,
            constraints=normalized_constraints,
            operator_guidance=operator_guidance,
        )

    def confirm_goal_execution(
        self,
        goal_id: str,
        revision: int,
        confirmed: bool,
    ) -> str:
        """Accept or reject the exact staged goal revision.

        A true confirmation starts a fresh Planner cycle.  The runtime then
        publishes only the first task and owns all subsequent replanning.

        Args:
            goal_id: Exact goal_id returned by request_goal_preview.
            revision: Exact revision returned by request_goal_preview.
            confirmed: True only after the user explicitly confirms this goal.

        Returns:
            JSON describing the resulting runtime state.
        """

        return self._invoke_runtime(
            "confirm_goal",
            goal_id=goal_id,
            revision=revision,
            confirmed=confirmed,
        )

    def resume_current_task(self) -> str:
        """Republish and resume the current task after recoverable attention.

        Returns:
            JSON describing the resulting runtime state.
        """

        return self._invoke_runtime("resume_current_task")

    def request_execution_replan(
        self,
        operator_guidance: str | None = None,
    ) -> str:
        """Request a fresh Planner cycle while preserving the frozen goal.

        Args:
            operator_guidance: Optional new observation or instruction to use.

        Returns:
            JSON describing the resulting runtime state.
        """

        return self._invoke_runtime(
            "request_replan",
            operator_guidance=operator_guidance,
        )

    def call_memory_agent(self, message: str) -> str:
        """Call preference memory with a labeled retrieve or mutate request.

        Args:
            message: A request beginning with RETRIEVE REQUEST: or MUTATE REQUEST:.

        Returns:
            The Memory Agent's JSON result.
        """

        return self.call_sub_agent(
            sub_agent_name="MEMORY_AGENT",
            message=message,
        )

    def _runtime_current_frame(self):
        """Return the attached runtime's image block, when one is available."""

        runtime = getattr(self, "runtime", None)
        frame_source = getattr(runtime, "frame_source", None)
        if not callable(frame_source):
            return None

        frame = frame_source()
        try:
            return frame.image_block
        except AttributeError as error:
            raise TypeError(
                "runtime.frame_source must return an object with image_block"
            ) from error

    def _record_completed_memory_mutation(
        self,
        request: str,
        output: str,
    ) -> None:
        if not request.lstrip().startswith("MUTATE REQUEST:"):
            return

        try:
            result = json.loads(output)
        except (TypeError, json.JSONDecodeError):
            return

        if not isinstance(result, dict):
            return

        status = result.get("status")
        if status not in {
            "REMEMBERED",
            "UPDATED",
            "FORGOTTEN",
            "UNCHANGED",
            "SUCCESS",
        }:
            return

        receipt = {
            "request": request,
            "status": status,
        }
        for key in (
            "id",
            "text",
            "previous_text",
            "new_text",
            "forgotten_text",
        ):
            if key in result:
                receipt[key] = result[key]

        receipts = getattr(self, "completed_memory_mutations", [])
        if receipt not in receipts:
            receipts.append(receipt)
        self.completed_memory_mutations = receipts[-20:]

    def call_sub_agent(
        self,
        sub_agent_name:  Literal["PLANNER_AGENT", "MEMORY_AGENT", "EXECUTION_AGENT", "VALIDATOR_AGENT"], 
        message: str
        ) -> str:
        """Call a sub-agent by providing the agent name and a request.

        Args:
            sub_agent_name: Name of the sub-agent to call.
            message: Natural language message to the sub-agent.

        Returns:
            The output from the sub-agent.
        """
        # Placeholder implementation; replace with actual sub-agent invocation logic.

        request_message = message
        message = [HumanMessage(content=message)]

        if sub_agent_name == "PLANNER_AGENT":
            planner_state = self.planner_agent.run(messages=message)
            planner_output = planner_state["messages"][-1].content
            if isinstance(planner_output, str):
                return planner_output
            return json.dumps(
                planner_output,
                ensure_ascii=False,
                default=str,
            )

            
        if sub_agent_name == "MEMORY_AGENT":
            current_frame = self._runtime_current_frame()
            if current_frame is None:
                memory_state = self.memory_agent.run(messages=message)
            else:
                memory_state = self.memory_agent.run(
                    messages=message,
                    current_frame=current_frame,
                )
            memory_output = memory_state["messages"][-1].content
            if not isinstance(memory_output, str):
                memory_output = json.dumps(
                    memory_output,
                    ensure_ascii=False,
                    default=str,
                )
            self._record_completed_memory_mutation(
                request_message,
                memory_output,
            )
            return memory_output


        
        if sub_agent_name == "EXECUTION_AGENT":
            return f"Agent under development. Reply the user: \"Called the EXECUTION_AGENT, agent under development.\""
        if sub_agent_name == "VALIDATOR_AGENT":
            return f"Agent under development. Reply the user: \"Called the VALIDATOR_AGENT, agent under development.\""
        else:
            raise ValueError(f"Unknown sub-agent name: {sub_agent_name}")

    def build_agent(self):
        agent_builder = StateGraph(
            MessagesState, 
            context_schema=CurrentFrameContext
        )
        agent_builder.add_node("HRI Agent", self.llm_call)
        agent_builder.add_node("Tool Node", self.tool_node)

        agent_builder.add_edge(START, "HRI Agent")
        agent_builder.add_conditional_edges(
            "HRI Agent", 
            self.should_continue, 
            ["Tool Node", END]
            )
        agent_builder.add_edge("Tool Node", "HRI Agent")

        checkpointer = InMemorySaver()
        
        agent = agent_builder.compile(checkpointer=checkpointer)

        return agent

    
    def tool_node(self, state: dict):
        """Perform model-requested tools without crashing the HRI session."""

        result = []
        for index, tool_call in enumerate(state["messages"][-1].tool_calls):
            tool_name = tool_call.get("name") or "unknown_tool"
            tool_call_id = tool_call.get("id") or f"prefmem-tool-call-{index}"
            try:
                tool = self.TOOLS_BY_NAME[tool_name]
                if self._runtime_shutdown_requested():
                    observation = json.dumps(
                        {
                            "status": "ERROR",
                            "error": (
                                "Emergency stop is latched; tool execution "
                                "is disabled."
                            ),
                        }
                    )
                else:
                    observation = tool.invoke(tool_call["args"])
            except Exception as error:
                observation = json.dumps(
                    {
                        "status": "ERROR",
                        "error": f"{type(error).__name__}: {error}",
                        "instruction": (
                            "Correct the tool name or arguments before trying "
                            "again; do not repeat the unchanged call."
                        ),
                    },
                    ensure_ascii=False,
                    default=str,
                )
            result.append(
                ToolMessage(
                    content=observation,
                    name=tool_name,
                    tool_call_id=tool_call_id,
                )
            )
        return {"messages": result}


    @staticmethod
    def _messages_for_model(messages: list[AnyMessage]) -> list[AnyMessage]:
        """Exclude completed prior-turn tool traces from the next prompt.

        Tool calls and results after the latest human message belong to the
        current graph turn and must remain available to the next HRI call.
        Earlier tool traces are implementation details; prior user messages and
        final assistant responses remain as conversational context.
        """

        latest_human_index = max(
            (
                index
                for index, message in enumerate(messages)
                if isinstance(message, HumanMessage)
            ),
            default=0,
        )

        model_messages = []
        for index, message in enumerate(messages):
            if index >= latest_human_index:
                model_messages.append(message)
                continue

            if isinstance(message, ToolMessage):
                continue
            if isinstance(message, AIMessage) and (
                message.tool_calls or message.invalid_tool_calls
            ):
                continue
            if isinstance(message, AIMessage):
                additional_kwargs = {
                    key: value
                    for key, value in message.additional_kwargs.items()
                    if key not in {"reasoning", "reasoning_content"}
                }
                message = message.model_copy(
                    update={"additional_kwargs": additional_kwargs}
                )
            model_messages.append(message)

        return model_messages


    def llm_call(self, state: dict, runtime: Runtime[CurrentFrameContext]):
        """LLM decides whether to call a tool or not"""

        model_messages = self._messages_for_model(state["messages"])
        request_options = (
            {"reasoning_effort": "high"}
            if getattr(self, "thinking_enabled", False)
            else {}
        )
        for index in range(len(model_messages) - 1, -1, -1):
            message = model_messages[index]
            if isinstance(message, HumanMessage):
                model_messages[index] = message.model_copy(
                    update={
                        "content": [
                            runtime.context["current_frame"],
                            {"type": "text", "text": message.content},
                        ]
                    }
                )
                break

        system_prompt = getattr(
            self,
            "system_prompt",
            self.config.system_prompt,
        )
        completed_memory_mutations = getattr(
            self,
            "completed_memory_mutations",
            [],
        )
        if completed_memory_mutations:
            system_prompt += (
                "\n\nCOMPLETED_MEMORY_MUTATIONS: These mutations already "
                "succeeded in this conversation. Do not repeat them unless "
                "the user explicitly requests a new memory change. A later "
                "task-plan confirmation is not a new memory request.\n"
                + json.dumps(
                    completed_memory_mutations,
                    ensure_ascii=False,
                    default=str,
                )
            )

        prefmem_runtime = getattr(self, "runtime", None)
        if prefmem_runtime is not None:
            try:
                runtime_state = prefmem_runtime.context_json()
            except Exception as error:
                runtime_state = json.dumps(
                    {
                        "state": "UNAVAILABLE",
                        "error": str(error),
                    },
                    ensure_ascii=False,
                )
            system_prompt += (
                "\n\nPREFMEM_RUNTIME_STATE: This is authoritative live "
                "execution state. Do not infer completion from conversation "
                "text or from a nominal plan.\n"
                + runtime_state
            )

        prompt_messages = [SystemMessage(content=system_prompt)] + model_messages

        started = perf_counter()
        answer = self.hri_llm.invoke(
            prompt_messages,
            **request_options,
        )
        if answer.invalid_tool_calls and not answer.tool_calls:
            content = answer.content
            has_visible_content = (
                bool(content.strip()) if isinstance(content, str) else bool(content)
            )
            if not has_visible_content:
                answer = answer.model_copy(
                    update={
                        "content": (
                            "I could not form a valid internal tool request, "
                            "so nothing was executed. Please repeat or rephrase "
                            "the request."
                        )
                    }
                )
        elapsed_seconds = perf_counter() - started

        metrics = getattr(self, "metrics", None)
        if metrics is not None:
            metrics.record(
                agent="HRI Agent",
                prompt_messages=prompt_messages,
                response=answer,
                elapsed_seconds=elapsed_seconds,
            )

        return {
            "messages": [answer],
            "llm_calls": state.get("llm_calls", 0) + 1,
        }


    def should_continue(self, state: MessagesState):
        """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

        messages = state["messages"]
        last_message = messages[-1]

        # If the LLM makes a tool call, then perform an action
        if last_message.tool_calls:
            return "Tool Node"

        # Otherwise, we stop (reply to the user)
        return END

        
    def invoke_agent(self, messages, current_frame) -> dict:
        config = {
            "configurable": {
                "thread_id": getattr(self, "conversation_id", "legacy-1")
            },
            "recursion_limit": 20,
        }

        active_section = None
        active_tool_index = None
        streamed_content = False
        streamed_reasoning = False
        streamed_tool_calls = False
        raw_response_count = 0

        def start_section(label):
            nonlocal active_section
            if active_section == label:
                return
            if active_section is not None:
                print()
            print(colored(f"{label}:", "black", "on_white"), flush=True)
            active_section = label

        for part in self.hri_agent.stream(
            input={"messages": messages},
            config=config,
            context={"current_frame": current_frame},
            stream_mode=["messages", "updates"],
            version="v2",
        ):
            if self._runtime_shutdown_requested():
                break
            if self._runtime_output_pending():
                if active_section is not None:
                    print()
                    active_section = None
                self._print_runtime_notifications()
            if part["type"] == "messages":
                chunk, metadata = part["data"]

                # Ignore Planner and tool-message chunks in the HRI renderer.
                if not isinstance(chunk, AIMessage):
                    continue

                if metadata.get("langgraph_node") != "HRI Agent":
                    continue

                reasoning = chunk.additional_kwargs.get("reasoning")
                if reasoning:
                    start_section("HRI Agent Thinking")
                    print(reasoning, end="", flush=True)
                    streamed_reasoning = True

                if isinstance(chunk.content, str) and chunk.content:
                    start_section("HRI Agent Response")
                    print(chunk.content, end="", flush=True)
                    streamed_content = True

                for call in getattr(chunk, "tool_call_chunks", []):
                    start_section("HRI Agent Tool Call")
                    call_index = call.get("index")
                    if call_index != active_tool_index:
                        if active_tool_index is not None:
                            print()
                        print(f"[{call_index}] ", end="", flush=True)
                        active_tool_index = call_index
                    if call.get("name"):
                        print(
                            f"{call['name']} arguments=",
                            end="",
                            flush=True,
                        )
                    if call.get("args"):
                        print(call["args"], end="", flush=True)
                    streamed_tool_calls = True

            elif part["type"] == "updates":
                for node_name, update in part["data"].items():
                    if not isinstance(update, dict):
                        continue

                    completed_messages = update.get("messages", [])
                    if node_name == "HRI Agent":
                        for message in completed_messages:
                            if not isinstance(message, AIMessage):
                                continue

                            reasoning = message.additional_kwargs.get("reasoning")
                            if reasoning and not streamed_reasoning:
                                start_section("HRI Agent Thinking")
                                print(reasoning, end="", flush=True)

                            if message.tool_calls and not streamed_tool_calls:
                                start_section("HRI Agent Tool Call")
                                for index, tool_call in enumerate(
                                    message.tool_calls
                                ):
                                    if index:
                                        print()
                                    print(
                                        json.dumps(
                                            tool_call,
                                            ensure_ascii=False,
                                            default=str,
                                        ),
                                        end="",
                                        flush=True,
                                    )

                            if message.content and not streamed_content:
                                start_section("HRI Agent Response")
                                print(message.content, end="", flush=True)

                            if self.args.print_raw:
                                raw_response_count += 1
                                start_section(
                                    "HRI Agent Raw Response "
                                    f"#{raw_response_count}"
                                )
                                print(
                                    json.dumps(
                                        message.model_dump(mode="json"),
                                        indent=2,
                                        ensure_ascii=False,
                                        default=str,
                                    ),
                                    flush=True,
                                )

                        active_tool_index = None
                        streamed_content = False
                        streamed_reasoning = False
                        streamed_tool_calls = False

                    elif node_name == "Tool Node":
                        for message in completed_messages:
                            if not isinstance(message, ToolMessage):
                                continue
                            start_section("HRI Agent Tool Result")
                            tool_name = message.name or message.tool_call_id
                            print(
                                f"{tool_name}: {message.content}",
                                end="",
                                flush=True,
                            )

        if active_section is not None:
            print()
        return self.hri_agent.get_state(config).values
        


    def run(self):

        if self.args.query_file:
            with open(self.args.query_file, 'r') as f:
                queries = json.load(f)
        query_index = 0
        response = None

        turn_index = 0
        while True:

            if self._runtime_shutdown_requested():
                self._print_runtime_notifications()
                print("Emergency stop is latched. Exiting PrefMem.")
                break

            print("="* 20 + f"Turn {turn_index + 1}" + "="*20)
            turn_index += 1

            self._print_runtime_notifications()
            print(colored("User:", "black", "on_white"))
            user_input = self._read_user_input_interruptibly()
            if user_input is None:
                self._print_runtime_notifications()
                print("Emergency stop is latched. Exiting PrefMem.")
                break
            if user_input == "" and self.args.query_file:
                if query_index < len(queries):
                    user_input = queries[query_index]
                    print(f"Using query idx={query_index} from query file: \n{user_input}")
                    query_index += 1
                else:
                    print("No more queries in the file. Conversation ended, Exiting PrefMem.")
                    break
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting PrefMem.")
                break
            if user_input.lower() in ["print_full"]:
                if response is None:
                    print("No model response is available yet.")
                else:
                    for m in response['messages']:
                        m.pretty_print()
                continue

            # Use the runtime-owned frame source so simulation and physical
            # camera modes share the exact same HRI/Planner observation.
            start_frame = self._runtime_current_frame()
            if start_frame is None:
                start_frame = get_live_frame()
            
            # build message
            messages = [
                HumanMessage(content=user_input),
            ]

            print(colored(f"\nHRI Agent:", "white", "on_green"))
            self.metrics.reset()
            turn_started = perf_counter()
            response = self.invoke_agent(messages, current_frame=start_frame)
            turn_seconds = perf_counter() - turn_started

            if self._runtime_shutdown_requested():
                self._print_runtime_notifications()
                print("Emergency stop is latched. Exiting PrefMem.")
                break

            if self.args.print_usage:
                print("\n")
                print(colored("Token Metrics:", "black", "on_white"))
                print(
                    json.dumps(
                        self.metrics.summary(turn_seconds=turn_seconds),
                        indent=2,
                    )
                )

    def _runtime_shutdown_requested(self) -> bool:
        runtime = getattr(self, "runtime", None)
        shutdown_event = getattr(runtime, "shutdown_event", None)
        return bool(shutdown_event is not None and shutdown_event.is_set())

    def _runtime_output_pending(self) -> bool:
        runtime = getattr(self, "runtime", None)
        for name in ("notifications", "monitor_events"):
            channel = getattr(runtime, name, None)
            if channel is None:
                continue
            try:
                if not channel.empty():
                    return True
            except Exception:
                continue
        return False

    def _print_runtime_notifications(self) -> None:
        runtime = getattr(self, "runtime", None)
        notifications = getattr(runtime, "notifications", None)
        authoritative_state = None
        context_dict = getattr(runtime, "context_dict", None)
        if callable(context_dict):
            try:
                runtime_context = context_dict()
            except Exception:
                runtime_context = None
            if isinstance(runtime_context, dict):
                authoritative_state = self._enum_text(
                    runtime_context.get("state")
                )
        if notifications is not None:
            while True:
                try:
                    notification = notifications.get_nowait()
                except queue.Empty:
                    break
                rendered = self._render_runtime_notification(
                    notification,
                    authoritative_state=authoritative_state,
                )
                print(colored(f"PrefMem: {rendered}", "white", "on_green"))
        self._print_monitor_events()

    def _print_monitor_events(self) -> None:
        runtime = getattr(self, "runtime", None)
        events = getattr(runtime, "monitor_events", None)
        if events is None:
            return
        args = getattr(self, "args", None)
        mode = str(getattr(args, "monitor_events", "off")).casefold()
        while True:
            try:
                event = events.get_nowait()
            except queue.Empty:
                return
            mapping = self._notification_mapping(event)
            if mapping is None or mode == "off":
                continue
            event_name = self._enum_text(mapping.get("event"))
            if mode == "summary" and event_name in {
                "INFERENCE_STARTED",
                "INFERENCE_COMPLETED",
            }:
                continue
            rendered = self._render_monitor_event(mapping)
            if rendered:
                print(colored(rendered, "cyan"))

    @classmethod
    def _render_monitor_event(cls, event: dict) -> str:
        event_name = cls._enum_text(event.get("event")) or "EVENT"
        publication_id = cls._brief_text(event.get("publication_id"))
        frame_sequence = event.get("frame_sequence")
        step_id = cls._brief_text(event.get("step_id"))
        prefix = f"[Monitor] {event_name.lower().replace('_', ' ')}"
        details: list[str] = []
        if frame_sequence is not None:
            details.append(f"frame={frame_sequence}")
        if step_id:
            details.append(f"step={step_id}")
        if publication_id:
            details.append(f"publication={publication_id}")
        elapsed = event.get("elapsed_seconds")
        if isinstance(elapsed, (int, float)):
            details.append(f"latency={float(elapsed):.2f}s")

        if event_name == "ASSESSMENT":
            status = cls._enum_text(event.get("task_status")) or "UNKNOWN"
            disposition = cls._enum_text(event.get("disposition")) or "UNKNOWN"
            details.extend((f"status={status}", f"disposition={disposition}"))
            executor_state = cls._enum_text(event.get("executor_state"))
            if executor_state:
                details.append(f"executor={executor_state}")
            confirmation_key = (
                "failure_confirmation" if status == "FAIL" else "success_confirmation"
            )
            confirmation = cls._notification_mapping(event.get(confirmation_key))
            if confirmation is not None:
                count = confirmation.get("count")
                required = confirmation.get("required")
                if isinstance(count, int) and isinstance(required, int) and count:
                    details.append(f"confirmation={count}/{required}")

        lines = [prefix + (" " + " ".join(details) if details else "")]
        if event_name == "TASK_PUBLISHED":
            instruction = cls._brief_text(event.get("instruction"))
            if instruction:
                lines.append(f"  instruction: {instruction}")
        if event_name == "ASSESSMENT":
            criteria = event.get("criteria") or []
            if isinstance(criteria, (list, tuple)):
                for raw in criteria:
                    criterion = cls._notification_mapping(raw)
                    if criterion is None:
                        continue
                    state = cls._enum_text(criterion.get("state")) or "UNKNOWN"
                    label = cls._brief_text(
                        criterion.get("description") or criterion.get("id")
                    )
                    if label:
                        lines.append(f"  {state}: {label}")
            observation = cls._brief_text(event.get("observation"))
            if observation:
                lines.append(f"  observation: {observation}")
            failure = cls._notification_mapping(event.get("failure"))
            if failure is not None:
                description = cls._brief_text(failure.get("description"))
                if description:
                    lines.append(f"  failure: {description}")
        message = cls._brief_text(event.get("message"))
        if message:
            lines.append(f"  message: {message}")
        return "\n".join(lines)

    @staticmethod
    def _notification_mapping(value) -> dict | None:
        """Return a mapping view without depending on Validator contracts."""

        if isinstance(value, dict):
            return value
        for method_name in ("model_dump", "to_dict"):
            method = getattr(value, method_name, None)
            if callable(method):
                try:
                    mapped = method()
                except Exception:
                    continue
                if isinstance(mapped, dict):
                    return mapped
        try:
            mapped = vars(value)
        except TypeError:
            return None
        return mapped if isinstance(mapped, dict) else None

    @staticmethod
    def _enum_text(value) -> str:
        if value is None:
            return ""
        value = getattr(value, "value", value)
        return str(value).strip().upper()

    @staticmethod
    def _brief_text(value) -> str:
        if value is None:
            return ""
        return " ".join(str(value).split())

    @classmethod
    def _validation_report_payload(cls, notification) -> dict | None:
        """Recognize a final-validation notification by shape or marker.

        Runtime may publish a contract object, a plain dictionary, or a small
        notification wrapper.  Keeping this adapter structural prevents the
        user-facing HRI layer from importing the Validator's evolving types.
        """

        outer = cls._notification_mapping(notification)
        if outer is None:
            return None

        payload = outer
        wrapped = False
        for key in ("validation_report", "final_validation_report", "report"):
            nested = cls._notification_mapping(outer.get(key))
            if nested is not None:
                payload = {**outer, **nested}
                wrapped = True
                break

        marker = cls._enum_text(
            payload.get("kind")
            or payload.get("type")
            or payload.get("notification_type")
            or ""
        )
        marked = marker in {
            "VALIDATION_REPORT",
            "FINAL_VALIDATION_REPORT",
            "FINAL_VALIDATION",
        }
        status_value = (
            payload.get("overall_status")
            or payload.get("validation_status")
            or payload.get("status")
            or payload.get("outcome")
        )
        status = cls._VALIDATION_STATUS_ALIASES.get(
            cls._enum_text(status_value),
            cls._enum_text(status_value),
        )
        checklist_present = any(
            key in payload
            for key in ("broad_checklist", "brief_checklist", "checklist")
        )
        if status not in cls._VALIDATION_REPORT_STATUSES:
            return None
        if not (wrapped or marked or checklist_present):
            return None

        return {**payload, "_render_status": status}

    @classmethod
    def _render_runtime_notification(
        cls,
        notification,
        *,
        authoritative_state: str | None = None,
    ) -> str:
        """Render final validation for a human, preserving ordinary strings.

        The detailed Validator checklist is deliberately never read here.
        Only the broad, user-facing checklist and controller-level result are
        eligible for the terminal task report.
        """

        if isinstance(notification, str):
            return notification

        report = cls._validation_report_payload(notification)
        if report is None:
            return str(notification)

        status = report["_render_status"]
        raw_checklist = (
            report.get("broad_checklist")
            or report.get("brief_checklist")
            or report.get("checklist")
            or []
        )
        if isinstance(raw_checklist, (str, bytes, dict)):
            raw_checklist = []

        checklist: list[tuple[str, str, str, str]] = []
        for raw_item in raw_checklist:
            item = cls._notification_mapping(raw_item)
            if item is None:
                continue
            item_status = cls._enum_text(
                item.get("state")
                or item.get("status")
                or item.get("result")
                or "UNKNOWN"
            )
            symbol = cls._CHECKLIST_SYMBOLS.get(item_status, "?")
            label = cls._brief_text(
                item.get("label")
                or item.get("description")
                or item.get("item")
                or item.get("criterion")
                or item.get("requirement")
                or item.get("title")
                or item.get("name")
            )
            if not label:
                continue
            evidence = cls._brief_text(
                item.get("evidence")
                or item.get("brief_evidence")
                or item.get("observation")
            )
            checklist.append((item_status, symbol, label, evidence))

        total = len(checklist)
        met = sum(item_status == "MET" for item_status, *_ in checklist)
        unknown = sum(
            item_status == "UNKNOWN" for item_status, *_ in checklist
        )
        controller_state = cls._enum_text(authoritative_state)
        completion_unconfirmed = (
            status == "COMPLETE"
            and bool(controller_state)
            and controller_state != "COMPLETE"
        )
        summary = cls._brief_text(report.get("summary"))
        if completion_unconfirmed:
            summary = (
                "Final validation met its checklist, but PrefMem has not "
                "marked the task complete "
                f"(controller state: {controller_state})."
            )
        elif not summary and status == "COMPLETE":
            summary = (
                f"Task complete — all {total} required outcomes were verified."
                if total
                else "Task complete — all required outcomes were verified."
            )
        elif not summary and status == "INCOMPLETE":
            summary = (
                f"Task incomplete — {met} of {total} required outcomes were verified."
                if total
                else "Task incomplete — one or more required outcomes were not met."
            )
        elif not summary:
            summary = (
                "Validation needs more evidence — "
                f"{unknown} of {total} required outcomes could not be verified."
                if total
                else (
                    "Validation needs more evidence — completion could not "
                    "yet be verified."
                )
            )

        lines = [summary]
        for _item_status, symbol, label, evidence in checklist:
            line = f"{symbol} {label}"
            if evidence:
                line += f" — {evidence}"
            lines.append(line)

        next_action = cls._brief_text(
            report.get("next_action") or report.get("recommended_next_action")
        )
        if next_action:
            lines.append(f"Next action: {next_action}")

        evidence_requests = (
            report.get("evidence_requests")
            or report.get("evidence_needed")
            or report.get("evidence_request")
            or []
        )
        if isinstance(evidence_requests, str):
            evidence_requests = [evidence_requests]
        elif not isinstance(evidence_requests, (list, tuple)):
            evidence_requests = []
        evidence_requests = [
            cls._brief_text(request)
            for request in evidence_requests
            if cls._brief_text(request)
        ]
        if len(evidence_requests) == 1:
            lines.append(f"Evidence needed: {evidence_requests[0]}")
        elif evidence_requests:
            lines.append("Evidence needed:")
            lines.extend(f"- {request}" for request in evidence_requests)

        return "\n".join(lines)

    def _read_user_input_interruptibly(self) -> str | None:
        """Read stdin without letting it mask a Monitor emergency stop."""

        runtime = getattr(self, "runtime", None)
        shutdown_event = getattr(runtime, "shutdown_event", None)
        if shutdown_event is None:
            return input()

        result_queue: queue.Queue[tuple[bool, object]] = queue.Queue(maxsize=1)

        def read_input() -> None:
            try:
                result_queue.put((True, input()))
            except BaseException as error:
                result_queue.put((False, error))

        threading.Thread(
            target=read_input,
            name="prefmem-console-input",
            daemon=True,
        ).start()

        while not shutdown_event.wait(0.1):
            runtime_check_timeout = getattr(runtime, "check_timeout", None)
            if callable(runtime_check_timeout):
                try:
                    runtime_check_timeout()
                except Exception:
                    # Timeout polling must not make console input unavailable.
                    pass
            self._print_runtime_notifications()
            try:
                succeeded, value = result_queue.get_nowait()
            except queue.Empty:
                continue
            if succeeded:
                return str(value)
            raise value

        return None


    def get_agent_graph(self):
        display(Image(self.hri_agent.get_graph(xray=True).draw_mermaid_png()))

        graph_path = Path("artifacts/prefmem-graph.png")
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        graph_path.write_bytes(
            self.hri_agent.get_graph(xray=True).draw_mermaid_png()
        )
        print(f"Graph saved to {graph_path}")
