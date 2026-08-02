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
from pydantic import BaseModel, Field
from langgraph.checkpoint.memory import InMemorySaver  
from langgraph.runtime import Runtime
from prefmem.agents.vision import image_data_url, get_start_end_frames, get_live_frame
from prefmem.agents.planner import Planner_Agent
from prefmem.agents.memory import Memory_Agent
import json

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
    def __init__(self, model_config, args):
        self.config = HRI_Config(model_config)
        self.args = args
        
        self.hri_llm = VLLMChatOpenAI(
            model=self.config.model,
            api_key="EMPTY",
            base_url=self.config.model_base_url,
            max_tokens=2048,  # Smaller while debugging
            temperature=0,
            streaming=True,
            stream_usage=True,
            timeout=300,
        )
        self.call_sub_agent_tool = StructuredTool.from_function(
            func=self.call_sub_agent
        )
        self.TOOLS = [self.call_sub_agent_tool]
        self.hri_llm = self.hri_llm.bind_tools(self.TOOLS)
        self.TOOLS_BY_NAME = {tool.name: tool for tool in self.TOOLS}
        self.hri_agent = self.build_agent()
        self.system_prompt = self.config.system_prompt
        self.thinking_enabled = bool(self.args.think and (self.args.think == "all" or "HRI" in self.args.think))
        self.completed_memory_mutations: list[dict] = []

        self.metrics = TurnMetrics(
            VLLMTokenCounter(
                base_url=self.config.model_base_url,
                model=self.config.model,
            )
            if model_config == "vllm"
            else None
        )

        self.planner_agent = Planner_Agent(
            model_config=model_config,
            args=args,
            metrics=self.metrics,
        )

        self.memory_agent = Memory_Agent(
            model_config=model_config,
            args=args,
            metrics=self.metrics,
        )

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
            memory_state = self.memory_agent.run(messages=message)
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
        """Performs the tool call"""

        result = []
        for tool_call in state["messages"][-1].tool_calls:
            tool = self.TOOLS_BY_NAME[tool_call["name"]]
            observation = tool.invoke(tool_call["args"])
            result.append(
                ToolMessage(
                    content=observation,
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
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

        prompt_messages = [SystemMessage(content=system_prompt)] + model_messages

        started = perf_counter()
        answer = self.hri_llm.invoke(
            prompt_messages,
            **request_options,
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
            "configurable": {"thread_id": "1"},
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

            print("="* 20 + f"Turn {turn_index + 1}" + "="*20)
            turn_index += 1

            print(colored("User:", "black", "on_white"))
            user_input = input()
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

            # start_frame, last_frame = get_start_end_frames(self.args)
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

            if self.args.print_usage:
                print("\n")
                print(colored("Token Metrics:", "black", "on_white"))
                print(
                    json.dumps(
                        self.metrics.summary(turn_seconds=turn_seconds),
                        indent=2,
                    )
                )


    def get_agent_graph(self):
        display(Image(self.hri_agent.get_graph(xray=True).draw_mermaid_png()))

        graph_path = Path("artifacts/prefmem-graph.png")
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        graph_path.write_bytes(
            self.hri_agent.get_graph(xray=True).draw_mermaid_png()
        )
        print(f"Graph saved to {graph_path}")
