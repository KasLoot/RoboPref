from langchain.tools import tool
from langchain.chat_models import init_chat_model
import os
from langchain.messages import AIMessage, AnyMessage
from typing_extensions import TypedDict, Annotated, Literal
from prefmem.agents.config import Planner_Config
from langchain.messages import ToolMessage

from langgraph.graph import StateGraph, START, END
import operator
from langchain.messages import SystemMessage, HumanMessage

from langchain_openai import ChatOpenAI

from IPython.display import Image, display
from pathlib import Path
from pydantic import BaseModel, Field
from langgraph.checkpoint.memory import InMemorySaver  
from langgraph.runtime import Runtime
from prefmem.agents.vision import image_data_url, get_start_end_frames
import json
from time import perf_counter

from prefmem.agents.metrics import TurnMetrics


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





class Planner_Agent:
    def __init__(
        self,
        model_config: str = "vllm",
        args=None,
        metrics: TurnMetrics | None = None,
    ):
        self.config = Planner_Config(model_config)
        self.args = args
        self.metrics = metrics
        self.llm = VLLMChatOpenAI(
            model=self.config.model,
            api_key="EMPTY",
            base_url=self.config.model_base_url,
            max_tokens=2048,  # Smaller while debugging
            temperature=0,
            streaming=True,
            stream_usage=True,
            timeout=300,
        )
        # self.TOOLS = []
        # self.TOOLS_BY_NAME = {tool.name: tool for tool in self.TOOLS}
        # self.llm = self.llm.bind_tools(self.TOOLS)
        self.agent = self.build_agent()
        self.system_prompt = self.config.system_prompt
        self.thinking_enabled = bool(args.think and (args.think == "all" or "Planner" in args.think))
        self.print_raw = bool(getattr(args, "print_raw", False))

    def build_agent(self):
        agent_builder = StateGraph(
            MessagesState, 
            context_schema=CurrentFrameContext
        )
        agent_builder.add_node("Planner Agent", self.llm_call)
        agent_builder.add_node("Tool Node", self.tool_node)

        agent_builder.add_edge(START, "Planner Agent")
        agent_builder.add_conditional_edges(
            "Planner Agent", 
            self.should_continue, 
            ["Tool Node", END]
        )
        agent_builder.add_edge("Tool Node", "Planner Agent")

        agent = agent_builder.compile()

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
    
    
    def llm_call(self, state: dict, runtime: Runtime[CurrentFrameContext]):
        """LLM decides whether to call a tool or not"""

        model_messages = list(state["messages"])
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

        prompt_messages = [
            SystemMessage(
                content=getattr(
                    self,
                    "system_prompt",
                    self.config.system_prompt,
                )
            )
        ] + model_messages

        started = perf_counter()
        answer = self.llm.invoke(
            prompt_messages,
            **request_options,
        )
        elapsed_seconds = perf_counter() - started

        if self.metrics is not None:
            self.metrics.record(
                agent="Planner Agent",
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
    
            final_state = None
            for part in self.agent.stream(
                input={"messages": messages},
                config=config,
                context={"current_frame": current_frame},
                stream_mode=["messages", "updates", "values"],
                version="v2",
            ):
                if part["type"] == "values":
                    final_state = part["data"]
                    continue

                if part["type"] == "messages":
                    chunk, metadata = part["data"]

                    # Ignore messages belonging to any nested agent or tool.
                    if not isinstance(chunk, AIMessage):
                        continue

                    if metadata.get("langgraph_node") != "Planner Agent":
                        continue
    
                    reasoning = chunk.additional_kwargs.get("reasoning")
                    if reasoning:
                        start_section("Planner Agent Thinking")
                        print(reasoning, end="", flush=True)
                        streamed_reasoning = True
    
                    if isinstance(chunk.content, str) and chunk.content:
                        start_section("Planner Agent Response")
                        print(chunk.content, end="", flush=True)
                        streamed_content = True
    
                    for call in getattr(chunk, "tool_call_chunks", []):
                        start_section("Planner Agent Tool Call")
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
                        if node_name == "Planner Agent":
                            for message in completed_messages:
                                if not isinstance(message, AIMessage):
                                    continue
    
                                reasoning = message.additional_kwargs.get("reasoning")
                                if reasoning and not streamed_reasoning:
                                    start_section("Planner Agent Thinking")
                                    print(reasoning, end="", flush=True)
    
                                if message.tool_calls and not streamed_tool_calls:
                                    start_section("Planner Agent Tool Call")
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
                                    start_section("Planner Agent Response")
                                    print(message.content, end="", flush=True)
    
                                if getattr(self, "print_raw", False):
                                    raw_response_count += 1
                                    start_section(
                                        "Planner Agent Raw Response "
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
                                start_section("Planner Agent Tool Result")
                                tool_name = message.name or message.tool_call_id
                                print(
                                    f"{tool_name}: {message.content}",
                                    end="",
                                    flush=True,
                                )
    
            if active_section is not None:
                print()

            if final_state is None:
                raise RuntimeError("Planner completed without producing final state.")

            return final_state
            
    
    
    def run(self, messages):

        start_frame, last_frame = get_start_end_frames(self.args)
        print(colored(f"\nPlanner Agent:", "white", "on_blue"))
        response = self.invoke_agent(messages, current_frame=start_frame)

        return response
