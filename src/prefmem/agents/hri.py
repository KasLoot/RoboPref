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
from prefmem.agents.vision import image_data_url, get_start_end_frames
from prefmem.agents.planner import Planner_Agent
import json


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

        self.planner_agent = Planner_Agent(model_config=model_config, args=args)

    def call_sub_agent(
        self,
        sub_agent_name:  Literal["PLANNER_AGENT"], 
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

        message = [HumanMessage(content=message)]

        if sub_agent_name == "PLANNER_AGENT":
            return self.planner_agent.run(messages=message)

            
        if sub_agent_name == "MEMORY_AGENT":
            return f"Finished: The robot has seen a red block, a blue block, and a green block."
        if sub_agent_name == "EXECUTION_AGENT":
            return f"Finished: The robot can pick up blocks and stack them."
        if sub_agent_name == "VALIDATOR_AGENT":
            return f"Finished: The blocks are in the correct order."
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

        return {
            "messages": [
                self.hri_llm.invoke(
                    [
                        SystemMessage(
                            content=getattr(
                                self,
                                "system_prompt",
                                self.config.system_prompt,
                            )
                        )
                    ]
                    + model_messages,
                    **request_options
                )
            ],
            "llm_calls": state.get('llm_calls', 0) + 1
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
        
        while True:

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

            start_frame, last_frame = get_start_end_frames(self.args)
            
            # build message
            messages = [
                HumanMessage(content=user_input),
            ]

            print(colored(f"\nHRI Agent:", "white", "on_green"))
            response = self.invoke_agent(messages, current_frame=start_frame)


    def get_agent_graph(self):
        display(Image(self.hri_agent.get_graph(xray=True).draw_mermaid_png()))

        graph_path = Path("artifacts/prefmem-graph.png")
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        graph_path.write_bytes(
            self.hri_agent.get_graph(xray=True).draw_mermaid_png()
        )
        print(f"Graph saved to {graph_path}")
