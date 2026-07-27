from langchain.tools import tool
from langchain.chat_models import init_chat_model
import os
from langchain.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
from prefmem.agents.config import HRI_Config
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
from prefmem.agents.vision import image_data_url



from colorama import init
from termcolor import colored
init()

@tool
def call_sub_agent(sub_agent_name: str, sub_agent_input: str) -> str:
    """Call a sub-agent with the given name and input.

    Args:
        sub_agent_name: Name of the sub-agent to call.
        sub_agent_input: Input to provide to the sub-agent.

    Returns:
        The output from the sub-agent.
    """
    # Placeholder implementation; replace with actual sub-agent invocation logic.
    print(f"Calling sub-agent '{sub_agent_name}' with input: {sub_agent_input}")

    if sub_agent_name == "Planner":
        return f"Finished: Stack the blocks in the order: red, blue, green."
    if sub_agent_name == "Memory":
        return f"Finished: The robot has seen a red block, a blue block, and a green block."
    if sub_agent_name == "VLA":
        return f"Finished: The robot can pick up blocks and stack them."
    if sub_agent_name == "Validator":
        return f"Finished: The blocks are in the correct order."
    else:
        raise ValueError(f"Unknown sub-agent name: {sub_agent_name}")



class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int


class HRIContext(TypedDict):
    current_frame: dict



class HRI_Agent:
    def __init__(self, model_config):
        self.config = HRI_Config(model_config)
        self.llm = ChatOpenAI(
                    model=self.config.model,
                    api_key="EMPTY",
                    base_url=self.config.model_base_url,
                    max_tokens=16384,
                    temperature=0,
        )
        self.TOOLS = [call_sub_agent]
        self.TOOLS_BY_NAME = {tool.name: tool for tool in self.TOOLS}
        self.llm = self.llm.bind_tools(self.TOOLS)
        self.agent = self.build_agent()

    def build_agent(self):
        agent_builder = StateGraph(MessagesState, context_schema=HRIContext)
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
            result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
        return {"messages": result}


    def llm_call(self, state: dict, runtime: Runtime[HRIContext]):
        """LLM decides whether to call a tool or not"""

        model_messages = list(state["messages"])
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
                self.llm.invoke(
                    [
                        SystemMessage(
                            content=self.config.system_prompt
                        )
                    ]
                    + model_messages
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

    def get_start_end_frames(self, args):
        if args.dataset:
            dataset_path = Path(args.dataset)
            if not dataset_path.exists():
                raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")

            # sort the files in the dataset directory to ensure consistent ordering
            image_files = sorted(dataset_path.glob("*.png"))
            if not image_files:
                raise FileNotFoundError(f"No PNG files found in dataset path {dataset_path}. Convert to PNG or provide a valid dataset.")
            
            start_frame = {
                "type": "image_url",
                "image_url": {"url": image_data_url(image_files[0].read_bytes())},
            }
            last_frame = {
                "type": "image_url",
                "image_url": {"url": image_data_url(image_files[-1].read_bytes())},
            }

            return start_frame, last_frame

        
    def invoke_agent(self, messages, current_frame) -> dict:

        response = self.agent.invoke(
            input = {"messages": messages},
            config = {"configurable": {"thread_id": "1"}},
            context = {"current_frame": current_frame},
            )
        return response
        


    def run(self, args):
        
        while True:
            user_input = input("User: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting PrefMem.")
                break
            if user_input.lower() in ["print_full"]:
                for m in response['messages']:
                    m.pretty_print()
                continue

            start_frame, last_frame = self.get_start_end_frames(args)
            
            # build message
            messages = [
                HumanMessage(content=user_input),
            ]
            response = self.invoke_agent(messages, current_frame=start_frame)
            print(colored(f"\nHRI Agent:", "white", "on_green"))
            print(colored(f"{response['messages'][-1].content}\n", "green"))
            

    def get_agent_graph(self):
        display(Image(self.agent.get_graph(xray=True).draw_mermaid_png()))

        graph_path = Path("artifacts/prefmem-graph.png")
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        graph_path.write_bytes(
            self.agent.get_graph(xray=True).draw_mermaid_png()
        )
        print(f"Graph saved to {graph_path}")
