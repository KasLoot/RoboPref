from langchain.tools import tool
from langchain.chat_models import init_chat_model
import os
from langchain.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
import operator
from langchain.messages import SystemMessage, HumanMessage

from langchain_openai import ChatOpenAI

from IPython.display import Image, display
from pathlib import Path


api_key = os.environ["OLLAMA_API_KEY"]

# # Ollama server model initialization
# model = init_chat_model(
#     model="gemma4:31b-cloud",
#     model_provider="ollama",
#     base_url="https://ollama.com",
#     client_kwargs={
#         "headers": {
#             "Authorization": f"Bearer {api_key}",
#         }
#     },
#     temperature=0,
# )

"""
vLLM server model initialization
First, run the following command to set up an SSH tunnel to the vLLM server:
ssh -N -L 8000:127.0.0.1:8000 \
  -p 51457 -i ~/.ssh/id_ed25519 \
  root@157.157.221.30
"""

inference_server_url = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")

llm = ChatOpenAI(
    model=os.getenv("VLLM_MODEL", "/workspace/models/gemma-4-26B-A4B-it"),
    # vLLM accepts any non-empty value when API-key authentication is disabled.
    api_key=os.getenv("VLLM_API_KEY", "EMPTY"),
    base_url=inference_server_url,
    max_tokens=16384,
    temperature=0,
)


# Define tools
@tool
def multiply(a: int, b: int) -> int:
    """Multiply `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Adds `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a + b


@tool
def divide(a: int, b: int) -> float:
    """Divide `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a / b


# Augment the LLM with tools
tools = [add, multiply, divide]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = llm.bind_tools(tools)




class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int






def llm_call(state: dict):
    """LLM decides whether to call a tool or not"""

    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
                    )
                ]
                + state["messages"]
            )
        ],
        "llm_calls": state.get('llm_calls', 0) + 1
    }


from langchain.messages import ToolMessage


def tool_node(state: dict):
    """Performs the tool call"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}



from typing import Literal
from langgraph.graph import StateGraph, START, END


def should_continue(state: MessagesState):
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        return "tool_node"

    # Otherwise, we stop (reply to the user)
    return END




# Build workflow
agent_builder = StateGraph(MessagesState)

# Add nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

# Add edges to connect nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END]
)
agent_builder.add_edge("tool_node", "llm_call")

# Compile the agent
agent = agent_builder.compile()

# Show the agent
display(Image(agent.get_graph(xray=True).draw_mermaid_png()))

graph_path = Path("artifacts/prefmem-graph.png")
graph_path.parent.mkdir(parents=True, exist_ok=True)
graph_path.write_bytes(
    agent.get_graph(xray=True).draw_mermaid_png()
)
print(f"Graph saved to {graph_path}")

# Invoke
messages = [HumanMessage(content="Calculate 2 + 3 * 4 / 2")]
messages = agent.invoke({"messages": messages})
for m in messages["messages"]:
    m.pretty_print()