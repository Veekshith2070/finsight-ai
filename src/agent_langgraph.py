import os
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

from src.config import GEMINI_API_KEY
from src.vector_store import query_vector_store
from src.tools import (
    calculate_percentage_growth,
    calculate_financial_margin,
    claculate_financial_ratio,
)
from src.logger import logger

os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

# 1. Define langChain-compatible tools
@tool
def search_report(query: str) -> str:
    """Searches he Apple 10-Q financial report for revenue, expenses, and notes."""
    logger.info(f"LangGraph Tool Executed: search_report('{query}')")
    results = query_vector_store(query, top_k=2)
    if not results:
        return "no relevant exerpts found."
    return "\n\n".join(f"[Page {r['page_number']}]\n{r['text']}" for r in results)

@tool
def calculate_margin(numerator: float, denominator: float, margin_name: str = "Margin") -> str:
    """Calculates financial margin percentage: (numerator / denominator) * 100."""
    res = calculate_financial_margin(numerator, denominator, margin_name)
    return str(res)

@tool
def calculate_growth(old_value: float, new_value: float) -> str:
    """Calculates percentage growth: ((new_value - old_value) / old_value) * 100."""
    res = calculate_percentage_growth(old_value, new_value)
    return str(res)

tools = [search_report, calculate_margin, calculate_growth]

# 2. build the LangGraph State Machine
def build_financial_agent_graph():
    logger.info("Building LangGraph State machine for FinSight AI...")

    # Initialize Model with bound tools
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        temperature=0.0,
    )
    llm_with_tools = llm.bind_tools(tools)

    # Node 1: The Agent Node (calls LLM)
    def call_model(state: MessagesState):
        messages = state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}
    
    # Node 2: The Tool Node (executes tools)
    tool_node = ToolNode(tools)

    # 3. Construct the Graph
    workflow = StateGraph(MessagesState)

    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)

    workflow.add_edge(START, "agent")

    # Conditional Edge: Check if model called a tool or finished
    workflow.add_conditional_edges(
        "agent",
        tools_condition, # Built-in router: routes to 'tool' or END
    )

    # Cyclic Edge: After tools run, loop BACH to the agent
    workflow.add_edge("tools", "agent")

    # Compile the graph into an executable runnable
    app = workflow.compile()
    return app

if __name__ == "__main__":
    app = build_financial_agent_graph()

    test_query = (
        "For the three months ended June 27, 2026, find the Apple's Products revenue "
        "and Services revenue in the report, and claculate what percentage Services is of Products."
    )

    print("\n=======================================================")
    print("      FinSight AI - LangGraph State Machine Demo       ")
    print("=======================================================\n")
    print(f"Query: {test_query}\n")

    # Stream the graph execution to watch node transitions live!
    inputs = {"messages": [("user", test_query)]}

    for event in app.stream(inputs, stream_mode="values"):
        latest_message = event["messages"][-1]
        sender = latest_message.type
        print(f"[{sender.upper()}]: {latest_message.content[:200] if latest_message.content else 'Called a tool...'}")