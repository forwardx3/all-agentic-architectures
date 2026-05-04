import os
import re
from typing import List, Annotated, TypedDict, Optional
from dotenv import load_dotenv

# LangChain components
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, ToolMessage
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langchain_tavily import TavilySearch

# LangGraph components
from langgraph.graph import StateGraph, END
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# For pretty printing
from rich.console import Console
from rich.markdown import Markdown

# --- API Key and Tracing Setup ---
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Agentic Architecture - Planning (Nebius)"

# Check that the keys are set
for key in ["DEEPSEEK_API_KEY", "LANGCHAIN_API_KEY", "TAVILY_API_KEY"]:
    if not os.environ.get(key):
        print(f"{key} not found. Please create a .env file and set it.")

print("Environment variables loaded and tracing is set up.")

console = Console()

# Define the state for our graphs
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# 1. Define the base tool from the tavily package
tavily_search_tool = TavilySearch(max_results=2)

# 2. THE FIX: Simplify the custom tool. 
#    The .invoke() method already returns a clean string, so we just pass it through.
@tool
def web_search(query: str) -> str:
    """Performs a web search using Tavily and returns the results as a string."""
    console.print(f"--- TOOL: Searching for '{query}'...")
    results = tavily_search_tool.invoke(query)
    return results

llm = ChatOpenAI(
    model="deepseek-v4-flash",
    temperature=0.2,
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base="https://api.deepseek.com/v1",
    model_kwargs={
        "extra_body": {
            "thinking": {"type": "disabled"}  # 禁用思考模式
        }
    }
)

llm_with_tools = llm.bind_tools(tools=[web_search])

def react_agent_node(state: AgentState):
    console.print("--- REACTIVE AGENT: Thinking... ---")
    messages_with_system_prompt = [
        SystemMessage(content="You are a helpful research assistant. You must call one and only one tool at a time. Do not call multiple tools in a single turn. After receiving the result from a tool, you will decide on the next step.")
    ] + state["messages"]

    response = llm_with_tools.invoke(messages_with_system_prompt)

    return {"messages": [response]}

tool_node = ToolNode([web_search])

react_graph_builder = StateGraph(AgentState)
react_graph_builder.add_node("agent", react_agent_node)
react_graph_builder.add_node("tools", tool_node)
react_graph_builder.add_conditional_edges("agent", tools_condition)
react_graph_builder.add_edge("tools", "agent")

react_graph_builder.set_entry_point("agent")
react_agent_app = react_graph_builder.compile()

print("Reactive (ReAct) agent compiled successfully.")

plan_centric_query = """
Find the population of the capital cities of France, Germany, and Italy. 
Then calculate their combined total. 
Finally, compare that combined total to the population of the United States, and say which is larger.
"""

console.print(f"[bold yellow]Testing REACTIVE agent on a plan-centric query:[/bold yellow] '{plan_centric_query}'\n")

final_react_output = None
for chunk in react_agent_app.stream({"messages": [("user", plan_centric_query)]}, stream_mode="values"):
    final_react_output = chunk
    console.print(f"--- [bold purple]Current State Update[/bold purple] ---")
    chunk["messages"][-1].pretty_print()
    console.print("\n")

console.print("\n--- [bold red]Final Output from Reactive Agent[/bold red] ---")
console.print(Markdown(final_react_output['messages'][-1].content))

class Plan(BaseModel):
    steps: List[str] = Field(description="A list of tool calls that, when executed, will answer the query.")

class PlanningState(TypedDict):
    user_request: str
    plan: Optional[List[str]]
    intermediate_steps: List[ToolMessage]
    final_answer: Optional[str]

def planner_node(state: PlanningState):
    console.print("--- PLANNER: Decomposing task... ---")
    planner_llm = llm.with_structured_output(Plan, method='json_mode')

    # THE FIX: A much more explicit prompt with a clear example (few-shot prompting)
    prompt = f"""You are an expert planner. Your job is to create a step-by-step plan to answer the user's request.
        Each step in the plan must be a single call to the `web_search` tool.

        **Instructions:**
        1. Analyze the user's request.
        2. Break it down into a sequence of simple, logical search queries.
        3. Format the output as a list of strings, where each string is a single valid tool call.

        **Example:**
        Request: "What is the capital of France and what is its population?"
        Correct Plan Output:
        [
            "web_search('capital of France')",
            "web_search('population of Paris')"
        ]

        **User's Request:**
        {state['user_request']}

        Your JSON response MUST include these exact keys:
        - "steps": list of str
    """

    plan_result = planner_llm.invoke(prompt)

    console.print(f"--- PLANNER: Generated Plan: {plan_result.steps} ---")
    return {"plan": plan_result.steps}

def executor_node(state: PlanningState):
    console.print("--- EXECUTOR: Running next step... ---")
    plan = state["plan"]
    next_step = plan[0]

    # Robust regex to handle both single and double quotes
    match = re.search(r"(\w+)\((?:\"|\')(.*?)(?:\"|\')\)", next_step)
    if not match:
        tool_name = "web_search"
        query = next_step
    else:
        tool_name, query = match.groups()[0], match.groups()[1]

    console.print(f"--- EXECUTOR: Calling tool '{tool_name}' with query '{query}' ---")
    
    result = tavily_search_tool.invoke(query)

    tool_message = ToolMessage(
    content=str(result),
    name=tool_name,
    tool_call_id=f"manual-{hash(query)}"
    )
    
    return {
        "plan": plan[1:], # Pop the executed step from the plan
        "intermediate_steps": state["intermediate_steps"] + [tool_message]
    }

def synthesizer_node(state: PlanningState):
    console.print("--- SYNTHESIZER: Generating final answer... ---")
    
    context = "\n".join([f"Tool {msg.name} returned: {msg.content}" for msg in state["intermediate_steps"]])

    prompt = f"""You are an expert synthesizer. Based on the user's request and the collected data, provide a comprehensive final answer.
    
    Request: {state['user_request']}
    Collected Data:
    {context}

    Your JSON response MUST include these exact keys:
    - "user_request": string
    - "plan": list of string
    - "intermediate_steps": list of tool message
    - "final_answer": string
    """
    llm_result = llm.invoke(prompt)
    final_answer = llm_result.content
    return {"final_answer": final_answer}

print("Planner, Executor, and Synthesizer nodes defined.")

def planning_router(state: PlanningState):
    if not state["plan"]:
        console.print("--- ROUTER: Plan complete. Moving to synthesizer. ---")
        return "synthesize"
    else:
        console.print("--- ROUTER: Plan has more steps. Continuing execution. ---")
        return "execute"

planning_graph_builder = StateGraph(PlanningState)
planning_graph_builder.add_node("plan", planner_node)
planning_graph_builder.add_node("execute", executor_node)
planning_graph_builder.add_node("synthesize", synthesizer_node)

planning_graph_builder.set_entry_point("plan")
planning_graph_builder.add_conditional_edges("plan", planning_router, {"execute": "execute", "synthesize": "synthesize"}) # Route after planning
planning_graph_builder.add_conditional_edges("execute", planning_router, {"execute": "execute", "synthesize": "synthesize"})
planning_graph_builder.add_edge("synthesize", END)

planning_agent_app = planning_graph_builder.compile()
print("Planning agent compiled successfully.")

console.print(f"[bold green]Testing PLANNING agent on the same plan-centric query:[/bold green] '{plan_centric_query}'\n")

# Remember to initialize the state correctly, especially the list for intermediate steps
initial_planning_input = {"user_request": plan_centric_query, "intermediate_steps": []}

final_planning_output = planning_agent_app.invoke(initial_planning_input)

console.print("\n--- [bold green]Final Output from Planning Agent[/bold green] ---")
console.print(Markdown(final_planning_output['final_answer']))

class ProcessEvaluation(BaseModel):
    """Schema for evaluating an agent's problem-solving process."""
    task_completion_score: int = Field(description="Score 1-10 on whether the agent successfully completed the task.")
    process_efficiency_score: int = Field(description="Score 1-10 on the efficiency and directness of the agent's process. A higher score means a more logical and less roundabout path.")
    justification: str = Field(description="A brief justification for the scores.")

judge_llm = llm.with_structured_output(ProcessEvaluation, method='json_mode')

def evaluate_agent_process(query: str, final_state: dict):
    # For the ReAct agent, the trace is in 'messages'. For Planning, it's in 'intermediate_steps'.
    if 'messages' in final_state:
        trace = "\n".join([f"{m.type}: {str(m.content)}" for m in final_state['messages']])
    else:
        trace = f"Plan: {final_state.get('plan', [])}\nSteps: {final_state.get('intermediate_steps', [])}"
        
    prompt = f"""You are an expert judge of AI agents. Evaluate the agent's process for solving the task on a scale of 1-10.
    Focus on whether the process was logical and efficient.
    
    **User's Task:** {query}
    **Full Agent Trace:**\n```\n{trace}\n```

    Your JSON response MUST include these exact keys:
    - "task_completion_score": int
    - "process_efficiency_score": int
    - "justification": string
    """
    return judge_llm.invoke(prompt)

console.print("--- Evaluating Reactive Agent's Process ---")
react_agent_evaluation = evaluate_agent_process(plan_centric_query, final_react_output)
console.print(react_agent_evaluation.model_dump())

console.print("\n--- Evaluating Planning Agent's Process ---")
planning_agent_evaluation = evaluate_agent_process(plan_centric_query, final_planning_output)
console.print(planning_agent_evaluation.model_dump())