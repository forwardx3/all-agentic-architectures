import os
import json
from typing import List, Annotated, TypedDict, Optional
from dotenv import load_dotenv

# LangChain components
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

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

def flaky_web_search(query: str) -> str:
    console.print(f"--- TOOL: Searching for '{query}'... ---")
    if "employee count" in query.lower():
        console.print("--- TOOL: [bold red]Simulating API failure![/bold red] ---")
        return "Error: Could not retrieve data. The API endpoint is currently unavailable."
    else:
        result = TavilySearch(max_results=2).invoke(query)
        if isinstance(result, (dict, list)):
            return json.dumps(result, indent=2)
        return str(result)
    
class BasicPEState(TypedDict):
    user_request: str
    plan: Optional[List[str]]
    intermediate_steps: List[str]
    final_answer: Optional[str]

class Plan(BaseModel):
    steps: List[str] = Field(description="A list of tool calls to execute.")

def basic_plan_node(state: BasicPEState):
    console.print("--- (Basic) PLANNER: Creating plan... ---")
    planner_llm = llm.with_structured_output(Plan, method='json_mode')

    prompt = f"""
    You are a planning agent. 
    Your job is to decompose the user's request into a list of clear tool queries.

    - Only return JSON that matches this schema: {{ "steps": [ "query1", "query2", ... ] }}
    - Do NOT return any prose or explanation.
    - Always use the 'flaky_web_search' tool for queries.

    Your JSON response MUST include these exact keys:
    - "steps": list of str

    User's request: "{state['user_request']}"
    """
    plan = planner_llm.invoke(prompt)
    return {"plan": plan.steps}

def basic_executor_node(state: BasicPEState):
    console.print("--- (Basic) EXECUTOR: Running next step... ---")
    next_step = state["plan"][0]

    result = flaky_web_search(next_step)
    return {"plan": state["plan"][1:], "intermediate_steps": state["intermediate_steps"] + [result]}

def basic_synthesizer_node(state: BasicPEState):
    console.print("--- (Basic) SYNTHESIZER: Generating final answer... ---")
    context = "\n".join(state["intermediate_steps"])

    prompt = f"Synthesize an answer for '{state['user_request']}' using this data:\n{context}"
    answer = llm.invoke(prompt).content
    return {"final_answer": answer}

pe_graph_builder = StateGraph(BasicPEState)
pe_graph_builder.add_node("plan", basic_plan_node)
pe_graph_builder.add_node("executor", basic_executor_node)
pe_graph_builder.add_node("synthesize", basic_synthesizer_node)

pe_graph_builder.set_entry_point("plan")
pe_graph_builder.add_conditional_edges("plan", lambda s: "executor" if s["plan"] else "synthesize")
pe_graph_builder.add_conditional_edges("executor", lambda s: "executor" if s["plan"] else "synthesize")
pe_graph_builder.add_edge("synthesize", END)

basic_pe_app = pe_graph_builder.compile()

print("Basic Planner-Executor agent compiled successfully.")

flaky_query = "What was Apple's R&D spend in their last fiscal year, and what was their total employee count? Calculate the R&D spend per employee."

console.print(f"[bold yellow]Testing BASIC P-E agent on a flaky query:[/bold yellow]\n'{flaky_query}'\n")

initial_pe_input = {"user_request": flaky_query, "intermediate_steps": []}
final_pe_output = basic_pe_app.invoke(initial_pe_input)

console.print("\n--- [bold red]Final Output from Basic P-E Agent[/bold red] ---")
console.print(Markdown(final_pe_output['final_answer']))

class VerificationResult(BaseModel):
    is_successful: bool = Field(description="True if the tool execution was successful and the data is valid.")
    reasoning: str = Field(description="Reasoning for the verification decision.")

class PEVState(TypedDict):
    user_request: str
    plan: Optional[List[str]]
    last_tool_result: Optional[str]
    intermediate_steps: List[str]
    final_answer: Optional[str]
    retries: int  # count how many times we’ve replanned

from langchain_core.exceptions import OutputParserException

class Plan(BaseModel):
    steps: List[str] = Field(description="List of queries (max 5).", max_length=5)

def pev_planner_node(state: PEVState):
    retries = state.get("retries", 0)
    if retries > 10:
        console.print("--- (PEV) PLANNER: Retry limit reached. Stopping. ---")
        return {
            "plan": [],
            "final_answer": "Error: Unable to complete task after multiple retries."
        }
    
    console.print(f"--- (PEV) PLANNER: Creating/revising plan (retry {retries})... ---")

    planner_llm = llm.with_structured_output(Plan, method='json_mode')
    past_context = "\n".join(state["intermediate_steps"])
    base_prompt = f"""
    You are a planning agent. 
    Create a plan to answer: '{state['user_request']}'. 
    Use the 'flaky_web_search' tool.

    Rules:
    - Return ONLY valid JSON in this exact format: {{ "steps": ["query1", "query2"] }}
    - Maximum 5 steps.
    - Do NOT repeat failed queries or endless variations.
    - Do NOT output explanations, only JSON.
    
    Your JSON response MUST include these exact keys:
    - "steps": list of str

    Previous attempts and results:
    {past_context}
    """

    for attempt in range(2):
        try:
            plan = planner_llm.invoke(base_prompt)
            return {"plan": plan.steps, "retries": retries + 1}
        except OutputParserException as e:
            console.print(f"[red]Planner parsing failed (attempt {attempt+1}): {e}[/red]")
            base_prompt = f"Return ONLY valid JSON with {{'steps': ['...']}}. {base_prompt}"

    return {"plan": ["Apple R&D spend last fiscal year"], "retries": retries + 1}

def pev_executor_node(state: PEVState):
    if not state.get("plan"):  # ✅ guard against empty plan
        console.print("--- (PEV) EXECUTOR: No steps left, skipping execution. ---")
        return {}
    
    console.print("--- (PEV) EXECUTOR: Running next step... ---")
    next_step = state["plan"][0]
    result = flaky_web_search(next_step)
    return {"plan": state["plan"][1:], "last_tool_result": result}

def verifier_node(state: PEVState):
    console.print("--- VERIFIER: Checking last tool result... ---")
    verifier_llm = llm.with_structured_output(VerificationResult, method='json_mode')
    prompt = f"""Verify if the following tool output is a successful result or an error message. 
        The task was '{state['user_request']}'.
        Tool Output: '{state['last_tool_result']}'

        Your JSON response MUST include these exact keys:
        - "is_successful": bool
        - "reasoning": str
    """
    verification = verifier_llm.invoke(prompt)
    console.print(f"--- VERIFIER: Judgment is '{'Success' if verification.is_successful else 'Failure'}' ---")

    if verification.is_successful:
        return {"intermediate_steps": state["intermediate_steps"] + [state["last_tool_result"]]}
    else:
        return {"plan": [], "intermediate_steps": state["intermediate_steps"] + [f"Verification Failed: {state['last_tool_result']}"]}
    
pev_synthesizer_node = basic_synthesizer_node # We can reuse the same synthesizer

def pev_router(state: PEVState):
    # ✅ If we already have a final answer (e.g. retry limit reached), stop
    if state.get("final_answer"):
        console.print("--- ROUTER: Final answer available. Moving to synthesizer. ---")
        return "synthesize"

    if not state["plan"]:
        # Check if plan is empty because of verification failure
        if state["intermediate_steps"] and "Verification Failed" in state["intermediate_steps"][-1]:
            console.print("--- ROUTER: Verification failed. Re-planning... ---")
            return "plan"
        else:
            console.print("--- ROUTER: Plan complete. Moving to synthesizer. ---")
            return "synthesize"
    else:
        console.print("--- ROUTER: Plan has more steps. Continuing execution. ---")
        return "execute"
    
pev_graph_builder = StateGraph(PEVState)
pev_graph_builder.add_node("plan", pev_planner_node)
pev_graph_builder.add_node("execute", pev_executor_node)
pev_graph_builder.add_node("verifier", verifier_node)
pev_graph_builder.add_node("synthesize", pev_synthesizer_node)

pev_graph_builder.set_entry_point("plan")
pev_graph_builder.add_edge("plan", "execute")
pev_graph_builder.add_edge("execute", "verifier")
pev_graph_builder.add_conditional_edges("verifier", pev_router)
pev_graph_builder.add_edge("synthesize", END)

pev_agent_app = pev_graph_builder.compile()
print("Planner-Executor-Verifier (PEV) agent compiled successfully.")

console.print(f"[bold green]Testing PEV agent on the same flaky query:[/bold green]\n'{flaky_query}'\n")

initial_pev_input = {"user_request": flaky_query, "intermediate_steps": [], "retries": 0}

final_pev_output = pev_agent_app.invoke(initial_pev_input)

console.print("\n--- [bold green]Final Output from PEV Agent[/bold green] ---")
console.print(Markdown(final_pev_output['final_answer']))