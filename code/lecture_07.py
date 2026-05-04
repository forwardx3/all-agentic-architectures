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

search_tool = TavilySearch(max_results=2)

class SequentialState(TypedDict):
    user_request: str
    news_report: Optional[str]
    technical_report: Optional[str]
    financial_report: Optional[str]
    final_report: Optional[str]

def news_report_seq(state: SequentialState):
    console.print("--- (Sequential) CALLING NEWS ANALYST ---")
    prompt = f"Your task is to act as an expert News Analyst. Find the latest major news about the topic in the user's request and provide a concise summary.\n\nUser Request: {state['user_request']}"
    agent = llm.bind_tools([search_tool])
    result = agent.invoke(prompt)
    return {"news_report": result.content}

def technical_report_seq(state: SequentialState):
    console.print("--- (Sequential) CALLING TECHNICAL ANALYST ---")
    prompt = f"Your task is to act as an expert Technical Analyst. Based on the following news report, conduct a technical analysis of the company's stock.\n\nNews Report:\n{state['news_report']}"
    agent = llm.bind_tools([search_tool])
    result = agent.invoke(prompt)
    return {"technical_report": result.content}

def financial_analyst_node_seq(state: SequentialState):
    console.print("--- (Sequential) CALLING FINANCIAL ANALYST ---")
    # This agent also uses the news report as context.
    prompt = f"Your task is to act as an expert Financial Analyst. Based on the following news report, analyze the company's recent financial performance.\n\nNews Report:\n{state['news_report']}"
    agent = llm.bind_tools([search_tool])
    result = agent.invoke(prompt)
    return {"financial_report": result.content}

def report_writer_node_seq(state: SequentialState):
    console.print("--- (Sequential) CALLING REPORT WRITER ---")
    prompt = f"""You are an expert report writer. Your task is to synthesize the information from the News, Technical, and Financial analysts into a single, cohesive report that directly answers the user's original request.
        User Request: {state['user_request']}

        Here are the reports to combine:
        ---
        News Report: {state['news_report']}
        ---
        Technical Report: {state['technical_report']}
        ---
        Financial Report: {state['financial_report']}
    """
    report = llm.invoke(prompt).content
    return {"final_report": report}

seq_graph_builder = StateGraph(SequentialState)
seq_graph_builder.add_node("news", news_report_seq)
seq_graph_builder.add_node("technical", technical_report_seq)
seq_graph_builder.add_node("financial", financial_analyst_node_seq)
seq_graph_builder.add_node("writer", report_writer_node_seq)

seq_graph_builder.set_entry_point("news")
seq_graph_builder.add_edge("news", "technical")
seq_graph_builder.add_edge("technical", "financial")
seq_graph_builder.add_edge("financial", "writer")
seq_graph_builder.add_edge("writer", END)

seq_agent_app = seq_graph_builder.compile()

console.print("Corrected sequential multi-agent system compiled successfully.")

dynamic_query = "Find the latest major news about Nvidia. Based on the sentiment of that news, conduct either a technical analysis (if the news is neutral or positive) or a financial analysis of their recent performance (if the news is negative)."

console.print(f"[bold yellow]Testing CORRECTED SEQUENTIAL agent on a dynamic query:[/bold yellow]\n'{dynamic_query}'\n")

# seq_agent_result = seq_agent_app.invoke({"user_request": dynamic_query})

# console.print("\n--- [bold red]Final Report from Sequential Agent[/bold red] ---")
# console.print(Markdown(seq_agent_result["final_report"]))

class BlackboardState(TypedDict):
    user_request: str
    # The central blackboard where agents post their findings as strings
    blackboard: List[str]
    # List of available agents for the controller to choose from
    available_agents: List[str]
    # The controller's next decision
    next_agent: Optional[str]
    writer_count: Annotated[int, lambda x, y: x + y]

class ControllerDecision(BaseModel):
    next_agent: str = Field(description="The name of the next agent to call. Must be one of ['News Analyst', 'Technical Analyst', 'Financial Analyst', 'Report Writer'] or 'FINISH'.")
    reasoning: str = Field(description="A brief reason for choosing the next agent.")

def create_blackboard_specialist(persona: str, agent_name: str):
    system_prompt = f"""You are an expert specialist agent: a {persona}.
    Your task is to contribute to a larger goal by performing your specific function.
    Read the initial User Request and the current Blackboard for context.
    Use your tools to find the required information.
    Finally, post your concise markdown report back to the blackboard. Your report should be signed with your name '{agent_name}'.
    """
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "User Request: {user_request}\n\nBlackboard (previous reports):\n{blackboard_str}")
    ])
    agent = prompt_template | llm.bind_tools([search_tool])

    def specialist_node(state: BlackboardState):
        console.print(f"--- (Blackboard) AGENT '{agent_name}' is working... ---")
        blackboard_str = "\n---\n".join(state["blackboard"])
        # messages = prompt_template.invoke({"user_request": state["user_request"], "blackboard_str": blackboard_str})
        # llm_with_tools = llm.bind_tools([search_tool])
        # result = llm_with_tools.invoke(messages)
        result = agent.invoke({
            "user_request": state["user_request"],
            "blackboard_str": blackboard_str
        })

        if result.tool_calls:
            tool_responses = []
            for tool_call in result.tool_calls:
                out = search_tool.invoke(tool_call["args"])
                tool_responses.append(f"Tool Result: {out}")
            
            final_prompt = f"Results found: {tool_responses}. Summarize this for the blackboard."
            result = llm.invoke(final_prompt)

        report = f"**Report from {agent_name}:**\n{result.content}"
        console.print(blackboard_str)
        console.print(f"**{agent_name} executed result: {result.content}**")
        # Append the new report to the list of blackboard entries
        update = {"blackboard": state["blackboard"] + [report]}
        
        if agent_name == "Report Writer":
            update["writer_count"] = 1

        return update

    return specialist_node

news_analyst_bb = create_blackboard_specialist("News Analyst", "News Analyst")
technical_analyst_bb = create_blackboard_specialist("Technical Analyst", "Technical Analyst")
financial_analyst_bb = create_blackboard_specialist("Financial Analyst", "Financial Analyst")
report_writer_bb = create_blackboard_specialist("Report Writer who synthesizes a final answer from the blackboard", "Report Writer")

def controller_node(state: BlackboardState):
    console.print("--- CONTROLLER: Analyzing blackboard... ---")

    if state["writer_count"] > 3:
        console.print("[bold red]--- CONTROLLER: Max Report Writer limit reached. Forcing FINISH. ---[/bold red]")
        return {"next_agent": "FINISH", "blackboard": state["blackboard"]}

    # Use a structured output LLM to make the decision
    controller_llm = llm.with_structured_output(ControllerDecision, method='json_mode')

    blackboard_content = "\n\n".join(state['blackboard'])
    agent_list = state['available_agents']

    # The new prompt is state-aware and goal-oriented.
    prompt = f"""You are the central controller of a multi-agent system. Your job is to analyze the shared blackboard and the original user request to decide which specialist agent should run next.

    **Original User Request:**
    {state['user_request']}

    **Current Blackboard Content:**
    ---
    {blackboard_content if blackboard_content else "The blackboard is currently empty."}
    ---

    **Available Specialist Agents:**
    {', '.join(agent_list)}

    **Your Task:**
    1.  Read the user request and the current blackboard content carefully.
    2.  Determine what the *next logical step* is to move closer to a complete answer.
    3.  Choose the single best agent to perform that step from the list of available agents.
    4.  If the user's request has been fully addressed and a final report has been written, choose 'FINISH'.

    **Critical Decision Logic:**
    1. If the 'Report Writer' has already posted a comprehensive final report that addresses the User Request, you MUST choose 'FINISH'.
    2. Do not call the 'Report Writer' multiple times unless the previous report was missing critical information.
    3. Check the Blackboard: If you see a signature from 'Report Writer' and the information is complete, STOP.

    Provide your decision in the required format.

    Your JSON response MUST include these exact keys:
    - "next_agent": str
    - "reasoning" : str
    """
    decision_result = controller_llm.invoke(prompt)
    console.print(f"--- CONTROLLER: Decision is to call '{decision_result.next_agent}'. Reason: {decision_result.reasoning} ---")

    # The dictionary returned here updates the 'next_agent' key in the graph's state
    return {"next_agent": decision_result.next_agent}

print("Blackboard components and corrected Controller node defined.")

def route_to_agent(state: BlackboardState):
    return state["next_agent"]

bb_graph_builder = StateGraph(BlackboardState)

bb_graph_builder.add_node("News Analyst", news_analyst_bb)
bb_graph_builder.add_node("Technical Analyst", technical_analyst_bb)
bb_graph_builder.add_node("Financial Analyst", financial_analyst_bb)
bb_graph_builder.add_node("Report Writer", report_writer_bb)
bb_graph_builder.add_node("controller", controller_node)

bb_graph_builder.set_entry_point("controller")
bb_graph_builder.add_conditional_edges(
    "controller",
    route_to_agent,
    {
        "News Analyst": "News Analyst",
        "Technical Analyst": "Technical Analyst",
        "Financial Analyst": "Financial Analyst",
        "Report Writer": "Report Writer",
        "FINISH": END
    }
)
bb_graph_builder.add_edge("News Analyst", "controller")
bb_graph_builder.add_edge("Technical Analyst", "controller")
bb_graph_builder.add_edge("Financial Analyst", "controller")
bb_graph_builder.add_edge("Report Writer", "controller")

bb_agent_app = bb_graph_builder.compile()
print("Blackboard system compiled successfully.")

console.print(f"[bold green]Testing BLACKBOARD system on the same dynamic query:[/bold green]\n'{dynamic_query}'\n")

agent_list = ["News Analyst", "Technical Analyst", "Financial Analyst", "Report Writer"]
initial_bb_input = {"user_request": dynamic_query, "blackboard": [], "available_agents": agent_list}

# We use stream to observe the step-by-step process
final_bb_output = None
for chunk in bb_agent_app.stream(initial_bb_input, {"recursion_limit": 100}):
    final_bb_output = chunk
    console.print("\n--- [bold purple]Current Blackboard State[/bold purple] ---")
    # Pretty print each report on the blackboard
    for i, report in enumerate(final_bb_output.get('blackboard', [])):
        console.print(f"--- Report {i+1} ---")
        console.print(Markdown(report))
    console.print("\n")

console.print("\n--- [bold green]Final Report from Blackboard System[/bold green] ---")
# The final report is the last item posted to the blackboard by the writer
# final_report_content = final_bb_output['blackboard'][-1]
console.print(Markdown(",".join(final_bb_output.keys())))