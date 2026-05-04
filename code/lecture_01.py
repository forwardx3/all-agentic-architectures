import os
import json
from typing import List, TypedDict, Optional
from dotenv import load_dotenv

from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field # Corrected import for Pydantic v2
from langgraph.graph import StateGraph, END

# For pretty printing
from rich.console import Console
from rich.markdown import Markdown
from rich.syntax import Syntax

# --- API Key and Tracing Setup ---
load_dotenv()

# Set up LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Agentic Architecture - Reflection (Nebius)"

# Check that the keys are set
if not os.environ.get("DEEPSEEK_API_KEY"):
    print("DEEPSEEK_API_KEY not found. Please create a .env file and set it.")
if not os.environ.get("LANGCHAIN_API_KEY"):
    print("LANGCHAIN_API_KEY not found. Please create a .env file and set it for tracing.")

print("Environment variables loaded and tracing is set up.")

class DraftModel(BaseModel):
    code: str = Field(description="The python code generated to solve the user's request.")
    explanation: str = Field(description="A brief explanation of how the code works.")

class Critique(BaseModel):
    has_errors: bool = Field(description="Does the code have any potential bugs or logical errors?")
    is_efficient: bool = Field(description="Is the code written in an efficient and optimal way?")
    suggested_improvements: List[str] = Field(description="Specific, actionable suggestions for improving the code.")
    critique_summary: str = Field(description="A summary of the critique.")

class RefinedCode(BaseModel):
    """Schema for the final, refined code after incorporating the critique."""
    refined_code: str = Field(description="The final, improved Python code.")
    refinement_summary: str = Field(description="A summary of the changes made based on the critique.")

print("Pydantic models for Draft, Critique, and RefinedCode have been defined.")

# Use a powerful Nebius model for generation and critique
# llm = ChatDeepSeek(
#     model="deepseek-v4-flash",  # 或使用 "deepseek-reasoner" 获得推理链
#     temperature=0.2,
#     # api_key 会自动从环境变量 DEEPSEEK_API_KEY 读取
# )

llm = ChatOpenAI(
    model="deepseek-v4-flash",
    temperature=0.2,
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base="https://api.deepseek.com/v1",
)

# Initialize console for pretty printing
console = Console()

print("Nebius LLM and Console are initialized.")

def generate_node(state):
    console.print("--- 1. Generating Initial Draft ---")
    generator_llm = llm.with_structured_output(DraftModel, method='json_mode')

    prompt = f"""You are an expert Python programmer. Write a Python function to solve the following request.
    Provide a simple, clear implementation and an explanation.
    
    Request: {state['user_request']}

    Your JSON response MUST include these exact keys:
    - "code": string
    - "explanation": string
    """

    draft = generator_llm.invoke(prompt)
    return {"draft": draft.model_dump()}

def critic_node(state):
    console.print("--- 2. Critiquing Draft ---")
    critic_llm = llm.with_structured_output(Critique, method='json_mode')

    code_to_critique = state['draft']['code']
    prompt = f"""You are an expert code reviewer and senior Python developer. Your task is to perform a thorough critique of the following code.
    
    Analyze the code for:
    1.  **Bugs and Errors:** Are there any potential runtime errors, logical flaws, or edge cases that are not handled?
    2.  **Efficiency and Best Practices:** Is this the most efficient way to solve the problem? Does it follow standard Python conventions (PEP 8)?
    
    Provide a structured critique with specific, actionable suggestions.
    
    Code to Review:
    ```python
    {code_to_critique}
    ```

    Your JSON response MUST include these exact keys:
    - "has_errors": boolean
    - "is_efficient": boolean
    - "suggested_improvements": list of strings
    - "critique_summary": string
    """
    
    critique = critic_llm.invoke(prompt)
    return {"critique": critique.model_dump()} # Corrected: use .model_dump()

def refine_node(state):
    console.print("--- 3. Refining Code ---")
    refiner_llm = llm.with_structured_output(RefinedCode, method='json_mode')

    draft_code = state['draft']['code']
    critique_suggestions = json.dumps(state['critique'], indent=2)

    prompt = f"""You are an expert Python programmer tasked with refining a piece of code based on a critique.
    
    Your goal is to rewrite the original code, implementing all the suggested improvements from the critique.
    
    **Original Code:**
    ```python
    {draft_code}
    ```
    
    **Critique and Suggestions:**
    {critique_suggestions}
    
    Please provide the final, refined code and a summary of the changes you made.

    Your JSON response MUST include these exact keys:
    - "refined_code": string
    - "refinement_summary": string
    """

    refined_code = refiner_llm.invoke(prompt)
    return {'refined_code': refined_code.model_dump()}


class ReflectionState(TypedDict):
    user_request: str
    draft: Optional[dict]
    critique: Optional[dict]
    refined_code: Optional[dict]

print("ReflectionState TypedDict defined.")

graph_builder = StateGraph(ReflectionState)
graph_builder.add_node("generator", generate_node)
graph_builder.add_node("critic", critic_node)
graph_builder.add_node("refiner", refine_node)

graph_builder.set_entry_point("generator")
graph_builder.add_edge("generator", "critic")
graph_builder.add_edge("critic", "refiner")
graph_builder.add_edge("refiner", END)

reflection_app = graph_builder.compile()

print("Reflection graph compiled successfully!")

try:
    from IPython.display import Image, display
    png_image = reflection_app.get_graph().draw_png()
    display(Image(png_image))
except Exception as e:
    print(f"Graph visualization failed: {e}. Please ensure pygraphviz is installed.")


user_request = "Write a Python function to find the nth Fibonacci number."
initial_input = {"user_request": user_request}

console.print(f"[bold cyan]🚀 Kicking off Reflection workflow for request:[/bold cyan] '{user_request}'\n")

# Corrected: This loop correctly captures the final, fully-populated state
final_state = None
for state_update in reflection_app.stream(initial_input, stream_mode="values"):
    final_state = state_update

console.print("\n[bold green]✅ Reflection workflow complete![/bold green]")

# Check if final_state is available and has the expected keys
if final_state and 'draft' in final_state and 'critique' in final_state and 'refined_code' in final_state:
    console.print(Markdown("--- ### Initial Draft ---"))
    console.print(Markdown(f"**Explanation:** {final_state['draft']['explanation']}"))
    # Use rich's Syntax for proper code highlighting
    console.print(Syntax(final_state['draft']['code'], "python", theme="monokai", line_numbers=True))

    console.print(Markdown("\n--- ### Critique ---"))
    console.print(Markdown(f"**Summary:** {final_state['critique']['critique_summary']}"))
    console.print(Markdown(f"**Improvements Suggested:**"))
    for improvement in final_state['critique']['suggested_improvements']:
        console.print(Markdown(f"- {improvement}"))

    console.print(Markdown("\n--- ### Final Refined Code ---"))
    console.print(Markdown(f"**Refinement Summary:** {final_state['refined_code']['refinement_summary']}"))
    console.print(Syntax(final_state['refined_code']['refined_code'], "python", theme="monokai", line_numbers=True))
else:
    console.print("[bold red]Error: The `final_state` is not available or is incomplete. Please check the execution of the previous cells.[/bold red]")


class CodeEvaluation(BaseModel):
    """Schema for evaluating a piece of code."""
    correctness_score: int = Field(description="Score from 1-10 on whether the code is logically correct.")
    efficiency_score: int = Field(description="Score from 1-10 on the code's algorithmic efficiency.")
    style_score: int = Field(description="Score from 1-10 on code style and readability (PEP 8). ")
    justification: str = Field(description="A brief justification for the scores.")

judge_llm = llm.with_structured_output(CodeEvaluation, method='json_mode')

def evaluate_code(code_to_evaluate: str):
    prompt = f"""You are an expert judge of Python code. Evaluate the following function on a scale of 1-10 for correctness, efficiency, and style. Provide a brief justification.
    
    Code:
    ```python
    {code_to_evaluate}
    ```

    Your JSON response MUST include these exact keys:
    - "correctness_score": int
    - "efficiency_score": int
    - "style_score": int
    - "justification": string
    """
    return judge_llm.invoke(prompt)

if final_state and 'draft' in final_state and 'refined_code' in final_state:
    console.print("--- Evaluating Initial Draft ---")
    initial_draft_evaluation = evaluate_code(final_state['draft']['code'])
    console.print(initial_draft_evaluation.model_dump()) # Corrected: use .model_dump()

    console.print("\n--- Evaluating Refined Code ---")
    refined_code_evaluation = evaluate_code(final_state['refined_code']['refined_code'])
    console.print(refined_code_evaluation.model_dump()) # Corrected: use .model_dump()
else:
    console.print("[bold red]Error: Cannot perform evaluation because the `final_state` is incomplete.[/bold red]")

