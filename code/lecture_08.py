import os
import json
import uuid
from typing import List, Annotated, TypedDict, Optional, Dict, Any, Tuple
from dotenv import load_dotenv

# LangChain components
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.embeddings import Embeddings
from openai import OpenAI

# LangGraph components
from langgraph.graph import StateGraph, END
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# For pretty printing
from rich.console import Console
from rich.markdown import Markdown

from langchain_community.vectorstores import FAISS
from langchain_neo4j import Neo4jGraph
from langchain_core.documents import Document
from langchain_community.graphs.graph_document import GraphDocument, Node as LCNode, Relationship as LCRelationship

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

api_key = os.getenv("DASHSCOPE_API_KEY", "your-api-key-here")
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

class QwenEmbeddings(Embeddings):
    """使用阿里云百炼的 text-embedding-v4 模型"""
    
    def __init__(self):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = "text-embedding-v4"  # 对应 Qwen3-Embedding 系列
        self.dimensions = 1024  # 支持 64-2048 维
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量生成文档向量"""
        completion = self.client.embeddings.create(
            model=self.model,
            input=texts,
            dimensions=self.dimensions,
            encoding_format="float"
        )
        return [item.embedding for item in completion.data]
    
    def embed_query(self, text: str) -> List[float]:
        """生成单个查询向量"""
        completion = self.client.embeddings.create(
            model=self.model,
            input=text,
            dimensions=self.dimensions,
            encoding_format="float"
        )
        return completion.data[0].embedding

embeddings = QwenEmbeddings()

try:
    episodic_vector_store = FAISS.from_texts(["Initial document to bootstrap the store"], embeddings)
except ImportError:
    console.print("[bold red]FAISS not installed. Please run `pip install faiss-cpu`.[/bold red]")
    episodic_vector_store = None

try:
    graph = Neo4jGraph(
        url=os.environ.get("NEO4J_URI"),
        username=os.environ.get("NEO4J_USERNAME"),
        password=os.environ.get("NEO4J_PASSWORD"),
        database="830416cc"
    )
    console.print("[bold green]connect neo4j database succ.[/bold green]")
    # Clear the graph for a clean run
    graph.query("MATCH (n) DETACH DELETE n")
    graph.query("""
        CREATE FULLTEXT INDEX entity IF NOT EXISTS 
        FOR (n:__Entity__) ON EACH [n.id]
    """)
    console.print("[bold green]Fulltext index 'entity' checked/created.[/bold green]")
except Exception as e:
    console.print(f"[bold red]Failed to connect to Neo4j: {e}. Please check your credentials and connection.[/bold red]")
    graph = None

# --- 3. Pydantic Models for the "Memory Maker" ---
# Define the structure of knowledge we want to extract.
class Node(BaseModel):
    id: str = Field(description="Unique identifier for the node, which can be a person's name, a company ticker, or a concept.")
    type: str = Field(description="The type of the node (e.g., 'User', 'Company', 'InvestmentPhilosophy').")
    properties: Dict[str, Any] = Field(default_factory=dict, description="A dictionary of properties for the node.")

class Relationship(BaseModel):
    source: Node = Field(description="The source node of the relationship.")
    target: Node = Field(description="The target node of the relationship.")
    type: str = Field(description="The type of the relationship (e.g., 'IS_A', 'INTERESTED_IN').")
    properties: Dict[str, Any] = Field(default_factory=dict, description="A dictionary of properties for the relationship.")

class KnowledgeGraph(BaseModel):
    """Represents the structured knowledge extracted from a conversation."""
    relationships: List[Relationship] = Field(description="A list of relationships to be added to the knowledge graph.")

# --- 4. The "Memory Maker" Agent ---
def create_memories(user_input: str, assistant_output: str):
    conversation = f"User: {user_input}\nAssistant: {assistant_output}"
    
    # 4a. Create Episodic Memory (Summarization)
    console.print("--- Creating Episodic Memory (Summary) ---")
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a summarization expert. Create a concise, one-sentence summary of the following user-assistant interaction. This summary will be used as a memory for future recall."),
        ("human", "Interaction:\n{interaction}")
    ])
    summarizer = summary_prompt | llm
    episodic_summary = summarizer.invoke({"interaction": conversation}).content
    
    new_doc = Document(page_content=episodic_summary, metadata={"created_at": uuid.uuid4().hex})
    episodic_vector_store.add_documents([new_doc])
    console.print(f"[green]Episodic memory created:[/green] '{episodic_summary}'")
    
    # 4b. Create Semantic Memory (Fact Extraction)
    console.print("--- Creating Semantic Memory (Graph) ---")
    extraction_llm = llm.with_structured_output(KnowledgeGraph, method='json_mode')
    extraction_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a knowledge extraction expert. 
            Your task is to identify key entities and their relationships from a conversation and model them as a graph. 
            Focus on user preferences, goals, and stated facts.
            
            Your JSON response MUST include these exact keys:
            - "relationships": list of Relationship
         
            Extract entities and their relationships into a graph structure.
            IMPORTANT: Every 'source' and 'target' MUST be a JSON object with 'id', 'type', and 'properties'.
            Example: "source": {{"id": "Alex", "type": "Person", "properties": {{}}}}
         """),
        ("human", "Extract all relationships from this interaction:\n{interaction}")
    ])
    extractor = extraction_prompt | extraction_llm
    try:
        kg_data = extractor.invoke({"interaction": conversation})
        if kg_data.relationships:
            all_lc_nodes = {} # 使用字典按 ID 去重，避免 unhashable 问题
            lc_rels = []
            
            for rel in kg_data.relationships:
                # 1. 提取并转换 Source Node
                s_id = rel.source.id
                if s_id not in all_lc_nodes:
                    all_lc_nodes[s_id] = LCNode(
                        id=s_id, 
                        type=rel.source.type, 
                        properties=rel.source.properties
                    )
                
                # 2. 提取并转换 Target Node
                t_id = rel.target.id
                if t_id not in all_lc_nodes:
                    all_lc_nodes[t_id] = LCNode(
                        id=t_id, 
                        type=rel.target.type, 
                        properties=rel.target.properties
                    )
                
                # 3. 构造 Relationship
                lc_rel = LCRelationship(
                    source=all_lc_nodes[s_id],
                    target=all_lc_nodes[t_id],
                    type=rel.type,
                    properties=rel.properties
                )
                lc_rels.append(lc_rel)
            
            # 构造文档对象
            doc = Document(page_content=conversation)
            # 这里的 nodes 已经是去重后的列表了
            graph_doc = GraphDocument(
                nodes=list(all_lc_nodes.values()), 
                relationships=lc_rels, 
                source=doc
            )
            
            graph.add_graph_documents([graph_doc])
            console.print(f"[green]Semantic memory created: Added {len(lc_rels)} relationships.[/green]")
        else:
            console.print("[yellow]No new semantic memories identified in this interaction.[/yellow]")
    except Exception as e:
        console.print(f"[red]Could not extract or save semantic memory: {e}[/red]")

if episodic_vector_store and graph:
    print("Memory components initialized successfully.")


class AgentState(TypedDict):
    user_input: str
    retrieved_memories: Optional[str]
    generation: str

def retrieve_memory(state: AgentState) -> Dict[str, Any]:
    console.print("--- Retrieving Memories ---")

    user_input = state['user_input']

    retrieved_docs = episodic_vector_store.similarity_search(user_input, k=2)
    episodic_memories = "\n".join([doc.page_content for doc in retrieved_docs])

    try:
        graph_schema = graph.get_schema
        semantic_memories = str(graph.query("""
            UNWIND $keywords AS keyword
            CALL db.index.fulltext.queryNodes("entity", keyword) YIELD node, score
            MATCH (node)-[r]-(related_node)
            RETURN node, r, related_node LIMIT 5
        """, {'keywords': user_input.split()}))
    except Exception as e:
        semantic_memories = f"Could not query graph: {e}"

    retrieved_content = f"Relevant Past Conversations (Episodic Memory):\n{episodic_memories}\n\nRelevant Facts (Semantic Memory):\n{semantic_memories}"
    console.print(f"[cyan]Retrieved Context:\n{retrieved_content}[/cyan]")
    
    return {"retrieved_memories": retrieved_content}

def generate_response(state: AgentState) -> Dict[str, Any]:
    console.print("--- Generating Response ---")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful and personalized financial assistant. Use the retrieved memories to inform your response and tailor it to the user. If the memories indicate a user's preference (e.g., they are a conservative investor), you MUST respect it."),
        ("human", "My question is: {user_input}\n\nHere are some memories that might be relevant:\n{retrieved_memories}")
    ])
    generator = prompt | llm
    generation = generator.invoke(state).content
    console.print(f"[green]Generated Response:\n{generation}[/green]")
    return {"generation": generation}

def update_memory(state: AgentState) -> Dict[str, Any]:
    """Node that updates the memory with the latest interaction."""
    console.print("--- Updating Memory ---")
    create_memories(state['user_input'], state['generation'])
    return {}

workflow = StateGraph(AgentState)

workflow.add_node("retrieve", retrieve_memory)
workflow.add_node("generate", generate_response)
workflow.add_node("update", update_memory)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", "update")
workflow.add_edge("update", END)

memory_agent = workflow.compile()
print("Memory-augmented agent graph compiled successfully.")

def run_interaction(query: str):
    result = memory_agent.invoke({"user_input": query})
    return result['generation']

console.print("\n--- 💬 INTERACTION 1: Seeding Memory ---")
run_interaction("Hi, my name is Alex. I'm a conservative investor, and I'm mainly interested in established tech companies.")

console.print("\n--- 💬 INTERACTION 2: Asking a specific question ---")
run_interaction("What do you think about Apple (AAPL)?")

console.print("\n--- 🧠 INTERACTION 3: THE MEMORY TEST ---")
run_interaction("Based on my goals, what's a good alternative to that stock?")

console.print("--- 🔍 Inspecting Episodic Memory (Vector Store) ---")
# We'll do a similarity search for a general concept to see what comes up
retrieved_docs = episodic_vector_store.similarity_search("User's investment strategy", k=3)
for i, doc in enumerate(retrieved_docs):
    print(f"{i+1}. {doc.page_content}")

console.print("\n--- 🕸️ Inspecting Semantic Memory (Graph Database) ---")
print(f"Graph Schema:\n{graph.get_schema}")

# Cypher query to see who is interested in what
query_result = graph.query("MATCH (n:User)-[r:INTERESTED_IN|HAS_GOAL]->(m) RETURN n, r, m")
print(f"Relationships in Graph:\n{query_result}")