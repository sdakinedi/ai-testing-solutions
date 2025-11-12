import os
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain_openai.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
import requests
import json

# Your existing Azure OpenAI setup
api_key = os.getenv("DIAL_API_KEY")

agent_llm = AzureChatOpenAI(
    api_version="2024-02-01",
    azure_endpoint="https://ai-proxy.lab.epam.com",
    api_key=api_key, # type: ignore
    temperature=0.0,
)

class MCPRAGClient:
    """Client to interact with MCP RAG server"""
    
    def __init__(self):
        # Import MCP server functions directly
        self._initialize_mcp_functions()
    
    def _initialize_mcp_functions(self):
        """Initialize MCP server functions"""
        try:
            # Import the server module and initialize if needed
            import sys
            import os
            
            # Add current directory to path if needed
            current_dir = os.path.dirname(os.path.abspath(__file__))
            if current_dir not in sys.path:
                sys.path.append(current_dir)
            
            # Import server functions
            from server import query_documents, get_vectordb_info, add_pdf_to_vectordb
            
            self.query_documents_func = query_documents
            self.get_vectordb_info_func = get_vectordb_info
            self.add_pdf_func = add_pdf_to_vectordb
            
            print("✓ MCP RAG functions loaded successfully")
            
        except Exception as e:
            print(f"✗ Error loading MCP functions: {e}")
            # Fallback functions
            self.query_documents_func = lambda q: f"Error: MCP server not available - {e}"
            self.get_vectordb_info_func = lambda: f"Error: MCP server not available - {e}"
            self.add_pdf_func = lambda p: f"Error: MCP server not available - {e}"
    
    def query_documents(self, question: str) -> str:
        """Query documents through MCP server"""
        try:
            return self.query_documents_func(question)
        except Exception as e:
            return f"Error querying documents: {str(e)}"
    
    def get_vectordb_info(self) -> str:
        """Get vector database info through MCP server"""
        try:
            return self.get_vectordb_info_func()
        except Exception as e:
            return f"Error getting vectordb info: {str(e)}"
    
    def add_pdf(self, pdf_path: str) -> str:
        """Add PDF through MCP server"""
        try:
            return self.add_pdf_func(pdf_path)
        except Exception as e:
            return f"Error adding PDF: {str(e)}"

# Initialize MCP client
mcp_client = MCPRAGClient()

# Define tools for the agent
def create_rag_tools():
    """Create tools that interface with MCP RAG server"""
    
    def query_tool(question: str) -> str:
        """Query the document database for information"""
        return mcp_client.query_documents(question)
    
    def info_tool(dummy: str = "") -> str:
        """Get information about the loaded documents"""
        return mcp_client.get_vectordb_info()
    
    def add_pdf_tool(pdf_path: str) -> str:
        """Add a new PDF to the document database"""
        return mcp_client.add_pdf(pdf_path)
    
    tools = [
        Tool(
            name="query_documents",
            func=query_tool,
            description="Query the document database with a question. Input should be a clear question about the documents."
        ),
        Tool(
            name="get_database_info",
            func=info_tool,
            description="Get information about the current document database including number of documents loaded."
        ),
        Tool(
            name="add_pdf_document",
            func=add_pdf_tool,
            description="Add a new PDF document to the database. Input should be the full path to the PDF file."
        )
    ]
    
    return tools

# Create React agent prompt
react_prompt = PromptTemplate.from_template("""
You are an intelligent document assistant that can help users query and manage documents using a RAG (Retrieval Augmented Generation) system.

You have access to the following tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

When users ask questions about documents:
1. First check if you need information about the database
2. Then query the documents with their specific question
3. Provide helpful, concise answers based on the retrieved information

Question: {input}
{agent_scratchpad}
""")

# Create and setup the agent
def create_agentic_rag():
    """Create the agentic RAG system"""
    
    tools = create_rag_tools()
    
    # Create the React agent
    agent = create_react_agent(
        llm=agent_llm,
        tools=tools,
        prompt=react_prompt
    )
    
    # Create agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=5,
        handle_parsing_errors=True
    )
    
    return agent_executor

class AgenticRAGSystem:
    """Main class for Agentic RAG with MCP"""
    
    def __init__(self):
        self.agent_executor = create_agentic_rag()
    
    def query(self, question: str) -> str:
        """Query the agentic RAG system"""
        try:
            print(f"🤔 Processing query: {question}")
            result = self.agent_executor.invoke({"input": question})
            return result["output"]
        except Exception as e:
            print(f"❌ Error details: {str(e)}")
            return f"Error in agentic query: {str(e)}"
    
    def chat(self):
        """Interactive chat interface"""
        print("Agentic RAG System with MCP - Interactive Mode")
        print("Type 'quit' to exit")
        print("-" * 50)
        
        while True:
            question = input("\nYou: ").strip()
            if question.lower() in ['quit', 'exit', 'q']:
                break
            
            if question:
                print("\nAgent:")
                response = self.query(question)
                print(response)

# Example usage
if __name__ == "__main__":
    print("🚀 Initializing Agentic RAG System with MCP...")
    
    # Initialize the agentic RAG system
    try:
        agentic_rag = AgenticRAGSystem()
        print("✓ Agentic RAG system initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize: {e}")
        exit(1)
    
    # Example queries
    test_queries = [
        "What information do you have in your database?",
        "What is the role of the Scrum Master in a Scrum team?",
        "How many documents are currently loaded?",
        "What are the key principles of Scrum methodology?"
    ]
    
    print("\n" + "="*50)
    print("Testing Agentic RAG System with MCP:")
    print("="*50)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n[{i}/{len(test_queries)}] Query: {query}")
        print("-" * 50)
        response = agentic_rag.query(query)
        print(f"Response: {response}")
        print("-" * 30)
    
    # Uncomment to start interactive chat
    # print("\n" + "="*50)
    print("Starting interactive chat...")
    agentic_rag.chat()