import os
import openai
import warnings
import asyncio
from functools import partial

from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import initialize_agent, AgentType
from langchain_openai.chat_models import AzureChatOpenAI
from langchain.tools import Tool, StructuredTool
from pydantic import BaseModel

from rag_mcp_server1 import *  # Import FastMCP server tools here

warnings.filterwarnings('ignore')

api_key = os.getenv("DIAL_API_KEY")
llm_model = "gpt-4-turbo"

#  Initialize LLM
generator_llm = AzureChatOpenAI(
    api_version="2024-02-01",
    azure_endpoint="https://ai-proxy.lab.epam.com",
    api_key=api_key,
    azure_deployment="gpt-4o",
    temperature=0.0,
)

llm = ChatOpenAI(temperature=0.0, model=llm_model)

#  Helper to display tools for debugging
def display_tool_details(tool):
    print(f"Tool Name: {tool.name}")
    print(f"  Description: {tool.description}")
    print(f"  Input Schema: {tool.inputSchema}")
    print(f"  Annotations: {tool.annotations}\n")

#  Fetch MCP tools dynamically
async def fetch_mcp_tools():
    tools = await mcp.list_tools()  # Get tools from MCP asynchronously
    if not tools:
        print("No tools found in MCP server!")
        return []

    # Debug print
    print("Available tools from MCP:")
    for tool in tools:
        display_tool_details(tool)

    return tools

# Sync wrapper for async MCP tool calls
def sync_mcp_tool_call(tool_name, arguments):
    try:
        loop = asyncio.get_running_loop()
        raise RuntimeError("Cannot run sync wrapper in async context")
    except RuntimeError:
        return asyncio.run(mcp.call_tool(tool_name, arguments))

# Convert MCP tools to simple LangChain tools
def convert_mcp_tools_to_agent_tools_v2(mcp_tools):
    langchain_tools = []

    for mcp_tool in mcp_tools:
        def make_tool_func(tool_name):
            async def async_tool_func(arguments):
                return await mcp.call_tool(tool_name, arguments)

            def sync_tool_func(query):
                # Parse input - handle both string and dict
                if isinstance(query, str):
                    arguments = {"question": query}
                elif isinstance(query, dict):
                    arguments = query
                else:
                    arguments = {"question": str(query)}

                try:
                    result = asyncio.run(async_tool_func(arguments))
                    return result
                except Exception as e:
                    return f"Error executing {tool_name}: {str(e)}"

            return sync_tool_func

        langchain_tool = Tool(
            name=mcp_tool.name,
            description=mcp_tool.description,
            func=make_tool_func(mcp_tool.name)
        )
        langchain_tools.append(langchain_tool)

    return langchain_tools

# Prepare MCP tools
async def prepare_tools():
    mcp_tools = await fetch_mcp_tools()
    langchain_tools = convert_mcp_tools_to_agent_tools_v2(mcp_tools)
    return langchain_tools

#  Run async preparation
mcp_tools = asyncio.run(prepare_tools())
print(f"Total MCP Tools converted: {len(mcp_tools)}")

# LangChain built-in tools
langchain_tools = load_tools(["llm-math", "wikipedia"], llm=generator_llm)

#  MANUAL PATCH for evaluate_rag tool as StructuredTool

# Step 1: Define the args schema
class EvaluateRagArgs(BaseModel):
    test_breadth: int
    test_depth: int

# Step 2: Define a sync wrapper for the MCP structured call
def evaluate_rag_sync(test_breadth: int, test_depth: int):
    try:
        return asyncio.run(
            mcp.call_tool("evaluate_rag", {
                "test_breadth": test_breadth,
                "test_depth": test_depth
            })
        )
    except Exception as e:
        return f"Error calling evaluate_rag: {str(e)}"

# Step 3: Create the StructuredTool
evaluate_rag_tool = StructuredTool.from_function(
    func=evaluate_rag_sync,
    name="evaluate_rag",
    description="Evaluate the RAG system with specified breadth & depth.",
    args_schema=EvaluateRagArgs
)

# Combine all tools
combined_tools = mcp_tools + langchain_tools
combined_tools.append(evaluate_rag_tool)  # Add patched structured tool

# Initialize the agent
agent = initialize_agent(
    combined_tools,
    generator_llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    handle_parsing_errors=True,
    verbose=True
)

# Test simple math query
response = agent("What is the result of 49366 / 432")
print("Agent Response (Math query):", response)

# Test normal RAG query
question = (
    "What are the requirements related to email confirmation? don't call rag_status tool. use other tools if available. "
)
result = agent(question)
print("Agent Response (RAG query):", result)

# Test evaluate_rag query
test_breadth = 1
test_depth = 1
question = (
    f"Please evaluate how good the RAG system is in the above context?"
    f"use {test_breadth} breadth and {test_depth} depth for the evaluation."
)
result = agent(question)
print("Agent Response (evaluate RAG query):", result)
