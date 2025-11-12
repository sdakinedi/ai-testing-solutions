import os
import openai
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain_openai.chat_models import AzureChatOpenAI
from rag_mcp_server1 import *  # Import FastMCP server tools here
import warnings
warnings.filterwarnings('ignore')

api_key = os.getenv("DIAL_API_KEY")

llm_model = "gpt-4-turbo"

# Initialize LLM
generator_llm = AzureChatOpenAI(
    api_version="2024-02-01",
    azure_endpoint="https://ai-proxy.lab.epam.com",
    api_key=api_key,
    azure_deployment="gpt-4o",
    temperature=0.0,
)

llm = ChatOpenAI(temperature=0.0, model=llm_model)

import asyncio

# Define a helper function to display tools clearly
def display_tool_details(tool):
    print(f"Tool Name: {tool.name}")
    print(f"  Description: {tool.description}")
    print(f"  Input Schema: {tool.inputSchema}")
    print(f"  Annotations: {tool.annotations}\n")


# Fetch tools from MCP dynamically
async def fetch_mcp_tools():
    tools = await mcp.list_tools()  # Get tools from FastMCP asynchronously
    if not tools:
        print("No tools found in MCP server!")
        return []

    # Display tools (for debugging)
    print("Available tools from MCP:")
    for tool in tools:
        display_tool_details(tool)

    return tools


from functools import partial

# Convert MCP tools into LangChain-compatible tools
def convert_mcp_tools_to_agent_tools(mcp_tools):
    """
    Convert MCP tools to LangChain-compatible tools by defining their appropriate behavior.
    """
    langchain_tools = []

    from langchain.tools import Tool  # Import LangChain's Tool module

    for mcp_tool in mcp_tools:
        # Use `functools.partial` to bind `mcp_tool.name` properly
        tool_func = partial(mcp.call_tool, mcp_tool.name)  # Partial binds the tool name to the call

        # Create a LangChain tool that properly handles inputs
        langchain_tool = Tool(
            name=mcp_tool.name,
            description=mcp_tool.description,
            func=tool_func  # Use the bound partial function
        )
        langchain_tools.append(langchain_tool)

    return langchain_tools

# Run the async function to fetch and convert MCP tools
async def prepare_tools():
    mcp_tools = await fetch_mcp_tools()  # Get MCP tools dynamically
    langchain_tools = convert_mcp_tools_to_agent_tools(mcp_tools)  # Convert to LangChain tools
    return langchain_tools


### Synchronously prepare MCP tools
mcp_tools = asyncio.run(prepare_tools())
print(f"Total MCP Tools converted: {len(mcp_tools)}")

# Load LangChain tools like LLM Math and Wikipedia
langchain_tools = load_tools(["llm-math", "wikipedia"], llm=generator_llm)

# Combine MCP tools and LangChain tools
combined_tools = mcp_tools + langchain_tools  # All tools combined

# Initialize the agent with combined tools
agent = initialize_agent(
    combined_tools,  # Pass combined tools here
    generator_llm, 
    agent=AgentType.OPENAI_FUNCTIONS,  # Use OpenAI Functions agent type
    handle_parsing_errors=True,
    verbose=True
)

# Test the agent for basic math operation
response = agent("What is the result of 49366 / 432")
print("Agent Response (Math query):", response)


feature = "Email confirmation to user"
requirements = "User should receive an email confirmation for order confirmation"
context = f"""the scope pertains to the E-Store product features for making Marvel Electronics and
                    Home Entertainment project live. It focuses on the company, the stakeholders and applications,
                    which allow for online sales, distribution and marketing of electronics."""

question = (
    f"What are the requirements related to {feature} don't call rag_status tool. use other tools if available."
)
result = agent(question)
print("Agent Response (Wikipedia query):", result)

# # Test normal RAG query
# question = (
#     f"can you write the test cases for the requirements {feature} {requirements} {context}?"
# )
# result = agent(question)
# print("Agent Response (RAG query):", result)


# Test evaluate_rag query
test_breadth = 1
test_depth = 1
question = (
    f"Please evaluate how good the RAG system is in the above context? "
    f"use {test_breadth} breadth and {test_depth} depth for the evaluation."
)
result = agent(question)
print("Agent Response (evaluate RAG query):", result)