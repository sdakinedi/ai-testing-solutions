import asyncio
import os
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain_openai import AzureChatOpenAI
#from mcp.client.fastmcp import FastMCPClient  # Assuming FastMCP client available
from mcp.server.fastmcp import FastMCPClient  # Assuming FastMCP server available


# ==============================
# MCP SERVER CONFIG
# ==============================
MCP_SERVER_URL = "http://localhost:8501"  # Change if MCP runs on different port
DIAL_API_KEY = os.getenv("DIAL_API_KEY")

# ==============================
# Initialize the LLM for reasoning
# ==============================
generator_llm = AzureChatOpenAI(
    api_version="2024-02-01",
    azure_endpoint="https://ai-proxy.lab.epam.com",
    api_key=DIAL_API_KEY,
    azure_deployment="gpt-4o",
    temperature=0.1,
)

# ==============================
# 1. Fetch list of MCP tools dynamically
# ==============================
async def fetch_mcp_tools():
    client = FastMCPClient(MCP_SERVER_URL)
    tools_metadata = await client.list_tools()
    return client, tools_metadata

# ==============================
# 2. Wrap MCP tool metadata into LangChain Tool
# ==============================
def wrap_mcp_tool(mcp_client: FastMCPClient, tool_meta: dict) -> Tool:
    """
    Converts MCP tool metadata into a LangChain-compatible Tool.
    """

    name = tool_meta["name"]
    desc = tool_meta.get("description", f"MCP tool: {name}")
    params = tool_meta.get("parameters", {})

    async def async_executor(**kwargs):
        """
        Executes MCP tool asynchronously.
        """
        result = await mcp_client.call_tool(name, **kwargs)
        return result

    def sync_executor(query: str):
        """
        Sync wrapper for LangChain Tool compatibility.
        """
        # Assume single param "question" by default if no params specified
        if params and "question" not in kwargs:
            kwargs = {k: query for k in params.keys()}
        else:
            kwargs = {"question": query}

        return asyncio.run(async_executor(**kwargs))

    return Tool(
        name=name,
        func=sync_executor,
        description=desc
    )

# ==============================
# 3. Build LangChain toolset dynamically
# ==============================
async def build_agent_toolset():
    mcp_client, tools_metadata = await fetch_mcp_tools()
    lc_tools = []
    for meta in tools_metadata:
        lc_tools.append(wrap_mcp_tool(mcp_client, meta))
    return lc_tools

# ==============================
# 4. Create the Agent dynamically
# ==============================
async def create_dynamic_agent():
    tools = await build_agent_toolset()
    agent = initialize_agent(
        tools,
        generator_llm,
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        verbose=True
    )
    return agent

# ==============================
# 5. Run an Agentic Query
# ==============================
async def run_agentic_query(user_query: str):
    agent = await create_dynamic_agent()
    result = await asyncio.to_thread(agent.run, user_query)
    return result

# ==============================
# MAIN ENTRY POINT
# ==============================
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        # Default test query
        query = "How many email confirmation requirements are there?"

    print(f"\n🤖 Running Agentic RAG Query: {query}\n")

    response = asyncio.run(run_agentic_query(query))

    print("\n✅ Final Agent Response:\n", response)
