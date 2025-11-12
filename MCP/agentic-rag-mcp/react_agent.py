from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain.agents.toolkits.mcp.toolkit import MCPToolkit
from langchain_openai import ChatOpenAI
import asyncio

async def run_agent_query(query: str):
    # 1. Connect to MCP server and fetch tools dynamically
    mcp_toolkit = MCPToolkit.from_url("http://localhost:8000")  # Change if your MCP is remote
    tools = await mcp_toolkit.get_tools()

    # 2. Initialize LLM
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # 3. Create React Agent using fetched tools
    from langchain.agents.react.base import REACT_CHAT_SYSTEM_MESSAGE
    agent = create_react_agent(llm=llm, tools=tools, prompt=REACT_CHAT_SYSTEM_MESSAGE)

    # 4. Create executor and invoke
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    result = await agent_executor.ainvoke({"input": query})
    return result['output']

# Run if invoked directly
if __name__ == "__main__":
    query = "What are the key points mentioned in the SRS document?"
    output = asyncio.run(run_agent_query(query))
    print("Agent Output:\n", output)
