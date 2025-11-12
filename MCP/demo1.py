from smolagents import InferenceClientModel, CodeAgent, ToolCollection
from mcp.client.stdio import StdioServerParameters
import os

api_key = os.getenv("HF_KEY")
model = InferenceClientModel(api_key=api_key)

server_parameters = StdioServerParameters(command="uv", args=["run", "server.py"])

with ToolCollection.from_mcp(
    server_parameters, trust_remote_code=True
) as tool_collection:
    agent = CodeAgent(tools=[*tool_collection.tools], model=model)
    agent.run("What's the weather in Tokyo?")
