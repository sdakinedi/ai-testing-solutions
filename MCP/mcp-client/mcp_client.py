from smolagents.mcp_client import MCPClient

import gradio as gr
import os

from mcp import StdioServerParameters
from smolagents import InferenceClientModel, CodeAgent, ToolCollection, MCPClient

with MCPClient(
    {"url": "https://satyaprasad3186-mcp-sentiment1.hf.space/gradio_api/mcp/sse"}
) as tools:
    # Tools from the remote server are available
    print("\n".join(f"{t.name}: {t.description}" for t in tools))


