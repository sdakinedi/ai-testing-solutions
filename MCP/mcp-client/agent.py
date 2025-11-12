import gradio as gr
import os

from smolagents import InferenceClientModel, CodeAgent, MCPClient, ToolCallingAgent
from agent1 import *
from openai import AzureOpenAI


# Load API key from environment variable
api_key = os.getenv("DIAL_API_KEY")

# Initialize Azure OpenAI client
azure_client = AzureOpenAI(
    api_key=api_key,
    api_version="2024-06-01",
    azure_endpoint="https://ai-proxy.lab.epam.com"
)

try:
    mcp_client = MCPClient(
        {"url": "https://satyaprasad3186-mcp-sentiment1.hf.space/gradio_api/mcp/sse"}
    )
    tools = mcp_client.get_tools()

    #model = InferenceClientModel(token=os.getenv("HF_TOKEN"))  # Use Hugging Face token if available

    wrapped_model = SmolAzureModelWrapper(azure_client, "gpt-35-turbo")

    print(f"Using model: {wrapped_model}")
   # agent = CodeAgent(tools=[*tools], model=wrapped_model, additional_authorized_imports=["json", "ast", "urllib", "base64"])
    agent = ToolCallingAgent(tools=[*tools], model=wrapped_model)

    demo = gr.ChatInterface(
        fn=lambda message, history: str(agent.run(message)),
        type="messages",
        examples=["Analyze the sentiment of the following text 'This is awesome'"],
        title="Agent with MCP Tools",
        description="This is a simple agent that uses MCP tools to answer questions.",
    )

    demo.launch(share=True)
finally:
    mcp_client.disconnect()