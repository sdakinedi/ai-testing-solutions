import gradio as gr
import os
import logging
from openai import AzureOpenAI
from smolagents import CodeAgent
from smolagents import MCPClient
from types import SimpleNamespace 
from smolagents import ChatMessage, MessageRole


# Set up logging (optional)
logging.basicConfig(level=logging.INFO)

# Load API key from environment variable
api_key = os.getenv("DIAL_API_KEY")

# Initialize Azure OpenAI client
azure_client = AzureOpenAI(
    api_key=api_key,
    api_version="2024-06-01",
    azure_endpoint="https://ai-proxy.lab.epam.com"
)

# Wrapper class to adapt AzureOpenAI to smolagents expected interface
class SmolAzureModelWrapper:
    def __init__(self, client, model):
        self.client = client
        self.model = model

    def generate(self, messages, **kwargs):
        safe_messages = []
        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")
            name = msg.get("name", "function_tool")

            # Convert tool-call -> function and tool-response -> function
            if role in ("tool-call", "tool-response"):
                safe_messages.append({
                    "role": "function",
                    "name": name,
                    "content": content
                })
            else:
                safe_messages.append(msg)

        # Call Azure OpenAI
        response = self.client.chat.completions.create(
            model=self.model,
            messages=safe_messages,
            temperature=0.1,
            top_p=1,
            max_tokens=kwargs.get("max_tokens", 1024)
        )

        content = response.choices[0].message.content
        return ChatMessage(role=MessageRole.ASSISTANT, content=content)

try:
    # Initialize MCP client
    mcp_client = MCPClient(
        {"url": "https://satyaprasad3186-mcp-sentiment1.hf.space/gradio_api/mcp/sse"}
    )

    # Fetch tool list
    tools = mcp_client.get_tools()

    # Wrap the Azure client
    wrapped_model = SmolAzureModelWrapper(azure_client, "gpt-35-turbo")

    # Initialize the agent with tools and wrapped model
    agent = CodeAgent(
        tools=[*tools],
        model=wrapped_model,
        additional_authorized_imports=["json", "ast", "urllib", "base64"]
    )

    # Launch the Gradio chat interface
    demo = gr.ChatInterface(
        fn=lambda message, history: str(agent.run(message)),
        type="messages",
        examples=["Analyze the sentiment of the following text 'This is awesome'"],
        title="Agent with MCP Tools",
        description="This is a simple agent that uses MCP tools to answer questions.",
    )

    demo.launch()
finally:
    # Ensure clean disconnect
    mcp_client.disconnect()
