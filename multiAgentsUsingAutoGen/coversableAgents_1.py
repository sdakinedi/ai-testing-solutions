import os
import openai
from langchain_openai.chat_models import AzureChatOpenAI
from autogen import ConversableAgent

openai.api_key  = os.environ['OPENAI_API_KEY']


# llm = AzureChatOpenAI(
#     openai_api_version="2023-07-01-preview",
#     api_key="b13467a9fde7430dae79d77dcbf023dd",
#     azure_endpoint="https://ai-proxy.lab.epam.com",
#     azure_deployment="gpt-35-turbo",
# )

# print("LLM is: ", llm.model_name)

llm_config = {"model": "gpt-3.5-turbo"}

#llm_config = {"model": llm.model_name}

agent = ConversableAgent(
    name="chatbot",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

reply = agent.generate_reply(
    messages=[{"content": "Tell me a joke.", "role": "user"}]
)
print(reply)
