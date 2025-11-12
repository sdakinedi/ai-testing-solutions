import os
import openai
from langchain_openai.chat_models import AzureChatOpenAI
from autogen import ConversableAgent

import warnings
warnings.filterwarnings('ignore')

openai.api_key  = os.environ['OPENAI_API_KEY']

llm_config = {"model": "gpt-3.5-turbo"}


agent = ConversableAgent(
    name="chatbot",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

#prompt = "tell me a joke"

prompt = "decompose the user story: Book a ride from A to B on Uber App \
         into 3 Scenarios with 3 ACs in given , when , then format."

reply = agent.generate_reply(
    messages=[{"content": prompt, "role": "user"}]
)
print(reply)
