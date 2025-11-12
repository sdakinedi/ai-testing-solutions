import os
import openai
from langchain_openai.chat_models import AzureChatOpenAI
from autogen import ConversableAgent

openai.api_key  = os.environ['OPENAI_API_KEY']


llm_config = {"model": "gpt-3.5-turbo"}


##### Enable conversation between 2 different agents
BA = ConversableAgent(
    name="Cathy",
    system_message=
    "Your name is Cathy(Business Analyst) with solid experience in understanding requirements , \
    , features, user stories and decomposing them into Acceptance criteria etc \
    You can start decomposing the user story given to you as input into Scenarios and ACs in Gherkin format... ",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

PO = ConversableAgent(
    name="Joe",
    system_message=
    "Your name is Joe and you are a product owner "
    "YOu can start identifying the requirements , features required for your Application context \
    and let BA work on their work of identifying the US, Scenarios, TCs etc",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

userstory = "Book a ride from A to B on Uber App"
context = "Uber is a ride booking service app..."

message = f"I am Joe. Cathy, let's work together on the feature {userstory} and {context}"

print(message)

chat_result = PO.initiate_chat(
    recipient=BA, 
    message=message,
    max_turns=2,
)
