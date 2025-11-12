import os
import openai
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI

import warnings
warnings.filterwarnings('ignore')

import autogen

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']

llm_config = {"model": "gpt-3.5-turbo"}

llm_config1 = {"model": "gpt-4-turbo"}


task = '''
        Book a ride from A to B on Uber App.
       '''


BA = autogen.AssistantAgent(
    name="Business Analysts",
    system_message="Your name is Cathy(Business Analyst) with solid experience in understanding requirements , \
                   , features, user stories and decomposing them into Acceptance criteria etc \
                    You can start decomposing the user story given to you as input into Scenarios and ACs in Gherkin format...\
                    Only return your final work without additional comments.",
    llm_config=llm_config,
)

reply = BA.generate_reply(messages=[{"content": task, "role": "user"}])

print(reply)

###### Adding the reflection ########

critic = autogen.AssistantAgent(
    name="Critic",
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
    llm_config=llm_config1,
    system_message="You are a critic. You review the work of "
                "the Writer /Business Analysts and provide constructive "
                "feedback to help improve the quality of the content.",
)


result = critic.initiate_chat(
    recipient=BA,
    message=task,
    max_turns=2,
    summary_method="last_msg"
)


#print("result is: >>>>>>>>>>>>>>>>>>>>>>>>>>>> ", result)
print(result.cost)
print(result.summary)