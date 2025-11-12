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


task = '''
        Write a concise but engaging blogpost about
       DeepLearning.AI. Make sure the blogpost is
       within 100 words.
       '''
import autogen

writer = autogen.AssistantAgent(
    name="Writer",
    system_message="You are a writer. You write engaging and concise " 
        "blogpost (with title) on given topics. You must polish your "
        "writing based on the feedback you receive and give a refined "
        "version. Only return your final work without additional comments.",
    llm_config=llm_config,
)

reply = writer.generate_reply(messages=[{"content": task, "role": "user"}])

print(reply)

###### Adding the reflection ########

critic = autogen.AssistantAgent(
    name="Critic",
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
    llm_config=llm_config,
    system_message="You are a critic. You review the work of "
                "the writer and provide constructive "
                "feedback to help improve the quality of the content.",
)


result = critic.initiate_chat(
    recipient=writer,
    message=task,
    max_turns=2,
    summary_method="last_msg"
)


#print("result is: >>>>>>>>>>>>>>>>>>>>>>>>>>>> ", result)
print(result.cost)
print(result.summary)