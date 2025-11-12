import os
import openai
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import tool
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType

import warnings
warnings.filterwarnings('ignore')

import autogen

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']

llm_config={"model": "gpt-4-turbo"}


task = "Write a blogpost about the stock price performance of "\
"Nvidia in the past month. Today's date is 2024-04-23."

user_proxy = autogen.ConversableAgent(
    name="Admin",
    system_message="Give the task, and send "
    "instructions to writer to refine the blog post.",
    code_execution_config=False,
    llm_config=llm_config,
    human_input_mode="ALWAYS",
)


planner = autogen.ConversableAgent(
    name="Planner",
    system_message="Given a task, please determine "
    "what information is needed to complete the task. "
    "Please note that the information will all be retrieved using"
    " Python code. Please only suggest information that can be "
    "retrieved using Python code. "
    "After each step is done by others, check the progress and "
    "instruct the remaining steps. If a step fails, try to "
    "workaround",
    description="Planner. Given a task, determine what "
    "information is needed to complete the task. "
    "After each step is done by others, check the progress and "
    "instruct the remaining steps",
    llm_config=llm_config,
)

engineer = autogen.AssistantAgent(
    name="Engineer",
    llm_config=llm_config,
    description="An engineer that writes code based on the plan "
    "provided by the planner.",
)


# executor = autogen.ConversableAgent(
#     name="Executor",
#     system_message="Execute the code written by the "
#     "engineer and report the result.",
#     human_input_mode="NEVER",
#     code_execution_config={
#         "last_n_messages": 3,
#         "work_dir": "coding",
#         "use_docker": False,
#     },
# )

writer = autogen.ConversableAgent(
    name="Writer",
    llm_config=llm_config,
    system_message="Writer."
    "Please write blogs in markdown format (with relevant titles)"
    " and put the content in pseudo ```md``` code block. "
    "You take feedback from the admin and refine your blog.",
    description="Writer."
    "Write blogs based on the code execution results and take "
    "feedback from the admin to refine the blog."
)


### Define the group chat..

groupchat = autogen.GroupChat(
    agents=[user_proxy, engineer, writer, planner],
    messages=[],
    max_round=15,
)

# manager to manage the group conversation
manager = autogen.GroupChatManager(
    groupchat=groupchat, llm_config=llm_config
)

# start the group chat
groupchat_result = user_proxy.initiate_chat(
    manager,
    message=task,
)
