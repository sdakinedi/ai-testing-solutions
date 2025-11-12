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

llm_config={"model": "gpt-3.5-turbo"}


task = "Perform the testing activities including the automation of the user story mentioned below\ \
    user story: Booking a ride on Uber app from A to B "

user_proxy = autogen.ConversableAgent(
    name="Admin",
    system_message="Give the task, and send "
    "instructions to writer to refine the test cases or automation code that are written.",
    code_execution_config=False,
    llm_config=llm_config,
    human_input_mode="ALWAYS",
)


planner = autogen.ConversableAgent(
    name="Planner",
    system_message="Given a task, please determine "
    "what information is needed to complete the task. "
    "Please only suggest information that can be "
    "relevant to testing the above user story."
    "use Jira mas management tool and "
    "use Selenium with Python TestNG as web automation techstack"
    "After each step is done by others, check the progress and "
    "instruct the remaining steps. If a step fails, try to "
    "workaround",
    description="Planner. Given a task, determine what "
    "information is needed to complete the task. "
    "After each step is done by others, check the progress and "
    "instruct the remaining steps",
    llm_config=llm_config,
)


BA = autogen.AssistantAgent(
    name="Business Analyst",
    llm_config=llm_config,
    description="A software business analysts that decomposes the user story into ACs, Scenarios into gherkin format "
)

Tester = autogen.AssistantAgent(
    name="tester",
    llm_config=llm_config,
    description="An engineer that writes Test cases for the given ACs and Scenarios provided by the Business Analyst. "
        "Writes as many Test cases as possible in the tabular format with columns as below"
        "Table consisting of columns as below "
        "Sno, TC id, Mapping to requirements, test description, test steps, "
        "type of test (positive or negative), priority, test data to test with , and expected result, automatable (Yes/No)"
)

AutomationTester = autogen.AssistantAgent(
    name="automation tester",
    llm_config=llm_config,
    description="An engineer that writes code based on the plan "
            "provided by the Tester. "
            "Writes code only for the automatable tests"
)

executor = autogen.ConversableAgent(
    name="Executor",
    system_message="Execute the code written by the "
    "AutomationTester and report the result.",
    human_input_mode="NEVER",
    code_execution_config={
        "last_n_messages": 10,
        "work_dir": "coding",
        "use_docker": False,
    },
)

Reporter = autogen.ConversableAgent(
    name="Reporter",
    llm_config=llm_config,
    system_message="Reporter."
    "Please write the report of the above conversation/ in tabular format"
    " and put the content in a file called testing.txt file "
    "You take feedback from the admin and refine your report.",
    description="Reporter."
    "Write the report of the above conversation in tabular format"
    "Get feedback from the admin to refine your writing."
)


### Define the group chat..

groupchat = autogen.GroupChat(
    agents=[user_proxy, planner, BA, Tester, AutomationTester],
    messages=[],
    max_round=15,
)

# manager to manage the group conversation
manager = autogen.GroupChatManager(
    groupchat=groupchat, llm_config=llm_config
)

# Talk to manager only
# Its the manager who directs the output to relevant team member/agent.
groupchat_result = user_proxy.initiate_chat(
    manager,
    message=task,
)
