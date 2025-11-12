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
    "instructions to Planner to refine the test cases or automation code that are written.",
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
    "use Jira was management tool and "
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
    name="BusinessAnalyst",
    llm_config=llm_config,
    description="A software business analysts that decomposes the user story into ACs, Scenarios into gherkin format as given by the Planner and Give them to Tester"
)

Tester = autogen.AssistantAgent(
    name="Tester",
    llm_config=llm_config,
    description="An engineer that writes Test cases for the given ACs and Scenarios provided by the BusinessAnalyst. "
        "Writes as many Test cases as possible in the tabular format with columns as below"
        "Table consisting of columns as below "
        "Sno, TC id, Mapping to requirements, test description, test steps, "
        "type of test (positive or negative), priority, test data to test with , and expected result, automatable (Yes/No)"
)

AutomationTester = autogen.AssistantAgent(
    name="AutomationTester",
    llm_config=llm_config,
    description="An engineer that writes code based on the plan provided by the planner for the test cases "
            "provided by the Tester. "
            "Writes code only for the automatable tests"
)



### Define the group chat..

groupchat = autogen.GroupChat(
    agents=[user_proxy, planner, BA, Tester],
    messages=[],
    max_round=15,
)

groupchat = autogen.GroupChat(
    agents=[user_proxy,  planner, BA, Tester, AutomationTester],
    messages=[],
    max_round=10,
    allowed_or_disallowed_speaker_transitions={
        user_proxy: [planner, BA, Tester, AutomationTester],
        planner: [BA, Tester, AutomationTester],
        BA: [Tester, AutomationTester],
        Tester: [user_proxy,AutomationTester, BA],
        AutomationTester: [user_proxy, Tester],
    },
    speaker_transitions_type="allowed",
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
