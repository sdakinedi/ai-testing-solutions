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

llm_config = {"model": "gpt-3.5-turbo"}

llm = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo")


@tool
def ActAsBusinessAnalyst(text: str) -> str:
    """
    Act as a business analysts with vast experience in seeing the business \
    side of the application and interact with different stakeholder \
    decompose the requirement/feature/stories/scenario and provide\
    output in a tabular format as asked.
    """
    return str

@tool
def ActAsTester(text: str) -> str:
    """
    You are a Software Tester with vast testing experience into functional and non-functional testing areas. \
    You are great at understanding the scenarios and writing test cases for them. 
    You are so good because you are able to break down 
    those scenarios into confined test cases with clear description, testability and steps to reproduce \
     and match them to requirements using traceability matrix \
    Write the TCs such a way that they follow the test case design techniques like Boundary values analysis, equivalence partitioning etc.,. \
    """
    return str("")


# Create agent
agent= initialize_agent(
    [ActAsBusinessAnalyst, ActAsTester], 
    llm, 
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose = True)


prompt = "decompose the user story: \
          Booking a ride on Uber app, \
          Decompose it into 3 ACs and 3 scenarios each with Given, When, Then\
          Write the POsitive and Negative test cases for each scenario. \
          Write everything in a table format with \
          Requirement ID, Test case ID, Test case description, Test data, Expected result \
          "


# try:
#     result = agent.run(prompt)
#     print(result)
# except: 
#     print("exception on external access")

result = agent.run(prompt)
print(result)

