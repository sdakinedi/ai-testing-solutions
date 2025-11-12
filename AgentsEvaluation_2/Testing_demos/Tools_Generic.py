import os
import openai
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
#from langchain.tools.python.tool import PythonREPLTool
from langchain_community.utilities.python import PythonREPL
from langchain.agents import tool
import warnings
warnings.filterwarnings('ignore')


openai.api_key  = os.environ['OPENAI_API_KEY']

llm_model = "gpt-4-turbo"

llm = ChatOpenAI(temperature=0.0, model=llm_model)

tools = load_tools(["llm-math","wikipedia"], llm=llm)


agent= initialize_agent(
    tools, 
    llm, 
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose = True)

agent("What is the result of 49366 / 432")


####### Connect to wikipedia

question = "Tom M. Mitchell is an American computer scientist \
and the Founders University Professor at Carnegie Mellon University (CMU)\
what book did he write?"
result = agent(question) 


