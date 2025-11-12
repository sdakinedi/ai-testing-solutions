import os
from langchain.agents import initialize_agent, AgentType
from langchain_experimental.tools import PythonREPLTool
from langchain_experimental.tools import PythonAstREPLTool
from langchain_community.agent_toolkits.jira.toolkit import JiraToolkit
from langchain_community.utilities.jira import JiraAPIWrapper
from langchain_openai import OpenAI
from langchain.tools import Tool

# Configure your Jira credentials
JIRA_URL = "https://myproject3186.atlassian.net"
JIRA_EMAIL = "satya.prasad.3186@gmail.com"
JIRA_API_TOKEN = "<JIRA_TOKEN>"

api_key = os.getenv("OPENAI_API_KEY")

os.environ["JIRA_INSTANCE_URL"] = JIRA_URL
os.environ["JIRA_API_TOKEN"] = JIRA_API_TOKEN
os.environ["JIRA_USERNAME"] = JIRA_EMAIL
os.environ["OPENAI_API_KEY"] = api_key
os.environ["JIRA_CLOUD"] = "myproject3186.atlassian.net"

llm = OpenAI(temperature=0)

jira = JiraAPIWrapper()
toolkit = JiraToolkit.from_jira_api_wrapper(jira)

toolkit = JiraToolkit.from_jira_api_wrapper(jira)
agent = initialize_agent(
    toolkit.get_tools(), llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

# Run the agent with a sample query
agent.invoke("Create a new Jira ticket for a bug in the login feature in the project named: project1")
