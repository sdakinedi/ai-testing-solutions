import os
import openai
#from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI

from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain,RouterOutputParser
from langchain.prompts import PromptTemplate

import warnings
warnings.filterwarnings('ignore')


from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

# Add Phoenix API Key for tracing
PHOENIX_API_KEY = "692a40cd57ca8e0f86a:57e4727"
os.environ["PHOENIX_CLIENT_HEADERS"] = f"api_key={PHOENIX_API_KEY}"
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "https://app.phoenix.arize.com"

# account for deprecation of LLM model
import datetime
# Get the current date
current_date = datetime.datetime.now().date()

# Define the date after which the model should be set to "gpt-3.5-turbo"
target_date = datetime.date(2024, 6, 12)

# Set the model variable based on the current date
if current_date > target_date:
    llm_model = "gpt-3.5-turbo"
else:
    llm_model = "gpt-3.5-turbo"

llm = ChatOpenAI(temperature=0.0, model=llm_model)


BA_template = """You are a Business Analyst \
You are great at understanding the requirements, converting them into functional and non-functional\
and decompose them into features, stories, Accentance criteria and Scenarios etc  \
When you don't know the answer to a question you admit\
that you don't know. And Ask for a clarification\

Here is a question:
{input}"""


SoftwareTester_template = """You are a Software Tester with vast testing experience into functional and non-functional testing areas. \
You are great at understanding the scenarios and writing test cases for them. 
You are so good because you are able to break down 
those scenarios into confined test cases with clear description, testability and steps to reproduce and match them to requirements using traceability matrix, 
Write the TCs such a way that they follow the test case design techniques like Boundary values analysis, equivalence partitioning etc.,.

Here is a question:
{input}"""

SoftwareAutomationTester_template = """You are a very good automation tester with extensive skills in Test automation of WEB, API, Mobile and Desktop application etc. \
You have an excellent knowledge of and understanding of Test scenarios/cases  
and come up an automation code with best practices using the frameworks and/or programming languages. 
You write the code in a way that is adhering to Clean Code Standards like SOLID, DRY, KISS, YAGNI principle and make sure the tech debt is minimum \
based on your judgements.

Here is a question:
{input}"""



prompt_infos = [
    {
        "name": "BA", 
        "description": "Good for answering questions from a business analysts role", 
        "prompt_template": BA_template
    },
    {
        "name": "SoftwareTester", 
        "description": "Good for answering questions from a Tester Role", 
        "prompt_template": SoftwareTester_template
    },
    {
        "name": "AutomationTester", 
        "description": "Good for answering questions from an Automation Tester Role", 
        "prompt_template": SoftwareAutomationTester_template
    } 
]

llm = ChatOpenAI(temperature=0, model=llm_model)


destination_chains = {}

for p_info in prompt_infos:
    name = p_info["name"]
    prompt_template = p_info["prompt_template"]
    prompt = ChatPromptTemplate.from_template(template=prompt_template)
    chain = LLMChain(llm=llm, prompt=prompt)
    destination_chains[name] = chain  
    
destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
destinations_str = "\n".join(destinations)

print("@@@@@@@@@@@@ destinations_str @@@@@@: ", destinations_str)


default_prompt = ChatPromptTemplate.from_template("{input}")
default_chain = LLMChain(llm=llm, prompt=default_prompt)


MULTI_PROMPT_ROUTER_TEMPLATE = """Given a raw text input to a \
language model select the model prompt best suited for the input. \
You will be given the names of the available prompts and a \
description of what the prompt is best suited for. \
You may also revise the original input if you think that revising\
it will ultimately lead to a better response from the language model.

<< FORMATTING >>
Return a markdown code snippet with a JSON object formatted to look like:
```json
{{{{
    "destination": string \ name of the prompt to use or "DEFAULT"
    "next_inputs": string \ a potentially modified version of the original input
}}}}
```

REMEMBER: "destination" MUST be one of the candidate prompt \
names specified below OR it can be "DEFAULT" if the input is not\
well suited for any of the candidate prompts.
REMEMBER: "next_inputs" can just be the original input \
if you don't think any modifications are needed.

<< CANDIDATE PROMPTS >>
{destinations}

<< INPUT >>
{{input}}

<< OUTPUT (markdown)>>"""


router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(
    destinations=destinations_str
)
router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser(),
)

router_chain = LLMRouterChain.from_llm(llm, router_prompt)


chain = MultiPromptChain(router_chain=router_chain, 
                         destination_chains=destination_chains, 
                         default_chain=default_chain, verbose=True
                        )



## Give this context to LLM first up.

input = "as a business analyst: Decompose the user story: Buying a pair of shoes from online shopping cart"

response1 = chain.run(input)
print(response1)


scenarios = " \
Feature: Online Shopping \
   - Story: As a customer, I want to be able to browse and purchase items online.\
     - Acceptance Criteria:\
       - The website should have a user-friendly interface for easy navigation.\
       - Customers should be able to search for specific items, such as shoes.\
       - Customers should be able to add items to their shopping cart.\
       - Customers should be able to view their shopping cart and proceed to checkout.\
"

#response2 = chain.run("Generate 3 positive TCs with steps 1..N for each of the above scenarios {scenarios}. \  Write in a tabular form and check if they can be automatable.")

# response2 = chain.run("Generate 3 positive TCs with steps 1..N for each of the above scenarios {response1}. \
#                             Write in a tabular form and check if they can be automatable.")
#print(response2)


# you can give the context of the application and the language and framework to use here
# to generate the automation code
# role = "AutomationTester"
# response3 = chain.run("As a {role} Generate Automation Code for each of the above TCs in {response2}. \
#                            Language: Java, Framework: Selenium.")
# print(response3)

# role = "Code Reviewer"