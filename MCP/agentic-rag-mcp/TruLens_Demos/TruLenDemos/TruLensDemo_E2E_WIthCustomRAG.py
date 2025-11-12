import os
import openai
import sys
#sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from datasets import Dataset
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai.chat_models import AzureChatOpenAI
from langchain_openai.embeddings import AzureOpenAIEmbeddings

from ragas.testset import TestsetGenerator
from ragas import evaluate
from trulens.core import Feedback

_ = load_dotenv(find_dotenv()) # read local .env file


from trulens.apps.langchain import TruChain
from trulens.core import TruSession

session = TruSession()
session.reset_database()

openai.api_key  = os.environ['OPENAI_API_KEY']

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "<your_langchain_api_key>" 
os.environ["LANGCHAIN_PROJECT"] = "ragasdemo10"


# Load PDF
loaders = [
    # Duplicate documents on purpose - messy data
    PyPDFLoader("C:\\Users\\Satyaprasad_Dakinedi\\Desktop\\guideToScrum.pdf"),
   
]
docs = []
for loader in loaders:
    docs.extend(loader.load())

documents = docs

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 150,
    length_function=len
)

chunks = text_splitter.split_documents(documents)

print("############ chunks is:", len(chunks))

# Time to create the embeddings
api_key = "b13467a9fde7430dae79d77dcbf023dd"


embeddings = AzureOpenAIEmbeddings(
    openai_api_version="2023-07-01-preview",
    api_key=api_key,
    azure_endpoint="https://ai-proxy.lab.epam.com",
    azure_deployment="text-embedding-ada-002",
    #model=azure_configs["embedding_name"],
)

# Define LLM
llm_1 = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
llm = AzureChatOpenAI(
    openai_api_version="2023-07-01-preview",
    api_key=api_key,
    azure_endpoint="https://ai-proxy.lab.epam.com",
    azure_deployment="gpt-35-turbo",
    #model=azure_configs["model_name"],
    #validate_base_url=False,
)

# from trulens_eval import Tru
# from trulens_eval.tru_custom_app import instrument
# tru = Tru()
# tru.reset_database()

persist_directory = "C:\\Users\\Satyaprasad_Dakinedi\\Desktop\\LangChain_demos\\langchaindemos\\vectorstore_13"

####### Create Vector Store

#from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=OpenAIEmbeddings(),
    persist_directory=persist_directory
)

print("Total entries in the vectordb is:", vectordb._collection.count())

retriever = vectordb.as_retriever()

##########################
####### Prepare the Prompt Template ########
##########################


# Define prompt template
template = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use two sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""

prompt = ChatPromptTemplate.from_template(template)

##########################
# Setup RAG pipeline
##########################

rag_chain = (
    {"context": retriever,  "question": RunnablePassthrough()} 
    | prompt 
    | llm_1
    | StrOutputParser() 
)

response = rag_chain.invoke("What is the role of Scrum master")
print(response)


import numpy as np
from trulens.core import Feedback
from trulens.providers.openai import OpenAI

# Initialize provider class
provider = OpenAI()

# select context to be used in feedback. the location of context is app specific.
context = TruChain.select_context(rag_chain)

# Define a groundedness feedback function
f_groundedness = (
    Feedback(
        provider.groundedness_measure_with_cot_reasons, name="Groundedness"
    )
    .on(context.collect())  # collect context chunks into a list
    .on_output()
)

# Question/answer relevance between overall question and answer.
f_answer_relevance = (
    Feedback(
        provider.relevance_with_cot_reasons, name="Answer Relevance"
    )
    .on_input_output()
)




# Context relevance between question and each context chunk.
f_context_relevance = (
    Feedback(
        provider.context_relevance_with_cot_reasons, name="Context Relevance"
    )
    .on_input()
    .on(context)
    .aggregate(np.mean)
)

# Instrument chain for logging with TruLens
tru_recorder = TruChain(
    rag_chain,
    app_name="ChatApplication",
    app_version="V1",
    feedbacks=[f_answer_relevance, f_context_relevance, f_groundedness],
)

with tru_recorder as recording:
    llm_response = rag_chain.invoke("What is the role of Scrum Master")

print(llm_response)

session.get_leaderboard()

from trulens.dashboard import run_dashboard

run_dashboard(session)