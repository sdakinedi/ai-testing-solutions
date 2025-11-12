import os
import openai
from dotenv import load_dotenv, find_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.agents import AgentType, initialize_agent
from langchain.tools import Tool
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from trulens.core import Feedback, TruSession
from trulens.apps.langchain import TruChain
from trulens.providers.openai import OpenAI
import numpy as np

# Load environment variables
_ = load_dotenv(find_dotenv()) 
openai.api_key = os.environ['OPENAI_API_KEY']

# Define API Keys
azure_api_key = "b13467a9fde7430dae79d77dcbf023dd"
azure_endpoint = "https://ai-proxy.lab.epam.com"

# Load and process PDF documents
pdf_loader = PyPDFLoader("C:\\Users\\Satyaprasad_Dakinedi\\Desktop\\guideToScrum.pdf")
documents = pdf_loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
chunks = text_splitter.split_documents(documents)

# Create vector store
persist_directory = "C:\\Users\\Satyaprasad_Dakinedi\\Desktop\\LangChain_demos\\langchaindemos\\vectorstore_16"
vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=AzureOpenAIEmbeddings(
        openai_api_version="2023-07-01-preview",
        api_key=azure_api_key,
        azure_endpoint=azure_endpoint,
        azure_deployment="text-embedding-ada-002"
    ),
    persist_directory=persist_directory
)
retriever = vectordb.as_retriever()

# Define LLM
llm = AzureChatOpenAI(
    openai_api_version="2023-07-01-preview",
    api_key=azure_api_key,
    azure_endpoint=azure_endpoint,
    azure_deployment="gpt-35-turbo"
)

# Define RAG retrieval agent
def retrieve_info(query):
    return qa_chain.run(query)

retrieval_agent = initialize_agent(
    tools=[Tool(
        name="KnowledgeBaseRetriever",
        func=retrieve_info,
        description="Retrieve information from the knowledge base using a question."
    )],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Define QA chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Define agent tools
def query_rag(query):
    return retrieval_agent.run(query)

rag_tool = Tool(
    name="RAGRetrieverAgent",
    func=query_rag,
    description="Use an agent to retrieve relevant information from RAG-based knowledge sources."
)

# Initialize memory
memory = ConversationBufferMemory(memory_key="chat_history")

# Create main agent with RAG retrieval agent as a tool
agent_executor = initialize_agent(
    tools=[rag_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=memory
)

# TruLens integration
session = TruSession()
feedback_provider = OpenAI()

context = TruChain.select_context(qa_chain)

f_groundedness = (
    Feedback(feedback_provider.groundedness_measure_with_cot_reasons, name="Groundedness")
    .on(context.collect())
    .on_output()
)
f_answer_relevance = (
    Feedback(feedback_provider.relevance_with_cot_reasons, name="Answer Relevance")
    .on_input_output()
)
f_context_relevance = (
    Feedback(feedback_provider.context_relevance_with_cot_reasons, name="Context Relevance")
    .on_input()
    .on(context)
    .aggregate(np.mean)
)

tru_recorder = TruChain(
    agent_executor,
    app_name="QABot_Agent",
    app_version="V3.1",
    feedbacks=[f_answer_relevance, f_context_relevance, f_groundedness]
)

# Run an example query
with tru_recorder as recording:
    response = agent_executor.run("What is the role of a Scrum Master?")
    print(response)
