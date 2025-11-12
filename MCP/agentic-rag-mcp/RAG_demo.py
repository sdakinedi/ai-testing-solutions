import os
#import openai
import sys
#sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai.chat_models import AzureChatOpenAI
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from langchain.retrievers.document_compressors import LLMChainFilter, CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
#from langchain.embeddings.openai import OpenAIEmbeddings


# _ = load_dotenv(find_dotenv()) # read local .env file

# openai.api_key  = os.environ['OPENAI_API_KEY']

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "<YOUR_LANGCHAIN_API_KEY>" 
os.environ["LANGCHAIN_PROJECT"] = "ragasdemo11"

# Load PDF
loaders = [
    PyPDFLoader("C:\\Users\\Satyaprasad_Dakinedi\\Desktop\\guideToScrum.pdf"),
   
]
docs = []
for loader in loaders:
    docs.extend(loader.load())

documents = docs

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 300,
    length_function=len
)

chunks = text_splitter.split_documents(documents)

print("chunks length:", len(chunks))


# Time to create the embeddings

api_key = os.getenv("DIAL_API_KEY")


embeddings = AzureOpenAIEmbeddings(
    api_key=api_key,
    api_version = "2024-02-01",
    azure_endpoint="https://ai-proxy.lab.epam.com",
    #azure_deployment="text-embedding-ada-002",    
)

# generator with openai models
generator_llm = AzureChatOpenAI(
    api_version = "2024-02-01",
    azure_endpoint = "https://ai-proxy.lab.epam.com",
    api_key=api_key,
    temperature=0.0,
)


critic_llm = AzureChatOpenAI(
    api_version = "2024-02-01",
    api_key=api_key,
    azure_endpoint="https://ai-proxy.lab.epam.com",
    azure_deployment="gpt-4",
)


# Define LLM
llm = AzureChatOpenAI(
    api_version="2024-02-01",
    api_key=api_key,
    azure_endpoint="https://ai-proxy.lab.epam.com",
    azure_deployment="gpt-35-turbo",
    
)


####### Create Vector Store
persist_directory = "C:\\Users\\Satyaprasad_Dakinedi\\Desktop\\LangChain_demos\\langchaindemos\\vectorstore_27June2025_6"
print("Creating Chroma vector store. at the location..", persist_directory)
vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=persist_directory,
)
print("Vector store created with", vectordb._collection.count(), "entries.")

# # load existing vector store if it exists
# if not vectordb._collection.count():
#     print("No existing vector store found, creating a new one.")
# else:
#     print("Loading existing vector store...")
#     vectordb = Chroma(
#         persist_directory=persist_directory,
#         embedding_function=embeddings
#     )

vectordb.persist()

print("Total entries in the vectordb is:", vectordb._collection.count())


# Chained retrieval pipeline
# The process is as follows:
# Similarity Retrieval (Basic Cosine) -> 
#   Compression Filter(To filter out irrelevant chunks) -> LLMChainFilter ->
#       Reranking -> Helps in reranking the retrieved chunks based on relevance -> (BGE or BERT ranker
#         Final Retrieval

retriever = vectordb.retriever()

similarity_retriever = vectordb.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}  # Retrieve top 5 similar chunks
)

filter = LLMChainFilter.from_llm(generator_llm)
compressed_retriever = ContextualCompressionRetriever(
    base_compressor=filter,
    base_retriever=similarity_retriever,
)   

reranker = HuggingFaceCrossEncoder(
    model_name="BAAI/bge-reranker-large",
)
reranker_compressor = CrossEncoderReranker(model=reranker, top_n=3)
# Retriever 3: Uses a reranker to rerank the retrieved results from the previous retriever
final_retriever = ContextualCompressionRetriever(
    base_compressor=reranker_compressor,
    base_retriever=compressed_retriever,
)


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
# use similarity_retriever if you want to use the basic similarity retriever
# use final_retriever if you want to use the chained retrieval pipeline with compression and reranking
rag_chain = (
    {"context": retriever,  "question": RunnablePassthrough()} 
    | prompt 
    | generator_llm
    | StrOutputParser() 
)


query = "What is the role of the Scrum Master in a Scrum team?"
response = rag_chain.invoke(query)

print("Query:", query)
print("Response:", response)

query = "Who is the prime minister of India?"
response = rag_chain.invoke(query)

print("Query:", query)
print("Response:", response)