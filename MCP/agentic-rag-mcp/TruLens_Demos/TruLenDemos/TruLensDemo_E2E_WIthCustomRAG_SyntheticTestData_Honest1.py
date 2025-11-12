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
from trulens.benchmark.generate.generate_test_set import GenerateTestSet

_ = load_dotenv(find_dotenv()) # read local .env file


from trulens.apps.langchain import TruChain
from trulens.core import TruSession

session = TruSession()
session.reset_database()

openai.api_key  = os.environ['OPENAI_API_KEY']

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "<YOUR_LANGCHAIN_API_KEY>" 
os.environ["LANGCHAIN_PROJECT"] = "truelensdemo1"


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
api_key = "d99c6d76e9b2438ea40741d7c1f7c357"


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
from datetime import datetime

script_directory = os.path.dirname(os.path.abspath(__file__))

current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

persist_directory = os.path.join(script_directory, f"vectorstore_{current_timestamp}")

if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)
    print(f"Vector store directory created: {persist_directory}")
else:
    print(f"Vector store directory already exists: {persist_directory}")

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
template = """You are an AI assistant for question-answering tasks. 
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

#response = rag_chain.invoke("What is the role of Scrum master")
#print(response)



##############

test = GenerateTestSet(app_callable=rag_chain.invoke)
test_set = test.generate_test_set(test_breadth=3, test_depth=2)

print("############## test_set is:", test_set)
#We can also provide a list of examples to help guide our app to the types of questions we want to test.

# examples = [
#     "What is the role of Scrum Master?",
#  #   "How much information can be stored in short term memory?",
# ]

# fewshot_test_set = test.generate_test_set(
#     test_breadth=3, test_depth=4, examples=examples
# )
# fewshot_test_set


import numpy as np
from trulens.core import Feedback
from trulens.providers.openai import OpenAI

# Initialize provider class
provider = OpenAI()

# select context to be used in feedback. the location of context is app specific.
context = TruChain.select_context(rag_chain)

from trulens_eval import Feedback
from trulens.feedback.embeddings import Embeddings


def calculate_compression_ratio(context: str, response: str) -> float:
    """
    Compute compression ratio as: len(response) / len(context).
    Lower compression ratio means higher summarization effectiveness.
    """
    if len(context) == 0:  # Handle edge cases
        return 0
    compression_ratio = len(response) / len(context)
    # Invert the ratio to derive a "higher is better" score for summaries (<1 is good)
    return 1 - compression_ratio


def calculate_semantic_similarity(context: str, response: str, embeddings: Embeddings) -> float:
    """
    Use embeddings (e.g., cosine similarity) to measure how semantically aligned the response is with the context.
    """
    # Generate embeddings for context and response
    context_emb = embeddings.embed(context)
    response_emb = embeddings.embed(response)
    
    # Calculate cosine similarity
    similarity = embeddings.cosine_distance(context_emb, response_emb)
    return similarity


# Summarization Capability Feedback Function
def summarization_quality(context_collected: list, response: str, embeddings: Embeddings) -> float:
    """
    Calculate the Summarization Capability score using collected context and model response.
    Combines compression ratio and semantic similarity.

    Args:
        context_collected: Retrieved context chunks (list of context strings).
        response: Model output from the RAG pipeline.
        embeddings: Embedding model instance to calculate cosine similarity.

    Returns:
        A score between 0 and 1 indicating summarization capability.
    """

    print(f"@@@@@@@@@@@@@@@@@@@@ Collected Context:", context_collected)
    print(f"@@@@@@@@@@@@@@@@@@@@ Response:", response)

    # Merge context into a single string for comparison
    context = " ".join(context_collected)

    # Sanity-check: Ensure both context and response are non-empty
    if not context or not response:
        print("Context or response is missing.")
        return 0.0

    print(f"@@@@@@@@@@@@@@@@@@@@ Context: {context}")

    # 1. Calculate compression ratio
    compression_score = calculate_compression_ratio(context, response)

    print(f"@@@@@@@@@@@@@@@@@@@@ Compression Score: {compression_score}")

    # 2. Compute semantic similarity
    semantic_similarity_score = calculate_semantic_similarity(context, response, embeddings)

    print(f"@@@@@@@@@@@@@@@@@@@@ Semantic Similarity Score: {semantic_similarity_score}")

    # Combine metrics
    combined_score = 0.65 * compression_score + 0.5 * semantic_similarity_score
    print(f"@@@@@@@@@@@@@@@@@@@@@@ Compression Score: {compression_score}, Semantic Similarity: {semantic_similarity_score}, Combined Score: {combined_score}")

    return combined_score

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

# Define the Feedback function for summarization evaluation
f_summarization_quality = (
    Feedback(
        lambda context_collected, response: summarization_quality(context_collected, response, embed),
        name="Summarization Capability",         
    )
    .on(context.collect())  # Attach context collector
    .on_output()  # Attach model response collector
)

# Define the Feedback function for summarization evaluation
f_summarization_quality1 = (
    Feedback(
        provider.summarization_with_cot_reasons, name="Summarization Capability"     
    )
    .on(context.collect())  # Attach context collector
    .on_output()  # Attach model response collector
)



from trulens.feedback.embeddings import Embeddings
embed = Embeddings(embed_model=embeddings)
f_embed_dist = Feedback(embed.cosine_distance).on_input().on(context)


honest_feedbacks = [
    f_answer_relevance,
    f_context_relevance,
    f_groundedness,
    f_summarization_quality    
]


# Instrument chain for logging with TruLens
tru_recorder = TruChain(
    rag_chain,
    app_name="QABot_SyntheticTestData",
    app_version="V3.2",
    feedbacks=honest_feedbacks,
    #feedbacks=[f_summarization_quality]
)

from trulens.core import TruSession
from trulens.dashboard import run_dashboard

session = TruSession()
run_dashboard(session)
session.reset_database()

#Evaluate the application with our generated test set
with tru_recorder as recording:
    for category in test_set:
        recording.record_metadata = dict(prompt_category=category)
        test_prompts = test_set[category]
        for test_prompt in test_prompts:
            llm_response = rag_chain.invoke(test_prompt)
            print(f"Test Prompt: {test_prompt}, Response: {llm_response}") 