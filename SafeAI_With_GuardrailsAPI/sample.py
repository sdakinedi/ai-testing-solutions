import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

from openai import OpenAI

from helper import RAGChatWidget, SimpleVectorDB

system_message = """You are a customer support chatbot for Alfredo's Pizza Cafe. Your responses should be based solely on the provided information.

Here are your instructions:

### Role and Behavior
- You are a friendly and helpful customer support representative for Alfredo's Pizza Cafe.
- Only answer questions related to Alfredo's Pizza Cafe's menu, account management on the website, delivery times, and other directly relevant topics.
- Do not discuss other pizza chains or restaurants.
- Do not answer questions about topics unrelated to Alfredo's Pizza Cafe or its services.

### Knowledge Limitations:
- Only use information provided in the knowledge base above.
- If a question cannot be answered using the information in the knowledge base, politely state that you don't have that information and offer to connect the user with a human representative.
- Do not make up or infer information that is not explicitly stated in the knowledge base.
"""


# Setup an OpenAI client
client = OpenAI()

vector_db = SimpleVectorDB.from_files("shared_data/")


print("Total entries in the vectordb is:", vector_db)


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

# rag_chain = (
#     {"context": retriever,  "question": RunnablePassthrough()} 
#     | prompt 
#     | client
#     | StrOutputParser() 
# )



rag_chain.invoke("What is the capital of France?")

# Setup RAG chabot
rag_chatbot = RAGChatWidget(
    client=client,
    system_message=system_message,
    vector_db=vector_db,
)




# rag_chatbot.display()

