from fastapi import FastAPI
from pydantic import BaseModel
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from smolagent import SmolAgent, Tool
from langchain.chat_models import ChatOpenAI

from mcp.server.fastmcp import FastMCP

# Step 1: Load documents and create vector DB
pdf_dir = os.path.join(os.path.dirname(__file__), "PDFs")
all_docs = []
for filename in os.listdir(pdf_dir):
    if filename.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(pdf_dir, filename))
        all_docs.extend(loader.load())

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(all_docs)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")

# Step 2: Define Retriever Tool
def rag_tool_fn(question: str) -> str:
    docs = vectordb.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    return f"Relevant context:\n{context}"

retriever_tool = Tool(
    name="RAG Retriever",
    description="Retrieves relevant document context for a question",
    func=rag_tool_fn
)

# Step 3: Define SmolAgent
llm = ChatOpenAI(temperature=0)

agent = SmolAgent(
    tools=[retriever_tool],
    llm=llm,
    system_prompt="You are an expert answering questions using tools. Always call the RAG Retriever tool before answering."
)

# Step 4: Setup FastAPI + MCP
app = FastAPI()
mcp = FastMCP(name="SmolAgenticRAG", version="1.0")

class Query(BaseModel):
    question: str

@app.post("/query")
async def query_agent(q: Query):
    result = agent.run(q.question)
    return {"answer": result}
