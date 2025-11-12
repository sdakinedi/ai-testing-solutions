from mcp.server.fastmcp import FastMCP
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma
from langchain_openai.chat_models import AzureChatOpenAI
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from langchain.retrievers.document_compressors import LLMChainFilter, CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# Create an MCP server
mcp = FastMCP("RAG Service")

# Global variables to store RAG components
vectordb = None
rag_chain = None
generator_llm = None
embeddings = None

def initialize_rag_system():
    """Initialize the RAG system with your existing configuration"""
    global vectordb, rag_chain, generator_llm, embeddings
    
    # Environment setup
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    # Set your LangChain API key and project 
    os.environ["LANGCHAIN_API_KEY"] = ""
    os.environ["LANGCHAIN_PROJECT"] = "ragasdemo11"
    
    api_key = os.getenv("DIAL_API_KEY")
    
    # Initialize embeddings
    embeddings = AzureOpenAIEmbeddings(
        api_key=api_key,
        api_version="2024-02-01",
        azure_endpoint="https://ai-proxy.lab.epam.com",
        azure_deployment="text-embedding-ada-002",
    )
    
    # Initialize generator LLM
    generator_llm = AzureChatOpenAI(
        api_version="2024-02-01",
        azure_endpoint="https://ai-proxy.lab.epam.com",
        api_key=api_key,
        temperature=0.0,
    )
    
    # Create new vector store in current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    persist_directory = os.path.join(current_dir, "vectorstore_new")
    
    try:
        # Create an empty vector store - it will be populated when PDFs are added
        vectordb = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
            collection_metadata={"hnsw:space": "cosine"}
        )
        print(f"✓ Created new vector store at: {persist_directory}")
        print("📝 Vector store is empty - use add_pdf_to_vectordb to populate it")
    except Exception as e:
        print(f"✗ Error creating vector store: {e}")
        return False
    
    # Setup retrieval pipeline
    similarity_retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    
    # Setup prompt template
    template = """You are an assistant for question-answering tasks. 
            Use the following pieces of retrieved context to answer the question. 
            If you don't know the answer, just say that you don't know. 
            Use two sentences maximum and keep the answer concise.
            Question: {question} 
            Context: {context} 
            Answer:
            """
    prompt = ChatPromptTemplate.from_template(template)
    
    # Create RAG chain
    rag_chain = (
        {"context": similarity_retriever, "question": RunnablePassthrough()}
        | prompt
        | generator_llm
        | StrOutputParser()
    )
    
    return True

def create_vectordb_from_documents(chunks):
    """Create vector database from document chunks"""
    global vectordb, embeddings
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    persist_directory = os.path.join(current_dir, "vectorstore_new")
    
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    return vectordb

@mcp.tool()
def query_documents(question: str) -> str:
    """Query the RAG system with a question about the loaded documents."""
    global rag_chain, vectordb
    
    if rag_chain is None:
        if not initialize_rag_system():
            return "Error: RAG system not initialized properly"
    
    # Check if vector store has any documents
    try:
        if vectordb._collection.count() == 0:
            return "Error: Vector store is empty. Please add some PDFs first using add_pdf_to_vectordb."
    except Exception as e:
        return f"Error checking vector store: {str(e)}"
    
    try:
        response = rag_chain.invoke(question)
        return response
    except Exception as e:
        return f"Error processing query: {str(e)}"

@mcp.tool()
def get_vectordb_info() -> str:
    """Get information about the loaded vector database."""
    global vectordb
    
    if vectordb is None:
        if not initialize_rag_system():
            return "Error: Vector database not loaded"
    
    try:
        count = vectordb._collection.count()
        return f"Vector database contains {count} document chunks"
    except Exception as e:
        return f"Error getting vectordb info: {str(e)}"

@mcp.tool()
def add_pdf_to_vectordb(pdf_path: str) -> str:
    """Add a new PDF to the vector database."""
    global vectordb, generator_llm, embeddings
    
    if not os.path.exists(pdf_path):
        return f"Error: PDF file not found at {pdf_path}"
    
    if vectordb is None:
        if not initialize_rag_system():
            return "Error: RAG system not initialized"
    
    try:
        # Load PDF
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        
        if not docs:
            return f"Error: No content found in PDF {pdf_path}"
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=300,
            length_function=len
        )
        chunks = text_splitter.split_documents(docs)
        
        # Check if this is the first document being added
        current_count = vectordb._collection.count()
        
        if current_count == 0:
            # If vector store is empty, recreate it with the new documents
            vectordb = create_vectordb_from_documents(chunks)
            print(f"✓ Created vector store with {len(chunks)} chunks from first PDF")
        else:
            # Add to existing vector store
            vectordb.add_documents(chunks)
        
        # Persist the changes
        vectordb.persist()
        
        new_count = vectordb._collection.count()
        return f"Successfully added {len(chunks)} chunks from {pdf_path}. Vector database now contains {new_count} total chunks."
        
    except Exception as e:
        return f"Error adding PDF: {str(e)}"

@mcp.tool()
def create_sample_vectordb() -> str:
    """Create a sample vector database with some test documents (for testing purposes)."""
    global vectordb, embeddings
    
    if vectordb is None:
        if not initialize_rag_system():
            return "Error: RAG system not initialized"
    
    try:
        # Create some sample documents for testing
        from langchain.schema import Document
        
        sample_docs = [
            Document(page_content="Python is a high-level programming language known for its simplicity and readability.", metadata={"source": "sample1"}),
            Document(page_content="Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.", metadata={"source": "sample2"}),
            Document(page_content="RAG (Retrieval-Augmented Generation) combines information retrieval with text generation for better AI responses.", metadata={"source": "sample3"}),
        ]
        
        # Create vector store with sample documents
        vectordb = create_vectordb_from_documents(sample_docs)
        vectordb.persist()
        
        count = vectordb._collection.count()
        return f"Successfully created sample vector database with {count} documents for testing purposes."
        
    except Exception as e:
        return f"Error creating sample vector database: {str(e)}"

@mcp.resource("rag://query/{question}")
def rag_resource(question: str) -> str:
    """Provide RAG query results as a resource."""
    return query_documents(question)

@mcp.prompt()
def rag_prompt(context: str, question: str) -> str:
    """Create a RAG-based prompt with context and question."""
    return f"""You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use two sentences maximum and keep the answer concise.

Context: {context}
Question: {question}

Answer:"""

# Initialize RAG system on startup
print(" Initializing RAG system...")
if initialize_rag_system():
    print("✓ RAG system initialized successfully")
else:
    print("✗ RAG system initialization failed")

# Run the server
if __name__ == "__main__":
    print("🌐 Starting MCP RAG Server...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print("📁 Vector store directory:", os.path.join(current_dir, "vectorstore_new"))
    print("🔧 Available tools: query_documents, get_vectordb_info, add_pdf_to_vectordb, create_sample_vectordb")
    print("📝 Note: Vector store starts empty - add PDFs using add_pdf_to_vectordb or create sample data with create_sample_vectordb")
    mcp.run()