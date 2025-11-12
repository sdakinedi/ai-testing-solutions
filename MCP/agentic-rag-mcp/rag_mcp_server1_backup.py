import os
import asyncio
from typing import Any, Dict, List, Optional
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv, find_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai.chat_models import AzureChatOpenAI
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from langchain.retrievers.document_compressors import LLMChainFilter, CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from openai import AzureOpenAI
from datetime import datetime

# Load environment variables
load_dotenv(find_dotenv())

# Set up LangChain tracing (optional)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
# Set your LangChain API key and project
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["LANGCHAIN_PROJECT"] = "ragasdemo11"

api_key = os.getenv("DIAL_API_KEY")
client = AzureOpenAI(
  api_key = api_key,
  api_version = "2024-02-01",
  azure_endpoint = "https://ai-proxy.lab.epam.com"
)
MODEL="gpt-35-turbo"

# Initialize FastMCP
mcp = FastMCP("RAG Server")

class RAGSystem:
    def __init__(self):
        self.vectordb = None
        self.rag_chain = None
        self.is_initialized = False
        
    def initialize(self, pdf_path: str, persist_directory: str):
        """Initialize the RAG system with PDF and vector store"""
        if self.is_initialized:
            return
            
        # Get API key
        api_key = os.getenv("DIAL_API_KEY")
        
        if not api_key:
            raise ValueError("DIAL_API_KEY environment variable not set")
        
        # Load PDF
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=300,
            length_function=len
        )
        chunks = text_splitter.split_documents(docs)
        
        # Create embeddings
        embeddings = AzureOpenAIEmbeddings(
            api_key=api_key, # type: ignore
            api_version="2024-02-01",
            azure_endpoint="https://ai-proxy.lab.epam.com",
            azure_deployment="text-embedding-ada-002",
        )
        
        # Create or load vector store
        try:
            # Try to load existing vector store
            self.vectordb = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings
            )
            if self.vectordb._collection.count() == 0:
                raise Exception("Empty vector store")
            print(f"Loaded existing vector store with {self.vectordb._collection.count()} entries")
        except:
            # Create new vector store
            print("Creating new vector store...")
            self.vectordb = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=persist_directory,
                collection_metadata={"hnsw:space": "cosine"}
            )
            self.vectordb.persist()
            print(f"Created vector store with {self.vectordb._collection.count()} entries")
        
        # Initialize LLM
        generator_llm = AzureChatOpenAI(
            api_version="2024-02-01",
            azure_endpoint="https://ai-proxy.lab.epam.com",
            api_key=api_key,
            azure_deployment="gpt-4o",
            temperature=0.0,
        )
        
        # Set up retrieval pipeline
        similarity_retriever = self.vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        # Optional: Set up advanced retrieval with compression and reranking
        # Uncomment the following lines if you want to use the advanced pipeline
        """
        filter = LLMChainFilter.from_llm(generator_llm)
        compressed_retriever = ContextualCompressionRetriever(
            base_compressor=filter,
            base_retriever=similarity_retriever,
        )
        
        reranker = HuggingFaceCrossEncoder(
            model_name="BAAI/bge-reranker-large",
        )
        reranker_compressor = CrossEncoderReranker(model=reranker, top_n=3)
        final_retriever = ContextualCompressionRetriever(
            base_compressor=reranker_compressor,
            base_retriever=compressed_retriever,
        )
        retriever = final_retriever
        """
        
        # Use simple similarity retriever for now
        retriever = similarity_retriever
        
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
        
        # Setup RAG pipeline
        self.rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | generator_llm
            | StrOutputParser()
        )
        
        self.is_initialized = True
        print("RAG system initialized successfully!")

# Global RAG system instance
rag_system = RAGSystem()

@mcp.tool()
def rag_query(
    question: str,
    #pdf_path: str = "C:\\Users\\Satyaprasad_Dakinedi\\Desktop\\guideToScrum.pdf"
    pdf_path: str = "C:\\Users\\Satyaprasad_Dakinedi\\Desktop\\AI_Solutions_Implementations\\MCP\\agentic-rag-mcp\\PDFs\\SRS.pdf",
    persist_directory: str = "C:\\Users\\Satyaprasad_Dakinedi\\Desktop\\LangChain_demos\\langchaindemos\\vectorstore_23June2025_5"
) -> str:
    """
    Perform a RAG (Retrieval-Augmented Generation) query against a PDF document.
    
    Args:
        question: The question to ask about the document
        pdf_path: Path to the PDF file to query (optional, uses default if not provided)
        persist_directory: Directory to store/load the vector database (optional, uses default if not provided)
    
    Returns:
        Answer to the question based on the PDF content
    """
    try:
        # Initialize the RAG system if not already done
        if not rag_system.is_initialized:
            rag_system.initialize(pdf_path, persist_directory)
        
        # Perform the query
        response = rag_system.rag_chain.invoke(question)
        
        return response
        
    except Exception as e:
        return f"Error processing RAG query: {str(e)}"

@mcp.tool()
def rag_status() -> str:
    """
    Check the status of the RAG system.
    
    Returns:
        Status information about the RAG system
    """
    if rag_system.is_initialized:
        count = rag_system.vectordb._collection.count() if rag_system.vectordb else 0
        return f"RAG system is initialized with {count} document chunks in vector store"
    else:
        return "RAG system is not initialized"
    
TEST_CASE_GENERATION_PROMPT = """
                            Generate test cases based on the following:

                            Feature: {feature}
                            Requirements: {requirements}
                            Context: {context}

                            Format:
                            1. Test Case ID
                            2. Description
                            3. Preconditions
                            4. Steps
                            5. Expected Results

                            Provide at least one happy path, one edge case, and one error path. 
                            """

@mcp.tool()
def generate_test_cases(feature: str, requirements: str, context: str = "") -> str:
    """Generate test cases for a given feature based on requirements"""
    try:
        formatted_prompt = TEST_CASE_GENERATION_PROMPT.format(
            feature=feature,
            requirements=requirements,
            context=context
        )

        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": formatted_prompt}],
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating test cases: {str(e)}"
    

TEST_AUTOMATION_PROMPT = """
                        Write automated test code for the following test case:
                        Test Case:
                        {test_case}
                        Test Framework: {framework}
                        Programming Language: {language}
                        Additional requirements:
                        {requirements}
                        Please include appropriate assertions, setup, and teardown logic.
                        Follow Best recommended practices for the specified framework and language.
                        Write only the code that is executable, do not include any explanations or comments.
                        """

@mcp.tool()
def write_test_automation_code(test_case: str, framework: str = "pytest", language: str = "python", requirements: str = "") -> str:
    """Write test automation code based on a test case description. Write only the code. Do not write anything else."""
    try:
        formatted_prompt = TEST_AUTOMATION_PROMPT.format(
            test_case=test_case,
            framework=framework,
            language=language,
            requirements=requirements
        )  
            
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": formatted_prompt}],
        )
            
        result = response.choices[0].message.content
       
        
        # Extract code from markdown if present
        code = result
        if "```" in result:
            code_blocks = result.split("```")
            for i in range(1, len(code_blocks), 2):
                if code_blocks[i].startswith("python"):
                    code = code_blocks[i][6:].strip()
                    break
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_filename = f"test_{timestamp}.py"

        # Save the code to a file in the current directory
        with open(test_filename, "w") as f:
            f.write(code)
        
        return f"Test code written to {test_filename}:\n\n{code}"
    except Exception as e:
        return f"Error writing test automation: {str(e)}"


if __name__ == "__main__":
    # Run the MCP server
    print("Starting RAG MCP Server...")
    mcp.run()


