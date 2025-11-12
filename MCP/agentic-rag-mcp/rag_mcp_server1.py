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
import numpy as np

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
  azure_endpoint = "https://ai-proxy.lab.epam.com",
azure_deployment = "gpt-35-turbo"
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
            azure_deployment="gpt-35-turbo",
        )
        
        # Set up retrieval pipeline
        similarity_retriever = self.vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        # Use simple similarity retriever for now
        #retriever = similarity_retriever
        retriever = self.vectordb.as_retriever()
        
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
        print("===> RAG system initialized successfully!")

# Global RAG system instance
rag_system = RAGSystem()

# TruLens imports - moved after RAG system definition
try:
    from trulens.core import TruSession, Feedback
    from trulens.providers.openai import AzureOpenAI
    from trulens.apps.langchain import TruChain
    from trulens.benchmark.generate.generate_test_set import GenerateTestSet
    
    TRULENS_AVAILABLE = True
    print("========>> TruLens imports successful!")
    
    # Initialize TruLens components
    provider = AzureOpenAI(
                deployment_name="gpt-35-turbo",
                api_key=api_key,
                api_version="2024-02-01", 
                azure_endpoint="https://ai-proxy.lab.epam.com")

   
    # Define feedback functions
    def setup_feedback_functions():
        context = TruChain.select_context(rag_system.rag_chain)
        
        f_groundedness = (
            Feedback(provider.groundedness_measure_with_cot_reasons, name="Groundedness")
            .on(context.collect())
            .on_output()
        )
        
        f_answer_relevance = (
            Feedback(provider.relevance_with_cot_reasons, name="Answer Relevance")
            .on_input_output()
        )
        
        f_context_relevance = (
            Feedback(provider.context_relevance_with_cot_reasons, name="Context Relevance")
            .on_input()
            .on(context)
            .aggregate(np.mean)
        )
        
        return f_groundedness, f_answer_relevance, f_context_relevance
    
except ImportError as e:
    TRULENS_AVAILABLE = False
    print(f" TruLens not available: {e}")
    print("RAG functionality will work, but evaluation features will be disabled.")

print(" ====> All other imports successful!")




@mcp.tool()
def rag_query(
    question: str,
    pdf_path: str = "C:\\Users\\Satyaprasad_Dakinedi\\Desktop\\AI_Solutions_Implementations\\MCP\\agentic-rag-mcp\\PDFs\\SRS.pdf",
    persist_directory: str = "C:\\Users\\Satyaprasad_Dakinedi\\Desktop\\LangChain_demos\\langchaindemos\\vectorstore_02"
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

@mcp.tool()
def evaluate_rag_chain(test_breadth: int = 2, test_depth: int = 3) -> str:
    """
    Evaluate the RAG chain using TruLens with synthetic test data.
    
    Args:
        test_breadth: Number of test categories
        test_depth: Number of prompts per category
    
    Returns:
        Summary string about evaluation
    """
    if not TRULENS_AVAILABLE:
        return "TruLens evaluation not available due to import issues. Please check TruLens installation."
    
    try:
        if not rag_system.is_initialized:
            return "RAG system not initialized. Run a query first."

        # Set up feedback functions
        f_groundedness, f_answer_relevance, f_context_relevance = setup_feedback_functions()

        # Generate synthetic test set
        test_generator = GenerateTestSet(app_callable=rag_system.rag_chain.invoke)
        test_set = test_generator.generate_test_set(test_breadth=test_breadth, test_depth=test_depth)

        print("========> Generated test set:", test_set)


        #session = TruSession()
    

        # Set up TruChain
        tru_recorder = TruChain(
            rag_system.rag_chain,
            app_name="RAG Evaluation",
            app_version="V1",
            feedbacks=[f_answer_relevance, f_context_relevance, f_groundedness],
        )

        from trulens.dashboard import run_dashboard
        #session = TruSession()

        session = TruSession()
        #session.reset_database()
        run_dashboard(session)

        # run_dashboard = True  # Set to False to skip dashboard
        # if run_dashboard:
        #     tru_recorder.run_dashboard(session)
        # #session = TruSession()
        # #session.reset_database()  # Optional: Reset before each eval

        # Evaluate each test
        try:
             with tru_recorder as recording:
                for category in test_set:
                    recording.record_metadata = dict(prompt_category=category)
                    test_prompts = test_set[category]
                    for test_prompt in test_prompts:
                        llm_response = rag_system.rag_chain.invoke(test_prompt)
                        print("LLM Response:", llm_response)
        finally:
             print("Recording complete. Saving results to database...")
        import time
        time.sleep(5)  # Ensure feedbacks are persisted

        records = session.get_records_and_feedback()
        print("Recorded evaluations:", records)

        return f"Evaluation completed with {len(test_set)} categories and feedbacks logged. Launch dashboard via `trulens.dashboard.run_dashboard()`."

    except Exception as e:
        return f"Error during evaluation: {str(e)} {str(e.__traceback__)} {e.__cause__}"

if __name__ == "__main__":
    # Run the MCP server
    print("====> Starting RAG MCP Server...")
    mcp.run()