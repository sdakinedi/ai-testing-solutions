from rag_mcp_server1 import rag_query, rag_status
from rag_mcp_server1 import *
import os

if __name__ == "__main__":
    # Absolute paths (update if needed)
    # base_dir = os.path.dirname(__file__)
    # pdf_path = os.path.join(base_dir, "PDFs", "guideToScrum.pdf")
    # persist_dir = os.path.join(base_dir, "rag_vector_store1")

    # Check status before initialization
    print("Before init:", rag_status())

    # Trigger query (which will initialize if not yet done)
    question = "What are the requirements related to email confirmation?"
    response = rag_query(question=question)
    print("Query Response:", response)

    # Status after
    print("After init:", rag_status())

    # evalute the RAG system
    question = "please evalute the RAG performance in the above context"
    result = evaluate_rag(test_breadth=1, test_depth=1,launch_dashboard=True, dashboard_port=8501) 
    print("Result is", result)