from rag_mcp_server1 import rag_query, rag_status
from rag_mcp_server1 import *
import os

if __name__ == "__main__":

    # Check status before initialization
    print("Before init:", rag_status())

    # Trigger query (which will initialize if not yet done)
    question = "What are the requirements about providing customer support? Please provide detailed information about customer support requirements."
    response = rag_query(question=question)
    print("####################")
    print("Query Response:", response)
    print("####################")

    # Status after
    print("After init:", rag_status())


    feature = "Email confirmation to user"
    requirements = "User should receive an email confirmation for order confirmation"
    context = f"""the scope pertains to the E-Store product features for making Marvel Electronics and
                    Home Entertainment project live. It focuses on the company, the stakeholders and applications,
                    which allow for online sales, distribution and marketing of electronics."""

    response = generate_test_cases(feature=feature, requirements=requirements, context=context)
    print("####################")
    print("Query Response:", response)
    print("####################")


    #evalute the RAG system

    # question = "please evalute the RAG performance in the above context"
    # result = evaluate_rag(test_breadth=1, test_depth=1,launch_dashboard=True, dashboard_port=8501) 
    # print("Result is", result)