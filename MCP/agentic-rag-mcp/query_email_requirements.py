from rag_mcp_server1 import rag_query

def main():
    print("Querying email notification requirements...")
    print("=" * 60)
    
    # Query 1: General email notifications
    question1 = "What are the requirements for email notifications and email functionality?"
    response1 = rag_query(question=question1)
    print("Email Notification Requirements:")
    print("-" * 40)
    print(response1)
    print()
    
    # Query 2: Email confirmation specifically
    question2 = "What are the requirements related to email confirmation?"
    response2 = rag_query(question=question2)
    print("Email Confirmation Requirements:")
    print("-" * 40)
    print(response2)
    print()
    
    # Query 3: Email triggers and events
    question3 = "What events or actions trigger email notifications in the system?"
    response3 = rag_query(question=question3)
    print("Email Triggers and Events:")
    print("-" * 40)
    print(response3)
    print()

if __name__ == "__main__":
    main()
