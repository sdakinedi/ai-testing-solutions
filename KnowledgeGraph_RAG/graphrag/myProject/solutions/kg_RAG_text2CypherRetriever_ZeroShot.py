from neo4j_graphrag.retrievers import Text2CypherRetriever
from neo4j_graphrag.generation import GraphRAG

from common import get_driver, get_llm
from utilities import explore_graph_schema, get_entity_info
from queries import query_knowledge_graph


# Neo4j driver
driver = get_driver()
print("Connected to Neo4j database successfully!")


# LLM
print("\nInitializing OpenAI LLM for Text2Cypher & RAG...")
llm = get_llm() 


# Text2CypherRetriever configuration
TEXT2CYPHER_SYSTEM_PROMPT = """
You are an expert Neo4j Cypher generator for a Scrum knowledge graph.

The graph contains:
- Entities with labels such as: Role, Artifact, Event, Practice, Principle, Team, Chunk.
- Relationships such as: FACILITATES, MANAGES, PARTICIPATES_IN, CREATES,
  USES, OWNS, ATTENDS, RESPONSIBLE_FOR, FROM_CHUNK.

General guidelines:
- Use concise Cypher.
- When the user asks about roles (e.g., Scrum Master, Product Owner),
  query Role nodes and their relationships.
- When the user asks about Scrum events (Sprint Planning, Daily Scrum,
  Sprint Review, Sprint Retrospective), query Event nodes and connect
  them to related entities (roles, artifacts, practices).
- If the question is broad, return multiple nodes and relevant relationships.

Return results that are useful as textual context for answering the question.
Do NOT modify the database. Use read-only queries (MATCH, OPTIONAL MATCH, RETURN).
"""


# Text2CypherRetriever
print("\nInitializing Text2CypherRetriever...")

text2cypher_retriever = Text2CypherRetriever(
    driver=driver,
    llm=llm,
    #custom_prompt=TEXT2CYPHER_SYSTEM_PROMPT,
)

print("Text2CypherRetriever initialized successfully!")


# GraphRAG pipeline using Text2CypherRetriever
print("\nCreating GraphRAG pipeline (Text2Cypher)...")

rag = GraphRAG(
    retriever=text2cypher_retriever,
    llm=llm,
)

print("GraphRAG (Text2Cypher) pipeline ready!")


# Main
if __name__ == "__main__":
    print("\n======================================================")
    print("KNOWLEDGE GRAPH RAG SYSTEM READY (Text2CypherRetriever)")
    print("\n======================================================")

    # Example queries – reuse the shared query helper
    print("\n\nRunning example queries with Text2CypherRetriever...\n")

    query_knowledge_graph(rag, "What is the role of Scrum Master?", None)
    query_knowledge_graph(rag, "How does the Product Owner manage the Product Backlog?", None)
    query_knowledge_graph(rag, "What happens during Sprint Planning?", None)
   
    # graph schema exploration
    print("\n\n")
    explore_graph_schema(driver)

    # specific entity info
    print("\n\n")
    get_entity_info(driver, "Scrum Master")

    # Interactive mode
    print("\n======================================================")
    print("INTERACTIVE MODE (Text2CypherRetriever)")
    print("======================================================")
    print("\nYou can now ask questions. Type 'exit' to quit.\n")

    try:
        while True:
            user_question = input("Your question: ")
            if user_question.lower() in ["exit", "quit", "q"]:
                print("\nClosing connection...")
                driver.close()
                print("Goodbye!")
                break

            if user_question.strip():
                # top_k is not really meaningful for Text2Cypher, so we just use the default
                query_knowledge_graph(rag, user_question, top_k=None)
            else:
                print("Please enter a valid question.")
    finally:
        driver.close()
        print("Driver closed.")
