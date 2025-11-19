from neo4j_graphrag.retrievers import Text2CypherRetriever
from neo4j_graphrag.generation import GraphRAG

from common import get_driver, get_llm
from utilities import explore_graph_schema, get_entity_info
from queries import query_knowledge_graph

# ---------------------------------------------------------------------
# Neo4j driver
# ---------------------------------------------------------------------

print("Connecting to Neo4j database (Text2Cypher + few-shot)...")
driver = get_driver()
print("Connected successfully!")

# ---------------------------------------------------------------------
# LLM (shared for Text2Cypher + RAG answer generation)
# ---------------------------------------------------------------------

print("\nInitializing OpenAI LLM for Text2Cypher & RAG...")
llm = get_llm()
t2c_llm = llm  # just an alias to mirror your original pattern

# ---------------------------------------------------------------------
# Few-shot examples for Text2CypherRetriever
# ---------------------------------------------------------------------
# These examples are specific to the given Scrum KG schema.
# They teach the LLM how to map natural language to Cypher queries.

examples = [
    # Role-focused question
    (
        "USER INPUT: 'What is the role of the Scrum Master?' "
        "QUERY: MATCH (r:Role {name: 'Scrum Master'})-[rel]-(n) "
        "RETURN r AS role, type(rel) AS relationship, labels(n) AS nodeLabels, n.name AS nodeName"
    ),

    # Product Owner & Product Backlog relationships
    (
        "USER INPUT: 'How does the Product Owner manage the Product Backlog?' "
        "QUERY: MATCH (po:Role {name: 'Product Owner'})-[rel]->(a:Artifact {name: 'Product Backlog'}) "
        "RETURN po AS role, type(rel) AS relationship, a AS artifact"
    ),

    # Event-focused: Sprint Planning
    (
        "USER INPUT: 'What happens during Sprint Planning?' "
        "QUERY: MATCH (e:Event {name: 'Sprint Planning'})-[rel]-(n) "
        "RETURN e AS event, type(rel) AS relationship, labels(n) AS nodeLabels, n.name AS nodeName"
    ),

    # All events and participating roles
    (
        "USER INPUT: 'List all Scrum events and which roles participate in them' "
        "QUERY: MATCH (r:Role)-[rel:PARTICIPATES_IN|ATTENDS]->(e:Event) "
        "RETURN r AS role, type(rel) AS relationship, e AS event "
        "ORDER BY e.name, r.name"
    ),
]

# ---------------------------------------------------------------------
# Text2CypherRetriever
# ---------------------------------------------------------------------

print("\nInitializing Text2CypherRetriever with few-shot examples...")

retriever = Text2CypherRetriever(
    driver=driver,
    llm=t2c_llm,
    examples=examples,
)

print("Text2CypherRetriever initialized successfully!")

# ---------------------------------------------------------------------
# GraphRAG pipeline using Text2CypherRetriever
# ---------------------------------------------------------------------

print("\nCreating GraphRAG pipeline (Text2Cypher + few-shot)...")

rag = GraphRAG(
    retriever=retriever,
    llm=llm,
)

print("GraphRAG (Text2Cypher + few-shot) pipeline ready!")

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("KNOWLEDGE GRAPH RAG SYSTEM READY (Text2CypherRetriever + examples)")
    print("=" * 70)

    # Example queries — reuse shared query helper
    print("\n\nRunning example queries with Text2CypherRetriever (few-shot)...\n")

    query_knowledge_graph(rag, "What is the role of Scrum Master?", top_k=None)
    query_knowledge_graph(rag, "How does the Product Owner manage the Product Backlog?", top_k=None)
    query_knowledge_graph(rag, "What happens during Sprint Planning?", top_k=None)

    # Optional: schema exploration
    print("\n\n")
    explore_graph_schema(driver)

    # Optional: specific entity info
    print("\n\n")
    get_entity_info(driver, "Scrum Master")

    # Interactive mode
    print("\n" + "=" * 70)
    print("INTERACTIVE MODE (Text2CypherRetriever with examples)")
    print("=" * 70)
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
                query_knowledge_graph(rag, user_question, top_k=None)
            else:
                print("Please enter a valid question.")
    finally:
        driver.close()
        print("Driver closed.")
