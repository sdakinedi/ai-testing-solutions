import os
from dotenv import load_dotenv

from neo4j import GraphDatabase

# LLM
from neo4j_graphrag.llm import OpenAILLM

# Retriever + RAG
from neo4j_graphrag.retrievers import Text2CypherRetriever
from neo4j_graphrag.generation import GraphRAG

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

load_dotenv()  # NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, OPENAI_API_KEY

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ---------------------------------------------------------------------
# Neo4j driver
# ---------------------------------------------------------------------

print("Connecting to Neo4j database...")
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"), 
    auth=(
        os.getenv("NEO4J_USERNAME"), 
        os.getenv("NEO4J_PASSWORD")
    )
)

print("Connected successfully!")

# ---------------------------------------------------------------------
# LLM (shared for Text2Cypher + RAG answer generation)
# ---------------------------------------------------------------------

print("\nInitializing OpenAI LLM for Text2Cypher & RAG...")
llm = OpenAILLM(
    model_name="gpt-3.5-turbo",
    model_params={
        "max_tokens": 1000,
    },
    api_key=OPENAI_API_KEY,
)

# Use a separate variable name if you like, as in your snippet
t2c_llm = llm

# ---------------------------------------------------------------------
# Few-shot examples for Text2CypherRetriever
# ---------------------------------------------------------------------
# These examples are specific to your Scrum KG schema.
# They teach the LLM how to map natural language to Cypher queries.

examples = [
    # 1) Role-focused question
    (
        "USER INPUT: 'What is the role of the Scrum Master?' "
        "QUERY: MATCH (r:Role {name: 'Scrum Master'})-[rel]-(n) "
        "RETURN r AS role, type(rel) AS relationship, labels(n) AS nodeLabels, n.name AS nodeName"
    ),

    # 2) Product Owner & Product Backlog relationships
    (
        "USER INPUT: 'How does the Product Owner manage the Product Backlog?' "
        "QUERY: MATCH (po:Role {name: 'Product Owner'})-[rel]->(a:Artifact {name: 'Product Backlog'}) "
        "RETURN po AS role, type(rel) AS relationship, a AS artifact"
    ),

    # 3) Event-focused: Sprint Planning
    (
        "USER INPUT: 'What happens during Sprint Planning?' "
        "QUERY: MATCH (e:Event {name: 'Sprint Planning'})-[rel]-(n) "
        "RETURN e AS event, type(rel) AS relationship, labels(n) AS nodeLabels, n.name AS nodeName"
    ),

    # 5) All events and participating roles
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

print("\nCreating GraphRAG pipeline (Text2Cypher)...")

rag = GraphRAG(
    retriever=retriever,
    llm=llm,
)

print("GraphRAG (Text2Cypher) pipeline ready!")

# ---------------------------------------------------------------------
# Query function
# ---------------------------------------------------------------------


def query_knowledge_graph_text2cypher(question: str):
    print("\n" + "=" * 70)
    print(f"Question: {question}")
    print("=" * 70)

    try:
        response = rag.search(query_text=question)

        print(f"\nAnswer:\n{response.answer}")

        # Show retriever result items (graph-based context)
        if response.retriever_result:
            print(
                f"\n--- Retrieved Graph Context "
                f"({len(response.retriever_result.items)} items) ---"
            )
            for idx, item in enumerate(response.retriever_result.items, 1):
                print(f"\nItem {idx}:")
                preview = item.content[:400] if len(item.content) > 400 else item.content
                print(preview)
                if hasattr(item, "metadata") and item.metadata:
                    print(f"Metadata: {item.metadata}")

        return response

    except Exception as e:
        print(f"Error during query: {e}")
        return None


# ---------------------------------------------------------------------
# Optional helpers: schema / entity exploration (same as other files)
# ---------------------------------------------------------------------


def explore_graph_schema():
    print("\n" + "=" * 70)
    print("GRAPH SCHEMA INFORMATION")
    print("=" * 70)

    with driver.session() as session:
        result = session.run("CALL db.labels()")
        labels = [record["label"] for record in result]
        print(f"\nNode Labels: {', '.join(labels)}")

        result = session.run("CALL db.relationshipTypes()")
        rel_types = [record["relationshipType"] for record in result]
        print(f"\nRelationship Types: {', '.join(rel_types)}")

        print("\nNode Counts by Label:")
        for label in labels:
            result = session.run(f"MATCH (n:{label}) RETURN count(n) as count")
            count = result.single()["count"]
            print(f"  {label}: {count}")

        result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
        rel_count = result.single()["count"]
        print(f"\nTotal Relationships: {rel_count}")


def get_entity_info(entity_name: str):
    print("\n" + "=" * 70)
    print(f"ENTITY INFORMATION: {entity_name}")
    print("=" * 70)

    with driver.session() as session:
        query = """
        MATCH (e {name: $name})
        OPTIONAL MATCH (e)-[r]-(related)
        RETURN e.name as entity, 
               labels(e) as entity_type,
               collect(DISTINCT {
                   relationship: type(r),
                   related_entity: related.name,
                   related_type: labels(related)[0]
               }) as connections
        """
        result = session.run(query, name=entity_name)

        record = result.single()
        if record:
            print(f"\nEntity: {record['entity']}")
            print(f"Type: {record['entity_type']}")
            print("\nConnections:")
            for conn in record["connections"]:
                if conn["related_entity"]:
                    print(
                        f"  {conn['relationship']} -> "
                        f"{conn['related_entity']} ({conn['related_type']})"
                    )
        else:
            print(f"No entity found with name: {entity_name}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("KNOWLEDGE GRAPH RAG SYSTEM READY (Text2CypherRetriever + examples)")
    print("=" * 70)

    # Example queries
    print("\n\nRunning example queries with Text2CypherRetriever...\n")

    query_knowledge_graph_text2cypher("What is the role of Scrum Master?")
    query_knowledge_graph_text2cypher("How does the Product Owner manage the Product Backlog?")
    query_knowledge_graph_text2cypher("What happens during Sprint Planning?")
    query_knowledge_graph_text2cypher("Explain the Daily Scrum ceremony")
    query_knowledge_graph_text2cypher("List all Scrum events and which roles participate in them")

    # Optional: schema exploration
    print("\n\n")
    explore_graph_schema()

    # Optional: specific entity info
    print("\n\n")
    get_entity_info("Scrum Master")

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
                query_knowledge_graph_text2cypher(user_question)
            else:
                print("Please enter a valid question.")
    finally:
        driver.close()
        print("Driver closed.")
