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
# LLM
# ---------------------------------------------------------------------

print("\nInitializing OpenAI LLM for Text2Cypher & RAG...")
llm = OpenAILLM(
    model_name="gpt-3.5-turbo",
    model_params={
        "max_tokens": 1000,
    },
    api_key=OPENAI_API_KEY,
)

# ---------------------------------------------------------------------
# Text2CypherRetriever configuration
# ---------------------------------------------------------------------

# You can tune this “system prompt” to your Scrum schema:
#   - What labels exist
#   - What relationships exist
#   - How to answer typical questions
TEXT2CYPHER_SYSTEM_PROMPT = """
You are an expert Neo4j Cypher generator for a Scrum knowledge graph.

The graph contains:
- Entities with labels such as: Role, Artifact, Event, Practice, Principle, Team, Chunk.
- Relationships such as: FACILITATES, MANAGES, PARTICIPATES_IN, CREATES, USES, OWNS, ATTENDS, RESPONSIBLE_FOR, FROM_CHUNK.

General guidelines:
- Use concise Cypher.
- When user asks about roles (e.g., Scrum Master, Product Owner), query Role nodes and their relationships.
- When user asks about Scrum events (Sprint Planning, Daily Scrum, Sprint Review, Sprint Retrospective),
  query Event nodes and connect them to related entities (roles, artifacts, practices) via relationships.
- If the question is broad, return multiple nodes and relevant relationships.

Return results that are useful as textual context for answering the question.
Do NOT modify the database. Use read-only queries (MATCH, OPTIONAL MATCH, RETURN).
"""



# ---------------------------------------------------------------------
# Text2CypherRetriever
# ---------------------------------------------------------------------

print("\nInitializing Text2CypherRetriever...")

text2cypher_retriever = Text2CypherRetriever(
    driver=driver,
    llm=llm,
    #custom_prompt=TEXT2CYPHER_SYSTEM_PROMPT   
)

print("Text2CypherRetriever initialized successfully!")

# ---------------------------------------------------------------------
# GraphRAG pipeline using Text2CypherRetriever
# ---------------------------------------------------------------------

print("\nCreating GraphRAG pipeline (Text2Cypher)...")

rag = GraphRAG(
    retriever=text2cypher_retriever,
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
        response = rag.search(
            query_text=question, return_context=True)

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
# Optional helpers: schema / entity exploration (same as others)
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
    print("KNOWLEDGE GRAPH RAG SYSTEM READY (Text2CypherRetriever)")
    print("=" * 70)

    # Example queries
    print("\n\nRunning example queries with Text2CypherRetriever...\n")

    query_knowledge_graph_text2cypher("What is the role of Scrum Master?")
    query_knowledge_graph_text2cypher("How does the Product Owner manage the Product Backlog?")
    query_knowledge_graph_text2cypher("What happens during Sprint Planning?")
    query_knowledge_graph_text2cypher("Explain the Daily Scrum ceremony")
    query_knowledge_graph_text2cypher("Show me all Scrum events and which roles participate in them")

    # Optional: schema exploration
    print("\n\n")
    explore_graph_schema()

    # Optional: specific entity info
    print("\n\n")
    get_entity_info("Scrum Master")

    # Interactive mode
    print("\n" + "=" * 70)
    print("INTERACTIVE MODE (Text2CypherRetriever)")
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
