import os
from dotenv import load_dotenv

from neo4j import GraphDatabase

# LLM + Embeddings
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings

# Retriever + RAG
from neo4j_graphrag.retrievers import VectorCypherRetriever
from neo4j_graphrag.generation import GraphRAG

# Optional: ensure vector index (idempotent)
from vector_index import ensure_chunk_vector_index

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

load_dotenv()  # Loads NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, OPENAI_API_KEY

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
# LLM & Embeddings
# ---------------------------------------------------------------------

print("\nInitializing OpenAI LLM...")
llm = OpenAILLM(
    model_name="gpt-3.5-turbo",
    model_params={
        "max_tokens": 1000,
    },
    api_key=OPENAI_API_KEY,
)

print("Initializing embeddings...")
embedder = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    api_key=OPENAI_API_KEY,
)

# ---------------------------------------------------------------------
# Ensure vector index (Chunk nodes) – safe to call even if it exists
# ---------------------------------------------------------------------

print("\nEnsuring vector index on Chunk nodes...")
ensure_chunk_vector_index(
    driver=driver,
    index_name="chunk_embeddings",
    label="Chunk",
    embedding_property="embedding",
    dimensions=1536,
)
print("Vector index step completed.")

# ---------------------------------------------------------------------
# Cypher-based graph-aware retrieval query
# ---------------------------------------------------------------------
# This query:
#  - Starts from the 'node' (the chunk returned by vector search) and its score.
#  - Finds entities connected via :FROM_CHUNK.
#  - Expands 1–2 hops from those entities, excluding :FROM_CHUNK relationships.
#  - Filters related nodes to roles, artifacts, and events.
#  - Returns enriched text with mentioned entities and related concepts.

retrieval_query = """
WITH node AS chunk, score
MATCH (chunk)<-[:FROM_CHUNK]-(entity)

OPTIONAL MATCH (entity)-[rel*1..2]-(relatedEntity)
WHERE all(r IN rel WHERE type(r) <> 'FROM_CHUNK')
  AND (relatedEntity:Role OR relatedEntity:Artifact OR relatedEntity:Event)

WITH
  chunk,
  score,
  collect(DISTINCT entity.name) AS entities,
  collect(DISTINCT relatedEntity.name) AS relatedEntities

WITH
  chunk.text AS chunkText,
  score,
  [e IN entities WHERE e IS NOT NULL] AS entityList,
  [re IN relatedEntities WHERE re IS NOT NULL] AS relatedList

RETURN
  chunkText +
  CASE
    WHEN size(entityList) > 0
      THEN '\n\nMentioned Entities: ' + apoc.text.join(entityList, ', ')
    ELSE ''
  END +
  CASE
    WHEN size(relatedList) > 0
      THEN '\nRelated Concepts: ' + apoc.text.join(relatedList, ', ')
    ELSE ''
  END AS text,
  score
ORDER BY score DESC
"""

# ---------------------------------------------------------------------
# VectorCypherRetriever
# ---------------------------------------------------------------------

print("\nInitializing VectorCypherRetriever...")

vector_cypher_retriever = VectorCypherRetriever(
    driver=driver,
    index_name="chunk_embeddings",
    embedder=embedder,
    retrieval_query=retrieval_query,
)

print("VectorCypherRetriever initialized successfully!")

# ---------------------------------------------------------------------
# GraphRAG pipeline using VectorCypherRetriever
# ---------------------------------------------------------------------

print("\nCreating GraphRAG pipeline (VectorCypher)...")

rag = GraphRAG(
    retriever=vector_cypher_retriever,
    llm=llm,
)

print("GraphRAG (VectorCypher) pipeline ready!")

# ---------------------------------------------------------------------
# Query function
# ---------------------------------------------------------------------


def query_knowledge_graph_vector_cypher(question: str, top_k: int = 3):
    print("\n" + "=" * 70)
    print(f"Question: {question}")
    print("=" * 70)

    try:
        response = rag.search(
            query_text=question,
            retriever_config={"top_k": top_k},
        )

        print(f"\nAnswer:\n{response.answer}")

        # Show retriever result items (context chunks)
        if response.retriever_result:
            print(
                f"\n--- Retrieved Context Chunks "
                f"({len(response.retriever_result.items)} items) ---"
            )
            for idx, item in enumerate(response.retriever_result.items, 1):
                print(f"\nChunk {idx}:")
                preview = item.content[:400] if len(item.content) > 400 else item.content
                print(preview)
                if hasattr(item, "metadata") and item.metadata:
                    print(f"Metadata: {item.metadata}")

        return response

    except Exception as e:
        print(f"Error during query: {e}")
        return None


# ---------------------------------------------------------------------
# Optional helpers: schema / entity exploration
# ---------------------------------------------------------------------


def explore_graph_schema():
    print("\n" + "=" * 70)
    print("GRAPH SCHEMA INFORMATION")
    print("=" * 70)

    with driver.session() as session:
        # Node labels
        result = session.run("CALL db.labels()")
        labels = [record["label"] for record in result]
        print(f"\nNode Labels: {', '.join(labels)}")

        # Relationship types
        result = session.run("CALL db.relationshipTypes()")
        rel_types = [record["relationshipType"] for record in result]
        #print(f"\nRelationship Types: {', '.join(rel_types)}")

        # Count nodes by label
        print("\nNode Counts by Label:")
        for label in labels:
            result = session.run(f"MATCH (n:{label}) RETURN count(n) as count")
            count = result.single()["count"]
            print(f"  {label}: {count}")

        # Count relationships
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
    print("KNOWLEDGE GRAPH RAG SYSTEM READY (VectorCypherRetriever)")
    print("=" * 70)

    # Example queries
    print("\n\nRunning example queries with VectorCypherRetriever...\n")

    query_knowledge_graph_vector_cypher("What is the role of Scrum Master?")
    query_knowledge_graph_vector_cypher("How does the Product Owner manage the Product Backlog?")
    query_knowledge_graph_vector_cypher("What happens during Sprint Planning?")
    query_knowledge_graph_vector_cypher("Explain the Daily Scrum ceremony")

    # Optional: schema exploration
    print("\n\n")
    explore_graph_schema()

    # Optional: specific entity info
    print("\n\n")
    get_entity_info("Scrum Master")

    # Interactive mode
    print("\n" + "=" * 70)
    print("INTERACTIVE MODE (VectorCypherRetriever)")
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
                query_knowledge_graph_vector_cypher(user_question)
            else:
                print("Please enter a valid question.")
    finally:
        driver.close()
        print("Driver closed.")
