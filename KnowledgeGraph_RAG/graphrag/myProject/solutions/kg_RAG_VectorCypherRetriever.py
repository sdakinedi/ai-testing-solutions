from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.retrievers import VectorCypherRetriever
from neo4j_graphrag.generation import GraphRAG

from common import get_driver, get_llm, get_embedder
from utilities import explore_graph_schema, get_entity_info
from queries import query_knowledge_graph
from vector_index import ensure_chunk_vector_index


# Neo4j driver
driver = get_driver()
print("Connected to Neo4j database successfully!")


# LLM & Embeddings
print("\nInitializing LLM...")
llm = get_llm()  

print("Initializing embeddings...")
embedder = get_embedder()  


#vector index (Chunk nodes) 
print("\ vector index on Chunk nodes...")
ensure_chunk_vector_index(
    driver=driver,
    index_name="chunk_embeddings",
    label="Chunk",
    embedding_property="embedding",
    dimensions=1536,
)
print("Vector index Step completed.")


# Cypher-based graph-aware retrieval query

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


# VectorCypherRetriever
print("\nInitializing VectorCypherRetriever...")

vector_cypher_retriever = VectorCypherRetriever(
    driver=driver,
    index_name="chunk_embeddings",
    embedder=embedder,
    retrieval_query=retrieval_query,
)

print("VectorCypherRetriever initialized successfully!")


# GraphRAG pipeline using VectorCypherRetriever
print("\nCreating GraphRAG pipeline (VectorCypher)...")

rag = GraphRAG(
    retriever=vector_cypher_retriever,
    llm=llm,
)

print("GraphRAG (VectorCypher) pipeline ready!")


# Main
if __name__ == "__main__":
    print("\n======================================================")
    print("KNOWLEDGE GRAPH RAG SYSTEM READY (VectorCypherRetriever)")
    print("======================================================")

    # Example queries using the shared query helper
    print("\n\nRunning example queries with VectorCypherRetriever...\n")

    query_knowledge_graph(rag, "What is the role of Scrum Master?", top_k=3)
    query_knowledge_graph(rag, "How does the Product Owner manage the Product Backlog?", top_k=3)
    query_knowledge_graph(rag, "What happens during Sprint Planning?", top_k=3)
    query_knowledge_graph(rag, "Explain the Daily Scrum ceremony", top_k=3)

    # Optional: schema exploration
    print("\n\n")
    explore_graph_schema(driver)

    # Optional: specific entity info
    print("\n\n")
    get_entity_info(driver, "Scrum Master")

    # Interactive mode
    print("\n======================================================")
    print("INTERACTIVE MODE (VectorCypherRetriever)")
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
                query_knowledge_graph(rag, user_question, top_k=3)
            else:
                print("Please enter a valid question.")
    finally:
        driver.close()
        print("Driver closed.")
