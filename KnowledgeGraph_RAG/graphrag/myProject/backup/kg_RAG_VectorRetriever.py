import os
import asyncio
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

# KG building
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import (
    FixedSizeSplitter,
)

# LLM + Embeddings
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings

# Retrievers + RAG
from neo4j_graphrag.retrievers import VectorRetriever
from neo4j_graphrag.generation import GraphRAG

# Local helper for vector index
from vector_index import ensure_chunk_vector_index

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

load_dotenv()  # NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, OPENAI_API_KEY, etc.

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PDF_PATH = r"C:\\Users\\Satyaprasad_Dakinedi\\Desktop\\guideToScrum.pdf"

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
        "temperature": 0.1,
        #"max_tokens": 1000,
        # "response_format": {"type": "json_object"},
    },
    api_key=OPENAI_API_KEY,
)

print("Initializing embeddings...")
embedder = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    api_key=OPENAI_API_KEY,
)

# ---------------------------------------------------------------------
# Text splitter
# ---------------------------------------------------------------------

print("Initializing text splitter...")
text_splitter = FixedSizeSplitter(
    chunk_size=500,
    chunk_overlap=150,
)

# ---------------------------------------------------------------------
# Domain entities & relations (Scrum)
# ---------------------------------------------------------------------

entities = ["Role", "Artifact", "Event", "Practice", "Principle", "Team"]
relations = [
    "FACILITATES",
    "MANAGES",
    "PARTICIPATES_IN",
    "CREATES",
    "USES",
    "OWNS",
    "ATTENDS",
    "RESPONSIBLE_FOR",
]

prompt_template = """
You are an expert at extracting entities and relationships from Scrum documentation.

Extract entities of these types: Role, Artifact, Event, Practice, Principle, Team
Extract relationships of these types: FACILITATES, MANAGES, PARTICIPATES_IN, CREATES, USES, OWNS, ATTENDS, RESPONSIBLE_FOR

From the following text, identify all entities and their relationships.

Text: {text}

Return the result as a JSON object with entities and relationships.
"""

# ---------------------------------------------------------------------
# Knowledge Graph builder
# ---------------------------------------------------------------------

print("\nBuilding Knowledge Graph Pipeline...")
kg_builder = SimpleKGPipeline(
    llm=llm,
    driver=driver,
    text_splitter=text_splitter,
    embedder=embedder,
    entities=entities,
    relations=relations,
    prompt_template=prompt_template,
    from_pdf=True,
)


async def build_knowledge_graph():
    print("\nProcessing PDF and building Knowledge Graph...")
    try:
        result = await kg_builder.run_async(file_path=PDF_PATH)
        print("\nKnowledge Graph built successfully!")
        return result
    except Exception as e:
        print(f"Error building knowledge graph: {e}")
        return None


print("\nStarting Knowledge Graph construction...")
kg_result = asyncio.run(build_knowledge_graph())

if kg_result:
    print("\nKnowledge Graph Statistics:")
    print(f"Status: {kg_result}")

# ---------------------------------------------------------------------
# Vector index (separated into helper module)
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
# Vector Retriever (simple, like your movie example)
# ---------------------------------------------------------------------

print("\nInitializing Vector Retriever...")

# We assume your KG builder created nodes like:
#   (:Chunk { text: "...", embedding: <vector> })
retriever = VectorRetriever(
    driver=driver,
    index_name="chunk_embeddings",
    embedder=embedder,
    # This will become item.content
    return_properties=["text"],
)

print("Vector Retriever initialized successfully!")

# ---------------------------------------------------------------------
# GraphRAG pipeline
# ---------------------------------------------------------------------

print("\nCreating GraphRAG pipeline...")
rag = GraphRAG(
    retriever=retriever,
    llm=llm,
)
print("GraphRAG pipeline ready!")

# ---------------------------------------------------------------------
# Query function
# ---------------------------------------------------------------------


def query_knowledge_graph(question: str, top_k: int = 3):
    print("\n" + "=" * 70)
    print(f"Question: {question}")
    print("=" * 70)

    try:
        response = rag.search(
            query_text=question,
            retriever_config={"top_k": top_k},
        )

        print(f"\nAnswer:\n{response.answer}")

        # Raw retriever items (vector results)
        if response.retriever_result:
            print(f"\n--- Retrieved Context Chunks ({len(response.retriever_result.items)} items) ---")
            for idx, item in enumerate(response.retriever_result.items, 1):
                print(f"\nChunk {idx}:")
                preview = item.content[:300] if len(item.content) > 300 else item.content
                print(preview)
                if hasattr(item, "metadata") and item.metadata:
                    print(f"Metadata: {item.metadata}")
        else:
            print("No context chunks retrieved.")

        return response

    except Exception as e:
        print(f"Error during query: {e}")
        return None


# ---------------------------------------------------------------------
# Utility functions (schema + entity exploration) – unchanged logic
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
        print(f"\nRelationship Types: {', '.join(rel_types)}")

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
    print("KNOWLEDGE GRAPH RAG SYSTEM READY (VectorRetriever)")
    print("=" * 70)

    # Example queries
    print("\n\nRunning example queries...\n")

    query_knowledge_graph("What is the role of Scrum Master?")
    query_knowledge_graph("How does the Product Owner manage the Product Backlog?")
    query_knowledge_graph("What happens during Sprint Planning?")
    query_knowledge_graph("Explain the Daily Scrum ceremony")

    # Explore the graph
    print("\n\n")
    explore_graph_schema()

    # Get specific entity info
    print("\n\n")
    get_entity_info("Scrum Master")

    # Interactive query mode
    print("\n" + "=" * 70)
    print("INTERACTIVE MODE")
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
                query_knowledge_graph(user_question)
            else:
                print("Please enter a valid question.")
    finally:
        # Safety close
        driver.close()
        print("Driver closed.")
