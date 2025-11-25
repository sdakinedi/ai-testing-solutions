import asyncio

# KG building
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import (
    FixedSizeSplitter,
)

# Retrievers + RAG
from neo4j_graphrag.retrievers import VectorRetriever
from neo4j_graphrag.generation import GraphRAG

# Shared project modules
from common import get_driver, get_llm, get_embedder
from domain_constants import (
    SCRUM_ENTITIES,
    SCRUM_RELATIONS,
    SCRUM_KG_PROMPT_TEMPLATE,
)
from utilities import explore_graph_schema, get_entity_info
from queries import query_knowledge_graph
from vector_index import ensure_chunk_vector_index

# ---------------------------------------------------------------------
# source files path
# ---------------------------------------------------------------------

PDF_PATH = r"C:\Users\Satyaprasad_Dakinedi\Desktop\guideToScrum.pdf"

# ---------------------------------------------------------------------
# Neo4j driver
# ---------------------------------------------------------------------

print("Connecting to Neo4j database...")
driver = get_driver()
print("Connected successfully!")

# ---------------------------------------------------------------------
# LLM & Embeddings
# ---------------------------------------------------------------------

print("\nInitializing OpenAI LLM...")
llm = get_llm()  

print("Initializing embeddings...")
embedder = get_embedder()

# ---------------------------------------------------------------------
# Text splitter
# ---------------------------------------------------------------------

print("Initializing text splitter...")
text_splitter = FixedSizeSplitter(
    chunk_size=500,
    chunk_overlap=150,
)

# ---------------------------------------------------------------------
# Knowledge Graph builder
# ---------------------------------------------------------------------

print("\nBuilding Knowledge Graph Pipeline...")
kg_builder = SimpleKGPipeline(
    llm=llm,
    driver=driver,
    text_splitter=text_splitter,
    embedder=embedder,
    entities=SCRUM_ENTITIES,
    relations=SCRUM_RELATIONS,
    prompt_template=SCRUM_KG_PROMPT_TEMPLATE,
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
# Vector index on Chunk nodes
# ---------------------------------------------------------------------

print("\nEnsuring vector index on Chunk nodes...")
ensure_chunk_vector_index(
    driver=driver,
    index_name="chunk_embeddings",
    label="Chunk",
    embedding_property="embedding",
    dimensions=1536,
)
print("Vector index has been completed.")

# ---------------------------------------------------------------------
# Vector Retriever 
# ---------------------------------------------------------------------

print("\nInitializing Vector Retriever...")


#   (:Chunk { text: '...', embedding: <vector> })
retriever = VectorRetriever(
    driver=driver,
    index_name="chunk_embeddings",
    embedder=embedder,
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
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("KNOWLEDGE GRAPH RAG SYSTEM READY (VectorRetriever)")
    print("=" * 70)

    # Example queries (uses shared query helper)
    print("\n\nRunning example queries...\n")

    query_knowledge_graph(rag, "What is the role of Scrum Master?")
    query_knowledge_graph(rag, "How does the Product Owner manage the Product Backlog?")
    query_knowledge_graph(rag, "What happens during Sprint Planning?")

    # Explore the graph (from utilities.py)
    print("\n\n")
    explore_graph_schema(driver)

    # Get specific entity info
    print("\n\n")
    get_entity_info(driver, "Scrum Master")

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
                query_knowledge_graph(rag, user_question)
            else:
                print("Please enter a valid question.")
    finally:
        # Safety close
        driver.close()
        print("Driver closed.")
