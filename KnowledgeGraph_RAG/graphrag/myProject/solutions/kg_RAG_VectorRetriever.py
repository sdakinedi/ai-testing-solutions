import asyncio

# KG building
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import (
    FixedSizeSplitter,
)

# Retrievers + RAG
from neo4j_graphrag.retrievers import VectorRetriever, VectorCypherRetriever
from neo4j_graphrag.generation import GraphRAG

# Shared project modules
from common import get_driver, get_llm, get_embedder
from domain_constants import (
    SCRUM_ENTITIES,
    SCRUM_RELATIONS,
    SCRUM_KG_PROMPT_TEMPLATE,
    KG_BUILDER_CONFIG,
)
from utilities import explore_graph_schema, get_entity_info
from queries import query_knowledge_graph
from vector_index import ensure_chunk_vector_index


# Configuration


PDF_PATH = r"C:\\Users\\SatyaprasadDakinedi\\Desktop\\guideToScrum.pdf"

# Set this to True to rebuild the KG, False to skip if already built
REBUILD_KG = True


# Neo4j driver
print("Connecting to Neo4j database...")
driver = get_driver()
print("Connected successfully!")


# Check if KG already exists
def check_kg_exists(driver):
    """Check if knowledge graph has entity nodes"""
    with driver.session() as session:
        result = session.run("""
            MATCH (n)
            WHERE NOT n:Chunk AND NOT n:Document AND NOT n:__KGBuilder__
            RETURN count(n) as entity_count
        """)
        record = result.single()
        return record["entity_count"] > 0 if record else False

kg_exists = check_kg_exists(driver)
print(f"\nKnowledge Graph exists: {kg_exists}")


# LLM & Embeddings
print("\nInitializing LLM...")
llm = get_llm()  

print("Initializing embeddings...")
embedder = get_embedder()

print("\nLLM type:", type(llm))
print("Embedder type:", type(embedder))


# Text splitter
print("\nInitializing text splitter...")
text_splitter = FixedSizeSplitter(
    chunk_size=1000,
    chunk_overlap=150,
)


# Knowledge Graph builder
async def build_knowledge_graph():
    """Build the knowledge graph from PDF"""
    print("\nBuilding Knowledge Graph Pipeline...")
    try:
        kg_builder = SimpleKGPipeline(
            llm=llm,
            driver=driver,
            text_splitter=text_splitter,
            embedder=embedder,
            entities=SCRUM_ENTITIES,
            relations=SCRUM_RELATIONS,
            prompt_template=SCRUM_KG_PROMPT_TEMPLATE,
            from_pdf=True,
            **KG_BUILDER_CONFIG,  # Add error handling config
        )
        print("Knowledge Graph Pipeline created successfully!")
        
        print("\nProcessing PDF and building Knowledge Graph...")
        print("This may take several minutes depending on document size...")
        
        result = await kg_builder.run_async(file_path=PDF_PATH)
        print("\nKnowledge Graph built successfully!")
        return result
    except Exception as e:
        print(f"Error building knowledge graph: {e}")
        import traceback
        traceback.print_exc()
        return None

# Build or skip KG construction
if REBUILD_KG or not kg_exists:
    print("\nStarting Knowledge Graph construction...")
    kg_result = asyncio.run(build_knowledge_graph())
    
    if kg_result:
        print("\nKnowledge Graph Statistics:")
        print(f"Status: {kg_result}")
    else:
        print("\nKnowledge Graph construction failed. Exiting...")
        driver.close()
        exit(1)
else:
    print("\nSkipping KG construction (already exists)")


# Vector index on Chunk nodes
print("\nEnsuring vector index on Chunk nodes...")
try:
    ensure_chunk_vector_index(
        driver=driver,
        index_name="chunk_embeddings",
        label="Chunk",
        embedding_property="embedding",
        dimensions=1536,
    )
    print("Vector index has been completed.")
except Exception as e:
    print(f"Error creating vector index: {e}")
    import traceback
    traceback.print_exc()


# Verify Entity Creation
print("\nVerifying entity extraction...")
with driver.session() as session:
    # Check for entity nodes
    result = session.run("""
        MATCH (n)
        WHERE NOT n:Chunk AND NOT n:Document AND NOT n:__KGBuilder__
        RETURN labels(n)[0] as label, count(*) as count
        ORDER BY count DESC
        LIMIT 10
    """)
    
    entities = list(result)
    if entities:
        print("✓ Entity nodes found:")
        for record in entities:
            print(f"  - {record['label']}: {record['count']} nodes")
    else:
        print("⚠ WARNING: No entity nodes found!")
        print("  The KG may not have extracted entities properly.")
        print("  Check your LLM configuration and prompt template.")


# Vector Retriever Setup
print("\nInitializing Vector Retriever...")

try:
    retriever = VectorRetriever(
        driver=driver,
        index_name="chunk_embeddings",
        embedder=embedder,
        return_properties=["text"],
    )
    print("Vector Retriever initialized successfully!")
except Exception as e:
    print(f"Error initializing Vector Retriever: {e}")
    import traceback
    traceback.print_exc()
    driver.close()
    exit(1)

# GraphRAG pipeline
print("\nCreating GraphRAG pipeline...")
try:
    rag = GraphRAG(
        retriever=retriever,
        llm=llm,
    )
    print("GraphRAG pipeline ready!")
except Exception as e:
    print(f"Error creating GraphRAG pipeline: {e}")
    import traceback
    traceback.print_exc()
    driver.close()
    exit(1)


# Main
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("KNOWLEDGE GRAPH RAG SYSTEM READY (VectorRetriever)")
    print("=" * 70)

    # Example queries
    print("\n\nRunning example queries...\n")

    try:
        query_knowledge_graph(rag, "What is the role of Scrum Master?")
        query_knowledge_graph(rag, "How does the Product Owner manage the Product Backlog?")
        query_knowledge_graph(rag, "What happens during Sprint Planning?")
    except Exception as e:
        print(f"Error running queries: {e}")
        import traceback
        traceback.print_exc()

    # Explore the graph
    print("\n\n")
    try:
        explore_graph_schema(driver)
    except Exception as e:
        print(f"Error exploring graph schema: {e}")

    # Get specific entity info
    print("\n\n")
    try:
        # First, let's find actual entity names
        with driver.session() as session:
            result = session.run("""
                MATCH (n)
                WHERE n.name IS NOT NULL 
                AND NOT n:Chunk AND NOT n:Document
                RETURN DISTINCT n.name as name
                LIMIT 5
            """)
            entity_names = [record["name"] for record in result]
            
            if entity_names:
                print(f"Sample entities found: {entity_names}")
                get_entity_info(driver, entity_names[0])
            else:
                print("No named entities found in the graph yet.")
                
    except Exception as e:
        print(f"Error getting entity info: {e}")

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
                try:
                    query_knowledge_graph(rag, user_question)
                except Exception as e:
                    print(f"Error processing query: {e}")
            else:
                print("Please enter a valid question.")
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    finally:
        driver.close()
        print("Driver closed.")