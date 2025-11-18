import os
import asyncio
from pathlib import Path
from neo4j import GraphDatabase
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
from neo4j_graphrag.indexes import create_vector_index
from neo4j_graphrag.retrievers import VectorCypherRetriever
from neo4j_graphrag.generation import GraphRAG

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEO4J_URI = "neo4j+s://3f713ff7.databases.neo4j.io"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "Vi4P4Y53IA99vUfrklzp9OCYMX2s6M313JC5xYCLWys"
PDF_PATH = r"C:\Users\Satyaprasad_Dakinedi\Desktop\guideToScrum.pdf"

# Initialize Neo4j driver
print("Connecting to Neo4j...")
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
print("Connected successfully!")

# Initialize LLM
print("\nInitializing OpenAI LLM...")
llm = OpenAILLM(
    model_name="gpt-3.5-turbo",
    model_params={
        "max_tokens": 1000,
        "response_format": {"type": "json_object"},
    },
    api_key=OPENAI_API_KEY
)

# Initialize embeddings
print("Initializing embeddings...")
embedder = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    api_key=OPENAI_API_KEY
)

# Initialize text splitter
print("Initializing text splitter...")
text_splitter = FixedSizeSplitter(
    chunk_size=500,
    chunk_overlap=150
)

# Define entities and relationships for Scrum domain
entities = ["Role", "Artifact", "Event", "Practice", "Principle", "Team"]
relations = ["FACILITATES", "MANAGES", "PARTICIPATES_IN", "CREATES", "USES", "OWNS", "ATTENDS", "RESPONSIBLE_FOR"]

# Custom prompt for better entity extraction
prompt_template = """
You are an expert at extracting entities and relationships from Scrum documentation.

Extract entities of these types: Role, Artifact, Event, Practice, Principle, Team
Extract relationships of these types: FACILITATES, MANAGES, PARTICIPATES_IN, CREATES, USES, OWNS, ATTENDS, RESPONSIBLE_FOR

From the following text, identify all entities and their relationships.

Text: {text}

Return the result as a JSON object with entities and relationships.
"""

# Build Knowledge Graph Pipeline
print("\nBuilding Knowledge Graph Pipeline...")
kg_builder = SimpleKGPipeline(
    llm=llm,
    driver=driver,
    text_splitter=text_splitter,
    embedder=embedder,
    entities=entities,
    relations=relations,
    prompt_template=prompt_template,
    from_pdf=True
)

# Function to build the knowledge graph
async def build_knowledge_graph():
    print("\nProcessing PDF and building Knowledge Graph...")
    print("This may take a few minutes depending on PDF size...")
    
    try:
        result = await kg_builder.run_async(file_path=PDF_PATH)
        print("\nKnowledge Graph built successfully!")
        return result
    except Exception as e:
        print(f"Error building knowledge graph: {e}")
        return None

# Run the KG building process
print("\nStarting Knowledge Graph construction...")
result = asyncio.run(build_knowledge_graph())

if result:
    print("\nKnowledge Graph Statistics:")
    print(f"Status: {result}")

# Create vector index for semantic search
print("\nCreating vector index...")
try:
    create_vector_index(
        driver,
        name="chunk_embeddings",
        label="Chunk",
        embedding_property="embedding",
        dimensions=1536,
        similarity_fn="cosine"
    )
    print("Vector index created successfully!")
except Exception as e:
    print(f"Vector index may already exist or error: {e}")

# Setup VectorCypherRetriever
print("\nInitializing Vector + Graph Retriever...")

retrieval_query = """
WITH node AS chunk, score
MATCH (chunk)<-[:FROM_CHUNK]-(entity)

OPTIONAL MATCH (entity)-[rel:!FROM_CHUNK*1..2]-(relatedEntity)
WHERE relatedEntity:Role OR relatedEntity:Artifact OR relatedEntity:Event

WITH chunk, score, 
     collect(DISTINCT entity.name) AS entities,
     collect(DISTINCT type(rel[0])) AS relationships,
     collect(DISTINCT relatedEntity.name) AS relatedEntities

WITH chunk.text AS chunkText,
     score,
     [e IN entities WHERE e IS NOT NULL | e] AS entityList,
     [r IN relationships WHERE r IS NOT NULL | r] AS relationshipList,
     [re IN relatedEntities WHERE re IS NOT NULL | re] AS relatedList

RETURN chunkText + 
       CASE WHEN size(entityList) > 0 
            THEN '\n\nMentioned Entities: ' + apoc.text.join(entityList, ', ')
            ELSE '' END +
       CASE WHEN size(relatedList) > 0 
            THEN '\nRelated Concepts: ' + apoc.text.join(relatedList, ', ')
            ELSE '' END AS text,
       score
ORDER BY score DESC
"""

vector_cypher_retriever = VectorCypherRetriever(
    driver=driver,
    index_name="chunk_embeddings",
    embedder=embedder,
    retrieval_query=retrieval_query
)

print("Vector Cypher Retriever initialized successfully!")

# Create GraphRAG pipeline
print("\nCreating GraphRAG pipeline...")
rag = GraphRAG(
    retriever=vector_cypher_retriever,
    llm=llm
)

print("GraphRAG pipeline ready!")

# Query function
def query_knowledge_graph(question, top_k=3):
    print("\n" + "="*70)
    print(f"Question: {question}")
    print("="*70)
    
    try:
        response = rag.search(
            query_text=question, 
            retriever_config={"top_k": top_k}
        )
        
        print(f"\nAnswer:\n{response.answer}")
        
        print(f"\n--- Retrieved Context Chunks ({len(response.items)} items) ---")
        for idx, item in enumerate(response.items, 1):
            print(f"\nChunk {idx}:")
            content_preview = item.content[:300] if len(item.content) > 300 else item.content
            print(f"{content_preview}...")
            if hasattr(item, 'metadata') and item.metadata:
                print(f"Metadata: {item.metadata}")
        
        return response
    
    except Exception as e:
        print(f"Error during query: {e}")
        return None

# Utility function to explore the graph
def explore_graph_schema():
    print("\n" + "="*70)
    print("GRAPH SCHEMA INFORMATION")
    print("="*70)
    
    with driver.session() as session:
        # Get node labels
        result = session.run("CALL db.labels()")
        labels = [record["label"] for record in result]
        print(f"\nNode Labels: {', '.join(labels)}")
        
        # Get relationship types
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

# Function to find entity details
def get_entity_info(entity_name):
    print("\n" + "="*70)
    print(f"ENTITY INFORMATION: {entity_name}")
    print("="*70)
    
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
            for conn in record['connections']:
                if conn['related_entity']:
                    print(f"  {conn['relationship']} -> {conn['related_entity']} ({conn['related_type']})")
        else:
            print(f"No entity found with name: {entity_name}")

# Main execution
if __name__ == "__main__":
    print("\n" + "="*70)
    print("KNOWLEDGE GRAPH RAG SYSTEM READY")
    print("="*70)
    
    # Example queries
    print("\n\nRunning example queries...\n")
    
    # Query 1
    query_knowledge_graph("What is the role of Scrum Master?")
    
    # Query 2
    query_knowledge_graph("How does the Product Owner manage the Product Backlog?")
    
    # Query 3
    query_knowledge_graph("What happens during Sprint Planning?")
    
    # Query 4
    query_knowledge_graph("Explain the Daily Scrum ceremony")
    
    # Explore the graph
    print("\n\n")
    explore_graph_schema()
    
    # Get specific entity info
    print("\n\n")
    get_entity_info("Scrum Master")
    
    # Interactive query mode
    print("\n" + "="*70)
    print("INTERACTIVE MODE")
    print("="*70)
    print("\nYou can now ask questions. Type 'exit' to quit.\n")
    
    while True:
        user_question = input("Your question: ")
        if user_question.lower() in ['exit', 'quit', 'q']:
            print("\nClosing connection...")
            driver.close()
            print("Goodbye!")
            break
        
        if user_question.strip():
            query_knowledge_graph(user_question)
        else:
            print("Please enter a valid question.")
    
print("\nProgram completed.")