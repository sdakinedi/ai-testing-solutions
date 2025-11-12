import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from neo4j import GraphDatabase
import json

# STEP 1: Load and Process PDF Documents
loaders = [
    PyPDFLoader("C:\\Users\\Satyaprasad_Dakinedi\\Desktop\\guideToScrum.pdf"),
]
docs = []
for loader in loaders:
    docs.extend(loader.load())

documents = docs

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=150,
    length_function=len
)

chunks = text_splitter.split_documents(documents)
print(f"############ Total chunks: {len(chunks)}")

# STEP 2: Setup OpenAI
openai_api_key = os.getenv("OPENAI_API_KEY")  

embeddings = OpenAIEmbeddings(
    openai_api_key=openai_api_key,
    model="text-embedding-ada-002"
)

llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    model="gpt-3.5-turbo",
    temperature=0
)

#
# STEP 3: Knowledge Graph Extraction

class KnowledgeGraphBuilder:
    def __init__(self, llm):
        self.llm = llm
        prompt_template = """
        Extract entities and relationships from the following text about Scrum.
        Focus on:
        - Entities: Roles (Scrum Master, Product Owner, etc.), Artifacts (Sprint Backlog, Product Backlog), Events (Sprint, Daily Scrum), Concepts
        - Relationships: describes how entities relate to each other

        Text: {text}

        Return a JSON object with this structure:
        {{
        "entities": [
            {{"name": "entity_name", "type": "entity_type", "properties": {{}}}},
        ],
        "relationships": [
            {{"source": "entity1", "target": "entity2", "type": "relationship_type"}},
        ]
        }}

        JSON:
        """
        self.extraction_prompt = ChatPromptTemplate.from_template(prompt_template)
    
    def extract_kg_from_chunk(self, chunk_text):
        """Extract knowledge graph elements from a text chunk"""
        try:
            chain = self.extraction_prompt | self.llm | StrOutputParser()
            response = chain.invoke({"text": chunk_text})
            
            # Parse JSON response
            kg_data = json.loads(response)
            return kg_data
        except Exception as e:
            print(f"Error extracting KG: {e}")
            return {"entities": [], "relationships": []}


# STEP 4: Neo4j Knowledge Graph Storage
class Neo4jKnowledgeGraph:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()
    
    def create_entity(self, tx, entity):
        """Create or merge an entity node"""
        query = f"""
        MERGE (e:{entity['type']} {{name: $name}})
        SET e += $properties
        RETURN e
        """
        tx.run(query, name=entity['name'], properties=entity.get('properties', {}))
    
    def create_relationship(self, tx, rel):
        """Create a relationship between entities"""
        query = """
        MATCH (a {name: $source})
        MATCH (b {name: $target})
        MERGE (a)-[r:RELATES_TO {type: $rel_type}]->(b)
        RETURN r
        """
        tx.run(query, source=rel['source'], target=rel['target'], rel_type=rel['type'])
    
    def add_chunk_node(self, tx, chunk_id, chunk_text, embedding):
        """Add a document chunk node with embedding"""
        query = """
        CREATE (c:Chunk {id: $chunk_id, text: $text, embedding: $embedding})
        RETURN c
        """
        tx.run(query, chunk_id=chunk_id, text=chunk_text, embedding=embedding)
    
    def link_chunk_to_entities(self, tx, chunk_id, entity_names):
        """Link chunk to mentioned entities"""
        for entity_name in entity_names:
            query = """
            MATCH (c:Chunk {id: $chunk_id})
            MATCH (e {name: $entity_name})
            MERGE (c)-[:MENTIONS]->(e)
            """
            tx.run(query, chunk_id=chunk_id, entity_name=entity_name)
    
    def retrieve_context_by_entities(self, tx, query_entities, limit=5):
        """Retrieve chunks connected to query entities"""
        cypher_query = """
        MATCH (e) WHERE e.name IN $entities
        MATCH (c:Chunk)-[:MENTIONS]->(e)
        RETURN DISTINCT c.text as text, c.id as chunk_id
        LIMIT $limit
        """
        result = tx.run(cypher_query, entities=query_entities, limit=limit)
        return [record["text"] for record in result]
    
    def retrieve_related_entities(self, tx, entity_names):
        """Get entities related to the query entities"""
        cypher_query = """
        MATCH (e1) WHERE e1.name IN $entities
        MATCH (e1)-[r]-(e2)
        RETURN DISTINCT e2.name as entity, type(r) as relationship
        LIMIT 10
        """
        result = tx.run(cypher_query, entities=entity_names)
        return [(record["entity"], record["relationship"]) for record in result]

#####################################
# STEP 5: Build the Knowledge Graph
#####################################
# Initialize Neo4j Aura (cloud instance)
NEO4J_URI = "neo4j+s://3f713ff7.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "5Q-QUg1_JDCH7mSjk0Z2iQjd4dkOuirWz3yBuu95A4k"

kg = Neo4jKnowledgeGraph(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
kg_builder = KnowledgeGraphBuilder(llm)

print("Building Knowledge Graph...")

for idx, chunk in enumerate(chunks[:20]):  # Process first 20 chunks for demo
    print(f"Processing chunk {idx + 1}/{min(20, len(chunks))}")
    
    # Extract entities and relationships
    kg_data = kg_builder.extract_kg_from_chunk(chunk.page_content)
    
    # Generate embedding for the chunk
    chunk_embedding = embeddings.embed_query(chunk.page_content)
    
    # Store in Neo4j
    with kg.driver.session() as session:
        # Add chunk node
        session.execute_write(kg.add_chunk_node, f"chunk_{idx}", chunk.page_content, chunk_embedding)
        
        # Add entities
        entity_names = []
        for entity in kg_data.get('entities', []):
            session.execute_write(kg.create_entity, entity)
            entity_names.append(entity['name'])
        
        # Add relationships
        for rel in kg_data.get('relationships', []):
            session.execute_write(kg.create_relationship, rel)
        
        # Link chunk to entities
        if entity_names:
            session.execute_write(kg.link_chunk_to_entities, f"chunk_{idx}", entity_names)

print("Knowledge Graph built successfully!")

# STEP 6: Knowledge Graph RAG Pipeline for retrieval
class KnowledgeGraphRetriever:
    def __init__(self, kg, llm):
        self.kg = kg
        self.llm = llm
        prompt_template = """
            Extract the main entities (roles, artifacts, events, concepts) from this question about Scrum.
            Return only a JSON list of entity names.

            Question: {question}

            JSON list:

        """
        self.entity_extraction_prompt = ChatPromptTemplate.from_template(prompt_template)
    
    def extract_query_entities(self, question):
        """Extract entities from the user's question"""
        try:
            chain = self.entity_extraction_prompt | self.llm | StrOutputParser()
            response = chain.invoke({"question": question})
            entities = json.loads(response)
            return entities if isinstance(entities, list) else []
        except:
            return []
    
    def retrieve(self, question):
        """Retrieve context from knowledge graph"""
        # Extract entities from question
        query_entities = self.extract_query_entities(question)
        
        if not query_entities:
            return "No relevant entities found in the question."
        
        with self.kg.driver.session() as session:
            # Get related entities
            related = session.execute_read(self.kg.retrieve_related_entities, query_entities)
            
            # Get relevant chunks
            all_entities = query_entities + [e[0] for e in related]
            chunks = session.execute_read(self.kg.retrieve_context_by_entities, all_entities[:5])
            
            context = "\n\n".join(chunks)
            
            # Add relationship information
            if related:
                rel_info = "\n\nRelated concepts:\n" + "\n".join([f"- {e} ({r})" for e, r in related[:5]])
                context += rel_info
            
            return context

# Setup KG Retriever
kg_retriever = KnowledgeGraphRetriever(kg, llm)

# Define prompt template
template = """You are an AI assistant for question-answering tasks about Scrum.
Use the following context retrieved from a knowledge graph to answer the question.
The context includes relevant text chunks and related concepts.

If you don't know the answer, just say that you don't know.
Use two sentences maximum and keep the answer concise.

Question: {question}
Context: {context}
Answer:
"""

prompt = ChatPromptTemplate.from_template(template)

# Build RAG chain
kg_rag_chain = (
    {"context": lambda x: kg_retriever.retrieve(x), "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


# Test Your RAG:Query the Knowledge Graph RAG
question = "What is the role of Scrum master"
print(f"\nQuestion: {question}")
response = kg_rag_chain.invoke(question)
print(f"Answer: {response}")

# new query
question2 = "How does the Product Owner interact with the Sprint Backlog?"
print(f"\nQuestion: {question2}")
response2 = kg_rag_chain.invoke(question2)
print(f"Answer: {response2}")

# Close Neo4j connection
kg.close()