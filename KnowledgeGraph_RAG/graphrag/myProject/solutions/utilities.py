"""
Utility functions for exploring and debugging the Knowledge Graph
"""

def explore_graph_schema(driver):
    """
    Explore the complete graph schema including nodes, relationships, and counts
    """
    print("=" * 70)
    print("GRAPH SCHEMA INFORMATION")
    print("=" * 70)
    
    with driver.session() as session:
        # Get all node labels
        result = session.run("CALL db.labels()")
        labels = [record["label"] for record in result]
        print(f"\nNode Labels: {', '.join(labels)}")
        
        # Get all relationship types
        result = session.run("CALL db.relationshipTypes()")
        rel_types = [record["relationshipType"] for record in result]
        print(f"\nRelationship Types: {', '.join(rel_types)}")
        
        # Get counts for each label
        print("\nNode Counts:")
        for label in labels:
            result = session.run(f"MATCH (n:`{label}`) RETURN count(n) as count")
            count = result.single()["count"]
            print(f"  {label}: {count}")
        
        # Get total relationships
        result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
        total_rels = result.single()["count"]
        print(f"\nTotal Relationships: {total_rels}")
        
        # Show sample entity nodes (non-system nodes)
        print("\n" + "-" * 70)
        print("SAMPLE ENTITY NODES (non-system)")
        print("-" * 70)
        
        result = session.run("""
            MATCH (n)
            WHERE NOT n:Chunk AND NOT n:Document AND NOT n:__KGBuilder__
            RETURN labels(n) as labels, n.name as name, n.description as description
            LIMIT 10
        """)
        
        entities_found = False
        for record in result:
            entities_found = True
            label = record["labels"][0] if record["labels"] else "Unknown"
            name = record["name"] or "No name"
            desc = record["description"] or "No description"
            print(f"\n{label}: {name}")
            if desc != "No description":
                print(f"  Description: {desc[:100]}...")
        
        if not entities_found:
            print("\n ALERT!!!!! No entity nodes found!")
            print("  Only Chunk/Document nodes exist.")
            print("  Entity extraction may have failed.")


def get_entity_info(driver, entity_name):
    """
    Get detailed information about a specific entity and its connections
    """
    print("=" * 70)
    print(f"ENTITY INFORMATION: {entity_name}")
    print("=" * 70)
    
    with driver.session() as session:
        # Try to find entity by name property
        result = session.run("""
            MATCH (e)
            WHERE e.name = $name
            OPTIONAL MATCH (e)-[r]-(related)
            RETURN e.name AS entity,
                   labels(e) AS entity_type,
                   e.description AS description,
                   collect(DISTINCT {
                       relationship: type(r),
                       related_entity: related.name,
                       related_type: labels(related)[0]
                   }) AS connections
            LIMIT 1
        """, name=entity_name)
        
        record = result.single()
        
        if record:
            print(f"\nEntity: {record['entity']}")
            print(f"Type: {record['entity_type']}")
            
            if record['description']:
                print(f"Description: {record['description']}")
            
            print("\nConnections:")
            for conn in record['connections']:
                if conn['related_entity']:
                    print(f"  -{conn['relationship']}-> {conn['related_type']}: {conn['related_entity']}")
        else:
            print(f"\nNo entity found with name '{entity_name}'.")
            
            # Try fuzzy search
            print("\nSearching for similar entities...")
            result = session.run("""
                MATCH (e)
                WHERE e.name IS NOT NULL 
                AND toLower(e.name) CONTAINS toLower($search)
                RETURN e.name as name, labels(e) as type
                LIMIT 5
            """, search=entity_name)
            
            similar = list(result)
            if similar:
                print("Found similar entities:")
                for rec in similar:
                    print(f"  - {rec['type'][0]}: {rec['name']}")
            else:
                print("No similar entities found.")


def debug_kg_extraction(driver):
    """
    Comprehensive debugging of KG extraction status
    """
    print("\n" + "=" * 70)
    print("KNOWLEDGE GRAPH DEBUG REPORT")
    print("=" * 70)
    
    with driver.session() as session:
        # 1. Check for Chunk nodes
        result = session.run("MATCH (c:Chunk) RETURN count(c) as count")
        chunk_count = result.single()["count"]
        print(f"\n✓ Chunk nodes: {chunk_count}")
        
        # 2. Check for embeddings
        result = session.run("""
            MATCH (c:Chunk) 
            WHERE c.embedding IS NOT NULL 
            RETURN count(c) as count
        """)
        embedded_count = result.single()["count"]
        print(f"Chunks with embeddings: {embedded_count}")
        
        # 3. Check for entity nodes
        result = session.run("""
            MATCH (n)
            WHERE NOT n:Chunk AND NOT n:Document AND NOT n:__KGBuilder__
            RETURN count(n) as count
        """)
        entity_count = result.single()["count"]
        
        if entity_count > 0:
            print(f"Entity nodes: {entity_count}")
            
            # Show entity distribution
            result = session.run("""
                MATCH (n)
                WHERE NOT n:Chunk AND NOT n:Document AND NOT n:__KGBuilder__
                RETURN labels(n)[0] as type, count(*) as count
                ORDER BY count DESC
            """)
            print("\n  Entity distribution:")
            for record in result:
                print(f"    {record['type']}: {record['count']}")
        else:
            print("Entity nodes: 0")
            print("\n PROBLEM DETECTED: No entities extracted!")
            print("\nPossible causes:")
            print("  1. LLM failed to extract entities (check API keys/quotas)")
            print("  2. Prompt template not configured correctly")
            print("  3. Entity types don't match the content")
            print("  4. PDF processing failed")
            
        # 4. Check for relationships between entities
        result = session.run("""
            MATCH (a)-[r]->(b)
            WHERE NOT a:Chunk AND NOT b:Chunk 
            AND NOT a:Document AND NOT b:Document
            RETURN count(r) as count
        """)
        rel_count = result.single()["count"]
        
        if rel_count > 0:
            print(f"\nEntity relationships: {rel_count}")
        else:
            print("\nEntity relationships: 0")
            
        # 5. Show sample chunk content
        print("\n" + "-" * 70)
        print("SAMPLE CHUNK CONTENT")
        print("-" * 70)
        result = session.run("""
            MATCH (c:Chunk)
            RETURN c.text as text
            LIMIT 2
        """)
        
        for i, record in enumerate(result, 1):
            print(f"\nChunk {i}:")
            print(record["text"][:200] + "...")


def list_all_entities(driver, limit=20):
    """
    List all extracted entities in the knowledge graph
    """
    print("\n" + "=" * 70)
    print(f"ALL ENTITIES (limit: {limit})")
    print("=" * 70)
    
    with driver.session() as session:
        result = session.run("""
            MATCH (n)
            WHERE NOT n:Chunk AND NOT n:Document AND NOT n:__KGBuilder__
            AND n.name IS NOT NULL
            RETURN labels(n)[0] as type, n.name as name
            ORDER BY type, name
            LIMIT $limit
        """, limit=limit)
        
        entities = list(result)
        
        if entities:
            current_type = None
            for record in entities:
                if record['type'] != current_type:
                    current_type = record['type']
                    print(f"\n{current_type}:")
                print(f"  - {record['name']}")
        else:
            print("\nNo named entities found in the graph.")


def get_entity_relationships_summary(driver):
    """
    Get a summary of all relationship types and their frequencies
    """
    print("\n" + "=" * 70)
    print("RELATIONSHIP SUMMARY")
    print("=" * 70)
    
    with driver.session() as session:
        result = session.run("""
            MATCH (a)-[r]->(b)
            WHERE NOT a:Chunk AND NOT b:Chunk
            RETURN type(r) as relationship, 
                   labels(a)[0] as from_type,
                   labels(b)[0] as to_type,
                   count(*) as count
            ORDER BY count DESC
            LIMIT 20
        """)
        
        relationships = list(result)
        
        if relationships:
            for record in relationships:
                print(f"\n{record['from_type']} -{record['relationship']}-> {record['to_type']}")
                print(f"  Count: {record['count']}")
        else:
            print("\nNo entity relationships found.")