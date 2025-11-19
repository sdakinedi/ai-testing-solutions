# """
# Utility functions for graph exploration and debugging.
# Used across all retriever scripts.
# """

def explore_graph_schema(driver):
    """
    Print node labels, relationship types,
    node counts, and total relationship count.
    """
    print("\n" + "=" * 70)
    print("GRAPH SCHEMA INFORMATION")
    print("=" * 70)

    with driver.session() as session:
        # Labels
        labels_res = session.run("CALL db.labels()")
        labels = [record["label"] for record in labels_res]
        print(f"\nNode Labels: {', '.join(labels)}")

        # Relationship types
        rels_res = session.run("CALL db.relationshipTypes()")
        rel_types = [record["relationshipType"] for record in rels_res]
        print(f"\nRelationship Types: {', '.join(rel_types)}")

        # Count per label
        print("\nNode Counts:")
        for label in labels:
            count = session.run(f"MATCH (n:{label}) RETURN count(n) AS c").single()["c"]
            print(f"  {label}: {count}")

        # Total relationships
        total_rel_count = session.run(
            "MATCH ()-[r]->() RETURN count(r) AS c"
        ).single()["c"]
        print(f"\nTotal Relationships: {total_rel_count}")

# This is for debugging specific entities in the graph

def get_entity_info(driver, entity_name: str):
    """
    Print an entity, its type, and all outgoing/incoming relationships.
    """
    print("\n" + "=" * 70)
    print(f"ENTITY INFORMATION: {entity_name}")
    print("=" * 70)

    query = """
    MATCH (e {name: $name})
    OPTIONAL MATCH (e)-[r]-(related)
    RETURN e.name AS entity,
           labels(e) AS entity_type,
           collect(DISTINCT {
               relationship: type(r),
               related_entity: related.name,
               related_type: labels(related)[0]
           }) AS connections
    """

    with driver.session() as session:
        record = session.run(query, name=entity_name).single()

        if not record:
            print(f"No entity found with name '{entity_name}'.")
            return

        print(f"\nEntity: {record['entity']}")
        print(f"Type: {record['entity_type']}")

        print("\nConnections:")
        for conn in record["connections"]:
            if conn["related_entity"]:
                print(
                    f"  {conn['relationship']} -> "
                    f"{conn['related_entity']} ({conn['related_type']})"
                )
