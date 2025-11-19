from neo4j_graphrag.indexes import create_vector_index

def ensure_chunk_vector_index(
    driver,
    index_name: str = "chunk_embeddings",
    label: str = "Chunk",
    embedding_property: str = "embedding",
    dimensions: int = 1536,
    similarity_fn: str = "cosine",
):
    """
    Create (or ensure) the vector index on Chunk nodes.

    Expects nodes like:
      (:Chunk { embedding: <vector>, text: <str>, ... })
    """
    try:
        create_vector_index(
            driver=driver,
            name=index_name,
            label=label,
            embedding_property=embedding_property,
            dimensions=dimensions,
            similarity_fn=similarity_fn,
        )
        print(f"Vector index '{index_name}' created / already exists.")
    except Exception as e:
        print(f"Vector index '{index_name}' may already exist or failed: {e}")
