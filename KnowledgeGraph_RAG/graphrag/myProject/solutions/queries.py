def query_knowledge_graph(rag, question: str, top_k: int = 3):
    """
    Execute a query against a GraphRAG pipeline and print
    the answer and retrieved context.

    :param rag: GraphRAG instance
    :param question: user question
    :param top_k: number of context items to retrieve (if retriever supports it)
    """
    print("\n======================================================")
    print(f"Question: {question}")
    print("======================================================")

    try:
        retriever_config = {"top_k": top_k} if top_k is not None else {}

        response = rag.search(
            query_text=question,
            retriever_config=retriever_config,
            return_context=True
        )

        print(f"\nAnswer:\n{response.answer}")
        print("\nRetrieved Context:", response.retriever_result.items)
        print("\nRetrieved Result is::\n", response.retriever_result.metadata["cypher"])
  

    except Exception as e:
        print(f"Error during query: {e}")
        return None
