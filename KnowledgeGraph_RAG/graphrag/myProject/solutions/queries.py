"""
Shared query helpers for GraphRAG pipelines.
These functions take a GraphRAG instance and handle:
- Printing the question
- Executing the search
- Printing the answer and retrieved context
"""

def query_knowledge_graph(rag, question: str, top_k: int = 3):
    """
    Execute a query against a GraphRAG pipeline and print
    the answer and retrieved context.

    :param rag: GraphRAG instance
    :param question: user question
    :param top_k: number of context items to retrieve (if retriever supports it)
    """
    print("\n" + "=" * 70)
    print(f"Question: {question}")
    print("=" * 70)

    try:
        retriever_config = {"top_k": top_k} if top_k is not None else {}

        response = rag.search(
            query_text=question,
            retriever_config=retriever_config,
        )

        print(f"\nAnswer:\n{response.answer}")

        # Show retriever result items if available
        if response.retriever_result and response.retriever_result.items:
            print(
                f"\n--- Retrieved Context "
                f"({len(response.retriever_result.items)} items) ---"
            )
            for idx, item in enumerate(response.retriever_result.items, 1):
                print(f"\nItem {idx}:")
                content = getattr(item, "content", "")
                preview = content[:400] if len(content) > 400 else content
                print(preview)
                metadata = getattr(item, "metadata", None)
                if metadata:
                    print(f"Metadata: {metadata}")
        else:
            print("\nNo context items retrieved.")

        return response

    except Exception as e:
        print(f"Error during query: {e}")
        return None
