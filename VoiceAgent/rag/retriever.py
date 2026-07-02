"""Retrieve relevant KB chunks for RAG."""

import os
from pathlib import Path
import chromadb
from loguru import logger
from prompts.system_prompt import SIMILARITY_THRESHOLD
from rag.gemini_embedder import embed_query


CHROMA_PATH = Path(__file__).parent.parent / "chroma_db"
COLLECTION_NAME = "brightbox_kb_gemini"


_client = None
_collection = None


def _get_collection():
    """Initializes and caches a singleton ChromaDB client and collection."""
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        _collection = _client.get_collection(name=COLLECTION_NAME)
    return _collection


def retrieve(query: str, k: int = 3) -> tuple[str, list[str], list[float]]:
    """Embed query and retrieve top-k chunks + scores."""
    collection = _get_collection()
    query_embedding = embed_query(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "distances"]
    )

    docs = results["documents"][0] if results["documents"] else []
    distances = results["distances"][0] if results["distances"] else []

    # ChromaDB's cosine distance is 0 for identical, 1 for orthogonal, 2 for opposite.
    # Convert to similarity score where 1 is identical and 0 is orthogonal.
    similarities = [1 - d for d in distances]

    # Filter out chunks that don't meet the minimum similarity threshold.
    relevant = [d for d, s in zip(docs, similarities) if s >= SIMILARITY_THRESHOLD]
    scores = [s for s in similarities if s >= SIMILARITY_THRESHOLD]

    if not relevant:
        logger.warning(f"No relevant chunks found (threshold={SIMILARITY_THRESHOLD})")
        return "", [], []

    logger.info(f"Retrieved {len(relevant)} chunks, best score: {max(scores):.2f}")
    return "\n\n".join(relevant), [d for d in docs], similarities


if __name__ == "__main__":
    # Quick self-test
    test_queries = ["When will my box arrive?", "Where is order #12345?", "How do I cancel?"]
    for q in test_queries:
        result, _, _ = retrieve(q)
        print(f"\nQ: {q}\n{'→ Found match' if result else '→ No match (fallback needed)'}")