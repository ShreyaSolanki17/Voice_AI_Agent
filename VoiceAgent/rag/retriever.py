"""Retrieve relevant KB chunks for RAG."""

import os
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from loguru import logger
from prompts.system_prompt import SIMILARITY_THRESHOLD


CHROMA_PATH = Path(__file__).parent.parent / "chroma_db"
COLLECTION_NAME = "brightbox_kb"


_client = None
_collection = None


def _get_collection():
    global _client, _collection
    if _collection is None:
        api_key = os.getenv("OPENAI_API_KEY")
        embed_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name="text-embedding-3-small"
        )
        _client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        _collection = _client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embed_fn
        )
    return _collection


def retrieve(query: str, k: int = 3) -> tuple[str, list[str], list[float]]:
    """Embed query and retrieve top-k chunks + scores."""
    collection = _get_collection()
    results = collection.query(
        query_texts=[query],
        n_results=k,
        include=["documents", "distances"]
    )

    docs = results["documents"][0] if results["documents"] else []
    distances = results["distances"][0] if results["distances"] else []

    # Convert cosine distance to similarity (lower distance = higher similarity)
    # Chroma uses cosine distance by default: 0=identical, 2=opposite
    similarities = [1 - d for d in distances]

    # Check if any chunk meets threshold
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