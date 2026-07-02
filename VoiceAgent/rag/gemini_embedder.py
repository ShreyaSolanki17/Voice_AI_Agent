"""Gemini embedding helpers for ChromaDB."""

import os
from functools import lru_cache
from typing import Sequence

import google.genai as genai


# The embedding model to use. Defaults to a common, high-performance model.
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")


def _prepare_document(content: str, title: str | None = None) -> str:
    """Prepare a document for embedding by adding a task-specific prefix."""
    if title is None:
        title = "none"
    return f"title: {title} | text: {content}"


def _prepare_query(query: str) -> str:
    """Prepare a search query for embedding by adding a task-specific prefix."""
    return f"task: search result | query: {query}"


@lru_cache(maxsize=1)
def _client() -> genai.Client:
    """Create and cache a singleton Gemini client to avoid re-authentication."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY required for embeddings")
    return genai.Client(api_key=api_key)


def _extract_embedding(response) -> list[float]:
    """Extract the embedding values from a Gemini API response."""
    if hasattr(response, "embeddings") and response.embeddings:
        return response.embeddings[0].values
    if isinstance(response, dict) and response.get("embeddings"):
        return response["embeddings"][0]["values"]
    raise ValueError("Unexpected Gemini embedding response shape")


def _embed(content: str) -> list[float]:
    response = _client().models.embed_content(
        model=EMBEDDING_MODEL,
        contents=content,
        # Specify output dimensionality for consistent vector sizes.
        config={"output_dimensionality": 768},
    )
    return _extract_embedding(response)


def embed_documents(texts: Sequence[str]) -> list[list[float]]:
    """Embed a list of documents for storage in the vector database."""
    return [_embed(_prepare_document(text)) for text in texts]


def embed_query(text: str) -> list[float]:
    """Embed a single user query for searching the vector database."""
    return _embed(_prepare_query(text))