"""Gemini embedding helpers for ChromaDB."""

import os
from functools import lru_cache
from typing import Sequence

import google.genai as genai


EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")


def _prepare_document(content: str, title: str | None = None) -> str:
    if title is None:
        title = "none"
    return f"title: {title} | text: {content}"


def _prepare_query(query: str) -> str:
    return f"task: search result | query: {query}"


@lru_cache(maxsize=1)
def _client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY required for embeddings")
    return genai.Client(api_key=api_key)


def _extract_embedding(response) -> list[float]:
    if hasattr(response, "embeddings") and response.embeddings:
        return response.embeddings[0].values
    if isinstance(response, dict) and response.get("embeddings"):
        return response["embeddings"][0]["values"]
    raise ValueError("Unexpected Gemini embedding response shape")


def _embed(content: str) -> list[float]:
    response = _client().models.embed_content(
        model=EMBEDDING_MODEL,
        contents=content,
        config={"output_dimensionality": 768},
    )
    return _extract_embedding(response)


def embed_documents(texts: Sequence[str]) -> list[list[float]]:
    return [_embed(_prepare_document(text)) for text in texts]


def embed_query(text: str) -> list[float]:
    return _embed(_prepare_query(text))