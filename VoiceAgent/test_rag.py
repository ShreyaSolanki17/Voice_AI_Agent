"""Self-check for RAG retriever."""

# ponytail: one-liner asserts for critical paths
from rag.retriever import retrieve, CHROMA_PATH
from pathlib import Path

def test_kb_exists():
    assert Path("rag/kb").exists(), "KB directory missing"
    assert len(list(Path("rag/kb").glob("*.md"))) >= 3, "Need 3 KB files"

def test_chroma_ready():
    client_path = Path(CHROMA_PATH)
    # After ingestion, chroma_db should have files
    if client_path.exists():
        files = list(client_path.glob("**/*"))
        # Should have persisted data

def test_shipping_query():
    """Should find shipping info in KB."""
    context, docs, scores = retrieve("When will my box arrive?")
    assert context, "Shipping query should retrieve context"
    assert "3-7 business days" in context, "Should find delivery timeframe"

def test_escalation_query():
    """Account-specific query should return low similarity."""
    context, docs, scores = retrieve("Where is order #4521?")
    # This should NOT find good context since it's account-specific
    # Either empty context OR low similarity score triggers fallback
    if context and scores:
        assert max(scores) < 0.7, "Account query should have low similarity"

if __name__ == "__main__":
    # Quick test - run after `python -m rag.ingest`
    import os
    if os.getenv("OPENAI_API_KEY"):
        test_shipping_query()
        test_escalation_query()
        print("RAG tests passed")