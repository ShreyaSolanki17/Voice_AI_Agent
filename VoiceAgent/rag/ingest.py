"""Ingest BrightBox KB documents into ChromaDB."""

import os
from pathlib import Path
import chromadb
from dotenv import load_dotenv
from loguru import logger
from rag.gemini_embedder import embed_documents

load_dotenv()

KB_DIR = Path(__file__).parent / "kb"
CHROMA_PATH = Path(__file__).parent.parent / "chroma_db"
COLLECTION_NAME = "brightbox_kb_gemini"


def chunk_text(text: str, max_tokens: int = 200) -> list[str]:
    """Simple paragraph-based chunking."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""

    for para in paragraphs:
        if current and len(current) + len(para) > max_tokens * 4:  # rough char estimate
            chunks.append(current)
            current = para
        else:
            current = current + "\n\n" + para if current else para

    if current:
        chunks.append(current)
    return chunks


def main():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY required for embeddings")
        return

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    all_ids = []
    all_chunks = []
    all_metadatas = []

    for kb_file in sorted(KB_DIR.glob("*.md")):
        content = kb_file.read_text()
        chunks = chunk_text(content)

        for i, chunk in enumerate(chunks):
            chunk_id = f"{kb_file.stem}_{i}"
            all_ids.append(chunk_id)
            all_chunks.append(chunk)
            all_metadatas.append({"source": kb_file.name})
            logger.info(f"Ingested chunk {chunk_id} from {kb_file.name}")

    if all_ids:
        embeddings = embed_documents(all_chunks)
        collection.upsert(
            ids=all_ids,
            documents=all_chunks,
            embeddings=embeddings,
            metadatas=all_metadatas,
        )
        logger.success(f"Ingested {len(all_ids)} chunks into {COLLECTION_NAME}")


if __name__ == "__main__":
    main()