from __future__ import annotations

import argparse
from pathlib import Path

from rag_bot.chunker import chunk_documents
from rag_bot.document_loader import load_documents
from rag_bot.embeddings import EmbeddingModel
from rag_bot.vector_store import FaissVectorStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index documents into a persistent FAISS vector store.")
    parser.add_argument("--data-dir", default="data", help="Folder containing PDF, TXT, MD, or DOCX files.")
    parser.add_argument("--storage-dir", default="storage", help="Folder where the FAISS index is saved.")
    parser.add_argument("--embedding-model", default="hashing-tfidf-768")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=900)
    parser.add_argument("--chunk-overlap", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    storage_dir = Path(args.storage_dir)

    print(f"Loading documents from {data_dir}...")
    pages = load_documents(data_dir)
    print(f"Loaded {len(pages)} document pages/sections.")

    print("Chunking documents...")
    chunks = chunk_documents(pages, max_chars=args.chunk_size, overlap_chars=args.chunk_overlap)
    print(f"Created {len(chunks)} chunks.")

    print(f"Embedding chunks with {args.embedding_model} in batches of {args.batch_size}...")
    embedding_model = EmbeddingModel(args.embedding_model)
    embeddings = embedding_model.embed_texts([chunk.text for chunk in chunks], batch_size=args.batch_size)
    storage_dir.mkdir(parents=True, exist_ok=True)
    embedding_model.save(storage_dir / "embedding_model.json")

    print(f"Saving FAISS vector store to {storage_dir}...")
    store = FaissVectorStore(storage_dir)
    store.save(embeddings, chunks)
    print("Indexing complete.")


if __name__ == "__main__":
    main()
