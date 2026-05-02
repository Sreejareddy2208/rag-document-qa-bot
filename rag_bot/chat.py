from __future__ import annotations

import argparse
from pathlib import Path

from rag_bot.config import load_settings
from rag_bot.embeddings import EmbeddingModel
from rag_bot.llm import LlmClient
from rag_bot.prompts import build_user_prompt
from rag_bot.vector_store import FaissVectorStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask questions against the indexed document collection.")
    parser.add_argument("--storage-dir", default="storage", help="Folder containing the saved FAISS index.")
    parser.add_argument("--embedding-model", default="hashing-tfidf-768")
    parser.add_argument("--top-k", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings()

    store = FaissVectorStore(Path(args.storage_dir))
    store.load()
    embedding_model = EmbeddingModel(args.embedding_model)
    embedding_model.load(Path(args.storage_dir) / "embedding_model.json")
    llm = LlmClient(settings)

    print("Document Q&A bot is ready. Type 'exit' or 'quit' to stop.")
    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break
        if not question:
            continue

        query_embedding = embedding_model.embed_query(question)
        results = store.search(query_embedding, top_k=args.top_k)
        prompt = build_user_prompt(question, results)

        print("\nAnswer:")
        print(llm.generate(prompt))

        print("\nRetrieved sources:")
        for number, result in enumerate(results, start=1):
            preview = result.chunk.text.replace("\n", " ")[:240]
            print(f"{number}. {result.chunk.citation} | score={result.score:.3f}")
            print(f"   {preview}...")


if __name__ == "__main__":
    main()
