from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from rag_bot.chunker import Chunk


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float


class FaissVectorStore:
    def __init__(self, storage_dir: Path) -> None:
        self.storage_dir = storage_dir
        self.index_path = storage_dir / "faiss.index"
        self.chunks_path = storage_dir / "chunks.json"
        self.index: faiss.Index | None = None
        self.chunks: list[Chunk] = []

    def save(self, embeddings: np.ndarray, chunks: list[Chunk]) -> None:
        if len(embeddings) != len(chunks):
            raise ValueError("Embedding count and chunk count do not match.")

        self.storage_dir.mkdir(parents=True, exist_ok=True)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)

        faiss.write_index(index, str(self.index_path))
        self.chunks_path.write_text(
            json.dumps([chunk.to_dict() for chunk in chunks], indent=2),
            encoding="utf-8",
        )
        self.index = index
        self.chunks = chunks

    def load(self) -> None:
        if not self.index_path.exists() or not self.chunks_path.exists():
            raise FileNotFoundError(
                "Vector store not found. Run: python -m rag_bot.index --data-dir data --storage-dir storage"
            )
        self.index = faiss.read_index(str(self.index_path))
        raw_chunks = json.loads(self.chunks_path.read_text(encoding="utf-8"))
        self.chunks = [Chunk.from_dict(chunk) for chunk in raw_chunks]

    def search(self, query_embedding: np.ndarray, top_k: int = 4) -> list[SearchResult]:
        if self.index is None:
            raise RuntimeError("Vector store is not loaded.")

        scores, indexes = self.index.search(query_embedding, top_k)
        results: list[SearchResult] = []
        for score, index in zip(scores[0], indexes[0]):
            if index < 0:
                continue
            results.append(SearchResult(chunk=self.chunks[int(index)], score=float(score)))
        return results
