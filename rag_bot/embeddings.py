from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np


TOKEN_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9'-]{1,}")


class EmbeddingModel:
    """Small local embedding model based on hashed TF-IDF features."""

    def __init__(self, model_name: str = "hashing-tfidf-768", dimension: int = 768) -> None:
        self.model_name = model_name
        self.dimension = dimension
        self.idf: dict[str, float] = {}

    def fit(self, texts: list[str]) -> None:
        document_count = len(texts)
        document_frequency: Counter[str] = Counter()
        for text in texts:
            document_frequency.update(set(_tokenize(text)))
        self.idf = {
            token: math.log((1 + document_count) / (1 + frequency)) + 1
            for token, frequency in document_frequency.items()
        }

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {"model_name": self.model_name, "dimension": self.dimension, "idf": self.idf},
                indent=2,
            ),
            encoding="utf-8",
        )

    def load(self, path: Path) -> None:
        data = json.loads(path.read_text(encoding="utf-8"))
        self.model_name = str(data["model_name"])
        self.dimension = int(data["dimension"])
        self.idf = {str(token): float(value) for token, value in data["idf"].items()}

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            raise ValueError("No texts provided for embedding.")
        if not self.idf:
            self.fit(texts)

        batches: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            batches.append(np.vstack([self._embed_one(text) for text in batch]))
        return np.vstack(batches).astype("float32")

    def embed_query(self, query: str) -> np.ndarray:
        return np.array([self._embed_one(query)], dtype="float32")

    def _embed_one(self, text: str) -> np.ndarray:
        tokens = _tokenize(text)
        counts = Counter(tokens)
        vector = np.zeros(self.dimension, dtype="float32")
        if not counts:
            return vector

        for token, count in counts.items():
            index = _stable_index(token, self.dimension)
            sign = 1.0 if _stable_index(f"{token}:sign", 2) == 0 else -1.0
            tf = 1.0 + math.log(count)
            vector[index] += sign * tf * self.idf.get(token, 1.0)

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector


def _tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


def _stable_index(text: str, dimension: int) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dimension
