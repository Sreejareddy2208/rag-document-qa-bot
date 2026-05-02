from __future__ import annotations

from rag_bot.vector_store import SearchResult


SYSTEM_PROMPT = """You are a careful document Q&A assistant.
Answer only from the provided context.
If the answer is not present in the context, say you do not have enough information in the documents.
Include citations using the source labels provided with each context chunk.
Do not use outside knowledge."""


def build_context(results: list[SearchResult]) -> str:
    blocks = []
    for number, result in enumerate(results, start=1):
        blocks.append(
            f"[Source {number}: {result.chunk.citation}; similarity={result.score:.3f}]\n"
            f"{result.chunk.text}"
        )
    return "\n\n".join(blocks)


def build_user_prompt(question: str, results: list[SearchResult]) -> str:
    context = build_context(results)
    return f"""Context:
{context}

Question: {question}

Write a concise answer grounded in the context. End with a Sources line listing the citations used."""
