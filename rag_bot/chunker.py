from __future__ import annotations

from dataclasses import dataclass, asdict

from rag_bot.document_loader import DocumentPage


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    page: int | None
    chunk_number: int
    text: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "Chunk":
        return cls(
            id=str(data["id"]),
            source=str(data["source"]),
            page=data["page"] if data["page"] is None else int(data["page"]),
            chunk_number=int(data["chunk_number"]),
            text=str(data["text"]),
        )

    @property
    def citation(self) -> str:
        page = f", page {self.page}" if self.page is not None else ""
        return f"{self.source}{page}, chunk {self.chunk_number}"


def chunk_documents(
    pages: list[DocumentPage],
    max_chars: int = 900,
    overlap_chars: int = 180,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    per_source_counts: dict[str, int] = {}

    for page in pages:
        for text in _chunk_text(page.text, max_chars=max_chars, overlap_chars=overlap_chars):
            count = per_source_counts.get(page.source, 0) + 1
            per_source_counts[page.source] = count
            chunks.append(
                Chunk(
                    id=f"{page.source}::{page.page or 'document'}::{count}",
                    source=page.source,
                    page=page.page,
                    chunk_number=count,
                    text=text,
                )
            )
    return chunks


def _chunk_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            chunks.append(current)
            current = _overlap_tail(current, overlap_chars)

        if len(paragraph) > max_chars:
            chunks.extend(_split_long_paragraph(paragraph, max_chars, overlap_chars))
            current = ""
        else:
            current = f"{current}\n\n{paragraph}".strip() if current else paragraph

    if current:
        chunks.append(current)

    return chunks


def _split_long_paragraph(paragraph: str, max_chars: int, overlap_chars: int) -> list[str]:
    parts: list[str] = []
    start = 0
    while start < len(paragraph):
        end = min(start + max_chars, len(paragraph))
        parts.append(paragraph[start:end].strip())
        if end == len(paragraph):
            break
        start = max(0, end - overlap_chars)
    return [part for part in parts if part]


def _overlap_tail(text: str, overlap_chars: int) -> str:
    if overlap_chars <= 0:
        return ""
    tail = text[-overlap_chars:]
    first_space = tail.find(" ")
    return tail[first_space + 1 :].strip() if first_space >= 0 else tail.strip()
