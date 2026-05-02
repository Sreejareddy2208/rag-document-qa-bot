from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}


@dataclass(frozen=True)
class DocumentPage:
    source: str
    page: int | None
    text: str


def clean_text(text: str) -> str:
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?m)^\s*page\s+\d+\s*$", "", text, flags=re.IGNORECASE)
    return text.strip()


def load_documents(data_dir: Path) -> list[DocumentPage]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    pages: list[DocumentPage] = []
    for path in sorted(data_dir.iterdir()):
        if path.is_dir() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        pages.extend(load_document(path))

    if not pages:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"No supported documents found in {data_dir}. Supported: {supported}")
    return pages


def load_document(path: Path) -> list[DocumentPage]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path)
    if suffix in {".txt", ".md"}:
        return _load_text(path)
    if suffix == ".docx":
        return _load_docx(path)
    raise ValueError(f"Unsupported file type: {path}")


def _load_pdf(path: Path) -> list[DocumentPage]:
    reader = PdfReader(str(path))
    pages: list[DocumentPage] = []
    for index, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if text:
            pages.append(DocumentPage(source=path.name, page=index, text=text))
    return pages


def _load_text(path: Path) -> list[DocumentPage]:
    text = clean_text(path.read_text(encoding="utf-8"))
    return [DocumentPage(source=path.name, page=None, text=text)] if text else []


def _load_docx(path: Path) -> list[DocumentPage]:
    from docx import Document as DocxDocument

    document = DocxDocument(str(path))
    paragraphs = [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()]
    text = clean_text("\n\n".join(paragraphs))
    return [DocumentPage(source=path.name, page=None, text=text)] if text else []
