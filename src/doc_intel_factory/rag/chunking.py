"""Simple text chunking for retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..synthetic.corpus import Document


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_type: str
    text: str
    start: int
    end: int


def chunk_text(text: str, *, size: int = 180, overlap: int = 40) -> list[tuple[str, int, int]]:
    if not text:
        return []
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        end = min(n, i + size)
        # prefer break at newline/space
        if end < n:
            br = text.rfind("\n", i + size // 2, end)
            if br <= i:
                br = text.rfind(" ", i + size // 2, end)
            if br > i:
                end = br
        piece = text[i:end].strip()
        if piece:
            chunks.append((piece, i, end))
        if end >= n:
            break
        i = max(i + 1, end - overlap)
    return chunks


def chunk_documents(docs: Iterable[Document], *, size: int = 180, overlap: int = 40) -> list[Chunk]:
    out: list[Chunk] = []
    for doc in docs:
        for j, (piece, start, end) in enumerate(chunk_text(doc.text, size=size, overlap=overlap)):
            out.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}::c{j}",
                    doc_id=doc.doc_id,
                    doc_type=doc.doc_type,
                    text=piece,
                    start=start,
                    end=end,
                )
            )
    return out
