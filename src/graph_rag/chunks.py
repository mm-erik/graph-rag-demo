"""Chunks as the graph sees them: the vendored `Chunk` plus its embedding.

The vendored chunker's `Chunk` carries metadata this project does not store and is
shaped by the production service's needs. `ChunkRecord` is the narrow record the graph
write takes, and the mapping between the two is pure so the one thing that can go
wrong here — a chunk paired with another chunk's embedding — is testable without a
model or a database.

The deterministic id is passed straight through from the chunker. It is what makes
re-ingestion an upsert rather than an accumulation, so nothing here may re-derive it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from graph_rag.vendored.models import Chunk, Document


@dataclass(frozen=True)
class ChunkRecord:
    #: Deterministic, from `(document id, index)` — see the vendored chunker.
    id: str
    text: str
    index: int
    #: The page the chunk starts on; None when the parser reported no page at all.
    page: int | None
    embedding: list[float]


class ChunkAlignmentError(RuntimeError):
    """Chunks and embeddings did not come back in matching counts."""


def page_of(chunk: Chunk) -> int | None:
    """The page a chunk starts on.

    A chunk combines whole elements and so can straddle a page break; its metadata
    lists every page it touches. The first is recorded — a citation points at where the
    passage begins.
    """
    pages = chunk.metadata.get("pages") or []
    return min(pages) if pages else None


def embeddable(chunks: Sequence[Chunk]) -> list[Chunk]:
    """The chunks worth embedding, in reading order.

    A blank chunk is rejected by the embedding API and would be dead weight in both
    indexes; the reading-order chain is built over what survives.
    """
    return [chunk for chunk in chunks if chunk.text.strip()]


def to_chunk_records(
    chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]
) -> list[ChunkRecord]:
    """Pair each chunk with its embedding, positionally.

    The count is checked rather than zipped: `zip` truncates to the shorter list, which
    would write a document's first chunks, drop the rest, and report success.
    """
    if len(chunks) != len(embeddings):
        raise ChunkAlignmentError(
            f"{len(chunks)} chunks but {len(embeddings)} embeddings — refusing to "
            "write chunks that may carry each other's vectors"
        )
    return [
        ChunkRecord(
            id=chunk.id,
            text=chunk.text,
            index=chunk.index,
            page=page_of(chunk),
            embedding=[float(value) for value in embedding],
        )
        for chunk, embedding in zip(chunks, embeddings, strict=True)
    ]


def page_count(document: Document) -> int:
    """How many pages the parser saw. Zero when it reported no page numbers."""
    return max((element.page_number or 0 for element in document.elements), default=0)
