"""The content half of ingestion: discovered files to chunks in the graph.

The structural pass (ticket 04) has already written a Document node per file. This
module fills them in: parse, chunk, embed, write — per document, so a corpus of a
hundred PDFs makes progress visible and a single unreadable file costs one document
rather than the whole run.

Every external dependency is injected as a protocol: the parser, the chunker, the
embedding model and the graph driver. Nothing here constructs one.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from neo4j import Driver

from graph_rag.chunks import embeddable, page_count, to_chunk_records
from graph_rag.embedding import EmbeddingModel
from graph_rag.graph.chunks import write_chunks
from graph_rag.parsing import DocumentParser, parse_file
from graph_rag.vendored.chunking import Chunker


@dataclass(frozen=True)
class DocumentContent:
    path: Path
    chunks: int
    pages: int
    #: Chunks left over from a previous, longer version of this document.
    removed: int
    #: Why this document has no content, when it has none. Reported, not raised: one
    #: unreadable PDF must not discard the work done on the other ninety-nine.
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass(frozen=True)
class ContentReport:
    documents: list[DocumentContent]

    @property
    def chunks(self) -> int:
        return sum(document.chunks for document in self.documents)

    @property
    def removed(self) -> int:
        return sum(document.removed for document in self.documents)

    @property
    def failures(self) -> list[DocumentContent]:
        return [document for document in self.documents if not document.ok]


async def ingest_document(
    path: Path,
    *,
    parser: DocumentParser,
    chunker: Chunker,
    embedder: EmbeddingModel,
    driver: Driver,
    database: str,
) -> DocumentContent:
    """Parse, embed and write one document's chunks.

    Embedding and the graph write are synchronous clients, so they are handed to a
    thread rather than run on the event loop — otherwise one document's embedding call
    would stall every other document's parse.
    """
    parsed = await parse_file(path, parser, chunker)
    chunks = embeddable(parsed.chunks)
    texts = [chunk.text for chunk in chunks]
    embeddings = await asyncio.to_thread(embedder.embed, texts) if texts else []
    records = to_chunk_records(chunks, embeddings)
    pages = page_count(parsed.document)
    report = await asyncio.to_thread(
        write_chunks, driver, database, str(path), records, pages
    )
    return DocumentContent(
        path=path, chunks=report.written, pages=pages, removed=report.removed
    )


async def ingest_contents(
    paths: Sequence[Path],
    *,
    parser: DocumentParser,
    chunker: Chunker,
    embedder: EmbeddingModel,
    driver: Driver,
    database: str,
    concurrency: int,
    on_document: Callable[[DocumentContent], None] | None = None,
) -> ContentReport:
    """Fill in every discovered document, a bounded number at a time.

    Results come back in the order `paths` were given, whatever order they finished in,
    so a re-run over an unchanged corpus reports the same thing twice.
    """
    limit = asyncio.Semaphore(concurrency)

    async def one(path: Path) -> DocumentContent:
        async with limit:
            try:
                content = await ingest_document(
                    path,
                    parser=parser,
                    chunker=chunker,
                    embedder=embedder,
                    driver=driver,
                    database=database,
                )
            except Exception as error:  # reported per document, not raised
                content = DocumentContent(
                    path=path, chunks=0, pages=0, removed=0, error=str(error)
                )
            if on_document is not None:
                on_document(content)
            return content

    documents = await asyncio.gather(*(one(path) for path in paths))
    return ContentReport(documents=list(documents))
