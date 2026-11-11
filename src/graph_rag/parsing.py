"""The input side of the ingestion pipeline: a PDF on disk to ordered chunks.

This is the project's own code; the work is done by the vendored production modules it
wires together. The parser client is a protocol satisfied by
`SelfHostedUnstructuredHTTPClient` and injected, never constructed inside a helper — the
pipeline seam the README describes.
"""

from __future__ import annotations

import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from uuid import NAMESPACE_URL, uuid5

import httpx

from graph_rag.cache import CachingUnstructuredClient, DiskCache, parsing_signature
from graph_rag.config import ChunkingSettings, ParsingSettings
from graph_rag.vendored.chunking import BasicChunker, Chunker, TitleChunker
from graph_rag.vendored.models import Chunk, Document, DocumentMetadata, RawDocument
from graph_rag.vendored.unstructured import UnstructuredClient, UnstructuredProvider
from graph_rag.vendored.unstructured_self_hosted_http import (
    SelfHostedUnstructuredHTTPClient,
)

DEFAULT_MIME_TYPE = "application/pdf"


class DocumentParser(Protocol):
    """Turns a raw document into an ordered, normalised `Document`."""

    async def parse(
        self, raw: RawDocument, document_id: str | None = None
    ) -> Document: ...


def document_id_for(path: Path) -> str:
    """Stable identity for a document on disk.

    Derived from the resolved path rather than the file's content hash: two byte-identical
    reports filed under two different clients are *different documents* in this corpus,
    and a content hash would give them the same chunk ids and silently collapse them.
    """
    return str(uuid5(NAMESPACE_URL, f"urn:document:{path.resolve()}"))


def make_chunker(settings: ChunkingSettings) -> Chunker:
    """Build the chunker named by configuration.

    Replaces the vendored `make_chunker(Settings)` that read the monorepo's settings
    object; the parameters it passes are the same.
    """
    strategy = settings.strategy.lower()
    common = {
        "max_characters": settings.max_characters,
        "new_after_n_chars": settings.new_after_n_chars,
        "overlap": settings.overlap,
    }
    if strategy == "by_title":
        return TitleChunker(**common, combine_under_n_chars=settings.combine_under_n_chars)
    if strategy == "basic":
        return BasicChunker(**common)
    raise ValueError(f"Unknown chunk strategy: {settings.strategy!r}")


def make_parser(
    settings: ParsingSettings,
    http_client: httpx.AsyncClient | None = None,
    cache: DiskCache | None = None,
) -> DocumentParser:
    """Construct the self-hosted Unstructured parser. Called at the composition root.

    The cache wraps the *transport*, inside the provider: a parse is a pure function of
    the file's bytes and the parsing configuration only once the document's identity
    has been taken out of it. See `cache.CachingUnstructuredClient`.
    """
    client: UnstructuredClient = SelfHostedUnstructuredHTTPClient(
        base_url=settings.url,
        strategy=settings.strategy,
        image_block_types=settings.image_block_types,
        languages=settings.languages,
        timeout=settings.timeout_seconds,
        max_retries=settings.max_retries,
        http_client=http_client,
    )
    if cache is not None:
        client = CachingUnstructuredClient(
            client, cache, signature=parsing_signature(settings)
        )
    return UnstructuredProvider(client)


def read_document(path: Path) -> RawDocument:
    mime_type, _ = mimetypes.guess_type(path.name)
    return RawDocument(
        content=path.read_bytes(),
        filename=path.name,
        mime_type=mime_type or DEFAULT_MIME_TYPE,
        metadata=DocumentMetadata(source=str(path)),
    )


@dataclass(frozen=True)
class ParsedDocument:
    path: Path
    document: Document
    chunks: list[Chunk]


async def parse_file(path: Path, parser: DocumentParser, chunker: Chunker) -> ParsedDocument:
    """Parse one file into its ordered chunks. Both dependencies are injected."""
    document = await parser.parse(read_document(path), document_id=document_id_for(path))
    return ParsedDocument(path=path, document=document, chunks=chunker.chunk(document))
