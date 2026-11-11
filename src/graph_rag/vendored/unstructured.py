"""Unstructured-backed parsing provider.

Converts documents into an ordered list of *elements* (Title, NarrativeText,
Table, Image, …) via the Unstructured API. The default transport is
:class:`~providers.parsing.unstructured_http.UnstructuredHTTPClient`.

---

VENDORED — not demo-authored code.

Origin: ``apps/ai/providers/parsing/unstructured.py`` in the sibling ``monorepo`` checkout (the production ingestion
service). Copied wholesale, with that system's settings/DI framework stripped; every
deviation from the original is marked ``LOCAL CHANGE``.

Do not improve this file. Its value is that behaviour here matches production, so the
demo predicts the real thing; edits silently decouple the two.
"""
from typing import Protocol
from uuid import uuid4

from graph_rag.vendored.models import Document, DocumentMetadata, RawDocument
from graph_rag.vendored.normalize import normalize_elements
from graph_rag.vendored.errors import ProviderError

__all__ = ["UnstructuredClient", "UnstructuredProvider"]


class UnstructuredClient(Protocol):
    """Transport that turns a raw document into Unstructured element dicts."""

    async def partition(self, raw: RawDocument) -> list[dict]:
        """Submit ``raw`` to Unstructured and return its element dicts."""


class UnstructuredProvider:
    """Parse and normalise documents using an Unstructured backend."""

    def __init__(self, client: UnstructuredClient) -> None:
        self._client = client

    async def parse(self, raw: RawDocument, document_id: str | None = None) -> Document:
        try:
            elements = await self._client.partition(raw)
        except ProviderError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalise all backend failures
            raise ProviderError(
                f"Unstructured parsing failed for {raw.filename!r}: {exc}"
            ) from exc

        if not isinstance(elements, list):
            raise ProviderError(
                "Unstructured returned an unexpected payload: "
                f"expected a list of elements, got {type(elements).__name__}"
            )

        metadata = raw.metadata or DocumentMetadata(source=raw.filename)
        return normalize_elements(
            elements,
            # LOCAL CHANGE: the origin always minted a fresh uuid4 here, because the
            # document id came from the surrounding pipeline. Chunk ids are derived
            # from it, and this project needs them stable across re-ingestion, so the
            # caller may supply the identity instead.
            document_id=document_id or str(uuid4()),
            metadata=metadata,
        )
