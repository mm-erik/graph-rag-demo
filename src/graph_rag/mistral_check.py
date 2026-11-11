"""Verify the configured Mistral model identifiers against the live API.

The model table in the spec is explicitly unverified, and the vector index is sized from
the embedding model, so a silently renamed model would produce a graph that cannot be
searched. This reports discrepancies rather than guessing.
"""

from __future__ import annotations

from dataclasses import dataclass

from mistralai.client import Mistral

from graph_rag.config import MistralSettings


@dataclass(frozen=True)
class ModelCheck:
    role: str
    identifier: str
    available: bool
    note: str | None = None


@dataclass(frozen=True)
class ModelReport:
    checks: list[ModelCheck]
    embedding_dimensions: int | None
    configured_dimensions: int
    listing_error: str | None = None

    @property
    def ok(self) -> bool:
        return (
            self.listing_error is None
            and all(check.available for check in self.checks)
            and self.embedding_dimensions == self.configured_dimensions
        )


def _known_identifiers(client: Mistral) -> set[str]:
    """Every id the API lists, plus its aliases.

    A card is listed under its canonical id (`mistral-small-2506`) with the `-latest`
    pointer we configure sitting in `aliases`, so ignoring aliases would report a
    perfectly good configuration as NOT FOUND. Cards of a model type this SDK version
    does not know carry no `id` at all, so neither field may be assumed present.
    """
    identifiers: set[str] = set()
    for model in client.models.list().data or []:
        identifier = getattr(model, "id", None)
        if identifier:
            identifiers.add(identifier)
        identifiers.update(getattr(model, "aliases", None) or [])
    return identifiers


def verify_models(client: Mistral, settings: MistralSettings) -> ModelReport:
    """Check each configured identifier exists, and measure the real embedding width."""
    listing_error = None
    try:
        available = _known_identifiers(client)
    except Exception as error:  # reported, not raised — this command is a diagnostic
        available = set()
        listing_error = str(error)

    checks = [
        ModelCheck("extraction", settings.extraction_model, settings.extraction_model in available),
        ModelCheck("chat", settings.chat_model, settings.chat_model in available),
        ModelCheck("embedding", settings.embedding_model, settings.embedding_model in available),
    ]

    dimensions = None
    try:
        response = client.embeddings.create(
            model=settings.embedding_model, inputs=["Dimensionsprüfung"]
        )
        # The SDK types both as optional; a null vector here would otherwise surface as
        # "NoneType has no len()" from a command whose whole job is a clear diagnostic.
        vector = response.data[0].embedding if response.data else None
        if not vector:
            raise ValueError("the API returned no embedding vector")
        dimensions = len(vector)
    except Exception as error:  # reported, not raised — this command is a diagnostic
        checks = [
            check
            if check.role != "embedding"
            else ModelCheck(check.role, check.identifier, check.available, str(error))
            for check in checks
        ]

    return ModelReport(
        checks=checks,
        embedding_dimensions=dimensions,
        configured_dimensions=settings.embedding_dimensions,
        listing_error=listing_error,
    )
