"""The embedding tier, adapted to the narrow protocol the pipeline asks for.

Constructed at the composition root and injected. Everything above this file sees only
`EmbeddingModel.embed`, so no caller depends on the Mistral SDK's request shape.

Two properties are load-bearing and both fail silently if broken:

- **Order.** `embed` returns one vector per input, in input order. The caller zips them
  onto chunks by position, so a batcher that reordered or dropped an input would give
  every chunk a neighbour's vector — retrieval would still work, just wrongly.
- **Width.** The vectors must be exactly as wide as the Neo4j vector index. Neo4j skips
  a node whose vector does not match the index width, with no error at all: the write
  succeeds, the chunk never appears in a search result.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Protocol

from mistralai.client import Mistral

from graph_rag.config import MistralSettings


class EmbeddingModel(Protocol):
    """Embeds texts, returning one vector per input, in input order."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...

    @property
    def dimensions(self) -> int: ...


class EmbeddingError(RuntimeError):
    """The API answered with something that cannot be trusted onto chunks."""


def batch_spans(
    texts: Sequence[str], max_count: int, max_characters: int
) -> Iterator[tuple[int, int]]:
    """`(start, stop)` slices covering `texts` once each, in order.

    Spans rather than lists of text so order and coverage are structural rather than
    something the caller has to re-establish.

    Two caps apply: the number of inputs, and their combined length. A single text over
    the character budget is still sent — on its own, so a request the API may refuse
    cannot take a batch of innocent chunks down with it.
    """
    start = 0
    characters = 0
    for position, text in enumerate(texts):
        if position > start and (
            position - start >= max_count or characters + len(text) > max_characters
        ):
            yield start, position
            start, characters = position, 0
        characters += len(text)
    if start < len(texts):
        yield start, len(texts)


class MistralEmbeddingModel:
    """`EmbeddingModel` backed by the Mistral embeddings endpoint."""

    def __init__(
        self,
        client: Mistral,
        model: str,
        dimensions: int,
        batch_size: int,
        batch_characters: int,
    ) -> None:
        self._client = client
        self._model = model
        self._dimensions = dimensions
        self._batch_size = batch_size
        self._batch_characters = batch_characters

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start, stop in batch_spans(texts, self._batch_size, self._batch_characters):
            vectors.extend(self._embed_batch(texts[start:stop]))
        return vectors

    def _embed_batch(self, batch: Sequence[str]) -> list[list[float]]:
        response = self._client.embeddings.create(model=self._model, inputs=list(batch))
        data = response.data or []
        if len(data) != len(batch):
            raise EmbeddingError(
                f"asked for {len(batch)} embeddings, got {len(data)} back"
            )
        # The API labels each vector with the position of its input. Sorting by that
        # label rather than trusting arrival order costs nothing and removes the one
        # way a batched call can misalign without anything looking wrong.
        ordered = sorted(data, key=lambda item: getattr(item, "index", 0))
        vectors = []
        for item in ordered:
            vector = item.embedding
            if not vector:
                raise EmbeddingError("the API returned an empty embedding vector")
            if len(vector) != self._dimensions:
                raise EmbeddingError(
                    f"{self._model} returned {len(vector)}-dim vectors but the graph's "
                    f"vector index is {self._dimensions}-dim. Neo4j silently ignores "
                    "vectors of the wrong width — fix EMBEDDING_DIMENSIONS and re-run "
                    "`just init`."
                )
            vectors.append([float(value) for value in vector])
        return vectors


def make_embedding_model(settings: MistralSettings) -> MistralEmbeddingModel:
    """Build the embedding model named by configuration."""
    return MistralEmbeddingModel(
        Mistral(api_key=settings.api_key),
        settings.embedding_model,
        dimensions=settings.embedding_dimensions,
        batch_size=settings.embedding_batch_size,
        batch_characters=settings.embedding_batch_characters,
    )
