"""The ingestion cache, where only one of the two failure modes is silent.

A cache that misses when it should hit costs money and minutes, and says so: the run
takes ten times as long and reports the calls it made. Nothing here tests for that.

A cache that *hits* when it should miss is the other kind. It answers with another
text's embedding, another chunk's extraction, or the parse of a file that has since
changed, and every one of those writes a graph that looks entirely plausible and is
wrong. These tests pin that: what the key must separate, and the one piece of
bookkeeping that can shift a vector onto a neighbour's chunk.

Per the project's testing posture, that criterion is the whole scope — there is no test
here that a payload comes back out of a file it was put into.
"""

import asyncio
from pathlib import Path

from graph_rag.cache import (
    CachingChatModel,
    CachingEmbeddingModel,
    CachingUnstructuredClient,
    DiskCache,
)
from graph_rag.vendored.models import DocumentMetadata, RawDocument



def test_clearing_empties_the_cache_but_touches_nothing_else(tmp_path: Path):
    """`--no-cache` must genuinely invalidate — and must not become `rm -rf` on
    whatever a mis-set CACHE_DIR points at."""
    bystander = tmp_path / "notes.md"
    bystander.write_text("not the cache's", encoding="utf-8")
    cache = DiskCache(tmp_path)
    cache.put("parse", "aaa", [{"text": "Projektbericht"}])
    cache.put("completion", "bbb", "Zusammenfassung")

    cache.clear()

    assert cache.get("parse", "aaa") is None
    assert cache.get("completion", "bbb") is None
    assert bystander.read_text(encoding="utf-8") == "not the cache's"


class FakeEmbeddingModel:
    """An embedding model that reports exactly which texts it was asked for."""

    def __init__(self, dimensions: int = 3) -> None:
        self._dimensions = dimensions
        self.calls: list[list[str]] = []

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, texts):
        self.calls.append(list(texts))
        # A vector that identifies its own text, so a misaligned pairing is visible.
        return [[float(len(text)), float(ord(text[0])), 0.5] for text in texts]


def test_a_partially_cached_batch_keeps_every_vector_on_its_own_text(tmp_path: Path):
    """The silent failure this cache could introduce: the uncached texts come back
    from the API as a shorter list, and reassembling it by position shifts every
    vector onto a neighbour's chunk. Retrieval still works — it simply answers from
    the wrong passage, which is this project's whole thesis failing quietly."""
    cache = DiskCache(tmp_path)
    model = FakeEmbeddingModel()
    expected = model.embed(["alpha", "beta", "gamma", "delta"])
    model.calls.clear()
    warm = CachingEmbeddingModel(model, cache, signature="mistral-embed/3")
    warm.embed(["beta", "delta"])
    model.calls.clear()

    vectors = warm.embed(["alpha", "beta", "gamma", "delta"])

    assert vectors == expected
    assert model.calls == [["alpha", "gamma"]]


def test_a_different_embedding_model_does_not_answer_from_the_old_one(tmp_path: Path):
    """The model identifier is part of the key. Vectors from two models are not
    comparable, and a graph holding both is silently unsearchable."""
    cache = DiskCache(tmp_path)
    model = FakeEmbeddingModel()
    CachingEmbeddingModel(model, cache, signature="mistral-embed/3").embed(["alpha"])
    model.calls.clear()

    CachingEmbeddingModel(model, cache, signature="other-embed/3").embed(["alpha"])

    assert model.calls == [["alpha"]]


class FakeChatModel:
    """A chat model that records every prompt it was actually asked."""

    def __init__(self, reply: str = "{}") -> None:
        self.reply = reply
        self.calls: list[tuple[str, str, bool]] = []

    def complete(self, system: str, user: str, *, json_object: bool = False) -> str:
        self.calls.append((system, user, json_object))
        return self.reply


def test_an_edited_prompt_invalidates_itself(tmp_path: Path):
    """The system prompt is part of the key, so iterating on an extraction prompt
    re-extracts without needing the cache cleared — and, far more importantly, cannot
    answer a new prompt with the old prompt's reply."""
    cache = DiskCache(tmp_path)
    model = FakeChatModel()
    cached = CachingChatModel(model, cache, signature="mistral-small-latest")
    cached.complete("Regeln", "Abschnitt A")
    model.calls.clear()

    cached.complete("Regeln, überarbeitet", "Abschnitt A")

    assert model.calls == [("Regeln, überarbeitet", "Abschnitt A", False)]


def test_two_chunks_that_differ_are_extracted_separately(tmp_path: Path):
    cache = DiskCache(tmp_path)
    model = FakeChatModel()
    cached = CachingChatModel(model, cache, signature="mistral-small-latest")
    cached.complete("Regeln", "Abschnitt A")
    model.calls.clear()

    cached.complete("Regeln", "Abschnitt B")

    assert model.calls == [("Regeln", "Abschnitt B", False)]


def test_an_empty_reply_is_not_pinned_for_good(tmp_path: Path):
    """An empty answer is a transient failure, not a result. Stored, it would become
    permanent: the document keeps no summary and no re-run can produce one."""
    cache = DiskCache(tmp_path)
    model = FakeChatModel(reply="")
    cached = CachingChatModel(model, cache, signature="mistral-large-latest")
    cached.complete("Fasse zusammen", "Bericht")
    model.calls.clear()
    model.reply = "Der Bericht beschreibt…"

    assert cached.complete("Fasse zusammen", "Bericht") == "Der Bericht beschreibt…"


class FakeUnstructuredClient:
    """A parsing transport that records the documents it was actually sent."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def partition(self, raw: RawDocument) -> list[dict]:
        self.calls.append(raw.filename)
        return [{"type": "Title", "text": raw.content.decode("utf-8")}]


def _raw(filename: str, content: str) -> RawDocument:
    return RawDocument(
        content=content.encode("utf-8"),
        filename=filename,
        mime_type="application/pdf",
        metadata=DocumentMetadata(source=filename),
    )


def test_an_unchanged_file_is_parsed_once_and_a_changed_one_is_reparsed(tmp_path: Path):
    """Content hash, not path: the acceptance criterion is that changing one
    document's content reprocesses exactly that document."""
    client = FakeUnstructuredClient()
    cached = CachingUnstructuredClient(
        client, DiskCache(tmp_path), signature="hi_res/deu,eng"
    )
    first = asyncio.run(cached.partition(_raw("bericht.pdf", "Projektbericht 2023")))
    client.calls.clear()

    again = asyncio.run(cached.partition(_raw("bericht.pdf", "Projektbericht 2023")))
    changed = asyncio.run(cached.partition(_raw("bericht.pdf", "Projektbericht 2024")))

    assert again == first
    assert changed == [{"type": "Title", "text": "Projektbericht 2024"}]
    assert client.calls == ["bericht.pdf"]


def test_changed_parsing_settings_are_not_answered_from_the_old_ones(tmp_path: Path):
    """A different strategy or language set is a different parse of the same bytes."""
    cache = DiskCache(tmp_path)
    client = FakeUnstructuredClient()
    asyncio.run(
        CachingUnstructuredClient(client, cache, signature="hi_res/deu,eng").partition(
            _raw("bericht.pdf", "Projektbericht")
        )
    )
    client.calls.clear()

    asyncio.run(
        CachingUnstructuredClient(client, cache, signature="fast/deu,eng").partition(
            _raw("bericht.pdf", "Projektbericht")
        )
    )

    assert client.calls == ["bericht.pdf"]
