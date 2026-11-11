"""Paying for an external call once, keyed by everything that call depends on.

An ingest of the demo corpus is minutes of layout analysis and a model call per chunk.
Almost none of that work changes between two runs, and all of the expensive parts of it
are already behind injected protocols — the parsing transport, the embedding model, the
chat model. So the cache is not a stage of the pipeline: it is three decorators wrapped
around those protocols where they are constructed, which is the composition root and the
factory in `parsing.py` next to it. No *pipeline* function knows it exists — `ingestion`,
`summaries` and `entities` still see only the protocols they were always given.

Two rules shape what is stored.

**Cache the raw result of the external call, never the artefact derived from it.** The
elements the parser returned, not the chunks; the model's reply, not the entities parsed
out of it. Everything downstream of the call is pure and cheap, so it re-runs on every
cache hit — which means changing the chunk size, the evidence limit or the ambiguity
rule costs nothing and needs no cache bust, and a stale cache cannot pin an old version
of that logic.

**The key is every input the call depends on, and nothing else.** A key that forgets an
input is the dangerous defect in this file: it does not fail, it answers with the result
of a different call. Because the system prompt is part of the key, an edited extraction
prompt invalidates its own entries; because the content hash is part of the key, a
changed document re-parses and an unchanged one never does.

The cache is content-addressed on disk rather than in the graph. The graph is dropped
and rebuilt by `just init`; the point of this file is that doing so costs seconds rather
than a re-ingest.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from shutil import rmtree
from typing import Protocol

from graph_rag.config import MistralSettings, ParsingSettings
from graph_rag.embedding import EmbeddingError, EmbeddingModel
from graph_rag.vendored.models import RawDocument
from graph_rag.vendored.unstructured import UnstructuredClient


class ChatModel(Protocol):
    """The widest of this project's three chat-model protocols.

    `summaries.ChatModel`, `entities.ExtractionModel` and
    `resolution.AdjudicationModel` are each a narrowing of this one, so a single
    decorator satisfies all three and every model call in the pipeline goes through
    one cache.
    """

    def complete(self, system: str, user: str, *, json_object: bool = False) -> str: ...

#: Bumped when the shape of a stored payload changes, so entries written by an older
#: version are missed rather than misread.
CACHE_VERSION = "1"

#: The namespaces, which are also the only directories `clear` will remove.
PARSE = "parse"
EMBED = "embed"
COMPLETION = "completion"
NAMESPACES: tuple[str, ...] = (PARSE, EMBED, COMPLETION)


def _is_reply(payload: object) -> bool:
    """A usable cached completion: a string with something in it. An empty one is a
    transient failure that was never stored, or an entry from before that rule."""
    return isinstance(payload, str) and bool(payload)


def _is_elements(payload: object) -> bool:
    return isinstance(payload, list)


def cache_key(*parts: str) -> str:
    """The key for one external call, over the exact inputs it depends on.

    Parts are separated, so `("ab", "c")` and `("a", "bc")` are different calls rather
    than the same one — a collision between two prompts that happened to concatenate
    alike would be answered from the cache and never noticed.
    """
    hasher = sha256()
    for part in parts:
        hasher.update(part.encode("utf-8"))
        hasher.update(b"\x00")
    return hasher.hexdigest()


@dataclass(frozen=True)
class CacheStats:
    """What one namespace did this run. Reported, because a cache that has quietly
    stopped hitting looks exactly like a cache that is working until the bill arrives."""

    hits: int
    misses: int


class DiskCache:
    """A content-addressed store under one directory.

    Payloads are JSON, written atomically: a run interrupted mid-write would otherwise
    leave a truncated entry that reads back as a shorter embedding or a half a reply.
    """

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._lock = threading.Lock()
        self._stats: dict[str, CacheStats] = {}

    @property
    def root(self) -> Path:
        return self._root

    def get(
        self, namespace: str, key: str, usable: Callable[[object], bool] = bool
    ) -> object | None:
        """The stored payload, or None when there is none this caller can use.

        An unreadable or corrupt entry is a miss, not an error: the remedy for both is
        to make the call again, and an ingest that aborted over a damaged cache file
        would be a cache that costs more than it saves.

        `usable` is the caller's own test of the payload — a vector of the right width,
        a reply that is a non-empty string. It belongs here rather than at the call site
        because this is where a hit is counted: an entry that parsed as JSON but is then
        rejected upstairs is a *miss*, and counting it as a hit would inflate exactly
        the number the run's cache report exists to make trustworthy.
        """
        try:
            payload = json.loads(self._path(namespace, key).read_text("utf-8"))
        except (OSError, ValueError):
            self._count(namespace, hit=False)
            return None
        if not usable(payload):
            self._count(namespace, hit=False)
            return None
        self._count(namespace, hit=True)
        return payload

    def put(self, namespace: str, key: str, payload: object) -> None:
        """Store `payload`, or give up quietly.

        A cache that cannot be written — a full disk, a read-only directory — must not
        fail an ingest whose real work succeeded. The next run simply pays again.
        """
        path = self._path(namespace, key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
            try:
                with os.fdopen(handle, "w", encoding="utf-8") as file:
                    json.dump(payload, file)
                os.replace(temporary, path)
            except BaseException:
                Path(temporary).unlink(missing_ok=True)
                raise
        except (OSError, TypeError, ValueError):
            return

    def clear(self, *namespaces: str) -> None:
        """Delete what this cache has written, in the namespaces named — all by default.

        Narrowable because the namespaces cost wildly different amounts to refill:
        re-extracting is an afternoon's iteration, while re-parsing the corpus is
        twenty minutes of layout analysis that nobody asked for. A command that only
        wants fresh model replies says so.

        Only namespace directories are removed, never the root itself: `--no-cache`
        must not become `rm -rf` on whatever a mis-set `CACHE_DIR` points at.
        """
        for namespace in namespaces or NAMESPACES:
            rmtree(self._root / namespace, ignore_errors=True)

    def stats(self) -> dict[str, CacheStats]:
        """Hits and misses per namespace that saw any traffic this run."""
        with self._lock:
            return dict(sorted(self._stats.items()))

    def _path(self, namespace: str, key: str) -> Path:
        # Sharded on the first byte of the key: a corpus of a hundred documents is a
        # few thousand chunks, and that many entries in one directory makes every
        # listing of it slow for no reason.
        return self._root / namespace / key[:2] / f"{key}.json"

    def _count(self, namespace: str, *, hit: bool) -> None:
        # Locked because the parse, embedding and extraction passes all run their calls
        # across threads, and a lost increment here would misreport the one number that
        # says whether the cache is working.
        with self._lock:
            counts = self._stats.get(namespace, CacheStats(hits=0, misses=0))
            self._stats[namespace] = CacheStats(
                hits=counts.hits + hit, misses=counts.misses + (not hit)
            )


class CachingEmbeddingModel:
    """`EmbeddingModel` that embeds a given text once, ever.

    Per text rather than per batch: the batches a corpus produces differ between runs
    as soon as one document changes, and a batch-keyed cache would miss every one of
    them. A text's vector depends on the text and the model and on nothing else, so
    that is the key.

    The bookkeeping below is the delicate part. The wrapped model is handed only the
    texts that missed, and so answers with a *shorter* list than it was asked about;
    putting those answers back in the right places is what keeps each vector on its own
    chunk. Getting it wrong does not raise — every chunk simply carries a neighbour's
    vector, and retrieval answers confidently from the wrong passage.
    """

    def __init__(
        self, model: EmbeddingModel, cache: DiskCache, signature: str
    ) -> None:
        self._model = model
        self._cache = cache
        self._signature = signature

    @property
    def dimensions(self) -> int:
        return self._model.dimensions

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        keys = [cache_key(CACHE_VERSION, EMBED, self._signature, text) for text in texts]
        # Keyed by position throughout, and read back out by position at the end: the
        # one arrangement in which a vector cannot drift onto a neighbour's text.
        found = {
            position: vector
            for position, key in enumerate(keys)
            if (vector := self._cached(key)) is not None
        }
        missing = [position for position in range(len(texts)) if position not in found]
        if missing:
            fresh = self._model.embed([texts[position] for position in missing])
            if len(fresh) != len(missing):
                # Checked rather than zipped, as in `to_chunk_records`: `zip` would
                # truncate and leave the tail of the batch holding each other's
                # vectors, which is exactly the misalignment this class must not
                # introduce.
                raise EmbeddingError(
                    f"asked for {len(missing)} embeddings, got {len(fresh)} back"
                )
            for position, vector in zip(missing, fresh, strict=True):
                found[position] = vector
                self._cache.put(EMBED, keys[position], vector)
        return [found[position] for position in range(len(texts))]

    def _cached(self, key: str) -> list[float] | None:
        """A stored vector, if it is one of the right width.

        The width is already in the signature, so a mismatch here means a foreign or
        damaged entry — and a vector of the wrong width is the one thing Neo4j accepts
        and then silently omits from every search.
        """
        payload = self._cache.get(EMBED, key, self._is_vector)
        return [float(value) for value in payload] if payload is not None else None

    def _is_vector(self, payload: object) -> bool:
        return (
            isinstance(payload, list)
            and len(payload) == self._model.dimensions
            and all(isinstance(value, (int, float)) for value in payload)
        )


class CachingChatModel:
    """A chat model that answers a repeated prompt from disk.

    One class for all four call sites — extraction, summaries, level inference and the
    resolution tiebreak — because they differ only in the prompt and the tier, and both
    are in the key. Extraction is the one that matters: a call per chunk is the dominant
    cost of an ingest and the reason this file exists.

    A changed prompt therefore invalidates its own entries, with no flag and no bust.
    That is the property worth protecting here — a cache that answered a rewritten
    extraction prompt with the previous prompt's reply would make every experiment look
    like it changed nothing.
    """

    def __init__(self, model: ChatModel, cache: DiskCache, signature: str) -> None:
        self._model = model
        self._cache = cache
        self._signature = signature

    def complete(self, system: str, user: str, *, json_object: bool = False) -> str:
        key = cache_key(
            CACHE_VERSION,
            COMPLETION,
            self._signature,
            system,
            user,
            "json" if json_object else "text",
        )
        cached = self._cache.get(COMPLETION, key, _is_reply)
        if isinstance(cached, str):
            return cached
        reply = self._model.complete(system, user, json_object=json_object)
        # An empty reply is a transient failure rather than an answer. Stored, it would
        # be permanent: the document keeps no summary, the chunk yields no entities,
        # and no amount of re-running produces either.
        if reply:
            self._cache.put(COMPLETION, key, reply)
        return reply


class CachingUnstructuredClient:
    """`UnstructuredClient` that sends a given file to the parsing service once.

    Wrapped around the *transport* rather than around `UnstructuredProvider`, and that
    choice is load-bearing. The provider stamps the document's identity onto the parse,
    and chunk ids are derived from that identity — so a cache keyed above it would hand
    two byte-identical reports filed under two different clients the same chunk ids and
    silently collapse them into one document. Below it, a parse is a pure function of
    the bytes and the parsing configuration, which is all the key needs to be.

    Parsing is seconds of layout analysis per PDF and the largest non-model cost of an
    ingest; it is also what makes a re-ingest after `just init` cheap, since the graph
    can be rebuilt without the parsing service being up at all.
    """

    def __init__(
        self, client: UnstructuredClient, cache: DiskCache, signature: str
    ) -> None:
        self._client = client
        self._cache = cache
        self._signature = signature

    async def partition(self, raw: RawDocument) -> list[dict]:
        key = cache_key(
            CACHE_VERSION, PARSE, self._signature, sha256(raw.content).hexdigest()
        )
        cached = self._cache.get(PARSE, key, _is_elements)
        if isinstance(cached, list):
            return cached
        elements = await self._client.partition(raw)
        self._cache.put(PARSE, key, elements)
        return elements


def parsing_signature(settings: ParsingSettings) -> str:
    """Everything about the parsing configuration that changes the elements returned.

    A parameter missing from here is a stale parse that nothing reports: the strategy
    is changed to `hi_res`, every document answers from its `fast` parse, and the
    chunks look plausible. The service's URL is deliberately absent — moving the same
    service to another port does not change what it returns.
    """
    return "|".join(
        (
            settings.strategy,
            ",".join(settings.languages),
            ",".join(settings.image_block_types),
        )
    )


def embedding_signature(settings: MistralSettings) -> str:
    """The model and the width it was asked for — vectors from two models are not
    comparable, and a width mismatch is invisible to Neo4j."""
    return f"{settings.embedding_model}/{settings.embedding_dimensions}"
