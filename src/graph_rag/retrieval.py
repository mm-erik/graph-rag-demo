"""The lexical-and-semantic retrieval leg: vector search and full-text search, fused.

Both, not either. Every document in this corpus is near-identical boilerplate, so every
chunk embeds to nearly the same point and dense retrieval is weak here *by construction*
— a distinctive literal token, a project name or a person, is often the only signal that
separates the right client's report from four lookalikes. The full-text index is what
rescues those questions, and dropping it for simplicity would quietly turn "Neo4j only"
into "dense only", which is the risk the README names.

Fusing on *rank* rather than on score is what makes that possible at all: a cosine
similarity and a Lucene BM25 score share no scale, and no weighting of the two numbers
means anything.

This module is pure. The searches are a protocol, implemented against Neo4j in
`graph/retrieval.py`, and the embedder is injected.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from graph_rag.ancestry import Ancestor

#: Results per leg. Ten is enough to show fusion changing the order and few enough to
#: read off a screen; it is a query parameter so it can be turned up mid-demo.
DEFAULT_TOP_K = 10

#: The constant from the original reciprocal-rank-fusion paper. It damps the difference
#: between the top ranks, which is what stops either leg's first place from dominating
#: a chunk both legs agree on.
RRF_K = 60


@dataclass(frozen=True)
class Fused:
    key: str
    score: float
    #: 1-based rank per leg, holding only the legs that returned this key at all.
    ranks: Mapping[str, int]


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[str]], *, k: int = RRF_K
) -> list[Fused]:
    """Fuse ranked lists of keys into one order, best first.

    **Ties keep the order `rankings` put them in.** Two keys each ranked first by one
    leg and absent from the other score identically, and the formula has nothing
    further to say about them — so a caller that cares which wins passes the deciding
    leg first. `Mode.legs` is where this project decides that.
    """
    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}
    for leg, keys in rankings.items():
        for position, key in enumerate(keys, start=1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + position)
            ranks.setdefault(key, {})[leg] = position
    return [
        Fused(key=key, score=score, ranks=ranks[key])
        for key, score in sorted(scores.items(), key=lambda item: -item[1])
    ]


@dataclass(frozen=True)
class Chunk:
    """One chunk of a document, with the structure around it.

    Defined here rather than next to its Cypher so the leg's logic — and its tests —
    stay free of the graph driver. `graph/retrieval.py` builds these.
    """

    id: str
    document_path: str
    filename: str
    index: int
    page: int | None
    text: str
    #: Outermost first, excluding the corpus root itself.
    ancestry: list[Ancestor]


@dataclass(frozen=True)
class ChunkHit:
    """A chunk as one search ranked it.

    The score is kept *outside* the chunk, because it belongs to the search rather
    than to the chunk: it is a cosine similarity in one leg and a Lucene BM25 score in
    the other, comparable within a leg and meaningless across them. A fused result
    therefore carries a `Chunk` and no index score at all — there is no honest single
    value to put there, and a field holding whichever leg happened to be read last
    would be a number later tickets could quietly trust.
    """

    chunk: Chunk
    score: float


class ChunkIndex(Protocol):
    """The two searches over chunks, each returning its own ranked list."""

    def by_vector(self, vector: list[float], limit: int) -> list[ChunkHit]: ...

    def by_text(self, query: str, limit: int) -> list[ChunkHit]: ...


class Embedder(Protocol):
    """The embedding tier, narrowed to the one call a question makes."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class Leg(StrEnum):
    """The two searches. These are what a result's per-leg ranks are keyed on and what
    the CLI prints, so they are named once here and nowhere else."""

    FULLTEXT = "fulltext"
    VECTOR = "vector"


class Mode(StrEnum):
    """Which searches run. Isolation is a demo feature, not only a debugging aid: the
    case for the lexical leg is made by showing the same question answered without it."""

    HYBRID = "hybrid"
    VECTOR = "vector"
    FULLTEXT = "fulltext"

    @property
    def legs(self) -> tuple[Leg, ...]:
        """The searches this mode runs, **in the order that breaks a tie**.

        The order is load-bearing and lives here so that it is one readable line
        rather than an emergent property of where two `if` blocks happen to sit.
        Ties are common and arbitrary — two chunks each ranked first by one leg and
        absent from the other score identically, and the formula has nothing further
        to say about them — so something has to break them, and the choice should be
        the one this corpus argues for. A full-text hit is a literal token the
        document actually contains; a vector hit on near-identical boilerplate is a
        similarity of 0.93 against a field of 0.92s. Measured: "Welche Risiken nennt
        der Bericht zu Projekt Taktzeit?" is ranked first by both legs — by the vector
        index with a chunk from Projekt *Atlas*, and by the full-text index with
        Projekt Taktzeit.
        """
        if self is Mode.HYBRID:
            return (Leg.FULLTEXT, Leg.VECTOR)
        return (Leg(self.value),)

    def runs(self, leg: Leg) -> bool:
        return leg in self.legs


#: Characters Lucene reads as syntax. Escaped rather than deleted: `-` and `/` sit
#: inside exactly the tokens this leg exists to catch — `Atlas-Referenzarchitektur`,
#: `SAP/4HANA` — and deleting them would run the two halves together into a word that
#: appears in no document.
_LUCENE_SPECIAL = set('+-&|!(){}[]^"~*?:\\/')


def lucene_query(question: str) -> str:
    """A natural-language question as a query the full-text index will accept.

    Questions arrive with the punctuation of German prose, and Lucene reads half of it
    as syntax: an unescaped `?`, `!` or `:` is a parse error and a leading `-` silently
    inverts a term into an exclusion. Neither can be allowed to reach the index — the
    first takes the leg down mid-demo, the second removes the right answer and reports
    nothing.

    Terms are left to the index's German analyser rather than filtered here: it already
    strips German stopwords and stems what remains, so a hand-maintained stopword list
    would only be a second, worse copy of that. Terms are implicitly OR-ed, so a long
    question is not required to match in full — the rarer the term, the more it scores.
    """
    terms = []
    for word in question.split():
        if not any(character.isalnum() for character in word):
            continue
        terms.append(
            "".join(
                f"\\{character}" if character in _LUCENE_SPECIAL else character
                for character in word
            )
        )
    return " ".join(terms)


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk in the leg's fused order, carrying how each search ranked it."""

    chunk: Chunk
    score: float
    #: 1-based rank per leg, holding only the legs that returned this chunk at all.
    ranks: Mapping[Leg, int]

    def rank_in(self, leg: Leg) -> int | None:
        """Where `leg` placed this chunk, or None if that leg never returned it.

        The honest answer to "would dense retrieval alone have found this?", which on
        this corpus is the question the whole lexical leg is here to answer.
        """
        return self.ranks.get(leg)


@dataclass(frozen=True)
class LegResult:
    """Everything the leg did, not only what it concluded.

    The intermediate lists are kept because the leg has to be inspectable on its own:
    seeing that a chunk was eighth by vector and first by full text is what makes the
    case for fusing them.
    """

    question: str
    #: What the full-text index was actually asked, after escaping. Empty when the
    #: question held no searchable term, or when that leg was not run.
    query: str
    #: What each leg that ran returned, in its own order. A leg that did not run is
    #: absent rather than empty — "found nothing" and "was not asked" are different
    #: answers to "is the lexical index working?".
    hits: Mapping[Leg, list[ChunkHit]]
    #: Those lists fused by reciprocal rank fusion, best first.
    results: list[RetrievedChunk]

    def found_by(self, leg: Leg) -> list[ChunkHit]:
        return self.hits.get(leg, [])


def search_chunks(
    *,
    question: str,
    index: ChunkIndex,
    embedder: Embedder,
    top_k: int = DEFAULT_TOP_K,
    mode: Mode = Mode.HYBRID,
) -> LegResult:
    """Retrieve chunks for a question by semantic and lexical search, fused.

    `top_k` is per leg, so hybrid mode considers up to twice that many chunks; capping
    the combined context is the next leg's job, not this one's.

    The index and the embedder are injected. Nothing here constructs a client.
    """
    query = lucene_query(question) if mode.runs(Leg.FULLTEXT) else ""
    searches = {
        Leg.FULLTEXT: lambda: index.by_text(query, top_k) if query else [],
        Leg.VECTOR: lambda: index.by_vector(embedder.embed([question])[0], top_k),
    }
    hits = {leg: searches[leg]() for leg in mode.legs}

    # One `Chunk` per id however many legs found it: the copies are identical, the
    # per-leg score having been left behind on the `ChunkHit` that carried it.
    by_id = {hit.chunk.id: hit.chunk for found in hits.values() for hit in found}
    rankings = {leg: [hit.chunk.id for hit in found] for leg, found in hits.items()}
    return LegResult(
        question=question,
        query=query,
        hits=hits,
        results=[
            RetrievedChunk(chunk=by_id[fused.key], score=fused.score, ranks=fused.ranks)
            for fused in reciprocal_rank_fusion(rankings)
        ],
    )
