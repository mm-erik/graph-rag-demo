"""The entity traversal retrieval leg: from names in the question to documents.

This is the leg the project exists for. The first leg finds chunks that *resemble* the
question; this one finds chunks that are *connected* to it. The target demo question —
"was verbindet das Projekt Atlas von Nordlicht mit dem von Brandt & Söhne?" — is
unanswerable by any amount of vector search, because no single chunk contains the
connection: it exists only as a path through the graph, and only because entities are
global rather than folder-owned.

Four steps:

1. **Extract** the entities the question names. One extraction-tier call per question —
   cheap, unlike the per-chunk extraction an ingest pays for.
2. **Match** them into the graph by normalised name *and* by embedding, because a
   question will not spell an entity the way the documents do.
3. **Walk** `RELATES` edges outward to a hop depth given at call time.
4. **Collect** the chunks that mention everything reached.

Everything is kept, not only the chunks: which entities matched, which edges were
walked at which hop, which chunk each reached entity led to. Tickets 14 and 15 render
that path, and a leg that returned chunks alone would have to be rewritten to produce
it.

The module is pure. The graph is a protocol, implemented against Neo4j in
`graph/traversal.py`; the embedder and the extraction model are injected.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from graph_rag.entities import ENTITY_TYPES, EntityIdentity, identify
# `DEFAULT_TOP_K` is re-exported rather than redefined: both legs return ten
# results so that ticket 12 fuses them with comparable weight, and two constants
# that have to stay equal are one that will not.
from graph_rag.retrieval import DEFAULT_TOP_K, Chunk, Embedder

# The matching stage's two thresholds are environment settings and live in `config.py`
# with every other one, so they arrive here as arguments and have no default to drift
# from. The two below are call-time knobs with no environment counterpart — the same
# split as `retrieval.DEFAULT_TOP_K`.

#: Hops walked when the caller says nothing. One hop is precise, two is where
#: cross-document connections appear, three is noise — and this is expected to be
#: turned up and down live, so it is a default rather than a constant.
DEFAULT_HOPS = 2


class Match(StrEnum):
    """How a question entity reached a graph entity. Both are possible at once."""

    NAME = "name"
    EMBEDDING = "embedding"


@dataclass(frozen=True)
class QuestionEntity:
    """An entity the question names, as the question spells it.

    The identity carries no scope: a question is asked from outside the hierarchy, so
    nothing on it says which client's `Projekt Atlas` is meant — and a question naming
    one should reach both rather than neither.
    """

    surface: str
    identity: EntityIdentity

    @property
    def type(self) -> str:
        return self.identity.type


@dataclass(frozen=True)
class GraphEntity:
    """An entity node as the graph holds it."""

    identity: EntityIdentity
    #: The name the documents write, for display.
    name: str
    mentions: int

    @property
    def scope(self) -> str:
        return self.identity.scope


@dataclass(frozen=True)
class MatchedEntity:
    """A question entity and the graph node it started the walk from."""

    question: QuestionEntity
    entity: GraphEntity
    #: Name match, embedding match, or both.
    how: tuple[Match, ...]
    #: Cosine similarity, when an embedding was involved. A name match alone has none
    #: — and a 0.0 standing in for "not measured" is a number later code could trust.
    similarity: float | None


@dataclass(frozen=True)
class Neighbour:
    """One `RELATES` edge, as the graph hands it back from a one-hop read.

    `entity` is the node the walk arrives at; `source` and `target` are the edge's own
    endpoints, which are **not** the same thing. The walk follows edges in either
    direction — a question naming a person has to reach the project they lead, and a
    question naming the project has to reach the person — but the edge itself has a
    direction and a type that only reads correctly one way round. `leitet` drawn
    backwards says the project leads the consultant.
    """

    entity: GraphEntity
    source: EntityIdentity
    target: EntityIdentity
    type: str
    evidence: str
    source_chunk: str


@dataclass(frozen=True)
class WalkedEdge:
    """An edge the traversal crossed, in its own direction, and when.

    `hop` is 1-based and is the hop on which this edge was first crossed, so the path
    can be replayed outward from the matched entities rather than guessed at from the
    node distances. `source` and `target` are the edge as the graph stores it, not the
    order the walk happened to meet it in.
    """

    source: EntityIdentity
    target: EntityIdentity
    type: str
    evidence: str
    source_chunk: str
    hop: int


@dataclass(frozen=True)
class ReachedEntity:
    """An entity the traversal arrived at, and how far out it was.

    Hop 0 is an entity the question named directly.
    """

    entity: GraphEntity
    hop: int


@dataclass(frozen=True)
class ChunkMention:
    """A chunk naming one reached entity — an id and an identity, no payload.

    Deliberately not the chunk itself. Scoring has to see *every* mention of every
    reached entity to be exact, and a cap on rows that carry chunk text and ancestry is
    a cap that truncates: a question naming two clients reads one client's mentions
    first, fills the budget, and returns an answer about that client only, with nothing
    on screen to say the other was cut. These rows are small enough to need no cap at
    all, and the text is fetched afterwards for the handful that survive scoring.
    """

    chunk_id: str
    entity: EntityIdentity


@dataclass(frozen=True)
class TraversedChunk:
    """A chunk this leg reached, carrying what led to it."""

    chunk: Chunk
    #: Every reached entity this chunk names, nearest hop first.
    via: tuple[ReachedEntity, ...]
    score: float

    @property
    def hop(self) -> int:
        """The shortest walk that reaches this chunk."""
        return min(reached.hop for reached in self.via)


class EntityGraph(Protocol):
    """The reads this leg makes over the entity graph.

    One call per hop rather than one variable-length path query, because the hop a node
    was reached at is provenance the UI renders, and recovering it from a collapsed
    path result is guesswork.
    """

    def by_name(self, names: Sequence[str]) -> list[GraphEntity]: ...

    def by_vector(
        self, vector: Sequence[float], limit: int
    ) -> list[tuple[GraphEntity, float]]: ...

    def neighbours(self, identities: Sequence[EntityIdentity]) -> list[Neighbour]: ...

    def mentions_of(
        self, identities: Sequence[EntityIdentity]
    ) -> list[ChunkMention]: ...

    def chunks(self, chunk_ids: Sequence[str]) -> list[Chunk]: ...


# Short, because this runs once per question with an audience waiting, and because the
# question is one sentence rather than a page of a report. The ingestion prompt's rules
# about evidence and relations have nothing to work with here and are left out.
QUESTION_SYSTEM_PROMPT = (
    "Du findest die benannten Entitäten in einer Frage an ein Projektarchiv einer "
    "Unternehmensberatung.\n"
    "\n"
    "Erlaubte Typen — nichts anderes ist zulässig:\n"
    "- person: benannte Menschen\n"
    "- organisation: Unternehmen, Behörden, Kliniken, Geschäftsbereiche\n"
    "- system: Software, Plattformen, Produkte, technische Standards\n"
    "- konzept: Projekte, Vorhaben, Methoden, Verfahren\n"
    "- ergebnis: Arbeitsergebnisse, Dokumente, Konzepte, Kennzahlen, Meilensteine\n"
    "- risiko: benannte Risiken, Hindernisse, offene Punkte\n"
    "\n"
    "Regeln:\n"
    "1. Nenne jede Entität genau so, wie sie in der Frage steht.\n"
    "2. Nur Eigennamen. Keine Fragewörter, keine Gattungsbegriffe ('Projekt', "
    "'Bericht', 'Risiko', 'Kunde'), keine Zeiträume, keine Beträge.\n"
    "3. Erfinde nichts. Steht kein Eigenname in der Frage, gib eine leere Liste "
    "zurück.\n"
    "\n"
    'Antworte ausschliesslich als JSON: {"entities": [{"name": "...", '
    '"type": "..."}]}'
)


def parse_question_entities(reply: str) -> list[QuestionEntity]:
    """The model's reply as entities, with everything outside the taxonomy dropped.

    Identity is assigned through the same `identify` the ingestion side uses, so a
    question's `Projekt Atlas` normalises to exactly the string the graph stored — the
    one property the name match depends on. It is called with no ancestry and no
    ambiguous names, which is what leaves a question entity unscoped.
    """
    try:
        payload = json.loads(reply)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, dict):
        return []
    found: list[QuestionEntity] = []
    seen: set[EntityIdentity] = set()
    for row in payload.get("entities") or []:
        if not isinstance(row, dict):
            continue
        surface = str(row.get("name") or "").strip()
        entity_type = str(row.get("type") or "").strip().lower()
        if not surface or entity_type not in ENTITY_TYPES:
            continue
        identity = identify(surface, entity_type, (), frozenset())
        if identity is None or identity in seen:
            continue
        seen.add(identity)
        found.append(QuestionEntity(surface=surface, identity=identity))
    return found


class QuestionModel(Protocol):
    """The extraction-tier model, narrowed to the one call a question makes."""

    def complete(self, system: str, user: str, *, json_object: bool = False) -> str: ...


def question_entities(question: str, model: QuestionModel) -> list[QuestionEntity]:
    """The entities a question names. One call."""
    return parse_question_entities(
        model.complete(QUESTION_SYSTEM_PROMPT, question, json_object=True)
    )


def match_entities(
    questioned: Sequence[QuestionEntity],
    graph: EntityGraph,
    embedder: Embedder,
    *,
    threshold: float,
    candidates: int,
) -> list[MatchedEntity]:
    """Where in the graph the question's entities are, by name and by embedding.

    **The name match ignores both scope and type.** Scope, because a question naming
    `Projekt Atlas` must reach both clients' projects — that it reaches two nodes at
    once is the demo, not a defect. Type, because the question side and the ingestion
    side are two independent extractions of one name, and an artefact one of them calls
    `ergebnis` and the other `konzept` is still the node the walk has to start from; a
    type equality check would lose exactly the bridge entity and report nothing wrong.

    The embedding match is the loose half and the dangerous one, so it is the half with
    a threshold: it starts the walk somewhere the question never named.
    """
    if not questioned:
        return []
    # Keyed on the graph's identity, so a node both halves of the match found is one
    # starting point recording both ways in — never the same walk begun twice.
    found: dict[EntityIdentity, _Partial] = {}

    def record(
        question: QuestionEntity,
        entity: GraphEntity,
        how: Match,
        similarity: float | None,
    ) -> None:
        partial = found.setdefault(entity.identity, _Partial(question, entity))
        partial.how.add(how)
        if similarity is not None and (
            partial.similarity is None or similarity > partial.similarity
        ):
            partial.similarity = similarity

    wanted = {entity.identity.normalized_name: entity for entity in questioned}
    for entity in graph.by_name(list(wanted)):
        record(wanted[entity.identity.normalized_name], entity, Match.NAME, None)

    vectors = embedder.embed([entity.surface for entity in questioned])
    for question, vector in zip(questioned, vectors, strict=True):
        for entity, similarity in graph.by_vector(vector, candidates):
            if similarity >= threshold:
                record(question, entity, Match.EMBEDDING, similarity)

    return [
        MatchedEntity(
            question=partial.question,
            entity=partial.entity,
            # In declaration order — the exact match first, then the loose one.
            how=tuple(way for way in Match if way in partial.how),
            similarity=partial.similarity,
        )
        for partial in found.values()
    ]


@dataclass
class _Partial:
    """One starting point while both halves of the match are still contributing to it."""

    question: QuestionEntity
    entity: GraphEntity
    how: set[Match] = field(default_factory=set)
    similarity: float | None = None


@dataclass(frozen=True)
class TraversalResult:
    """Everything the leg did, not only what it concluded.

    The path is the product here as much as the chunks are: ticket 14 serialises it and
    ticket 15 draws it, and the demo's whole burden of proof rests on the audience
    seeing which edges were crossed to get from the question to a document that never
    mentions it.
    """

    question: str
    hops: int
    #: What the question was read as naming.
    entities: list[QuestionEntity]
    #: Where those names landed in the graph — the walk's starting points.
    matches: list[MatchedEntity]
    #: Every entity the walk arrived at, nearest first, matched entities at hop 0.
    reached: list[ReachedEntity]
    #: Every edge crossed, in the order the hops crossed them.
    edges: list[WalkedEdge]
    #: The chunks those entities are mentioned by, best first.
    results: list[TraversedChunk]


def traverse_entities(
    *,
    question: str,
    graph: EntityGraph,
    embedder: Embedder,
    model: QuestionModel,
    hops: int = DEFAULT_HOPS,
    top_k: int = DEFAULT_TOP_K,
    threshold: float,
    candidates: int,
) -> TraversalResult:
    """Retrieve chunks for a question by walking the entity graph.

    `hops` is a parameter and not a constant anywhere below it: it is adjusted live, and
    the difference between one and two is the difference between a precise answer and a
    cross-client one.

    The graph, the embedder and the model are injected. Nothing here constructs a
    client.
    """
    questioned = question_entities(question, model)
    matches = match_entities(
        questioned, graph, embedder, threshold=threshold, candidates=candidates
    )
    reached, edges = _walk(graph, [match.entity for match in matches], hops)
    return TraversalResult(
        question=question,
        hops=hops,
        entities=questioned,
        matches=matches,
        reached=reached,
        edges=edges,
        results=_collect(graph, reached, top_k),
    )


def _walk(
    graph: EntityGraph, start: Sequence[GraphEntity], hops: int
) -> tuple[list[ReachedEntity], list[WalkedEdge]]:
    """Breadth-first out to `hops`, one graph read per hop.

    An entity already reached is never re-reached at a greater depth: the first time it
    is seen is the shortest path to it, and recording it again would both inflate the
    picture and let a dense neighbourhood revisit itself until the walk never ends.
    """
    reached = [ReachedEntity(entity=entity, hop=0) for entity in start]
    seen = {entity.identity for entity in start}
    walked: set[tuple[EntityIdentity, EntityIdentity, str]] = set()
    edges: list[WalkedEdge] = []
    frontier = list(start)
    for hop in range(1, max(hops, 0) + 1):
        if not frontier:
            break
        next_frontier: list[GraphEntity] = []
        for neighbour in graph.neighbours([entity.identity for entity in frontier]):
            # An edge between two nodes of the frontier is met from both ends, and an
            # edge back to where the walk came from is met again on every later hop.
            # Recorded once, at the earliest hop that crossed it: a subgraph holding
            # the same edge at two depths draws a connection the walk never made.
            crossed = (neighbour.source, neighbour.target, neighbour.type)
            if crossed not in walked:
                walked.add(crossed)
                edges.append(
                    WalkedEdge(
                        source=neighbour.source,
                        target=neighbour.target,
                        type=neighbour.type,
                        evidence=neighbour.evidence,
                        source_chunk=neighbour.source_chunk,
                        hop=hop,
                    )
                )
            if neighbour.entity.identity in seen:
                continue
            seen.add(neighbour.entity.identity)
            reached.append(ReachedEntity(entity=neighbour.entity, hop=hop))
            next_frontier.append(neighbour.entity)
        frontier = next_frontier
    return reached, edges


#: How much a chunk gains from naming an entity reached `hop` steps out. A chunk naming
#: an entity the question itself named is worth 1.0; one two hops away, a third of that.
def _weight(hop: int) -> float:
    return 1.0 / (1.0 + hop)


def _collect(
    graph: EntityGraph, reached: Sequence[ReachedEntity], top_k: int
) -> list[TraversedChunk]:
    """The chunks mentioning everything reached, best first.

    Scored as the sum of `1/(1 + hop)` over the reached entities a chunk names, which
    rewards *both* proximity and breadth — and the bridge chunk this project is built to
    surface is precisely the one naming two reached entities at once, the named project
    and the artefact one hop out from it. Ties break on the shortest hop and then on the
    document, so two runs over one graph return one order.

    Scored over every mention and only then paid for: the text and the folder ancestry
    are read for the chunks that survive, which is what lets the scoring above see all
    of them.
    """
    if not reached:
        return []
    hops = {found.entity.identity: found for found in reached}
    via: dict[str, list[ReachedEntity]] = {}
    for mention in graph.mentions_of(list(hops)):
        found = hops.get(mention.entity)
        if found is None:
            continue
        reached_by = via.setdefault(mention.chunk_id, [])
        if found not in reached_by:
            reached_by.append(found)
    scores = {
        chunk_id: sum(_weight(found.hop) for found in reached_by)
        for chunk_id, reached_by in via.items()
    }
    best = sorted(
        via,
        key=lambda chunk_id: (
            -scores[chunk_id],
            min(found.hop for found in via[chunk_id]),
            chunk_id,
        ),
    )[:top_k]
    chunks = {chunk.id: chunk for chunk in graph.chunks(best)}
    return [
        TraversedChunk(
            chunk=chunks[chunk_id],
            via=tuple(sorted(via[chunk_id], key=lambda reached: reached.hop)),
            score=scores[chunk_id],
        )
        for chunk_id in best
        if chunk_id in chunks
    ]
