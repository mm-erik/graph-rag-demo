"""One silent failure guards this file: how far the traversal actually walked.

The entity leg is the only thing in this project that can reach a document sharing no
vocabulary with the question, and how far it reaches is a number — one hop is precise,
two is where cross-document connections appear, three is noise. Walk one hop fewer than
asked and the leg still returns plausible chunks from the entity that was named; it
simply never reaches the other client, and the demo's central claim quietly stops being
demonstrated. Walk one more and the answer drowns. Nothing raises in either case.

The graph below is a chain rather than the real corpus, so the hop each chunk sits at is
readable off the fixture instead of being recomputed the way the implementation does.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from graph_rag.ancestry import Ancestor
from graph_rag.entities import EntityIdentity
from graph_rag.retrieval import Chunk
from graph_rag.traversal import (
    ChunkMention,
    GraphEntity,
    Neighbour,
    traverse_entities,
)


def _identity(name: str) -> EntityIdentity:
    return EntityIdentity(type="konzept", normalized_name=name)


def _entity(name: str) -> GraphEntity:
    return GraphEntity(identity=_identity(name), name=name.title(), mentions=1)


@dataclass
class FakeGraph:
    """An entity graph as a chain of names, with one chunk naming each.

    `edges` is read both ways round, as the real one is: a question naming a person has
    to reach the project they lead whichever direction the extraction happened to write
    the edge in.
    """

    edges: list[tuple[str, str]]
    #: Extra chunks beyond the one per entity, as `(chunk id, entity names)`.
    extra: dict[str, list[str]] | None = None

    def by_name(self, names):
        known = {name for edge in self.edges for name in edge}
        return [_entity(name) for name in sorted(known & set(names))]

    def by_vector(self, vector, limit):
        return []

    def neighbours(self, identities):
        wanted = {identity.normalized_name for identity in identities}
        found = []
        for left, right in self.edges:
            # The edge keeps its stored direction whichever end the walk arrives from,
            # as the real one does: only the node reached changes.
            for here, there in ((left, right), (right, left)):
                if here in wanted:
                    found.append(
                        Neighbour(
                            entity=_entity(there),
                            source=_identity(left),
                            target=_identity(right),
                            type="basiert_auf",
                            evidence=f"{left} -> {right}",
                            source_chunk=f"chunk-{left}",
                        )
                    )
        return found

    def mentions_of(self, identities):
        wanted = {identity.normalized_name for identity in identities}
        mentioning = {f"chunk-{name}": [name] for name in sorted(wanted)}
        for chunk_id, names in (self.extra or {}).items():
            named = [name for name in names if name in wanted]
            if named:
                mentioning[chunk_id] = named
        return [
            ChunkMention(chunk_id=chunk_id, entity=_identity(name))
            for chunk_id, names in sorted(mentioning.items())
            for name in names
        ]

    def chunks(self, chunk_ids):
        return [_chunk(chunk_id) for chunk_id in chunk_ids]


def _chunk(chunk_id: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        document_path=f"corpus/{chunk_id}.pdf",
        filename=f"{chunk_id}.pdf",
        index=0,
        page=1,
        text=chunk_id,
        ancestry=[Ancestor(name="Kunde", level="Kunde")],
    )


class FakeModel:
    """Question-side extraction, pinned to the names the test asks about."""

    def __init__(self, *names: str) -> None:
        self._names = names

    def complete(self, system: str, user: str, *, json_object: bool = False) -> str:
        return json.dumps(
            {"entities": [{"name": name, "type": "konzept"} for name in self._names]}
        )


class FakeEmbedder:
    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


def _reached(question: str, graph: FakeGraph, hops: int, **kwargs) -> list[str]:
    result = traverse_entities(
        question=question,
        graph=graph,
        embedder=FakeEmbedder(),
        model=FakeModel(*kwargs.pop("names", ("a",))),
        hops=hops,
        threshold=kwargs.pop("threshold", 0.82),
        candidates=kwargs.pop("candidates", 5),
        **kwargs,
    )
    return sorted(found.chunk.id for found in result.results)


def test_one_hop_reaches_the_neighbours_of_the_named_entity_and_stops():
    """`a` is named, `b` is one edge away, `c` is two. At one hop the chunk naming `c`
    must not appear — a leg that quietly walked further would reach the other client on
    a question that asked about one of them."""
    graph = FakeGraph(edges=[("a", "b"), ("b", "c")])

    assert _reached("a?", graph, hops=1) == ["chunk-a", "chunk-b"]


def test_two_hops_reach_the_document_sharing_no_vocabulary_with_the_question():
    """The whole leg. `c` is named by no chunk the question's words could find, and two
    hops is the depth at which it arrives."""
    graph = FakeGraph(edges=[("a", "b"), ("b", "c")])

    assert _reached("a?", graph, hops=2) == ["chunk-a", "chunk-b", "chunk-c"]


class VectorOnlyGraph(FakeGraph):
    """A graph nothing in the question names exactly, reachable only by embedding."""

    def __init__(self, similarity: float) -> None:
        super().__init__(edges=[("atlas referenzarchitektur", "kafka")])
        self._similarity = similarity

    def by_name(self, names):
        return []

    def by_vector(self, vector, limit):
        return [(_entity("atlas referenzarchitektur"), self._similarity)]


def test_an_entity_the_question_spells_differently_is_matched_by_embedding():
    """A question will not write a name the way a report does. The matched node is the
    graph's, not the question's — it is what the walk starts from and what the UI
    labels the starting point with."""
    result = traverse_entities(
        question="Was ist die Atlas-Referenz?",
        graph=VectorOnlyGraph(similarity=0.91),
        embedder=FakeEmbedder(),
        model=FakeModel("Atlas-Referenz"),
        hops=1,
        threshold=0.82,
        candidates=5,
    )

    assert [match.entity.name for match in result.matches] == [
        "Atlas Referenzarchitektur"
    ]
    assert sorted(found.chunk.id for found in result.results) == [
        "chunk-atlas referenzarchitektur",
        "chunk-kafka",
    ]


def test_a_candidate_below_the_threshold_does_not_become_a_starting_point():
    """The resolution hazard of ticket 08 in miniature, and the same silent failure: a
    walk started from the wrong node returns genuinely connected material about
    something nobody asked after."""
    result = traverse_entities(
        question="Was ist die Atlas-Referenz?",
        graph=VectorOnlyGraph(similarity=0.74),
        embedder=FakeEmbedder(),
        model=FakeModel("Atlas-Referenz"),
        hops=2,
        threshold=0.82,
        candidates=5,
    )

    assert result.matches == []
    assert result.results == []


class TwoScopesGraph(FakeGraph):
    """One name, two nodes — the corpus's planted collision, as the graph holds it."""

    def __init__(self) -> None:
        super().__init__(edges=[("projekt atlas", "katharina vogt")])

    def by_name(self, names):
        return [
            GraphEntity(
                identity=EntityIdentity(
                    type="konzept", normalized_name="projekt atlas", scope=scope
                ),
                name="Projekt Atlas",
                mentions=1,
            )
            for scope in ("Brandt und Soehne Maschinenbau", "Nordlicht Energie AG")
        ]


def test_one_name_in_the_question_reaches_every_client_that_uses_it():
    """The demo. `Projekt Atlas` is two nodes the hierarchy ruled apart, and a question
    asked from outside the hierarchy names neither in particular — so it must start the
    walk at both. Matching on the scoped identity instead would reach one client's
    project, answer plausibly from it, and report nothing."""
    result = traverse_entities(
        question="Was verbindet die Atlas-Projekte?",
        graph=TwoScopesGraph(),
        embedder=FakeEmbedder(),
        model=FakeModel("Projekt Atlas"),
        hops=1,
        threshold=0.82,
        candidates=5,
    )

    assert sorted(match.entity.scope for match in result.matches) == [
        "Brandt und Soehne Maschinenbau",
        "Nordlicht Energie AG",
    ]


def test_the_path_records_each_edge_once_in_its_own_direction():
    """Ticket 15 draws this path and the audience is asked to believe it.

    Two ways to draw a connection nobody made. An edge recorded at the wrong hop puts a
    node closer to the question than the walk ever found it; an edge recorded in the
    direction the walk happened to meet it in, rather than the direction the graph
    stores, says the project leads the consultant. The walk also meets `a—b` again on
    its way out of `b`, and recording it twice would put one edge at two depths."""
    graph = FakeGraph(edges=[("a", "b"), ("b", "c")])

    result = traverse_entities(
        question="a?",
        graph=graph,
        embedder=FakeEmbedder(),
        model=FakeModel("a"),
        hops=2,
        threshold=0.82,
        candidates=5,
    )

    assert [
        (edge.source.normalized_name, edge.target.normalized_name, edge.hop)
        for edge in result.edges
    ] == [("a", "b", 1), ("b", "c", 2)]
    assert {
        reached.entity.identity.normalized_name: reached.hop
        for reached in result.reached
    } == {"a": 0, "b": 1, "c": 2}
