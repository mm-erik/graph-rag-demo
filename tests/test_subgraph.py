"""The one silent failure in the payload: which edges are flagged as crossing.

Everything else the service returns is checkable by reading it — an answer is prose,
a source list is numbered, a missing node is a hole in the picture. The cross-branch
flag is different: it is a boolean the frontend colours by, and both ways of getting
it wrong draw a picture that looks correct.

Flag too much and every edge is coloured, so the colour stops meaning anything and the
demo's one visual claim — *the graph reached across the hierarchy* — is made about
edges that stayed inside one client. Flag too little and the demo makes no claim at
all: the picture is a tidy tree with a connection the audience has to be told about.
Neither raises, and neither is visible from the terminal.

The fixture is the shape of the real demo question, reduced to what the flag depends
on: two clients, each with a project of the same name, and one global entity mentioned
from both — the bridge the corpus is engineered to plant.
"""

from __future__ import annotations

from graph_rag.ancestry import Ancestor
from graph_rag.answer import Answer, ContextChunk, Parameters, Source
from graph_rag.entities import EntityIdentity
from graph_rag.retrieval import Chunk
from graph_rag.subgraph import EdgeKind, NodeKind, build_subgraph, entity_id
from graph_rag.traversal import (
    GraphEntity,
    ReachedEntity,
    TraversalResult,
    WalkedEdge,
)

NORDLICHT = "Nordlicht Energie AG"
BRANDT = "Brandt und Soehne Maschinenbau"


def _chunk(chunk_id: str, client: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        document_path=f"corpus/{client}/2023/Projekt Atlas/bericht.pdf",
        filename="bericht.pdf",
        index=0,
        page=3,
        text=f"Text von {chunk_id}.",
        ancestry=[
            Ancestor(name=client, level="Kunde"),
            Ancestor(name="2023", level="Jahr"),
            Ancestor(name="Projekt Atlas", level="Projekt"),
        ],
    )


def _identity(name: str, scope: str = "") -> EntityIdentity:
    return EntityIdentity(type="konzept", normalized_name=name, scope=scope)


def _reached(name: str, hop: int, scope: str = "") -> ReachedEntity:
    identity = _identity(name, scope)
    return ReachedEntity(
        entity=GraphEntity(identity=identity, name=name.title(), mentions=2), hop=hop
    )


#: One client's project, the other's, and the artefact both of them use.
ATLAS_NORD = _reached("projekt atlas", hop=0, scope=NORDLICHT)
ATLAS_BRANDT = _reached("projekt atlas", hop=0, scope=BRANDT)
BRIDGE = _reached("atlas referenzarchitektur", hop=1)


def _answer(*, text: str = "Antwort [Quelle 1].") -> Answer:
    """The demo question's result: a chunk from each client, joined by one artefact."""
    context = [
        ContextChunk(
            chunk=_chunk("nord-1", NORDLICHT),
            score=0.5,
            ranks={Source.TRAVERSAL: 1},
            via=(ATLAS_NORD, BRIDGE),
        ),
        ContextChunk(
            chunk=_chunk("brandt-1", BRANDT),
            score=0.4,
            ranks={Source.TRAVERSAL: 2, Source.SEARCH: 1},
            via=(ATLAS_BRANDT, BRIDGE),
        ),
    ]
    traversal = TraversalResult(
        question="Was verbindet die beiden Atlas-Projekte?",
        hops=2,
        entities=[],
        matches=[],
        reached=[ATLAS_NORD, ATLAS_BRANDT, BRIDGE],
        edges=[
            WalkedEdge(
                source=ATLAS_NORD.entity.identity,
                target=BRIDGE.entity.identity,
                type="nutzt",
                evidence="…",
                source_chunk="nord-1",
                hop=1,
            ),
            WalkedEdge(
                source=ATLAS_BRANDT.entity.identity,
                target=BRIDGE.entity.identity,
                type="nutzt",
                evidence="…",
                source_chunk="brandt-1",
                hop=1,
            ),
        ],
        results=[],
    )
    return Answer(
        question=traversal.question,
        text=text,
        context=context,
        parameters=Parameters(),
        traversal=traversal,
    )


def _edges(subgraph, kind: EdgeKind) -> list:
    return [edge for edge in subgraph.edges if edge.kind is kind]


def _node(subgraph, node_id: str):
    return next(node for node in subgraph.nodes if node.id == node_id)


def _assert_closed(subgraph) -> None:
    """Every edge points at a node that is in the payload.

    A renderer given an edge to a missing node either drops it or invents one. Both
    are silent: the first loses the crossing, the second draws an unlabelled node in
    the middle of the picture.
    """
    ids = {node.id for node in subgraph.nodes}
    assert ids
    for edge in subgraph.edges:
        assert edge.source in ids
        assert edge.target in ids


def test_the_hierarchy_is_never_flagged_as_crossing_itself():
    """A folder containing a folder stays inside its branch, by construction.

    The colour means "this edge left the tree". A containment edge that carried it
    would paint the tree itself and leave nothing for the entity web to stand out
    against.
    """
    subgraph = build_subgraph(_answer())
    structural = _edges(subgraph, EdgeKind.CONTAINS) + _edges(
        subgraph, EdgeKind.HAS_CHUNK
    )
    assert structural
    assert not [edge for edge in structural if edge.crosses_branches]


def test_a_relation_into_an_entity_both_clients_mention_is_flagged():
    """The payoff edge: one client's project to the artefact the other uses too.

    The two ends are *not* disjoint — the bridge is mentioned from this client as
    well — so a rule that asked for disjoint branches would leave the demo's one
    interesting edge uncoloured.
    """
    subgraph = build_subgraph(_answer())
    relations = _edges(subgraph, EdgeKind.RELATES)
    assert len(relations) == 2
    assert all(edge.crosses_branches for edge in relations)
    assert all(
        sorted(edge.branches) == sorted([NORDLICHT, BRANDT]) for edge in relations
    )


def test_a_relation_inside_one_client_is_not_flagged():
    """The same walk with nothing shared: an edge that stays in one subtree.

    Guards the other direction — a rule that flagged on "the subgraph spans two
    clients" rather than on the edge's own two ends would colour this one too.
    """
    answer = _answer()
    local = _reached("taktzeitanalyse", hop=1, scope=NORDLICHT)
    context = [
        ContextChunk(
            chunk=_chunk("nord-1", NORDLICHT),
            score=0.5,
            ranks={Source.TRAVERSAL: 1},
            via=(ATLAS_NORD, local),
        ),
        *answer.context[1:],
    ]
    walked = WalkedEdge(
        source=ATLAS_NORD.entity.identity,
        target=local.entity.identity,
        type="erzeugt",
        evidence="…",
        source_chunk="nord-1",
        hop=1,
    )
    traversal = TraversalResult(
        question=answer.question,
        hops=2,
        entities=[],
        matches=[],
        reached=[ATLAS_NORD, ATLAS_BRANDT, local],
        edges=[walked],
        results=[],
    )
    subgraph = build_subgraph(
        Answer(
            question=answer.question,
            text=answer.text,
            context=context,
            parameters=Parameters(),
            traversal=traversal,
        )
    )
    _assert_closed(subgraph)
    (relation,) = _edges(subgraph, EdgeKind.RELATES)
    assert not relation.crosses_branches
    assert relation.branches == (NORDLICHT,)


def test_a_chunk_the_lexical_leg_found_alone_still_names_its_entities():
    """The bridge has to count the clients that reached it through either leg.

    A chunk the entity walk never returned is reached through no entity, so drawn
    from the walk alone it hangs in the picture unconnected and its client is missing
    from the bridge's branch set — and on this corpus the lexical leg is exactly the
    leg that rescues a question spelled with a literal project name. The crossing
    would then render one-sided, uncoloured, precisely when the lexical leg did the
    work. The graph's own answer to "which of these entities does this chunk name" is
    passed in to close that.
    """
    answer = _answer()
    lexical = ContextChunk(
        chunk=_chunk("brandt-2", BRANDT),
        score=0.3,
        ranks={Source.SEARCH: 2},
        via=(),
    )
    with_lexical = Answer(
        question=answer.question,
        text=answer.text,
        context=[answer.context[0], lexical],
        parameters=Parameters(),
        traversal=answer.traversal,
    )

    blind = build_subgraph(with_lexical)
    assert _node(blind, entity_id(BRIDGE.entity.identity)).branches == (NORDLICHT,)
    assert not [e for e in blind.edges if e.source == "chunk:brandt-2"]

    seeing = build_subgraph(
        with_lexical, {"brandt-2": [BRIDGE.entity.identity, _identity("unbekannt")]}
    )
    assert sorted(_node(seeing, entity_id(BRIDGE.entity.identity)).branches) == sorted(
        [NORDLICHT, BRANDT]
    )
    (mention,) = [e for e in seeing.edges if e.source == "chunk:brandt-2"]
    assert mention.kind is EdgeKind.MENTIONS
    assert mention.crosses_branches
    # An entity the walk never reached is not drawn in from a mention row: the
    # picture is the walk's, and a node with no hop could not be traced to it.
    _assert_closed(seeing)


def test_an_entity_no_drawn_chunk_mentions_is_not_claimed_to_cross():
    """An entity the walk reached but no drawn chunk names sits nowhere in the tree.

    Flagging its edges would colour a crossing the picture cannot show — the second
    branch would be nowhere on screen, and the audience would be asked to take the
    colour on faith, which is the one thing this visualisation exists not to do.
    """
    answer = _answer()
    outer = _reached("vektor consulting", hop=2)
    traversal = TraversalResult(
        question=answer.question,
        hops=2,
        entities=[],
        matches=[],
        reached=[*answer.reached, outer],
        edges=[
            *answer.edges,
            WalkedEdge(
                source=BRIDGE.entity.identity,
                target=outer.entity.identity,
                type="betreut",
                evidence="…",
                source_chunk="nord-1",
                hop=2,
            ),
        ],
        results=[],
    )
    subgraph = build_subgraph(
        Answer(
            question=answer.question,
            text=answer.text,
            context=answer.context,
            parameters=Parameters(),
            traversal=traversal,
        )
    )
    unplaced = _node(subgraph, entity_id(outer.entity.identity))
    assert unplaced.branches == ()
    (edge,) = [
        edge for edge in subgraph.edges if edge.target == unplaced.id
    ]
    assert not edge.crosses_branches


def test_two_clients_projects_of_the_same_name_stay_two_nodes():
    """Scope is part of an entity's identity, here as in the graph.

    Dropping it would merge the two clients' `Projekt Atlas` into one node, and the
    picture would then show a single project both clients worked on — the precise
    claim the corpus is built to make false.
    """
    subgraph = build_subgraph(_answer())
    entities = [node for node in subgraph.nodes if node.kind is NodeKind.ENTITY]
    assert len(entities) == 3
    assert entity_id(ATLAS_NORD.entity.identity) != entity_id(
        ATLAS_BRANDT.entity.identity
    )


def test_the_bridge_entity_is_the_only_node_spanning_both_clients():
    subgraph = build_subgraph(_answer())
    assert [node.label for node in subgraph.nodes if node.bridge] == [
        BRIDGE.entity.name
    ]
    assert subgraph.branches == sorted([BRANDT, NORDLICHT])


def test_a_chunk_node_carries_the_citation_number_the_answer_uses():
    """`[Quelle 1]` in the prose has to find its node, or the citation panel of
    ticket 15 points at nothing."""
    subgraph = build_subgraph(_answer())
    chunks = {
        node.detail["source"]: node
        for node in subgraph.nodes
        if node.kind is NodeKind.CHUNK
    }
    assert sorted(chunks) == [1, 2]
    assert chunks[1].detail["cited"] is True
    assert chunks[2].detail["cited"] is False
    assert chunks[1].detail["chunk_id"] == "nord-1"
    assert chunks[2].detail["ranks"] == {"traversal": 2, "search": 1}


def test_every_edge_points_at_a_node_that_is_in_the_payload():
    _assert_closed(build_subgraph(_answer()))
