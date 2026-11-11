"""The answer's subgraph, shaped for a renderer — and the cross-branch flag.

Ticket 12 returns everything the answer path did. This module turns that into the one
thing ticket 15 draws: typed nodes with stable identities, typed edges between them,
and — the reason this file exists at all — a flag on every edge that leaves one branch
of the hierarchy for another.

**The cross-branch determination is made here, on the server.** Not because the client
could not loop over edges, but because the branch a node belongs to is known here and
is not in the payload unless this module puts it there. A frontend recomputing it would
be reimplementing the project's central claim from whatever happened to be serialised,
and getting it subtly wrong in a way that still draws a picture.

Three decisions shape the result.

**The chunks drawn are the context the answer was written from, and nothing else.**
Both legs ranked more than that; the subgraph is the evidence behind *this* answer, so
it holds the dozen chunks the model actually read. A picture showing chunks that were
retrieved and then dropped would overstate the case by exactly the number the cap
removed.

**A node's branch set is read off what is in the picture.** A folder, a document and a
chunk each sit under exactly one top-level folder — the client. An entity sits under
none: entities are global by design, which is the modelling decision that makes a
cross-client path real structure. So an entity's branches are the branches of the
chunks *in this subgraph* that mention it, which makes every flag traceable on screen:
an edge drawn as crossing is one whose crossing the audience can follow node by node.
Deriving it from the whole graph instead would flag edges whose second branch is
nowhere in the picture — a claim the visual cannot support. Which entities a drawn
chunk mentions is the one thing this module cannot work out alone, and the caller
passes it in: see `_mentioned`.

**An edge crosses when its two ends, taken together, sit under more than one branch.**
Not when they are disjoint: the demo's bridge entity is mentioned by both clients, so
the edge from one client's project into it has overlapping ends and is still the edge
on which the walk leaves that client's subtree. Containment edges never cross, by
construction — which is the point, since the hierarchy is the thing being escaped.

The module is pure: an `Answer` in, a `Subgraph` out, no driver and no client.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from graph_rag.ancestry import Ancestor, branch
from graph_rag.answer import Answer, ContextChunk, Source
from graph_rag.entities import EntityIdentity
from graph_rag.retrieval import Chunk
from graph_rag.traversal import MatchedEntity, ReachedEntity, WalkedEdge


class NodeKind(StrEnum):
    """What a node *is*, so the renderer can draw four shapes rather than one.

    The four labels of the graph schema, spelled the way the schema spells them.
    """

    FOLDER = "folder"
    DOCUMENT = "document"
    CHUNK = "chunk"
    ENTITY = "entity"


class EdgeKind(StrEnum):
    """What an edge *is*. The first three are the hierarchy and never cross branches;
    the last two are the entity web, which is the only thing that can."""

    CONTAINS = "contains"
    HAS_CHUNK = "has_chunk"
    MENTIONS = "mentions"
    RELATES = "relates"


def folder_id(names: Sequence[str]) -> str:
    """A folder's identity: its name chain from the top-level branch down.

    Built from the ancestry a chunk already carries rather than from a `Folder.path`,
    because ancestry is what both retrieval legs return and the path of a folder is
    not. Two folders with the same chain are the same folder, which is exactly the
    property the hierarchy guarantees.
    """
    return "folder:" + "/".join(names)


def entity_id(identity: EntityIdentity) -> str:
    """An entity's identity, as the graph's uniqueness constraint defines it.

    All three fields, including the scope — `Projekt Atlas` under one client and under
    another are two nodes, and collapsing them here would draw the demo's central
    distinction away.
    """
    return f"entity:{identity.type}:{identity.scope}:{identity.normalized_name}"


@dataclass(frozen=True)
class Node:
    """One node of the drawn subgraph.

    Flat rather than one class per kind: this is serialised to a renderer that
    switches on `kind`, and a union of four shapes would be four shapes for the
    frontend to learn.
    """

    id: str
    kind: NodeKind
    #: What to write on it.
    label: str
    #: The top-level folders this node sits under. One for anything in the hierarchy;
    #: any number for an entity, which belongs to no folder and is reached from all of
    #: them. More than one is a bridge — the demo.
    branches: tuple[str, ...] = ()
    #: Folder ancestry, outermost first, for everything that has one.
    ancestry: tuple[Ancestor, ...] = ()
    #: Kind-specific detail: the page of a chunk, the hop of an entity, the level name
    #: of a folder. Open rather than typed per kind, for the same reason as `kind`.
    detail: dict[str, object] = field(default_factory=dict)

    @property
    def bridge(self) -> bool:
        """Whether this node alone already spans two branches of the hierarchy.

        Only an entity can: every other kind sits at one place in the tree.
        """
        return len(self.branches) > 1


@dataclass(frozen=True)
class Edge:
    """One edge of the drawn subgraph, and whether it leaves its branch."""

    id: str
    kind: EdgeKind
    source: str
    target: str
    #: The relation type a `RELATES` edge carries — `leitet`, `nutzt`. Empty for the
    #: structural kinds, whose type *is* their kind.
    label: str = ""
    #: True when the two ends, together, sit under more than one top-level branch.
    #: Determined here; ticket 15 renders these in a distinct colour and must not
    #: recompute it.
    crosses_branches: bool = False
    #: The branches the two ends span, for the tooltip that has to justify the colour.
    branches: tuple[str, ...] = ()
    detail: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class Subgraph:
    """The nodes and edges behind one answer, ready to draw."""

    nodes: list[Node]
    edges: list[Edge]

    @property
    def crossing(self) -> list[Edge]:
        """The edges that leave a branch — the visual payoff, as a list."""
        return [edge for edge in self.edges if edge.crosses_branches]

    @property
    def branches(self) -> list[str]:
        """Every top-level branch the subgraph touches."""
        return sorted({name for node in self.nodes for name in node.branches})


def _ancestry_names(ancestry: Sequence[Ancestor]) -> list[str]:
    return [ancestor.name for ancestor in ancestry]


def document_id(chunk: Chunk) -> str:
    """A document's identity: its path, which the graph constrains to be unique."""
    return f"document:{chunk.document_path}"


def chunk_detail(
    position: int, found: ContextChunk, cited: bool, hop: int | None
) -> dict[str, object]:
    """One context chunk as the payload describes it, wherever it is described.

    The chunk node carries this and so does the numbered source beside the answer —
    the same chunk, read twice by one screen. Built here once so the two cannot
    disagree about which leg found it or what it was ranked.
    """
    chunk = found.chunk
    return {
        # The citation scheme: `[Quelle 3]` in the answer is the chunk whose source
        # number is 3, which is how ticket 15 connects a citation in the prose to a
        # node in the picture.
        "source": position,
        "cited": cited,
        "chunk_id": chunk.id,
        "document": chunk.document_path,
        "filename": chunk.filename,
        "index": chunk.index,
        "page": chunk.page,
        "text": chunk.text,
        "score": found.score,
        # Which leg found it, and where that leg placed it. A chunk the lexical leg
        # never returned is a chunk the graph found, which is the claim the whole
        # demo is making.
        "legs": sorted(source.value for source in found.ranks),
        "ranks": {source.value: rank for source, rank in found.ranks.items()},
        #: The nearest reached entity named by this chunk, or None if it names none.
        "hop": hop,
    }


def chunk_id(chunk: Chunk) -> str:
    """A chunk's identity: the deterministic id the ingest gave it.

    Public, with `entity_id` and `folder_id`, because the service points at these
    nodes from outside the subgraph — a source in the answer panel names the node it
    is drawn as — and two spellings of one identity would join nothing.
    """
    return f"chunk:{chunk.id}"


class _Builder:
    """Accumulates nodes and edges, keeping the first of each identity.

    First rather than last, and silently: the same folder is reached once per chunk
    under it, and the second arrival carries exactly what the first did. An identity
    that genuinely differed would be a bug upstream of here.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, Node] = {}
        self._edges: dict[str, Edge] = {}

    def node(self, node: Node) -> str:
        self._nodes.setdefault(node.id, node)
        return node.id

    def edge(self, edge: Edge) -> None:
        self._edges.setdefault(edge.id, edge)

    @property
    def nodes(self) -> list[Node]:
        return list(self._nodes.values())

    @property
    def edges(self) -> list[Edge]:
        return list(self._edges.values())

    def branches_of(self, node_id: str) -> tuple[str, ...]:
        node = self._nodes.get(node_id)
        return node.branches if node else ()


def _hierarchy(
    builder: _Builder,
    position: int,
    found: ContextChunk,
    cited: bool,
    names: Sequence[ReachedEntity],
) -> str:
    """The chain of folders down to this chunk, and the chunk itself.

    Returns the chunk node's id. Every folder on the way is created once and linked to
    the one above it, so the drawn tree is the corpus's tree rather than a star of
    documents around a root.
    """
    chunk = found.chunk
    folders = _ancestry_names(chunk.ancestry)
    home = branch(chunk.ancestry)
    branches = (home,) if home else ()

    parent: str | None = None
    for depth, ancestor in enumerate(chunk.ancestry, start=1):
        current = builder.node(
            Node(
                id=folder_id(folders[:depth]),
                kind=NodeKind.FOLDER,
                label=ancestor.name,
                branches=branches,
                ancestry=tuple(chunk.ancestry[:depth]),
                # The depth is the graph's own depth: ancestry excludes the corpus
                # root, which is depth 0 and not a level of the hierarchy.
                detail={"depth": depth, "level": ancestor.level},
            )
        )
        if parent is not None:
            builder.edge(_edge(EdgeKind.CONTAINS, parent, current, builder))
        parent = current

    document = builder.node(
        Node(
            id=document_id(chunk),
            kind=NodeKind.DOCUMENT,
            label=chunk.filename,
            branches=branches,
            ancestry=tuple(chunk.ancestry),
            detail={"path": chunk.document_path},
        )
    )
    if parent is not None:
        builder.edge(_edge(EdgeKind.CONTAINS, parent, document, builder))

    node = builder.node(
        Node(
            id=chunk_id(chunk),
            kind=NodeKind.CHUNK,
            label=f"Quelle {position}",
            branches=branches,
            ancestry=tuple(chunk.ancestry),
            detail=chunk_detail(
                position,
                found,
                cited,
                min((reached.hop for reached in names), default=None),
            ),
        )
    )
    builder.edge(_edge(EdgeKind.HAS_CHUNK, document, node, builder))
    return node


def _edge(
    kind: EdgeKind,
    source: str,
    target: str,
    builder: _Builder,
    *,
    label: str = "",
    detail: dict[str, object] | None = None,
) -> Edge:
    """One edge, with the cross-branch question answered.

    Containment edges go through this too rather than being hardcoded to False: the
    rule answers False for them anyway, and a hardcoded answer would stop being
    checked the day the hierarchy gained a shortcut.

    The flag is the union of the two ends' branches: an edge whose ends together sit
    under more than one top-level folder is an edge the hierarchy alone does not
    explain. Both ends must be placed for the question to have an answer — an entity
    no drawn chunk mentions sits nowhere in particular, and claiming it crosses would
    be a colour the picture cannot justify.
    """
    from_branches = builder.branches_of(source)
    to_branches = builder.branches_of(target)
    spanned = tuple(sorted(set(from_branches) | set(to_branches)))
    return Edge(
        id=f"{kind.value}:{source}->{target}" + (f":{label}" if label else ""),
        kind=kind,
        source=source,
        target=target,
        label=label,
        crosses_branches=bool(from_branches and to_branches) and len(spanned) > 1,
        branches=spanned,
        detail=detail or {},
    )


def _mentioned(
    answer: Answer, mentions: Mapping[str, Sequence[EntityIdentity]]
) -> list[tuple[ContextChunk, tuple[ReachedEntity, ...]]]:
    """Each drawn chunk paired with the reached entities it names, nearest hop first.

    Two sources, because the traversal leg only knows about the chunks *it* returned.
    A chunk the lexical leg found alone can name the bridge entity just as well — and
    on this corpus it routinely does, since the lexical leg is the one that rescues a
    question spelled with a literal project name. Drawn from `via` alone, that chunk
    would sit in the picture as an island and its client would be missing from the
    bridge's branch set, so a real crossing would render one-sided and uncoloured
    precisely when the lexical leg did the work.

    `mentions` is the graph's answer to "which of these entities does this chunk
    name", keyed by chunk id, and is supplied by the caller — this module stays pure.
    Empty is a valid argument: the result is then the walk's own view, which is what
    the CLI has.
    """
    reached = {found.entity.identity: found for found in _reached_entities(answer)}
    rows = []
    for found in answer.context:
        # A set would do, but identities are compared and not ordered, and a picture
        # that redraws itself differently between two runs over one graph is a demo
        # that cannot be rehearsed.
        named = [entity.entity.identity for entity in found.via]
        named += [
            identity
            for identity in mentions.get(found.chunk.id, ())
            if identity not in named
        ]
        entities = [reached[identity] for identity in named if identity in reached]
        rows.append((found, tuple(sorted(entities, key=lambda reached: reached.hop))))
    return rows


def _entity_branches(
    mentioned: Sequence[tuple[ContextChunk, tuple[ReachedEntity, ...]]],
) -> dict[EntityIdentity, set[str]]:
    """Which branches each entity is named from, within the drawn context.

    This is where an entity acquires a place in the hierarchy it deliberately does not
    have in the graph. Two branches here is the definition of a bridge, and it is read
    off the chunks on screen so that the claim is checkable against the picture.
    """
    branches: dict[EntityIdentity, set[str]] = {}
    for found, entities in mentioned:
        home = branch(found.chunk.ancestry)
        if not home:
            continue
        for reached in entities:
            branches.setdefault(reached.entity.identity, set()).add(home)
    return branches


def _entity_node(
    reached: ReachedEntity,
    branches: Iterable[str],
    matched: MatchedEntity | None,
) -> Node:
    entity = reached.entity
    return Node(
        id=entity_id(entity.identity),
        kind=NodeKind.ENTITY,
        label=entity.name,
        branches=tuple(sorted(branches)),
        detail={
            "type": entity.identity.type,
            # The scope is the client a name had to be split by. Empty for a global
            # entity, which is the kind that can bridge.
            "scope": entity.identity.scope,
            "normalized_name": entity.identity.normalized_name,
            "mentions": entity.mentions,
            #: 0 for an entity the question named; higher is how far the walk went.
            "hop": reached.hop,
            "matched": matched is not None,
            "matched_as": matched.question.surface if matched else None,
            "match_how": [way.value for way in matched.how] if matched else [],
            "similarity": matched.similarity if matched else None,
        },
    )


def _reached_entities(answer: Answer) -> list[ReachedEntity]:
    """Every entity the picture has to hold a node for, at its nearest hop.

    The walk's own list *and* the entities the drawn chunks were reached through.
    They are the same list in practice — a chunk is in the context because the walk
    reached an entity it mentions — but an edge whose end is missing is drawn by a
    force layout as a node with no label, or dropped along with the crossing it was
    carrying, and neither failure says anything.
    """
    nearest: dict[EntityIdentity, ReachedEntity] = {}
    for reached in [*answer.reached, *(r for found in answer.context for r in found.via)]:
        current = nearest.get(reached.entity.identity)
        if current is None or reached.hop < current.hop:
            nearest[reached.entity.identity] = reached
    return list(nearest.values())


def build_subgraph(
    answer: Answer, mentions: Mapping[str, Sequence[EntityIdentity]] | None = None
) -> Subgraph:
    """The subgraph behind an answer: the hierarchy it read from, and the walk.

    `mentions` maps a context chunk's id to the entities it names, as the graph
    knows it — see `_mentioned` for why the walk's own view is not enough. Omitted,
    the picture holds only what the traversal leg reached, which is correct but
    smaller.

    Built in drawing order — the tree first, then the entity web over it — so that
    every edge is created after both of its ends and can be asked which branches they
    sit under.
    """
    builder = _Builder()
    cited = set(answer.citations)
    mentioned = _mentioned(answer, mentions or {})

    for position, (found, entities) in enumerate(mentioned, start=1):
        _hierarchy(builder, position, found, position in cited, entities)

    branches = _entity_branches(mentioned)
    matched = {match.entity.identity: match for match in answer.matches}
    # Nearest hop first, so the walk reads outward from what the question named.
    for reached in sorted(_reached_entities(answer), key=lambda reached: reached.hop):
        builder.node(
            _entity_node(
                reached,
                branches.get(reached.entity.identity, set()),
                matched.get(reached.entity.identity),
            )
        )

    # Mentions after every entity exists, so a chunk's edge to a bridge is flagged
    # against the bridge's full branch set rather than against half of it.
    for found, entities in mentioned:
        for reached in entities:
            builder.edge(
                _edge(
                    EdgeKind.MENTIONS,
                    chunk_id(found.chunk),
                    entity_id(reached.entity.identity),
                    builder,
                    detail={"hop": reached.hop},
                )
            )

    for walked in answer.edges:
        builder.edge(_relation(walked, builder))

    return Subgraph(nodes=builder.nodes, edges=builder.edges)


def _relation(walked: WalkedEdge, builder: _Builder) -> Edge:
    """One `RELATES` edge, drawn in its own direction.

    The edge's stored direction, never the direction the walk met it in: `leitet`
    drawn backwards says the project leads the consultant, and an audience reading the
    picture has no way to know it is reversed.
    """
    return _edge(
        EdgeKind.RELATES,
        entity_id(walked.source),
        entity_id(walked.target),
        builder,
        label=walked.type,
        detail={
            "hop": walked.hop,
            "evidence": walked.evidence,
            "source_chunk": walked.source_chunk,
        },
    )


def leg_balance(answer: Answer) -> dict[str, int]:
    """How many context chunks each leg contributed, and how many both did.

    The demo's claim as four numbers: `traversal_only` is the count of chunks the
    lexical leg never returned, which is the only honest way to say what the graph
    added without a flat-RAG baseline to put beside it.
    """
    search = [found for found in answer.context if found.found_by(Source.SEARCH)]
    traversal = [found for found in answer.context if found.found_by(Source.TRAVERSAL)]
    return {
        "context": len(answer.context),
        "search": len(search),
        "traversal": len(traversal),
        "both": len([found for found in answer.context if len(found.ranks) == 2]),
        "traversal_only": len(
            [found for found in traversal if not found.found_by(Source.SEARCH)]
        ),
    }
