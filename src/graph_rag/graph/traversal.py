"""The entity-graph reads the traversal leg walks, against Neo4j.

One read per step of the leg, and one read per *hop*, rather than a single
variable-length path query. A `MATCH (a)-[:RELATES*1..2]-(b)` would be one round trip,
but it collapses the walk: the hop each node was reached at is provenance ticket 15
draws, and recovering it from a path result is guesswork the picture cannot afford.
Two round trips on a graph this size cost nothing worth the ambiguity.

Chunks come back with their **folder ancestry**, from the same projection the other leg
uses, because an answer built from a chunk that arrives as a path string can only cite
a path string.

The driver is injected; nothing here constructs one.
"""

from __future__ import annotations

from collections.abc import Sequence

from neo4j import Driver

from graph_rag.entities import EntityIdentity
from graph_rag.graph.ancestry import ancestry_projection, to_ancestors
from graph_rag.graph.schema import ENTITY_VECTOR_INDEX
from graph_rag.retrieval import Chunk
from graph_rag.traversal import ChunkMention, GraphEntity, Neighbour

_ENTITY_COLUMNS = (
    "{normalized_name: %(n)s.normalized_name, type: %(n)s.type, "
    "scope: coalesce(%(n)s.scope, ''), name: %(n)s.name, "
    "mentions: coalesce(%(n)s.mentions, 0)}"
)


def _entity_row(variable: str) -> str:
    return _ENTITY_COLUMNS % {"n": variable}


def _identity(row) -> EntityIdentity:
    """An entity's identity from any row carrying the three properties that make it."""
    return EntityIdentity(
        type=row["type"],
        normalized_name=row["normalized_name"],
        scope=row["scope"],
    )


def _entity(row) -> GraphEntity:
    return GraphEntity(
        identity=_identity(row), name=row["name"], mentions=row["mentions"]
    )


# Matched on the normalised name alone — not on the type and not on the scope. Both
# omissions are deliberate and both are explained in `traversal.match_entities`: a
# question naming `Projekt Atlas` has to reach every client that runs one.
_BY_NAME = f"""
UNWIND $names AS name
MATCH (e:Entity {{normalized_name: name}})
RETURN {_entity_row('e')} AS entity
ORDER BY e.mentions DESC, e.name, e.scope
"""


def match_by_name(driver: Driver, database: str, names: Sequence[str]) -> list[GraphEntity]:
    """Every entity node whose normalised name is one of `names`."""
    if not names:
        return []
    records, _, _ = driver.execute_query(
        _BY_NAME, {"names": list(names)}, database_=database
    )
    return [_entity(record["entity"]) for record in records]


_BY_VECTOR = f"""
CALL db.index.vector.queryNodes('{ENTITY_VECTOR_INDEX}', $limit, $vector)
YIELD node, score
RETURN {_entity_row('node')} AS entity, score
ORDER BY score DESC
"""

#: As with the chunk index, never ask the vector index for fewer candidates than this.
#: `limit` is also roughly what an approximate index spends searching, and the entity
#: names in this corpus are short and crowd together — a two-candidate search settles
#: in the wrong neighbourhood and the leg then starts its walk from the wrong node.
ENTITY_CANDIDATE_FLOOR = 10


def match_by_vector(
    driver: Driver, database: str, vector: Sequence[float], limit: int
) -> list[tuple[GraphEntity, float]]:
    """The `limit` nearest entity nodes to `vector`, with their similarity."""
    records, _, _ = driver.execute_query(
        _BY_VECTOR,
        {"vector": list(vector), "limit": max(limit, ENTITY_CANDIDATE_FLOOR)},
        database_=database,
    )
    return [(_entity(record["entity"]), record["score"]) for record in records][:limit]


# Undirected on purpose. `RELATES` carries a direction because `leitet` reads one way
# round, but a question naming a person must reach the project they lead and a question
# naming the project must reach the person — the walk is about connectedness, and
# following only the stored direction would make half the corpus unreachable depending
# on which end the question happened to name.
_NEIGHBOURS = f"""
UNWIND $keys AS key
MATCH (a:Entity {{normalized_name: key.normalized_name, type: key.type,
                  scope: key.scope}})-[r:RELATES]-(b:Entity)
RETURN {_entity_row('b')} AS reached,
       {_entity_row('startNode(r)')} AS source, {_entity_row('endNode(r)')} AS target,
       r.type AS type, coalesce(r.evidence, '') AS evidence,
       coalesce(r.source_chunk, '') AS source_chunk
ORDER BY b.mentions DESC, b.name, r.type
"""


def _keys(identities: Sequence[EntityIdentity]) -> list[dict]:
    return [
        {
            "normalized_name": identity.normalized_name,
            "type": identity.type,
            "scope": identity.scope,
        }
        for identity in identities
    ]


def neighbours(
    driver: Driver, database: str, identities: Sequence[EntityIdentity]
) -> list[Neighbour]:
    """Every entity one `RELATES` edge from any of `identities`, either direction."""
    if not identities:
        return []
    records, _, _ = driver.execute_query(
        _NEIGHBOURS, {"keys": _keys(identities)}, database_=database
    )
    return [
        Neighbour(
            entity=_entity(record["reached"]),
            source=_identity(record["source"]),
            target=_identity(record["target"]),
            type=record["type"],
            evidence=record["evidence"],
            source_chunk=record["source_chunk"],
        )
        for record in records
    ]


_MENTIONS = """
UNWIND $keys AS key
MATCH (c:Chunk)-[:MENTIONS]->(:Entity {normalized_name: key.normalized_name,
                                       type: key.type, scope: key.scope})
RETURN c.id AS id, key.normalized_name AS normalized_name, key.type AS type,
       key.scope AS scope
"""


def mentions_of(
    driver: Driver, database: str, identities: Sequence[EntityIdentity]
) -> list[ChunkMention]:
    """One row per `(chunk, reached entity)`, uncapped and without the chunk's text.

    Uncapped on purpose. Scoring has to weigh every reached entity a chunk names, and a
    cap here truncates by whichever client the planner happened to read first — which
    on the one question this project exists to answer silently returns half the answer.
    """
    if not identities:
        return []
    records, _, _ = driver.execute_query(
        _MENTIONS, {"keys": _keys(identities)}, database_=database
    )
    return [
        ChunkMention(
            chunk_id=record["id"],
            entity=_identity(record),
        )
        for record in records
    ]


# The ancestry projection is the one the other leg uses, for the same reason: a chunk
# that arrives as a path string can only be cited as a path string.
_CHUNKS = (
    """
MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)
WHERE c.id IN $ids
WITH c, d
"""
    + ancestry_projection("c, d")
    + """
RETURN c.id AS id, c.index AS index, c.page AS page, c.text AS text,
       d.path AS document_path, d.filename AS filename, ancestry
"""
)


def chunks(driver: Driver, database: str, chunk_ids: Sequence[str]) -> list[Chunk]:
    """The named chunks, with their folder ancestry. The caller holds the order."""
    if not chunk_ids:
        return []
    records, _, _ = driver.execute_query(
        _CHUNKS, {"ids": list(chunk_ids)}, database_=database
    )
    return [
        Chunk(
            id=record["id"],
            document_path=record["document_path"],
            filename=record["filename"],
            index=record["index"],
            page=record["page"],
            text=record["text"],
            ancestry=to_ancestors(record["ancestry"]),
        )
        for record in records
    ]


class GraphEntityGraph:
    """`EntityGraph` backed by Neo4j, binding the injected driver to a database."""

    def __init__(self, driver: Driver, database: str) -> None:
        self._driver = driver
        self._database = database

    def by_name(self, names: Sequence[str]) -> list[GraphEntity]:
        return match_by_name(self._driver, self._database, names)

    def by_vector(
        self, vector: Sequence[float], limit: int
    ) -> list[tuple[GraphEntity, float]]:
        return match_by_vector(self._driver, self._database, vector, limit)

    def neighbours(self, identities: Sequence[EntityIdentity]) -> list[Neighbour]:
        return neighbours(self._driver, self._database, identities)

    def mentions_of(
        self, identities: Sequence[EntityIdentity]
    ) -> list[ChunkMention]:
        return mentions_of(self._driver, self._database, identities)

    def chunks(self, chunk_ids: Sequence[str]) -> list[Chunk]:
        return chunks(self._driver, self._database, chunk_ids)
