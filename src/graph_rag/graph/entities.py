"""Reading what extraction works from, and writing the knowledge graph back.

This is where the folder tree stops being the whole structure and the graph becomes a
knowledge graph. Two things here are load-bearing and easy to get silently wrong:

- **Entity nodes carry no folder edge of any kind.** Not `CONTAINS`, not an ownership
  property pointing at a path. An entity named by two clients' documents is one node
  reachable from both subtrees through `MENTIONS`, and that is what makes the
  cross-client path real structure rather than a join computed at query time. Every
  write in this file is checked against that rule.
- **Edges are replaced, not accumulated.** A re-ingest with different thresholds would
  otherwise leave the previous run's edges in place underneath the new ones, and a
  graph that holds both readings of the corpus is wrong in a way nothing reports.

The driver is injected; nothing here constructs one.
"""

from __future__ import annotations

from dataclasses import dataclass

from neo4j import Driver, ManagedTransaction

from graph_rag.entities import ChunkSource, GraphRelation, Mention
from graph_rag.resolution import ResolvedEntity

# Ancestry is ordered by the folder's recorded depth, as in the summary read, and the
# corpus root is skipped — it is not a level of the hierarchy.
_READ_SOURCES = """
MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)
OPTIONAL MATCH (f:Folder)-[:CONTAINS*]->(d)
WITH d, c, f ORDER BY f.depth
WITH d, c, [folder IN collect(f) WHERE folder.depth > 0 | folder.name] AS ancestry
RETURN c.id AS id, c.text AS text, ancestry
ORDER BY d.path, c.index
"""

# Every folder paired with the top-level branch it sits under. `*0..` includes the
# branch folder itself, so a client whose name repeats as a project name is caught too.
_READ_BRANCH_FOLDERS = """
MATCH (branch:Folder {depth: 1})-[:CONTAINS*0..]->(f:Folder)
RETURN branch.name AS branch, f.name AS name
"""


def read_chunk_sources(driver: Driver, database: str) -> list[ChunkSource]:
    """Every chunk to extract from, with the folder path of its document.

    The whole corpus, never a subset: resolution has to see every mention of a name
    before it can tell a shared entity from a coincidence of spelling, and the edges
    this pass replaces are keyed on the chunks it was given.
    """
    records, _, _ = driver.execute_query(_READ_SOURCES, database_=database)
    return [
        ChunkSource(
            id=record["id"],
            text=record["text"] or "",
            ancestry=list(record["ancestry"]),
        )
        for record in records
    ]


def read_branch_folders(driver: Driver, database: str) -> list[tuple[str, str]]:
    """`(branch, folder name)` for every folder, for the ambiguity rule.

    Read from the graph rather than from the walk, so the rule sees the *whole* corpus
    even when this run was pointed at one branch of it. A name is only ambiguous
    relative to everything else in the archive.
    """
    records, _, _ = driver.execute_query(_READ_BRANCH_FOLDERS, database_=database)
    return [(record["branch"], record["name"]) for record in records]


_CLEAR_MENTIONS = """
MATCH (c:Chunk)-[m:MENTIONS]->(:Entity)
WHERE c.id IN $chunk_ids
DELETE m
"""

# Matched by the property rather than per chunk id, so this is one scan of the
# relationship type instead of one per chunk.
_CLEAR_RELATIONS = """
MATCH (:Entity)-[r:RELATES]->(:Entity)
WHERE r.source_chunk IN $chunk_ids
DELETE r
"""

# `db.create.setNodeVectorProperty` rather than a plain SET, for the same reason as on
# Chunk: it guarantees the property lands as a list of floats, and a list that came out
# as integers is invisible to the vector index with no error anywhere.
_MERGE_ENTITIES = """
UNWIND $entities AS entity
MERGE (e:Entity {normalized_name: entity.normalized_name, type: entity.type,
                 scope: entity.scope})
SET e.name = entity.name, e.mentions = entity.mentions, e.aliases = entity.aliases
WITH e, entity
CALL db.create.setNodeVectorProperty(e, 'embedding', entity.embedding)
RETURN count(*) AS written
"""

_MERGE_MENTIONS = """
UNWIND $mentions AS mention
MATCH (c:Chunk {id: mention.chunk_id})
MATCH (e:Entity {normalized_name: mention.normalized_name, type: mention.type,
                 scope: mention.scope})
MERGE (c)-[m:MENTIONS]->(e)
SET m.evidence = mention.evidence, m.surface = mention.surface
RETURN count(*) AS written
"""

# The relation type is part of the MERGE pattern, so two entities can be joined by more
# than one kind of edge — a person both leads a project and works on it.
_MERGE_RELATIONS = """
UNWIND $relations AS relation
MATCH (a:Entity {normalized_name: relation.source_name, type: relation.source_type,
                 scope: relation.source_scope})
MATCH (b:Entity {normalized_name: relation.target_name, type: relation.target_type,
                 scope: relation.target_scope})
MERGE (a)-[r:RELATES {type: relation.type}]->(b)
SET r.evidence = relation.evidence, r.source_chunk = relation.source_chunk,
    r.support = relation.support
RETURN count(*) AS written
"""

# An entity nothing mentions has no evidence behind it and cannot be justified on
# screen. These are the leftovers of a previous run whose every mention has since
# resolved elsewhere.
_DELETE_ORPHANS = """
MATCH (e:Entity)
WHERE NOT (:Chunk)-[:MENTIONS]->(e)
DETACH DELETE e
"""


@dataclass(frozen=True)
class EntityWriteReport:
    entities: int
    mentions: int
    relations: int
    #: Entities from a previous run that no chunk mentions any more.
    removed: int


def write_entities(
    driver: Driver,
    database: str,
    entities: list[ResolvedEntity],
    mentions: list[Mention],
    relations: list[GraphRelation],
    chunk_ids: list[str],
) -> EntityWriteReport:
    """Replace the knowledge graph derived from `chunk_ids`.

    One transaction: a failure between clearing the old edges and writing the new ones
    would leave a graph with entities and nothing pointing at them, which reads on
    screen as a successful ingest that found no connections.
    """
    entity_rows = [
        {
            "normalized_name": entity.identity.normalized_name,
            "type": entity.identity.type,
            "name": entity.surface,
            "scope": entity.identity.scope,
            "mentions": entity.mentions,
            "aliases": entity.aliases,
            "embedding": entity.embedding,
        }
        for entity in entities
    ]
    mention_rows = [
        {
            "chunk_id": mention.chunk_id,
            "normalized_name": mention.identity.normalized_name,
            "type": mention.identity.type,
            "scope": mention.identity.scope,
            "surface": mention.surface,
            "evidence": mention.evidence,
        }
        for mention in mentions
    ]
    relation_rows = [
        {
            "source_name": relation.source.normalized_name,
            "source_type": relation.source.type,
            "source_scope": relation.source.scope,
            "target_name": relation.target.normalized_name,
            "target_type": relation.target.type,
            "target_scope": relation.target.scope,
            "type": relation.type,
            "evidence": relation.evidence,
            "source_chunk": relation.source_chunk,
            "support": relation.support,
        }
        for relation in relations
    ]

    def write(transaction: ManagedTransaction) -> EntityWriteReport:
        transaction.run(_CLEAR_MENTIONS, {"chunk_ids": chunk_ids})
        transaction.run(_CLEAR_RELATIONS, {"chunk_ids": chunk_ids})
        written = _count(transaction, _MERGE_ENTITIES, {"entities": entity_rows})
        mentioned = _count(transaction, _MERGE_MENTIONS, {"mentions": mention_rows})
        related = _count(transaction, _MERGE_RELATIONS, {"relations": relation_rows})
        removed = (
            transaction.run(_DELETE_ORPHANS).consume().counters.nodes_deleted
        )
        return EntityWriteReport(
            entities=written, mentions=mentioned, relations=related, removed=removed
        )

    with driver.session(database=database) as session:
        return session.execute_write(write)


def _count(transaction: ManagedTransaction, cypher: str, parameters: dict) -> int:
    """Run a write that returns a count, tolerating an empty batch.

    `UNWIND []` yields no rows at all, so `.single()` would be None rather than zero.
    """
    if not next(iter(parameters.values())):
        return 0
    record = transaction.run(cypher, parameters).single()
    return record["written"] if record else 0


@dataclass(frozen=True)
class EntityCoverage:
    total: int
    by_type: dict[str, int]
    mentions: int
    relations: int
    chunks_with_entities: int
    chunks: int
    #: Entities carrying a folder edge. Must be zero — see the module docstring.
    folder_owned: int


# Separate statements rather than one query of subqueries: an empty graph makes every
# one of these match nothing, and a `CALL` subquery that returns no rows eliminates the
# row it was called for — so a single combined query answers an un-ingested graph with
# no result at all rather than with zeroes.
_BY_TYPE = "MATCH (e:Entity) RETURN e.type AS type, count(*) AS count"
_MENTION_COUNT = "MATCH (:Chunk)-[m:MENTIONS]->(:Entity) RETURN count(m) AS count"
_RELATION_COUNT = "MATCH (:Entity)-[r:RELATES]->(:Entity) RETURN count(r) AS count"
_CHUNK_COUNT = "MATCH (c:Chunk) RETURN count(c) AS count"
_COVERED_CHUNKS = (
    "MATCH (c:Chunk) WHERE (c)-[:MENTIONS]->(:Entity) RETURN count(c) AS count"
)
# The modelling rule, asserted rather than assumed: any edge at all between a Folder
# and an Entity, in either direction, is a violation.
_FOLDER_OWNED = "MATCH (:Folder)--(e:Entity) RETURN count(DISTINCT e) AS count"


def _scalar(driver: Driver, database: str, cypher: str) -> int:
    records, _, _ = driver.execute_query(cypher, database_=database)
    return records[0]["count"] if records else 0


def entity_coverage(driver: Driver, database: str) -> EntityCoverage:
    """What the knowledge graph actually holds — including the one thing it must not."""
    records, _, _ = driver.execute_query(_BY_TYPE, database_=database)
    by_type = {record["type"]: record["count"] for record in records}
    return EntityCoverage(
        total=sum(by_type.values()),
        by_type=by_type,
        mentions=_scalar(driver, database, _MENTION_COUNT),
        relations=_scalar(driver, database, _RELATION_COUNT),
        chunks_with_entities=_scalar(driver, database, _COVERED_CHUNKS),
        chunks=_scalar(driver, database, _CHUNK_COUNT),
        folder_owned=_scalar(driver, database, _FOLDER_OWNED),
    )


@dataclass(frozen=True)
class BridgeEntity:
    """An entity named by documents under more than one top-level branch.

    These are the demo. Everything else in the graph can be reached by walking down the
    folder tree; a bridge is the only kind of node that joins two branches of it, and
    it exists because the corpus plants a consultant and an artefact that genuinely
    cross from one client to another.
    """

    name: str
    type: str
    branches: list[str]
    mentions: int


_BRIDGES = """
MATCH (branch:Folder {depth: 1})-[:CONTAINS*]->(:Document)-[:HAS_CHUNK]->(c:Chunk)
MATCH (c)-[:MENTIONS]->(e:Entity)
WHERE $contains = '' OR toLower(e.name) CONTAINS toLower($contains)
WITH e, collect(DISTINCT branch.name) AS branches
WHERE size(branches) > 1
RETURN e.name AS name, e.type AS type, branches,
       coalesce(e.mentions, 0) AS mentions
ORDER BY size(branches) DESC, mentions DESC, name
LIMIT $limit
"""


def read_bridges(
    driver: Driver, database: str, contains: str = "", limit: int = 25
) -> list[BridgeEntity]:
    """Entities joining two or more branches of the hierarchy."""
    records, _, _ = driver.execute_query(
        _BRIDGES, {"contains": contains, "limit": limit}, database_=database
    )
    return [
        BridgeEntity(
            name=record["name"],
            type=record["type"],
            branches=sorted(record["branches"]),
            mentions=record["mentions"],
        )
        for record in records
    ]


_BRIDGE_COUNT = """
MATCH (branch:Folder {depth: 1})-[:CONTAINS*]->(:Document)-[:HAS_CHUNK]->(c:Chunk)
MATCH (c)-[:MENTIONS]->(e:Entity)
WITH e, count(DISTINCT branch) AS branches
WHERE branches > 1
RETURN count(e) AS count
"""


def count_bridges(driver: Driver, database: str) -> int:
    """How many entities join two or more branches, uncapped.

    Separate from `read_bridges` because that one takes a `--limit`: reporting the
    length of a truncated list reads as a verdict on resolution rather than as a page
    of it.
    """
    return _scalar(driver, database, _BRIDGE_COUNT)


@dataclass(frozen=True)
class StoredEntity:
    name: str
    type: str
    scope: str
    mentions: int
    aliases: list[str]
    relations: int


_READ_ENTITIES = """
MATCH (e:Entity)
WHERE ($contains = '' OR toLower(e.name) CONTAINS toLower($contains))
  AND ($type = '' OR e.type = $type)
OPTIONAL MATCH (e)-[r:RELATES]-(:Entity)
RETURN e.name AS name, e.type AS type, coalesce(e.scope, '') AS scope,
       coalesce(e.mentions, 0) AS mentions, coalesce(e.aliases, []) AS aliases,
       count(r) AS relations
ORDER BY mentions DESC, name
LIMIT $limit
"""


def read_entities(
    driver: Driver,
    database: str,
    contains: str = "",
    entity_type: str = "",
    limit: int = 20,
) -> list[StoredEntity]:
    """Stored entities, most-mentioned first — the view for checking resolution."""
    records, _, _ = driver.execute_query(
        _READ_ENTITIES,
        {"contains": contains, "type": entity_type, "limit": limit},
        database_=database,
    )
    return [
        StoredEntity(
            name=record["name"],
            type=record["type"],
            scope=record["scope"],
            mentions=record["mentions"],
            aliases=list(record["aliases"]),
            relations=record["relations"],
        )
        for record in records
    ]
