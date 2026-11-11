"""The two chunk searches the lexical-and-semantic leg fuses, read out of Neo4j.

Both reads carry the chunk's **folder ancestry** back with it, in the same shape a
summary is written from. That is not decoration: a chunk that arrives as a path string
can only be cited as a path string, and an answer that recites
`corpus/Nordlicht Energie AG/2023/...` instead of saying *Kunde Nordlicht Energie AG,
Jahr 2023* throws away the one advantage the graph has over flat retrieval.

Both are one round trip. The obvious alternative — search, then fetch each hit's
ancestry — is ten extra queries per leg per question, on the path an audience is
watching.

The driver is injected; nothing here constructs one.
"""

from __future__ import annotations

from neo4j import Driver

from graph_rag.graph.ancestry import ancestry_projection, to_ancestors
from graph_rag.graph.schema import CHUNK_FULLTEXT_INDEX, CHUNK_VECTOR_INDEX
from graph_rag.retrieval import Chunk, ChunkHit

_WITH_ANCESTRY = (
    "MATCH (d:Document)-[:HAS_CHUNK]->(node)\n"
    + ancestry_projection("node, score, d")
    + """
RETURN node.id AS id, score, d.path AS document_path, d.filename AS filename,
       node.index AS index, node.page AS page, node.text AS text, ancestry
ORDER BY score DESC
"""
)

_VECTOR_SEARCH = (
    f"CALL db.index.vector.queryNodes('{CHUNK_VECTOR_INDEX}', $limit, $vector)\n"
    "YIELD node, score\n" + _WITH_ANCESTRY
)

_FULLTEXT_SEARCH = (
    f"CALL db.index.fulltext.queryNodes('{CHUNK_FULLTEXT_INDEX}', $query, "
    "{limit: $limit})\n"
    "YIELD node, score\n" + _WITH_ANCESTRY
)


def _hits(records) -> list[ChunkHit]:
    return [
        ChunkHit(
            score=record["score"],
            chunk=Chunk(
                id=record["id"],
                document_path=record["document_path"],
                filename=record["filename"],
                index=record["index"],
                page=record["page"],
                text=record["text"],
                ancestry=to_ancestors(record["ancestry"]),
            ),
        )
        for record in records
    ]


#: Never ask the vector index for fewer candidates than this, however few results the
#: caller wants.
#:
#: Neo4j's vector index is approximate, and `limit` is also what it spends on the
#: search. On this corpus — where every chunk embeds into the same small region — a
#: small limit makes it give up in the wrong part of the graph and return a *worse*
#: neighbourhood entirely, silently. Measured against the real corpus with the question
#: "Prüfschema Berechtigungsrollen": asked for 5 it returned five 0.88-similarity
#: chunks, none from the right project; asked for 10 and truncated to 5 it returned the
#: 0.93 chunks, all six of which are the right ones. Twelve is the smallest value that
#: recovered every question tried, including at `--top-k 3`; above it nothing changes.
VECTOR_CANDIDATE_FLOOR = 12


def vector_search(
    driver: Driver, database: str, vector: list[float], limit: int
) -> list[ChunkHit]:
    """The `limit` nearest chunks to `vector` by cosine similarity."""
    records, _, _ = driver.execute_query(
        _VECTOR_SEARCH,
        {"vector": vector, "limit": max(limit, VECTOR_CANDIDATE_FLOOR)},
        database_=database,
    )
    return _hits(records)[:limit]


def fulltext_search(
    driver: Driver, database: str, query: str, limit: int
) -> list[ChunkHit]:
    """The `limit` best chunks for a Lucene query over the German-analysed index."""
    records, _, _ = driver.execute_query(
        _FULLTEXT_SEARCH, {"query": query, "limit": limit}, database_=database
    )
    return _hits(records)


class GraphChunkIndex:
    """`ChunkIndex` backed by Neo4j, binding the injected driver to a database."""

    def __init__(self, driver: Driver, database: str) -> None:
        self._driver = driver
        self._database = database

    def by_vector(self, vector: list[float], limit: int) -> list[ChunkHit]:
        return vector_search(self._driver, self._database, vector, limit)

    def by_text(self, query: str, limit: int) -> list[ChunkHit]:
        return fulltext_search(self._driver, self._database, query, limit)
