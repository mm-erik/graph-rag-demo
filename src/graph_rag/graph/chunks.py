"""Writing a document's chunks into the graph, and reporting what landed.

Chunks are merged on their deterministic id, so re-ingesting a document overwrites its
chunks in place. That alone is not enough to make re-ingestion idempotent: a document
whose text changed produces *fewer* chunks than before, and the tail of the previous
run would survive, unreferenced by nothing and perfectly retrievable. The stale tail is
deleted explicitly.

The reading-order chain is rebuilt rather than merged, for the same reason: a shortened
document would otherwise keep an edge pointing into a chunk that no longer exists.

Reading chunks back out for retrieval is `graph/retrieval.py`; what remains here is
the coverage report, which asks a question about the write rather than about a
question.

The driver is injected; nothing here constructs one.
"""

from __future__ import annotations

from dataclasses import dataclass

from neo4j import Driver, ManagedTransaction

from graph_rag.chunks import ChunkRecord

# Deleted before the upsert, so a document that shrank does not keep the tail of its
# previous ingestion. DETACH because a chunk carries MENTIONS edges to entities.
_DELETE_STALE_CHUNKS = """
MATCH (d:Document {path: $document_path})-[:HAS_CHUNK]->(c:Chunk)
WHERE NOT c.id IN $ids
DETACH DELETE c
"""

_CLEAR_READING_ORDER = """
MATCH (:Document {path: $document_path})-[:HAS_CHUNK]->(:Chunk)-[r:NEXT]->(:Chunk)
DELETE r
"""

# `db.create.setNodeVectorProperty` rather than `SET c.embedding = ...`: it is the
# documented way to write a vector, and it guarantees the property lands as a list of
# floats. A list that came out as integers is indistinguishable in the Browser and
# invisible to the vector index.
_MERGE_CHUNKS = """
MATCH (d:Document {path: $document_path})
UNWIND $chunks AS chunk
MERGE (c:Chunk {id: chunk.id})
SET c.text = chunk.text, c.index = chunk.index, c.page = chunk.page
MERGE (d)-[:HAS_CHUNK]->(c)
WITH c, chunk
CALL db.create.setNodeVectorProperty(c, 'embedding', chunk.embedding)
RETURN count(*) AS written
"""

_LINK_READING_ORDER = """
UNWIND $pairs AS pair
MATCH (a:Chunk {id: pair.from})
MATCH (b:Chunk {id: pair.to})
MERGE (a)-[:NEXT]->(b)
"""

_SET_PAGE_COUNT = """
MATCH (d:Document {path: $document_path})
SET d.page_count = $page_count
RETURN count(d) AS documents
"""


class UnknownDocument(RuntimeError):
    """No Document node at that path — the structural pass has not run, or ran against
    a different root. Writing chunks anyway would leave them attached to nothing."""


@dataclass(frozen=True)
class ChunkWriteReport:
    written: int
    #: Chunks from a previous ingestion of this document that no longer exist.
    removed: int


def write_chunks(
    driver: Driver,
    database: str,
    document_path: str,
    chunks: list[ChunkRecord],
    page_count: int,
) -> ChunkWriteReport:
    """Replace a document's chunks with `chunks`, in reading order.

    Every pass shares one transaction: a failure between the delete and the upsert
    would otherwise leave the document with no content at all.
    """
    ids = [chunk.id for chunk in chunks]
    rows = [
        {
            "id": chunk.id,
            "text": chunk.text,
            "index": chunk.index,
            "page": chunk.page,
            "embedding": chunk.embedding,
        }
        for chunk in chunks
    ]
    pairs = [
        {"from": ids[position], "to": ids[position + 1]}
        for position in range(len(ids) - 1)
    ]

    def write(transaction: ManagedTransaction) -> ChunkWriteReport:
        documents = transaction.run(
            _SET_PAGE_COUNT,
            {"document_path": document_path, "page_count": page_count},
        ).single()["documents"]
        if not documents:
            raise UnknownDocument(
                f"no Document node at {document_path}. Chunks are hung off the "
                "structure written earlier in the same ingest; run `ingest` over the "
                "directory rather than writing chunks on their own."
            )
        removed = (
            transaction.run(
                _DELETE_STALE_CHUNKS, {"document_path": document_path, "ids": ids}
            )
            .consume()
            .counters.nodes_deleted
        )
        transaction.run(_CLEAR_READING_ORDER, {"document_path": document_path})
        written = 0
        if rows:
            written = transaction.run(
                _MERGE_CHUNKS, {"document_path": document_path, "chunks": rows}
            ).single()["written"]
            transaction.run(_LINK_READING_ORDER, {"pairs": pairs})
        return ChunkWriteReport(written=written, removed=removed)

    with driver.session(database=database) as session:
        return session.execute_write(write)


_EMBEDDING_COVERAGE = """
MATCH (c:Chunk)
RETURN count(c) AS total,
       count(c.embedding) AS embedded,
       collect(DISTINCT size(c.embedding)) AS widths
"""


@dataclass(frozen=True)
class EmbeddingCoverage:
    """What is actually stored, as opposed to what the index claims to hold."""

    total: int
    embedded: int
    #: Every distinct vector width in the graph. More than one entry, or one that is
    #: not the index's width, means chunks the vector index is silently ignoring.
    widths: list[int]

    def matches(self, dimensions: int) -> bool:
        if self.total == 0:
            return True
        return self.embedded == self.total and self.widths == [dimensions]


def embedding_coverage(driver: Driver, database: str) -> EmbeddingCoverage:
    """How many chunks carry an embedding, and how wide those embeddings are."""
    records, _, _ = driver.execute_query(_EMBEDDING_COVERAGE, database_=database)
    record = records[0]
    return EmbeddingCoverage(
        total=record["total"],
        embedded=record["embedded"],
        widths=sorted(record["widths"]),
    )
