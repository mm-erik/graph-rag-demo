"""Reading what a summary is written from, and writing it back onto the document.

Two reads rather than one, deliberately. The *state* read is cheap — a property per
document — and decides which documents need a summary at all; only those then pay for
the *source* read, which pulls the document's chunk text. A single combined read would
drag the whole corpus's text across the wire on a re-ingest that is about to summarise
nothing.

The summary is stored next to the content hash it was written from, so the cache
decision is answered by the graph itself rather than by a sidecar file that can drift
away from it.

The driver is injected; nothing here constructs one.
"""

from __future__ import annotations

from dataclasses import dataclass

from neo4j import Driver

from graph_rag.graph.ancestry import ancestry_projection, to_ancestors
from graph_rag.summaries import DocumentSource, DocumentState

_READ_STATES = """
UNWIND $paths AS path
MATCH (d:Document {path: path})
RETURN d.path AS path, d.content_hash AS content_hash, d.summary AS summary,
       d.summary_hash AS summary_hash
"""

_READ_SOURCE = (
    "MATCH (d:Document {path: $path})\n"
    + ancestry_projection("d")
    + """
OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
WITH d, ancestry, c ORDER BY c.index
RETURN d.filename AS filename, coalesce(d.page_count, 0) AS page_count, ancestry,
       [chunk IN collect(c) | chunk.text] AS chunks
"""
)

_WRITE_SUMMARY = """
MATCH (d:Document {path: $path})
SET d.summary = $summary, d.summary_hash = $content_hash
RETURN count(d) AS documents
"""


class UnknownDocument(RuntimeError):
    """No Document node at that path — the structural pass has not run, or ran against
    a different root."""


def read_document_states(
    driver: Driver, database: str, paths: list[str]
) -> list[DocumentState]:
    """What the graph already holds for each document, without its text."""
    records, _, _ = driver.execute_query(
        _READ_STATES, {"paths": paths}, database_=database
    )
    return [
        DocumentState(
            path=record["path"],
            content_hash=record["content_hash"],
            summary=record["summary"],
            summary_hash=record["summary_hash"],
        )
        for record in records
    ]


def read_document_source(driver: Driver, database: str, path: str) -> DocumentSource:
    """One document's ancestry and chunk text, in reading order."""
    records, _, _ = driver.execute_query(
        _READ_SOURCE, {"path": path}, database_=database
    )
    if not records:
        raise UnknownDocument(f"no Document node at {path}")
    record = records[0]
    return DocumentSource(
        path=path,
        filename=record["filename"],
        page_count=record["page_count"],
        ancestry=to_ancestors(record["ancestry"]),
        chunks=[text for text in record["chunks"] if text],
    )


def write_summary(
    driver: Driver, database: str, path: str, summary: str, content_hash: str
) -> None:
    """Store the summary alongside the hash of the content it describes."""
    records, _, _ = driver.execute_query(
        _WRITE_SUMMARY,
        {"path": path, "summary": summary, "content_hash": content_hash},
        database_=database,
    )
    if not records or not records[0]["documents"]:
        raise UnknownDocument(f"no Document node at {path}")


@dataclass(frozen=True)
class SummaryCoverage:
    total: int
    summarised: int
    #: Summaries describing an older version of their document.
    stale: int


_COVERAGE = """
MATCH (d:Document)
RETURN count(d) AS total,
       count(d.summary) AS summarised,
       count(CASE WHEN d.summary IS NOT NULL AND d.summary_hash <> d.content_hash
                  THEN 1 END) AS stale
"""


def summary_coverage(driver: Driver, database: str) -> SummaryCoverage:
    """How many documents carry a summary, and how many carry an outdated one."""
    records, _, _ = driver.execute_query(_COVERAGE, database_=database)
    record = records[0]
    return SummaryCoverage(
        total=record["total"], summarised=record["summarised"], stale=record["stale"]
    )


@dataclass(frozen=True)
class StoredSummary:
    path: str
    filename: str
    summary: str | None
    stale: bool


_READ_SUMMARIES = """
MATCH (d:Document)
WHERE $contains = '' OR toLower(d.path) CONTAINS toLower($contains)
RETURN d.path AS path, d.filename AS filename, d.summary AS summary,
       coalesce(d.summary_hash <> d.content_hash, false) AS stale
ORDER BY d.path
LIMIT $limit
"""


def read_summaries(
    driver: Driver, database: str, contains: str = "", limit: int = 20
) -> list[StoredSummary]:
    """Stored summaries, for reading side by side.

    The one acceptance test this feature has is a human comparing two near-identical
    reports from different clients, so the comparison needs to be one command rather
    than two hand-written Cypher queries mid-demo.
    """
    records, _, _ = driver.execute_query(
        _READ_SUMMARIES, {"contains": contains, "limit": limit}, database_=database
    )
    return [
        StoredSummary(
            path=record["path"],
            filename=record["filename"],
            summary=record["summary"],
            stale=record["stale"],
        )
        for record in records
    ]


class GraphSummaryStore:
    """`SummaryStore` backed by Neo4j, binding the injected driver to a database."""

    def __init__(self, driver: Driver, database: str) -> None:
        self._driver = driver
        self._database = database

    def read_source(self, path: str) -> DocumentSource:
        return read_document_source(self._driver, self._database, path)

    def write(self, path: str, summary: str, content_hash: str) -> None:
        write_summary(self._driver, self._database, path, summary, content_hash)
