"""Readiness reporting: is the graph reachable, and is it indexed correctly?

`status` is advertised as the command that confirms the schema is right, so it must
reject every drift `apply_schema` repairs — otherwise the one command that could catch a
misconfigured index instead certifies it.
"""

from __future__ import annotations

from dataclasses import dataclass

from neo4j import Driver

from graph_rag.graph.chunks import EmbeddingCoverage, embedding_coverage
from graph_rag.graph.schema import (
    CHUNK_FULLTEXT_INDEX,
    CHUNK_VECTOR_INDEX,
    ENTITY_VECTOR_INDEX,
    constraint_drift,
    constraint_names,
    fulltext_index_analyzer,
    index_drift,
    list_constraints,
    list_indexes,
    schema_statements,
    vector_index_dimensions,
    vector_index_similarity,
)


@dataclass(frozen=True)
class IndexStatus:
    name: str
    state: str | None  # None when the index is missing entirely
    detail: str
    ok: bool

    @property
    def populating(self) -> bool:
        return self.state == "POPULATING"


@dataclass(frozen=True)
class GraphStatus:
    server: str
    database: str
    node_counts: dict[str, int]
    indexes: list[IndexStatus]
    constraints: list[str]
    missing_constraints: list[str]
    #: What the graph actually holds, as opposed to what the index claims to.
    embeddings: EmbeddingCoverage
    #: The width the vector index was built at, for comparison with `embeddings`.
    dimensions: int

    @property
    def embeddings_ok(self) -> bool:
        """Every chunk carries a vector the index can see.

        Separate from the index checks because the remedy is different: a correct index
        over chunks written at an older width needs a re-ingest, not a re-init, and
        nothing else in this report would ever say so.
        """
        return self.embeddings.matches(self.dimensions)

    @property
    def ready(self) -> bool:
        return (
            all(index.ok for index in self.indexes)
            and not self.missing_constraints
            and self.embeddings_ok
        )

    @property
    def only_populating(self) -> bool:
        """Not ready, but waiting is the remedy — re-running `init` would do nothing."""
        return (
            not self.missing_constraints
            and self.embeddings_ok
            and all(index.ok or index.populating for index in self.indexes)
        )


def _node_counts(driver: Driver, database: str) -> dict[str, int]:
    counts = {}
    for label in ("Folder", "Document", "Chunk", "Entity"):
        records, _, _ = driver.execute_query(
            f"MATCH (n:{label}) RETURN count(n) AS count", database_=database
        )
        counts[label] = records[0]["count"]
    return counts


def _server_description(driver: Driver, database: str) -> str:
    records, _, _ = driver.execute_query(
        "CALL dbms.components() YIELD name, versions, edition "
        "RETURN name, versions[0] AS version, edition",
        database_=database,
    )
    if not records:
        return "unknown"
    row = records[0]
    return f"{row['name']} {row['version']} ({row['edition']})"


def _describe(index: dict) -> str:
    name = index["name"]
    if name in (CHUNK_VECTOR_INDEX, ENTITY_VECTOR_INDEX):
        return f"{vector_index_dimensions(index)}-dim {vector_index_similarity(index)}"
    if name == CHUNK_FULLTEXT_INDEX:
        return f"analyzer={fulltext_index_analyzer(index)}"
    return ""


def graph_status(driver: Driver, database: str, dimensions: int) -> GraphStatus:
    """Connectivity plus whether every expected index and constraint is correct.

    The driver is injected.
    """
    driver.verify_connectivity()
    existing = {index["name"]: index for index in list_indexes(driver, database)}

    statuses = []
    for statement in schema_statements(dimensions):
        if statement.kind == "constraint":
            continue
        index = existing.get(statement.name)
        if index is None:
            statuses.append(
                IndexStatus(name=statement.name, state=None, detail="", ok=False)
            )
            continue
        drift = index_drift(index, dimensions)
        detail = drift or _describe(index)
        statuses.append(
            IndexStatus(
                name=statement.name,
                state=index["state"],
                detail=detail,
                ok=drift is None and index["state"] == "ONLINE",
            )
        )

    # A constraint over the wrong properties counts as missing: it exists under the
    # right name while rejecting the writes the project needs, and `status` is
    # advertised as the command that catches exactly that.
    constraints = list_constraints(driver, database)
    found_constraints = [
        c["name"] for c in constraints if constraint_drift(c) is None
    ]
    return GraphStatus(
        server=_server_description(driver, database),
        database=database,
        node_counts=_node_counts(driver, database),
        indexes=statuses,
        constraints=found_constraints,
        missing_constraints=[
            name for name in constraint_names() if name not in found_constraints
        ],
        embeddings=embedding_coverage(driver, database),
        dimensions=dimensions,
    )
