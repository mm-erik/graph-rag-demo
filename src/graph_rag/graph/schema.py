"""Constraints and indexes for the demo graph.

Every statement is idempotent: re-running `apply_schema` must succeed on an already
initialised graph. The case `IF NOT EXISTS` cannot cover is an index that exists under
the right *name* but the wrong *configuration* — Neo4j then skips creation and ignores
the OPTIONS entirely. A vector index at the wrong width cannot hold the model's vectors,
and a full-text index with the wrong analyzer silently stops stemming German. Both are
dropped and recreated, and the fact is reported.
"""

from __future__ import annotations

from dataclasses import dataclass

from neo4j import Driver

CHUNK_VECTOR_INDEX = "chunk_embedding"
ENTITY_VECTOR_INDEX = "entity_embedding"
CHUNK_FULLTEXT_INDEX = "chunk_text_fulltext"

# German analyzer: the corpus and every prompt in this project are German, and the
# lexical leg is the whole reason a full-text index exists alongside the vectors.
FULLTEXT_ANALYZER = "german"


@dataclass(frozen=True)
class SchemaStatement:
    name: str
    kind: str  # "constraint" | "vector index" | "fulltext index"
    cypher: str


def _constraints() -> list[SchemaStatement]:
    specs = [
        ("folder_path_unique", "(f:Folder) REQUIRE f.path IS UNIQUE"),
        ("document_path_unique", "(d:Document) REQUIRE d.path IS UNIQUE"),
        ("chunk_id_unique", "(c:Chunk) REQUIRE c.id IS UNIQUE"),
        # All three, because an entity's identity is all three: the same normalised
        # name under two scopes is two different things the hierarchy has already
        # ruled apart, and a two-property constraint would refuse to store the second.
        (
            "entity_identity_unique",
            "(e:Entity) REQUIRE (e.normalized_name, e.type, e.scope) IS UNIQUE",
        ),
    ]
    return [
        SchemaStatement(
            name=name,
            kind="constraint",
            cypher=f"CREATE CONSTRAINT {name} IF NOT EXISTS FOR {body}",
        )
        for name, body in specs
    ]


def _vector_index(name: str, label: str, dimensions: int) -> SchemaStatement:
    return SchemaStatement(
        name=name,
        kind="vector index",
        cypher=(
            f"CREATE VECTOR INDEX {name} IF NOT EXISTS "
            f"FOR (n:{label}) ON (n.embedding) "
            "OPTIONS {indexConfig: {"
            f"`vector.dimensions`: {dimensions}, "
            "`vector.similarity_function`: 'cosine'}}"
        ),
    )


def schema_statements(dimensions: int) -> list[SchemaStatement]:
    """All constraints and indexes the project needs, in creation order."""
    return [
        *_constraints(),
        _vector_index(CHUNK_VECTOR_INDEX, "Chunk", dimensions),
        _vector_index(ENTITY_VECTOR_INDEX, "Entity", dimensions),
        SchemaStatement(
            name=CHUNK_FULLTEXT_INDEX,
            kind="fulltext index",
            cypher=(
                f"CREATE FULLTEXT INDEX {CHUNK_FULLTEXT_INDEX} IF NOT EXISTS "
                "FOR (c:Chunk) ON EACH [c.text] "
                "OPTIONS {indexConfig: {"
                f"`fulltext.analyzer`: '{FULLTEXT_ANALYZER}'"
                "}}"
            ),
        ),
    ]


def list_indexes(driver: Driver, database: str) -> list[dict]:
    records, _, _ = driver.execute_query(
        "SHOW INDEXES YIELD name, type, entityType, labelsOrTypes, properties, state, options "
        "RETURN name, type, entityType, labelsOrTypes, properties, state, options ORDER BY name",
        database_=database,
    )
    return [record.data() for record in records]


def list_constraints(driver: Driver, database: str) -> list[dict]:
    records, _, _ = driver.execute_query(
        "SHOW CONSTRAINTS YIELD name, type, labelsOrTypes, properties "
        "RETURN name, type, labelsOrTypes, properties ORDER BY name",
        database_=database,
    )
    return [record.data() for record in records]


def _index_config(index: dict) -> dict:
    return (index.get("options") or {}).get("indexConfig") or {}


def vector_index_dimensions(index: dict) -> int | None:
    value = _index_config(index).get("vector.dimensions")
    return int(value) if value is not None else None


def vector_index_similarity(index: dict) -> str | None:
    value = _index_config(index).get("vector.similarity_function")
    return str(value).lower() if value is not None else None


def fulltext_index_analyzer(index: dict) -> str | None:
    value = _index_config(index).get("fulltext.analyzer")
    return str(value).lower() if value is not None else None


def index_drift(index: dict, dimensions: int) -> str | None:
    """Why this existing index does not match what the project needs, if it does not."""
    name = index["name"]
    # A FAILED index keeps its (correct) options, so only its state gives it away.
    # Without this, `status` rightly refuses to call the graph ready while `init` has
    # nothing to repair, and the operator loops between the two.
    if index.get("state") == "FAILED":
        return "state=FAILED"
    if name in (CHUNK_VECTOR_INDEX, ENTITY_VECTOR_INDEX):
        found = vector_index_dimensions(index)
        if found != dimensions:
            return f"{found}-dim, expected {dimensions}-dim"
        similarity = vector_index_similarity(index)
        if similarity != "cosine":
            return f"similarity={similarity}, expected cosine"
    elif name == CHUNK_FULLTEXT_INDEX:
        analyzer = fulltext_index_analyzer(index)
        if analyzer != FULLTEXT_ANALYZER:
            return f"analyzer={analyzer}, expected {FULLTEXT_ANALYZER}"
    return None


@dataclass(frozen=True)
class DroppedIndex:
    name: str
    reason: str
    # Dropping a vector index orphans embeddings already written at the old width:
    # Neo4j skips nodes whose vector does not match the index, with no error.
    needs_reingest: bool


def _drop_drifted_indexes(
    driver: Driver, database: str, dimensions: int, dropped: list[DroppedIndex]
) -> list[DroppedIndex]:
    """Drop indexes `IF NOT EXISTS` would otherwise leave misconfigured.

    Appends to the caller's list as it goes, so a failure midway still leaves the caller
    holding every drop that did happen.
    """
    managed = {CHUNK_VECTOR_INDEX, ENTITY_VECTOR_INDEX, CHUNK_FULLTEXT_INDEX}
    for index in list_indexes(driver, database):
        if index["name"] not in managed:
            continue
        reason = index_drift(index, dimensions)
        if reason is None:
            continue
        driver.execute_query(f"DROP INDEX {index['name']} IF EXISTS", database_=database)
        dropped.append(
            DroppedIndex(
                name=index["name"],
                reason=reason,
                needs_reingest=index["type"] == "VECTOR",
            )
        )
    return dropped


def constraint_names() -> list[str]:
    return [statement.name for statement in _constraints()]


#: What each managed constraint must be over. `CREATE CONSTRAINT ... IF NOT EXISTS`
#: matches on the *name*, so a constraint created by an older version of this project
#: over fewer properties survives re-initialisation untouched — and then rejects the
#: writes it was widened to allow. Same failure as a drifted index, same repair.
_CONSTRAINT_PROPERTIES = {
    "folder_path_unique": ["path"],
    "document_path_unique": ["path"],
    "chunk_id_unique": ["id"],
    "entity_identity_unique": ["normalized_name", "type", "scope"],
}


def constraint_drift(constraint: dict) -> str | None:
    """Why this existing constraint does not match what the project needs."""
    expected = _CONSTRAINT_PROPERTIES.get(constraint["name"])
    found = list(constraint.get("properties") or [])
    if expected is None or sorted(found) == sorted(expected):
        return None
    return f"over {found or 'nothing'}, expected {expected}"


def _drop_drifted_constraints(driver: Driver, database: str) -> list[DroppedIndex]:
    """Drop constraints `IF NOT EXISTS` would otherwise leave over the wrong properties.

    Reported as drops like a drifted index, but never as needing a re-ingest: a
    constraint holds no data, so recreating one costs nothing already written.
    """
    dropped = []
    for constraint in list_constraints(driver, database):
        reason = constraint_drift(constraint)
        if reason is None:
            continue
        driver.execute_query(
            f"DROP CONSTRAINT {constraint['name']} IF EXISTS", database_=database
        )
        dropped.append(
            DroppedIndex(name=constraint["name"], reason=reason, needs_reingest=False)
        )
    return dropped


@dataclass(frozen=True)
class AppliedStatement:
    statement: SchemaStatement
    created: bool  # False when it already existed and `IF NOT EXISTS` skipped it


@dataclass(frozen=True)
class SchemaReport:
    applied: list[AppliedStatement]
    dropped: list[DroppedIndex]


class SchemaApplicationError(RuntimeError):
    """Creation failed after drifted indexes were already dropped.

    Carries the drops so the operator is told what was destroyed — otherwise a single
    bad value silently leaves the graph with fewer indexes than it started with.
    """

    def __init__(self, dropped: list[DroppedIndex], cause: Exception) -> None:
        super().__init__(str(cause))
        #: Only the drops that were *not* recreated before the failure. Reporting a
        #: successfully recreated index as destroyed would send the operator off to
        #: re-ingest a corpus that is fine.
        self.dropped = dropped


def _existing_names(driver: Driver, database: str) -> set[str]:
    indexes = {index["name"] for index in list_indexes(driver, database)}
    return indexes | {constraint["name"] for constraint in list_constraints(driver, database)}


def apply_schema(driver: Driver, database: str, dimensions: int) -> SchemaReport:
    """Create every constraint and index, idempotently. The driver is injected."""
    dropped: list[DroppedIndex] = []
    applied: list[AppliedStatement] = []
    recreated: set[str] = set()
    try:
        # Inside the try as well: a failure partway through the drops would otherwise
        # surface as a bare Neo4j error, leaving the operator unaware that an index was
        # already destroyed — the silent shrinkage this error class exists to prevent.
        dropped = _drop_drifted_indexes(driver, database, dimensions, dropped)
        dropped.extend(_drop_drifted_constraints(driver, database))
        before = _existing_names(driver, database)

        for statement in schema_statements(dimensions):
            driver.execute_query(statement.cypher, database_=database)
            applied.append(
                AppliedStatement(statement=statement, created=statement.name not in before)
            )
            recreated.add(statement.name)
    except Exception as error:
        raise SchemaApplicationError(
            [drop for drop in dropped if drop.name not in recreated], error
        ) from error
    return SchemaReport(applied=applied, dropped=dropped)
