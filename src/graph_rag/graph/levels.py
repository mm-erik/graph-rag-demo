"""Reading the sibling folder names at each depth, and writing back what they mean.

Both halves work off the depth recorded *in the graph*, never the depth the walk
happened to produce. The two disagree whenever a subdirectory of an already-ingested
tree is ingested on its own, and writing a level name by the walk's depth would then
label the wrong folders — silently, which is the failure mode this project cannot
afford.

Both are also scoped to *one tree*: the one containing the walked root. A level is a
property of a hierarchy, not of the database. Two unrelated corpora ingested into the
same graph have their own meanings for depth 1 — one's clients are another's regions —
and sampling across both would describe neither and then stamp the answer onto both.

The driver is injected; nothing here constructs one.
"""

from __future__ import annotations

from dataclasses import dataclass

from neo4j import Driver

from graph_rag.levels import SAMPLE_LIMIT, Level, LevelSample

# The tree the walked root belongs to: the walked root may itself be a subdirectory of
# a larger ingested hierarchy, in which case the levels to name are the whole tree's.
_TREE = """
MATCH (walked:Folder {path: $root})
MATCH (top:Folder)-[:CONTAINS*0..]->(walked)
WHERE NOT (:Folder)-[:CONTAINS]->(top)
"""

# ORDER BY before collect fixes the sample: the same corpus yields the same prompt, so
# re-running an ingest cannot relabel a level because a different 40 names were drawn.
# Depth 0 is excluded — it is the tree's root, the corpus itself rather than a level of
# it, and the folder an operator's `--levels` list never means.
_READ_SAMPLES = (
    _TREE
    + """
MATCH (top)-[:CONTAINS*0..]->(f:Folder) WHERE f.depth > 0
WITH f.depth AS depth, f.name AS name ORDER BY name
WITH depth, count(name) AS folder_count, collect(DISTINCT name) AS names
RETURN depth, folder_count, size(names) AS distinct_count, names[0..$limit] AS sample
ORDER BY depth
"""
)

_WRITE_LEVELS = (
    _TREE
    + """
UNWIND $levels AS level
MATCH (top)-[:CONTAINS*0..]->(f:Folder) WHERE f.depth = level.depth
SET f.inferred_level_name = level.name,
    f.inferred_level_confidence = level.confidence
RETURN count(f) AS folders
"""
)


def read_level_samples(
    driver: Driver, database: str, root: str, limit: int = SAMPLE_LIMIT
) -> list[LevelSample]:
    """One sample of sibling folder names per depth of `root`'s tree — the cheapest
    possible input to the model: the names, how many folders there are, and how many
    are distinct."""
    records, _, _ = driver.execute_query(
        _READ_SAMPLES, {"root": root, "limit": limit}, database_=database
    )
    return [
        LevelSample(
            depth=record["depth"],
            folder_count=record["folder_count"],
            distinct_count=record["distinct_count"],
            names=list(record["sample"]),
        )
        for record in records
    ]


def write_levels(driver: Driver, database: str, root: str, levels: list[Level]) -> int:
    """Stamp each level's name and confidence onto every folder at that depth of
    `root`'s tree.

    Returns the number of folder nodes labelled.
    """
    if not levels:
        return 0
    payload = [
        {"depth": level.depth, "name": level.name, "confidence": level.confidence}
        for level in levels
    ]
    records, _, _ = driver.execute_query(
        _WRITE_LEVELS, {"root": root, "levels": payload}, database_=database
    )
    return records[0]["folders"] if records else 0


@dataclass(frozen=True)
class StoredLevel:
    """What a depth of the hierarchy was decided to mean, as the graph now holds it.

    Read back rather than recomputed: the operator endpoint and `status` report what
    an answer will actually cite, and a second inference run to find out would both
    cost a model call and be able to disagree with the stored labels.
    """

    depth: int
    #: None when this depth was never labelled — inference skipped, or failed.
    name: str | None
    confidence: float | None
    #: Folders at this depth carrying this label.
    folders: int


# Grouped by name as well as by depth, so two trees in one database that disagree
# about what depth 1 means show up as two rows rather than as whichever one the
# aggregation happened to keep. Depth 0 is the corpus root, not a level of it.
_READ_LEVELS = """
MATCH (f:Folder) WHERE f.depth > 0
WITH f.depth AS depth, f.inferred_level_name AS name,
     f.inferred_level_confidence AS confidence, count(*) AS folders
RETURN depth, name, confidence, folders
ORDER BY depth, folders DESC, name
"""


def read_levels(driver: Driver, database: str) -> list[StoredLevel]:
    """The inferred level name stamped on each depth of the hierarchy.

    Database-wide, unlike the two reads above, and deliberately: this answers an
    operator asking what the graph currently holds, and there is no walked root to
    scope it to. Two unrelated corpora in one database therefore show up as two rows
    at the same depth rather than as one averaged answer — which is the honest
    report, and the one that makes the collision visible.
    """
    records, _, _ = driver.execute_query(_READ_LEVELS, database_=database)
    return [
        StoredLevel(
            depth=record["depth"],
            name=record["name"],
            confidence=record["confidence"],
            folders=record["folders"],
        )
        for record in records
    ]
